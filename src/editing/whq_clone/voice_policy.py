"""voice_policy — 逐段「用户原声 vs 声音克隆」决策。

用户需求: 素材本就有口播文案时延用用户原声, 尤其是画面有人脸时(口型对得上);
没有人脸时, 若原声不符合成片整体文案, 才用克隆。

对 manifest 每段计算三个证据:
  1. window_speech: 该段实际使用的源窗口 [source_start, source_start+take] 内,
     用户 ASR 逐字条目(asr_items)的重叠文本 + 覆盖率(说话时长/窗口时长)。
  2. face: YuNet(models/face_detection_yunet_2023mar.onnx) 对窗口抽帧检测人脸。
     主流程环境无 cv2, 经子进程(env WHQ_CV_PYTHON, 默认 /root/miniconda3/bin/python3.13)
     跑 face_probe.py。
  3. tempo: 素材不足时 clone_builder 会 setpts 拉伸视频, 原声需同倍率 atempo 放慢;
     低于 WHQ_VOICE_MIN_ATEMPO(默认 0.75, 再慢听感发糊)则原声不可用。

决策(写进 plan JSON 的 voice_source + decision_basis):
  - 有口播 + 有人脸 + 变速可行  -> original(确定性保留原声)
  - 有口播 + 无人脸             -> llm_decide(交给脚本 LLM 判断原声是否贴合整体文案)
  - 无口播 / 变速不可行          -> clone
"""
import json
import os
import re
import subprocess

from _common import REPO, legacy

CV_PYTHON = os.getenv("WHQ_CV_PYTHON", "/root/miniconda3/bin/python3.13")
FACE_MODEL = os.getenv(
    "WHQ_FACE_MODEL",
    os.path.join(REPO, "models", "face_detection_yunet_2023mar.onnx"))
FACE_PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_probe.py")
# 窗口内至少这么多中文字才算「本就有口播文案」(过滤"嗯/好"之类口水音)
# 用户定的规则：**有人脸且这个人开口说话了就保留原声**，没说话才克隆。阈值定高会把
# "有人在说话"的段判成无口播 -> 换成克隆配音 -> 画面里人的口型和听到的话对不上。
# 所以只挡纯口水音（1~2 个字），可用 WHQ_VOICE_MIN_CHARS 调回严格。
MIN_SPEECH_CHARS = int(os.getenv("WHQ_VOICE_MIN_CHARS", "3"))
# 窗口内说话时长占比阈值(太稀疏说明只是零星词, 铺不满该段)。同上：宁可保原声也不要口型错位。
MIN_SPEECH_COVERAGE = float(os.getenv("WHQ_VOICE_MIN_COVERAGE", "0.25"))
# 抽帧有人脸帧占比阈值
MIN_FACE_RATIO = float(os.getenv("WHQ_FACE_MIN_RATIO", "0.4"))
# 原声允许的最慢变速(素材被拉伸时原声要同倍率放慢才对得上口型)
MIN_ATEMPO = float(os.getenv("WHQ_VOICE_MIN_ATEMPO", "0.75"))
# —— 句子级对窗(说完整句) ——
# 相邻字间隔>=此秒数视为一个分句结束(说话人的自然停顿)
SENT_GAP = float(os.getenv("WHQ_SENT_GAP", "0.35"))
# 原声允许的最快加速: 完整句组比槽位长时, 音画同倍率加速(不破口型), 超过则装不下
MAX_ATEMPO_UP = float(os.getenv("WHQ_VOICE_MAX_ATEMPO", "1.35"))
WIN_LEAD = 0.10   # 对窗后句首前留一点起势
WIN_TAIL = 0.12   # 句尾留衰减, 不贴字尾硬切


def _norm(text):
    return re.sub(r"\s+", "", str(text or "").strip())


def _utterances(rec, gap=None):
    """按逐字间隔把 ASR 切成分句: 相邻字间隔>=gap 视为一句结束。返回 [{start,end,text}]。"""
    gap = SENT_GAP if gap is None else gap
    words = []
    for it in rec.get("asr_items", []) or []:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        if e < s:
            continue
        words.append((s, e, _norm(it.get("text"))))
    words.sort()
    utts = []
    for s, e, t in words:
        if utts and s - utts[-1]["end"] < gap:
            utts[-1]["end"] = max(utts[-1]["end"], e)
            utts[-1]["text"] += t
        else:
            utts.append({"start": s, "end": e, "text": t})
    return utts


def _sent_guard():
    """是否启用「原声完整句护栏」(见 decide 内注释)。legacy 下关闭。"""
    if legacy():
        return False
    return os.getenv("WHQ_ORIGINAL_SENT_GUARD", "1") not in ("0", "false", "False")


def _ends_at_pause(rec, end_time, gap=None):
    """音频窗口末端 end_time 是否落在自然停顿处（说完整句/分句）。

    end_time 之后 gap 秒内还有字开口 -> 说话人还没停，是被截断的半句/残句(返回 False)。
    用于「原声完整句护栏」：锁定原声段前先确认它不是没说完就被硬切的残句。
    """
    gap = SENT_GAP if gap is None else gap
    for it in rec.get("asr_items", []) or []:
        try:
            s = float(it.get("start"))
        except (TypeError, ValueError):
            continue
        # 紧接 end_time 之后（gap 内）还有字 -> 句子还在继续，末端是半句
        if end_time - 0.05 < s < end_time + gap:
            return False
    return True


def align_window(rec, start, avail, target):
    """原声段「说完整句」对窗: 返回 None(维持原窗) 或调整方案 dict。

    问题: 段窗口按参考节拍定长硬切, 切点常落在用户句子中间(实测 S06 尾句催单
    「真的赶紧吧」整个被切掉, S02 停在「我比之前早起」半句上)。修法按优先级:
      1. full_sentence: 窗口终点改落在某个**句尾之后的停顿**里(音频说完整句、自然
         收束), 逐个「前 k 句」组合试探, 取变速倍率最接近 1 的方案。窗口整体伸缩后
         视频 setpts、音频 atempo **同倍率**回填 target_duration(倍率钳
         [MIN_ATEMPO, MAX_ATEMPO_UP], 音画同步口型不花)。窗口始终钳在该段素材资产
         边界内, 不侵占分给其他段的镜头(镜头不重复)。
      2. clause_cut: 任何句组都装不进弹性范围时维持原窗, 但音频只截到窗内最后一个
         **完整分句**的结尾(自然停顿处), 不再切在分句中间。
    """
    if not rec or target <= 0:
        return None
    take0 = min(avail, target) if avail > 0 else target
    asset_end = start + (avail if avail > 0 else target)
    w_end = start + take0
    utts = _utterances(rec)
    group = [u for u in utts if min(u["end"], w_end) - max(u["start"], start) > 0.05]
    if not group:
        return None
    gi = utts.index(group[0])
    # 句首不可早于资产起点(之前的画面属于别的段)
    gs = max(start, group[0]["start"] - WIN_LEAD)
    best = None
    for k, u in enumerate(group):
        if u["end"] > asset_end + 0.01:
            continue  # 该句句尾越过素材资产边界, 收不完整
        lo = min(u["end"] + WIN_TAIL, asset_end)          # 至少要包住第 k 句句尾
        nxt_idx = gi + k + 1
        hi = asset_end                                     # 最远到资产边界
        if nxt_idx < len(utts):
            hi = min(hi, utts[nxt_idx]["start"] - 0.05)    # 不能带进下一句的字
        if hi < lo - 1e-6:
            continue
        end = min(max(gs + target, lo), hi)                # 停顿区间内取最贴近目标时长的终点
        span = end - gs
        ratio = span / target
        if not (MIN_ATEMPO <= ratio <= MAX_ATEMPO_UP):
            continue
        cand = {"mode": "full_sentence", "start": round(gs, 3), "take": round(span, 3),
                "text": "".join(x["text"] for x in group[:k + 1]),
                "audio_take": round(u["end"] - gs, 3),
                "note": ("对窗至完整句组[{:.2f},{:.2f}], 源窗{:.2f}s音画同倍率{:.2f}x回填"
                         "{:.2f}s槽, 句子在停顿处自然收束"
                         .format(gs, end, span, ratio, target))}
        if best is None or abs(ratio - 1.0) < best[0]:
            best = (abs(ratio - 1.0), cand)
    if best:
        return best[1]
    inside = [u for u in group if u["start"] >= start - 0.05 and u["end"] <= w_end + 0.05]
    if inside:
        return {"mode": "clause_cut", "start": round(start, 3), "take": round(take0, 3),
                "text": "".join(u["text"] for u in inside),
                "audio_take": round(max(0.0, inside[-1]["end"] - start), 3),
                "note": ("完整句组超出{:.2f}x变速弹性, 维持原窗, 音频收口到窗内最后一个"
                         "分句停顿{:.2f}s处".format(MAX_ATEMPO_UP, inside[-1]["end"]))}
    return None


def window_speech(rec, window_start, window_duration):
    """窗口内的口播文本/覆盖率/干净截止点。rec 是 all_source_asr.json 的一条记录。

    返回 (text, coverage, clean_end): text 只收**基本完整落在窗口内**的字(避免边界
    半个字进文案); clean_end 是这些字里最后一个的结束时刻(绝对时间) —— 原声音频应
    截到这里而不是硬切窗口末端, 否则最后一个字被拦腰砍断(实测 S06 "…还有活动真|"
    戛然而止, 字幕也带着悬空尾字)。
    """
    if not rec or window_duration <= 0:
        return "", 0.0, 0.0
    window_end = window_start + window_duration
    pieces = []
    voiced = 0.0
    clean_end = 0.0
    for it in rec.get("asr_items", []) or []:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        overlap = min(e, window_end) - max(s, window_start)
        if overlap <= 0:
            continue
        # 只收基本完整落在窗口内的字(>=80%), 边界上被切一半的字不要
        if overlap / (e - s) >= 0.8:
            pieces.append((s, _norm(it.get("text"))))
            clean_end = max(clean_end, min(e, window_end))
        voiced += overlap
    pieces.sort(key=lambda x: x[0])
    return "".join(t for _, t in pieces), min(1.0, voiced / window_duration), clean_end


def probe_faces(video, start, duration, frames=5):
    """子进程跑 YuNet。失败(无cv2/无模型/坏视频)按无人脸处理, 不阻断主流程。"""
    model = os.path.abspath(FACE_MODEL)
    if not os.path.exists(model):
        print("[voice_policy] 人脸检测跳过(按无人脸处理): 缺 YuNet 模型 {}".format(model), flush=True)
        return {"sampled": 0, "face_frames": 0, "face_ratio": 0.0, "max_scores": []}
    try:
        proc = subprocess.run(
            [CV_PYTHON, FACE_PROBE, "--video", str(video),
             "--model", model,
             "--start", "{:.3f}".format(start), "--duration", "{:.3f}".format(duration),
             "--frames", str(frames)],
            capture_output=True, text=True, timeout=120)
        lines = (proc.stdout or "").strip().splitlines()
        if not lines:
            # face_probe 无输出 = 子进程崩溃(缺 cv2/坏模型等); 带上 stderr 尾便于定位
            err = (proc.stderr or "").strip().splitlines()
            reason = err[-1] if err else "无输出(exit={})".format(proc.returncode)
            print("[voice_policy] 人脸检测失败(按无人脸处理): {}".format(reason[:160]), flush=True)
            return {"sampled": 0, "face_frames": 0, "face_ratio": 0.0, "max_scores": []}
        return json.loads(lines[-1])
    except Exception as exc:
        print("[voice_policy] 人脸检测失败(按无人脸处理): {}".format(str(exc)[:120]), flush=True)
        return {"sampled": 0, "face_frames": 0, "face_ratio": 0.0, "max_scores": []}


def decide(manifest, speech_records):
    """返回 {slot_id: decision}; decision 含 voice_source/window_text/coverage/
    face_ratio/atempo/decision_basis。voice_source ∈ original|llm_decide|clone。"""
    rec_by_path = {}
    for r in speech_records or []:
        p = str(r.get("source_path") or "")
        if p:
            rec_by_path[p] = r
    out = {}
    for m in manifest:
        sid = m.get("slot_id")
        src = str(m.get("source_path") or "")
        target = float(m.get("target_duration") or 0.0)
        avail = float(m.get("source_avail") or 0.0)
        start = float(m.get("source_start") or 0.0)
        take = float(m.get("source_take") or 0.0)
        if take <= 0:
            take = min(avail, target) if avail > 0 else target
        atempo = (take / target) if target > 0 else 1.0
        rec = rec_by_path.get(src)
        text, coverage, clean_end = window_speech(rec, start, take)
        has_speech = len(text) >= MIN_SPEECH_CHARS and coverage >= MIN_SPEECH_COVERAGE
        # 原声音频只截到最后一个完整字的边界(不切半个字); 无口播段无意义, 保持 take
        audio_take = round(max(0.0, clean_end - start), 3) if clean_end > start else take
        audio_take = min(audio_take, take)
        d = {"slot_id": sid, "window_text": text, "speech_coverage": round(coverage, 3),
             "audio_take": audio_take,
             "face_ratio": 0.0, "atempo": round(atempo, 3)}
        if not has_speech:
            d["voice_source"] = "clone"
            d["decision_basis"] = ("窗口无有效口播(字数{}/覆盖率{:.0%}), 用克隆配音"
                                   .format(len(text), coverage))
        elif atempo < MIN_ATEMPO:
            d["voice_source"] = "clone"
            d["decision_basis"] = ("窗口有口播但素材被拉伸{:.2f}x, 原声需放慢至{:.2f}倍"
                                   "(低于{}下限)听感失真, 用克隆配音"
                                   .format(1.0 / atempo if atempo else 0, atempo, MIN_ATEMPO))
        else:
            probe = probe_faces(src, start, take)
            d["face_ratio"] = probe.get("face_ratio", 0.0)
            if d["face_ratio"] >= MIN_FACE_RATIO:
                d["voice_source"] = "original"
                d["decision_basis"] = ("窗口有真实口播「{}」(覆盖率{:.0%}) 且画面有人脸"
                                       "({}/{}帧, 口型需对上), 保留用户原声"
                                       .format(text[:40], coverage,
                                               probe.get("face_frames", 0),
                                               probe.get("sampled", 0)))
            else:
                d["voice_source"] = "llm_decide"
                d["decision_basis"] = ("窗口有真实口播「{}」(覆盖率{:.0%}) 但画面无人脸"
                                       "({}/{}帧), 由脚本 LLM 判断原声是否贴合整体文案"
                                       .format(text[:40], coverage,
                                               probe.get("face_frames", 0),
                                               probe.get("sampled", 0)))
        if d["voice_source"] in ("original", "llm_decide"):
            # 句子级对窗: 说完整句(窗口终点落在句尾停顿), 音画同倍率伸缩; 装不下则
            # 音频收口到分句边界。win_start/win_take 供上游在剪画面前平移源窗口。
            # llm_decide 段只做分句收口不平移窗口: LLM 可能最终选克隆, 画面窗口必须
            # 与决策时一致, 否则克隆兜底时画面被无谓换掉。
            adj = align_window(rec, start, avail, target)
            if adj and (d["voice_source"] == "original" or adj["mode"] == "clause_cut"):
                d["window_text"] = adj["text"]
                d["audio_take"] = adj["audio_take"]
                if d["voice_source"] == "original":
                    d["win_start"] = adj["start"]
                    d["win_take"] = adj["take"]
                    d["atempo"] = round(adj["take"] / target, 3) if target > 0 else 1.0
                d["decision_basis"] += "; " + adj["note"]
            # 原声完整句护栏: 对窗后若音频末端仍不在自然停顿处(半句/残句被硬切),
            # 该原声不锁定, 降级为克隆配音(写完整短文案), 避免"一句话没说完就切镜头"。
            # 注意: 判据严(末端 SENT_GAP 内有字即判半句), 会把大量原声降级成克隆,
            # 与"尽量保留用户原声"冲突 -> WHQ_ORIGINAL_SENT_GUARD=0 / WHQ_LEGACY=1 可关。
            if d["voice_source"] == "original" and _sent_guard():
                win_s = d.get("win_start", start)
                aud_end = win_s + float(d.get("audio_take") or 0.0)
                if not _ends_at_pause(rec, aud_end):
                    tail = (d.get("window_text") or "")[-6:]
                    d["voice_source"] = "clone"
                    d["decision_basis"] = ("原声窗口是半句(末端「{}」后无自然停顿, 塞不下完整句)"
                                           "-> 降级克隆保文案完整".format(tail))
                    for k in ("win_start", "win_take"):
                        d.pop(k, None)
                    d["audio_take"] = take
                    d["atempo"] = round(atempo, 3)
        print("[voice_policy] {} -> {} | {}".format(sid, d["voice_source"],
                                                    d["decision_basis"]), flush=True)
        out[sid] = d
    return out
