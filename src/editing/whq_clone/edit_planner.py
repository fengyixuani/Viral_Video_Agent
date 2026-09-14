"""edit_planner — 结构级(而非逐镜)复刻规划: 把参考视频的 DNA 叙事拆成若干
「有意义的段落」(不是 12 个碎场景切点), 再给每段分配一条**不重复**的用户素材。

与旧 shot_matcher 的区别:
  - 段落来自 DNA key_beats(叙事节拍, 通常 5~8 段), 贴的是「结构/节奏」而非逐帧切点。
  - 素材分配是 **1:1 不复用**: 每条用户片段最多用于一个段落, 保证成片不重复镜头。
  - 目标是「逻辑合理 + 尽量贴近参考」, 不追求逐镜像素级复刻。

产出 plan:
  {reference:{duration,...}, segments:[
     {index, slot_id, beat_desc, target_duration, best_candidate, score, method, is_gap}
  ]}
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import COMMON, legacy  # noqa: F401
from shot_matcher import score_candidate_deterministic

try:
    from pipeline_utils import ask_qianfan, loads_with_repair, parallel_map
    _HAS_LLM = True
except Exception:  # pragma: no cover
    _HAS_LLM = False

    def parallel_map(fn, items, **_kw):  # 无网关时也要能跑确定性分配
        return [fn(x) for x in items]

# 原声优先: 匹配阶段偏向「窗口内自带用户口播」的素材, 让下游 voice_policy 有机会
# 保留原声(见 voice_policy: 有口播+人脸->original)。默认关闭, 由 WHQ_PREFER_ORIGINAL_VOICE=1
# 开启(对其他项目零影响)。口播度量复用 voice_policy 的窗口口播/覆盖率口径, 与决策一致。
try:
    from voice_policy import window_speech, MIN_SPEECH_CHARS, MIN_SPEECH_COVERAGE, is_filler_speech
    _HAS_VP = True
except Exception:  # pragma: no cover
    _HAS_VP = False
# 注意: 必须在**调用时**读环境变量, 不能在 import 时固化成模块常量。
# 否则终端链路(先 import edit_planner, runner 再 setdefault WHQ_PREFER_ORIGINAL_VOICE)
# 会因常量已 = False 而永久跳过 speech_map, 原声优先形同虚设(实测终端原声率恒为0)。
def _prefer_original():
    return os.getenv("WHQ_PREFER_ORIGINAL_VOICE", "0") not in ("0", "false", "False")


def _ov_det_bonus():
    # 确定性打分里给「有可用口播」的候选加的分(相关性满分约 1.0, 此加分为辅不喧宾夺主)
    return float(os.getenv("WHQ_ORIGINAL_VOICE_BONUS", "0.25"))


def _min_product_shots():
    # 成片至少保留这么多「商品展示」段(无口播的 B-roll: 配料/包装/冲泡/使用演示等)。
    # 原声优先会把带口播的讲解镜头铺满全片, 挤掉商品展示画面; 这里设下限强制留白给商品镜头。
    if legacy():
        return 0
    return int(os.getenv("WHQ_MIN_PRODUCT_SHOTS", "2"))


def _product_shot_penalty():
    # 每缺 1 段商品展示的扣分。默认 1.0 >= 原声率上限(1.0), 等于把下限当硬闸——选片会先
    # 凑够 B-roll 再谈贴合/原声, 是效果回退的主因之一; 调小(如 0.15)可退化为软偏好。
    return float(os.getenv("WHQ_PRODUCT_SHOT_PENALTY", "1.0"))


def build_speech_map(candidates, speech_records):
    """cand_id -> {chars, coverage, has_speech, text}: 候选**自身窗口**内的用户口播。

    口径与 voice_policy.window_speech 完全一致(只收 >=80% 落窗内的字, 覆盖率=说话
    时长/窗口时长), 因此「matcher 认为有可用口播」与「voice_policy 会判 original」对齐。
    注意用候选自身 [start,end] 作窗口: 长素材里口播集中在某段时, 拆分后的该 asset_segment
    局部覆盖率高, 不会被整片低覆盖率淹没(实测 225406 成分口播即此情形)。
    """
    if not (_HAS_VP and speech_records):
        return {}
    rec_by_path = {}
    for r in speech_records:
        p = str(r.get("source_path") or "")
        if p:
            rec_by_path[p] = r
    out = {}
    for c in candidates:
        rec = rec_by_path.get(str(c.get("source_path") or ""))
        start = float(c.get("start") or 0.0)
        dur = float(c.get("duration") or 0.0)
        if rec and dur > 0:
            text, cov, _ = window_speech(rec, start, dur)
            # 现场废话(拍摄口令/口水话)不算可用口播: 否则匹配阶段会偏向这类候选, 下游又只能
            # 判克隆 —— 白占了一个"自带口播"的名额。
            has = (len(text) >= MIN_SPEECH_CHARS and cov >= MIN_SPEECH_COVERAGE
                   and not is_filler_speech(text)[0])
            out[c["global_asset_id"]] = {"chars": len(text), "coverage": round(cov, 3),
                                         "has_speech": has, "text": text}
        else:
            out[c["global_asset_id"]] = {"chars": 0, "coverage": 0.0,
                                         "has_speech": False, "text": ""}
    return out


def _ref_cps_for_range(ref_asr_items, start, end, max_gap=0.35):
    """参考视频 [start,end] 秒内的真实语速(中文字/秒, 按**有说话时间**计)。

    用于让配音该段语速跟随参考对应段落: 参考说得快(cps 高)配音也快, 说得慢也慢。
    token 中心落在窗内即计入; 去空白后按字符数计(与 voiceover 的 max_tts_chars 口径一致)。
    分母不是窗口时长, 而是窗内逐字时间区间的并集(相邻间隔<=max_gap 秒视为连续
    说话, 更长的算停顿/静音剔除)——避免把留白/BGM 摊进语速而低估参考真实 cps。
    参考该段无口播(纯画面/BGM)-> 返回 None, 上游回退全局默认 cps。
    """
    if not ref_asr_items or end <= start:
        return None
    chars = 0
    spans = []
    for it in ref_asr_items:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        mid = (s + e) / 2.0
        if start - 0.01 <= mid < end + 0.01:
            chars += len(re.sub(r"\s+", "", str(it.get("text") or "")))
            spans.append((s, max(e, s)))
    if chars <= 0 or not spans:
        return None
    spans.sort()
    voiced = 0.0
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s - ce <= max_gap:
            ce = max(ce, e)
        else:
            voiced += ce - cs
            cs, ce = s, e
    voiced += ce - cs
    # 逐字时间戳可能退化(start==end 居多); 过短则兜底回窗口时长, 且不超过窗口
    voiced = min(voiced, end - start)
    if voiced < 0.2:
        voiced = end - start
    return round(chars / voiced, 3)


def build_segments_from_dna(dna, total_duration=None, min_seg=1.2, ref_asr_items=None):
    """DNA key_beats -> 结构级段落时间线。

    beats 之间常有空档(例:2-3s, 6-7s 未覆盖); 把整段参考时长按 beat 时长比例
    重新分配, 让各段首尾相接、总和≈参考总时长, 从而复刻整体节奏。

    ref_asr_items: 参考视频逐字 ASR(带 start/end/text)。给定时按每段 ref_time_range
    聚合出该段真实语速 ref_cps(中文字/秒), 挂到 segment; 缺失或窗内无口播 -> None。
    """
    from reference_shots import parse_key_beats
    beats = parse_key_beats(dna)
    total = float(total_duration or dna.get("duration_estimate") or 0.0)
    if not beats:
        # 无 beats: 退化成单段(整片)
        return [{"index": 1, "beat_desc": dna.get("topic_and_emotion", "") or "全片",
                 "target_duration": max(min_seg, total or 5.0),
                 "ref_cps": _ref_cps_for_range(ref_asr_items, 0.0, total) if total else None}]
    # 兜底: 部分/全部 beat 缺时间戳(key_beats 非 "MM:SS-MM:SS" 格式)时, 不能直接相减。
    # 对齐 reference_shots 的处理: 已知时长的按真实时长, 缺失的用「已知均值」或整片均分兜底。
    n = len(beats)
    raw = []
    for b in beats:
        s, e = b.get("start"), b.get("end")
        raw.append((e - s) if (s is not None and e is not None and e > s) else None)
    known = [x for x in raw if x]
    if not total or total <= 0:
        total = sum(known) if known else float(n) * 3.0  # 无任何时间戳: 每段约 3s 兜底
    avg = (sum(known) / len(known)) if known else (total / n)
    spans = [max(0.1, x if x else avg) for x in raw]
    span_sum = sum(spans) or 1.0
    segments = []
    cursor = 0.0
    for i, (b, sp) in enumerate(zip(beats, spans), 1):
        dur = round(max(min_seg, total * sp / span_sum), 2)
        s, e = b.get("start"), b.get("end")
        if s is None or e is None:  # 缺时间戳: 用累计游标合成首尾相接的展示区间
            s, e = cursor, cursor + sp
        cursor = float(e)
        segments.append({
            "index": i,
            "beat_desc": b.get("desc", "") or "",
            "ref_time_range": "{:.2f}-{:.2f}".format(float(s), float(e)),
            "target_duration": dur,
            "ref_cps": _ref_cps_for_range(ref_asr_items, float(s), float(e)),
        })
    return segments


def _slot_id(i):
    return "S{:02d}".format(i)


# ---------- 确定性 1:1 分配 ----------

def assign_deterministic(segments, candidates, speech_map=None):
    """全局贪心: 按 (段,候选) 得分降序依次锁定, 每条候选只用一次。

    speech_map 给定且开启原声优先时, 对「窗口内有可用口播」的候选加 OV_DET_BONUS,
    使叙事段更倾向选到自带原声的素材(相关性仍是主项, 加分为辅)。
    """
    prefer_ov = _prefer_original() and bool(speech_map)
    pairs = []
    for si, seg in enumerate(segments):
        for ci, cand in enumerate(candidates):
            sc = score_candidate_deterministic(seg["beat_desc"], cand)
            if prefer_ov and (speech_map.get(cand["global_asset_id"]) or {}).get("has_speech"):
                sc += _ov_det_bonus()
            pairs.append((sc, si, ci))
    pairs.sort(key=lambda x: x[0], reverse=True)
    seg_taken = {}
    cand_taken = set()
    for sc, si, ci in pairs:
        if si in seg_taken or ci in cand_taken:
            continue
        seg_taken[si] = (ci, sc)
        cand_taken.add(ci)
        if len(seg_taken) == len(segments):
            break
    result = []
    for si, seg in enumerate(segments):
        if si in seg_taken:
            ci, sc = seg_taken[si]
            result.append((seg, candidates[ci], round(sc, 4), "deterministic"))
        else:
            # 候选耗尽(候选数<段数): 该段无独占素材 -> 缺口
            result.append((seg, None, 0.0, "deterministic"))
    return result


# ---------- LLM 全局分配(带不重复约束) ----------

_LLM_PROMPT = """你是短视频剪辑导演。要用一批**用户素材片段**复刻一个参考视频的**叙事结构**(不是逐帧复刻)。
成片要求: 逻辑连贯、尽量贴近参考视频的节奏与内容; **每条用户素材最多只能用在一个段落**(不可重复使用)。

参考视频的叙事段落(按顺序, 每段附时长):
{seg_block}

可用用户素材片段:
{cand_block}
{ov_guidance}
请为每个段落分配**一条**最合适且**互不重复**的用户素材(global_asset_id)。
若某段实在没有合适且未被占用的素材, best_id 给空字符串 ""(该段会走补拍/降级)。
只输出 JSON:
{{"assignments": [{{"segment_index": 1, "best_id": "<global_asset_id 或 空>", "score": <0~1>, "reason": "<一句话>"}}, ...]}}"""

# 原声优先时追加的择偶导语: 让 LLM 在画面合理前提下优先选自带**贴题口播**的素材,
# 从而下游能保留用户真实原声。明确要求「口播须贴合该段旁白内容」以挡掉跑题闲聊。
_OV_GUIDANCE = """
【原声优先(重要)】部分素材标注了「原声口播:...」表示其画面内自带用户真实说话。
在保证画面与该段叙事合理的前提下, 请**优先**把口播内容**贴合该段旁白主题**的素材分给对应段落,
以便保留用户原声(更真实、口型可对上)。注意:
- 只有当该口播确实**贴合本段要讲的内容**时才优先(如成分段配讲成分的口播); 明显跑题的闲聊
  (如游戏/日常对话)**不要**因为它有口播就选, 宁可用无口播素材走克隆配音。
- 仍须满足互不重复与画面基本合理。
"""

# 商品展示下限导语: 与 _min_product_shots() 联动, 下限为 0(或 legacy)时不注入,
# 否则会和「原声优先」在同一个 prompt 里互相拉扯。
_PRODUCT_GUIDANCE = """【商品展示下限(重要)】不要把带口播的讲解镜头铺满全片。**配料表/包装特写/冲泡/兑牛奶/使用演示**
这类「展示商品本身」的节拍, 必须优先分给**真正展示该商品画面**的素材(即使它没有口播、要走克隆配音),
不要为了凑原声硬塞一个人物讲话镜头上去。整片**至少保留 {n} 段商品展示画面**, 否则成片全是大头讲解、
没有产品镜头, 带货效果差。
"""


def assign_llm(segments, candidates, model=None, speech_map=None, temperature=0.2):
    prefer_ov = _prefer_original() and bool(speech_map)
    seg_lines = ["- 段{} ({:.1f}s): {}".format(
        s["index"], s["target_duration"], s["beat_desc"] or "(无描述)") for s in segments]

    def _cand_line(c):
        base = "- {} [{}] {:.1f}s: {}".format(
            c["global_asset_id"], c.get("asset_type", ""), c.get("duration", 0.0),
            (c.get("text") or "")[:150])
        if prefer_ov:
            sm = speech_map.get(c["global_asset_id"]) or {}
            if sm.get("has_speech"):
                base += "  ｜原声口播:「{}」".format((sm.get("text") or "")[:60])
        return base

    cand_lines = [_cand_line(c) for c in candidates]
    _min_prod = _min_product_shots()
    guidance = (_OV_GUIDANCE if prefer_ov else "")
    if _min_prod > 0:
        guidance += _PRODUCT_GUIDANCE.format(n=_min_prod)
    prompt = _LLM_PROMPT.format(seg_block="\n".join(seg_lines),
                                cand_block="\n".join(cand_lines),
                                ov_guidance=guidance)
    text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=temperature)
    obj = loads_with_repair(text)
    import re
    by_id = {c["global_asset_id"]: c for c in candidates}
    by_norm = {re.sub(r"\s+", "", k): v for k, v in by_id.items()}

    def resolve(rid):
        if not rid:
            return None
        return by_id.get(rid) or by_norm.get(re.sub(r"\s+", "", str(rid)))

    assign_map = {}
    for a in (obj.get("assignments") or []):
        assign_map[int(a.get("segment_index", 0))] = a
    result = []
    used = set()
    for seg in segments:
        a = assign_map.get(seg["index"], {})
        cand = resolve(a.get("best_id"))
        # 兜底强制不重复: 若 LLM 选了已用素材, 视为无效
        if cand and cand["global_asset_id"] in used:
            cand = None
        if cand:
            used.add(cand["global_asset_id"])
            result.append((seg, cand, round(float(a.get("score", 0.5)), 4), "llm"))
        else:
            result.append((seg, None, 0.0, "llm_gap"))
    # 对 LLM 留下的缺口, 用确定性在**未用**候选里补一条(仍不重复)
    leftover = [c for c in candidates if c["global_asset_id"] not in used]
    if leftover:
        det = assign_deterministic([r[0] for r in result if r[1] is None], leftover,
                                   speech_map=speech_map)
        det_iter = iter(det)
        patched = []
        for seg, cand, sc, method in result:
            if cand is None:
                dseg, dcand, dsc, _ = next(det_iter)
                if dcand is not None:
                    used.add(dcand["global_asset_id"])
                    patched.append((seg, dcand, dsc, "deterministic_fallback"))
                    continue
            patched.append((seg, cand, sc, method))
        result = patched
    return result


# 催单/促销类节拍关键词(beat_desc 命中即视为 CTA 段)
_CTA_RE = re.compile(r"催单|催促|下单|赶紧|抢购|凑单|稀缺|链接|购买|囤")


def reorder_cta_last(segments):
    """带货叙事整形: 催单(CTA)节拍集中到结尾。

    参考视频的节拍顺序有时把催单插在中段(实测黑巧咖: S03 凑单催价在第 3 段,
    S06 催单在结尾, 中间隔着成分/喝法两个卖点段), 复刻后成片"先催单再讲卖点
    最后又催单", 叙事割裂。整形规则: 命中 _CTA_RE 的节拍整体后移到片尾(保持
    彼此相对顺序), 其余节拍相对顺序不变; 之后按新顺序重编 index/slot_id。
    ref_time_range/ref_cps 跟着节拍走(语速仍随参考对应段)。CTA 本就全在结尾
    时不动(等价 no-op)。
    """
    cta_ids = {id(s) for s in segments if _CTA_RE.search(str(s.get("beat_desc") or ""))}
    if not cta_ids:
        return segments
    rest = [s for s in segments if id(s) not in cta_ids]
    cta = [s for s in segments if id(s) in cta_ids]
    new = rest + cta
    if [id(s) for s in new] == [id(s) for s in segments]:
        return segments
    moved = [s.get("beat_desc") for s in cta]
    for i, s in enumerate(new, 1):
        s["index"] = i
        s["slot_id"] = _slot_id(i)
    print("[edit_planner] 催单节拍集中到结尾: {} 段后移 -> {}".format(
        len(cta), "; ".join(str(m)[:24] for m in moved)), flush=True)
    return new


def _apply_slot_pins(out_segments, candidates):
    """按 WHQ_SLOT_PIN 强制把某槽位钉到指定 global_asset_id(人工修文案/接续用)。

    格式: "S02=2025-04-20 222615::A2[,S05=<id>...]"(槽位为 reorder 后的最终 slot_id)。
    被钉候选若已分给别的槽, 那个槽让出(转缺口), 保证 1:1 不重复。找不到候选/槽则跳过。
    """
    pin_env = os.getenv("WHQ_SLOT_PIN", "").strip()
    if not pin_env:
        return out_segments
    by_id = {c["global_asset_id"]: c for c in candidates}
    by_norm = {re.sub(r"\s+", "", k): v for k, v in by_id.items()}
    seg_by_slot = {s["slot_id"]: s for s in out_segments}
    for token in pin_env.split(","):
        if "=" not in token:
            continue
        slot, aid = (x.strip() for x in token.split("=", 1))
        seg = seg_by_slot.get(slot)
        cand = by_id.get(aid) or by_norm.get(re.sub(r"\s+", "", aid))
        if not seg or not cand:
            print("[edit_planner] SLOT_PIN 跳过(槽或素材不存在): {}={}".format(slot, aid), flush=True)
            continue
        # 该候选若已占用别的槽, 让那个槽转缺口(维持 1:1)
        for other in out_segments:
            if other is not seg and (other.get("best_candidate") or {}).get("global_asset_id") == cand["global_asset_id"]:
                other["best_candidate"] = None
                other["is_gap"] = True
                other["method"] = "pinned_displaced"
                other["score"] = 0.0
        seg["best_candidate"] = cand
        seg["is_gap"] = False
        seg["method"] = "pinned"
        seg["score"] = 1.0
        seg["pinned"] = True  # 硬约束: 后续连贯修复不得改动被钉槽位
        print("[edit_planner] SLOT_PIN {} <- {}".format(slot, cand["global_asset_id"]), flush=True)
    return out_segments


def _assigned_to_out_segments(assigned, gap_threshold, det_gap_threshold,
                              candidates, cta_last=True):
    """把 (seg,cand,score,method) 列表整形为最终 out_segments(含 reorder + slot pin)。"""
    out_segments = []
    for seg, cand, score, method in assigned:
        thr = det_gap_threshold if method.startswith("deterministic") else gap_threshold
        is_gap = (cand is None) or (score < thr)
        out_segments.append({
            "index": seg["index"],
            "slot_id": _slot_id(seg["index"]),
            "beat_desc": seg["beat_desc"],
            "ref_time_range": seg.get("ref_time_range", ""),
            "target_duration": seg["target_duration"],
            "ref_cps": seg.get("ref_cps"),
            "best_candidate": cand,
            "score": score,
            "method": method,
            "is_gap": is_gap,
        })
    if cta_last:
        out_segments = reorder_cta_last(out_segments)
    out_segments = _apply_slot_pins(out_segments, candidates)
    return out_segments


# 计划择优目标: **原声率为绝对主目标**, 连贯只作平局项 + 给出最弱衔接供人工 pin(见 score_plan)。

_COHERENCE_PROMPT = """你是短视频带货成片的叙事质检员。下面是一条成片**按播放顺序**的每段内容
(有的段会用素材原声口播, 标了「原声:」; 没有的段只给了该段要讲的主题):

{seq_block}

请**只评估叙事顺序是否承接**: 钩子→体验/卖点→催单的推进是否自然, 相邻两段主题是否接得上、
有没有前言不搭后语/主题跳跃/原声内容与该段主题跑偏。
**重要: 原声段是真人实录口语, 允许口语化、不书面、有语气词或 ASR 小瑕疵——不要因为"不够书面/
念白粗糙"而扣分, 只看内容主题是否承接。** 口语顺畅但主题割裂要低分; 口语粗糙但主题承接要高分。
只输出 JSON: {{"coherence": <0~1 的小数, 1=主题承接非常顺, 0=完全割裂>, "weakest": "<最不连贯的相邻段, 如 S01->S02, 无则空>", "reason": "<一句话>"}}"""


def _plan_coherence(out_segments, speech_map, model):
    """LLM 对整条(排序后)方案打叙事连贯分 0~1。失败返回 (0.5, {}) 中性值。"""
    if not (_HAS_LLM and out_segments):
        return 0.5, {}
    lines = []
    for s in out_segments:
        cand = s.get("best_candidate") or {}
        sm = speech_map.get(cand.get("global_asset_id")) or {}
        beat = (s.get("beat_desc") or "")[:40]
        if sm.get("has_speech"):
            lines.append("- {} 原声:「{}」(该段主题: {})".format(
                s["slot_id"], (sm.get("text") or "")[:50], beat))
        else:
            lines.append("- {} (克隆配音, 该段主题: {})".format(s["slot_id"], beat))
    prompt = _COHERENCE_PROMPT.format(seq_block="\n".join(lines))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=0.2)
        obj = loads_with_repair(text)
        cval = float(obj.get("coherence"))
        return max(0.0, min(1.0, cval)), obj
    except Exception as exc:  # noqa: BLE001 —— 打分失败不阻断, 回中性值
        print("[edit_planner] 连贯打分失败(用中性0.5): {}".format(str(exc)[:120]), flush=True)
        return 0.5, {}


def score_plan(out_segments, speech_map, model):
    """原声优先择优打分（带商品展示下限）。返回 (total, detail)。

    **原声率为主目标**, 连贯只作极小平局项; 但加一道**商品展示下限**硬约束:
      成片至少 WHQ_MIN_PRODUCT_SHOTS 段是「无口播的商品展示画面」(配料/包装/冲泡/使用演示)。
      原声优先会把带口播的讲解镜头铺满全片、挤掉商品镜头, 故对不满足下限的方案按缺口重罚
      (每缺 1 条商品展示扣 1.0, 远超多 1 段原声带来的 1/n 收益), 迫使择优保留足够商品镜头。
      total = 原声率 + 0.01*连贯 + 0.001*贴合 - 1.0*商品展示缺口
    """
    n = len(out_segments) or 1
    ov = sum(1 for s in out_segments
             if (speech_map.get((s.get("best_candidate") or {}).get("global_asset_id")) or {}).get("has_speech"))
    ov_ratio = ov / n
    # 商品展示段 = 分到的候选无可用口播(即非讲解口播镜头, 通常是配料/包装/冲泡/使用演示等 B-roll)
    n_broll = sum(1 for s in out_segments
                  if (s.get("best_candidate"))
                  and not (speech_map.get((s.get("best_candidate") or {}).get("global_asset_id")) or {}).get("has_speech"))
    min_product = min(_min_product_shots(), max(0, n - 1))  # 至少留 1 段可给口播
    shortfall = max(0, min_product - n_broll)
    fits = [float(s.get("score") or 0.0) for s in out_segments]
    mean_fit = min(1.0, sum(fits) / n) if fits else 0.0
    coherence, cdetail = _plan_coherence(out_segments, speech_map, model)
    total = ov_ratio + 0.01 * coherence + 0.001 * mean_fit - _product_shot_penalty() * shortfall
    return total, {"coherence": round(coherence, 3), "ov_ratio": round(ov_ratio, 3),
                   "mean_fit": round(mean_fit, 3), "product_shots": n_broll,
                   "product_short": shortfall, "weakest": cdetail.get("weakest", "")}


_SLOT_RE = re.compile(r"S\d{2}")

_BRIDGE_PROMPT = """短视频带货成片里, {prev_slot} 段的内容是:「{prev_text}」。
下一段 {break_slot} 当前用的内容「{cur_text}」与上一段衔接不顺(该段原定主题「{break_beat}」仅供参考)。
下面是若干可改用的**用户原声口播**候选:
{options}
请选出最能**自然承接 {prev_slot}、让叙事顺下去**的一条。**承接连贯优先于贴合原定主题**——
只要不是明显无关的闲聊(如游戏/日常对话)即可选; 原定主题对不上没关系。实在都是无关闲聊才给空。
只输出 JSON: {{"best_id": "<候选 id 或 空>", "reason": "<一句话>"}}"""


def _pick_bridge_candidate(prev_seg, break_seg, pool, speech_map, model):
    """让 LLM 从未用口播候选里挑一条最能承接上一段的, 返回 candidate 或 None。"""
    if not (_HAS_LLM and pool):
        return None
    prev_c = prev_seg.get("best_candidate") or {}
    prev_sm = speech_map.get(prev_c.get("global_asset_id")) or {}
    prev_text = prev_sm.get("text") or (prev_seg.get("beat_desc") or "")
    cur_c = break_seg.get("best_candidate") or {}
    cur_sm = speech_map.get(cur_c.get("global_asset_id")) or {}
    opts = ["- {}: 「{}」".format(c["global_asset_id"],
            (speech_map.get(c["global_asset_id"]) or {}).get("text", "")[:50]) for c in pool[:12]]
    prompt = _BRIDGE_PROMPT.format(
        prev_slot=prev_seg["slot_id"], prev_text=prev_text[:50],
        break_slot=break_seg["slot_id"], break_beat=(break_seg.get("beat_desc") or "")[:40],
        cur_text=(cur_sm.get("text") or "")[:40], options="\n".join(opts))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=0.2)
        obj = loads_with_repair(text)
        bid = re.sub(r"\s+", "", str(obj.get("best_id") or ""))
        for c in pool:
            if re.sub(r"\s+", "", c["global_asset_id"]) == bid:
                return c
    except Exception as exc:  # noqa: BLE001
        print("[edit_planner]   桥接候选选取失败: {}".format(str(exc)[:120]), flush=True)
    return None


def _repair_coherence(out_segments, candidates, speech_map, model, cur_total, cur_detail,
                      max_rounds=2):
    """最弱链定向修复: 针对连贯打分标出的最弱衔接段, 让 LLM 从未用口播候选里挑桥接段换入重评。

    纯 best-of-K 重采样常采不到能桥接的排列(如把体验反转口播放到 S02)。这里做**局部搜索**:
    取 weakest('SXX->SYY')里的破坏段 SYY, 让 LLM 从未占用且自带口播的候选里挑一条最能承接
    SXX 的换入(_pick_bridge_candidate), 再全局重评(含连贯); 另试与前一段互换素材。采纳提升总分者。
    每轮至多 2 次连贯 LLM 调用, 最多 max_rounds 轮, 控成本。返回 (out, total, detail)。
    """
    import copy
    best_out, best_total, best_detail = out_segments, cur_total, cur_detail
    for _ in range(max_rounds):
        weakest = best_detail.get("weakest") or ""
        slots = _SLOT_RE.findall(weakest)
        if not slots:
            break
        break_slot = slots[-1]  # 'SXX->SYY' 里破坏承接的是后一段 SYY
        idx = next((i for i, s in enumerate(best_out) if s["slot_id"] == break_slot), None)
        if idx is None:
            break
        if best_out[idx].get("pinned"):
            print("[edit_planner]   {} 已被人工 pin, 跳过连贯修复(尊重硬约束)".format(break_slot),
                  flush=True)
            break
        used = {(s.get("best_candidate") or {}).get("global_asset_id") for s in best_out}
        pool = [c for c in candidates
                if c["global_asset_id"] not in used
                and (speech_map.get(c["global_asset_id"]) or {}).get("has_speech")]
        pool.sort(key=lambda c: (speech_map.get(c["global_asset_id"]) or {}).get("chars", 0),
                  reverse=True)
        round_best = None
        pick = _pick_bridge_candidate(best_out[idx - 1] if idx > 0 else best_out[idx],
                                      best_out[idx], pool, speech_map, model)
        # 候选试换池: LLM 选的桥接段 + 口播字数最多的前几条(兜底, 防 LLM 漏选)
        replace_cands = []
        if pick is not None:
            replace_cands.append(pick)
        for c in pool[:4]:
            if c["global_asset_id"] not in {x["global_asset_id"] for x in replace_cands}:
                replace_cands.append(c)
        trials = [("replace", c) for c in replace_cands[:5]]
        if idx > 0 and not best_out[idx - 1].get("pinned"):
            trials.append(("swap_prev", None))

        def _try(trial):
            kind, c = trial
            cand_out = copy.deepcopy(best_out)
            if kind == "replace":
                cand_out[idx]["best_candidate"] = c
                cand_out[idx]["method"] = "coherence_repair"
                cand_out[idx]["is_gap"] = False
                cand_out[idx]["score"] = max(cand_out[idx].get("score", 0.0), 0.5)
                tag = c["global_asset_id"]
            else:  # swap_prev
                cand_out[idx]["best_candidate"], cand_out[idx - 1]["best_candidate"] = (
                    cand_out[idx - 1]["best_candidate"], cand_out[idx]["best_candidate"])
                cand_out[idx]["method"] = cand_out[idx - 1]["method"] = "coherence_repair"
                tag = "swap({}<->{})".format(best_out[idx - 1]["slot_id"], break_slot)
            t, d = score_plan(cand_out, speech_map, model)
            print("[edit_planner]   修复试换 {} {}: 总分{:.3f} 连贯{}".format(
                break_slot, tag, t, d["coherence"]), flush=True)
            return (t, cand_out, d)

        # 每个试换都是一次独立的连贯打分 LLM 调用 -> 并发评估(原串行是本阶段主要耗时)
        for r in parallel_map(_try, trials):
            if isinstance(r, Exception):
                print("[edit_planner]   试换评估失败(跳过): {}".format(str(r)[:120]), flush=True)
                continue
            if round_best is None or r[0] > round_best[0]:
                round_best = r
        if round_best and round_best[0] > best_total + 1e-6:
            best_total, best_out, best_detail = round_best
            print("[edit_planner]   修复采纳: {} 总分->{:.3f} 连贯{} 弱点{}".format(
                break_slot, best_total, best_detail["coherence"],
                best_detail.get("weakest") or "无"), flush=True)
        else:
            break  # 本轮无提升, 停止
    return best_out, best_total, best_detail


def _suggest_pin(out_segments, candidates, speech_map, model, weakest):
    """打印「建议人工 pin」: 对 LLM 判出的最弱衔接段, 给出一条可锁定的桥接口播候选。

    连贯是主观项, 自动择优只保证原声最多; 若仍有断裂衔接, 这里把最弱段和一条建议的
    桥接素材打印成现成的 WHQ_SLOT_PIN, 用户想改一句就能一键锁定重跑。
    """
    slots = _SLOT_RE.findall(weakest or "")
    if not slots:
        return
    break_slot = slots[-1]
    idx = next((i for i, s in enumerate(out_segments) if s["slot_id"] == break_slot), None)
    if idx is None:
        return
    used = {(s.get("best_candidate") or {}).get("global_asset_id") for s in out_segments}
    pool = [c for c in candidates
            if c["global_asset_id"] not in used
            and (speech_map.get(c["global_asset_id"]) or {}).get("has_speech")]
    pool.sort(key=lambda c: (speech_map.get(c["global_asset_id"]) or {}).get("chars", 0),
              reverse=True)
    pick = _pick_bridge_candidate(out_segments[idx - 1] if idx > 0 else out_segments[idx],
                                  out_segments[idx], pool, speech_map, model)
    if not pick:
        print("[edit_planner] 建议人工复核衔接: {} (未找到更合适的桥接口播)".format(weakest),
              flush=True)
        return
    txt = (speech_map.get(pick["global_asset_id"]) or {}).get("text", "")[:40]
    print("[edit_planner] 建议人工 pin: 最弱衔接 {}; 如需改善可锁定 {}=「{}」".format(
        weakest, break_slot, txt), flush=True)
    print("[edit_planner]   WHQ_SLOT_PIN=\"{}={}\"".format(break_slot, pick["global_asset_id"]),
          flush=True)


def plan_edit(dna, candidates, total_duration=None, use_llm=True, model=None,
              gap_threshold=0.4, det_gap_threshold=0.06, min_seg=1.2, ref_asr_items=None,
              cta_last=True, speech_records=None):
    segments = build_segments_from_dna(dna, total_duration=total_duration, min_seg=min_seg,
                                       ref_asr_items=ref_asr_items)
    speech_map = build_speech_map(candidates, speech_records) if _prefer_original() else {}
    if speech_map:
        n_ov = sum(1 for v in speech_map.values() if v.get("has_speech"))
        print("[edit_planner] 原声优先: {} 条候选自带可用口播(共 {} 条)".format(
            n_ov, len(candidates)), flush=True)
    # best-of-K(原声优先择优): 采样 K 个分配方案, 按**原声率**择优(连贯仅平局项); 对选中的
    # 原声最多方案做连贯定向修复(在不减原声的前提下把断裂衔接换成桥接口播), 再打印 LLM 判出的
    # 最弱衔接作为「建议人工 pin」——连贯这种主观项交给人工一键锁定, 不让噪声判分挤掉原声。
    K = int(os.getenv("WHQ_PLAN_BEST_OF_K", "1"))
    if use_llm and _HAS_LLM and K > 1:
        temps = [0.2, 0.5, 0.8, 0.35, 0.65, 0.95]

        def _sample(i):
            t = temps[i % len(temps)]
            assigned = assign_llm(segments, candidates, model=model,
                                  speech_map=speech_map, temperature=t)
            out = _assigned_to_out_segments(assigned, gap_threshold, det_gap_threshold,
                                            candidates, cta_last=cta_last)
            total, detail = score_plan(out, speech_map, model)
            print("[edit_planner] 候选#{} (T={}) 原声{}/商品展示{}/连贯{}/贴合{} 弱点:{}".format(
                i + 1, t, detail["ov_ratio"], detail["product_shots"], detail["coherence"],
                detail["mean_fit"], detail["weakest"] or "无"), flush=True)
            return (total, out, detail)

        # K 个方案各自是独立的一次 LLM 分配+一次连贯打分, 彼此无依赖 -> 并发采样(网络等待型)
        cands = []
        for r in parallel_map(_sample, range(K)):
            if isinstance(r, Exception):
                print("[edit_planner] 候选采样失败(跳过): {}".format(str(r)[:120]), flush=True)
                continue
            cands.append(r)
        if not cands:  # 全部采样失败 -> 退确定性分配, 不阻断主流程
            print("[edit_planner] best-of-K 全部失败, 回退确定性分配", flush=True)
            assigned = assign_deterministic(segments, candidates, speech_map=speech_map)
            return {"segments": _assigned_to_out_segments(
                assigned, gap_threshold, det_gap_threshold, candidates, cta_last=cta_last)}
        if os.getenv("WHQ_PLAN_COHERENCE_REPAIR", "1") not in ("0", "false", "False"):
            # 连贯修复的基线取**综合总分最高**(已计入商品展示下限惩罚)的方案, 避免在违反
            # 商品展示下限的方案上修复; 修复中换口播桥接段若打破下限, 会被总分惩罚而不被选中。
            base = max(cands, key=lambda c: c[0])
            print("[edit_planner] 对综合最优候选(原声{}/商品展示{}/连贯{})做连贯定向修复".format(
                base[2]["ov_ratio"], base[2]["product_shots"], base[2]["coherence"]), flush=True)
            rep_out, rep_total, rep_detail = _repair_coherence(
                base[1], candidates, speech_map, model, base[0], base[2])
            cands.append((rep_total, rep_out, rep_detail))
        best = max(cands, key=lambda c: c[0])  # 综合总分(原声率主导+商品展示下限, 连贯平局)
        print("[edit_planner] best-of-{} 选定 原声{}/商品展示{}/连贯{}".format(
            K, best[2]["ov_ratio"], best[2]["product_shots"], best[2]["coherence"]), flush=True)
        _suggest_pin(best[1], candidates, speech_map, model, best[2].get("weakest"))
        return {"segments": best[1]}
    if use_llm and _HAS_LLM:
        assigned = assign_llm(segments, candidates, model=model, speech_map=speech_map)
    else:
        assigned = assign_deterministic(segments, candidates, speech_map=speech_map)
    out_segments = _assigned_to_out_segments(assigned, gap_threshold, det_gap_threshold,
                                             candidates, cta_last=cta_last)
    return {"segments": out_segments}


def main(argv=None):
    from reference_shots import load_dna
    from asset_index import load_candidates
    ap = argparse.ArgumentParser(description="结构级复刻规划(不重复分配)")
    ap.add_argument("--dna", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--total-duration", type=float)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    dna = load_dna(args.dna)
    cands = load_candidates(args.assets)
    plan = plan_edit(dna, cands, total_duration=args.total_duration,
                     use_llm=not args.no_llm, model=args.model)
    if args.out:
        json.dump(plan, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", args.out)
    for s in plan["segments"]:
        bc = s["best_candidate"]
        cid = bc["global_asset_id"] if bc else "(缺口)"
        flag = " GAP" if s["is_gap"] else ""
        print("  段{:>2} {:>4.1f}s <- {:28s} score={:.3f} [{}]{}  {}".format(
            s["index"], s["target_duration"], cid, s["score"], s["method"], flag,
            (s["beat_desc"] or "")[:32]))


if __name__ == "__main__":
    main()
