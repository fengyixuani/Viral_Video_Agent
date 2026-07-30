"""voiceover — 「两者结合」配音: 用用户素材真实口播(ASR)做内容底座, LLM 顺滑改写/
补全成贴近参考视频节奏的口播脚本, 再复用主流水线的 build_tts_overlay.py(CosyVoice3
零样本声音克隆)把配音叠加到无声 base 上。

流程:
  1. load_user_speech: 读 all_source_asr.json, 过滤出有意义的真实口播句(噪声/单字丢弃)。
  2. select_prompt: 挑一段够长(>=8s)的真实口播做**声音克隆参考**(prompt.wav+prompt_asr)。
  3. generate_script: LLM 结合 [真实口播内容 + DNA叙事 + 每段画面/时长] 生成逐段口播脚本。
  4. build_tts_plan: 按 base 累计时间线给每段配 start/end/duration, 组装 TTS_PLAN。
  5. add_voiceover: 调 generation/build_tts_overlay.py(TTS_PLAN_PATH 直供), 产出有声视频。

无用户口播时优雅降级: 脚本纯 LLM 生成(基于画面/商品), 声音克隆参考取最长可用音频。
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import COMMON, VENDOR, REPO, FFMPEG, legacy  # noqa: F401

try:
    from pipeline_utils import ask_qianfan, loads_with_repair
    _HAS_LLM = True
except Exception:  # pragma: no cover
    _HAS_LLM = False

import voice_policy

TTS_PYTHON = os.getenv("TTS_PYTHON", "/root/miniconda3/envs/viral-split-tts/bin/python")
BUILD_TTS_OVERLAY = os.path.join(REPO, "generation", "build_tts_overlay.py")
PROMPT_MIN_SECONDS = float(os.getenv("TTS_PROMPT_MIN_SECONDS", "8"))
PROMPT_MAX_SECONDS = float(os.getenv("TTS_PROMPT_MAX_SECONDS", "18"))
_NOISE_MAX_CHARS = 4   # <=4 字且无商品语义的多半是噪声/口水音

# 每段语速(中文字/秒): 优先用参考视频该段真实语速 ref_cps(让配音快慢跟随参考),
# 缺失(参考纯画面/BGM 段)回退全局默认; 再钳进合理区间防参考漏字/极快念白把某段推爆或过慢。
CPS_DEFAULT = float(os.getenv("TTS_CHARS_PER_SECOND", "3.8"))
CPS_MIN = float(os.getenv("TTS_CPS_MIN", "2.5"))
CPS_MAX = float(os.getenv("TTS_CPS_MAX", "6.0"))
SCRIPT_REVIEW_APPLY_REWRITE = os.getenv(
    "WHQ_SCRIPT_REVIEW_APPLY_REWRITE", "1"
) not in ("0", "false", "False")


def _seg_cps(ref_cps):
    """该段目标语速(字/秒): ref_cps 有效则用之, 否则全局默认, 统一钳进 [MIN,MAX]。"""
    try:
        cps = float(ref_cps) if ref_cps else CPS_DEFAULT
    except (TypeError, ValueError):
        cps = CPS_DEFAULT
    return min(CPS_MAX, max(CPS_MIN, cps))


def _norm(text):
    return re.sub(r"\s+", "", str(text or "").strip())


def load_user_speech(asr_path):
    """返回有意义的真实口播记录列表(按文本长度降序)。"""
    if not asr_path or not os.path.exists(asr_path):
        return []
    records = json.load(open(asr_path, encoding="utf-8"))
    if not isinstance(records, list):
        return []
    good = []
    for r in records:
        text = _norm(r.get("asr_text"))
        if not text:
            items = r.get("asr_items") or []
            text = _norm("".join(str(i.get("text") or "") for i in items))
        if len(text) <= _NOISE_MAX_CHARS:
            continue
        good.append({
            "source_video_id": r.get("source_video_id", ""),
            "source_path": r.get("source_path", ""),
            "audio_path": r.get("audio_path", ""),
            "duration": float(r.get("duration") or 0.0),
            "text": text,
            "asr_items": r.get("asr_items") or [],
        })
    good.sort(key=lambda x: len(x["text"]), reverse=True)
    return good


def _speech_range(rec):
    starts, ends = [], []
    for it in rec.get("asr_items", []) or []:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        if e > s:
            starts.append(s)
            ends.append(e)
    if starts:
        return min(starts), max(ends)
    return 0.0, rec.get("duration", 0.0)


def select_prompt(speech_records, out_dir):
    """挑最合适的一段真实口播做声音克隆参考, 抽 wav + 写 prompt_asr。"""
    os.makedirs(out_dir, exist_ok=True)
    if not speech_records:
        return None, None
    # 打分: 时长落在 [min,max] 加分, 文本长度加分
    best = None
    for rec in speech_records:
        if not rec.get("audio_path") or not os.path.exists(rec["audio_path"]):
            continue
        s, e = _speech_range(rec)
        dur = max(0.0, e - s)
        usable = dur if dur > 0 else rec["duration"]
        if usable < PROMPT_MIN_SECONDS:
            continue
        capped = min(usable, PROMPT_MAX_SECONDS)
        score = min(len(rec["text"]), 90) + capped * 6
        if PROMPT_MIN_SECONDS <= usable <= PROMPT_MAX_SECONDS:
            score += 30
        cand = (score, rec, s, capped)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None, None
    _score, rec, start, dur = best
    prompt_wav = os.path.join(out_dir, "prompt.wav")
    prompt_asr = os.path.join(out_dir, "prompt_asr.json")
    # prompt_text 必须严格对应 prompt_wav 的**那段**音频, 否则 CosyVoice 零样本会
    # "接着念参考"而忽略目标文本(实测: 转写用整条 rec[text] 而音频只截了 [start,start+dur],
    # 二者错配 -> 合成出来的是参考内容本身)。这里只取落在 [start, start+dur] 窗内的分词拼成转写。
    clip_end = start + dur
    clip_words = []
    for it in rec.get("asr_items", []) or []:
        try:
            s = float(it.get("start"))
        except (TypeError, ValueError):
            continue
        if start - 0.01 <= s < clip_end + 0.01:
            clip_words.append(str(it.get("text") or ""))
    clip_text = _norm("".join(clip_words)) or _norm(rec["text"])
    # 真实口播多带环境底噪(如煎制/油爆声), CosyVoice 零样本会把底噪一并克隆进每句 TTS,
    # 在切镜起点听感为"爆音/提示音"。抽参考时先降噪+高通+响度归一, 只留干净人声克隆。
    denoise = os.getenv("TTS_PROMPT_DENOISE",
                        "highpass=f=70,afftdn=nr=12:nf=-25,loudnorm=I=-18:TP=-2:LRA=11")
    # 参考**尾部补一小段静音**: 否则 CosyVoice 零样本会把参考结尾词(如"你信你信")的尾音
    # 泄漏到目标句开头(实测每句开头多一个"信/进"音节)。补静音让模型明确"参考已结束",
    # 实测(ASR 回验)可消除该首音节。可用 TTS_PROMPT_TAIL_SILENCE 调整/置 0 关闭。
    tail = float(os.getenv("TTS_PROMPT_TAIL_SILENCE", "0.4"))
    af = denoise + (",apad=pad_dur={:.3f}".format(tail) if tail > 0 else "")
    # -ss/-t 放在 -i 前(输入侧): 精确只读 dur 秒源音频, 之后 apad 追加静音 -> 输出 dur+tail。
    subprocess.run([FFMPEG, "-y", "-ss", "{:.3f}".format(start), "-t", "{:.3f}".format(dur),
                    "-i", rec["audio_path"], "-af", af,
                    "-ac", "1", "-ar", "16000", prompt_wav],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    json.dump({
        "model": "source_material_asr", "device": "source_material", "dtype": "source_material",
        "source_video_id": rec.get("source_video_id", ""),
        "source_path": rec.get("source_path", ""),
        "audio_path": rec.get("audio_path", ""),
        "results": [{"language": "Chinese", "text": clip_text, "time_stamps": None}],
    }, open(prompt_asr, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[voiceover] 声音克隆参考: {} ({:.1f}s) 转写={}".format(
        rec.get("source_video_id"), dur, repr(clip_text)[:40]))
    return prompt_wav, prompt_asr


_SCRIPT_PROMPT = """你是短视频带货口播导演。要给一条「复刻参考爆款视频」的成片写**配音口播脚本**。

成片中下面这些段落已**锁定使用用户素材原声**(text 就是素材里的原话, 不要为它们生成文案,
但你为其余段落写文案时必须与这些原话前后连贯):
{locked_block}

其余需要生成配音文案的段落如下(每段已定时长, 单位秒)。注意: 每段的「节拍说明」来自**另一条参考视频**,
只用来借它的**叙事结构**(开场钩子→体验→价格→成分→用法→催单), 里面出现的具体事实(成分名/百分比/
数字/喝法/品类) 是参考视频那个商品的, **不属于本商品, 严禁照搬**:
{seg_block}

商品名: {product}

用户真实口播素材(**唯一事实来源**, 卖点/成分/数字/用法只能出自这里):
{speech_block}

参考视频叙事基调(只借基调, 不借事实): {topic}

事实红线(最重要):
- text 中出现的每一个可核查断言(成分、含量、百分比、价格、功效、喝法、品类对比), 都必须能在上面
  「用户真实口播素材」或该段「本段实际画面/素材内容」里找到出处; 找不到出处的一律不写。
- **喝法/搭配动作(如兑茶、兑东方树叶、兑牛奶、兑某某饮料)必须是本段素材画面/口播里真实出现过的**;
  用户素材没做这个动作就**绝对不能写**(例:参考视频兑的是东方树叶茶, 但本商品用户没兑茶, 就不许写"兑茶")。
- 参考节拍说明里的事实(如某成分含量、兑某饮料的喝法)如果用户口播/素材没有, 就替换成用户素材里
  真实存在的对应卖点; 没有对应卖点就只描述画面/氛围, 不下断言。
- 喝法/品类逻辑必须符合本商品常识, 不得把参考商品的吃喝搭配安到本商品头上。

其他要求:
1. 逐段产出一句口播 text, 读起来自然、口语化, 整体连贯成一条完整的带货叙事。
{strict_block}2. 每段都给了「目标字数≤N」(不同段语速不同, 这是为了跟随参考视频对应段落的快慢节奏);
   该段 text 长度必须不超过它的目标字数, 宁短勿长。字数多的段说得快、字数少的段说得慢, 属正常。
3. 优先复用用户真实口播里的卖点/表达; 缺失处只作画面/氛围衔接, 不要编造不存在的信息。
4. 不要堆叠标点、不要括号/引号/省略号; 每句给一个 tone(如 轻快安利/自然解释/强调提醒/收束种草)。
5. 每句必须给 basis: 逐条说明这句 text 里每个断言的出处(引用用户口播原句片段), 纯衔接句写"画面衔接, 无事实断言"。
6. 个别段标注了「该段素材自带原声口播」: 若那段原话内容贴合整条叙事, 请给该段输出 "use_original": true
   (不用写 text, 会直接用素材原声); 若原话与整体文案脱节才写克隆 text 替代。
7. 只输出 JSON:
{{"items": [{{"slot_id": "S01", "text": "...", "tone": "...", "basis": "断言A出自用户口播「...」; 断言B出自「...」", "use_original": false}}, ...]}}"""


# —— 事实核验: 文案中的可核查数字断言必须能在用户真实口播里找到出处 ——
# 参考视频节拍(beat_desc)带着参考商品的事实(51.7%膳食纤维/29.9元6盒/兑奶像抹茶奶绿...),
# LLM 很容易把它们照搬给本商品(实测黑巧咖成片翻车)。数字类断言可确定性核验: 提取
# text 里的数字 token, 逐个在用户口播语料里找子串, 找不到=编造/搬参考 -> 责令重写。
_NUM_TOKEN_RE = re.compile(
    r"百分之[零一二三四五六七八九十百千点两0-9]+"
    r"|[0-9]+(?:\.[0-9]+)?%?"
    r"|[一二三四五六七八九十百千两]{1,6}(?:点[一二三四五六七八九十]+)?"
    r"(?:块|元|折|盒|袋|包|条|斤|克|毫升|升|倍|周|天|年|个月|小时)"
)


def _fact_violations(text, corpus):
    """返回 text 中在 corpus(用户口播全集, 已去空白)找不到出处的数字断言 token。"""
    bad = []
    for tok in _NUM_TOKEN_RE.findall(_norm(text)):
        if tok not in corpus:
            bad.append(tok)
    return bad


_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _clean(text):
    return _PUNCT_RE.sub("", _norm(text))


def _ref_leak_violations(text, user_corpus, ref_corpus, n=4):
    """返回 text 中「在参考视频口播里出现、但用户口播里没有」的片段(疑似照搬参考事实)。

    数字断言之外的搬运(如参考的喝法"兑牛奶像抹茶奶绿"、成分名)没有数字可查,
    用 n-gram 对照兜底: 连续 n 字命中参考语料且不命中用户语料 -> 泄漏。
    """
    if not ref_corpus:
        return []
    body = _clean(text)
    flags = [False] * len(body)
    for i in range(0, max(0, len(body) - n + 1)):
        gram = body[i:i + n]
        if gram in ref_corpus and gram not in user_corpus:
            for j in range(i, i + n):
                flags[j] = True
    leaks, cur = [], ""
    for ch, f in zip(body, flags):
        if f:
            cur += ch
        elif cur:
            leaks.append(cur)
            cur = ""
    if cur:
        leaks.append(cur)
    return leaks


def _fact_check_script(by_slot, segments, speech_records, product_name="", model=None,
                       rounds=2, ref_text=""):
    """确定性核验 + 违规段责令 LLM 重写(最多 rounds 轮)。返回 (by_slot, report)。

    两类违规: ①数字断言在用户口播查无出处 ②非数字片段照搬参考视频口播(n-gram 泄漏)。
    """
    corpus = _norm("".join(r.get("text", "") for r in speech_records))
    user_clean = _clean("".join(r.get("text", "") for r in speech_records))
    ref_clean = _clean(ref_text)
    tc_by_slot = {s["slot_id"]: max(6, int(float(s["target_duration"]) * _seg_cps(s.get("ref_cps"))))
                  for s in segments}
    report = {}
    for round_no in range(rounds + 1):
        bad_slots = {}
        leak_by_slot = {}
        for sid, it in by_slot.items():
            viol = _fact_violations(it.get("text", ""), corpus)
            leaks = _ref_leak_violations(it.get("text", ""), user_clean, ref_clean)
            if viol or leaks:
                bad_slots[sid] = viol + ["搬参考:" + x for x in leaks]
                leak_by_slot[sid] = leaks
        report = {sid: {"unverified_claims": v} for sid, v in bad_slots.items()}
        if not bad_slots or round_no >= rounds or not _HAS_LLM:
            break
        payload = [{"slot_id": sid,
                    "text": by_slot[sid]["text"],
                    "目标字数上限": tc_by_slot.get(sid, 20),
                    "违规断言(查无出处数字/照搬参考)": viol} for sid, viol in bad_slots.items()]
        fix_prompt = (
            "你是短视频带货文案事实审查员。下面这些口播句子里出现了违规断言: 要么是**在用户"
            "真实口播素材里找不到出处的数字断言**, 要么是**从另一条参考视频照搬来的事实/喝法/"
            "成分说法**(标记为 搬参考:xx)。两类都属于编造, 必须改写:\n"
            "- 删掉或替换这些违规断言, 只保留能在用户口播素材里找到出处的说法;\n"
            "- 没有可替换的真实卖点就改成画面/氛围描述, 不下任何可核查断言;\n"
            "- 每句 text 不得超过其目标字数上限; 口语自然, 不用括号引号省略号;\n"
            "- 每句更新 basis 说明出处, 纯衔接句写\"画面衔接, 无事实断言\"。\n\n"
            "用户真实口播素材(唯一事实来源):\n{speech}\n\n商品名: {product}\n\n待改写:\n{items}\n\n"
            "只输出 JSON: {{\"items\": [{{\"slot_id\": \"S01\", \"text\": \"...\", \"tone\": \"...\", \"basis\": \"...\"}}]}}"
        ).format(speech="\n".join("- " + r["text"] for r in speech_records[:6]),
                 product=product_name or "该商品",
                 items=json.dumps(payload, ensure_ascii=False, indent=2))
        try:
            text, _ = ask_qianfan([{"role": "user", "content": fix_prompt}], model=model, temperature=0.3)
            obj = loads_with_repair(text)
        except Exception as exc:
            print("[voiceover] 事实核验重写失败: {}".format(str(exc)[:160]), flush=True)
            break
        for it in (obj.get("items") or []):
            sid = _norm(it.get("slot_id"))
            txt = _norm(it.get("text"))
            if sid in bad_slots and txt:
                by_slot[sid]["text"] = txt
                by_slot[sid]["tone"] = (it.get("tone") or by_slot[sid].get("tone") or "自然安利").strip()
                by_slot[sid]["basis"] = str(it.get("basis") or "").strip() or by_slot[sid].get("basis", "")
    for sid, it in by_slot.items():
        viol = report.get(sid, {}).get("unverified_claims", [])
        it["fact_check"] = ("failed: " + ",".join(viol)) if viol else "passed"
        if viol:
            print("[voiceover] FACT_CHECK_FAILED {} 违规: {}".format(sid, viol), flush=True)
    return by_slot, report


def _fallback_script(segments):
    out = {}
    for s in segments:
        out[s["slot_id"]] = {"slot_id": s["slot_id"],
                             "text": _norm(s.get("beat_desc"))[:20] or "好物推荐",
                             "tone": "自然安利",
                             "basis": "降级兜底: 取参考节拍描述, 未经事实核验"}
    return out


def _original_script_item(sid, decision):
    """锁定原声段的脚本条目: text 即素材原话(字幕用), 无需核验(本就是用户说的)。"""
    return {"slot_id": sid,
            "text": decision.get("window_text", ""),
            "tone": "用户原声",
            "voice_source": "original",
            "audio_start": decision.get("audio_start"),
            "audio_take": decision.get("audio_take"),
            "basis": decision.get("decision_basis", ""),
            "fact_check": "passed(用户素材原声, 未改写)"}


def _parse_script_text_overrides():
    raw = os.getenv("WHQ_SCRIPT_TEXT_OVERRIDE", "").strip()
    if not raw:
        return {}
    out = {}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            return {_norm(k): str(v) for k, v in obj.items() if _norm(k) and str(v).strip()}
        except Exception as exc:
            print("[voiceover] WHQ_SCRIPT_TEXT_OVERRIDE JSON 解析失败: {}".format(str(exc)[:120]), flush=True)
            return {}
    for part in raw.split("||"):
        if "=" not in part:
            continue
        sid, text = part.split("=", 1)
        sid = _norm(sid)
        text = text.strip()
        if sid and text:
            out[sid] = text
    return out


def apply_script_text_overrides(by_slot):
    for sid, text in _parse_script_text_overrides().items():
        it = by_slot.get(sid) or {"slot_id": sid, "voice_source": "clone", "tone": "用户指定"}
        if it.get("voice_source") == "original":
            print("[voiceover] SCRIPT_OVERRIDE {} 是原声段, 不覆盖音频实录".format(sid), flush=True)
            continue
        old = it.get("text", "")
        it["text"] = text
        it["script_override"] = "user_env"
        it["basis"] = (it.get("basis", "") + "; 用户指定文案覆盖").strip("; ")
        by_slot[sid] = it
        print("[voiceover] SCRIPT_OVERRIDE {}: {} -> {}".format(sid, old, text), flush=True)
    return by_slot




_STRICT_SCRIPT_RULES = """- 每段 text **只讲该段【本段实际画面/素材内容】里真实有的东西**(锚定本段素材, 不要引入别的段的卖点)。
- **同一个卖点/说法(如某种喝法、某个成分)全片只能出现一次**, 不要在多段重复。
- **整条必须能顺读成一段话**：从第 2 段起，句首尽量用自然的衔接/过渡词承接上一段
  (如"而且""再加上""关键是""所以""最后""难怪"), 段与段之间有逻辑递进(痛点→卖点→演示→
  价格→催单), 不要是一堆互不相干的短句硬拼; 但每段仍紧扣本段画面、别硬凑。
- **催单/购买号召(赶紧、下单、冲、手慢无、链接、抢、囤)只能出现在最后一段, 且只出现一次**;
  中间任何一段都严禁出现催单/号召语气(价格段只客观讲价格/优惠, 不喊"赶紧冲")。
"""


def _strict_script():
    """是否启用「锚定本段素材/禁重复卖点/催单只在末段」这组文案硬约束。

    这组约束彼此拉扯(锚死本段素材 + 禁重复 + 又要顺读成一段话), 实测会让文案变干、
    连贯下降。WHQ_STRICT_SCRIPT=0 / WHQ_LEGACY=1 关闭, 回到迁移前的宽松版 prompt。
    """
    if legacy():
        return False
    return os.getenv("WHQ_STRICT_SCRIPT", "1") not in ("0", "false", "False")


def _target_chars(seg):
    """该段文案目标字数上限。

    迁移后放宽成 max(8, dur*cps + 4)，每段多 4 字会破坏「跟随参考语速」的初衷
    (ref_cps 算出来的就是参考对应段的字/秒)，legacy 下回到 max(6, dur*cps)。
    """
    raw = int(float(seg["target_duration"]) * _seg_cps(seg.get("ref_cps")))
    if legacy():
        return max(6, raw)
    slack = int(os.getenv("WHQ_TARGET_CHARS_SLACK", "4"))
    return max(8, raw + slack)


def generate_script(segments, dna, speech_records, product_name="", model=None, ref_text="",
                    decisions=None):
    decisions = decisions or {}
    locked = {s["slot_id"]: decisions[s["slot_id"]] for s in segments
              if (decisions.get(s["slot_id"]) or {}).get("voice_source") == "original"}
    optional = {s["slot_id"]: decisions[s["slot_id"]] for s in segments
                if (decisions.get(s["slot_id"]) or {}).get("voice_source") == "llm_decide"}
    gen_segments = [s for s in segments if s["slot_id"] not in locked]
    if not (_HAS_LLM and segments):
        by_slot = _fallback_script(gen_segments)
        for sid, d in locked.items():
            by_slot[sid] = _original_script_item(sid, d)
        return by_slot
    seg_lines = []
    for s in gen_segments:
        tc = _target_chars(s)
        line = "- {} ({:.1f}s, 目标字数≤{}): {}".format(
            s["slot_id"], s["target_duration"], tc, s.get("beat_desc") or "(画面)")
        mat = (s.get("best_candidate") or {}).get("text") or ""
        if mat and _strict_script():
            line += "\n  【本段实际画面/素材内容(文案只能据此写, 讲这段素材里真实有的东西, 不要照搬节拍里参考商品的说法, 也不要用别的段的卖点): {}】".format(mat[:140])
        opt = optional.get(s["slot_id"])
        if opt:
            line += "\n  【该段素材自带原声口播: 「{}」。若这段原话贴合整条叙事请返回 use_original=true; 与整体文案脱节才写克隆 text 替代】".format(
                opt.get("window_text", "")[:60])
        seg_lines.append(line)
    locked_lines = ["- {} (素材原话): {}".format(sid, d.get("window_text", "")[:80])
                    for sid, d in locked.items()] or ["(无)"]
    speech_lines = ["- {}".format(r["text"]) for r in speech_records[:6]] or ["(无真实口播, 请据画面与商品生成)"]
    topic = dna.get("topic_and_emotion") if isinstance(dna, dict) else ""
    if isinstance(topic, dict):
        topic = topic.get("core_topic") or json.dumps(topic, ensure_ascii=False)
    prompt = _SCRIPT_PROMPT.format(
        locked_block="\n".join(locked_lines),
        seg_block="\n".join(seg_lines), product=product_name or "该商品",
        speech_block="\n".join(speech_lines), topic=str(topic)[:120],
        strict_block=(_STRICT_SCRIPT_RULES if _strict_script() else ""))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=0.5)
        obj = loads_with_repair(text)
    except Exception as exc:
        print("[voiceover] LLM 生成脚本失败, 降级: {}".format(str(exc)[:160]), flush=True)
        by_slot = _fallback_script(gen_segments)
        for sid, d in locked.items():
            by_slot[sid] = _original_script_item(sid, d)
        return by_slot
    by_slot = {}
    for it in (obj.get("items") or []):
        sid = _norm(it.get("slot_id"))
        if sid in locked:
            continue
        # 无人脸但有原声的段: LLM 判定原话贴合整体文案 -> 也用原声
        if sid in optional and it.get("use_original") in (True, "true", "True", 1):
            d = optional[sid]
            d["voice_source"] = "original"
            d["decision_basis"] += "; LLM 判定该原话贴合整体叙事, 采用原声"
            by_slot[sid] = _original_script_item(sid, d)
            print("[voiceover] {} LLM 判定用原声".format(sid), flush=True)
            continue
        txt = _norm(it.get("text"))
        if sid and txt:
            by_slot[sid] = {"slot_id": sid, "text": txt,
                            "tone": (it.get("tone") or "自然安利").strip(),
                            "voice_source": "clone",
                            "basis": str(it.get("basis") or "").strip()}
    # 缺段兜底
    for s in gen_segments:
        by_slot.setdefault(s["slot_id"], _fallback_script([s])[s["slot_id"]])
    # 锁定原声段: LLM 成功路径同样要用 _original_script_item 加回(text=对窗 window_text),
    # 否则下游 build_tts_plan 会兜底用整段完整文本当字幕, 而音频只是对窗窗口 -> 字幕≠语音。
    for sid, d in locked.items():
        by_slot[sid] = _original_script_item(sid, d)
    # 事实核验只针对克隆段(原声段本就是用户说的话, 无需核验)
    clone_slots = {sid: it for sid, it in by_slot.items()
                   if it.get("voice_source") != "original"}
    clone_slots, _report = _fact_check_script(clone_slots, gen_segments, speech_records,
                                              product_name=product_name, model=model,
                                              ref_text=ref_text)
    by_slot.update(clone_slots)
    for sid, d in locked.items():
        by_slot[sid] = _original_script_item(sid, d)
    return by_slot


_REVIEW_PROMPT = """你是短视频带货成片的文案质检员。下面是一条成片的**最终逐段口播文案**
(按播放顺序; 有的段是用户素材原声实录, 有的段是克隆配音), 请整体审查是否合理:

商品名: {product}

逐段文案:
{script_block}

用户真实口播素材(事实来源, 供比对):
{speech_block}

审查维度(逐段给结论):
1. coherent: 该段与前后段是否连贯, 整条叙事(钩子→体验→价格/活动→成分→用法→催单)是否顺;
2. complete: 该句是否完整、不是半句话/悬空断句(如说到一半戛然而止);
3. sensible: 内容是否符合商品与生活常识, 有没有互相矛盾或莫名其妙的说法{strict_dims}

注意:
- 原声段的 text 是用户真实说话的实录, **不能改写**; 若它有问题只能指出(verdict=warn/fail)并说明。
- 原声段若实录文本里有明显的 ASR 同音错字(如「壁垒」实为「避雷」), 在 caption_text 给出
  **仅替换错字**的字幕更正版: 必须与原文等长、逐字同音/近音替换, 不得增删或改写句子; 否则留空。
- 克隆段若有问题, 给出改进后的 text(不超过原 text 字数+{slack},
  不得引入用户口播素材之外的新事实断言)。
- 都没问题就全部 verdict=pass。

只输出 JSON:
{{"overall": "整体结论一句话",
  "items": [{{"slot_id": "S01", "verdict": "pass|warn|fail", "issues": "问题说明, 无则空",
             "suggested_text": "仅克隆段且需要修改时给, 否则空",
             "caption_text": "仅原声段有同音错字时给等长更正版, 否则空"}}]}}"""


_STRICT_REVIEW_DIMS = """;
4. no_repeat: **同一个卖点/说法全片只能出现一次**(如"兑椰奶口感丝滑"不能在两段都讲);
   若某卖点重复出现, 只保留最贴合其段落功能的那一段, 其余重复段给 suggested_text
   改写成**该段自己应讲的内容**(按其在叙事里的位置: 配料段讲配料、用法段讲喝法等);
5. cta_ending: **催单/购买号召(赶紧/下单/冲/手慢无/链接/抢/囤)只能出现在最后一段, 且只出现一次**。
   - 若**最后一段不是**购买号召, 必须给 suggested_text 改写成一句购买号召;
   - 若**中间某段出现了**催单/号召, 必须给该段 suggested_text 改写成只讲本段内容(去掉催单语气)。"""


def review_script(by_slot, segments, speech_records, product_name="", model=None, ref_text=""):
    """成片文案整体 LLM 合理性审查(含原声段)。

    逐段过 fact_check 只保证「不编造」, 保证不了整条连贯/断句完整(实测 S02 原声停在
    「我比之前早起」半句上没人拦)。此审查把六段最终文案(原声+克隆)整体给 LLM 看:
    - 原声段是实录不可改写, 只标注 verdict/问题(供人排查, 窗口问题由 voice_policy 对窗修);
    - 克隆段可按建议改写(改写后再走一遍数字/搬参考核验, 不通过则不采纳)。
    结论写进每段 script_review 字段, 随 plan JSON 落盘。
    """
    if not (_HAS_LLM and by_slot):
        return by_slot
    order = [s["slot_id"] for s in segments if s["slot_id"] in by_slot]
    order += [sid for sid in by_slot if sid not in order]
    script_lines = []
    for sid in order:
        it = by_slot[sid]
        kind = "原声实录" if it.get("voice_source") == "original" else "克隆配音"
        script_lines.append("- {} [{}]: {}".format(sid, kind, it.get("text", "")))
    speech_lines = ["- " + r["text"] for r in speech_records[:6]] or ["(无)"]
    prompt = _REVIEW_PROMPT.format(product=product_name or "该商品",
                                   script_block="\n".join(script_lines),
                                   speech_block="\n".join(speech_lines),
                                   strict_dims=(_STRICT_REVIEW_DIMS if _strict_script() else "。"),
                                   slack=(4 if legacy() else 6))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=0.3)
        obj = loads_with_repair(text)
    except Exception as exc:
        print("[voiceover] 文案整体审查失败(跳过): {}".format(str(exc)[:160]), flush=True)
        for it in by_slot.values():
            it.setdefault("script_review", "review_unavailable")
        return by_slot
    overall = str(obj.get("overall") or "").strip()
    print("[voiceover] SCRIPT_REVIEW overall: {}".format(overall), flush=True)
    corpus = _norm("".join(r.get("text", "") for r in speech_records))
    user_clean = _clean("".join(r.get("text", "") for r in speech_records))
    ref_clean = _clean(ref_text)
    reviewed = set()
    for r in (obj.get("items") or []):
        sid = _norm(r.get("slot_id"))
        it = by_slot.get(sid)
        if not it:
            continue
        verdict = str(r.get("verdict") or "pass").strip().lower()
        issues = str(r.get("issues") or "").strip()
        it["script_review"] = verdict + ((": " + issues) if issues else "")
        reviewed.add(sid)
        # 原声段字幕错字更正: 音频是实录不动, 只更正烧录字幕用的展示文本。
        # 只接受"等长且改动字数很少"的更正(防 LLM 借机改写句子), 存 caption_text
        # 供 finisher 优先烧录; text 仍保留 ASR 实录原文(如实记录音频内容)。
        cap = _norm(r.get("caption_text"))
        cur = str(it.get("text") or "")
        if (cap and cap != cur and it.get("voice_source") == "original"
                and len(cap) == len(cur)
                and sum(a != b for a, b in zip(cap, cur)) <= max(2, len(cur) // 4)):
            it["caption_text"] = cap
            print("[voiceover] SCRIPT_REVIEW {} 字幕错字更正(音频不动): {} -> {}".format(
                sid, cur, cap), flush=True)
        sug = _norm(r.get("suggested_text"))
        # 只对克隆段采纳改写建议, 且改写后须重过事实核验(不引入新的编造)
        if (sug and sug != it.get("text") and verdict in ("warn", "fail")
                and it.get("voice_source") != "original"):
            if not SCRIPT_REVIEW_APPLY_REWRITE:
                print("[voiceover] SCRIPT_REVIEW {} 建议改写已关闭, 保留原文: {}".format(sid, it.get("text")), flush=True)
            elif not _fact_violations(sug, corpus) and not _ref_leak_violations(sug, user_clean, ref_clean):
                print("[voiceover] SCRIPT_REVIEW {} 改写: {} -> {}".format(sid, it.get("text"), sug), flush=True)
                it["text"] = sug
                it["basis"] = (it.get("basis", "") + "; 整体审查改写: " + issues).strip("; ")
            else:
                print("[voiceover] SCRIPT_REVIEW {} 建议改写含未核实断言, 不采纳: {}".format(sid, sug), flush=True)
        if verdict != "pass":
            print("[voiceover] SCRIPT_REVIEW {} {}: {}".format(sid, verdict, issues), flush=True)
    for sid, it in by_slot.items():
        if sid not in reviewed:
            it.setdefault("script_review", "pass")
    return by_slot


def _extract_original_audio(m, out_wav, audio_take=None):
    """从源素材抽该段实际使用窗口的原声, 变速对齐到 target_duration。

    clone_builder 对 avail<target 的段做 setpts 视频拉伸, 原声须同倍率 atempo 放慢
    (take/target < 1), 才与画面口型对齐; avail>=target 时 take=target, atempo=1。
    audio_take(voice_policy 按 ASR 字边界算出的干净截止点, <=take): 音频只截到最后
    一个完整字的结束, 不在半个字上硬切(段尾余下部分自然留静音)。
    """
    src = str(m.get("source_path") or "")
    target = float(m.get("target_duration") or 0.0)
    avail = float(m.get("source_avail") or 0.0)
    start = float(m.get("source_start") or 0.0)
    # voice_policy 句子级对窗后 manifest 带 source_take(源窗口长, 可长/短于 target);
    # 画面按 target/source_take 做了 setpts, 音频须同倍率 atempo(>1 加速 <1 放慢)。
    take_full = float(m.get("source_take") or 0.0)
    if take_full <= 0:
        take_full = min(avail, target) if avail > 0 else target
    take = take_full
    if not src or take <= 0:
        return None
    try:
        cut = float(audio_take or 0.0)
    except (TypeError, ValueError):
        cut = 0.0
    if 0 < cut < take:
        take = cut
    atempo = take_full / target if target > 0 else 1.0
    af = "atempo={:.5f},".format(atempo) if abs(atempo - 1.0) > 0.01 else ""
    af += "afade=t=in:d=0.02:curve=hsin,areverse,afade=t=in:d=0.02:curve=hsin,areverse"
    cmd = [FFMPEG, "-y", "-ss", "{:.3f}".format(start), "-t", "{:.3f}".format(take),
           "-i", src, "-vn", "-af", af, "-ar", "44100", "-ac", "2", str(out_wav)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return str(out_wav)


def build_tts_plan(manifest, script_by_slot, work_dir=None):
    """manifest 来自 clone_builder(含 target_time_range/累计); 组装 build_tts_overlay 直供 items。

    原声段(voice_source=original)在此处直接抽好源音频(voice_wav), build_tts_overlay
    见到 voice_wav 会跳过 CosyVoice 合成直接叠加。
    """
    items = []
    for m in manifest:
        sid = m.get("slot_id")
        rng = m.get("target_time_range", "")
        mt = re.match(r"\s*([0-9.]+)\s*-\s*([0-9.]+)", rng)
        if not mt:
            continue
        start, end = float(mt.group(1)), float(mt.group(2))
        sc = script_by_slot.get(sid, {})
        text = sc.get("text", "")
        if not text:
            continue
        dur = end - start
        seg_cps = _seg_cps(m.get("ref_cps"))
        items.append({
            "slot_id": sid,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration_seconds": round(dur, 2),
            "max_tts_chars": max(6, int(dur * seg_cps)),
            "ref_cps": m.get("ref_cps"),
            "seg_cps": round(seg_cps, 3),
            "text": text,
            "tone": sc.get("tone", "自然安利"),
            "voice_source": sc.get("voice_source", "clone"),
            "audio_take": sc.get("audio_take"),
            "basis": sc.get("basis", ""),
            "fact_check": sc.get("fact_check", ""),
            "script_review": sc.get("script_review", ""),
            "caption_text": sc.get("caption_text", ""),
            "script_override": sc.get("script_override", ""),
        })
        if sc.get("voice_source") == "original" and work_dir:
            try:
                wav = _extract_original_audio(m, os.path.join(work_dir, "orig_{}.wav".format(sid)),
                                              audio_take=sc.get("audio_take"))
                if wav:
                    items[-1]["voice_wav"] = wav
            except Exception as exc:
                print("[voiceover] {} 原声抽取失败, 回退克隆: {}".format(sid, str(exc)[:120]), flush=True)
                items[-1]["voice_source"] = "clone"
                items[-1]["basis"] += "; 原声抽取失败回退克隆"
    return {"items": items}


def add_voiceover(base_video, tts_plan, prompt_wav, prompt_asr, out_video,
                  work_dir, product_name=""):
    """调 generation/build_tts_overlay.py 叠加配音(TTS_PLAN_PATH 直供 items)。"""
    os.makedirs(work_dir, exist_ok=True)
    plan_path = os.path.join(work_dir, "tts_plan_input.json")
    json.dump(tts_plan, open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [COMMON, VENDOR] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update({
        "TTS_PLAN_PATH": plan_path,
        "TTS_OVERLAY_DIR": work_dir,
        "TTS_PROMPT_WAV": prompt_wav,
        "TTS_PROMPT_ASR": prompt_asr,
        "BASE_VIDEO": base_video,
        "FINAL_OUTPUT": out_video,
        "TARGET_PRODUCT_NAME": product_name or "该商品",
        "TTS_PYTHON": TTS_PYTHON,
        "FFMPEG": FFMPEG,
        # 默认 1.15 会把 CosyVoice 峰值(~0.89)推过满幅致削波失真(听感"不清晰"), 降回 1.0
        "TTS_OVERLAY_VOLUME": os.getenv("TTS_OVERLAY_VOLUME", "1.0"),
        # whq 段落首尾相接, 单段音频绝不能溢出到下一段(否则与字幕错位/叠音)。
        # build_tts_overlay 默认 TTS_MAX_ATEMPO=1.35, 对本模型偏慢的 TTS(实测~1.2字/秒)
        # 压不进槽位就放任溢出 -> 整轨相对字幕漂移。这里放宽上限, 让每段压进自己的
        # target_duration(源语速慢, 压 ~2.4x 后≈2.9字/秒, 既对齐又不失真); 并多给一轮
        # 文案压缩(先缩短再变速, 更自然)。可用同名 env 覆盖。
        "TTS_MAX_ATEMPO": os.getenv("TTS_MAX_ATEMPO", os.getenv("WHQ_TTS_MAX_ATEMPO", "2.6")),
        "TTS_RESYNTH_ROUNDS": os.getenv("TTS_RESYNTH_ROUNDS", "2"),
        # 把合成脚本指到薄封装: 真实合成后就地修掉每段起点硬起瞬态 + clarity EQ,
        # 让每段原始 tts_S0x.wav 本身就干净(不止最终成片干净)。
        # 真实合成脚本默认 CosyVoice3; 可用 WHQ_REAL_TTS_SCRIPT 换模型(如 VoxCPM2), 换模型时
        # 用 WHQ_REAL_TTS_PYTHON 指定其独立解释器(该模型所在 conda env)。
        "TTS_SCRIPT": os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_clean_wrap.py"),
        "WHQ_REAL_TTS_SCRIPT": os.getenv(
            "WHQ_REAL_TTS_SCRIPT", os.path.join(REPO, "generation", "run_cosyvoice3_zero_shot.py")),
    })
    if os.getenv("WHQ_REAL_TTS_PYTHON"):
        env["WHQ_REAL_TTS_PYTHON"] = os.getenv("WHQ_REAL_TTS_PYTHON")
    # build_tts_overlay 用 sys.executable 之外的 TTS_PYTHON 跑 cosyvoice; 本体用带
    # requests 的解释器即可。用主匹配 env(viral-split) 跑本体。
    py = sys.executable
    print("[voiceover] build_tts_overlay -> {}".format(out_video), flush=True)
    subprocess.run([py, BUILD_TTS_OVERLAY], cwd=REPO, env=env, check=True)
    return out_video


def _media_duration(path):
    try:
        out = subprocess.run(
            [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def declick_voiceover(video, item_starts, out_video, fade=0.020):
    """每句配音起止做短淡入淡出, 消除 CosyVoice 每句起点的宽带瞬态(切镜"提示音/杂音")。

    build_tts_overlay 用 adelay 把每句 TTS 硬贴到时间线, CosyVoice 合成每句开头 ~2~3ms 是
    高幅宽带瞬态(样本从 0 瞬跳 ±0.4 并高频振荡), 从静音突现在切镜点听感为"咔哒/提示音/杂音"。
    每句被约束在自己片段内(start=段起, end=段止), 故按句边界切片 + afade + 重接去瞬态。
    用 **hsin(升余弦)曲线**: 起点斜率为 0, 对最初 2~3ms 的爆发抑制远强于线性 tri,
    fade 取 20ms(足够盖住瞬态, 对语音起音自然无损感)。
    """
    dur = _media_duration(video)
    bounds = sorted(set([s for s in item_starts if s > 0.0] + [0.0]))
    if dur > 0:
        bounds.append(dur)
    bounds = sorted(set(bounds))
    if len(bounds) < 2:
        return video
    parts, labels = [], []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg_len = e - s
        if seg_len <= 2 * fade:
            continue
        lab = "a{}".format(i)
        labels.append("[{}]".format(lab))
        parts.append(
            "[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
            "afade=t=in:curve=hsin:st=0:d={d},"
            "afade=t=out:curve=hsin:st={fo:.3f}:d={d}[{lab}]".format(
                s=s, e=e, d=fade, fo=max(0.0, seg_len - fade), lab=lab))
    if not parts:
        return video
    fg = ";".join(parts) + ";" + "".join(labels) + \
        "concat=n={}:v=0:a=1[a]".format(len(labels))
    subprocess.run(
        [FFMPEG, "-y", "-i", video, "-filter_complex", fg,
         "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", out_video],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print("[voiceover] 去切镜爆音 -> {}".format(out_video), flush=True)
    return out_video


def run_voiceover(manifest, dna, asr_path, base_video, out_video, work_dir,
                  product_name="", model=None, ref_text="", decisions=None):
    speech = load_user_speech(asr_path)
    prompt_wav, prompt_asr = select_prompt(speech, os.path.join(work_dir, "prompt"))
    if not prompt_wav:
        print("[voiceover] 无可用声音克隆参考, 跳过配音(输出无声 base)", flush=True)
        return None, {"voiced": False, "reason": "no_prompt"}
    segments = [{"slot_id": m["slot_id"], "target_duration": m["target_duration"],
                 "beat_desc": m.get("beat_desc"), "ref_cps": m.get("ref_cps")} for m in manifest]
    # 逐段决策: 素材本就有口播+有人脸 -> 锁定原声; 有口播无人脸 -> LLM 判贴合度; 其余克隆。
    # run_clone 会在剪画面前先决策(原声段句子级对窗须在剪画面前平移窗口)并传入, 此处不重复跑。
    if decisions is None:
        decisions = voice_policy.decide(manifest, speech)
    script = generate_script(segments, dna, speech, product_name=product_name, model=model,
                             ref_text=ref_text, decisions=decisions)
    # 成片文案整体合理性审查(含原声段): 连贯性/断句完整性/叙事逻辑, 结论写入 plan
    script = review_script(script, segments, speech, product_name=product_name,
                           model=model, ref_text=ref_text)
    script = apply_script_text_overrides(script)
    plan = build_tts_plan(manifest, script, work_dir=work_dir)
    if not plan["items"]:
        return None, {"voiced": False, "reason": "empty_script"}
    add_voiceover(base_video, plan, prompt_wav, prompt_asr, out_video, work_dir, product_name)
    # 消除每句配音硬起造成的"切镜提示音/爆音"
    starts = [float(it.get("start") or 0.0) for it in plan["items"]]
    declick_tmp = os.path.splitext(out_video)[0] + "_declick.mp4"
    try:
        declick_voiceover(out_video, starts, declick_tmp)
        os.replace(declick_tmp, out_video)
    except Exception as exc:
        print("[voiceover] 去爆音失败, 保留原配音: {}".format(str(exc)[:160]), flush=True)
    return out_video, {"voiced": True, "n_items": len(plan["items"]),
                       "prompt_wav": prompt_wav, "script": script}


def main(argv=None):
    from reference_shots import load_dna
    ap = argparse.ArgumentParser(description="两者结合配音(用户ASR+LLM改写+声音克隆)")
    ap.add_argument("--manifest", required=True, help="clone_builder 输出 manifest json")
    ap.add_argument("--dna", required=True)
    ap.add_argument("--asr", required=True, help="all_source_asr.json")
    ap.add_argument("--base", required=True, help="无声 base mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--product-name", default="")
    ap.add_argument("--model")
    args = ap.parse_args(argv)
    man = json.load(open(args.manifest, encoding="utf-8"))
    manifest = man["segments"] if isinstance(man, dict) and "segments" in man else man
    dna = load_dna(args.dna)
    out, stat = run_voiceover(manifest, dna, args.asr, args.base, args.out,
                              args.work_dir, product_name=args.product_name, model=args.model)
    print("[voiceover]", stat)
    if out:
        print("voiced ->", out)


if __name__ == "__main__":
    main()
