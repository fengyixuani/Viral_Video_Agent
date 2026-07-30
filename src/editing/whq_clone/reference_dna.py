"""reference_dna — 从参考视频重建 DNA 的 key_beats（Agent 移植新增）。

背景：原 whq 的 key_beats 来自 Split understanding 阶段 VLM 撰写的 DNA md
(``content_structure.key_beats``)——每条是「画面语义 + 叙事功能」的语义节拍。Agent 的
``AnalysisResult.template`` 只有 role/duration，没有参考视频的时间轴叙事节拍。

按用户要求恢复到 Split 的语义质量，这里**首选用 VLM 观看参考视频**产出语义 key_beats
（``_vlm_dna``）；VLM 不可用/失败时**退化**为「场景检测 + 逐字 ASR」的弱重建
（``_scene_asr_dna``，只有口播文本、无画面语义）。

产物是一个 ``dna`` dict，形状与 ``reference_shots.parse_key_beats`` /
``edit_planner.build_segments_from_dna`` 期望的一致：
    {
      "duration_estimate": <秒>,
      "content_structure": {"narrative_structure": <可选>, "key_beats": ["M:SS-M:SS：<语义描述>", ...]},
      "topic_and_emotion": {"topic": <主题>},
    }

key_beats 的语义质量直接决定下游 edit_planner 给用户素材配镜 + voiceover 生成文案的效果，
所以务必是「这个节拍在讲什么 + 画面呈现什么」，而非参考视频的原始口播逐字。
"""
import asyncio
import os
import re

import _common  # noqa: F401  确保 src/shared 挂上 sys.path（供 import as_core）
import as_core
from reference_shots import detect_scene_cuts, _video_duration


# --------------------------------------------------------------------------- #
# 首选：VLM 观看参考视频 → 语义 key_beats（对齐 Split DNA 质量）
# --------------------------------------------------------------------------- #
_VLM_SYSTEM = (
    "你是爆款短视频的叙事结构分析专家。观看给定的参考视频，输出该视频的叙事 DNA，"
    "严格以 JSON 返回。"
)

_VLM_USER_TMPL = (
    "分析这条参考视频（总时长约 {duration:.1f} 秒），拆解出它的叙事结构 DNA。要求：\n"
    "1. 把视频切成 5~6 个**有语义的叙事节拍**(key_beats)，不是逐帧切点，而是"
    "「一个完整的叙事段落」。段落数量宁少勿多，保证每段承载一个完整的口播句/表达。\n"
    "2. 每个 key_beat 用一句话描述：**这个节拍在讲什么 + 画面呈现了什么 + 承担的叙事功能**"
    "（如开场钩子/痛点/卖点特写/使用演示/催单）。**不要只抄口播原话**，要概括画面与意图。\n"
    "3. **不要标注时间戳**，只写语义描述。例如："
    "`开场展示羽衣甘蓝粉兑东方树叶的“王炸”喝法，字幕打出搭配公式，抛出“以为是智商税”的反转钩子`。\n"
    "4. 另给 duration_estimate（秒）、topic（主题一句话）、narrative_structure（用 -> 连接的结构概述）。\n"
    "{asr_hint}"
    "只返回如下 JSON（不要额外文字）：\n"
    '{{"viral_dna_template":{{"duration_estimate":<秒>,'
    '"topic_and_emotion":{{"topic":"..."}},'
    '"content_structure":{{"narrative_structure":"...","key_beats":["语义描述1", "语义描述2"]}}}}}}'
)

# 从 VLM 描述里剥掉可能残留的时间戳前缀（如 "00:00-00:05：" / "0:02～0:05 "），
# 只保留语义描述——目的是让 build_segments_from_dna 走「均分时长」而非按精细时间戳
# 切成长短不一的碎段（后者会把用户原声句拦腰截断，且逻辑更跳，见与 Split 原版对比）。
_TS_PREFIX = re.compile(r"^\s*\d{1,2}:\d{2}(?:\.\d+)?\s*[-~到]\s*\d{1,2}:\d{2}(?:\.\d+)?\s*[：:]?\s*")


def _strip_ts(beat):
    return _TS_PREFIX.sub("", str(beat)).strip()


def _run_async(coro):
    """在无事件循环的线程/CLI 中同步跑协程（whq 全链路是同步、经 to_thread 调度）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        # 极少数场景：已在事件循环里被直接调用 → 用独立线程跑一个新循环。
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def _ref_asr_text(ref_asr_items, limit=600):
    txt = "".join(str(it.get("text") or "") for it in (ref_asr_items or [])).strip()
    return txt[:limit]


def _vlm_dna(ref_video, duration, ref_asr_items, topic):
    """让 VLM 观看参考视频，产出语义 key_beats 的 dna dict；失败/为空返回 None。"""
    asr_text = _ref_asr_text(ref_asr_items)
    asr_hint = (
        "参考视频的口播转写（仅辅助理解，可能有识别错字，不要照抄）：{}\n".format(asr_text)
        if asr_text else ""
    )
    user = _VLM_USER_TMPL.format(duration=duration or 0.0, asr_hint=asr_hint)
    obj = _run_async(as_core.complete_json(
        _VLM_SYSTEM, user, vision=True,
        media=[{"type": "video", "url": ref_video}]))
    tpl = obj.get("viral_dna_template", obj) if isinstance(obj, dict) else {}
    cs = (tpl.get("content_structure") or {}) if isinstance(tpl, dict) else {}
    key_beats = cs.get("key_beats") or []
    if not key_beats:
        return None
    # 剥掉可能残留的时间戳前缀 -> 无时间戳的纯语义节拍 -> 下游按均分时长切段
    # （与 Split 原版一致，避免精细时间戳切出过短段导致原声句被截、镜头跳切）。
    key_beats = [_strip_ts(b) for b in key_beats if _strip_ts(b)]
    if not key_beats:
        return None
    return {
        "duration_estimate": round(float(tpl.get("duration_estimate") or duration or 0.0), 3),
        "content_structure": {
            "narrative_structure": cs.get("narrative_structure", ""),
            "key_beats": key_beats,
        },
        "topic_and_emotion": {"topic": (tpl.get("topic_and_emotion") or {}).get("topic")
                              or topic or ""},
        "_reconstructed": True,
        "_vlm": True,
    }


# --------------------------------------------------------------------------- #
# 兜底：场景检测 + 逐字 ASR（无画面语义，仅口播文本；VLM 不可用时用）
# --------------------------------------------------------------------------- #
def _fmt_ts(t):
    """秒 -> 'M:SS.ss'，满足 parse_key_beats 的 (\\d{1,2}:\\d{2}(?:\\.\\d+)?) 正则。"""
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - 60 * m
    return "{:d}:{:05.2f}".format(m, s)


def _merge_bounds(cuts, duration, target_beats, min_shot):
    """场景切点 -> 段边界；把过短的段并进相邻段，直到段数 <= target_beats。"""
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    bounds = sorted(set(round(b, 3) for b in bounds))
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < min_shot and b != duration:
            continue
        merged.append(b)
    if merged[-1] != duration and duration:
        merged.append(duration)
    segs = [(merged[i], merged[i + 1]) for i in range(len(merged) - 1)]
    while len(segs) > target_beats:
        i = min(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if (segs[i - 1][1] - segs[i - 1][0]) <= (segs[i + 1][1] - segs[i + 1][0]) else i + 1
        lo = min(i, j)
        segs[lo] = (segs[min(i, j)][0], segs[max(i, j)][1])
        del segs[max(i, j)]
    return segs


def _speech_in_window(ref_asr_items, start, end):
    """窗口 [start,end] 内参考口播文本（token 中心落窗即计入）。"""
    if not ref_asr_items:
        return ""
    toks = []
    for it in ref_asr_items:
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        mid = (s + e) / 2.0
        if start - 0.01 <= mid < end + 0.01:
            toks.append(str(it.get("text") or ""))
    return "".join(toks).strip()


def _scene_asr_dna(ref_video, duration, ref_asr_items, target_beats, min_shot,
                   scene_threshold, topic):
    cuts = detect_scene_cuts(ref_video, scene_threshold)
    segs = _merge_bounds(cuts, duration, target_beats, min_shot) or [(0.0, duration)]
    key_beats = []
    for (s, e) in segs:
        desc = _speech_in_window(ref_asr_items, s, e)
        beat = "{}-{}".format(_fmt_ts(s), _fmt_ts(e))
        if desc:
            beat = "{}：{}".format(beat, desc)
        key_beats.append(beat)
    return {
        "duration_estimate": round(duration, 3),
        "content_structure": {"key_beats": key_beats},
        "topic_and_emotion": {"topic": topic or ""},
        "_reconstructed": True,
        "_vlm": False,
        "_n_scene_cuts": len(cuts),
    }


def build_dna_from_reference(ref_video, ref_asr_items=None, *, target_beats=6,
                             min_shot=0.8, scene_threshold=0.30, topic="", use_vlm=True):
    """重建参考视频的 DNA（首选 VLM 语义节拍，失败退化场景+ASR）。

    ref_asr_items: 参考视频逐字 ASR（asr_tokens.reference_asr_items 的产物），作为 VLM
    辅助上下文 + 兜底路径的描述来源。
    """
    if not ref_video or not os.path.exists(ref_video):
        raise ValueError("重建 key_beats 需要可读的参考视频: {}".format(ref_video))
    duration = _video_duration(ref_video)
    if not duration or duration <= 0:
        raise ValueError("无法探测参考视频时长: {}".format(ref_video))

    if use_vlm:
        try:
            dna = _vlm_dna(ref_video, duration, ref_asr_items, topic)
            if dna:
                print("[reference_dna] VLM 语义 key_beats: {} 段".format(
                    len(dna["content_structure"]["key_beats"])), flush=True)
                return dna
            print("[reference_dna] VLM 未产出 key_beats，退化为场景+ASR 重建", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("[reference_dna] VLM 重建失败({}), 退化为场景+ASR 重建".format(
                str(exc)[:160]), flush=True)

    return _scene_asr_dna(ref_video, duration, ref_asr_items, target_beats,
                          min_shot, scene_threshold, topic)
