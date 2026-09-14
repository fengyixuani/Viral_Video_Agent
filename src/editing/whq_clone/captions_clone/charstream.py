"""逐字时间流: 对【成片】跑一次词级 ASR, 给字幕块提供逐字揭示时间。

copy_zimu/v2 的逐字揭示依赖词级(逐字)时间戳。whq_clone 手上只有 tts_overlay_plan 的
**句级** start/end(段落窗口), 直接线性铺字会与真实语音漂移(尤其原声段与克隆段语速不同)。
所以这里对成片(已配音、未烧字幕的那一版)跑一次 Qwen3-ASR 词级强制对齐, 复用 whq_clone
已有的 asr_tokens.run_asr(同一套环境/模型/缓存, 不新增依赖)。

对外三个函数:
  - char_stream(video, cache_dir): 成片全篇逐字流 [{ch,start,end}](按时间排序)。
  - build_lines(tts_items, stream): 逐句念白 [{id,text,start,end,chars}]。
    文本以 tts plan 为准(它才是实际念出来的、且原声段有 caption_text 更正),
    时间优先用 ASR 逐字对齐; 在 ASR 流里定位不到的句子退化为窗口内线性铺字。
  - lines_from_stream(stream): 没有 tts plan 时(比如「字幕特效模仿」直接对任意成片跑),
    纯按停顿把 ASR 逐字流切成句子。

ASR 不可用(无环境/失败)时 char_stream 返回 [], build_lines 全部走线性铺字 —— 等价于旧行为,
不阻断出片; lines_from_stream 则返回 [](没有 ASR 就没有文本来源)。

每个字还带 ``brk``: 可以在这个字**之后**断开(短语/词边界)。来源两处 ——
  ① ASR 原文(asr_text, 带标点)里该字后面紧跟标点 -> 句读边界;
  ② 中文分词(jieba)的词尾 -> 词边界。
asr_items 是**纯单字无标点**的, 字间隔还量化到 0.08s(词内/词间都常是 0.000), 所以词边界只能这么
还原; 切句与 target_match 拆块都只在 brk 处下刀, 避免把「丝滑」切成「丝」「滑」。
jieba 缺失时退化为只有标点边界(仍好过按字数硬切)。
"""
import os
import re

import asr_tokens

# 句读标点: 出现在某字之后 -> 该字是短语尾, 允许断块
_PUNCT = set("，,。.！!？?；;：:、…~—－·　\"'“”‘’()（）【】[]{}《》<>")
_CUT = None            # 分词器句柄: None 未初始化 / False 不可用 / callable


def _expand(token_text, start, end):
    """多字 ASR token 拆成单字, 词内按字数线性插值。"""
    text = re.sub(r"\s+", "", str(token_text or ""))
    n = len(text)
    if n == 0:
        return []
    if n == 1:
        return [{"ch": text, "start": float(start), "end": float(end), "brk": False}]
    step = (float(end) - float(start)) / n
    return [{"ch": ch, "start": float(start) + i * step, "end": float(start) + (i + 1) * step,
             "brk": False}
            for i, ch in enumerate(text)]


def _cutter():
    """中文分词函数(jieba.cut); 不可用返回 None(只靠标点边界)。"""
    global _CUT
    if _CUT is None:
        try:
            import jieba
            _CUT = jieba.cut
        except Exception as exc:  # noqa: BLE001
            print("[captions_clone] jieba 不可用, 词边界退化为只用标点: {}".format(
                str(exc)[:120]), flush=True)
            _CUT = False
    return _CUT or None


def _mark_punct(chars, asr_text):
    """按 ASR 原文(带标点)标句读边界: 该字后面紧跟标点。

    asr_text 去掉标点/空白后应与 chars 逐字对齐; 对不上就整条放弃, 宁可不标也不要错位标。
    """
    text = str(asr_text or "")
    if not text:
        return False
    marks, i = [], 0
    for ch in text:
        if ch.isspace():
            continue
        if ch in _PUNCT:
            if i > 0:
                marks.append(i - 1)
            continue
        if i >= len(chars) or chars[i]["ch"] != ch:
            return False
        i += 1
    if i != len(chars):
        return False
    for idx in marks:
        chars[idx]["brk"] = True
        chars[idx]["punct"] = True         # 句读边界(比词边界更值得断句)
    return True


def _mark_words(chars, asr_text=None):
    """按分词标词边界: 每个词的最后一个字。

    分词跑在**带标点的 asr_text** 上(标点是分词的重要线索: 去掉标点后「丝滑。搅匀」会被切成
    「丝/滑/搅匀」), 再按字映射回 chars(跳过标点/空白)。asr_text 缺失/对不齐时退化用纯字流。
    """
    cut = _cutter()
    if not cut:
        return
    text = str(asr_text or "") or "".join(c["ch"] for c in chars)
    i = 0
    for word in cut(text):
        for ch in word:
            if ch.isspace() or ch in _PUNCT:
                continue
            if i >= len(chars) or chars[i]["ch"] != ch:
                return                      # 对不上, 停在这里(已标的仍有效)
            i += 1
        if 0 < i <= len(chars):
            chars[i - 1]["brk"] = True


def _mark_breaks(chars, asr_text):
    """给 chars 打可断点(brk): 标点句读 + 分词词尾, 末字天然是边界。返回可断点数。"""
    if not chars:
        return 0
    ok = _mark_punct(chars, asr_text)
    _mark_words(chars, asr_text if ok else None)
    chars[-1]["brk"] = True
    chars[-1]["punct"] = True
    return sum(1 for c in chars if c.get("brk"))


def char_stream(video, cache_dir):
    """对成片跑词级 ASR -> 全篇逐字流 [{ch,start,end}]。失败返回 []。"""
    out_json = asr_tokens.run_asr(video, cache_dir)
    if not out_json:
        return []
    import json
    try:
        with open(out_json, encoding="utf-8") as f:
            records = json.load(f) or []
    except (OSError, ValueError) as exc:
        print("[captions_clone] 成片 ASR 解析失败(降级线性铺字): {}".format(str(exc)[:160]),
              flush=True)
        return []
    stream, marked = [], 0
    for r in records:
        chars = []
        for it in r.get("asr_items") or []:
            chars.extend(_expand(it.get("text"), it.get("start") or 0.0, it.get("end") or 0.0))
        marked += _mark_breaks(chars, r.get("asr_text"))
        stream.extend(chars)
    stream.sort(key=lambda c: c["start"])
    print("[captions_clone] 成片逐字 ASR: {} 字, 可断点(标点+词边界) {} 处".format(
        len(stream), marked), flush=True)
    return stream


def _linear(text, start, end):
    n = len(text)
    if n == 0 or end <= start:
        return []
    step = (end - start) / n
    chars = [{"ch": ch, "start": start + i * step, "end": start + (i + 1) * step, "brk": False}
             for i, ch in enumerate(text)]
    _mark_words(chars)          # 线性铺字也要有词边界, 否则拆块又会切在词中间
    if chars:
        chars[-1]["brk"] = True
        chars[-1]["punct"] = True
    return chars


def plan_text(item):
    """配音 plan 一条的字幕文本: 去空白与标点(逐字流里没有标点, 要能对齐)。

    caption_text 优先(原声段的同音错字更正版), 否则 text。
    """
    raw = str(item.get("caption_text") or item.get("text") or "")
    return "".join(ch for ch in raw if not ch.isspace() and ch not in _PUNCT)


def lines_from_stream(stream, max_gap=None, max_chars=None, quiet=False):
    """没有 tts plan 时: 把 ASR 逐字流切成句子 [{id,text,start,end,chars}]。

    切句规则: 句读处(标点)只要够 min_chars 字就断句; 另外相邻字间隔 > max_gap 或攒到 max_chars
    字也要断, 但**只在可断点落刀** —— 优先句读, 其次词边界(攒够 max_chars 才允许), 都等不到就
    在 1.5*max_chars 处硬断兜底。否则 18 字硬切会把「丝滑」切成上一句的「丝」和下一句的「滑」。
    默认值可用 env 调。
    """
    if not stream:
        return []
    max_gap = float(max_gap if max_gap is not None else os.getenv("WHQ_CAPTION_SPLIT_GAP", "0.45"))
    max_chars = int(max_chars if max_chars is not None
                    else os.getenv("WHQ_CAPTION_SPLIT_CHARS", "18"))
    min_chars = int(os.getenv("WHQ_CAPTION_MIN_CHARS", "6"))
    hard_chars = max_chars + max_chars // 2
    has_brk = any(c.get("brk") for c in stream)
    groups, cur = [], []
    for c in stream:
        prev = cur[-1] if cur else None
        if prev is not None:
            n = len(cur)
            want = (float(c["start"]) - float(prev["end"]) > max_gap or n >= max_chars
                    or (prev.get("punct") and n >= min_chars))
            can = (not has_brk or prev.get("punct")
                   or (prev.get("brk") and n >= max_chars) or n >= hard_chars)
            if want and can:
                groups.append(cur)
                cur = []
        cur.append(dict(c))
    if cur:
        groups.append(cur)
    lines = []
    for g in groups:
        text = "".join(x["ch"] for x in g)
        if not text:
            continue
        lines.append({"id": len(lines) + 1, "text": text,
                      "start": float(g[0]["start"]), "end": float(g[-1]["end"]), "chars": g})
    if not quiet:
        print("[captions_clone] 按停顿切句: {} 句(间隔>{}s 或 >{}字断句)".format(
            len(lines), max_gap, max_chars), flush=True)
    return lines


def _uncovered_runs(stream, used, busy, min_chars, min_dur):
    """逐字流里既没被 plan 行吃掉、也不落在已有字幕时间窗内的连续段。"""
    runs, cur = [], []
    for i, c in enumerate(stream):
        mid = (float(c["start"]) + float(c["end"])) / 2.0
        if not used[i] and not any(a <= mid <= b for a, b in busy):
            cur.append(c)
            continue
        if cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs
            if len(r) >= min_chars and float(r[-1]["end"]) - float(r[0]["start"]) >= min_dur]


def build_lines(tts_items, stream):
    """tts plan items + 成片逐字流 -> 逐句念白(带逐字时间)。

    **文本以 plan 为准**(caption_text 优先, 否则 text): 那是实际念出来的文案, 没有 ASR 的同音
    错字（实测成片 ASR 把「侧颜」听成「侧眼」）。时间优先用 ASR 逐字对齐, 在流里定位不到的句子
    (文案与识别结果有差异时)退化为窗口内线性铺字。

    对齐完还要**补漏**: 原声保留段的 plan 文本只是匹配到的那一句(``loop._caption_items`` 取
    ``spoken_text``), 而 editor 保留的音频窗口往往更长——实测 S01 窗口 0-5.72s 里还念了
    「用的粉底越贵卡的纹路越清晰」(2.88-5.52s), plan 里没这句, 那 2.6s 就有声音没字幕。这里把
    没被任何 plan 行覆盖的逐字段按停顿切句补上, 文本随后走同音纠错(``WHQ_CAPTION_GAPFILL=0``
    可关)。
    """
    stream_text = "".join(c["ch"] for c in stream)
    used = [False] * len(stream)
    lines, cursor, hit = [], 0, 0
    for it in tts_items or []:
        text = plan_text(it)
        start = float(it.get("start") or 0.0)
        end = float(it.get("end") or 0.0)
        if not text or end <= start:
            continue
        chars = []
        if stream_text:
            idx = stream_text.find(text, cursor)
            if idx < 0:
                idx = stream_text.find(text)
            if idx >= 0:
                seg = stream[idx: idx + len(text)]
                if seg:
                    chars = [dict(c) for c in seg]
                    cursor = idx + len(text)
                    hit += 1
                    for i in range(idx, min(idx + len(text), len(used))):
                        used[i] = True
        if not chars:
            chars = _linear(text, start, end)
        lines.append({"id": len(lines) + 1, "text": text,
                      "start": chars[0]["start"], "end": chars[-1]["end"], "chars": chars})
    if not lines:
        return lines
    print("[captions_clone] 逐句对齐: {}/{} 句命中 ASR 逐字".format(hit, len(lines)), flush=True)
    if stream and os.getenv("WHQ_CAPTION_GAPFILL", "1").strip().lower() not in (
            "0", "off", "false", "no"):
        busy = [(float(l["start"]), float(l["end"])) for l in lines]
        runs = _uncovered_runs(stream, used, busy,
                               int(os.getenv("WHQ_CAPTION_GAPFILL_CHARS", "4")),
                               float(os.getenv("WHQ_CAPTION_GAPFILL_DUR", "0.6")))
        extra = []
        for run in runs:
            extra.extend(lines_from_stream(run, quiet=True))
        if extra:
            print("[captions_clone] 补漏字幕 {} 句(plan 未覆盖的成片念白): {}".format(
                len(extra), " / ".join("{:.1f}-{:.1f}s {}".format(
                    l["start"], l["end"], l["text"]) for l in extra)), flush=True)
            lines.extend(extra)
    lines.sort(key=lambda l: float(l["start"]))
    for i, ln in enumerate(lines, 1):
        ln["id"] = i
    return lines

