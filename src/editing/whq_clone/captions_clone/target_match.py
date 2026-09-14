"""目标编排: 成片念白 + 参考字幕风格 -> 单条字幕流(带角色/位置/颜色/字号/特效)。

移植 copy_zimu/v2/target_match.py, 两处改动:
  1. 参考风格不再从 samples/<ref>_style_profile.json 读, 直接吃 profile_cache 给的 dict。
  2. LLM 走 Agent 的 wenchain 网关(pipeline_utils.ask_qianfan), 不再走 Split 的 keyline_selector。
  3. 产物直接是内存 seq 交给 ass_burn(不再经《清单.md》round-trip); md 只作留档。

编排要点(与 Split 版一致):
  * 位置: 一律底部居中(WHQ_CAPTION_POS=mixed 可回到"彩色大字上顶部"的旧分布)。
  * 字号: **全片统一**取参考的口播档(base_style) —— 不再让 emphasis 整块跳大字。
  * 配色: 正文取参考口播色; 参考的彩色只用在**块内关键词**上(accent_color + keyword)。
  * 特效: LLM 只能在该角色的【参考特效集】里挑; 没挑则整套套用该角色参考特效。
  * 关键词(成分/数据/功效/品牌)留在所在块里, 由 LLM 用 keyword 标出, 块内变大换色。
  * 同音错字: correct_typos 在拆块前做一遍等长纠错(原声段字幕只能来自 ASR)。
"""
import json
import os
import re

from pipeline_utils import ask_qianfan, loads_with_repair

# 念白超过该字数的句子必须拆多段(对齐参考的短字幕节奏)。
SPLIT_THRESHOLD = 14

_POS_CN = {"top": "顶部", "middle": "中部", "bottom": "底部"}
_ROLE_CN = {"narration": "口播", "emphasis": "强调大字", "highlight": "句内高亮",
            "hook": "标题钩子", "label": "标签/角标"}
# profile 的 effects 数组里可能混入角色名(分析噪声), 复刻特效时剔除。
_EFFECT_NOISE = {"narration", "emphasis", "highlight", "label", "hook"}
# 放射线需要集中线 overlay 素材, 烧录端不支持, 直接剔除。
_EFFECT_DROP = {"放射线"}
_ROLE_COLOR_FALLBACK = {"narration": "#FFFFFF", "emphasis": "#EA3717",
                        "highlight": "#FFF9C4", "label": "#FFFFE0", "hook": "#FFFFFF"}
_ROLE_DEFAULT_POS = {"narration": "底部居中", "emphasis": "顶部居中",
                     "highlight": "顶部居中", "label": "顶部居中", "hook": "顶部居中"}
# 短彩色块(≤该字数)可在顶部偏左/偏右交替增加活泼度, 更长则顶部居中(避免出画面)。
_TOP_OFFSET_MAX = 4
_SIZE_PX = {"big": 120, "normal": 66, "small": 40}
_ROLE_AN = {"narration": 2, "emphasis": 8, "highlight": 8, "label": 8, "hook": 8}


def hex_to_ass(hexstr):
    """#RRGGBB -> ASS &H00BBGGRR。"""
    h = (hexstr or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        return "&H00FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return ("&H00" + b + g + r).upper()


def ref_palette(profile):
    """参考 style_profile -> (styles_for_llm, by_role)。

    颜色/字号/an/特效全部取参考分析的校准值(不写死); profile 缺字段才用兜底常量。
    """
    for_llm, by_role = [], {}
    for name, st in (profile.get("styles") or {}).items():
        role = st.get("role", name)
        eff = [e for e in (st.get("effects") or [])
               if e not in _EFFECT_NOISE and e not in _EFFECT_DROP]
        # 参考检测到倾斜但没显式给"斜排"特效, 视为斜排(忠实复刻参考的倾斜)。
        if int(st.get("slant") or 0) != 0 and "斜排" not in eff:
            eff = eff + ["斜排"]
        color = (st.get("fill_hex") or st.get("fill_hex_hint")
                 or _ROLE_COLOR_FALLBACK.get(role, "#FFFFFF"))
        outline = st.get("outline_hex") or st.get("outline_hex_hint") or "#101010"
        size_kw = st.get("size_kw") or "normal"
        size_px = int(st.get("size") or _SIZE_PX.get(size_kw, 66))
        an = int(st.get("an") or _ROLE_AN.get(role, 2))
        band = _POS_CN.get((st.get("position") or "").strip(),
                           _ROLE_DEFAULT_POS.get(role, "底部居中")[:2])
        by_role[role] = {"size_kw": size_kw, "size_px": size_px, "effects": eff,
                         "annotations": list(st.get("annotations") or []),
                         "color": color, "outline": outline, "an": an,
                         "band": band, "slant": int(st.get("slant") or 0)}
        for_llm.append({"角色": _ROLE_CN.get(role, role), "role": role,
                        "参考色": color, "字号档": size_kw, "特效": eff})
    return for_llm, by_role


def allowed_cuts(chars):
    """可以在其后断块的字下标集合(来自 charstream 的 brk: ASR 原文里该字后面有标点)。

    末字那一刀不算内部断点。空集 = 这句没有标点信息(比如线性铺字的 tts 段), 此时不做约束。
    """
    n = len(chars or [])
    return {i for i, c in enumerate(chars) if c.get("brk") and i < n - 1}


def marked_text(line):
    """句子文本, 在可断块处插入 `/` 给 LLM 看; 没有标记就是原文。"""
    chars = line.get("chars") or []
    if not chars:
        return line.get("text") or ""
    cuts = allowed_cuts(chars)
    out = []
    for i, c in enumerate(chars):
        out.append(c["ch"])
        if i in cuts:
            out.append("/")
    return "".join(out)


TYPO_SYSTEM = (
    "下面是带货视频口播的**语音识别**文案, 里面常有同音/近音错字。请逐句改成正确的字。\n"
    "【硬约束】每句改完后的**字数必须和原句完全一致**: 只允许把某个字换成同音/近音的正确字, "
    "**不许增字、不许删字、不许改标点、不许调整语序、不许润色改写**。字幕要和语音逐字对齐, "
    "多一个字少一个字都会让整句时间轴错位。\n"
    "【怎么判断该改成什么】\n"
    "  1) 先看**整段上下文**语义: 正确的写法必须让这句在这条视频里读得通。\n"
    "  2) 优先考虑**带货口播的高频词和网络用语**, 它们最常被识别错: 避雷(不是「壁垒」)、"
    "踩雷、闭眼冲、无脑入、回购、性价比、直接冲、别犹豫、划算、绝了、上脸、显白、氛围感、"
    "配料表、0 蔗糖、致敏、锁水、平替。\n"
    "  3) **绝对不要凭猜测编出品牌名、人名、地名或型号**(把「壁垒」改成某个品牌名这种是严重错误)。"
    "只有上下文明确出现该品牌时才可以。\n"
    "  4) **没把握就原样返回**: 留着一个错字, 比改成一个猜出来的词好得多。\n"
    "没有错字的句子原样返回。只输出 JSON: "
    "{\"lines\":[{\"id\":1,\"text\":\"改好的整句\",\"why\":\"改了什么/为什么, 没改就填空\"}]}"
)


def correct_typos(lines, product_name="", model=None):
    """同音错字纠正(就地改 lines 的 text 与 chars[i]['ch'])。返回改动的句数。

    为什么需要: 原声保留段没有配音文案, 字幕文本只能来自成片 ASR, Qwen3-ASR-0.6B 又没有热词/
    领域纠错能力。实测出过「壁垒这个黑巧」(应为品牌名)、「侧眼」(应为「侧颜」)。

    **只接受等长改写**: 逐字时间来自 chars, 改了字数就与 els/时间轴错位。长度不符的整句丢弃
    (宁可留着错字, 也不要字幕跟语音对不上)。LLM 失败/异常一律跳过, 不阻断烧录。
    """
    items = [{"id": l["id"], "text": l.get("text") or ""} for l in lines if l.get("text")]
    if not items:
        return 0
    user = ("商品: {}\n\n识别文案(逐句):\n{}".format(
        product_name or "（未指定）",
        json.dumps(items, ensure_ascii=False, indent=None)))
    try:
        content, _raw = ask_qianfan(
            [{"role": "system", "content": TYPO_SYSTEM}, {"role": "user", "content": user}],
            model=model, max_tokens=4096, temperature=0.1)
        data = loads_with_repair(content)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 同音纠错 LLM 失败(保留 ASR 原文): {}".format(str(exc)[:200]),
              flush=True)
        return 0
    fixed = {}
    for it in data.get("lines") or []:
        try:
            fixed[int(it.get("id"))] = (re.sub(r"\s+", "", str(it.get("text") or "")),
                                        str(it.get("why") or ""))
        except (TypeError, ValueError):
            continue
    changed, skipped = 0, 0
    for ln in lines:
        new, why = fixed.get(ln["id"], ("", ""))
        old = ln.get("text") or ""
        if not new or new == old:
            continue
        if len(new) != len(old):
            skipped += 1
            continue
        ln["text"] = new
        for c, ch in zip(ln.get("chars") or [], new):
            c["ch"] = ch
        changed += 1
        print("[captions_clone] 纠错 [{}] {} -> {}{}".format(
            ln["id"], old, new, "（{}）".format(why[:60]) if why else ""), flush=True)
    if skipped:
        print("[captions_clone] 同音纠错丢弃 {} 句(改后字数不一致, 会让时间轴错位)".format(skipped),
              flush=True)
    return changed




SYSTEM = (
    "你是抖音爆款带货视频的字幕特效设计师。要把【参考视频字幕风格】迁移到【用户视频念白】上, "
    "编排成【单条字幕流】(所有块按语音时间首尾相接, 同一时刻只出现一条)。\n"
    "【角色】只决定**特效**, 不决定字号和颜色(字号全片统一, 颜色只给关键词):\n"
    "  - narration 口播: 主体念白, 大多数块都是这个。\n"
    "  - emphasis 强调: 最戳人的短句(只影响特效, 不会整块变色变大)。\n"
    "  - label 标签/角标: 小标签角标。\n"
    "【拆块规则】(最重要):\n"
    "  1) 把每句念白按语义拆成若干【短块】, 每块是该句里【逐字连续出现的子串】(改一个字都对不上音)。\n"
    "  2) 念白里用 `/` 标出了**可以断块的位置**(标点/词边界)。**只能在 `/` 处断开**, "
    "**绝不能在词内部下刀**(如「丝滑」不能拆成「丝」+「滑」, 「配料表」不能拆成「配」+「料表」, "
    "「赶紧」不能拆成「赶」+「紧」)。`/` 是**允许**断的位置、不是必须断: 每块**尽量 4~12 字**, "
    "不要每个 `/` 都断成一两个字的碎块。phrase 里**不要带 `/`**。\n"
    "  3) 关键词(成分/数据/功效/品牌)**不要单独切成一块**, 让它留在所在的口播块里, 用 keyword "
    "字段把它标出来即可 —— 系统会在块内把这几个字变大换色, 整块其余字保持普通白字。\n"
    "  4) 念白【超过约14字】的句子尽量拆成【两段以上】连续短块; 若句内没有 `/` 可断, 就整句一块, "
    "**不要为了凑长度硬切**。\n"
    "  5) 一句话所有短块【按原文顺序、拼起来就是原句】(不改字、不漏字、不重叠) —— **每个字都必须"
    "被某个块覆盖**, 只挑关键词、丢掉其余部分是错的(那几秒会有声音没字幕)。\n"
    "【关键词 keyword】每块**最多一个**, 必须是该块 phrase 里【逐字连续出现的子串】, 建议 2~6 字, "
    "挑成分/数据/功效/品牌/卖点词(如「配料表」「0 蔗糖」「不挤脚」)。整块都是关键词是错的 —— "
    "**keyword 必须比 phrase 短**; 块里没有值得突出的词就填空字符串。\n"
    "【逐块特效】(复刻参考, 按语义挑, 只能从该 role 参考特效里选, 不合适就不套):\n"
    "  弹入=开场/重磅登场; 对比标签=对比/并列词; 斜排=强调短句的冲击感(口播不斜排); "
    "描边发光=普通口播基础; 放大镜=带关键词的块。\n"
    "对每个短块输出:\n"
    "  - src_id: 原句编号(整数)\n"
    "  - order: 块在原句内顺序(从1)\n"
    "  - phrase: 块文本(原样逐字, 该句连续子串)\n"
    "  - role: narration | emphasis | label\n"
    "  - keyword: 块内要变大换色的关键词(phrase 的连续子串, 比 phrase 短; 没有就填 \"\")\n"
    "  - effects: 数组, 本块特效(从该 role 参考特效里按语义挑, 可空)\n"
    "  - reason: 一句话说明(角色/关键词/特效为何这样)\n"
    "只返回 JSON: {\"blocks\":[{\"src_id\":1,\"order\":1,\"phrase\":\"..\",\"role\":\"..\","
    "\"keyword\":\"..\",\"effects\":[],\"reason\":\"..\"}]}"
)


def match(lines, styles_for_llm, model=None):
    """LLM: 把每句念白拆成短块并套参考风格。返回 {src_id: [block,...]}; 失败返回 {}。"""
    numbered = "\n".join("[{}] {:.2f}-{:.2f}s ({}字) {}".format(
        l["id"], l["start"], l["end"], len(l["text"]), marked_text(l)) for l in lines)
    user = ("参考风格调色板(角色/迁移色/字号/特效, JSON):\n"
            + json.dumps(styles_for_llm, ensure_ascii=False)
            + "\n\n用户视频全部念白(按时间, 编号 src_id, 已标字数; `/` = 允许断块的位置):\n" + numbered
            + "\n\n请按参考风格把每句念白拆成单条流短块(**只能在 `/` 处断开**, 关键词留在块内用 "
              "keyword 标出、不要单独成块), 给每块配 role/keyword/effects。")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        content, _raw = ask_qianfan(messages, model=model, max_tokens=8192, temperature=0.3)
        data = loads_with_repair(content)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 拆块 LLM 失败(全走兜底拆块): {}".format(str(exc)[:200]), flush=True)
        return {}
    by_id = {}
    for b in data.get("blocks", []):
        sid = b.get("src_id")
        if sid is None:
            continue
        by_id.setdefault(int(sid), []).append(b)
    for sid in by_id:
        by_id[sid].sort(key=lambda b: b.get("order", 0))
    return by_id


def _split_fallback(text, cuts=None):
    """LLM 未给某句分块时的兜底: 过长就在**离中点最近的合法断点**拆开, 否则整句一块。

    cuts = 允许断块的字下标(其后可断)。没有合法断点就整句一块 —— 宁可一块长字幕,
    也不要按字数硬切出「丝」+「滑」这种。
    """
    text = re.sub(r"\s+", "", text)
    if len(text) <= SPLIT_THRESHOLD:
        return [text]
    mid = (len(text) + 1) // 2
    inner = sorted(i for i in (cuts or ()) if 0 <= i < len(text) - 1)
    if not inner:
        return [text]
    i = min(inner, key=lambda x: abs(x + 1 - mid))
    return [text[:i + 1], text[i + 1:]]


def _fill_gaps(chars, parts):
    """把没被任何块覆盖的字补成口播块 —— 否则那几秒**有声音没字幕**。

    LLM 经常只挑走一句里的关键词(实测一条 124 字的成片, 它对某句只返回了「智商水」一个
    highlight 块, 剩下 26 字直接消失, 成片里近 5 秒有口播没字幕)。原来只有"整句没块"才兜底,
    这里补上"句内漏字"。
    """
    cuts = allowed_cuts(chars)
    out, cursor, filled = [], 0, 0

    def gap(a, b):
        """[a, b) 没被覆盖 -> 拆成 1~2 个口播块补上。"""
        text = "".join(c["ch"] for c in chars[a:b])
        pieces, at = [], a
        for piece in _split_fallback(text, {i - a for i in cuts if a <= i < b}):
            pieces.append([at, at + len(piece),
                           {"role": "narration", "effects": [], "order": 0,
                            "reason": "补齐 LLM 漏掉的念白"}])
            at += len(piece)
        return pieces

    for st, en, b in parts:
        if st > cursor:
            out.extend(gap(cursor, st))
            filled += st - cursor
        out.append([st, en, b])
        cursor = en
    if cursor < len(chars):
        out.extend(gap(cursor, len(chars)))
        filled += len(chars) - cursor
    if filled:
        print("[captions_clone] 补齐 LLM 漏掉的念白 {} 字(否则这段有声音没字幕)".format(filled),
              flush=True)
    return out


def _snap_blocks(chars, parts):
    """把 LLM 的块边界对齐到标点短语边界; 对不齐就把两块并回一块。

    parts: [[start, end, block], ...](按序、首尾相接)。避免「丝滑」被切成「丝」「滑」。
    """
    cuts = allowed_cuts(chars)
    if not cuts or len(parts) < 2:
        return parts
    out = [parts[0]]
    for st, en, b in parts[1:]:
        p_st, p_en, p_b = out[-1]
        if p_en != st or (p_en - 1) in cuts:
            out.append([st, en, b])            # 边界本就落在短语尾(或两块不相邻)
            continue
        near = [i for i in cuts if p_st <= i < en - 1]
        if near:                               # 挪到最近的合法断点, 两块都保住
            i = min(near, key=lambda x: abs(x - (p_en - 1)))
            out[-1][1] = i + 1
            out.append([i + 1, en, b])
        else:                                  # 无处可断 -> 合并, 角色取较长那半
            out[-1] = [p_st, en, b if (en - st) > (p_en - p_st) else p_b]
    return out


def position_mode():
    """字幕位置策略: "bottom"(全部底部居中, 默认) / "mixed"(口播底部 + 强调/高亮顶部)。

    用户拍板: 位置统一固定在画面最下方, 参考视频那边只模仿颜色/字号/特效。
    """
    v = (os.getenv("WHQ_CAPTION_POS") or "bottom").strip().lower()
    return "mixed" if v in ("mixed", "top", "auto") else "bottom"


def _coerce_position(role, phrase, top_alt):
    """位置: 默认全部底部居中; mixed 模式下彩色大字上顶部(短块偏左右交替), 都不用中部(挡脸)。"""
    if role == "narration" or position_mode() == "bottom":
        return "底部居中"
    if len(phrase) <= _TOP_OFFSET_MAX:
        return "顶部偏左" if top_alt % 2 == 0 else "顶部偏右"
    return "顶部居中"


def base_style(by_role):
    """全片统一的正文样式(字号/填充/描边): 取参考的**口播**档。

    用户拍板: 字号全片一致, 不再让 emphasis 整块跳到 120px「突然变大」; 彩色也不整块套,
    只给块内关键词(见 ``_mark_keyword``)。参考没给口播档才用兜底常量。
    """
    rp = by_role.get("narration") or {}
    return {"size": int(rp.get("size_px") or _SIZE_PX["normal"]),
            "color": rp.get("color") or _ROLE_COLOR_FALLBACK["narration"],
            "outline": rp.get("outline") or "#101010"}


def accent_color(by_role):
    """关键词强调色: 参考里 emphasis/highlight 的校准色(排除白色), 都没有就用兜底红。"""
    for role in ("emphasis", "highlight", "hook", "label"):
        c = ((by_role.get(role) or {}).get("color") or "").strip()
        if c and c.upper() not in ("#FFFFFF", "#FFF", "#FEFEFE"):
            return c
    return _ROLE_COLOR_FALLBACK["emphasis"]


def _mark_keyword(els, phrase, keyword):
    """把 keyword 命中的字标 ``hot=True``(烧录时块内变大换色)。返回命中字数。

    keyword 覆盖整块时不标 —— 那又变回「整句都是特殊颜色」, 用户明确不要。
    """
    kw = re.sub(r"[\s/]+", "", str(keyword or ""))
    if not kw or len(kw) >= len(phrase):
        return 0
    i = phrase.find(kw)
    if i < 0:
        return 0
    for e in els[i:i + len(kw)]:
        e["hot"] = True
    return len(kw)


def build_sequence(lines, by_id, by_role):
    """LLM 分块按原句顺序光标推进定位回逐字时间 -> 单条流(每块带 els 逐字时间)。

    定位后再按标点短语边界校正一遍(``_snap_blocks``): LLM 常按字数对半切, 会把「配料表」
    切成「配」+「料表」, 靠 prompt 说服不彻底, 这里兜死。

    字号/填充色全块统一走 ``base_style``; 只有 LLM 标出的 keyword 在块内变大换色。
    """
    seq = []
    top_alt = 0
    base = base_style(by_role)
    accent = accent_color(by_role)
    for ln in lines:
        chars = ln.get("chars") or []
        text = "".join(c["ch"] for c in chars)
        blocks = by_id.get(ln["id"])
        if not blocks:
            blocks = [{"phrase": p, "role": "narration", "effects": [], "position": "底部居中",
                       "reason": "兜底拆块(LLM 未覆盖)", "order": i + 1}
                      for i, p in enumerate(_split_fallback(text or ln["text"],
                                                            allowed_cuts(chars)))]
        parts, cursor = [], 0
        for b in blocks:
            phrase = re.sub(r"[\s/]+", "", b.get("phrase") or "")
            if not phrase:
                continue
            idx = text.find(phrase, cursor) if text else -1
            if idx < 0:
                # 定位不到(越权/重叠/错位) -> 丢弃, 保证单条流不重叠、能拼回原句
                continue
            parts.append([idx, idx + len(phrase), b])
            cursor = idx + len(phrase)
        for st, en, b in _snap_blocks(chars, _fill_gaps(chars, parts)):
            els = [dict(c) for c in chars[st:en]]
            if not els:
                continue
            phrase = "".join(e["ch"] for e in els)
            role = b.get("role") or "narration"
            if role not in by_role and role not in _ROLE_COLOR_FALLBACK:
                role = "narration"
            pos = _coerce_position(role, phrase, top_alt)
            if pos.startswith("顶部偏"):
                top_alt += 1
            rp = by_role.get(role, {})
            allowed = rp.get("effects", [])
            eff = [e for e in (b.get("effects") or []) if e in allowed]
            if role == "narration":
                eff = [e for e in eff if e != "斜排"]  # 口播一行正字, 不斜排
            # LLM 没给特效时套该角色的参考特效集, 保证"有特效的角色不会变成纯文字"
            if not eff:
                eff = [e for e in allowed if not (role == "narration" and e == "斜排")]
            hot = _mark_keyword(els, phrase, b.get("keyword"))
            seq.append({
                "src_id": ln["id"], "order": b.get("order", 0), "phrase": phrase,
                "role": role, "position": pos,
                "color": base["color"], "outline": base["outline"], "size": base["size"],
                "accent": accent,
                "keyword": "".join(e["ch"] for e in els if e.get("hot")) if hot else "",
                "effects": eff, "start": float(els[0]["start"]), "end": float(els[-1]["end"]),
                "els": els, "reason": b.get("reason") or "-",
            })
    seq.sort(key=lambda s: (s["start"], s["src_id"], s["order"]))
    return seq


def plan(lines, profile, model=None):
    """参考 profile + 逐句念白 -> (seq, by_role)。对外主入口。"""
    styles_for_llm, by_role = ref_palette(profile)
    seq = build_sequence(lines, match(lines, styles_for_llm, model=model), by_role)
    kw = sum(1 for s in seq if s.get("keyword"))
    print("[captions_clone] 字幕块: {} (带关键词高亮 {} / 强调 {} / 口播 {}); 字号统一 {}px, "
          "关键词色 {}".format(
              len(seq), kw, sum(1 for s in seq if s["role"] == "emphasis"),
              sum(1 for s in seq if s["role"] == "narration"),
              (seq[0]["size"] if seq else "-"), (seq[0].get("accent") if seq else "-")),
          flush=True)
    return seq, by_role


def _fx_cell(s):
    parts = [_ROLE_CN.get(s["role"], s["role"])]
    for e in s.get("effects") or []:
        if e not in parts:
            parts.append(e)
    return " ".join(parts)


def render_md(seq, by_role, target_name, ref_name):
    """镜像参考《字幕清单.md》两表(留档用, 不参与烧录)。"""
    total = max((s["end"] for s in seq), default=0.0) or 1.0
    base = seq[0] if seq else {}
    L = ["# 字幕清单（参考风格迁移·单条字幕流）: {}".format(target_name), "",
         "> whq_clone/captions_clone 生成: 成片念白 + 参考《{}》字幕风格迁移。".format(ref_name),
         "> 字号全片统一（参考口播档）; 位置统一底部居中; 参考的彩色只用在**块内关键词**上"
         "（关键词变大换色, 整块不变色）; 特效由 LLM 在该角色的参考特效集内按语义逐块匹配。", "",
         "## 一、样式与颜色规范", "",
         "| 项 | 值 |", "|---|---|",
         "| 正文字号（全片统一） | {} |".format(base.get("size", "-")),
         "| 正文填充色 | `{}` / ASS `{}` |".format(
             base.get("color", "-"), hex_to_ass(base.get("color"))),
         "| 描边色 | `{}` |".format(base.get("outline", "-")),
         "| 关键词强调色 | `{}` / ASS `{}` |".format(
             base.get("accent", "-"), hex_to_ass(base.get("accent"))),
         "| 关键词放大倍数 | 见 ass_burn.hot_scale（WHQ_CAPTION_HOT_SCALE） |",
         "", "### 各角色特效占比", "",
         "| 角色 | 特效 | 占比 |", "|---|---|---|"]
    for role in ("emphasis", "highlight", "label", "hook", "narration"):
        segs = [s for s in seq if s["role"] == role]
        if not segs:
            continue
        rp = by_role.get(role, {})
        cov = sum(s["end"] - s["start"] for s in segs)
        L.append("| {} | {} | {:.3f} |".format(
            _ROLE_CN.get(role, role), "、".join(rp.get("effects") or []) or "-", cov / total))
    L += ["", "## 二、完整时间轴", "",
          "| 时间(s) | 内容 | 关键词 | 样式/特效 |", "|---|---|---|---|"]
    for s in seq:
        L.append("| {:.1f}–{:.1f} | {} | {} | {} |".format(
            s["start"], s["end"], s["phrase"], s.get("keyword") or "-", _fx_cell(s)))
    return "\n".join(L) + "\n"


def dump_md(seq, by_role, out_path, target_name, ref_name):
    """写留档 md; 失败不阻断烧录。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_md(seq, by_role, target_name, ref_name))
    except OSError as exc:
        print("[captions_clone] 字幕清单落盘失败(忽略): {}".format(str(exc)[:120]), flush=True)
    return out_path
