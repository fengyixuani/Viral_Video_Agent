"""agent_tools — 把 whq 的「保留用户原声」能力做成剪辑 Agent 的一个工具。

whq_clone 走 Agent 形态（src/editing/loop.py 的剪辑-审片循环）后，选片权交给了剪辑
Agent。但 whq 之所以听感真实，靠的是**句子级对窗**：原声段的窗口要落在说话人的自然停顿
上，不能把一句话切一半，且字幕必须等于窗口内实际念出来的字。普通 ``place`` 不做这件事。

本模块注册一个 ``place_original`` 工具：语义 = "用这条素材的原声放这一段"，执行时
  1) 用素材池里随 connector_context 下发的逐字 ASR（whq_speech.asr_items）找出窗口内的分句；
  2) 把窗口吸附到「说完整句」的边界（句首/句尾都不切半句）；
  3) 字幕 = 窗口内逐字文本（字幕 == 语音 == 画面）。

不改 loop.py 的主循环：声明走 tools.register_edit_tool，执行走 tools.edit_tool_handler
（loop.py 已有的扩展工具插件点）。import 本模块即完成挂载。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_EDITING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDITING not in sys.path:
    sys.path.insert(0, _EDITING)

# 必须拿到**与 loop.py 相同的** tools 模块对象，否则注册的 handler 落在另一份
# _TOOL_HANDLERS 上，loop 里 get_tool_handler 找不到（loop 用 `from editing import tools`）。
try:
    from editing import tools as edit_tools
except ImportError:  # 独立跑（sys.path 只挂到 src/editing）时退化
    import tools as edit_tools

# 相邻字间隔 >= 此秒数视为一个分句结束（与 voice_policy.SENT_GAP 同源，保持行为一致）
SENT_GAP = float(os.getenv("WHQ_SENT_GAP", "0.35"))
WIN_LEAD = 0.10   # 句首前留一点起势
WIN_TAIL = 0.12   # 句尾留衰减，不贴字尾硬切
# 末句被素材切片边界截断时，允许把结束点往后延多少秒去把那句说完
EXTEND_TAIL = float(os.getenv("WHQ_ORIGINAL_EXTEND_TAIL", "3.0"))


def skip_slot(slot_id: str, reason: str = ""):
    """跳过某个节拍：当参考爆款这一段讲的内容（如价格活动、下单截图演示、某个参考商品独有的卖点）在**用户素材里根本不存在**时，用本工具把该段从成片里去掉。宁可少一段，也不要为了填满结构而编造素材里没有的事实、或硬塞一个不相关的画面。

    Args:
        slot_id: 要跳过的槽位 id
        reason: 为什么跳过（说明用户素材里缺什么）
    """
    raise NotImplementedError("schema 声明；执行见本模块的 handler")


def place_original(slot_id: str, global_asset_id: str, note: str = ""):
    """把某素材片段以「保留用户原声」的方式放入某 slot：系统会自动把截取窗口吸附到说话人的自然停顿处（保证这一句说完整、不切半句），并把字幕设成窗口内真实念出的原话。**该片段自带原声口播（召回结果里 has_original_voice=true）时优先用本工具**，比 place + tts_clone 更真实。原声是拍摄现场的口令/口水话（"行行行往上走这个都够了"）时不算自带口播，本工具会拒绝——那种段请 place + tts_clone。

    Args:
        slot_id: 目标槽位 id
        global_asset_id: 要放入的片段 id（须自带口播，召回结果 has_original_voice=true）
        note: 可选，本步思路一句话
    """
    raise NotImplementedError("schema 声明；执行见本模块的 handler")


def _utterances(asr_items, gap=SENT_GAP):
    """按逐字间隔切分句：返回 [{start, end, text}]。"""
    utts = []
    for it in sorted(asr_items or [], key=lambda x: float(x.get("start") or 0.0)):
        try:
            s, e = float(it.get("start")), float(it.get("end"))
        except (TypeError, ValueError):
            continue
        t = str(it.get("text") or "")
        if utts and s - utts[-1]["end"] < gap:
            utts[-1]["end"] = max(utts[-1]["end"], e)
            utts[-1]["text"] += t
        else:
            utts.append({"start": s, "end": e, "text": t})
    return utts


def _text_in_window(asr_items, win_s, win_e):
    """窗口内**真正念出来**的字。字幕只能用这个，不能用整句文本。"""
    out = []
    for it in sorted(asr_items or [], key=lambda x: float(x.get("start") or 0.0)):
        try:
            mid = (float(it.get("start")) + float(it.get("end"))) / 2.0
        except (TypeError, ValueError):
            continue
        if win_s <= mid <= win_e:
            out.append(str(it.get("text") or ""))
    return "".join(out)


def align_original_window(asr_items, start, end, target=None):
    """把 [start, end] 吸附到「说完整句」的窗口。返回 (win_start, win_end, text)。

    取与窗口相交的分句，整句纳入（句首往前不超过 WIN_LEAD，句尾留 WIN_TAIL）；目标时长
    给定时，从窗口内的整句里取能装下的最长连续句组，装不下一整句就退回原窗口。

    关键：素材切片的边界常常**把一句话切成两半**（如"小分子柠檬酸特工负责钻"后面还有话）。
    素材池下发的 asr_items 带了窗口外 ±1.5s 的上下文，这里允许把结束点**延到那句真正说完**
    （最多 EXTEND_TAIL 秒），否则保住了原声却仍然半句被截断。

    返回的 text **一律按最终窗口重算**（``_text_in_window``），不是句组的整句文本：句子起点
    落在候选窗之前时（实测 S02 那句横跨源 1.2-6.08s，候选窗只有 4.75-6.20），整句文本里
    有一半的字根本没播出来，拿它当字幕就是「字幕比人说的话多出十几个字」。
    """

    utts = [u for u in _utterances(asr_items) if u["end"] > start and u["start"] < end]
    if not utts:
        return round(start, 3), round(end, 3), ""
    # 末句被切断时（窗口结束点落在句中/紧接其后还有字），把该句补完
    tail_limit = end + EXTEND_TAIL
    for u in _utterances(asr_items):
        if u["start"] < end < u["end"] or (end - 0.05 < u["start"] < end + SENT_GAP):
            if u["end"] <= tail_limit:
                if u not in utts:
                    utts.append(u)
                end = min(tail_limit, u["end"])
    utts.sort(key=lambda x: x["start"])
    if target and target > 0:
        best = None
        for i in range(len(utts)):
            for j in range(i, len(utts)):
                # 句首**整句纳入**：不再夹回候选窗(原来是 max(start-WIN_LEAD, ...)，句子起点在
                # 候选窗之前就只截到半句尾巴)，否则窗口里播的和字幕写的就不是一回事。
                s = max(0.0, utts[i]["start"] - WIN_LEAD)
                e = min(end + WIN_TAIL, utts[j]["end"] + WIN_TAIL)
                span = e - s
                if span <= 0 or span > target * 1.35:  # 与 voice_policy 的 atempo 上限一致
                    continue
                text = _text_in_window(asr_items, s, e)
                if not text:
                    continue
                if best is None or span > best[2] - best[1]:
                    best = (text, s, e)
        if best:
            return round(max(0.0, best[1]), 3), round(best[2], 3), best[0]
    # 没有整句能装进这个节拍：退回原候选窗（docstring 说的「装不下一整句就退回原窗口」），
    # 字幕同样按窗口重算 —— 宁可字幕是半句、也要和听到的话逐字一致。
    s = max(0.0, max(start - WIN_LEAD, utts[0]["start"] - WIN_LEAD))
    e = min(end + WIN_TAIL, utts[-1]["end"] + WIN_TAIL)
    return round(s, 3), round(e, 3), _text_in_window(asr_items, s, e)


@edit_tools.edit_tool_handler("place_original")
async def _handle_place_original(ctx: dict, action: dict) -> dict:
    """执行 place_original：句子级对窗后写入 ctx['placements']（结构与 place 一致）。"""
    sid = str(action.get("slot_id") or "")
    gid = str(action.get("global_asset_id") or "")
    slot_meta = ctx.get("slot_meta") or {}
    toolbox = ctx.get("toolbox")
    if not sid or sid not in slot_meta or not toolbox:
        return {"ok": False, "error": "slot_id 无效"}
    seg = (getattr(toolbox, "by_gid", {}) or {}).get(gid)
    if not seg:
        return {"ok": False, "error": f"未知 global_asset_id：{gid}"}
    ws = seg.get("whq_speech") or {}
    items = ws.get("asr_items") or []
    start, end = edit_tools._parse_range(seg.get("source_time_range", ""))
    target = float(slot_meta[sid].get("target_duration") or 0.0)
    if not items:
        # 老编排产物只给了整段原声文本、没有逐字 ASR：做不了精细对窗，但仍要保住原声
        # （换克隆配音会口型错位）。退化为用该候选原窗口 + 整段原话作字幕。
        text = (ws.get("text") or "").strip()
        if len(text) < 2:
            return {"ok": False, "error": "该片段没有可用的原声；请改用 place，需要口播就再用 tts_clone 克隆配音"}
        win_s, win_e = start, (min(end, start + target) if target > 0 else end)
        degraded = True
    else:
        win_s, win_e, text = align_original_window(items, start, end, target)
        degraded = False
        if not text:
            return {"ok": False, "error": "窗口内没有成句的原声；请改用 place"}
    # 该段的原声已被判定不可用（太短/是现场杂音或 ASR 乱识别，如"哥四妹把脸闭住啊"），
    # 保留它等于成片里放一段听不懂的杂音 —— 那种情况必须走克隆配音。
    # 但**只要对窗后拿到的是成句真实口播**（≥ AGENT_LIPSYNC_HARD_CHARS 字）就得保原声：
    # 画面里的人在说这句话，换配音口型就对不上，这一条优先于编排的 clone 判定。
    if sid in (ctx.get("allow_tts_slots") or ()) \
            and len(text.strip()) < int(os.getenv("AGENT_LIPSYNC_HARD_CHARS", "8")):
        return {"ok": False, "error": (
            f"{sid} 的原声已判定不可用（太短或只是现场杂音/ASR 乱识别），不能保留原声。"
            "请对它用 tts_clone 配一段贴合本镜画面的解说。")}
    # 现场废话不算"可用原声"：字数够、也真有人在说，但内容是拍摄口令/口水话
    # （"行行行行往上走这个都够了"、"OK然后捏一捏那个泡沫"）。保留它成片就是一段没有
    # 信息量的现场录音，用户明确说过"原声是没有用的啊 都是废话"。
    _filler, _why = edit_tools.is_filler_speech(text)
    if not _filler:
        # 电平判据：离机位喊的口令字面像正常句子，文本规则兜不住；取窗电平极低
        # （峰值/均值双阈值）就判为现场杂音，不能当"可用原声"保留。
        _filler, _why = edit_tools.is_offmic_quiet(
            seg.get("source_path", ""), "{:.2f}-{:.2f}".format(win_s, win_e))
    if _filler:
        return {"ok": False, "error": (
            f"{sid} 这段原声是拍摄现场的废话（「{text[:24]}」：{_why}），没有信息量，不能保留。"
            "优先换一个**没有人说话**的片段（retrieve + place）再 tts_clone 配解说；"
            "本镜画面非用不可时，直接对它 place + tts_clone（口型会略有出入，但好过留一段现场口令）。")}

    rng = "{:.2f}-{:.2f}".format(win_s, win_e)
    ctx["placements"][sid] = {
        "global_asset_id": gid, "source_path": seg.get("source_path", ""),
        "source_time_range": rng,
        "target_duration": slot_meta[sid].get("target_duration"),
        "caption": text, "speech": text, "burn_caption": True, "speed": 1.0,
        "voice_source": "original",
    }
    # 保留原声的段不该再叠克隆配音
    tts_by_slot = ctx.get("tts_by_slot")
    if isinstance(tts_by_slot, dict):
        tts_by_slot.pop(sid, None)
    # 登记进重叠检测 + 从待填列表移除（与 loop.py 的 place 分支等价）
    reg = ctx.get("register")
    if callable(reg):
        reg(sid, dict(ctx["placements"][sid]))
    unfilled = ctx.get("unfilled")
    if isinstance(unfilled, list) and sid in unfilled:
        unfilled.remove(sid)
    notes = ctx.get("notes")
    if action.get("note") and isinstance(notes, list):
        notes.append(str(action["note"]))
    return {"ok": True, "slot_id": sid, "source_time_range": rng, "caption": text,
            "note": ("已保留用户原声（该编排产物没有逐字 ASR，未做句子级对窗；"
                     "重跑「生成复刻方案」可获得精确对窗）" if degraded else
                     "窗口已吸附到自然停顿（说完整句），字幕=窗口内原话")}


@edit_tools.edit_tool_handler("skip_slot")
async def _handle_skip_slot(ctx: dict, action: dict) -> dict:
    """执行 skip_slot：把该节拍从成片里去掉（清 placement/配音、从待办移除）。

    参考爆款的某些节拍（价格活动、下单演示等）在用户素材里没有对应内容，硬填只会得到
    编造的文案或不相关的画面。允许少一段，成片的其余结构照旧。
    """
    sid = str(action.get("slot_id") or "")
    if not sid or sid not in (ctx.get("slot_meta") or {}):
        return {"ok": False, "error": "slot_id 无效"}
    reason = str(action.get("reason") or "").strip() or "用户素材里没有该节拍所需的内容"
    (ctx.get("placements") or {}).pop(sid, None)
    tts_by_slot = ctx.get("tts_by_slot")
    if isinstance(tts_by_slot, dict):
        tts_by_slot.pop(sid, None)
    used = ctx.get("used")
    if isinstance(used, list):
        used[:] = [u for u in used if u.get("slot_id") != sid]
    unfilled = ctx.get("unfilled")
    if isinstance(unfilled, list) and sid in unfilled:
        unfilled.remove(sid)
    skipped = ctx.setdefault("skipped_slots", {})
    skipped[sid] = reason
    notes = ctx.get("notes")
    if isinstance(notes, list):
        notes.append("跳过 {}：{}".format(sid, reason))
    return {"ok": True, "slot_id": sid, "skipped": True, "reason": reason,
            "note": "该节拍已从成片去掉，不用再为它选片或配音"}


edit_tools.register_edit_tool(place_original)
edit_tools.register_edit_tool(skip_slot)
