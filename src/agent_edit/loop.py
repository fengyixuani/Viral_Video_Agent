"""纯 Agent 剪辑链路：剪辑 Agent + 审片 Agent + 重剪循环（agent_cut 分支）。

流程（最多 MAX_LOOPS 轮）：
  1. 剪辑 Agent（ReAct）：不再局限于每个 slot 的预分配候选，而是用召回工具从**全量素材池**
     自由检索、按需 VLM 验证、再放入；放入时后台检测重叠（同片段/同口播被多槽占用），
     重叠触发审核 Agent 裁决归属，输家 slot 同轮重新召回另选。字幕用所放片段自己的口播 speech。
  2. 剪辑器（editor.build_video，纯 ffmpeg）：执行 placements → 样片 + 逐操作记录 ops。
  3. 审片 Agent（默认 qwen 视觉 / 可选 Gemini 画面+声音）：直接看样片 + 对照 DNA + 看本轮 ops
     + 历轮简要，判定是否符合预期；不符合就给出重剪命令。
  4. 通过则输出；否则带反馈回到 1 重剪。达到轮次上限仍不达标 → 输出历轮最佳样片并标注
     「素材受限」。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time

import as_core
import asr_cache
import obs
from skills import get as get_skill
from agent_edit import editor
from agent_edit import gemini_review
from agent_edit import arbiter
from agent_edit import tts as edit_tts
from agent_edit import tools as edit_tools
from agent_edit.tools import EditToolbox, ranges_overlap, normalize_speech, render_tools_spec

_log = obs.get_logger("agent_edit_loop")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BGM = os.path.join(os.path.dirname(AGENT_ROOT), "Viral_Video", "带货素材", "dataset", "可商用bgm",
                           "时尚动感放克律动 Funk Caravan Main_爱给网_aigei_com.mp3")
MAX_LOOPS = int(os.getenv("AGENT_EDIT_MAX_LOOPS", "5"))


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(AGENT_ROOT, path))


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _load_context(strategy_abs: str, strategy: dict) -> dict:
    """加载全量素材池 + DNA 分镜结构。

    优先读 strategy 同目录的 connector_context.json（含 shots 结构 + 全部 segments 素材池）；
    缺失时退化用 strategy.user_asset_bank 里的 primary+alternates 拼一个小池，保证不断链。
    """
    ctx_path = os.path.join(os.path.dirname(strategy_abs), "connector_context.json")
    if os.path.isfile(ctx_path):
        try:
            with open(ctx_path, "r", encoding="utf-8") as fh:
                ctx = json.load(fh)
            if isinstance(ctx, dict) and ctx.get("segments"):
                return ctx
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("connector_context 读取失败(%s)，退化用 strategy 素材池", exc)
    # 退化：从 user_asset_bank 收集片段（primary + alternates）
    segs, seen = [], set()
    for asset in strategy.get("user_asset_bank", []) or []:
        for src in [asset] + list(asset.get("alternate_assets", []) or []):
            gid = src.get("asset_id") or src.get("global_asset_id") or ""
            path = src.get("source_path", "")
            key = gid or f"{path}::{src.get('source_time_range','')}"
            if not path or key in seen:
                continue
            seen.add(key)
            segs.append({
                "global_asset_id": gid or key,
                "source_path": path,
                "source_time_range": src.get("source_time_range", ""),
                "one_sentence_summary": src.get("summary", "") or src.get("visual_description", ""),
                "visual_description": src.get("visual_description", ""),
                "speech_or_text": src.get("speech_or_text", ""),
                "keywords": src.get("strengths", []) or [],
            })
    shots = [{"slot_id": (a.get("concrete_editing_plan") or {}).get("slot_id") or a.get("slot_id"),
              "role": (a.get("suitable_roles") or [""])[0] if isinstance(a.get("suitable_roles"), list) else a.get("suitable_roles", ""),
              "want": a.get("visual_description", "")}
             for a in strategy.get("user_asset_bank", []) or []]
    return {"shots": shots, "segments": segs}


def _load_inputs(strategy: dict, context: dict) -> dict:
    """抽出 DNA 概要 + 待填 slot（slot_id/角色/意图/目标时长/breakdown）。

    slot 结构来自 connector_context 的 shots（DNA 角色/意图更完整），按 slot_id 对齐；
    候选片段不再预分配——剪辑 Agent 会用召回工具从全量 segments 池自己找。
    """
    shots_meta = {s.get("slot_id"): s for s in (context.get("shots") or []) if isinstance(s, dict)}
    slots = []
    for item in strategy.get("editing_timeline", []) or []:
        if item.get("action") != "use_user_asset":
            continue
        sid = item.get("slot_id", "")
        meta = shots_meta.get(sid, {})
        t0, t1 = _parse_range(item.get("target_time_range", ""))
        target_dur = round(t1 - t0, 2) if t1 > t0 else float(meta.get("duration") or 3.0)
        slots.append({
            "slot_id": sid,
            "role": meta.get("role", "") or item.get("role", ""),
            "want": meta.get("want", "") or item.get("caption_text", ""),
            "breakdown": meta.get("breakdown", []) or [],
            "target_duration": target_dur,
        })
    meta = strategy.get("metadata", {}) or {}
    return {
        "product_name": meta.get("scheme", "") or meta.get("project_name", "") or "目标商品",
        "narrative_structure": strategy.get("overall_editing_strategy", {}) if isinstance(strategy.get("overall_editing_strategy"), dict) else {},
        "slots": slots,
    }


_EDIT_FALLBACK = (
    "你是短视频剪辑 Agent，用 ReAct 方式工作：每次只输出**一个** JSON 动作，我执行后把结果反馈给你，"
    "你再决定下一步，直到所有 slot 都放好片、输出 finish。\n"
    "目标：为爆款 DNA 的每个 slot 从**全量用户素材池**里挑最贴合其角色/意图的片段，剪出一条竖屏带货成片。\n"
    "你**不再被限制在某个预分配候选池**——用召回工具自己从全池找素材、（拿不准时）验证、再放入。\n"
    "可用工具（每次只输出其中一个动作的 JSON）：\n{{tools}}\n"
    "硬规则：\n"
    "- **成片保留素材原声，字幕必须用你所放那个片段自己的 speech（口播原话）**，别用参考爆款字幕、别照抄；"
    "speech 为空则 burn_caption=false。caption 可轻微精简断句但语义须与 speech 一致。\n"
    "- 一个片段/同一句口播只能用于一个 slot，不要重复占用。\n"
    "- target_duration 参考 slot 目标时长，可按节奏微调；必要时 speed 0.8~1.6。\n"
    "- 有 review_feedback（上一轮审片问题/命令）时，必须针对性地重新召回替换、改 trim/字幕/时长。\n"
    "- 优先把关键转化节点（痛点/产品登场等靠前 slot）配到最贴合的素材。"
)

_REVIEW_FALLBACK = (
    "你是短视频审片 Agent。你会看到一条**剪好的成片**（视频）以及它对照的爆款 DNA、本轮剪辑操作记录、"
    "以及历轮简要。你的任务：判断这条成片是否符合爆款 DNA 的预期（结构/节奏/卖点表达/字幕与画面匹配/"
    "整体观感），并指出问题、给出可执行的重剪命令。\n"
    "评估要点：镜头顺序与叙事是否贴合 DNA；单镜时长/节奏是否合适；字幕是否与画面匹配、有无错位；"
    "有无明显重复/突兀/黑屏/空镜；整体是否像一条完整的带货短视频。\n"
    "严格只输出 JSON：{\n"
    '  "pass": true/false,            // 是否已符合预期\n'
    '  "score": 0-100,                // 综合评分\n'
    '  "problems": [{"slot_id":"S0x","issue":"具体问题","fix":"该怎么改"}],\n'
    '  "commands": ["给剪辑 Agent 的重剪指令（具体到镜头/操作）"],\n'
    '  "thought": "一句话总体判断",\n'
    '  "material_limited": true/false // 若判断是"素材本身撑不起 DNA、再剪也只能这样"，置 true\n'
    "}"
)


def _skill_prompt(skill_id: str, fallback: str) -> str:
    """读 skills_defs 里该 Agent 的 SKILL.md 作为 system prompt；缺失时用内置兜底。"""
    sk = get_skill(skill_id)
    return sk.prompt_hint if (sk and sk.prompt_hint) else fallback


def edit_system_prompt() -> str:
    """剪辑 Agent 的 system prompt = SKILL.md 角色/约束 + 工具注册表渲染出的工具清单。"""
    return _skill_prompt("agent_edit_editor", _EDIT_FALLBACK).replace("{{tools}}", render_tools_spec())


def review_prompt() -> str:
    return _skill_prompt("agent_edit_reviewer", _REVIEW_FALLBACK)


async def _run_llm(system: str, user: str, *, media=None, vision=False, tag: str = ""):
    """跑一次 LLM，捕获思考(reasoning)+输出(content)。传 tag 时把两者都写进 log，方便事后查原因。"""
    content = ""
    reasoning_parts = []
    async for item in as_core.stream(system, user, vision=vision, media=media):
        if item.get("reasoning"):
            reasoning_parts.append(str(item["reasoning"]))
        if "content" in item:
            content = item["content"]
    reasoning = "".join(reasoning_parts).strip()
    if tag:
        if reasoning:
            _log.info("%s reasoning: %s", tag, reasoning[:2000])
        _log.info("%s output: %s", tag, (content or "").strip()[:4000])
    return as_core.parse_json(content) if content.strip() else {}


def _placements_to_clips(placements: dict, slots: list, tts_by_slot: dict = None) -> list:
    """把每个 slot 的最终 placement 按 slot 顺序转成 editor 需要的 clips。

    若该 slot 用了 tts_clone（克隆配音），则该镜改用 TTS 音频替换原声、字幕=配音文案。
    """
    tts_by_slot = tts_by_slot or {}
    clips = []
    for s in slots:
        p = placements.get(s["slot_id"])
        if not p:
            continue
        tts = tts_by_slot.get(s["slot_id"])
        if tts and tts.get("audio_path"):
            # 克隆配音：字幕=改写文案，音轨=TTS wav，时长以配音为准
            clips.append({
                "slot_id": s["slot_id"],
                "source_path": p["source_path"],
                "source_time_range": p.get("source_time_range", ""),
                "target_duration": float(tts.get("duration") or p.get("target_duration") or s.get("target_duration") or 3.0),
                "caption_text": (tts.get("text") or "").strip(),
                "burn_caption": bool((tts.get("text") or "").strip()),
                "tts_audio_path": tts["audio_path"],
                "speed": 1.0,
            })
            continue
        caption = (p.get("caption") or "").strip() or (p.get("speech") or "").strip()
        burn = p.get("burn_caption", True) is not False
        clips.append({
            "slot_id": s["slot_id"],
            "source_path": p["source_path"],
            "source_time_range": p.get("source_time_range", ""),
            "target_duration": float(p.get("target_duration") or s.get("target_duration") or 3.0),
            "caption_text": caption if burn else "",
            "burn_caption": burn,
            "speed": float(p.get("speed") or 1.0),
        })
    return clips


@edit_tools.edit_tool_handler("tts_clone")
async def _handle_tts_clone(ctx: dict, action: dict) -> dict:
    """tts_clone 工具执行器：用参考片段音色把改写文案念出来，产出 wav 记到该 slot。"""
    slot_id = action.get("slot_id", "")
    ref_gid = action.get("ref_global_asset_id", "")
    text = (action.get("text") or "").strip()
    toolbox = ctx.get("toolbox")
    seg = toolbox.by_gid.get(ref_gid) if (toolbox and ref_gid) else None
    if not slot_id or slot_id not in ctx.get("slot_meta", {}):
        return {"ok": False, "error": "slot_id 无效"}
    if not seg:
        return {"ok": False, "error": f"参考片段无效：{ref_gid}"}
    if not text:
        return {"ok": False, "error": "缺少要配音的文案 text"}
    tts_dir = os.path.join(AGENT_ROOT, "uploads", "tts")
    os.makedirs(tts_dir, exist_ok=True)
    out_wav = os.path.join(tts_dir, f"tts_{slot_id}_{int(time.time() * 1000) % 1000000}.wav")
    res = await asyncio.to_thread(edit_tts.clone, seg.get("source_path", ""), seg.get("source_time_range", ""),
                                  seg.get("speech_or_text", ""), text, out_wav)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "TTS 失败")}
    ctx.setdefault("tts_by_slot", {})[slot_id] = {
        "audio_path": out_wav, "text": text, "duration": res.get("duration", 0.0)}
    return {"ok": True, "slot_id": slot_id, "duration": res.get("duration", 0.0),
            "note": "已生成克隆配音；该镜将改用此配音、字幕=该文案"}


async def _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_system,
                          base_placements=None, base_tts=None, revise_slots=None):
    """剪辑 Agent 的 ReAct 循环：自由召回 → 按需验证 → 放入（后台查重叠→仲裁）→ finish。

    增量重剪：base_placements/base_tts 是上一轮的成片基线（继承过来，不重排）；
    revise_slots 是审片点名要改的 slot——只有这些进入待办，其余 slot 默认保留上一轮结果，
    避免"为修一个镜头把其它已 OK 的镜头也重排坏了"的回归。首轮 base 为空则填全部 slot。

    作为异步生成器：过程中 yield step 事件；最后 yield
    ``{"__edit_result__": True, "placements": {...}, "notes": [...], "tts_by_slot": {...}}``。
    """
    placements: dict = dict(base_placements or {})   # 继承上一轮成片基线
    tts_by_slot: dict = dict(base_tts or {})
    used: list = []                # [{gid, source_path, range, speech_norm, slot_id}]
    slot_meta = {s["slot_id"]: s for s in slots}
    scratch: list = []             # 最近若干步 {action, observation}
    notes: list = []
    max_steps = len(slots) * 6 + 12
    tag = f"[{rid}] loop{loop} 剪辑Agent"

    def _register(slot_id, seg):
        used[:] = [u for u in used if u["slot_id"] != slot_id]
        used.append({"gid": seg.get("global_asset_id", ""), "source_path": seg.get("source_path", ""),
                     "range": seg.get("source_time_range", ""),
                     "speech_norm": normalize_speech(seg.get("speech", "")), "slot_id": slot_id})

    # 把继承来的基线 placements 登记进 used，保证重叠检测/召回去重对"保留的镜头"也生效
    for _sid, _p in placements.items():
        _register(_sid, _p)

    if base_placements and revise_slots:
        # 增量重剪：只把审片点名的 slot 列为待办；其余保留基线
        unfilled = [s for s in revise_slots if s in slot_meta]
    else:
        # 首轮（或没有可解析的点名）：填全部 slot
        unfilled = [s["slot_id"] for s in slots]

    def _overlap_slot(slot_id, path, rng, speech):
        sn = normalize_speech(speech)
        for u in used:
            if u["slot_id"] == slot_id:
                continue
            if ranges_overlap(path, rng, u["source_path"], u["range"]):
                return u["slot_id"]
            if sn and sn == u["speech_norm"]:
                return u["slot_id"]
        return None

    incremental = bool(base_placements and revise_slots)
    step = 0
    while step < max_steps:
        step += 1
        # 按 slot 顺序给出「叙事上下文」：每镜的目的(role/want) + 当前这镜的语音内容
        # （tts 配音文案 > 原声口播/字幕 > 未定）。tts_clone 写配音时要据此衔接前后镜头。
        narrative = []
        for s in slots:
            sid = s["slot_id"]
            p = placements.get(sid)
            voice = ""
            if sid in tts_by_slot:
                voice = tts_by_slot[sid].get("text", "")
            elif p:
                voice = p.get("caption") or p.get("speech") or ""
            narrative.append({"slot_id": sid, "role": s["role"], "want": s["want"],
                              "voice": voice or "(未定)"})
        agent_user = json.dumps({
            "dna": dna,
            "slots": [{"slot_id": s["slot_id"], "role": s["role"], "want": s["want"],
                       "target_duration": s["target_duration"]} for s in slots],
            "narrative": narrative,
            "review_feedback": review_feedback,
            "history_brief": [{"loop": h["loop"], "problems": h.get("problems", [])} for h in history[-2:]],
            "baseline_placements": [{"slot_id": k, "gid": v.get("global_asset_id"),
                                     "caption": v.get("caption") or v.get("speech", ""),
                                     "tts": k in tts_by_slot} for k, v in placements.items()],
            "slots_to_revise": unfilled,
            "recent_steps": scratch[-8:],
            "instruction": (
                ("增量重剪模式：baseline_placements 是上一轮已成片的镜头，**只修订 slots_to_revise 里点名的 slot**，"
                 "其余镜头保持不动、不要重新召回或改动。改完点名的 slot 就输出 {\"action\":\"finish\"}。")
                if incremental else
                "只输出一个动作 JSON。所有 slot 都放好后输出 {\"action\":\"finish\"}。"),
        }, ensure_ascii=False)
        action = await _run_llm(edit_system, agent_user, tag=(tag if step == 1 else ""))
        act = str(action.get("action", "")).lower()

        if act == "finish" or (not act and not unfilled):
            break

        if act == "retrieve":
            sid = action.get("slot_id", "")
            query = action.get("query", "") or slot_meta.get(sid, {}).get("want", "")
            res = toolbox.retrieve(query, top_k=int(action.get("top_k") or 6),
                                   exclude_gids=[u["gid"] for u in used])
            cands = res["candidates"]
            obs = {"ok": True, "slot_id": sid, "backend": res.get("backend"), "pool_size": res.get("pool_size"),
                   "candidates": [{"global_asset_id": c["global_asset_id"], "summary": c["summary"],
                                   "speech": c["speech"], "source_time_range": c["source_time_range"],
                                   "duration": c["duration"], "keywords": c["keywords"]} for c in cands]}
            _log.info("%s retrieve slot=%s query=%s -> %d cand", tag, sid, query, len(cands))
            scratch.append({"action": f"retrieve[{sid}] {query}", "observation": obs})

        elif act == "verify":
            gid = action.get("global_asset_id", "")
            vr = await toolbox.verify(gid, action.get("question", ""))
            obs = {"ok": not vr.get("error"), "global_asset_id": gid,
                   "observation": vr.get("observation", "")[:600], "error": vr.get("error", "")}
            _log.info("%s verify gid=%s -> %s", tag, gid, (vr.get("observation") or vr.get("error"))[:120])
            scratch.append({"action": f"verify {gid}", "observation": obs})

        elif act == "place":
            sid = action.get("slot_id", "")
            gid = action.get("global_asset_id", "")
            seg = toolbox.by_gid.get(gid)
            if not sid or sid not in slot_meta or not seg:
                scratch.append({"action": f"place {sid}", "observation": {"ok": False, "error": "slot_id 或 global_asset_id 无效"}})
                continue
            path = seg.get("source_path", "")
            rng = action.get("source_time_range") or seg.get("source_time_range", "")
            speech = seg.get("speech_or_text", "")
            conflict_slot = _overlap_slot(sid, path, rng, speech)
            if conflict_slot:
                # 后台重叠 → 触发仲裁 Agent
                yield _estep(rid, f"arb{loop}-{sid}", "重叠检测触发审核", state="running",
                             thought=f"{sid} 想放的素材/口播与 {conflict_slot} 重叠，交审核 Agent 裁决归属")
                verdict = await arbiter.arbitrate_overlap(
                    {"summary": seg.get("one_sentence_summary", ""), "speech": speech, "source_time_range": rng},
                    [{"slot_id": sid, **{k: slot_meta[sid][k] for k in ("role", "want")}},
                     {"slot_id": conflict_slot, **{k: slot_meta[conflict_slot][k] for k in ("role", "want")}}])
                winner = verdict["winner_slot_id"]
                _log.info("%s overlap %s vs %s -> winner=%s (%s)", tag, sid, conflict_slot, winner, verdict.get("reason"))
                yield _estep(rid, f"arb{loop}-{sid}", "审核裁决重叠归属",
                             thought=f"审核判定 {winner} 保留该素材（{verdict.get('reason','')}），{'; '.join(verdict.get('losers',[]))} 需另选",
                             observation=json.dumps(verdict, ensure_ascii=False))
                if winner != sid:
                    # 本次放入的是输家 → 拒绝，要求另选
                    scratch.append({"action": f"place {sid}", "observation": {
                        "ok": False, "overlap_with": conflict_slot,
                        "arbitration": f"该素材判归 {conflict_slot}；{sid} 请用 retrieve 另选一个不冲突的片段",
                        "used_gids": [u["gid"] for u in used]}})
                    continue
                # 本次是赢家 → 抢占，把输家 conflict_slot 的 placement 移除并重新列入待填
                placements.pop(conflict_slot, None)
                tts_by_slot.pop(conflict_slot, None)
                used[:] = [u for u in used if u["slot_id"] != conflict_slot]
                if conflict_slot not in unfilled:
                    unfilled.append(conflict_slot)
                scratch.append({"action": f"place {sid}", "observation": {
                    "ok": True, "won_over": conflict_slot,
                    "note": f"{sid} 抢到该素材；{conflict_slot} 需重新召回另选"}})
            else:
                scratch.append({"action": f"place {sid}", "observation": {"ok": True}})
            # 重新放片 = 该 slot 的画面变了，清掉旧的克隆配音（如需要 Agent 会再调 tts_clone）
            tts_by_slot.pop(sid, None)
            placements[sid] = {
                "global_asset_id": gid, "source_path": path, "source_time_range": rng,
                "target_duration": action.get("target_duration") or slot_meta[sid]["target_duration"],
                "caption": action.get("caption", ""), "speech": speech,
                "burn_caption": action.get("burn_caption", True), "speed": action.get("speed", 1.0),
            }
            _register(sid, {"global_asset_id": gid, "source_path": path, "source_time_range": rng, "speech": speech})
            if sid in unfilled:
                unfilled.remove(sid)
            if action.get("note"):
                notes.append(str(action["note"]))
        else:
            # 扩展工具的执行插件点：别人用 @edit_tool_handler 注册的新工具走这里
            handler = edit_tools.get_tool_handler(act)
            if handler:
                ctx = {"toolbox": toolbox, "used": used, "placements": placements,
                       "slot_meta": slot_meta, "notes": notes, "tts_by_slot": tts_by_slot}
                try:
                    obs = await handler(ctx, action)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("%s tool '%s' handler failed: %s", tag, act, exc)
                    obs = {"ok": False, "error": str(exc)[:200]}
                scratch.append({"action": str(act), "observation": obs if isinstance(obs, dict) else {"ok": True, "result": obs}})
            else:
                scratch.append({"action": str(act), "observation": {"ok": False, "error": "未知动作，请用 retrieve/verify/place/finish"}})

    yield {"__edit_result__": True, "placements": placements, "notes": notes, "tts_by_slot": tts_by_slot}


def _estep(rid, key, title, *, thought="", state="done", observation=None):
    ev = {"type": "step", "phase": "Agent 剪辑", "key": f"{key}-{rid}", "state": state,
          "title": title, "thought": thought}
    if observation is not None:
        ev["observation"] = observation
    return ev


def _ops_brief(result: dict) -> list:
    out = []
    for o in result.get("ops", []):
        if o.get("op") == "trim":
            out.append(f"{o.get('slot_id')}: {o.get('source')} {o.get('source_time_range')} {o.get('target_duration')}s "
                       f"speed={o.get('speed')} 字幕{'已烧' if o.get('caption_burned') else '未烧'}"
                       f"{' 克隆配音' if o.get('tts') else ''} {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
        else:
            out.append(f"{o.get('op')}: {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
    return out


def _flagged_slots(review: dict, all_slot_ids: list) -> list:
    """从审片结果里解析出「被点名要改」的 slot（problems.slot_id + commands 文本里的 Sxx）。

    解析不到（如只写"全局"）时返回全部 slot——安全兜底为全量重排。
    """
    bits = [str(p.get("slot_id", "")) for p in (review.get("problems", []) or [])]
    bits += [str(c) for c in (review.get("commands", []) or [])]
    combined = " ".join(bits)
    flagged = []
    for sid in all_slot_ids:
        num = sid.lstrip("S").lstrip("0") or "0"
        if sid in combined or re.search(rf"S0*{num}\b", combined):
            flagged.append(sid)
    return flagged or list(all_slot_ids)


def _asr_clip_check(clips: list) -> list:
    """用缓存的 ASR 句级时间戳，逐镜判断结束点是否切在句子中间（口播被截断），并给出该镜内的 ASR 文本。

    供审片 Agent 参考：ASR 时间戳比"凭听感"更客观地判断截断；ASR 文本也帮审片对比"听到的"
    是否与"ASR 以为说的"一致（识别成杂音/指导语时两者会对不上）。仅对有 ASR 缓存的源片段产出。
    """
    rows = []
    for c in clips or []:
        sp = c.get("source_path", "")
        if not sp:
            continue
        asr = asr_cache.get(sp)
        segs = (asr or {}).get("segments") or []
        if not segs:
            continue
        try:
            a, b = str(c.get("source_time_range", "")).split("-")
            s, e = float(a), float(b)
        except (ValueError, AttributeError):
            continue
        played = [g for g in segs if float(g.get("end", 0)) > s + 0.05 and float(g.get("start", 0)) < e - 0.05]
        row = {"slot_id": c.get("slot_id"), "clip_range": c.get("source_time_range", ""),
               "asr_in_clip": "".join(g.get("text", "") for g in played)[:140]}
        # 结束截断：有句子在 e 之前开始、却在 e 之后才结束 → 这句被切断
        for g in segs:
            gs, ge = float(g.get("start", 0)), float(g.get("end", 0))
            if gs < e - 0.15 and ge > e + 0.25 and ge > s:
                row["truncated_at_end"] = True
                row["cut_sentence"] = g.get("text", "")
                row["suggest_extend_end_to"] = round(ge + 0.15, 2)
                break
        rows.append(row)
    return rows


async def agent_edit_stream(strategy_path: str, *, enable_bgm: bool = True, max_loops: int = None,
                            review_model: str = "qwen"):
    """产出 step / edit_log / agent_edit_done / error 事件。

    review_model: 审片 Agent 用哪个视觉后端——``qwen``（默认，as_core 视觉，只看画面）或
    ``gemini``（同时看画面+听声音，能判定字幕与口播是否一致）。
    """
    rid = time.strftime("%H%M%S")
    max_loops = max_loops or MAX_LOOPS
    review_model = (review_model or "qwen").lower()
    if review_model not in ("qwen", "gemini"):
        review_model = "qwen"

    def step(key, title, thought, state="done", observation=None):
        ev = {"type": "step", "phase": "Agent 剪辑", "key": f"{key}-{rid}", "state": state,
              "title": title, "thought": thought}
        if observation is not None:
            ev["observation"] = observation
        return ev

    strategy_abs = _abspath(strategy_path)
    if not strategy_abs or not os.path.isfile(strategy_abs):
        _log.warning("[%s] agent_edit abort: strategy not found %s", rid, strategy_path)
        yield {"type": "error", "message": f"找不到编排脚本：{strategy_path}"}
        return
    with open(strategy_abs, "r", encoding="utf-8") as fh:
        strategy = json.load(fh)
    context = _load_context(strategy_abs, strategy)
    inputs = _load_inputs(strategy, context)
    slots = inputs["slots"]
    if not slots:
        _log.warning("[%s] agent_edit abort: no editable slots (strategy=%s)", rid, strategy_abs)
        yield {"type": "error", "message": "没有可剪辑的镜头（都是补拍/AIGC 生成或候选缺失）。"}
        return
    segments = [s for s in (context.get("segments") or []) if isinstance(s, dict)]
    if not segments:
        _log.warning("[%s] agent_edit abort: empty material pool", rid)
        yield {"type": "error", "message": "素材池为空，无法召回剪辑。"}
        return
    task_id = "agentedit_" + hashlib.sha1(strategy_abs.encode("utf-8")).hexdigest()[:12]
    toolbox = EditToolbox(task_id, segments)
    edit_sys = edit_system_prompt()   # SKILL.md 角色/约束 + 工具注册表渲染的工具清单
    review_p = review_prompt()        # 审片 Agent 的 SKILL.md
    dna = {"product_name": inputs["product_name"], "narrative_structure": inputs["narrative_structure"],
           "slots": [{"slot_id": s["slot_id"], "role": s["role"], "want": s["want"],
                      "target_duration": s["target_duration"]} for s in slots]}
    bgm = DEFAULT_BGM if enable_bgm and os.path.isfile(DEFAULT_BGM) else ""
    _log.info("[%s] agent_edit start strategy=%s slots=%d pool=%d max_loops=%d bgm=%s review_model=%s",
              rid, os.path.basename(strategy_abs), len(slots), toolbox.pool_size(), max_loops, bool(bgm), review_model)

    final_dir = os.path.join(AGENT_ROOT, "uploads", "final")
    os.makedirs(final_dir, exist_ok=True)
    slug = "agentcut_" + time.strftime("%Y%m%d_%H%M%S")

    history = []          # 每轮简要（供审片看历轮）
    best = None           # {score, video_uri, path, loop}
    review_feedback = {}
    prev_placements = {}  # 上一轮成片基线（增量重剪继承）
    prev_tts = {}
    revise_slots = None   # 审片点名要改的 slot；None=全量
    all_slot_ids = [s["slot_id"] for s in slots]

    yield step("prep", "读取编排脚本 + 建全池召回",
               f"{len(slots)} 个 DNA 槽位待填，素材池 {toolbox.pool_size()} 段可自由召回，进入 Agent 剪辑-审片循环（最多 {max_loops} 轮）",
               observation="\n".join(f"{s['slot_id']} {s['role']} 目标{s['target_duration']}s" for s in slots))

    for loop in range(1, max_loops + 1):
        # 1) 剪辑 Agent：ReAct 自由召回 → 验证 → 放入（后台查重叠→仲裁）
        incr = bool(prev_placements and revise_slots)
        yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent{'（增量修订：' + '、'.join(revise_slots) + '）' if incr else '自由召回选片'}",
                   ("只重剪审片点名的镜头，其余保留上一轮" if incr else "从全池按 DNA 角色召回、按需验证、放入并后台查重叠"),
                   state="running")
        placements, notes = {}, []
        tts_by_slot = {}
        async for ev in _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_sys,
                                        base_placements=(prev_placements or None), base_tts=(prev_tts or None),
                                        revise_slots=revise_slots):
            if ev.get("__edit_result__"):
                placements, notes = ev["placements"], ev["notes"]
                tts_by_slot = ev.get("tts_by_slot", {})
                continue
            yield ev
        # 本轮的成片计划成为下一轮的基线（下一轮据本轮审片只改被点名的 slot）
        prev_placements, prev_tts = placements, tts_by_slot
        clips = _placements_to_clips(placements, slots, tts_by_slot)
        note = "；".join(notes[:3])
        if not clips:
            _log.warning("[%s] loop%d 剪辑Agent 未产出任何 placement", rid, loop)
            yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 选片", "本轮未放入任何镜头，跳过", state="done")
            history.append({"loop": loop, "edit_note": note, "ops": [], "review": "本轮未产出", "problems": ["剪辑 Agent 未放入镜头"]})
            review_feedback = {"commands": ["上一轮没有放入任何镜头，请用 retrieve 召回后 place 每个 slot"]}
            revise_slots = None  # 全量重排
            continue
        _log.info("[%s] loop%d 剪辑Agent placed=%d note=%s detail=%s", rid, loop, len(clips), note,
                  json.dumps([{"slot": c["slot_id"], "src": os.path.basename(c["source_path"]),
                               "range": c["source_time_range"], "dur": c["target_duration"],
                               "speed": c["speed"], "cap": c["caption_text"]} for c in clips], ensure_ascii=False)[:2000])
        yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 选片完成",
                   note or f"放入 {len(clips)} 个镜头",
                   observation=json.dumps([{"slot": c["slot_id"], "src_range": c["source_time_range"],
                                            "dur": c["target_duration"], "speed": c["speed"],
                                            "cap": c["caption_text"]} for c in clips], ensure_ascii=False))

        # 2) 执行剪辑
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"ffmpeg 逐镜 trim/字幕/拼接{'/BGM' if bgm else ''}", state="running")
        out_path = os.path.join(final_dir, f"{slug}_loop{loop}.mp4")
        result = editor.build_video(clips, out_path, bgm_path=bgm, width=720, height=1080, fps=30)
        ops_brief = _ops_brief(result)
        _log.info("[%s] loop%d 剪辑执行 clip_count=%s error=%s ops=%s", rid, loop,
                  result.get("clip_count"), result.get("error", ""),
                  json.dumps(ops_brief, ensure_ascii=False)[:3000])
        if result.get("error") or not result.get("output"):
            yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"剪辑失败：{result.get('error','')}", state="done",
                       observation="\n".join(ops_brief))
            history.append({"loop": loop, "edit_note": note, "ops": ops_brief,
                            "review": "本轮剪辑失败", "problems": [result.get("error", "")]})
            review_feedback = {"commands": [f"上一轮剪辑失败：{result.get('error','')}，请调整选片/时长后重试"]}
            revise_slots = None  # 剪辑失败 → 全量重排
            continue
        video_uri = f"uploads/final/{os.path.basename(out_path)}"
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"成片 {result['clip_count']} 镜已生成",
                   observation="\n".join(ops_brief))
        yield {"type": "agent_edit_sample", "loop": loop, "video_uri": video_uri}

        # 3) 审片 Agent（看视频）
        model_label = "Gemini（画面+声音）" if review_model == "gemini" else "默认视觉模型"
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 看片", f"用 {model_label} 对照 DNA 审阅成片，定位问题", state="running")
        # 审片用的 DNA 去掉 target_duration —— 避免审片盯着"单镜差几秒"扣分/要求降速凑时长（优先看效果）
        review_dna = dict(dna)
        review_dna["slots"] = [{k: v for k, v in s.items() if k != "target_duration"} for s in dna.get("slots", [])]
        asr_check = _asr_clip_check(clips)  # ASR 句级时间戳：客观判断口播是否被截断 + 该镜 ASR 文本
        review_user = json.dumps({
            "dna": review_dna, "this_loop_ops": ops_brief,
            "asr_check": asr_check,
            "history": [{"loop": h["loop"], "note": h.get("edit_note", ""), "problems": h.get("problems", [])} for h in history[-4:]],
            "instruction": "请观看视频，对照 DNA 判定是否符合预期；镜头时长以内容表达自然为准，不要要求与目标秒数一致。"
                           "参考 asr_check 里的 ASR 时间戳判断口播是否被截断；并判断每镜声音是否为真实产品口播（而非环境杂音/拍摄现场指导语）。",
        }, ensure_ascii=False)
        if review_model == "gemini":
            review = await asyncio.to_thread(gemini_review.review_video, review_p, review_user, out_path)
            if review.get("_error"):
                # Gemini 审片失败 → 回退默认视觉模型，避免整轮卡死
                _log.warning("[%s] loop%d Gemini 审片失败(%s)，回退默认视觉模型", rid, loop, review.get("_error"))
                yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 看片",
                           f"Gemini 审片失败（{review.get('_error')}），回退默认视觉模型", state="running")
                review = await _run_llm(review_p, review_user, vision=True,
                                        media=[{"type": "video", "url": video_uri}],
                                        tag=f"[{rid}] loop{loop} 审片Agent(回退qwen)")
            else:
                _log.info("[%s] loop%d 审片Agent(gemini) output: %s", rid, loop,
                          json.dumps(review, ensure_ascii=False)[:4000])
        else:
            review = await _run_llm(review_p, review_user, vision=True,
                                    media=[{"type": "video", "url": video_uri}],
                                    tag=f"[{rid}] loop{loop} 审片Agent")
        score = float(review.get("score") or 0)
        passed = bool(review.get("pass"))
        problems = review.get("problems", []) or []
        if best is None or score > best["score"]:
            best = {"score": score, "video_uri": video_uri, "path": out_path, "loop": loop}
        _log.info("[%s] loop%d 审片Agent pass=%s score=%.0f material_limited=%s thought=%s problems=%s commands=%s",
                  rid, loop, passed, score, bool(review.get("material_limited")),
                  review.get("thought", ""),
                  json.dumps([{"slot": p.get("slot_id", ""), "issue": p.get("issue", ""), "fix": p.get("fix", "")}
                              for p in problems], ensure_ascii=False)[:2000],
                  json.dumps(review.get("commands", []), ensure_ascii=False)[:1500])
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 判定",
                   f"{'通过' if passed else '未通过'} · {score:.0f} 分 · {review.get('thought','')}",
                   observation=json.dumps(review, ensure_ascii=False)[:1500])

        history.append({"loop": loop, "edit_note": note, "ops": ops_brief,
                        "review": review.get("thought", ""), "score": score,
                        "problems": [f"{p.get('slot_id','')}: {p.get('issue','')}" for p in problems]})

        if passed:
            _log.info("[%s] agent_edit done PASS loop=%d score=%.0f video=%s", rid, loop, score, video_uri)
            yield {"type": "agent_edit_done", "video_uri": video_uri, "final_path": out_path,
                   "loops": loop, "score": score, "material_limited": False,
                   "review_model": review_model,
                   "verdict": review.get("thought", "符合预期"), "history": history}
            return
        if review.get("material_limited"):
            _log.info("[%s] loop%d 审片判定素材受限，输出当前最佳样片", rid, loop)
            yield step("stop", "审片判定素材受限", "审片 Agent 认为再剪也只能这样，直接输出当前最佳样片", state="done")
            break
        review_feedback = {"problems": problems, "commands": review.get("commands", []),
                           "last_score": score}
        revise_slots = _flagged_slots(review, all_slot_ids)  # 下一轮只增量修订被点名的 slot
        _log.info("[%s] loop%d 下一轮增量修订 slots=%s", rid, loop, revise_slots)

    # 达到上限 / 素材受限：输出最佳样片
    b = best or {"video_uri": "", "path": "", "score": 0, "loop": max_loops}
    _log.info("[%s] agent_edit done BEST loops=%d best_loop=%s score=%s video=%s",
              rid, len(history), b.get("loop"), b.get("score"), b.get("video_uri"))
    yield {"type": "agent_edit_done", "video_uri": b["video_uri"], "final_path": b["path"],
           "loops": len(history), "score": b["score"], "material_limited": True,
           "review_model": review_model,
           "verdict": "已达轮次上限/素材受限，输出历轮最佳样片", "history": history}
