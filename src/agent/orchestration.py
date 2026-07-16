"""结构优先复刻编排 Agent。

参考 Viral_Video_Split 的结构优先思路：只参考爆款的整体结构 DNA（叙事结构、节奏、
Hook、卖点顺序、CTA、总时长），用**用户素材池**自由编排出一份满足 DNA 的新脚本。
Slot 数量/时长/内容由 Agent 决定，不再逐镜头映射到参考视频。

产出：
- ``decisions``：per-shot feasibility_like（driver 用的）
- ``template``：编排后的 shot_slots 新模板（供下游 _build_plan / scriptgen 使用）
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import as_core
import obs
from skills import get as get_skill

_log = obs.get_logger("orchestrate")
SKILL_ID = "orchestration_script"


def _collect_material_pool(material_understanding: dict) -> tuple[list, dict]:
    """从素材理解结果收集素材池（一句话描述 + 该片段实际口播文字）。

    给每条素材分配一个**短稳定 id**（M01、M02…）供编排 LLM 复述——真实的 global_asset_id
    长达 80+ 字符（含 @/-/:: 与时间戳），LLM 无法逐字 echo 回来，会导致 asset_id 校验全灭、
    所有镜头误判为"生成"。``index`` 按 short_id 建键；entry 里同时保留真实 ``asset_id``。
    """
    pool = []
    index = {}
    for parsed in (material_understanding or {}).values():
        if not isinstance(parsed, dict):
            continue
        svid = parsed.get("source_video_id", "")
        parent_path = parsed.get("source_path", "")
        for seg in parsed.get("asset_segments", []) or []:
            if not isinstance(seg, dict):
                continue
            gid = seg.get("global_asset_id") or f"{svid}::{seg.get('asset_id', '')}"
            summary = seg.get("one_sentence_summary") or seg.get("visual_description", "")
            speech = seg.get("speech_or_text") or ""
            short_id = f"M{len(pool) + 1:02d}"
            entry = {
                "short_id": short_id,
                "asset_id": gid,
                "source_video_id": svid,
                "source_path": seg.get("source_path") or parent_path,
                "summary": summary,
                "speech_or_text": speech,
                "time_range": seg.get("source_time_range", ""),
            }
            pool.append(entry)
            index[short_id] = entry
    return pool, index


def _extract_structure_dna(reference_template: dict) -> dict:
    tpl = reference_template or {}
    return {
        "industry": tpl.get("industry", ""),
        "narrative_structure": tpl.get("narrative_structure", []) or [],
        "hook": tpl.get("hook", {}) or {},
        "rhythm": tpl.get("rhythm", {}) or {},
        "selling_points_order": tpl.get("selling_points_order", []) or [],
        "cta": tpl.get("cta", {}) or {},
        "total_duration_sec": tpl.get("total_duration_sec", 0) or 0,
        "packaging": tpl.get("packaging", {}) or {},
    }


def _slot_id_of(shot_id, index: int) -> str:
    if isinstance(shot_id, str) and shot_id.startswith("S"):
        return shot_id
    return f"S{int(index) + 1:02d}"


def _shot_caption(shot: dict) -> str:
    for d in shot.get("breakdown", []) or []:
        if not isinstance(d, dict):
            continue
        dim = d.get("dim", "") or ""
        if "字幕" in dim or "口播" in dim:
            return d.get("value", "") or ""
    return ""


def _set_shot_caption(shot: dict, caption: str):
    if not caption:
        return
    breakdown = shot.setdefault("breakdown", [])
    for d in breakdown:
        if isinstance(d, dict) and ("字幕" in (d.get("dim", "") or "") or "口播" in (d.get("dim", "") or "")):
            d["value"] = caption
            return
    breakdown.insert(0, {"dim": "字幕贴片", "value": caption})


_FROM_FEAS_PROMPT = (
    "你是短视频复刻的编排 Agent。素材可行性验证已经为每个镜头位（slot）在【用户素材】里找好了候选片段，"
    "每个候选带一句话描述 summary 和该片段的用户口播 speech_or_text。\n"
    "你的任务：参考爆款结构 DNA，为每个 slot 写出最终的字幕/口播文案，并从该 slot 的 candidates 里选一个首选片段。\n"
    "规则：\n"
    "- 严格按输入的 slot 顺序输出（slot_id 原样返回），**不要重新排序**。\n"
    "- **允许提前收尾**：如果剧本在某个 slot 达到自然结尾（CTA / 号召 / 收官口播，如「赶紧下单」「冲了」「去做 SPA」「就完事了」这类），"
    "请**只输出到该 slot 为止**，把它之后的所有 slot 从 slots 数组里省略——省略即视为丢弃（避免强行凑够参考视频的镜头数导致成片突兀）。"
    "如果没到自然结尾，请把每个 slot 都写出来。\n"
    "- caption 用简体中文：优先改写/采用首选候选的 speech_or_text（用户自己的口播），使其自然、符合该 slot 的叙事作用；"
    "该候选没有口播时，就依据 summary 写一句贴合画面的简短口播。禁止照抄参考视频的字幕/口播文案。\n"
    "- primary_asset_id 必须从该 slot 的 candidates.asset_id 里选；尽量让不同 slot 选不同片段以避免重复。\n"
    "- feasible=false 的 slot 说明没有合适素材，保持它需要生成/补拍，caption 可据 want 写一句。\n"
    "严格只输出 JSON：{\"slots\":[{\"slot_id\":\"S01\",\"caption\":\"...\",\"primary_asset_id\":\"...\",\"reason\":\"...\"}],"
    "\"end_reason\":\"如果提前收尾，说明在哪个 slot 收尾及原因；否则空字符串\",\"overall_note\":\"...\"}"
)


async def orchestrate_from_feasibility(reference_template: dict, feasibility: dict, *,
                                       scheme_name="", material_strategy="faithful",
                                       selected_dimensions=None, selected_trends=None,
                                       intent="") -> AsyncGenerator[dict, None]:
    """基于素材可行性验证结果编排最终脚本：把每个镜头已找到的候选片段（一句话描述）交给
    编排 Agent，让它写最终字幕并选首选；每个 slot 的多候选原样继承下去，供后续 Split 挑不重复片段。

    结束时 yield ``{"__orchestration_result__": True, "decisions": {shot_id: ...}, "template": <更新后的参考模板>}``。
    """
    template = reference_template or {}
    shots = template.get("shot_slots", []) or []
    structure_dna = _extract_structure_dna(template)
    skill_name = "结构编排（基于可行性候选）"
    base = {"lane": "orchestrate", "lane_label": "编排 Agent", "lane_skill": skill_name, "phase": "编排"}

    shot_briefs = []
    feas_by_shot = {}
    cands_by_shot = {}
    for idx, shot in enumerate(shots):
        sid = shot.get("id")
        feas = feasibility.get(str(sid)) or feasibility.get(sid) or {}
        feas_by_shot[sid] = feas
        cands = [c for c in (feas.get("candidates") or []) if isinstance(c, dict) and c.get("asset_id")]
        cands_by_shot[sid] = cands
        shot_briefs.append({
            "slot_id": _slot_id_of(sid, idx),
            "role": shot.get("role", ""),
            "want": shot.get("want", ""),
            "feasible": feas.get("status") in ("direct", "partial"),
            "candidates": [{"asset_id": c["asset_id"], "summary": c.get("summary", ""),
                            "speech_or_text": c.get("speech_or_text", ""),
                            "source_time_range": c.get("source_time_range", "")} for c in cands],
        })

    total_cands = sum(len(v) for v in cands_by_shot.values())
    yield {"type": "step", **base, "key": "orch-head", "state": "running",
           "title": "基于可行性候选编排最终脚本",
           "thought": f"把 {len(shots)} 个镜头、共 {total_cands} 条候选片段的一句话描述交给编排 Agent 写最终脚本"}

    user = json.dumps({
        "structure_dna": structure_dna,
        "shots": shot_briefs,
        "user_choices": {"scheme": scheme_name, "strategy": material_strategy,
                         "selected_dimensions": list(selected_dimensions or []),
                         "selected_trends": list(selected_trends or []), "intent": intent or ""},
    }, ensure_ascii=False)

    content = ""
    try:
        async for item in as_core.stream(_FROM_FEAS_PROMPT, user):
            if item.get("reasoning"):
                ev = {"type": "reasoning", "text": item["reasoning"]}
                ev.update(base)
                yield ev
            elif "content" in item:
                content = item["content"]
        data = as_core.parse_json(content) if content.strip() else {}
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        _log.warning("orchestrate_from_feasibility failed: %s", exc)
        data = {}

    out_by_slot = {s.get("slot_id"): s for s in (data.get("slots", []) or []) if isinstance(s, dict)}
    decisions: dict = {}
    use_count = 0
    kept_shots: list = []
    dropped_slot_ids: list = []
    for idx, shot in enumerate(shots):
        sid = shot.get("id")
        sid_str = _slot_id_of(sid, idx)
        feas = feas_by_shot.get(sid, {})
        cands = cands_by_shot.get(sid, [])
        out = out_by_slot.get(sid_str)
        # 编排 Agent 未在输出里返回的 slot = 主动省略（自然收尾之后的镜头），从最终脚本里丢掉
        if out is None:
            dropped_slot_ids.append(sid_str)
            continue
        kept_shots.append(shot)
        chosen_id = out.get("primary_asset_id")
        primary = next((c for c in cands if c["asset_id"] == chosen_id), None) \
            or next((c for c in cands if c["asset_id"] == feas.get("matched_asset_id")), None) \
            or (cands[0] if cands else None)
        caption = (out.get("caption") or "").strip() or _shot_caption(shot)
        if feas.get("status") in ("direct", "partial") and primary:
            use_count += 1
            alternates = [c for c in cands if c["asset_id"] != primary["asset_id"]]
            decisions[sid] = {
                "shot_id": sid, "status": feas.get("status", "direct"),
                "matched_asset_id": primary["asset_id"],
                "matched_source_video_id": primary.get("source_video_id", ""),
                "matched_source_path": primary.get("source_path", ""),
                "matched_time_range": primary.get("source_time_range", ""),
                "matched_summary": primary.get("summary", ""),
                "score": feas.get("score", primary.get("score", 0.0)),
                "replicable_part": feas.get("replicable_part", ""),
                "reason": out.get("reason") or feas.get("reason", "编排 Agent 基于可行性候选选定"),
                "caption": caption,
                "alternates": alternates,
            }
            _set_shot_caption(shot, caption)
        else:
            decisions[sid] = {
                "shot_id": sid, "status": "none", "matched_asset_id": "", "matched_summary": "",
                "reason": out.get("reason") or feas.get("reason", "候选里没有合适片段，需补拍或 AIGC 生成"),
                "generation_prompt": shot.get("want", ""),
                "caption": caption, "alternates": [],
            }
            _set_shot_caption(shot, caption)

    # 让编排 Agent 决定自然结尾之后就把参考模板剪短，避免下游强行凑够 12 镜产生突兀尾巴
    if dropped_slot_ids and kept_shots:
        template = dict(template)
        template["shot_slots"] = kept_shots
        template.setdefault("metadata", {})
        template.setdefault("total_duration_sec", sum(float(s.get("duration") or 0.0) for s in kept_shots))
        template["dropped_trailing_slots"] = {
            "dropped_slot_ids": dropped_slot_ids,
            "reason": data.get("end_reason") or "编排 Agent 判定叙事已收尾，之后的 slot 视为多余尾巴丢弃",
        }

    yield {"type": "step", **base, "key": "orch-head", "state": "done",
           "title": "最终脚本编排完成",
           "thought": (f"用素材 {use_count} 镜，需生成/补拍 {len(kept_shots) - use_count} 镜；"
                       + (f"编排 Agent 丢弃了尾部 {len(dropped_slot_ids)} 个多余 slot（{', '.join(dropped_slot_ids)}）" if dropped_slot_ids else "保留全部参考镜头")),
           "observation": (data.get("end_reason") or "") + ("\n" + data.get("overall_note", "") if data.get("overall_note") else "")}
    yield {"__orchestration_result__": True, "decisions": decisions, "template": template}


async def orchestrate_structure_first(reference_template: dict, material_understanding: dict, *,
                                      scheme_name="", material_strategy="faithful",
                                      selected_dimensions=None, selected_trends=None,
                                      intent="") -> AsyncGenerator[dict, None]:
    """结构优先编排：Agent 参考结构 DNA + 用户素材池，自由编排新脚本。

    结束时 yield ``{"__orchestration_result__": True, "decisions": {shot_id: feasibility_like}, "template": {shot_slots: [...]}}``。
    """
    pool, index = _collect_material_pool(material_understanding)
    structure_dna = _extract_structure_dna(reference_template)
    skill = get_skill(SKILL_ID)
    prompt = skill.prompt_hint if skill else "参考结构 DNA，用素材池自由编排出满足 DNA 的新脚本，输出 JSON。"
    skill_name = skill.name if skill else "结构优先编排"
    base = {"lane": "orchestrate", "lane_label": "编排 Agent", "lane_skill": skill_name, "phase": "编排"}

    yield {"type": "step", **base, "key": "orch-head", "state": "running",
           "title": "结构优先编排",
           "thought": (f"参考结构 DNA（{'/'.join(structure_dna.get('narrative_structure', []))}），"
                       f"从 {len(pool)} 条素材描述里自由编排新脚本")}

    user = json.dumps({
        "structure_dna": structure_dna,
        "material_pool": [
            {"asset_id": e["short_id"], "source_video_id": e["source_video_id"],
             "summary": e["summary"], "speech_or_text": e["speech_or_text"], "time_range": e["time_range"]}
            for e in pool
        ],
        "user_choices": {
            "scheme": scheme_name,
            "strategy": material_strategy,
            "selected_dimensions": list(selected_dimensions or []),
            "selected_trends": list(selected_trends or []),
            "intent": intent or "",
        },
    }, ensure_ascii=False)

    content = ""
    try:
        async for item in as_core.stream(prompt, user):
            if item.get("reasoning"):
                ev = {"type": "reasoning", "text": item["reasoning"]}
                ev.update(base)
                yield ev
            elif "content" in item:
                content = item["content"]
        data = as_core.parse_json(content) if content.strip() else {}
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        _log.warning("orchestration failed: %s", exc)
        data = {}

    slot_specs = [x for x in data.get("slots", []) if isinstance(x, dict)]
    decisions: dict = {}
    new_slots = []
    for idx, spec in enumerate(slot_specs):
        raw_sid = spec.get("slot_id") or f"S{idx + 1:02d}"
        try:
            sid_int = int(str(raw_sid).lstrip("S").lstrip("s")) if str(raw_sid).lower().lstrip("s").isdigit() else idx + 1
        except ValueError:
            sid_int = idx + 1
        duration = float(spec.get("duration") or 0.0) or 3.0
        role = spec.get("role") or ""
        goal = spec.get("goal") or spec.get("caption") or role
        caption = spec.get("caption") or ""
        action = spec.get("action")
        # 兼容旧 schema：asset_id 单值 → 单候选；新 schema：candidates 列表
        raw_candidates = spec.get("candidates") or []
        if not raw_candidates and spec.get("asset_id"):
            raw_candidates = [{"asset_id": spec.get("asset_id"), "source_time_range": "", "reason": spec.get("reason", "")}]
        # 校验候选（asset_id 必须存在于 pool），最多保留 3 个
        valid = []
        for c in raw_candidates:
            if not isinstance(c, dict):
                continue
            aid = c.get("asset_id", "")
            if aid and aid in index and aid not in {v["short_id"] for v in valid}:
                entry = index[aid]
                valid.append({
                    "short_id": aid,
                    "asset_id": entry["asset_id"],  # 映射回真实 global_asset_id 供下游解析
                    "source_video_id": entry["source_video_id"],
                    "source_path": entry.get("source_path", ""),
                    "source_time_range": c.get("source_time_range") or entry["time_range"],
                    "summary": entry["summary"],
                    "speech_or_text": entry.get("speech_or_text", ""),
                    "reason": c.get("reason", ""),
                })
            if len(valid) >= 3:
                break
        breakdown = []
        if caption:
            breakdown.append({"dim": "字幕贴片", "value": caption})
        if role:
            breakdown.append({"dim": "叙事结构", "value": role})
        new_slots.append({
            "id": sid_int,
            "want": goal,
            "duration": duration,
            "role": role,
            "breakdown": breakdown,
        })
        if action == "use_user_asset" and valid:
            primary = valid[0]
            alternates = valid[1:]
            user_speech = primary.get("speech_or_text") or ""
            final_caption = user_speech or caption
            decisions[sid_int] = {
                "shot_id": sid_int, "status": "direct",
                "matched_asset_id": primary["asset_id"],
                "matched_source_video_id": primary["source_video_id"],
                "matched_source_path": primary.get("source_path", ""),
                "matched_time_range": primary["source_time_range"],
                "matched_summary": primary["summary"],
                "score": 1.0,
                "replicable_part": "",
                "reason": primary.get("reason") or spec.get("reason", "结构优先编排首选"),
                "caption": final_caption,
                "alternates": alternates,
            }
            if breakdown and breakdown[0].get("dim") == "字幕贴片":
                breakdown[0]["value"] = final_caption
            elif final_caption:
                breakdown.insert(0, {"dim": "字幕贴片", "value": final_caption})
        else:
            decisions[sid_int] = {
                "shot_id": sid_int, "status": "none",
                "matched_asset_id": "", "matched_summary": "",
                "reason": spec.get("reason") or "素材池无合适片段，需 AIGC 生成",
                "generation_prompt": spec.get("generation_prompt", ""),
                "caption": caption,
                "alternates": [],
            }

    total_dur = sum(s.get("duration", 0.0) for s in new_slots)
    new_template = {
        "template_id": "structure-first",
        "industry": structure_dna.get("industry", "ecom"),
        "total_duration_sec": total_dur,
        "hook": structure_dna.get("hook", {}),
        "narrative_structure": structure_dna.get("narrative_structure", []),
        "rhythm": structure_dna.get("rhythm", {}),
        "selling_points_order": structure_dna.get("selling_points_order", []),
        "cta": structure_dna.get("cta", {}),
        "packaging": structure_dna.get("packaging", {}),
        "shot_slots": new_slots,
    }
    use_count = sum(1 for d in decisions.values() if d["status"] == "direct")
    yield {"type": "step", **base, "key": "orch-head", "state": "done",
           "title": "结构优先编排完成",
           "thought": f"编排出 {len(new_slots)} 个 slot（用素材 {use_count} 镜，AIGC {len(new_slots) - use_count} 镜）",
           "observation": data.get("overall_note", "")}
    yield {"__orchestration_result__": True, "decisions": decisions, "template": new_template}
