"""复刻方案 → Split 兼容脚本导出。

按 Viral_Video_Split 编排层的输出契约生成两份可被下游消费的 JSON：

1. ``asset_guided_edit_plan.json``：DNA 重写后的 per-slot 编排源
   （viral_dna_template 顶层，含 slots + packaging）。
2. ``selected_editing_strategy.json``：绑定用户素材后的 slot 匹配 + 剪辑时间线
   （metadata / user_asset_bank / slot_matching / editing_timeline / missing_assets /
   overall_editing_strategy），Split 的 execute_tool_plan / rebuild_asr_edit /
   map_editing_to_tools 均从这里读取。

不引用 Split 代码，只按其字段命名与语义生成。
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime


def _slug(text: str) -> str:
    text = re.sub(r"\s+", "_", (text or "").strip())
    text = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fa5\-]", "", text)
    return (text or "reproject")[:60]


def _slot_id(shot_id, index: int) -> str:
    if isinstance(shot_id, str) and shot_id.startswith("S"):
        return shot_id
    return f"S{int(index) + 1:02d}"


def _time_range(start: float, dur: float) -> str:
    return f"{max(0.0, float(start)):.2f}-{max(float(start), float(start) + float(dur)):.2f}"


def _split_time_range(text: str):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _find_asset_segment(material_understanding: dict, asset_id: str, source_video_id: str):
    """在素材理解结果里回查 asset segment 与 source_path。

    先做全局精确匹配（global_asset_id 完全相等）；找不到再退回按局部 asset_id 兜底，
    且兜底必须**限定在 source_video_id 对应的那条素材内**——否则不同视频重名的局部 id
    （A1/A2…，每条素材都有）会被列表里排第一的素材截胡，导致所有镜头都塌缩到同一条素材。
    """
    if not material_understanding or not asset_id:
        return None, None
    local_id = asset_id.split("::")[-1]
    # 1) 全局精确匹配 global_asset_id（正确视频优先，不受字典顺序影响）
    for parsed in material_understanding.values():
        if not isinstance(parsed, dict):
            continue
        svid = parsed.get("source_video_id", "")
        source_path = parsed.get("source_path", "")
        for seg in parsed.get("asset_segments", []) or []:
            if not isinstance(seg, dict):
                continue
            gid = seg.get("global_asset_id") or f"{svid}::{seg.get('asset_id', '')}"
            if gid == asset_id:
                return seg, (source_path or svid)
    # 2) 兜底：按局部 asset_id 匹配，但只在指定 source_video_id 的素材内找
    if source_video_id:
        for parsed in material_understanding.values():
            if not isinstance(parsed, dict) or parsed.get("source_video_id", "") != source_video_id:
                continue
            source_path = parsed.get("source_path", "")
            svid = parsed.get("source_video_id", "")
            for seg in parsed.get("asset_segments", []) or []:
                if isinstance(seg, dict) and seg.get("asset_id", "") == local_id:
                    return seg, (source_path or svid)
    return None, None


def _scheme(schemes: list, scheme_name: str):
    for scheme in schemes or []:
        if isinstance(scheme, dict) and (scheme.get("name") == scheme_name or scheme.get("id") == scheme_name):
            return scheme
    return schemes[0] if schemes else {}


def build_asset_guided_edit_plan(*, project_name: str, template: dict, scheme: dict,
                                 selected_dimensions: list, selected_trends: list,
                                 material_strategy: str, feasibility: dict,
                                 material_understanding: dict, target_product_name: str = "") -> dict:
    """产出 Viral_Video_Split 兼容的 asset_guided_edit_plan 结构。"""
    shots = template.get("shot_slots", []) or []
    slots = []
    accumulated = 0.0
    for index, shot in enumerate(shots):
        sid = _slot_id(shot.get("id"), index)
        duration = float(shot.get("duration") or 0.0)
        time_range = _time_range(accumulated, duration)
        accumulated += duration
        feas = (feasibility or {}).get(shot.get("id")) or (feasibility or {}).get(sid) or {}
        breakdown = shot.get("breakdown", []) or []
        keywords = [d.get("value", "") for d in breakdown if isinstance(d, dict) and d.get("value")]
        packaging = []
        for dim in breakdown:
            if not isinstance(dim, dict):
                continue
            name = (dim.get("dim") or "").strip()
            value = dim.get("value") or ""
            if not name or not value:
                continue
            if any(tag in name for tag in ("字幕", "贴片", "标题", "口播", "CTA")):
                packaging.append({
                    "packaging_type": "subtitle_bar" if "字幕" in name else "headline_text",
                    "visual_style": name,
                    "text_or_visual": value,
                    "bbox_normalized": [0.05, 0.72, 0.95, 0.92],
                    "rotation_deg": 0.0,
                    "time_range_relative": _time_range(0.0, duration),
                    "reference_beat": name,
                })
        slots.append({
            "slot_id": sid,
            "time_range": time_range,
            "target_duration": round(duration, 2),
            "role": shot.get("role", ""),
            "goal": shot.get("want", ""),
            "required_user_asset": feas.get("matched_summary") or shot.get("want", ""),
            "asset_search_query": shot.get("want", ""),
            "must_have_visual_evidence": keywords[:5],
            "keywords": keywords,
            "speech_or_caption": next((d.get("value", "") for d in breakdown
                                        if isinstance(d, dict) and ("字幕" in (d.get("dim") or "") or "口播" in (d.get("dim") or ""))), ""),
            "source_reference_description": shot.get("want", ""),
            "needs_generated_asset": feas.get("status") == "none",
            "generation_prompt": (feas.get("reason") or shot.get("want", "")) if feas.get("status") == "none" else "",
            "packaging": packaging,
        })
    return {
        "viral_dna_template": {
            "reference_video": project_name or "reference",
            "duration_estimate": round(accumulated, 2),
            "asset_guided_plan": True,
            "target_product_name": target_product_name or (scheme.get("name") if scheme else ""),
            "product_desc": scheme.get("desc", "") if isinstance(scheme, dict) else "",
            "creative_brief": {
                "scheme_name": scheme.get("name", "") if isinstance(scheme, dict) else "",
                "strategy": material_strategy,
                "selected_dimensions": list(selected_dimensions or []),
                "selected_trends": list(selected_trends or []),
            },
            "slots": slots,
            "source_dna_path": "",
            "user_asset_index": "",
            "raw_response": "",
        }
    }


def _tool_plan_hint(source_start: float, source_end: float, target_duration: float,
                    caption: str, is_generated: bool) -> list:
    dur = max(0.05, source_end - source_start)
    hint = []
    if not is_generated:
        hint.append({"order": 1, "operation": "trim",
                     "params": {"start": round(source_start, 2), "end": round(source_end, 2), "keep_audio": True},
                     "reason": "截取素材对应片段"})
        if abs(dur - target_duration) > 0.15 and target_duration > 0:
            speed = round(dur / max(0.1, target_duration), 3)
            hint.append({"order": 2, "operation": "speed_count_cut",
                         "params": {"speed": speed, "target_duration": round(target_duration, 2),
                                     "speed_reason": "对齐目标 slot 时长"},
                         "reason": "对齐 slot 时长"})
    if caption:
        hint.append({"order": len(hint) + 1, "operation": "title_card_overlay",
                     "params": {"title_text": caption, "font_size": 56, "alignment": 2,
                                 "margin_v": 180, "persist": True,
                                 "font_color_ass": "&H00FFFFFF", "border_color_ass": "&H00000000"},
                     "reason": "叠加核心字幕"})
    hint.append({"order": len(hint) + 1, "operation": "transition_out",
                 "params": {"transition": "hard_cut", "duration": 0.2},
                 "reason": "衔接下一 slot"})
    return hint


def build_selected_editing_strategy(*, project_name: str, template: dict, scheme: dict,
                                    selected_dimensions: list, selected_trends: list,
                                    material_strategy: str, feasibility: dict,
                                    material_understanding: dict, dna_path: str = "") -> dict:
    shots = template.get("shot_slots", []) or []
    user_asset_bank = []
    slot_matching = []
    editing_timeline = []
    missing_assets = []
    accumulated_target = 0.0
    for index, shot in enumerate(shots):
        sid = _slot_id(shot.get("id"), index)
        duration = float(shot.get("duration") or 0.0)
        target_range = _time_range(accumulated_target, duration)
        accumulated_target += duration
        breakdown = shot.get("breakdown", []) or []
        caption = next((d.get("value", "") for d in breakdown
                        if isinstance(d, dict) and ("字幕" in (d.get("dim") or "") or "口播" in (d.get("dim") or ""))), "")
        feas = (feasibility or {}).get(shot.get("id")) or (feasibility or {}).get(sid) or {}
        status = feas.get("status", "none")

        if status in ("direct", "partial") and feas.get("matched_asset_id"):
            asset_gid = feas.get("matched_asset_id")
            svid = feas.get("matched_source_video_id") or asset_gid.split("::")[0]
            src_start, src_end = _split_time_range(feas.get("matched_time_range", ""))
            segment, found_path = _find_asset_segment(material_understanding, asset_gid, svid)
            # source_path 以决策里的 matched_source_path 为准（编排/可行性已给出正确路径），
            # 回查结果仅用于补充 segment 元数据；两者都缺时才为空。
            source_path = feas.get("matched_source_path") or found_path or ""
            visual_description = (segment or {}).get("visual_description", "") if segment else feas.get("matched_summary", "")
            speech = (segment or {}).get("speech_or_text", "") if segment else ""
            keywords = (segment or {}).get("keywords", []) if segment else []
            source_time_range = f"{src_start:.2f}-{src_end:.2f}" if src_end > src_start else feas.get("matched_time_range", "0.00-0.00")
            asset_id = f"asset_{sid}_{asset_gid}"
            concrete_plan = {
                "slot_id": sid,
                "candidate_id": asset_gid.split("::")[-1] if "::" in asset_gid else asset_gid,
                "source_video_id": svid,
                "source_path": source_path or "",
                "source_time_range": source_time_range,
                "target_duration": round(duration, 2),
                "fit_score": round(float(feas.get("score", 0.0)), 3),
                "editing_intent": feas.get("reason") or shot.get("want", ""),
                "tool_plan_hint": _tool_plan_hint(src_start, src_end if src_end > src_start else src_start + duration,
                                                  duration, caption, is_generated=False),
                "final_caption_text": caption,
                "sales_copy": {
                    "script_text": caption,
                    "estimated_tts_duration": round(max(0.6, len(caption) / 5.5), 2) if caption else 0.0,
                    "target_slot_duration": round(duration, 2),
                    "duration_overrun": 0.0,
                    "duration_fit": True,
                    "speech_rate_chars_per_second": 5.5,
                    "rewrite_note": "",
                },
                "visual_focus": visual_description,
                "recommended_transition_to_next": "hard_cut",
                "editing_risks": [] if status == "direct" else [feas.get("reason") or "部分可复刻，需注意衔接"],
                "why_not_other_slots": [],
                "caption_suggestion": caption,
            }
            user_asset_bank.append({
                "asset_id": asset_id,
                "source_video_id": svid,
                "source_path": source_path or "",
                "source_time_range": source_time_range,
                "visual_description": visual_description,
                "speech_or_text": speech,
                "asset_type": (segment or {}).get("asset_type", ""),
                "source_asset_type": "video",
                "quality_score": round(float((segment or {}).get("quality_score", feas.get("score", 0.0))), 3),
                "suitable_roles": [shot.get("role", "")] if shot.get("role") else [],
                "strengths": keywords[:3],
                "weaknesses": [] if status == "direct" else [feas.get("replicable_part") or "部分可复刻"],
                "shot_description": {},
                "concrete_editing_plan": concrete_plan,
                "selection_decision": "use_user_asset",
                "must_have_checklist": [],
                "missing_hard_features": [],
                "sales_copy": concrete_plan["sales_copy"],
                "alternate_assets": [
                    {
                        "asset_id": alt.get("asset_id", ""),
                        "source_video_id": alt.get("source_video_id", ""),
                        "source_path": alt.get("source_path", ""),
                        "source_time_range": alt.get("source_time_range") or alt.get("time_range", ""),
                        "summary": alt.get("summary", ""),
                        "reason": alt.get("reason", ""),
                    }
                    for alt in (feas.get("alternates") or [])
                    if isinstance(alt, dict)
                ],
            })
            slot_matching.append({
                "slot_id": sid,
                "status": "matched" if status == "direct" else "partial_matched",
                "matched_asset_id": asset_id,
                "fit_score": concrete_plan["fit_score"],
                "reason": feas.get("reason", ""),
                "fallback_plan": [] if status == "direct" else ["补充镜头素材"],
                "missing_hard_features": [] if status == "direct" else [feas.get("replicable_part", "")],
            })
            editing_timeline.append({
                "slot_id": sid,
                "target_time_range": target_range,
                "source_path": source_path or "",
                "source_time_range": source_time_range,
                "caption_text": caption,
                "transition_to_next": "hard_cut",
                "action": "use_user_asset",
            })
        else:
            slot_matching.append({
                "slot_id": sid,
                "status": "missing",
                "matched_asset_id": "",
                "fit_score": 0.0,
                "reason": feas.get("reason", "用户素材无匹配"),
                "fallback_plan": ["T2V 生成" if material_strategy != "faithful" else "补拍或补素材"],
                "missing_hard_features": [shot.get("want", "")],
            })
            missing_assets.append({
                "slot_id": sid,
                "role": shot.get("role", ""),
                "goal": shot.get("want", ""),
                "action": "generate" if material_strategy != "faithful" else "reshoot",
                "generation_prompt": feas.get("reason") or shot.get("want", ""),
                "target_duration": round(duration, 2),
            })
            editing_timeline.append({
                "slot_id": sid,
                "target_time_range": target_range,
                "source_path": "",
                "source_time_range": "",
                "caption_text": caption,
                "transition_to_next": "hard_cut",
                "action": "generate" if material_strategy != "faithful" else "reshoot",
            })

    matched_count = sum(1 for s in slot_matching if s["status"] in ("matched", "partial_matched"))
    total = max(1, len(shots))
    return {
        "metadata": {
            "matching_mode": "preunderstood_user_assets",
            "template_source": "viral_dna",
            "script_path": "",
            "dna_path": dna_path,
            "slot_duration_fill": {
                "enabled": True,
                "policy": "short slots keep selected clip duration; gaps are filled by next-best candidates not owned by higher-scored slots",
            },
            "scheme": scheme.get("name", "") if isinstance(scheme, dict) else "",
            "strategy": material_strategy,
            "selected_dimensions": list(selected_dimensions or []),
            "selected_trends": list(selected_trends or []),
            "project_name": project_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "user_asset_bank": user_asset_bank,
        "slot_matching": slot_matching,
        "editing_timeline": editing_timeline,
        "missing_assets": missing_assets,
        "overall_editing_strategy": {
            "structure_fit_score": round(matched_count / total, 3),
            "material_sufficiency_score": round(matched_count / total, 3),
            "recommended_similarity_level": "high" if matched_count == total else "medium",
            "main_risks": [] if matched_count == total else ["部分镜头需生成或补拍"],
            "final_notes": scheme.get("desc", "") if isinstance(scheme, dict) else "",
        },
    }


def _shots_context(template: dict) -> list:
    """导出镜头结构（含 slot_id / role / want / breakdown），供 connector 逐镜建 query。"""
    shots = []
    for index, shot in enumerate(template.get("shot_slots", []) or []):
        if not isinstance(shot, dict):
            continue
        shots.append({
            "slot_id": _slot_id(shot.get("id"), index),
            "role": shot.get("role", ""),
            "want": shot.get("want", ""),
            "duration": float(shot.get("duration") or 0.0),
            "breakdown": shot.get("breakdown", []) or [],
        })
    return shots


def _flatten_segments(material_understanding: dict) -> list:
    """把全部用户素材理解结果拍平成候选片段池（对齐 Split 的 user asset 全集）。"""
    segments = []
    seen = set()
    for parsed in (material_understanding or {}).values():
        if not isinstance(parsed, dict):
            continue
        svid = parsed.get("source_video_id", "")
        parent_path = parsed.get("source_path", "")
        for seg in parsed.get("asset_segments", []) or []:
            if not isinstance(seg, dict):
                continue
            seg_svid = seg.get("source_video_id") or svid
            gid = seg.get("global_asset_id") or f"{seg_svid}::{seg.get('asset_id', '')}"
            if not gid or gid in seen:
                continue
            source_path = seg.get("source_path") or parent_path
            if not source_path:
                continue
            seen.add(gid)
            segments.append({
                "global_asset_id": gid,
                "asset_id": seg.get("asset_id", ""),
                "source_video_id": seg_svid,
                "source_path": source_path,
                "source_time_range": seg.get("source_time_range", ""),
                "asset_type": seg.get("asset_type", ""),
                "one_sentence_summary": seg.get("one_sentence_summary", ""),
                "visual_description": seg.get("visual_description", ""),
                "speech_or_text": seg.get("speech_or_text", ""),
                "keywords": seg.get("keywords", []) or [],
                "visual_evidence_tags": seg.get("visual_evidence_tags", []) or [],
                "quality_score": seg.get("quality_score", 0.0),
            })
    return segments


def export_scripts(*, project_name: str, template: dict, schemes: list, scheme_name: str,
                    selected_dimensions: list, selected_trends: list, material_strategy: str,
                    feasibility: dict, material_understanding: dict,
                    target_product_name: str = "", output_root: str = "") -> dict:
    """一次生成两份 Split 兼容脚本，落盘并返回 ``{dir, edit_plan_path, strategy_path, plan, strategy}``。"""
    scheme = _scheme(schemes, scheme_name)
    slug = _slug(project_name)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = output_root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", slug)
    edit_dir = os.path.join(root, "edit_plan")
    os.makedirs(edit_dir, exist_ok=True)
    edit_plan = build_asset_guided_edit_plan(
        project_name=project_name, template=template, scheme=scheme,
        selected_dimensions=selected_dimensions, selected_trends=selected_trends,
        material_strategy=material_strategy, feasibility=feasibility,
        material_understanding=material_understanding, target_product_name=target_product_name,
    )
    edit_plan_path = os.path.join(edit_dir, f"{slug}_asset_guided_edit_plan_{stamp}.json")
    with open(edit_plan_path, "w", encoding="utf-8") as stream:
        json.dump(edit_plan, stream, ensure_ascii=False, indent=2)
    strategy = build_selected_editing_strategy(
        project_name=project_name, template=template, scheme=scheme,
        selected_dimensions=selected_dimensions, selected_trends=selected_trends,
        material_strategy=material_strategy, feasibility=feasibility,
        material_understanding=material_understanding, dna_path=edit_plan_path,
    )
    strategy_path = os.path.join(root, f"{slug}_selected_editing_strategy_{stamp}.json")
    with open(strategy_path, "w", encoding="utf-8") as stream:
        json.dump(strategy, stream, ensure_ascii=False, indent=2)
    # 连接器上下文：镜头结构 + 全量用户素材片段池，供 connector 给每个 slot 建富候选池
    # （对齐 Split 的 all_scores 多候选设计，避免下游只有单候选导致成片重复）
    context = {
        "shots": _shots_context(template),
        "segments": _flatten_segments(material_understanding),
    }
    context_path = os.path.join(root, "connector_context.json")
    with open(context_path, "w", encoding="utf-8") as stream:
        json.dump(context, stream, ensure_ascii=False, indent=2)
    return {
        "dir": root,
        "edit_plan_path": edit_plan_path,
        "strategy_path": strategy_path,
        "context_path": context_path,
        "plan": edit_plan,
        "strategy": strategy,
    }
