"""素材可行性验证阶段：按参考视频 shot 数启动多子 Agent（≤10），
为每个镜头在【用户素材】里查找可复刻片段，判定 direct/partial/none。

每个验证 Agent 加载隐藏 skill ``material_coarse_match``，先用向量库
(search_user_materials) 做语义粗检索；是否再用视觉（VL）二次验证由 Agent
自行判断。事件带 lane，前端竖排展示，每个 Agent 一行。
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

import as_core
import obs
from skills import get as get_skill
from tools import Retriever, VLMTool

_log = obs.get_logger("feasibility")

MAX_AGENTS = int(os.getenv("FEASIBILITY_MAX_AGENTS", "10"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILL_ID = "material_coarse_match"
# 每个镜头保留的候选数（供下一步编排 Agent 参考 + Split 挑不重复片段）
TOP_CANDIDATES = int(os.getenv("FEASIBILITY_TOP_CANDIDATES", "4"))
_VLM = VLMTool()
# VL 二次核验单镜最多一轮，避免 need_vl 自循环
VL_MAX_ROUNDS = 1


def _targets_from_matches(matches, asset_ids=None) -> list:
    """把召回候选（含 meta.source_path / source_time_range）转成 VLMTool 的 targets。"""
    wanted = set(asset_ids or [])
    out = []
    for m in matches or []:
        if wanted and m.get("id") not in wanted:
            continue
        meta = m.get("meta", {}) or {}
        if not meta.get("source_path"):
            continue
        out.append({
            "asset_id": m.get("id"),
            "source_path": meta.get("source_path", ""),
            "source_time_range": meta.get("source_time_range", ""),
            "source_video_id": meta.get("source_video_id", ""),
        })
    return out


async def _run_skill(skill_prompt, user_json, base, queue) -> dict:
    """跑一次 skill LLM（流式 reasoning 转发），返回解析后的 JSON。"""
    content = ""
    try:
        async for item in as_core.stream(skill_prompt, user_json):
            if item.get("reasoning"):
                ev = {"type": "reasoning", "text": item["reasoning"]}
                ev.update(base)
                await queue.put(ev)
            elif "content" in item:
                content = item["content"]
        return as_core.parse_json(content) if content.strip() else {}
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        _log.warning("skill run failed: %s", exc)
        return {}


def _local_media(uri: str) -> str:
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for cand in (uri, os.path.join(PROJECT_ROOT, uri)):
        if os.path.isfile(cand):
            return cand
    return ""


async def _verify_shot(lane, lane_label, key, shot, skill_prompt, skill_name, queue, retriever):
    shot_id = shot.get("id")
    base = {"lane": lane, "lane_label": lane_label, "lane_skill": skill_name, "phase": "可行性验证"}

    def step(state, title, thought, observation=None):
        ev = {"type": "step", "state": state, "title": title, "thought": thought, "key": key}
        ev.update(base)
        if observation is not None:
            ev["observation"] = observation
        return ev

    await queue.put(step("running", f"镜头 {shot_id} 匹配", f"在用户素材中检索镜头「{shot.get('want', '')}」"))
    # 多 query（L1 意图 + L2 视觉）→ 各自余弦排序 → RRF 融合，判定权交给 LLM 二判
    search = retriever.retrieve_for_shot(shot, top_k=5)
    matches = search.get("matches", [])
    queries = search.get("query_list", [])
    backend = search.get("backend", "qianfan")
    top = matches[0] if matches else None
    top_meta = (top or {}).get("meta", {})
    top_score = (top or {}).get("rrf_score", 0.0)
    _log.info("shot=%s queries=%s top1=%s rrf=%.4f backend=%s",
              shot_id, queries, (top or {}).get("id", ""), top_score, backend)

    def _user_json(extra=None):
        payload = {
            "shot_id": shot_id,
            "shot_want": shot.get("want", ""),
            "shot_breakdown": shot.get("breakdown", []),
            "queries": queries,
            "candidates": matches,
            "retrieval_backend": backend,
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)

    # 第一遍：Agent 自主判定；它也可以要求「视觉核验」(need_vl + vl_prompt)
    decision = await _run_skill(skill_prompt, _user_json(), base, queue)
    used_vl = False
    # Agent 自行选择是否用 VLM tool（prompt 由它自己给），最多一轮
    if decision.get("need_vl") and matches:
        targets = _targets_from_matches(matches, decision.get("vl_asset_ids"))
        vl_prompt = str(decision.get("vl_prompt") or "").strip() or \
            f"请核对以下候选片段能否承担镜头「{shot.get('want', '')}」的功能，如实描述可见主体/动作/景别。"
        if targets:
            await queue.put(step("running", f"镜头 {shot_id} 视觉核验", vl_prompt[:60]))
            vl = await _VLM.inspect(vl_prompt, targets=targets)
            used_vl = True
            await queue.put(step("done", f"镜头 {shot_id} 视觉核验完成",
                                 (vl.get("error") or vl.get("observation", ""))[:80] or "已核验",
                                 observation=json.dumps(vl, ensure_ascii=False)[:1200]))
            # 第二遍：带上视觉观察，给出最终判定（不再请求 need_vl）
            decision2 = await _run_skill(
                skill_prompt,
                _user_json({"vl_observation": vl.get("observation", ""),
                            "vl_error": vl.get("error", ""),
                            "instruction": "已完成视觉核验，请据 vl_observation 给出最终 status，不要再返回 need_vl。"}),
                base, queue)
            if decision2.get("status") in ("direct", "partial", "none"):
                decision = decision2

    status = decision.get("status")
    if status not in ("direct", "partial", "none"):
        # LLM 失败兜底：不依赖 embedding 分数——有候选就至少 partial，无候选才 none
        status = "partial" if matches else "none"
    # 每个镜头保留多候选（含一句话描述 + source_path），供下一步编排 Agent 与 Split 继承
    candidates = []
    for m in matches[:TOP_CANDIDATES]:
        cmeta = m.get("meta", {}) or {}
        candidates.append({
            "asset_id": m.get("id", ""),
            "source_video_id": cmeta.get("source_video_id", ""),
            "source_path": cmeta.get("source_path", ""),
            "source_time_range": cmeta.get("source_time_range", ""),
            "summary": cmeta.get("one_sentence_summary", "") or cmeta.get("visual_description", ""),
            "speech_or_text": cmeta.get("speech_or_text", ""),
            "score": round(float(m.get("rrf_score", 0.0)), 4),
        })
    matched_asset_id = decision.get("matched_asset_id") or (top["id"] if top else "")
    alternates = [c for c in candidates if c["asset_id"] != matched_asset_id]
    result = {
        "shot_id": shot_id,
        "status": status,
        "matched_asset_id": matched_asset_id,
        "matched_summary": decision.get("matched_summary") or top_meta.get("one_sentence_summary", ""),
        "matched_source_video_id": top_meta.get("source_video_id", ""),
        "matched_source_path": top_meta.get("source_path", ""),
        "matched_time_range": top_meta.get("source_time_range", ""),
        "score": decision.get("score", top_score),
        "replicable_part": decision.get("replicable_part", ""),
        "reason": decision.get("reason", ""),
        "used_vl": bool(decision.get("used_vl", used_vl)),
        "candidates": candidates,
        "alternates": alternates,
    }
    label_map = {"direct": "直接复刻", "partial": "部分可复刻", "none": "需补充/AIGC 生成"}
    await queue.put(step("done", f"镜头 {shot_id}：{label_map[status]}",
                         result.get("reason") or label_map[status],
                         observation=json.dumps(result, ensure_ascii=False)))
    return result


async def verify_materials(shots, *, max_agents=None, task_id="") -> AsyncGenerator[dict, None]:
    """为每个 shot 启动一个验证 Agent（≤max_agents），并行判定可复刻性。

    ``task_id`` 指向 per-task 向量库；结束时 yield ``{"__feasibility_result__": True, "results": {...}}``。
    """
    shots = [s for s in (shots or []) if isinstance(s, dict)]
    if not shots:
        yield {"__feasibility_result__": True, "results": {}}
        return
    retriever = Retriever(task_id)
    if retriever.size() == 0:
        yield {"type": "step", "phase": "可行性验证", "key": "feasibility-head", "state": "done",
               "title": "跳过素材可行性验证", "thought": "用户素材向量库为空，所有镜头均需补充或 AIGC 生成"}
        yield {"__feasibility_result__": True,
               "results": {s.get("id"): {"shot_id": s.get("id"), "status": "none",
                                          "reason": "无用户素材"} for s in shots}}
        return

    skill = get_skill(SKILL_ID)
    skill_prompt = skill.prompt_hint if skill else "为镜头在用户素材里找可复刻片段，输出 status。"
    skill_name = skill.name if skill else SKILL_ID
    num_agents = max(1, min(max_agents or MAX_AGENTS, len(shots)))
    _log.info("feasibility start shots=%d agents=%d", len(shots), num_agents)

    yield {"type": "step", "phase": "可行性验证", "key": "feasibility-head", "state": "running",
           "lane_skill": skill_name,
           "title": "素材可行性验证",
           "thought": f"按 {len(shots)} 个镜头启动 {num_agents} 个验证 Agent，在用户素材中查找可复刻片段"}

    out_q: "asyncio.Queue" = asyncio.Queue()
    results: dict = {}
    assignments = {w: [shots[i] for i in range(w, len(shots), num_agents)] for w in range(num_agents)}

    async def worker(w):
        lane = f"verify-{w}"
        lane_label = f"验证 Agent #{w + 1}"
        for shot in assignments.get(w, []):
            res = await _verify_shot(lane, lane_label, f"{lane}-{shot.get('id')}", shot,
                                     skill_prompt, skill_name, out_q, retriever)
            results[shot.get("id")] = res
        await out_q.put({"__worker_done__": w})

    tasks = [asyncio.create_task(worker(w)) for w in range(num_agents)]
    done = 0
    while done < num_agents:
        ev = await out_q.get()
        if ev.get("__worker_done__") is not None:
            done += 1
            continue
        yield ev
    await asyncio.gather(*tasks, return_exceptions=True)
    counts = {"direct": 0, "partial": 0, "none": 0}
    for r in results.values():
        counts[r.get("status", "none")] = counts.get(r.get("status", "none"), 0) + 1
    _log.info("feasibility done: %s", counts)
    yield {"type": "step", "phase": "可行性验证", "key": "feasibility-head", "state": "done",
           "lane_skill": skill_name,
           "title": "素材可行性验证完成",
           "thought": f"直接复刻 {counts['direct']} 镜，部分可复刻 {counts['partial']} 镜，需补充/生成 {counts['none']} 镜"}
    yield {"__feasibility_result__": True, "results": results}


AUDIT_SKILL_ID = "material_arbitration"


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return None


def _overlap(x, y) -> bool:
    return bool(x and y and x[0] < y[1] and y[0] < x[1])


def _find_conflicts(results: dict) -> list:
    """找出争抢同一用户素材片段（相同 asset 或时间重叠）的 shot 分组。"""
    claims = [(sid, r) for sid, r in results.items()
              if r.get("status") in ("direct", "partial")
              and (r.get("matched_asset_id") or r.get("matched_time_range"))]
    groups = []
    used = set()
    for i, (sid, r) in enumerate(claims):
        if sid in used:
            continue
        group = [(sid, r)]
        used.add(sid)
        for sid2, r2 in claims[i + 1:]:
            if sid2 in used:
                continue
            same_asset = r.get("matched_asset_id") and r.get("matched_asset_id") == r2.get("matched_asset_id")
            same_time = (r.get("matched_source_video_id")
                         and r.get("matched_source_video_id") == r2.get("matched_source_video_id")
                         and _overlap(_parse_range(r.get("matched_time_range")), _parse_range(r2.get("matched_time_range"))))
            if same_asset or same_time:
                group.append((sid2, r2))
                used.add(sid2)
        if len(group) > 1:
            groups.append(group)
    return groups


async def arbitrate_conflicts(shots, results: dict) -> AsyncGenerator[dict, None]:
    """审核 Agent：当多个 shot 抢同一用户素材片段/时间段重叠时，裁决片段归属。

    只有检测到冲突才触发。裁决后 loser 镜头会被降级（改为需补充/AIGC 生成，
    并注明片段已分配给谁）。结束时 yield ``{"__arbitration_result__": True, ...}``。
    """
    shot_map = {s.get("id"): s for s in (shots or []) if isinstance(s, dict)}
    groups = _find_conflicts(results)
    if not groups:
        yield {"__arbitration_result__": True, "results": results, "conflicts": 0}
        return

    skill = get_skill(AUDIT_SKILL_ID)
    audit_prompt = skill.prompt_hint if skill else (
        "你是素材分配审核 Agent。多个镜头争抢同一用户素材片段，请判定该片段归属哪个镜头最合理，"
        "其余镜头改为需补充或 AIGC 生成。严格输出 JSON："
        '{"winner_shot_id":0,"reason":"","losers":[{"shot_id":0,"action":"none|partial","note":""}]}')
    skill_name = skill.name if skill else "素材分配审核"
    lane = "audit"
    base = {"lane": lane, "lane_label": "审核 Agent", "lane_skill": skill_name, "phase": "审核"}
    _log.info("arbitration triggered: %d conflict group(s)", len(groups))

    yield {"type": "step", "phase": "审核", "key": "audit-head", "state": "running",
           "lane_skill": skill_name, "lane": lane, "lane_label": "审核 Agent",
           "title": "素材分配审核", "thought": f"检测到 {len(groups)} 组镜头争抢同一素材，触发审核裁决"}

    for gi, group in enumerate(groups):
        asset_id = group[0][1].get("matched_asset_id", "")
        competitors = [{
            "shot_id": sid,
            "want": shot_map.get(sid, {}).get("want", ""),
            "score": r.get("score", 0.0),
            "reason": r.get("reason", ""),
        } for sid, r in group]
        key = f"audit-{gi}"
        yield {"type": "step", **base, "key": key, "state": "running",
               "title": f"裁决片段 {asset_id or gi + 1}",
               "thought": f"镜头 {[c['shot_id'] for c in competitors]} 争抢，判定归属"}
        user = json.dumps({"asset_id": asset_id, "competitors": competitors}, ensure_ascii=False)
        content = ""
        try:
            async for item in as_core.stream(audit_prompt, user):
                if item.get("reasoning"):
                    ev = {"type": "reasoning", "text": item["reasoning"]}
                    ev.update(base)
                    yield ev
                elif "content" in item:
                    content = item["content"]
            decision = as_core.parse_json(content) if content.strip() else {}
        except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            _log.warning("arbitration group=%s failed: %s", gi, exc)
            decision = {}

        # 审核 Agent 自行选择是否用 VLM tool 核对争抢片段（prompt 由它自己给），最多一轮
        if decision.get("need_vl"):
            vl_targets, seen_clip = [], set()
            for sid, _r in group:
                sp = results[sid].get("matched_source_path", "")
                tr = results[sid].get("matched_time_range", "")
                ckey = (sp, tr)
                if sp and ckey not in seen_clip:
                    seen_clip.add(ckey)
                    vl_targets.append({"asset_id": results[sid].get("matched_asset_id", ""),
                                       "source_path": sp, "source_time_range": tr})
            vl_prompt = str(decision.get("vl_prompt") or "").strip() or \
                f"请对比这些争抢同一片段的镜头需求，描述该片段可见内容最契合哪种镜头功能。"
            if vl_targets:
                yield {"type": "step", **base, "key": key, "state": "running",
                       "title": f"裁决片段 {asset_id or gi + 1} · 视觉核验", "thought": vl_prompt[:60]}
                vl = await _VLM.inspect(vl_prompt, targets=vl_targets)
                yield {"type": "step", **base, "key": key, "state": "running",
                       "title": f"裁决片段 {asset_id or gi + 1} · 视觉核验完成",
                       "thought": (vl.get("error") or vl.get("observation", ""))[:80] or "已核验"}
                content2 = ""
                user2 = json.dumps({"asset_id": asset_id, "competitors": competitors,
                                    "vl_observation": vl.get("observation", ""),
                                    "vl_error": vl.get("error", ""),
                                    "instruction": "已完成视觉核验，请据 vl_observation 给出最终裁决，不要再返回 need_vl。"},
                                   ensure_ascii=False)
                try:
                    async for item in as_core.stream(audit_prompt, user2):
                        if item.get("reasoning"):
                            ev = {"type": "reasoning", "text": item["reasoning"]}
                            ev.update(base)
                            yield ev
                        elif "content" in item:
                            content2 = item["content"]
                    decision2 = as_core.parse_json(content2) if content2.strip() else {}
                    if decision2.get("winner_shot_id") in shot_map:
                        decision = decision2
                except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
                    _log.warning("arbitration group=%s vl re-decide failed: %s", gi, exc)

        # 兜底：分数最高者胜出
        winner = decision.get("winner_shot_id")
        if winner not in shot_map:
            winner = max(competitors, key=lambda c: c.get("score", 0.0))["shot_id"]
        losers = {l.get("shot_id"): l for l in decision.get("losers", []) if isinstance(l, dict)}
        # 记录本组已被占用的 asset_id + 时间段（winner 保留主片段）
        taken_ranges = {}
        winner_meta = results[winner]
        taken_ranges.setdefault(winner_meta.get("matched_source_video_id", ""), []).append(
            (_parse_range(winner_meta.get("matched_time_range", "")), winner_meta.get("matched_asset_id", "")))
        for sid, r in group:
            if sid == winner:
                continue
            loser_spec = losers.get(sid, {})
            # 尝试用 loser 自己的 alternates 顶替（结构优先编排会为每个 slot 提供 2 个备选）
            alternates = results[sid].get("alternates", []) or []
            replaced = None
            for alt in alternates:
                aid = alt.get("asset_id", "")
                svid = alt.get("source_video_id", "")
                rng = _parse_range(alt.get("source_time_range") or alt.get("time_range", ""))
                # 与已占用检查
                conflict = False
                for existing in taken_ranges.get(svid, []):
                    if existing[1] == aid or _overlap(existing[0], rng):
                        conflict = True
                        break
                if not conflict:
                    replaced = alt
                    taken_ranges.setdefault(svid, []).append((rng, aid))
                    break
            if replaced:
                results[sid]["status"] = "direct"
                results[sid]["matched_asset_id"] = replaced["asset_id"]
                results[sid]["matched_source_video_id"] = replaced.get("source_video_id", "")
                results[sid]["matched_time_range"] = replaced.get("source_time_range") or replaced.get("time_range", "")
                results[sid]["matched_summary"] = replaced.get("summary", "")
                results[sid]["reason"] = f"首选与镜头 {winner} 冲突，改用备选：{replaced.get('reason', '') or replaced['asset_id']}"
                results[sid]["arbitrated"] = True
                # 从 alternates 里移除已使用的，剩余的继续保留供后续冲突使用
                results[sid]["alternates"] = [a for a in alternates if a is not replaced]
            else:
                action = loser_spec.get("action") if loser_spec.get("action") in ("none", "partial") else "none"
                note = loser_spec.get("note") or f"片段已判给镜头 {winner}，本镜备选均冲突，需 AIGC 生成或补充素材"
                results[sid]["status"] = action
                results[sid]["arbitrated"] = True
                results[sid]["reason"] = note
                if action == "none":
                    results[sid]["matched_asset_id"] = ""
                    results[sid]["matched_summary"] = ""
        yield {"type": "step", **base, "key": key, "state": "done",
               "title": f"片段判给镜头 {winner}",
               "thought": decision.get("reason") or f"片段归镜头 {winner}，其他镜头尝试备选",
               "observation": json.dumps({"winner": winner, "group": [c["shot_id"] for c in competitors]}, ensure_ascii=False)}

    yield {"type": "step", "phase": "审核", "key": "audit-head", "state": "done",
           "lane_skill": skill_name, "lane": lane, "lane_label": "审核 Agent",
           "title": "素材分配审核完成", "thought": f"完成 {len(groups)} 组裁决"}
    yield {"__arbitration_result__": True, "results": results, "conflicts": len(groups)}
