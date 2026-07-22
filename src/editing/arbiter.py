"""编辑期重叠仲裁 Agent（agent_cut）。

剪辑 Agent 放片时若某素材片段与已放入的另一个 slot 重叠（同段素材/同一句口播被用到两处），
触发本仲裁 Agent：结合各 slot 的 DNA 角色/意图，判定这段素材归谁最合理；输家 slot 需另选。
"""
from __future__ import annotations

import json

import as_core
import obs
from skills import get as get_skill

_log = obs.get_logger("agent_edit_arbiter")

_FALLBACK = (
    "你是短视频剪辑的素材分配仲裁 Agent。现在有一段用户素材被同时放进了多个镜头槽位（重复占用），"
    "这不允许。请结合每个槽位的 DNA 角色(role)与意图(want)，判定这段素材放在**哪个槽位最合理**，"
    "其余槽位必须改用别的素材。\n"
    "判定原则：素材的画面/口播语义与哪个槽位的角色意图最契合，就归哪个槽位；越靠前的关键转化节点"
    "（如痛点、产品登场）优先拿到最贴合的素材。\n"
    "严格只输出 JSON：{\"winner_slot_id\":\"S0x\",\"reason\":\"一句话理由\",\"losers\":[\"S0y\"]}"
)


def _prompt() -> str:
    sk = get_skill("agent_edit_arbiter")
    return sk.prompt_hint if (sk and sk.prompt_hint) else _FALLBACK


async def arbitrate_overlap(contested: dict, competitors: list) -> dict:
    """contested={global_asset_id,summary,speech,source_time_range}；competitors=[{slot_id,role,want},...]。

    返回 {winner_slot_id, reason, losers:[...]}；失败时默认第一个 competitor 为 winner。
    """
    comps = [c for c in (competitors or []) if isinstance(c, dict) and c.get("slot_id")]
    if len(comps) <= 1:
        return {"winner_slot_id": comps[0]["slot_id"] if comps else "", "reason": "无竞争", "losers": []}
    user = json.dumps({
        "contested_segment": {
            "summary": contested.get("summary", ""),
            "speech": contested.get("speech", ""),
            "source_time_range": contested.get("source_time_range", ""),
        },
        "competing_slots": [{"slot_id": c["slot_id"], "role": c.get("role", ""), "want": c.get("want", "")}
                            for c in comps],
    }, ensure_ascii=False)
    content = ""
    try:
        async for item in as_core.stream(_prompt(), user):
            if "content" in item:
                content = item["content"]
        data = as_core.parse_json(content) if content.strip() else {}
    except (ValueError, TypeError, RuntimeError) as exc:
        _log.warning("arbitrate_overlap failed: %s", exc)
        data = {}
    winner = data.get("winner_slot_id") or comps[0]["slot_id"]
    valid_ids = {c["slot_id"] for c in comps}
    if winner not in valid_ids:
        winner = comps[0]["slot_id"]
    losers = [c["slot_id"] for c in comps if c["slot_id"] != winner]
    return {"winner_slot_id": winner, "reason": data.get("reason", "仲裁判定"), "losers": losers}
