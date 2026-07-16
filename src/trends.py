"""LLM-backed trending viral-factor candidates."""
import json

import as_core

SYSTEM = '''你是短视频趋势研究员。结合行业、意图和场景提示，给出8到12条可融入复刻的热门爆款因素。来源只能是爆款库沉淀、网络热点、平台趋势。所有字段值必须使用简体中文，不要输出英文或拼音。严格只输出JSON：{"trends":[{"keyword":"str","phrase":"str","source":"爆款库沉淀|网络热点|平台趋势","reason":"str"}]}'''


async def fetch_trends(industry_id, intent="", skill_hint=""):
    user = json.dumps({"industry": industry_id, "intent": intent, "skill_hint": skill_hint}, ensure_ascii=False)
    try:
        data = await as_core.complete_json(SYSTEM, user)
    except (RuntimeError, ValueError, TypeError):
        data = {"trends": []}
    result = []
    for trend in data.get("trends", []) if isinstance(data, dict) else []:
        if not isinstance(trend, dict):
            continue
        item = {
            "keyword": str(trend.get("keyword", "")).strip(),
            "phrase": str(trend.get("phrase", "")).strip(),
            "source": str(trend.get("source", "平台趋势")).strip(),
            "reason": str(trend.get("reason", "")).strip(),
        }
        if item["keyword"]:
            result.append(item)
    return result
