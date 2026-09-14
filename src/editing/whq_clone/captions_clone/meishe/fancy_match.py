"""给已排好的字幕块挑【美摄花字】与【美摄字体】—— 移植 copy_zimu/v2/meishe/target_match_meishe.py
的 LLM 部分。

与 Split 版差别: 拆块/位置/时间已由 captions_clone.target_match 完成(单条流已定), 这里只做
「给每块选一款花字 + 一款字体」这一件事 —— 避免同一件事(拆块)让 LLM 做两遍、两个后端块序
还可能不一致。LLM 走 Agent 网关。
"""
import json
import re

from pipeline_utils import ask_qianfan, loads_with_repair

from . import catalog as C

# 醒目角色套花字; 口播只套字体不套花字。
FANCY_ROLES = {"emphasis", "highlight", "label", "hook"}
_ROLE_CN = {"narration": "口播", "emphasis": "强调大字", "highlight": "句内高亮",
            "label": "标签/角标", "hook": "标题钩子"}

SYSTEM = (
    "你是抖音爆款带货视频的字幕特效设计师, 手上有一套【美摄(meishe)花字 + 字体】资产。\n"
    "下面给你一条已经排好的【单条字幕流】(每块的文字/时间/角色/位置都已定, 不要改动),\n"
    "你只需要给每一块挑资产:\n"
    "  - 醒目块(强调大字/句内高亮/标签/标题钩子): 从【花字目录】按标签与名称语义挑最贴合的一款, "
    "填该花字的【名称】(务必与目录里的名称完全一致)。\n"
    "  - 口播块: 不套花字(fancy 留空字符串)。\n"
    "  - 每一块都要从【字体目录】挑一款字体, 填字体名(与目录完全一致); 口播用清晰黑体。\n"
    "本视频是带货/好物测评类, 优先 review/unboxing/praise/impact/highlight/explanation 等语义的花字。\n"
    "对每块输出: {\"idx\": 块序号(与输入一致), \"fancy\": \"花字名称或空\", \"font\": \"字体名\"}\n"
    "只返回 JSON: {\"picks\":[{\"idx\":0,\"fancy\":\"..\",\"font\":\"..\"}]}"
)


def _fancy_catalog(fancy_list):
    return "\n".join("[{}] {} | 标签: {}".format(i, f["name"], ",".join(f["tags"]))
                     for i, f in enumerate(fancy_list))


def _font_catalog(font_list):
    return "\n".join("- {} | 风格: {} | 类: {} | 适配品类: {}".format(
        f["font_name"], f["style"], f["cls"], f["category"] or "通用") for f in font_list)


def _ask(seq, fancy_list, font_list, model=None):
    blocks = [{"idx": i, "文字": s["phrase"], "角色": _ROLE_CN.get(s["role"], s["role"]),
               "role": s["role"], "位置": s["position"]} for i, s in enumerate(seq)]
    user = ("美摄花字目录(编号/名称/标签):\n" + _fancy_catalog(fancy_list)
            + "\n\n美摄字体目录:\n" + _font_catalog(font_list)
            + "\n\n已排好的单条字幕流(JSON):\n" + json.dumps(blocks, ensure_ascii=False)
            + "\n\n请给每块挑花字(仅醒目块)与字体。")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    content, _raw = ask_qianfan(messages, model=model, max_tokens=8192, temperature=0.3)
    picks = {}
    for p in (loads_with_repair(content).get("picks") or []):
        try:
            picks[int(p.get("idx"))] = p
        except (TypeError, ValueError):
            continue
    return picks


def assign_assets(seq, model=None):
    """给 seq 每块补 font_name / fancy_name / fancy_style_id, 返回 meishe blocks 列表。

    LLM 失败时全部退化为默认字体 + 无花字(仍可渲染成普通字幕)。
    """
    fancy_list = C.load_fancy_texts()
    font_list = C.load_fonts()
    by_fancy = C.fancy_by_name(fancy_list)
    by_font = C.font_by_name(font_list)
    default = C.default_font(font_list)
    try:
        picks = _ask(seq, fancy_list, font_list, model=model)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 花字挑选 LLM 失败(退化默认字体/无花字): {}".format(
            str(exc)[:200]), flush=True)
        picks = {}

    blocks = []
    for i, s in enumerate(seq):
        p = picks.get(i) or {}
        font = by_font.get(p.get("font") or "") or by_font.get(default) or {}
        fancy_name = re.sub(r"\s+", "", p.get("fancy") or "")
        fancy = by_fancy.get(p.get("fancy") or "") or by_fancy.get(fancy_name)
        if s["role"] not in FANCY_ROLES or not fancy or not fancy.get("styleId"):
            fancy = None
        blocks.append({
            "start": s["start"], "end": s["end"], "phrase": s["phrase"],
            "role": s["role"], "position": s["position"],
            "font": font.get("font_name") or default,
            "fancy_name": fancy["name"] if fancy else "",
            "style_id": fancy["styleId"] if fancy else "",
        })
    print("[captions_clone] 美摄资产: {} 块, 其中花字块 {}".format(
        len(blocks), sum(1 for b in blocks if b["style_id"])), flush=True)
    return blocks
