"""读取美摄花字/字体资产目录快照(assets/ 随包携带) —— 移植 copy_zimu/v2/meishe/catalog.py。"""
import json
import os

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_fancy_texts():
    """可用花字列表 [{styleId, name, tags[], desc}]。

    只保留带 styleId 的条目(无 styleId 美摄渲染不了, 不进 LLM 候选)。
    """
    with open(os.path.join(ASSETS, "fancy_text_info.json"), encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for e in data:
        style_id = e.get("styleId") or ""
        if not style_id:
            continue
        out.append({
            "styleId": style_id,
            "name": e.get("display_name_zh") or e.get("display_name") or "",
            "tags": list(e.get("style_tags") or []),
            "desc": e.get("description") or "",
        })
    return out


def load_fonts():
    """字体列表 [{font_name, font_path, style, cls, category}]。"""
    with open(os.path.join(ASSETS, "font_info.json"), encoding="utf-8") as f:
        data = json.load(f)
    return [{
        "font_name": e.get("font_name", ""),
        "font_path": e.get("font_path", ""),
        "style": e.get("style", ""),
        "cls": e.get("class", ""),
        "category": e.get("ecommerce_category", ""),
    } for e in data]


def fancy_by_name(fancy_list):
    return {f["name"]: f for f in fancy_list}


def font_by_name(font_list):
    return {f["font_name"]: f for f in font_list}


def default_font(font_list, category="化妆品"):
    """默认字体: 优先给定品类, 否则第一款。"""
    for f in font_list:
        if f.get("category") == category:
            return f["font_name"]
    return font_list[0]["font_name"] if font_list else ""
