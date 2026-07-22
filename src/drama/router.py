"""参考视频路由 + 商品信息推断。

- route_reference：看参考视频（辅以"素材数量"信号），判断走「长带货短剧复刻(drama)」
  还是「电商理解增强(ecom)」。
- infer_product：商品名缺失时，用视觉模型从商品图/参考视频推断商品名/卖点/外观。
"""
import asyncio
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import as_core

_ROUTE_SYSTEM = """你是短视频复刻路由器。看这条参考视频，判断它更适合哪条复刻链路，只输出 JSON：
{
  "route": "drama" | "ecom",
  "video_type": "对视频类型的一句话判断",
  "reason": "判断依据",
  "confidence": 0-1
}
判定标准：
- drama（长带货短剧复刻）：有人物、有剧情/冲突/反转/对白的情景短剧、带货短剧、漫剧，靠"故事套路"种草。
- ecom（电商理解增强）：以商品展示/测评/开箱/口播卖点/使用演示为主，没有明显剧情故事线。
"""


def route_reference(video_path: str, product_images=None, materials_count: int = 0) -> dict:
    """返回 {route, video_type, reason, confidence, signals}。"""
    signals = {
        "product_images": len(product_images or []),
        "materials_count": materials_count,
    }
    if not video_path or not os.path.isfile(video_path):
        return {"route": "ecom", "video_type": "无参考视频",
                "reason": "缺少参考视频，默认走电商理解增强", "confidence": 0.3,
                "signals": signals}
    user = ("判断这条参考视频走 drama 还是 ecom 链路，按 schema 输出 JSON。"
            f" 附加信号：用户上传了 {signals['product_images']} 张商品图、"
            f"{materials_count} 个其它素材。素材很少（如仅 1 视频 + 1 商品图）时，"
            "若视频是剧情向短剧应判 drama，若是纯商品展示应判 ecom。")
    try:
        res = asyncio.run(as_core.complete_json(
            _ROUTE_SYSTEM, user, vision=True,
            media=[{"type": "video", "url": video_path}]))
    except Exception as exc:  # noqa: BLE001
        return {"route": "ecom", "video_type": "路由判断失败",
                "reason": f"视觉路由异常，回退 ecom：{exc}", "confidence": 0.3,
                "signals": signals}
    route = res.get("route") if res.get("route") in ("drama", "ecom") else "ecom"
    res["route"] = route
    res["signals"] = signals
    return res


_PRODUCT_SYSTEM = """你是电商商品信息标注员。看给定的商品图片，输出该商品的结构化信息，只输出 JSON：
{
  "name": "商品名（含品牌/品类，尽量准确）",
  "features": ["卖点/功能，3~5 条"],
  "notes": "商品外观描述（形态/包装/配色/logo/文字），供后续生图保持一致"
}"""


def infer_product(product: dict, product_images=None, video_path: str = "") -> dict:
    """商品名/卖点/外观任一缺失时，用视觉模型补全（仅填缺失字段，已填的保留）。"""
    product = dict(product or {})
    has_name = bool((product.get("name") or "").strip())
    has_feats = bool(product.get("features"))
    has_notes = bool((product.get("notes") or "").strip())
    if has_name and has_feats and has_notes:
        return product
    media = []
    imgs = [p for p in (product_images or []) if p and os.path.isfile(p)]
    for p in imgs[:3]:
        media.append({"type": "image", "url": p})
    if not media and video_path and os.path.isfile(video_path):
        media.append({"type": "video", "url": video_path})
    if not media:
        product.setdefault("name", "该商品")
        return product
    try:
        res = asyncio.run(as_core.complete_json(
            _PRODUCT_SYSTEM, "请标注这个商品的名称、卖点和外观，按 schema 输出 JSON。",
            vision=True, media=media))
    except Exception:  # noqa: BLE001
        product.setdefault("name", "该商品")
        return product
    if res.get("name") and not has_name:
        product["name"] = res["name"]
    if res.get("features") and not has_feats:
        product["features"] = res["features"]
    if res.get("notes") and not has_notes:
        product["notes"] = res["notes"]
    product["_inferred"] = True
    return product
