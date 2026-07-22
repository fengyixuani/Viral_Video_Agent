"""A 负责：理解 的路由（/api/analyze /api/route）。

编排（/api/replicate）已划归 B，见 routes_orchestration.py。
"""
import os

import obs
from understanding import UnderstandingAgent

_log = obs.get_logger("http.understanding")
_AGENT = UnderstandingAgent()


def handle_analyze(h):
    h._sse(_AGENT.analyze_stream)


def handle_route(h):
    """通用模式路由：看参考视频，决定走「长带货短剧复刻」还是「电商理解增强」。"""
    payload = h._payload()
    video_path = h._resolve_local(str(payload.get("video_uri", "")).strip())
    raw_imgs = payload.get("product_images") or []
    product_images = [h._resolve_local(str(p)) for p in raw_imgs]
    product_images = [p for p in product_images if p and os.path.isfile(p)]
    materials_count = int(payload.get("materials_count", 0) or 0)
    try:
        from drama import router as drama_router
        res = drama_router.route_reference(video_path, product_images, materials_count)
        h._json(res)
    except Exception as exc:  # noqa: BLE001
        _log.error("route failed: %s", exc, exc_info=True)
        h._json({"route": "ecom", "reason": f"路由异常，回退 ecom：{exc}", "confidence": 0.3}, 200)


GET = {}
POST = {
    "/api/analyze": handle_analyze,
    "/api/route": handle_route,
}
