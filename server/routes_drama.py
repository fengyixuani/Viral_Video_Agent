"""C 负责：AI 短剧复刻 路由（POST /api/drama_replicate）。"""
import os

from _paths import ROOT
import obs
from drama import pipeline as drama_pipeline

_log = obs.get_logger("http.drama")


def handle_drama_replicate(h):
    """AI 长带货短剧复刻：理解→脚本→三视图/故事板→分片段i2v→拼接→验证，流式回传。"""
    payload = h._payload()
    video_uri = str(payload.get("video_uri", "")).strip()
    video_path = h._resolve_local(video_uri)
    product = payload.get("product") or None
    raw_imgs = payload.get("product_images") or []
    product_images = [h._resolve_local(str(p)) for p in raw_imgs]
    product_images = [p for p in product_images if p and os.path.isfile(p)]
    max_iters = int(payload.get("max_iters", 2) or 2)

    def _servable(p):
        """把 outputs 下的本地绝对路径转成可通过 /outputs/ 访问的相对 URL。"""
        try:
            rel = os.path.relpath(os.path.realpath(p), os.path.realpath(ROOT))
            if not rel.startswith(".."):
                return "/" + rel.replace(os.sep, "/")
        except Exception:  # noqa: BLE001
            pass
        return ""

    def emit(event):
        if isinstance(event, dict) and event.get("video"):
            url = _servable(event["video"])
            if url:
                event = {**event, "video_url": url}
        h._emit(event)

    h._sse_headers()
    try:
        if not video_path or not os.path.isfile(video_path):
            emit({"type": "error", "message": f"参考视频不存在: {video_uri}"})
        else:
            for event in drama_pipeline.run_drama_replication(
                    video_path, product=product, product_images=product_images,
                    max_iters=max_iters):
                emit(event)
    except Exception as exc:  # noqa: BLE001
        _log.error("drama_replicate stream failed: %s", exc, exc_info=True)
        try:
            emit({"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
    h._sse_done()


GET = {}
POST = {
    "/api/drama_replicate": handle_drama_replicate,
}
