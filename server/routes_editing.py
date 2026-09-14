"""B 负责：编排 → 剪辑 的路由。

- GET  /api/agent_edit/tools   剪辑 + AIGC 工具 schema 展示
- GET  /api/debug/materials    素材调试页（按任务）
- POST /api/edit               旧 Split 剪辑链路（保留）
- POST /api/agent_edit         纯 Agent 剪辑（剪辑+审片重剪，含 AIGC 补镜）
- POST /api/caption_fx         字幕特效模仿（对成片烧参考视频同风格字幕）
"""
import json
import os
from urllib.parse import unquote, urlparse

from _paths import ROOT, UPLOAD_DIR
import obs
import connector
from editing import loop as agent_edit_loop
from editing import tools as agent_edit_tools

_log = obs.get_logger("http.editing")


def handle_agent_edit_tools(h):
    from editing import aigc as _aigc
    h._json({"tools": agent_edit_tools.tool_schemas() + _aigc.tool_schemas()})


def handle_edit(h):
    """把 Agent 编排脚本接到 Split 编排层之后，流式跑真实剪辑/TTS/字幕/BGM。"""
    payload = h._payload()
    strategy_path = str(payload.get("strategy_path", "")).strip()
    h._sse_headers()
    try:
        if not strategy_path:
            h._emit({"type": "error", "message": "缺少 strategy_path"})
        else:
            for event in connector.run_edit(
                strategy_path,
                enable_tts=bool(payload.get("enable_tts", True)),
                enable_bgm=bool(payload.get("enable_bgm", True)),
                enable_t2v=bool(payload.get("enable_t2v", False)),
                bgm_path=str(payload.get("bgm_path", "")).strip(),
            ):
                h._emit(event)
    except Exception as exc:  # noqa: BLE001
        _log.error("edit stream failed: %s", exc, exc_info=True)
        try:
            h._emit({"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
    h._sse_done()


def handle_agent_edit(h):
    """纯 Agent 剪辑链路（agent_cut）：剪辑 Agent + 审片 Agent 重剪循环，流式回传。

    成片连同「每镜实际文案 + 成片时间轴位置」落到 ``edit_cache``，这样之后调字幕样式可以直接
    对这条成片重烧（``/api/caption_fx``），不用为了拿回文案再跑一遍剪辑。
    """
    import asyncio
    from editing import edit_cache
    payload = h._payload()
    strategy_path = str(payload.get("strategy_path", "")).strip()
    max_loops = payload.get("max_loops")
    kw = dict(
        enable_bgm=bool(payload.get("enable_bgm", True)),
        review_model=str(payload.get("review_model", "qwen")).strip() or "qwen",
        enable_review=bool(payload.get("enable_review", True)),
        enable_tts=bool(payload.get("enable_tts", True)),
        pure_music=bool(payload.get("pure_music", False)),
        reuse_bgm=bool(payload.get("reuse_bgm", False)),
        reference_video=str(payload.get("reference_video", "")).strip(),
        beat_sync=bool(payload.get("beat_sync", False)),
        burn_subtitle=bool(payload.get("burn_subtitle", True)),
        missing_shot_mode=str(payload.get("missing_shot_mode", "aigc")).strip() or "aigc",
    )
    h._sse_headers()
    try:
        async def drain():
            if not strategy_path:
                h._emit({"type": "error", "message": "缺少 strategy_path"})
                return
            async for event in agent_edit_loop.agent_edit_stream(
                    strategy_path, max_loops=int(max_loops) if max_loops else None, **kw):
                if event.get("type") == "agent_edit_done" and event.get("video_uri"):
                    edit_cache.put(event["video_uri"],
                                   caption_items=event.get("caption_items"),
                                   reference_video=kw["reference_video"],
                                   burn_subtitle=kw["burn_subtitle"])
                h._emit(event)
        asyncio.run(drain())
    except Exception as exc:  # noqa: BLE001
        _log.error("agent_edit stream failed: %s", exc, exc_info=True)
        try:
            h._emit({"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
    h._sse_done()


def handle_caption_fx(h):
    """字幕特效模仿：对已有成片烧上「参考视频同风格」的字幕（captions_clone），流式回传。

    ``tts_items``（前端从上一轮 Agent 剪辑的 caption_items 带过来）是每镜的实际文案+成片时间轴
    位置。有它就用它当字幕文本，没有才退回对成片跑 ASR 取文本（ASR 会有同音错字）。

    两处归一化让「反复调字幕」可行：
      * ``video_uri`` 剥掉 ``_capfx`` 后缀，始终对**没烧过字幕的原片**重烧，不叠第二层字幕；
      * 前端没带 ``tts_items``（比如刷新过页面）时从 ``edit_cache`` 回填，不用重跑剪辑。
    """
    from editing import caption_fx, edit_cache
    payload = h._payload()
    video_uri = edit_cache.base_uri(str(payload.get("video_uri", "")).strip())
    reference_video = str(payload.get("reference_video", "")).strip()
    items = payload.get("tts_items") or []
    tts_items = [it for it in items
                 if isinstance(it, dict) and str(it.get("text") or it.get("caption_text") or "").strip()]
    if not tts_items and video_uri:
        cached = edit_cache.get(video_uri)
        tts_items = cached.get("caption_items") or []
        reference_video = reference_video or cached.get("reference_video", "")
        if tts_items:
            _log.info("caption_fx 用缓存的配音文案 %d 条（%s）", len(tts_items), video_uri)
    h._sse_headers()
    try:
        if not video_uri:
            h._emit({"type": "error", "message": "缺少 video_uri（要模仿字幕特效的成片）"})
        else:
            for event in caption_fx.run_caption_fx(video_uri, reference_video, tts_items):
                h._emit(event)
    except Exception as exc:  # noqa: BLE001
        _log.error("caption_fx stream failed: %s", exc, exc_info=True)
        try:
            h._emit({"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
    h._sse_done()


def handle_edit_cache(h):
    """可重烧字幕的成片列表（Agent 剪辑缓存）：刷新页面后也能挑一条直接重烧字幕。"""
    from editing import edit_cache
    h._json({"items": edit_cache.entries()})


GET = {
    "/api/agent_edit/tools": handle_agent_edit_tools,
    "/api/edit_cache": handle_edit_cache,
}
POST = {
    "/api/edit": handle_edit,
    "/api/agent_edit": handle_agent_edit,
    "/api/caption_fx": handle_caption_fx,
}
