"""B 负责：编排 → 剪辑 的路由。

- GET  /api/agent_edit/tools   剪辑 + AIGC 工具 schema 展示
- GET  /api/debug/materials    素材调试页（按任务）
- POST /api/edit               旧 Split 剪辑链路（保留）
- POST /api/agent_edit         纯 Agent 剪辑（剪辑+审片重剪，含 AIGC 补镜）
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
    """纯 Agent 剪辑链路（agent_cut）：剪辑 Agent + 审片 Agent 重剪循环，流式回传。"""
    import asyncio
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
                h._emit(event)
        asyncio.run(drain())
    except Exception as exc:  # noqa: BLE001
        _log.error("agent_edit stream failed: %s", exc, exc_info=True)
        try:
            h._emit({"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
    h._sse_done()


GET = {
    "/api/agent_edit/tools": handle_agent_edit_tools,
}
POST = {
    "/api/edit": handle_edit,
    "/api/agent_edit": handle_agent_edit,
}
