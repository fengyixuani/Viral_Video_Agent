"""工程调试路由（跨阶段中间产物，@A@B@C 共用）：素材调试页各 tab 的数据源。

- GET /api/debug/materials  素材理解（按任务，来自 uploads/manifests/）
- GET /api/debug/analyze    爆款理解（来自 uploads/cache/analyze_*.json）
- GET /api/debug/edit       Agent 剪辑中间输出（来自 uploads/debug/edit/*.json）
"""
import json
import os
from urllib.parse import unquote, urlparse

from _paths import ROOT, UPLOAD_DIR
import obs

_log = obs.get_logger("http.debug")


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def handle_debug_materials(h):
    """素材理解：按「任务」展示素材（可播放视频 + 完整理解 JSON）。带 ?task_id=xxx 看指定任务。"""
    query = urlparse(h.path).query
    want_task = ""
    for kv in query.split("&"):
        if kv.startswith("task_id="):
            want_task = unquote(kv[len("task_id="):]).strip()
    manifest_dir = os.path.join(UPLOAD_DIR, "manifests")
    tasks, by_id = [], {}
    try:
        names = [n for n in os.listdir(manifest_dir) if n.endswith(".json")]
    except OSError:
        names = []
    for name in names:
        man = _load_json(os.path.join(manifest_dir, name))
        if not man:
            continue
        tid = man.get("task_id") or name[:-len(".json")]
        by_id[tid] = man
        tasks.append({"task_id": tid, "created_at": man.get("created_at") or 0,
                      "created_at_str": man.get("created_at_str") or "",
                      "material_count": man.get("material_count") or len(man.get("materials") or []),
                      "segment_count": man.get("segment_count") or 0})
    tasks.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    selected = by_id.get(want_task) if want_task else (by_id.get(tasks[0]["task_id"]) if tasks else None)
    materials = []
    for m in (selected.get("materials") if selected else []):
        src = str(m.get("source_path") or "").strip()
        video_uri = src if src.startswith("uploads/") else (f"uploads/{os.path.basename(src)}" if src else "")
        materials.append({"cache_key": m.get("cache_key") or "",
                          "source_video_id": m.get("source_video_id") or "",
                          "video_uri": video_uri,
                          "video_exists": bool(video_uri) and os.path.isfile(os.path.join(ROOT, video_uri)),
                          "segment_count": m.get("segment_count") or len((m.get("payload") or {}).get("asset_segments") or []),
                          "payload": m.get("payload") or {}})
    h._json({"tasks": tasks, "task_id": (selected.get("task_id") if selected else ""),
             "materials": materials, "count": len(materials)})


def handle_debug_analyze(h):
    """爆款理解：列出每次参考视频理解缓存（template/分镜/方案/可行性 完整 JSON）。"""
    cache_dir = os.path.join(UPLOAD_DIR, "cache")
    items = []
    try:
        names = [n for n in os.listdir(cache_dir) if n.startswith("analyze_") and n.endswith(".json")]
    except OSError:
        names = []
    for name in names:
        body = _load_json(os.path.join(cache_dir, name))
        if not body:
            continue
        payload = body.get("payload") or {}
        tpl = payload.get("template") or {}
        items.append({
            "key": body.get("key") or name[len("analyze_"):-len(".json")],
            "cached_at": body.get("cached_at") or 0,
            "cached_at_str": body.get("cached_at_str") or "",
            "industry_guess": payload.get("industry_guess", ""),
            "shot_count": len(tpl.get("shot_slots") or []),
            "scheme_count": len(payload.get("schemes") or []),
            "feasibility_count": len(payload.get("feasibility") or {}),
            "material_count": len(payload.get("material_understanding") or {}),
            "payload": payload,
        })
    items.sort(key=lambda x: x.get("cached_at") or 0, reverse=True)
    h._json({"analyses": items, "count": len(items)})


def handle_debug_edit(h):
    """Agent 剪辑中间输出：每次剪辑的逐轮 note/ops/审片评语/评分/问题（uploads/debug/edit/）。"""
    edit_dir = os.path.join(UPLOAD_DIR, "debug", "edit")
    items = []
    try:
        names = [n for n in os.listdir(edit_dir) if n.endswith(".json")]
    except OSError:
        names = []
    for name in names:
        tr = _load_json(os.path.join(edit_dir, name))
        if not tr:
            continue
        items.append({
            "task_id": tr.get("task_id") or name[:-len(".json")],
            "created_at": tr.get("created_at") or 0,
            "created_at_str": tr.get("created_at_str") or "",
            "video_uri": tr.get("video_uri", ""),
            "final_score": tr.get("final_score"),
            "verdict": tr.get("verdict", ""),
            "review_model": tr.get("review_model", ""),
            "loops": tr.get("loops") or len(tr.get("history") or []),
            "strategy_path": tr.get("strategy_path", ""),
            "history": tr.get("history") or [],
        })
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    h._json({"edits": items, "count": len(items)})


GET = {
    "/api/debug/materials": handle_debug_materials,
    "/api/debug/analyze": handle_debug_analyze,
    "/api/debug/edit": handle_debug_edit,
}
POST = {}
