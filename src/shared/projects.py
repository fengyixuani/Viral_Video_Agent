"""历史项目（工程）持久化。

一个「项目」= 一次参考视频 + 一批用户素材 + 复刻意图。用户点击历史项目即可自动载入
其素材，无需重新上传。记录存到 ``uploads/projects.json``，按 (video_uri + 素材集合)
去重、按最近更新排序。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECTS_FILE = os.path.join(PROJECT_ROOT, "uploads", "projects.json")
_LOCK = threading.Lock()


def _load() -> list:
    if not os.path.isfile(PROJECTS_FILE):
        return []
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(projects: list):
    os.makedirs(os.path.dirname(PROJECTS_FILE), exist_ok=True)
    with open(PROJECTS_FILE, "w", encoding="utf-8") as stream:
        json.dump(projects, stream, ensure_ascii=False, indent=2)


def _material_uris(materials) -> list:
    """只保留已上传的文件类素材（uploads 路径），过滤掉纯文本素材。"""
    out = []
    for m in materials or []:
        if isinstance(m, str) and (m.startswith("uploads/") or ("/" in m and "." in os.path.basename(m))):
            if m not in out:
                out.append(m)
    return out


def _derive_name(video_uri: str, intent: str, materials: list) -> str:
    intent = (intent or "").strip()
    if intent:
        return intent[:24]
    if video_uri and not video_uri.startswith(("http://", "https://")):
        return os.path.basename(video_uri)
    if video_uri:
        return "URL 参考"
    if materials:
        return f"{os.path.basename(materials[0])} 等 {len(materials)} 素材"
    return "项目 " + time.strftime("%m-%d %H:%M")


def save_project(*, video_uri: str = "", video_desc: str = "", intent: str = "",
                 skill_id: str = "", materials=None) -> str:
    """按 (video_uri + 素材集合) 去重 upsert 一个项目，返回 project id；无可存内容返回空串。"""
    mats = _material_uris(materials or [])
    video_uri = video_uri or ""
    if not video_uri and not mats:
        return ""
    pid = hashlib.sha256(("|".join([video_uri] + sorted(mats))).encode("utf-8")).hexdigest()[:16]
    now = int(time.time())
    with _LOCK:
        projects = _load()
        existing = next((p for p in projects if p.get("id") == pid), None)
        record = {
            "id": pid,
            "name": (existing.get("name") if existing else None) or _derive_name(video_uri, intent, mats),
            "video_uri": video_uri,
            "video_desc": video_desc or "",
            "intent": intent or "",
            "skill_id": skill_id or "",
            "materials": mats,
            "materials_count": len(mats),
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        projects = [p for p in projects if p.get("id") != pid]
        projects.append(record)
        projects.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
        _save(projects)
    return pid


def list_projects() -> list:
    """返回全部项目（含 materials），按最近更新排序。"""
    return _load()


def rename_project(pid: str, name: str) -> bool:
    name = (name or "").strip()
    if not pid or not name:
        return False
    with _LOCK:
        projects = _load()
        hit = False
        for p in projects:
            if p.get("id") == pid:
                p["name"] = name[:40]
                hit = True
        if hit:
            _save(projects)
        return hit
