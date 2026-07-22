"""跨阶段共享 ASR 结果缓存。

素材理解阶段（生成 speech_or_text 用）与剪辑连接器（生成 Split 的
``all_source_asr.json`` 用）共用一套 key（abspath + mtime + size），命名空间 ``asr``；
一份素材只转写一次，之后所有阶段直接命中缓存。
"""
from __future__ import annotations

import os

import cache

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(uri: str) -> str:
    if not uri:
        return ""
    if os.path.isabs(uri):
        return uri
    return os.path.abspath(os.path.join(PROJECT_ROOT, uri))


def _digest(uri: str) -> dict:
    path = _resolve(uri)
    try:
        stat = os.stat(path)
        return {"path": path, "mtime": int(stat.st_mtime), "size": stat.st_size, "v": 1}
    except OSError:
        return {"path": path, "v": 1}


def get(uri: str):
    """命中返回 ``{text, segments, duration_seconds}``，未命中或失败结果返回 ``None``。"""
    if not uri:
        return None
    entry = cache.get("asr", _digest(uri))
    if not entry or not isinstance(entry.get("payload"), dict):
        return None
    payload = entry["payload"]
    if payload.get("error"):
        return None
    return payload


def set(uri: str, payload: dict):
    """缓存成功的 ASR 结果（无 error 即可，**包括"识别成功但无口播"的空结果**）。

    关键：一段素材如果本来就没有人声，ASR 会返回空 segments 但**不是失败**——这种也要缓存，
    否则每次都会当成"没缓存"重新转写（这正是"同样素材不 hit 缓存/每次都慢"的根因）。
    只有真正报错（transformers 异常、GPU 不可用等）才不缓存，留待下次重试。
    """
    if not uri or not isinstance(payload, dict):
        return
    if payload.get("error") or "segments" not in payload:
        return
    try:
        cache.set("asr", _digest(uri), {
            "text": payload.get("text", ""),
            "segments": payload.get("segments", []),
            "duration_seconds": float(payload.get("duration_seconds") or 0.0),
        })
    except OSError:
        pass
