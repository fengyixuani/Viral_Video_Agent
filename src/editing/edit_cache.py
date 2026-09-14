"""Agent 剪辑成片缓存：让「字幕特效模仿」能对同一条成片反复重烧，不用重跑剪辑。

为什么需要：调字幕样式是个要看效果、反复改的事，但字幕文本来自 Agent 剪辑那一轮的配音 plan
（``loop._caption_items`` -> ``agent_edit_done.caption_items``），原先只存在前端的
``window._lastCaptionItems`` 里。刷一下页面就丢了，只能重跑一次 Agent 剪辑（分钟级 + 占 GPU
跑 TTS）才能再点字幕。这里把「成片 uri -> 配音文案时间轴 + 参考视频」落盘，caption_fx 缺
tts_items 时自动回填，前端也能列出可重烧的成片。

只缓存**元信息**，视频本身就是 uploads/final 下那些文件，不复制。
"""
from __future__ import annotations

import json
import os
import re
import time

import obs

_log = obs.get_logger("edit_cache")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(AGENT_ROOT, "uploads", "final", "agent_edit_cache.json")
MAX_ENTRIES = int(os.getenv("AGENT_EDIT_CACHE_MAX", "30"))
# 字幕特效模仿的产物后缀。反复点会叠成 xxx_capfx_capfx.mp4（字幕烧两层），要剥回原片。
_CAPFX_RE = re.compile(r"(?:_capfx)+(?=\.[^.]+$)")


def base_uri(uri: str) -> str:
    """剥掉 ``_capfx`` 后缀链，拿回**没烧过字幕特效**的原始成片 uri。

    前端原来把 caption_fx 的产物又写回 ``_lastFinalVideo``，再点一次就是对已烧字幕的片子
    再烧一遍（uploads/final 里真出现过 ``..._loop1_capfx_capfx.mp4``）。统一在这里归一化。
    """
    u = str(uri or "").strip()
    if not u:
        return ""
    stripped = _CAPFX_RE.sub("", u)
    return stripped if os.path.isfile(abspath(stripped)) else u


def abspath(uri: str) -> str:
    u = str(uri or "").strip()
    if not u:
        return ""
    return u if os.path.isabs(u) else os.path.join(AGENT_ROOT, u.lstrip("/"))


def _load() -> dict:
    try:
        with open(STORE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        tmp = STORE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, STORE)
    except OSError as exc:
        _log.warning("缓存落盘失败(忽略): %s", str(exc)[:160])


def put(video_uri: str, caption_items=None, reference_video: str = "", **extra) -> None:
    """记一条成片。同一条 uri 重复写就覆盖（最后一轮才是最终成片）。"""
    key = base_uri(video_uri)
    if not key:
        return
    data = _load()
    data[key] = {
        "video_uri": key,
        "caption_items": [it for it in (caption_items or []) if isinstance(it, dict)],
        "reference_video": str(reference_video or ""),
        "updated_at": int(time.time()),
        **{k: v for k, v in extra.items() if v not in (None, "")},
    }
    # 只留最近 MAX_ENTRIES 条，且丢掉视频已被删的条目
    alive = {k: v for k, v in data.items() if os.path.isfile(abspath(k))}
    if len(alive) > MAX_ENTRIES:
        for k, _v in sorted(alive.items(), key=lambda kv: kv[1].get("updated_at", 0))[
                :len(alive) - MAX_ENTRIES]:
            alive.pop(k, None)
    _write(alive)
    _log.info("缓存成片 %s（字幕 %d 条）", key, len(data[key]["caption_items"]))


def get(video_uri: str) -> dict:
    """按成片 uri 取缓存（自动归一化 _capfx 后缀）。没有返回 {}。"""
    key = base_uri(video_uri)
    return _load().get(key) or {}


def entries() -> list:
    """按更新时间倒序列出可重烧的成片（视频文件还在的）。"""
    out = []
    for k, v in _load().items():
        p = abspath(k)
        if not os.path.isfile(p):
            continue
        out.append({
            "video_uri": k,
            "name": os.path.basename(k),
            "caption_count": len(v.get("caption_items") or []),
            "reference_video": v.get("reference_video", ""),
            "updated_at": v.get("updated_at", 0),
            "size_mb": round(os.path.getsize(p) / 1048576.0, 1),
        })
    out.sort(key=lambda e: -e["updated_at"])
    return out
