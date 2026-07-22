"""Tiny JSON file cache for expensive LLM/vision responses."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key(namespace: str, digest_input: dict) -> str:
    raw = json.dumps(digest_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _path(namespace: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{namespace}_{key}.json")


def get(namespace: str, digest_input: dict):
    """Return cached payload dict, or ``None`` if absent."""
    path = _path(namespace, _key(namespace, digest_input))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def set(namespace: str, digest_input: dict, payload) -> str:
    """Persist ``payload`` and return the cache key."""
    key = _key(namespace, digest_input)
    path = _path(namespace, key)
    body = {
        "cached_at": time.time(),
        "cached_at_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "key": key,
        "payload": payload,
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2)
    return key


def key_for(namespace: str, digest_input: dict) -> str:
    return _key(namespace, digest_input)
