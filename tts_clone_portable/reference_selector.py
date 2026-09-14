"""Select a stable zero-shot TTS reference from a material pool."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

PUNCT_RE = re.compile(r"[\s，。！？、,.!?~…\-—:：;；\"'“”‘’()（）]+")


def normalize_text(value: Any) -> str:
    return PUNCT_RE.sub("", str(value or ""))


def _speech_text(segment: dict) -> str:
    return normalize_text(segment.get("speech_or_text", "") or
                          (segment.get("whq_speech") or {}).get("text", ""))


def _duration(segment: dict) -> float:
    try:
        if segment.get("duration") is not None:
            return float(segment["duration"])
        start, end = str(segment.get("source_time_range", "")).split("-", 1)
        return float(end) - float(start)
    except (TypeError, ValueError):
        return 0.0


def _valid_candidates(segments: list[dict], min_chars: int, min_seconds: float,
                      max_seconds: float) -> list[tuple[str, dict, str]]:
    result = []
    for segment in segments:
        text = _speech_text(segment)
        if (len(text) >= min_chars and min_seconds <= _duration(segment) <= max_seconds
                and segment.get("source_path")):
            result.append((str(segment.get("global_asset_id", "")), segment, text))
    result.sort(key=lambda item: -len(item[2]))
    return result


def _default_llm(prompt: str, payload: dict) -> dict:
    import requests
    base = os.getenv("WENCHAIN_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("WENCHAIN_BASE_URL is not configured")
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    model = os.getenv("TEXT_LLM_MODEL", os.getenv("LLM_MODEL", "ali-qwen3.7-max"))
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {os.getenv('WENCHAIN_API_KEY', os.getenv('QIANFAN_API_KEY', ''))}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": 0, "max_tokens": 256,
              "messages": [{"role": "system", "content": prompt},
                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
        timeout=int(os.getenv("LLM_TIMEOUT", "300")),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    content = str(content).strip().strip("`")
    if content.startswith("json"):
        content = content[4:].strip()
    return json.loads(content[content.find("{"):])


def _reference(item: tuple[str, dict, str]) -> dict:
    gid, segment, text = item
    return {"global_asset_id": gid, "source_path": segment.get("source_path", ""),
            "source_time_range": segment.get("source_time_range", ""), "speech": text}


def pick_voice_reference(segments: list[dict], *, product_name: str = "",
                         llm: Callable[[str, dict], dict] | None = None,
                         reference_video: dict | None = None, min_chars: int = 8,
                         min_seconds: float = 1.5, max_seconds: float = 20.0,
                         top_k: int = 8) -> dict | None:
    """Pick one reference using the production filtering and LLM policy.

    ``llm`` receives ``(system_prompt, payload)`` and returns ``{"best": N}``.
    Passing it explicitly makes this module independent of any LLM gateway.
    """
    candidates = _valid_candidates(segments, min_chars, min_seconds, max_seconds)
    if not candidates:
        return reference_video
    candidates = candidates[:top_k]
    attempted_llm = False
    selected = 1
    if len(candidates) > 1:
        attempted_llm = True
        listing = "\n".join(f"{i}. 「{text[:60]}」" for i, (_gid, _seg, text) in enumerate(candidates, 1))
        prompt = ("从带货素材 ASR 转写中挑一段作为声音克隆参考音。标准：真人正常说话讲产品、"
                  "语义通顺、转写准确；排除现场杂音、拍摄口令、乱码。全部不可靠时返回 best=0。"
                  '只输出 JSON：{"best": 序号或0, "reason": "一句话"}')
        try:
            result = (llm or _default_llm)(
                prompt, {"product_name": product_name or "（未指定）", "segments": listing})
            selected = int(result.get("best") or 0)
        except Exception:
            selected = 1
    if attempted_llm and selected == 0:
        return reference_video
    if not 1 <= selected <= len(candidates):
        selected = 1
    return _reference(candidates[selected - 1])
