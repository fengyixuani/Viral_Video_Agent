"""Optional standalone checks used before/after calling the TTS backend."""
from __future__ import annotations

import re


def validate_request(reference: dict, text: str, *, slot_duration: float = 0.0,
                     ref_cps: float = 5.0, existing_texts: list[str] | None = None) -> list[str]:
    errors = []
    if not reference.get("source_path"):
        errors.append("reference source_path is empty")
    if not reference.get("source_time_range"):
        errors.append("reference source_time_range is empty")
    text = (text or "").strip()
    if not text:
        errors.append("text is empty")
    if slot_duration > 0:
        limit = max(8, int(slot_duration * (ref_cps or 5.0)))
        count = len(re.sub(r"[^\w\u4e00-\u9fff]+", "", text))
        if count > limit:
            errors.append(f"text is too long: {count} characters > {limit}")
    normalized = re.sub(r"[\s，。！？、,.!?~…\-—:：;；\"'“”‘’()（）]+", "", text).lower()
    for previous in existing_texts or []:
        other = re.sub(r"[\s，。！？、,.!?~…\-—:：;；\"'“”‘’()（）]+", "", previous or "").lower()
        if any(normalized[i:i + 6] in other for i in range(max(0, len(normalized) - 5))):
            errors.append("text repeats an existing six-character phrase")
            break
    return errors
