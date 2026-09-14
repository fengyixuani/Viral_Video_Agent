from __future__ import annotations

import tempfile
from pathlib import Path

from reference_selector import pick_voice_reference
from constraints import validate_request


segments = [
    {"global_asset_id": "too_short", "source_path": "a.mp4", "source_time_range": "0-1", "speech_or_text": "短"},
    {"global_asset_id": "best", "source_path": "b.mp4", "source_time_range": "0-3", "speech_or_text": "这是一段正常的产品介绍口播"},
    {"global_asset_id": "other", "source_path": "c.mp4", "source_time_range": "0-4", "speech_or_text": "另一个正常产品口播片段"},
]


def fake_llm(_prompt, _payload):
    return {"best": 2, "reason": "文本完整且像正常产品口播"}


picked = pick_voice_reference(segments, llm=fake_llm)
assert picked["global_asset_id"] == "other", picked
assert validate_request(picked, "短文案") == []
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "selected.json"
    output.write_text(str(picked), encoding="utf-8")
print("selfcheck passed")
