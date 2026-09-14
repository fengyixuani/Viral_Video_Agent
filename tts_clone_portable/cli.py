"""CLI: select a reference and synthesize one cloned voice clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constraints import validate_request
from .reference_selector import pick_voice_reference
from .synthesizer import clone


def main() -> int:
    parser = argparse.ArgumentParser(description="Portable TTS_clone reference selection and synthesis")
    parser.add_argument("--segments", required=True, help="JSON file containing a list or {segments: [...]}.")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--product-name", default="")
    parser.add_argument("--reference-video-json", default="", help="Fallback reference object JSON")
    parser.add_argument("--slot-duration", type=float, default=0.0)
    parser.add_argument("--ref-cps", type=float, default=5.0)
    parser.add_argument("--reference-output", default="", help="Where to save selected reference JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
    fallback = None
    if args.reference_video_json:
        fallback = json.loads(Path(args.reference_video_json).read_text(encoding="utf-8"))
    reference = pick_voice_reference(segments, product_name=args.product_name,
                                     reference_video=fallback)
    if not reference:
        print(json.dumps({"ok": False, "error": "no usable TTS reference"}, ensure_ascii=False))
        return 2
    errors = validate_request(reference, args.text, slot_duration=args.slot_duration,
                              ref_cps=args.ref_cps)
    if errors:
        print(json.dumps({"ok": False, "error": "; ".join(errors), "reference": reference}, ensure_ascii=False))
        return 2
    if args.reference_output:
        Path(args.reference_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reference_output).write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    result = clone(reference, args.text, args.output)
    result["reference"] = reference
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
