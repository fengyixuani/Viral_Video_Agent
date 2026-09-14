"""Backend-independent TTS clone adapter.

The configured backend must accept:
--prompt-wav FILE --prompt-asr FILE --text TEXT --output FILE
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import wave


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return _env("FFMPEG", "ffmpeg")


def _range(value: str) -> tuple[float, float]:
    try:
        start, end = str(value).split("-", 1)
        return max(0.0, float(start)), max(0.0, float(end))
    except (TypeError, ValueError):
        return 0.0, 0.0


def extract_prompt_wav(source_path: str, time_range: str, output: str) -> None:
    start, end = _range(time_range)
    duration = max(0.5, end - start) if end > start else 6.0
    command = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-i", source_path, "-t", f"{duration:.3f}",
               "-vn", "-ac", "1", "-ar", "16000", output]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode or not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise RuntimeError(f"prompt audio extraction failed: {result.stderr[-300:]}")


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as stream:
            return round(stream.getnframes() / float(stream.getframerate()), 2)
    except (wave.Error, OSError, ZeroDivisionError):
        return 0.0


def _backend() -> tuple[str, str]:
    python = _env("AGENT_TTS_PYTHON") or _env("TTS_PYTHON")
    script = _env("AGENT_TTS_SCRIPT") or _env("TTS_SCRIPT")
    return python, script


def _normalize_loudness(path: str) -> None:
    target = _env("AGENT_TTS_LUFS", "-16")
    if target.lower() in {"off", "no", "false"}:
        return
    ffmpeg = _ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", path,
         "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", os.devnull], capture_output=True, text=True, timeout=120)
    match = re.search(r"\{[^{}]*input_i[^{}]*\}", probe.stderr, re.S)
    if not match:
        return
    values = json.loads(match.group(0))
    filter_spec = (
        f"loudnorm=I={target}:TP=-1.5:LRA=11:"
        f"measured_I={values['input_i']}:measured_TP={values['input_tp']}:"
        f"measured_LRA={values['input_lra']}:measured_thresh={values['input_thresh']}:linear=true"
    )
    normalized = path + ".norm.wav"
    result = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                             "-i", path, "-af", filter_spec, "-ar", "44100", normalized],
                            capture_output=True, text=True, timeout=120)
    if result.returncode == 0 and os.path.isfile(normalized) and os.path.getsize(normalized):
        os.replace(normalized, path)
    elif os.path.exists(normalized):
        os.remove(normalized)


def clone(reference: dict, text: str, output: str, *, prompt_mode: str | None = None) -> dict:
    """Generate cloned speech from a selected material segment."""
    text = (text or "").strip()
    source = os.path.abspath(str(reference.get("source_path", "")))
    if not text:
        return {"ok": False, "error": "missing synthesis text"}
    if not os.path.isfile(source):
        return {"ok": False, "error": f"reference media not found: {source}"}
    python, script = _backend()
    if not python or not os.path.isfile(python) or not script or not os.path.isfile(script):
        return {"ok": False, "error": "configure AGENT_TTS_PYTHON/AGENT_TTS_SCRIPT or TTS_PYTHON/TTS_SCRIPT"}
    mode = (prompt_mode or _env("AGENT_TTS_PROMPT_MODE", "basic")).lower()
    if mode not in {"basic", "ultimate"}:
        return {"ok": False, "error": "prompt_mode must be basic or ultimate"}
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="tts_clone_")
    prompt_wav = os.path.join(temp_dir, "prompt.wav")
    prompt_asr = os.path.join(temp_dir, "prompt_asr.json")
    try:
        extract_prompt_wav(source, str(reference.get("source_time_range", "")), prompt_wav)
        with open(prompt_asr, "w", encoding="utf-8") as stream:
            json.dump({"results": [{"text": reference.get("speech", "") if mode == "ultimate" else ""}]}, stream)
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            env.pop(name, None)
        command = [python, script, "--prompt-wav", prompt_wav, "--prompt-asr", prompt_asr,
                   "--text", text, "--output", output]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=int(_env("AGENT_TTS_TIMEOUT", "600")), env=env)
        if result.returncode or not os.path.isfile(output) or os.path.getsize(output) == 0:
            return {"ok": False, "error": result.stderr[-500:] or "TTS backend produced no audio"}
        _normalize_loudness(output)
        return {"ok": True, "output": output, "duration": _wav_duration(output),
                "backend": os.path.basename(script), "prompt_mode": mode}
    finally:
        for path in (prompt_wav, prompt_asr):
            if os.path.exists(path):
                os.remove(path)
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
