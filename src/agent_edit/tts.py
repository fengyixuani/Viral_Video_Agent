"""TTS 声音克隆工具（agent_cut）：CosyVoice3 zero-shot 声音克隆。

用某个用户素材片段作**参考音频（音色来源）**，把改写后的文案念出来，产出 wav；
剪辑器可用它替换该镜原声，实现「改写文案 + 克隆原口播音色配音」。

底层复用 Split 的 CosyVoice3 脚本（generation/run_cosyvoice3_zero_shot.py），
用它自己的 conda env（cosyvoice_env）子进程调用；路径从 Split 的 .env 读取。
契约：--prompt-wav 参考音频 + --prompt-asr（参考音频转写 JSON）+ --text 目标文案 → --output wav。
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import wave

import obs

_log = obs.get_logger("agent_edit_tts")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

SPLIT_ROOT = os.getenv("VIRAL_VIDEO_SPLIT_ROOT", "/root/chengzhiyang/Viral_Video_Split")
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TTS_TIMEOUT = int(os.getenv("AGENT_TTS_TIMEOUT", "600"))


def _split_env() -> dict:
    env = {}
    path = os.path.join(SPLIT_ROOT, ".env")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as stream:
            for raw in stream:
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _split_env().get(key) or default


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(AGENT_ROOT, path))


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _extract_prompt_wav(src: str, time_range: str, out_wav: str) -> bool:
    """从参考素材片段抽 16k 单声道 wav 作 CosyVoice 的 prompt 音频。"""
    start, end = _parse_range(time_range)
    dur = max(0.5, end - start) if end > start else 6.0
    cmd = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, start):.3f}", "-i", src, "-t", f"{dur:.3f}",
           "-vn", "-ac", "1", "-ar", "16000", out_wav]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        return r.returncode == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0
    except (subprocess.SubprocessError, OSError) as exc:
        _log.warning("prompt wav extract failed: %s", exc)
        return False


def _wav_duration(path: str) -> float:
    """读 wav 时长；先用 stdlib wave（PCM），失败再手解析 RIFF 头（兼容 float wav）。"""
    try:
        with wave.open(path, "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            if rate:
                return round(frames / float(rate), 2)
    except (wave.Error, OSError):
        pass
    # 手解析 RIFF：找 fmt（采样率/声道/位深）与 data（字节数）
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return 0.0
        pos, rate, ch, bits, data_bytes = 12, 0, 1, 16, 0
        while pos + 8 <= len(data):
            cid = data[pos:pos + 4]
            size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            body = data[pos + 8:pos + 8 + size]
            if cid == b"fmt " and len(body) >= 16:
                ch = struct.unpack("<H", body[2:4])[0] or 1
                rate = struct.unpack("<I", body[4:8])[0]
                bits = struct.unpack("<H", body[14:16])[0] or 16
            elif cid == b"data":
                data_bytes = size
            pos += 8 + size + (size & 1)
        if rate and ch and bits:
            return round(data_bytes / float(rate * ch * (bits // 8)), 2)
    except (OSError, struct.error):
        pass
    return 0.0


def available() -> bool:
    """CosyVoice 脚本/解释器/模型是否就绪。"""
    return all(os.path.exists(_cfg(k)) for k in ("TTS_PYTHON", "TTS_SCRIPT", "TTS_MODEL_DIR") if _cfg(k)) \
        and bool(_cfg("TTS_PYTHON")) and bool(_cfg("TTS_SCRIPT"))


def clone(ref_source_path: str, ref_time_range: str, ref_speech: str, text: str, out_wav: str) -> dict:
    """用参考片段音色把 text 念出来，产出 wav。返回 {ok, output, duration, error}。"""
    text = (text or "").strip()
    ref_speech = (ref_speech or "").strip()
    if not text:
        return {"ok": False, "error": "缺少要配音的文案 text"}
    if not ref_speech:
        return {"ok": False, "error": "参考片段没有口播文本(speech)，无法做 zero-shot 克隆"}
    src = _abspath(ref_source_path)
    if not src or not os.path.isfile(src):
        return {"ok": False, "error": f"参考素材不存在：{ref_source_path}"}
    tts_python, tts_script = _cfg("TTS_PYTHON"), _cfg("TTS_SCRIPT")
    if not (tts_python and os.path.exists(tts_python) and tts_script and os.path.exists(tts_script)):
        return {"ok": False, "error": "CosyVoice 未配置（TTS_PYTHON/TTS_SCRIPT 缺失）"}

    tmpdir = tempfile.mkdtemp(prefix="agenttts_")
    prompt_wav = os.path.join(tmpdir, "prompt.wav")
    prompt_asr = os.path.join(tmpdir, "prompt_asr.json")
    try:
        if not _extract_prompt_wav(src, ref_time_range, prompt_wav):
            return {"ok": False, "error": "参考音频抽取失败"}
        with open(prompt_asr, "w", encoding="utf-8") as fh:
            json.dump({"results": [{"text": ref_speech}]}, fh, ensure_ascii=False)
        os.makedirs(os.path.dirname(_abspath(out_wav)) or ".", exist_ok=True)
        env = dict(os.environ)
        for k in ("TTS_REPO", "TTS_MODEL_DIR"):
            if _cfg(k):
                env[k] = _cfg(k)
        cmd = [tts_python, tts_script, "--prompt-wav", prompt_wav, "--prompt-asr", prompt_asr,
               "--text", text, "--output", _abspath(out_wav)]
        _log.info("tts clone: text=%r ref=%s", text[:40], os.path.basename(src))
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TTS_TIMEOUT, env=env)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "ignore")[-300:]
            _log.warning("cosyvoice failed: %s", err)
            return {"ok": False, "error": f"CosyVoice 失败：{err}"}
        outp = _abspath(out_wav)
        if not os.path.isfile(outp) or os.path.getsize(outp) == 0:
            return {"ok": False, "error": "CosyVoice 未产出音频"}
        return {"ok": True, "output": out_wav, "duration": _wav_duration(outp)}
    finally:
        for p in (prompt_wav, prompt_asr):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
