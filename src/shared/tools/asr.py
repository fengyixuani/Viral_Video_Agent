"""本机离线语音识别（ASR）工具（业务层）。

参考 Tool_Test/qwen3_asr_tool.py：复用 Viral_Video_Split 的本地 Qwen3-ASR-0.6B
管线（不走千帆/网络）：ffmpeg 抽 16k 单声道 wav → 在带 CUDA 的专用解释器里子进程
运行 vendored ``run_qwen3_asr_test.py``（Qwen3-ASR-0.6B + ForcedAligner-0.6B）拿
逐字时间戳 → 按标点/静音断句成句级 segments。

GPU 保证：校验 ASR 解释器 ``torch.cuda.is_available()``，自动挑空闲显存最大的 GPU，
强制 ``QWEN3_ASR_DEVICE=cuda``，不静默退回 CPU；无 CUDA 时返回 error。

跨阶段共享缓存：``cache_get`` / ``cache_set`` 按 ``(abspath, mtime, size)`` 落到 cache
的 ``asr`` 命名空间，供理解阶段（生成 speech_or_text 用）与剪辑连接器（生成 Split 的
``all_source_asr.json`` 用）共享——只跑一次，不再重复转写。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time

import cache
import obs

_log = obs.get_logger("asr")

_SPLIT_ROOT = os.getenv("VIRAL_VIDEO_SPLIT_ROOT", "/root/chengzhiyang/Viral_Video_Split")
try:
    import imageio_ffmpeg
    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _DEFAULT_FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

ASR_PYTHON = os.getenv("ASR_PYTHON", "/root/chengzhiyang/miniconda3/envs/qwen3-asr-cu128/bin/python")
ASR_SCRIPT = os.getenv("ASR_SCRIPT", os.path.join(_SPLIT_ROOT, "common", "vendor", "Viral_Video", "run_qwen3_asr_test.py"))
ASR_MODEL = os.getenv("QWEN3_ASR_MODEL", os.path.join(_SPLIT_ROOT, "models", "Qwen3-ASR-0.6B"))
ASR_FORCED_ALIGNER = os.getenv("QWEN3_FORCED_ALIGNER", os.path.join(_SPLIT_ROOT, "models", "Qwen3-ForcedAligner-0.6B"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pick_gpu() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        best_idx, best_free = 0, -1
        for line in out.splitlines():
            idx, free = (x.strip() for x in line.split(","))
            if int(free) > best_free:
                best_free, best_idx = int(free), int(idx)
        return best_idx
    except Exception:
        return 0


def _cuda_ok(py: str) -> bool:
    try:
        r = subprocess.run(
            [py, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _sentence_segments(items, duration):
    segments, current, last_end = [], [], None
    punctuation = set("。！？!?；;")
    for item in items:
        start = max(0.0, min(float(item["start"]), duration or float(item["start"])))
        end = max(start, min(float(item["end"]), duration or float(item["end"])))
        token = item["text"]
        if current and last_end is not None and start - last_end > 0.9:
            segments.append(current)
            current = []
        current.append({"text": token, "start": start, "end": end})
        last_end = end
        if token in punctuation:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    merged = []
    for group in segments:
        text = "".join(it["text"] for it in group).strip()
        if not text:
            continue
        s = min(it["start"] for it in group)
        e = max(it["end"] for it in group)
        merged.append({"start": round(s, 2), "end": round(max(e, s + 0.2), 2), "text": text})
    return merged


def _parse_payload(payload):
    results = payload.get("results") or []
    if not results:
        return "", []
    result = results[0] or {}
    text = str(result.get("text") or "").strip()
    items = (result.get("time_stamps") or {}).get("items") or []
    normalized = []
    for item in items:
        try:
            token = str(item.get("text") or "").strip()
            start = float(item.get("start_time"))
            end = float(item.get("end_time"))
        except Exception:
            continue
        if token and end >= start:
            normalized.append({"text": token, "start": start, "end": end})
    return text, normalized


def _duration(ffmpeg, path):
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", path, "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


class ASRTool:
    name = "语音识别"

    def _resolve(self, uri: str) -> str:
        if not uri or uri.startswith(("http://", "https://", "data:")):
            return ""
        for cand in (uri, os.path.join(PROJECT_ROOT, uri)):
            if os.path.isfile(cand):
                return cand
        return ""

    def transcribe(self, media_path: str, timestamps: bool = True, gpu_id: int = -1) -> dict:
        """本机离线转写音频/视频，返回 ``{text, segments, error?}``。"""
        local = self._resolve(media_path)
        if not local:
            return {"text": "", "segments": [], "error": f"media not found: {media_path}"}
        for label, path in (("asr_python", ASR_PYTHON), ("asr_script", ASR_SCRIPT),
                            ("model", ASR_MODEL), ("forced_aligner", ASR_FORCED_ALIGNER)):
            if not os.path.exists(path):
                return {"text": "", "segments": [], "error": f"{label} missing: {path}"}
        if not _cuda_ok(ASR_PYTHON):
            return {"text": "", "segments": [], "error": "CUDA not available in ASR interpreter; refusing CPU fallback"}
        chosen_gpu = gpu_id if gpu_id >= 0 else _pick_gpu()

        tmp_wav = asr_json = None
        try:
            duration = _duration(_DEFAULT_FFMPEG, local)
            fd, tmp_wav = tempfile.mkstemp(prefix="qasr_", suffix=".wav")
            os.close(fd)
            ex = subprocess.run(
                [_DEFAULT_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", local, "-vn", "-ac", "1", "-ar", "16000", tmp_wav],
                capture_output=True, text=True,
            )
            if ex.returncode != 0:
                return {"text": "", "segments": [], "error": f"ffmpeg extract failed: {ex.stderr[-300:]}"}
            fd, asr_json = tempfile.mkstemp(prefix="qasr_", suffix=".json")
            os.close(fd)
            cmd = [ASR_PYTHON, ASR_SCRIPT, tmp_wav, asr_json,
                   "--model", ASR_MODEL, "--forced-aligner", ASR_FORCED_ALIGNER]
            if timestamps:
                cmd.append("--timestamps")
            env = dict(os.environ)
            # 关键：清掉会污染 ASR 专用解释器导入的变量（PYTHONPATH=src、PYTHONHOME 等），
            # 否则子进程会串到调用方(base)的 site-packages，导致 transformers 版本错乱
            # （现象：cannot import name 'GenerationMixin'）。ASR 解释器用自己的 site-packages。
            for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
                env.pop(var, None)
            env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
            env["QWEN3_ASR_DEVICE"] = "cuda"
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            wall = time.time() - start
            if proc.returncode != 0:
                return {"text": "", "segments": [],
                        "error": f"Qwen3-ASR failed (exit={proc.returncode}): {proc.stderr[-300:]}"}
            with open(asr_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            return {"text": "", "segments": [], "error": f"ASR error: {exc!r}"}
        finally:
            for p in (tmp_wav, asr_json):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        text, items = _parse_payload(payload)
        segments = _sentence_segments(items, duration) if timestamps else []
        _log.info("ASR done gpu=%s dur=%.1fs chars=%d segs=%d wall=%.1fs",
                  chosen_gpu, duration, len(text), len(segments), wall)
        return {"text": text, "segments": segments, "duration_seconds": duration,
                "gpu_id": chosen_gpu, "engine": "qwen3-asr-0.6b"}
