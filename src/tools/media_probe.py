"""镜头切分与音乐节奏检测工具（业务层）。

- ``detect_shot_boundaries``: 用 ffmpeg ``select='gt(scene,THR)',showinfo`` 扫描
  视频，返回镜头切换点的 pts_time（秒）。
- ``detect_music_beats``: 用 librosa 做节拍跟踪，返回 tempo (BPM) 与 beat/
  onset 时间点。

两个函数都返回 AgentScope ``ToolChunk``，人类可读摘要写在 ``TextBlock``，
机器可读结果写在 ``metadata``；`src/agent/react_agents.py` 只做薄封装注册到
Toolkit，供 UnderstandingAgent / PlanningAgent 调用。
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import time

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

try:
    import imageio_ffmpeg  # type: ignore

    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _DEFAULT_FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv"}


def _resolve_ffmpeg(ffmpeg_bin: str = "") -> str:
    """允许调用方覆盖 ffmpeg 路径，否则用 imageio_ffmpeg 内置。"""
    candidate = ffmpeg_bin or os.getenv("FFMPEG_BIN", "") or _DEFAULT_FFMPEG
    return candidate


def _parse_scene_times(log_text: str) -> list[float]:
    return [float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", log_text)]


def detect_shot_boundaries(
    video_path: str,
    threshold: float = 0.18,
    fallback_threshold: float = 0.08,
    ffmpeg_bin: str = "",
) -> ToolChunk:
    """Detect shot-boundary timestamps in a video with ffmpeg scene score.

    Runs ``ffmpeg -vf select='gt(scene,THRESHOLD)',showinfo`` on the video and
    returns the pts_time (in seconds) of every detected shot cut. If the primary
    threshold yields zero cuts, the lower ``fallback_threshold`` is tried once
    so flat/slow videos still return something usable.

    Args:
        video_path: Absolute or workspace-relative path to the video file.
        threshold: Scene-change score threshold in [0, 1]. Higher is stricter.
        fallback_threshold: If the primary run finds no cuts and this is smaller
            than ``threshold`` and > 0, rerun once with this lower threshold.
        ffmpeg_bin: Optional override of the ffmpeg binary path.

    Returns:
        ``ToolChunk`` with a text summary in ``content`` and machine-readable
        fields (``boundaries``, ``boundary_count``, ``threshold_used``,
        ``fallback_used``, ``ffmpeg_seconds``) in ``metadata``.
    """
    if not os.path.isfile(video_path):
        return ToolChunk(
            content=[TextBlock(text=f"video_path not found: {video_path}")],
            state=ToolResultState.ERROR,
        )

    ffmpeg = _resolve_ffmpeg(ffmpeg_bin)
    if not (os.path.isfile(ffmpeg) and os.access(ffmpeg, os.X_OK)):
        return ToolChunk(
            content=[TextBlock(text=f"ffmpeg binary not executable: {ffmpeg}")],
            state=ToolResultState.ERROR,
        )

    def _run(thr: float) -> tuple[list[float], float]:
        cmd = [
            ffmpeg, "-hide_banner", "-i", video_path,
            "-vf", f"select='gt(scene,{thr})',showinfo",
            "-an", "-f", "null", "-",
        ]
        start = time.time()
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit={proc.returncode}) for cmd: "
                + " ".join(shlex.quote(x) for x in cmd)
                + "\n---STDOUT/STDERR---\n" + proc.stdout,
            )
        return _parse_scene_times(proc.stdout), elapsed

    try:
        boundaries, elapsed = _run(threshold)
        used_threshold = threshold
        fallback_used = False
        if (
            not boundaries
            and fallback_threshold > 0
            and fallback_threshold < threshold
        ):
            fb, fb_elapsed = _run(fallback_threshold)
            if fb:
                boundaries = fb
                used_threshold = fallback_threshold
                fallback_used = True
                elapsed += fb_elapsed
    except Exception as exc:  # noqa: BLE001
        return ToolChunk(
            content=[TextBlock(text=f"shot boundary detection failed: {exc}")],
            state=ToolResultState.ERROR,
        )

    summary = (
        f"Detected {len(boundaries)} shot boundary(ies) in "
        f"{os.path.basename(video_path)} "
        f"(threshold={used_threshold}, fallback_used={fallback_used}, "
        f"ffmpeg_seconds={elapsed:.2f})."
    )
    if boundaries:
        preview = ", ".join(f"{b:.3f}s" for b in boundaries[:10])
        if len(boundaries) > 10:
            preview += f", ... (+{len(boundaries) - 10} more)"
        summary += " Cuts at: " + preview

    return ToolChunk(
        content=[TextBlock(text=summary)],
        state=ToolResultState.SUCCESS,
        metadata={
            "video_path": os.path.abspath(video_path),
            "threshold_used": used_threshold,
            "fallback_used": fallback_used,
            "boundary_count": len(boundaries),
            "boundaries": boundaries,
            "ffmpeg_seconds": elapsed,
        },
    )


def _extract_audio_to_wav(src: str, ffmpeg_bin: str, sample_rate: int) -> str:
    """把任意媒体提取成 mono wav 到临时目录，调用方负责清理。"""
    fd, wav_path = tempfile.mkstemp(prefix="beat_", suffix=".wav")
    os.close(fd)
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-vn", "-ac", "1", "-ar", str(sample_rate), wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract failed (exit={proc.returncode}) for cmd: "
            + " ".join(shlex.quote(x) for x in cmd)
            + "\n---STDOUT/STDERR---\n" + proc.stdout,
        )
    return wav_path


def detect_music_beats(
    audio_path: str,
    include_onsets: bool = False,
    sample_rate: int = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
    ffmpeg_bin: str = "",
) -> ToolChunk:
    """Detect music beats (and optional onsets) of an audio or video file.

    Uses ``librosa.beat.beat_track`` on the waveform to estimate the global
    tempo in BPM and the list of beat timestamps (seconds). When
    ``include_onsets`` is true, also runs ``librosa.onset.onset_detect`` for a
    denser list of note onsets, useful for tight cut-on-the-beat editing.

    Non-audio media (mp4/mov/mkv/...) is transparently supported: the audio
    track is extracted with ffmpeg to a temporary mono wav before analysis,
    then cleaned up.

    Args:
        audio_path: Absolute or workspace-relative path to an audio/video file.
        include_onsets: If true, also return onset timestamps.
        sample_rate: Target sampling rate in Hz.
        hop_length: STFT hop length in samples.
        start_bpm: Initial tempo estimate to bias beat tracking.
        tightness: How strictly beats must follow the estimated tempo.
        ffmpeg_bin: Optional override of the ffmpeg binary path.

    Returns:
        ``ToolChunk`` with a text summary and machine-readable metadata
        (``tempo_bpm``, ``beat_count``, ``beats``, ``duration_seconds``,
        ``librosa_seconds`` plus optional ``onset_count`` / ``onsets``).
    """
    if not os.path.isfile(audio_path):
        return ToolChunk(
            content=[TextBlock(text=f"audio_path not found: {audio_path}")],
            state=ToolResultState.ERROR,
        )

    try:
        import librosa
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return ToolChunk(
            content=[TextBlock(text=f"librosa import failed: {exc}")],
            state=ToolResultState.ERROR,
        )

    ffmpeg = _resolve_ffmpeg(ffmpeg_bin)
    ext = os.path.splitext(audio_path)[1].lower()
    tmp_wav: str | None = None
    try:
        if ext in _AUDIO_EXTS:
            load_src = audio_path
        elif ext in _VIDEO_EXTS:
            if not (os.path.isfile(ffmpeg) and os.access(ffmpeg, os.X_OK)):
                return ToolChunk(
                    content=[TextBlock(
                        text=f"ffmpeg needed for video audio extraction but not executable: {ffmpeg}",
                    )],
                    state=ToolResultState.ERROR,
                )
            tmp_wav = _extract_audio_to_wav(audio_path, ffmpeg, sample_rate)
            load_src = tmp_wav
        else:
            load_src = audio_path

        start = time.time()
        y, sr = librosa.load(load_src, sr=sample_rate, mono=True)
        duration = float(len(y) / sr) if sr else 0.0

        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length,
            start_bpm=start_bpm, tightness=tightness, units="frames",
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        beats = [round(float(t), 6) for t in np.asarray(beat_times).tolist()]
        tempo_val = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0

        onsets: list[float] = []
        if include_onsets:
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=hop_length, units="frames",
            )
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sr, hop_length=hop_length,
            )
            onsets = [round(float(t), 6) for t in np.asarray(onset_times).tolist()]

        elapsed = time.time() - start
    except Exception as exc:  # noqa: BLE001
        return ToolChunk(
            content=[TextBlock(text=f"beat detection failed: {exc}")],
            state=ToolResultState.ERROR,
        )
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    summary = (
        f"Tempo {tempo_val:.2f} BPM, {len(beats)} beat(s) over "
        f"{duration:.2f}s of {os.path.basename(audio_path)} "
        f"(librosa_seconds={elapsed:.2f})."
    )
    if beats:
        preview = ", ".join(f"{b:.3f}s" for b in beats[:8])
        if len(beats) > 8:
            preview += f", ... (+{len(beats) - 8} more)"
        summary += " Beats at: " + preview
    if include_onsets:
        summary += f" | onsets={len(onsets)}"

    metadata: dict = {
        "audio_path": os.path.abspath(audio_path),
        "duration_seconds": duration,
        "sample_rate": int(sr),
        "hop_length": hop_length,
        "tempo_bpm": tempo_val,
        "beat_count": len(beats),
        "beats": beats,
        "librosa_seconds": elapsed,
    }
    if include_onsets:
        metadata["onset_count"] = len(onsets)
        metadata["onsets"] = onsets

    return ToolChunk(
        content=[TextBlock(text=summary)],
        state=ToolResultState.SUCCESS,
        metadata=metadata,
    )
