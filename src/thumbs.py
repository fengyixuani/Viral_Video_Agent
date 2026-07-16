"""Extract per-shot preview frames from a reference video using ffmpeg."""
import hashlib
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import imageio_ffmpeg  # type: ignore
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THUMB_ROOT = os.path.join(PROJECT_ROOT, "uploads", "thumbs")
os.makedirs(THUMB_ROOT, exist_ok=True)
# ffmpeg 抽帧的默认超时（seconds）与并行度：ffmpeg 快速 seek 抽单帧本应亚秒级完成；
# 15s 是防止个别损坏 MOV / 索引异常的兜底，超时后跳过而不是拖挂整条流水线。
THUMB_TIMEOUT_S = int(os.getenv("THUMB_FFMPEG_TIMEOUT", "15"))
THUMB_WORKERS = max(1, int(os.getenv("THUMB_FFMPEG_WORKERS", "6")))


def _resolve_local_path(uri: str) -> str:
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for candidate in (uri, os.path.join(PROJECT_ROOT, uri)):
        if os.path.isfile(candidate):
            return candidate
    return ""


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [_FFMPEG, "-hide_banner", "-i", path, "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stdout)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _probe_dimensions(path: str):
    result = subprocess.run(
        [_FFMPEG, "-hide_banner", "-i", path, "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    match = re.search(r"Stream #\S+ Video:.*?(\d{2,5})x(\d{2,5})", result.stdout)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def probe_aspect_ratio(video_uri: str):
    local = _resolve_local_path(video_uri)
    if not local:
        return None
    width, height = _probe_dimensions(local)
    if width <= 0 or height <= 0:
        return None
    return {"width": width, "height": height, "ratio": round(width / height, 4)}


def probe_duration_seconds(video_uri: str) -> float:
    local = _resolve_local_path(video_uri)
    if not local:
        return 0.0
    return _probe_duration(local)


def _run_thumb_ffmpeg(source: str, timestamp: float, target: str) -> bool:
    """跑一次 ffmpeg 快速 seek 抽单帧到 ``target``。超时 / 出错返回 False。

    - ``-ss`` 放 ``-i`` 前面走 keyframe fast-seek，通常 100~500ms 完成；
    - ``-loglevel error`` 只输出错误，避免管道被冗长 metadata 堵住；
    - ``timeout=THUMB_TIMEOUT_S`` 兜底防止个别损坏/索引异常的文件把线程挂住。
    """
    cmd = [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, timestamp):.3f}", "-i", source,
        "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "3", target,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=True, timeout=THUMB_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return os.path.isfile(target) and os.path.getsize(target) > 0


def _run_parallel(tasks):
    """并行执行 ``tasks``（每个 task 是无参 callable）；ffmpeg 是 I/O 主导，线程池就够。"""
    if not tasks:
        return
    workers = min(len(tasks), THUMB_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda fn: fn(), tasks))


def extract_shot_thumbs(video_uri: str, shot_slots):
    """Attach ``thumb`` URIs (served under ``/uploads/thumbs``) to each shot.

    Uses the declared shot durations to place the extraction at the midpoint of
    each shot. Falls back to evenly-spaced samples when total duration exceeds
    the media length. ffmpeg 抽帧走线程池并行。
    """
    local = _resolve_local_path(video_uri)
    if not local or not shot_slots:
        return
    duration = _probe_duration(local)
    if duration <= 0:
        return
    declared = sum(float(s.get("duration") or 0.0) for s in shot_slots) or duration
    scale = duration / declared if declared > 0 else 1.0
    stem = hashlib.md5(local.encode("utf-8")).hexdigest()[:10]
    tasks = []
    accumulated = 0.0
    for shot in shot_slots:
        d = float(shot.get("duration") or 0.0) * scale
        timestamp = accumulated + d / 2.0
        accumulated += d
        ts = max(0.05, min(duration - 0.05, timestamp))
        name = f"{stem}_shot{int(shot.get('id') or 0):02d}.jpg"
        target = os.path.join(THUMB_ROOT, name)
        if os.path.isfile(target):
            shot["thumb"] = f"uploads/thumbs/{name}"
            continue

        def make_task(shot_ref, ts_ref, target_ref, name_ref):
            def task():
                if _run_thumb_ffmpeg(local, ts_ref, target_ref):
                    shot_ref["thumb"] = f"uploads/thumbs/{name_ref}"
            return task

        tasks.append(make_task(shot, ts, target, name))
    _run_parallel(tasks)


def extract_remake_thumbs(shot_slots, decisions):
    """给每个镜头基于可行性验证选中的用户素材片段抽一张缩略图（并行执行）。

    读取每个 shot 对应决策里的 ``matched_source_path`` + ``matched_time_range``，
    在该片段中点位置抽一帧，写到 ``uploads/thumbs/`` 并把 URI 设到
    ``shot['remake_thumb']``。ffmpeg 抽帧走 ThreadPoolExecutor 并行，
    单条超时后跳过，不再拖挂整个流程。
    """
    if not shot_slots or not decisions:
        return
    tasks = []
    for shot in shot_slots:
        sid = shot.get("id")
        decision = decisions.get(sid) or decisions.get(str(sid)) or {}
        source_path = decision.get("matched_source_path", "")
        time_range = decision.get("matched_time_range", "")
        if not source_path or not time_range:
            continue
        local = _resolve_local_path(source_path)
        if not local:
            continue
        try:
            start_s, end_s = time_range.split("-")
            start, end = float(start_s), float(end_s)
        except (ValueError, AttributeError):
            continue
        if end <= start:
            continue
        mid = max(0.05, (start + end) / 2.0)
        digest_input = f"{local}::{start:.2f}-{end:.2f}"
        stem = hashlib.md5(digest_input.encode("utf-8")).hexdigest()[:10]
        slot_tag = re.sub(r"[^0-9A-Za-z]", "", str(sid)) or "x"
        name = f"remake_{stem}_{slot_tag}.jpg"
        target = os.path.join(THUMB_ROOT, name)
        if os.path.isfile(target):
            shot["remake_thumb"] = f"uploads/thumbs/{name}"
            continue

        def make_task(shot_ref, ts_ref, local_ref, target_ref, name_ref):
            def task():
                if _run_thumb_ffmpeg(local_ref, ts_ref, target_ref):
                    shot_ref["remake_thumb"] = f"uploads/thumbs/{name_ref}"
            return task

        tasks.append(make_task(shot, mid, local, target, name))
    _run_parallel(tasks)
