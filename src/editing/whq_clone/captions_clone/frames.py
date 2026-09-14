"""参考视频抽帧: 两路(移植自 copy_zimu/v1.frames + v2.frames)。

  - extract_frames       512px JPG: 供 VLM 读字/位置/特效, 省 token。
  - extract_frames_hires 原生分辨率 PNG: 供 color_calib 像素级取色。

归一 bbox(0~1)与分辨率无关: VLM 看小图给归一 bbox, 校准在原生 PNG 上按同一归一坐标裁剪。
逐时刻 seek(-ss t -frames:v 1)拿精确时间戳, 比 -vf fps= 更好对齐字幕出现时刻。
ffmpeg 走 _common.FFMPEG(需含 libass/常规解码)。
"""
import os
import re
import subprocess

from _common import FFMPEG


def duration_seconds(video):
    """用 ffmpeg -i 读时长(秒); 读不到返回 0.0。"""
    cmd = [FFMPEG, "-hide_banner", "-i", video, "-f", "null", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stdout)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


def _times(video, fps, extra_times=None):
    dur = duration_seconds(video) or 1.0
    step = 1.0 / fps if fps and fps > 0 else 1.0
    times, t = [], 0.0
    while t < dur:
        times.append(round(t, 2))
        t += step
    for et in extra_times or []:
        et = round(float(et), 2)
        if 0.0 <= et <= dur:
            times.append(et)
    return sorted(set(times))


def extract_frames(video, out_dir, fps=1.0, prefix="ref", scale_w=512, extra_times=None):
    """按 fps 抽 512px JPG(VLM 用), 返回 [(t_sec, jpg_path)](按时间排序)。"""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for t in _times(video, fps, extra_times):
        out = os.path.join(out_dir, "{}_{:07d}ms.jpg".format(prefix, int(round(t * 1000))))
        cmd = [FFMPEG, "-v", "error", "-y", "-ss", "{:.3f}".format(t), "-i", video,
               "-frames:v", "1", "-vf", "scale={}:-2".format(scale_w), "-q:v", "3", out]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if os.path.exists(out):
            frames.append((t, out))
    frames.sort(key=lambda x: x[0])
    return frames


def extract_frames_hires(video, out_dir, fps=1.0, prefix="hires", extra_times=None):
    """原生分辨率 PNG 抽帧(不缩放, 无损), 返回 [(t_sec, png_path)]。供像素取色。"""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for t in _times(video, fps, extra_times):
        out = os.path.join(out_dir, "{}_{:07d}ms.png".format(prefix, int(round(t * 1000))))
        cmd = [FFMPEG, "-v", "error", "-y", "-ss", "{:.3f}".format(t), "-i", video,
               "-frames:v", "1", out]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if os.path.exists(out):
            frames.append((t, out))
    frames.sort(key=lambda x: x[0])
    return frames
