"""visual_match — 让选片"画面贴近参考视频"：抽参考视频每个节拍的实际画面，用 VLM 和用户素材
画面做**视觉相似度**比对（景别/构图/主体/动作），选出画面最像参考镜头的用户片段。

之前的匹配是"文字语义"（beat 文字 ↔ 素材文字描述），画面未必像参考镜头。这里补上"视觉比对参考帧"。
节拍在深度理解后无时间戳，故对参考视频做场景切分、按节拍数切成有序窗口，逐窗抽代表帧对齐到各节拍。
"""
import os
import subprocess

import _common  # noqa: F401
import as_core
from _common import FFMPEG
from reference_shots import detect_scene_cuts, _video_duration
from reference_dna import _merge_bounds, _run_async


def _extract_frame(src, t, out_png):
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", "{:.3f}".format(max(0.0, t)),
           "-i", src, "-frames:v", "1", "-vf", "scale=384:-2", out_png]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_png)
    except Exception:  # noqa: BLE001
        return False


def reference_frames(ref_video, n_beats, work_dir):
    """把参考视频按节拍数切成 n_beats 个有序窗口，每窗抽一帧，返回 [frame_path, ...]（按节拍顺序）。"""
    if not (ref_video and os.path.exists(ref_video)) or n_beats <= 0:
        return []
    dur = _video_duration(ref_video)
    if not dur or dur <= 0:
        return []
    os.makedirs(work_dir, exist_ok=True)
    try:
        cuts = detect_scene_cuts(ref_video, 0.30)
    except Exception:  # noqa: BLE001
        cuts = []
    segs = _merge_bounds(cuts, dur, n_beats, 0.6) or [(0.0, dur)]
    # 段数可能与 n_beats 不完全相等；按顺序对齐，不足则用均分补
    if len(segs) < n_beats:
        step = dur / n_beats
        segs = [(i * step, (i + 1) * step) for i in range(n_beats)]
    frames = []
    for i in range(n_beats):
        s, e = segs[i] if i < len(segs) else (i * dur / n_beats, (i + 1) * dur / n_beats)
        png = os.path.join(work_dir, "ref_beat_{:02d}.png".format(i))
        frames.append(png if _extract_frame(ref_video, (s + e) / 2.0, png) else None)
    return frames


def visual_match(ref_frame, cand_frame, beat_desc):
    """VLM 比对：候选画面(图2)是否与参考镜头(图1)视觉相像。返回 (matches, reason)。失败→True 放过。"""
    if not (ref_frame and cand_frame and os.path.exists(ref_frame) and os.path.exists(cand_frame)):
        return True, "no_frame"
    system = ("你是短视频镜头审校专家。图1是参考爆款视频某节拍的真实画面，图2是要复刻该节拍的用户素材画面。"
              "判断图2是否在**镜头语言上贴近图1**，以 JSON 返回。")
    user = ("该节拍作用：「{}」。判断图2相对图1：**景别(特写/中景/全景)、构图、主体类型、动作/场景**"
            "是否大体一致(复刻同一类镜头)——商品不同没关系，只看镜头语言是否像。"
            "相像→matches=true；明显不同一类镜头(如图1是产品特写、图2是人物讲话/无关空镜)→matches=false。"
            "只返回 JSON：{{\"matches\": true/false, \"reason\": \"...\"}}").format(beat_desc)
    try:
        obj = _run_async(as_core.complete_json(
            system, user, vision=True,
            media=[{"type": "image", "url": ref_frame}, {"type": "image", "url": cand_frame}]))
        return bool(obj.get("matches", True)), obj.get("reason", "")
    except Exception as exc:  # noqa: BLE001
        print("[visual_match] VLM 比对失败(放过): {}".format(str(exc)[:120]), flush=True)
        return True, "vlm_error"
