"""参考风格分析 + 按参考视频指纹缓存 style_profile。

参考风格分析很贵(逐秒抽帧 + 多批 VLM + 原生 PNG 像素采样), 而同一条参考视频在多次
复刻/续跑里是完全一样的输入。这里按【参考视频文件指纹 + 抽帧率 + VLM 模型】做缓存:
命中直接读 style_profile.json, 未命中才跑分析。缓存目录默认落在 whq work_dir 下,
可用 WHQ_CAPTION_PROFILE_DIR 指到一个跨任务共享的目录(推荐, 换参考视频自动失效)。

也支持外部直接指定成品 profile: WHQ_CAPTION_PROFILE=<style_profile.json> 时跳过分析。
"""
import json
import os

import pipeline_utils

from . import color_calib, inventory_md, ref_analyzer, style_profile
from .frames import duration_seconds, extract_frames_hires

_ENV_FPS = "WHQ_CAPTION_REF_FPS"


def _slug(video):
    return os.path.splitext(os.path.basename(video or "ref"))[0]


def _cache_key(ref_video, fps, model):
    return pipeline_utils.fingerprint(
        pipeline_utils.file_fingerprint([ref_video]), fps, model)


def analyze_reference(ref_video, work_dir, fps=None, model=None, cache_dir=None):
    """参考视频 -> style_profile dict(带缓存)。分析失败/无参考视频时返回回退预设。

    产物(cache_dir 下): <slug>_style_profile.json / <slug>_raw_inventory.json /
    <slug>_字幕清单.md。profile 里挂 ``_fingerprint`` 供缓存校验。
    """
    preset = os.getenv("WHQ_CAPTION_PROFILE")
    if preset and os.path.exists(preset):
        print("[captions_clone] 用外部指定参考风格: {}".format(preset), flush=True)
        with open(preset, encoding="utf-8") as f:
            return json.load(f)

    fps = float(fps or os.getenv(_ENV_FPS, "1.0"))
    model = model or ref_analyzer.default_model()
    cache_dir = cache_dir or os.getenv("WHQ_CAPTION_PROFILE_DIR") or os.path.join(
        work_dir, "ref_style")
    os.makedirs(cache_dir, exist_ok=True)
    slug = _slug(ref_video)
    profile_path = os.path.join(cache_dir, slug + "_style_profile.json")

    if not (ref_video and os.path.exists(ref_video)):
        print("[captions_clone] 无参考视频, 用回退预设风格", flush=True)
        return style_profile.fallback(ref_video, 0.0, reason="no_ref_video")

    fp = _cache_key(ref_video, fps, model)
    cached = pipeline_utils.load_stage(profile_path, fp)
    if cached:
        print("[captions_clone] 复用参考风格缓存: {}".format(profile_path), flush=True)
        return cached

    work = os.path.join(cache_dir, "_work_" + slug)
    try:
        inventory = ref_analyzer.analyze(ref_video, work, model=model, fps=fps)
        try:
            hires = extract_frames_hires(ref_video, os.path.join(work, "frames_hires"), fps=fps)
            color_calib.calibrate(inventory, hires)
        except Exception as exc:  # noqa: BLE001
            print("[captions_clone] 像素校准跳过: {}".format(str(exc)[:160]), flush=True)
        profile = style_profile.distill(inventory)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 参考风格分析失败(回退预设): {}".format(str(exc)[:200]), flush=True)
        inventory = {"video": ref_video, "duration": duration_seconds(ref_video), "fps": fps,
                     "frame_count": 0, "captions": []}
        profile = style_profile.fallback(ref_video, inventory["duration"], reason=str(exc)[:60])

    try:
        with open(os.path.join(cache_dir, slug + "_raw_inventory.json"), "w",
                  encoding="utf-8") as f:
            json.dump(inventory, f, ensure_ascii=False, indent=2)
        with open(os.path.join(cache_dir, slug + "_字幕清单.md"), "w", encoding="utf-8") as f:
            f.write(inventory_md.render_md(inventory, profile))
    except OSError as exc:
        print("[captions_clone] 参考清单落盘失败(忽略): {}".format(str(exc)[:120]), flush=True)
    pipeline_utils.save_stage(profile_path, fp, profile)
    print("[captions_clone] 参考风格: density={} styles={} 条目={}".format(
        profile.get("density"), list(profile.get("styles", {})),
        len(inventory.get("captions") or [])), flush=True)
    return profile
