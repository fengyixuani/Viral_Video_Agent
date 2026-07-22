"""阶段④：seedance 2.0 逐「片段」图生视频（一个片段 = 一次 15s i2v，含台词发声）+ 拼接。"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tools import wenchain_media as media

STYLE = os.getenv(
    "DRAMA_STYLE",
    "3D动画电影风格（国漫/皮克斯质感），卡通渲染，明确非真人、非写实，夸张卡通造型",
)

SEG_CONCURRENCY = int(os.getenv("DRAMA_SEG_CONCURRENCY", "4"))


def _gen_one(seg, first_frame_url, outdir, order, regen_first_frame, emit):
    from drama.script import build_seedance_prompt

    idx = seg.get("idx", order)
    prompt = build_seedance_prompt(seg, STYLE)
    dur = float(seg.get("duration_sec", 12) or 12)
    dur = max(4.0, min(15.0, dur))
    url = None
    mode = None
    if first_frame_url:
        for attempt in range(2):
            try:
                if emit:
                    emit(f"片段 {idx} i2v {dur:.0f}s 第{attempt+1}次")
                url = media.gen_video_i2v(prompt, first_frame_url=first_frame_url, duration_sec=dur)
                mode = "i2v"
                break
            except media.MediaError as e:
                if emit:
                    emit(f"片段 {idx} i2v 失败：{e}")
                time.sleep(2)
        if url is None and regen_first_frame is not None:
            try:
                if emit:
                    emit(f"片段 {idx} 重画更卡通首帧后重试")
                ff2 = regen_first_frame(seg)
                if ff2:
                    url = media.gen_video_i2v(prompt, first_frame_url=ff2, duration_sec=dur)
                    mode = "i2v-regen"
            except media.MediaError as e:
                if emit:
                    emit(f"片段 {idx} 重试仍失败：{e}")
    if url is None:
        try:
            if emit:
                emit(f"片段 {idx} 回退 t2v")
            url = media.gen_video_t2v(prompt, duration_sec=dur)
            mode = "t2v"
        except media.MediaError as e:
            if emit:
                emit(f"片段 {idx} 彻底失败，跳过：{e}")
            return None
    local = os.path.join(outdir, f"seg_{idx:02d}.mp4")
    media.download(url, local)
    return {"idx": idx, "order": order, "url": url, "local": local, "mode": mode}


def gen_segment_clips(script, first_frames: dict, outdir: str, emit=None,
                      regen_first_frame=None, concurrency: int = None) -> list:
    """并发跑每个片段的 15s i2v。first_frames: {seg_idx: {"url": ...}}"""
    os.makedirs(outdir, exist_ok=True)
    segs = list(script.get("segments", []) or [])
    n = concurrency or SEG_CONCURRENCY
    if emit:
        emit(f"并发 seedance i2v：{len(segs)} 个片段，并发度 {n}")
    results = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {}
        for order, seg in enumerate(segs, 1):
            ff = (first_frames.get(seg.get("idx")) or {}).get("url")
            futs[pool.submit(_gen_one, seg, ff, outdir, order,
                              regen_first_frame, emit)] = order
        for fut in as_completed(futs):
            order = futs[fut]
            r = fut.result()
            if r:
                results[order] = r
                if emit:
                    emit(f"片段 {r['idx']} 完成 ({r['mode']})")
    return [results[o] for o in sorted(results)]


def concat_final(clips, dst: str, emit=None) -> str:
    locals_ = [c["local"] for c in sorted(clips, key=lambda c: c["order"]) if c.get("local")]
    if not locals_:
        raise media.MediaError("没有可拼接的片段")
    if emit:
        emit(f"拼接 {len(locals_)} 个片段为成片")
    return media.concat_videos(locals_, dst)
