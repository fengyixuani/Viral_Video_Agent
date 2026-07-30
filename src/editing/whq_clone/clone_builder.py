"""clone_builder — 按结构级规划(edit_planner)裁剪+归一化+硬剪拼接成**无声** base。

变更(对齐新需求):
  - 成片**移除素材原声**: 每段一律 -an; 拼接后补一条**静音**音轨(供后续 TTS 配音
    混流时 [0:a] 映射可用, 同时满足「移除参考/素材原始音频」)。
  - 输入是 edit_planner 的 segments(结构级、1:1 不重复分配), 而非逐镜 matches。
  - 每段时长 = 该段 target_duration(源自 DNA 节拍按比例铺满参考总时长)。
      源可用长度 L >= D: 从起点截 D 秒; L < D: 时间拉伸补到 D(--no-stretch 用 L)。
  - 统一 720x1280 / 30fps (scale+pad), 便于无缝硬剪。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import FFMPEG


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _build_segment(src, src_start, avail, target_dur, out_path,
                   width=720, height=1280, fps=30, stretch=True, source_take=None):
    """裁剪+归一化单段为**无声** mp4。

    source_take(voice_policy 句子级对窗给出的源窗口长): 显式指定从源截多长。
    与 target_dur 不等时 setpts **双向**变速回填(>target 加速/<target 放慢),
    原声音频走同倍率 atempo, 音画同步、口型不花——这是「原声段说完整句」的画面侧。
    """
    take = min(avail, target_dur) if avail > 0 else target_dur
    vf = ("scale={w}:{h}:force_original_aspect_ratio=decrease,"
          "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1").format(w=width, h=height)
    setpts_factor = None
    if source_take and source_take > 0:
        take = min(source_take, avail) if avail > 0 else source_take
        if abs(take - target_dur) > 0.02:
            setpts_factor = target_dur / take
            vf += ",setpts={:.5f}*PTS".format(setpts_factor)
    elif avail > 0 and avail < target_dur and stretch:
        setpts_factor = target_dur / avail
        vf += ",setpts={:.5f}*PTS".format(setpts_factor)
        take = avail
    cmd = [FFMPEG, "-y", "-ss", "{:.3f}".format(src_start)]
    if avail > 0 or (source_take and source_take > 0):
        cmd += ["-t", "{:.3f}".format(take)]
    cmd += ["-i", src, "-vf", vf, "-r", str(fps)]
    if setpts_factor:
        cmd += ["-t", "{:.3f}".format(target_dur)]      # 拉伸后钳到目标时长
    elif avail <= 0:
        cmd += ["-t", "{:.3f}".format(target_dur)]      # avail 未知(T2V整段)
    cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    _run(cmd)
    return out_path


def _segment_source(seg):
    """从 edit_planner segment 解析源: 返回 (src_path, start, avail) 或 None。"""
    if seg.get("is_t2v") and seg.get("t2v_path"):
        return seg["t2v_path"], 0.0, 0.0
    cand = seg.get("best_candidate")
    if not cand:
        return None
    return cand.get("source_path"), cand.get("start", 0.0), cand.get("duration", 0.0)


def build_base(segments, out_path, width=720, height=1280, fps=30,
               stretch=True, work_dir=None, add_silent_audio=True):
    """把 edit_planner segments 裁剪归一化后硬剪成无声 base, 末尾补静音音轨。"""
    work_dir = work_dir or tempfile.mkdtemp(prefix="whq_clone_")
    os.makedirs(work_dir, exist_ok=True)
    seg_paths, manifest = [], []
    for seg in segments:
        src_info = _segment_source(seg)
        seg_out = os.path.join(work_dir, "seg_{:02d}.mp4".format(seg["index"]))
        if not src_info or not src_info[0] or not os.path.exists(src_info[0]):
            print("[clone] 跳过段{}: 源不存在".format(seg["index"]), flush=True)
            continue
        src, src_start, avail = src_info
        # voice_policy 句子级对窗可平移源窗口(source_start_override/source_take):
        # 让原声段的窗口以句尾收束, 画面与原声同倍率伸缩保持口型对齐。
        if seg.get("source_start_override") is not None:
            src_start = float(seg["source_start_override"])
        source_take = seg.get("source_take")
        target_dur = float(seg["target_duration"])
        _build_segment(src, src_start, avail, target_dur, seg_out,
                       width=width, height=height, fps=fps, stretch=stretch,
                       source_take=source_take)
        seg_paths.append(seg_out)
        manifest.append({
            "segment_index": seg["index"],
            "slot_id": seg.get("slot_id"),
            "beat_desc": seg.get("beat_desc"),
            "target_duration": target_dur,
            "ref_time_range": seg.get("ref_time_range", ""),
            "ref_cps": seg.get("ref_cps"),
            "source_path": src,
            "source_start": src_start,
            "source_avail": avail,
            "source_take": source_take,
            "score": seg.get("score"),
            "method": seg.get("method"),
            "is_t2v": bool(seg.get("is_t2v")),
            "segment_file": seg_out,
        })
    if not seg_paths:
        raise RuntimeError("没有可拼接的片段")
    list_file = os.path.join(work_dir, "concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write("file '{}'\n".format(os.path.abspath(p)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    silent_concat = os.path.join(work_dir, "_concat_silent.mp4")
    _run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
          "-c", "copy", silent_concat])
    if add_silent_audio:
        # 补一条静音立体声音轨(44.1k), 供后续 TTS 混流 [0:a] 映射使用。
        _run([FFMPEG, "-y", "-i", silent_concat,
              "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
              "-c:a", "aac", "-shortest", out_path])
    else:
        os.replace(silent_concat, out_path)
    # 更新 target_time_range(累计) 便于配音/字幕对齐
    t = 0.0
    for m in manifest:
        d = m["target_duration"]
        m["target_time_range"] = "{:.2f}-{:.2f}".format(t, t + d)
        t += d
    print("[clone] DONE ->", out_path, flush=True)
    return out_path, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="按结构级规划拼接无声 base 视频")
    ap.add_argument("--plan", required=True, help="edit_planner 输出 json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--no-stretch", action="store_true", help="短素材不拉伸(总时长会略短)")
    ap.add_argument("--no-silent-audio", action="store_true")
    ap.add_argument("--work-dir")
    args = ap.parse_args(argv)
    data = json.load(open(args.plan, encoding="utf-8"))
    segments = data["segments"] if isinstance(data, dict) and "segments" in data else data
    out, manifest = build_base(
        segments, args.out, width=args.width, height=args.height, fps=args.fps,
        stretch=not args.no_stretch, work_dir=args.work_dir,
        add_silent_audio=not args.no_silent_audio)
    mpath = os.path.splitext(out)[0] + "_manifest.json"
    json.dump({"output": out, "segments": manifest}, open(mpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("manifest ->", mpath)


if __name__ == "__main__":
    main()
