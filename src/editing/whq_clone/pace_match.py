"""pace_match — 成片整体语速贴近参考视频(逐段加速)。

背景: whq 成片每段时长=参考节拍时长, 但口播字数受 CPS_MAX/atempo cap 限制,
实际语速(~3-4 字/秒)远低于参考爆款(~7 字/秒), 观感"慢半拍"。
做法: 对**烧完字幕的成片**按槽位边界逐段变速——
  每段倍率 r = 该段参考 ref_cps / 该段实际 cps, 钳 [min_speed, max_speed];
  两侧 cps 都按**有说话时间**计: ref_cps 由 edit_planner 按参考逐字时间并集算,
  实际 cps 用该段语音 wav 的 silencedetect 有声时长(而非槽时长)作分母,
  避免把段内留白摊进语速、systematically 低估两边的真实语速;
  视频 setpts/r, 音频 atempo=r(变速不变调); 字幕已是画面像素, 随段同步。
逐段而非全局: 原声段(如 S06)可能本就达到参考语速, 全局加速会把它推到 12+ 字/秒。
默认 KEEP_TOTAL=1 只提语速、画面段长不变, 因此**成片总时长 = 参考总时长**;
WHQ_PACE_KEEP_TOTAL=0 回到音画同倍率加速的旧行为(总时长会短于参考)。
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import FFMPEG

MAX_SPEED = float(os.getenv("WHQ_PACE_MAX_SPEED", "1.4"))
MIN_SPEED = float(os.getenv("WHQ_PACE_MIN_SPEED", "1.0"))
# 结果语速天花板(字/秒): 参考爆款常是连续 7+ 字/秒的自然快语者, 但配音/原声段
# 是短句满槽发声, 硬 atempo 拉到 7 字/秒会变成"录音快放"般的失真快感(实测比参考更快)。
# 故不追参考瞬时 cps, 只把每段结果语速钳到自然可懂的上限, 变速倍率同时受 MAX_SPEED 约束。
TARGET_CPS_MAX = float(os.getenv("WHQ_PACE_TARGET_CPS", "5.5"))
# 保总时长(默认开): 只把**语音**按倍率提速, 画面段长保持 = 参考节拍时长, 音频变短的部分补静音。
# 旧行为(=0)把画面一起加速, 成片总时长会比参考短一大截(实测参考 43.8s -> 成片 36.1s),
# 而画面节奏本就来自参考节拍, 再加速等于比参考更快。保总时长后: 画面节奏=参考、语速≈参考、
# 总时长≈参考, 代价是语速提上去后段尾会多出一点没有口播的画面留白。
KEEP_TOTAL = os.getenv("WHQ_PACE_KEEP_TOTAL", "1") not in ("0", "false", "False")

_PUNCT_RE = re.compile(r"[\s，。,.!？?、；;：:…~\-]")


def _cps(text, duration):
    chars = len(_PUNCT_RE.sub("", str(text or "")))
    return (chars / duration) if duration > 0 else 0.0


_SILENCE_RE = re.compile(r"silence_duration:\s*([0-9.]+)")


def _voiced_duration(wav, noise_db=-35, min_sil=0.25):
    """wav 的有声时长(总长-静音段), 用 ffmpeg silencedetect。失败返回 None。"""
    if not wav or not os.path.exists(wav):
        return None
    try:
        p = subprocess.run(
            [FFMPEG, "-i", wav, "-af",
             "silencedetect=noise={}dB:d={}".format(noise_db, min_sil),
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        err = p.stderr or ""
        dm = re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+)", err)
        if not dm:
            return None
        total = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
        sil = sum(float(x) for x in _SILENCE_RE.findall(err))
        voiced = total - sil
        return voiced if voiced > 0.2 else None
    except Exception:
        return None


def plan_speeds(items, max_speed=MAX_SPEED, min_speed=MIN_SPEED, target_cps_max=TARGET_CPS_MAX,
                prior_speeds=None, keep_total=KEEP_TOTAL):
    """按 tts_overlay_plan items 算每段变速倍率。返回 [{slot_id,start,end,speed,...}]。

    prior_speeds: {slot_id: 硬剪阶段已施加的倍率}(见 run_clone._prior_speeds)。原声段为对齐
    完整句常已被加速到 1.35，这里再乘一次会叠成 ~1.9 倍快放；故按 max_speed/已用倍率封顶，
    保证「累计」倍率不超过 max_speed。
    """
    prior_speeds = prior_speeds or {}
    plan = []
    for it in items:
        start = float(it.get("start") or 0.0)
        end = float(it.get("end") or 0.0)
        dur = end - start
        if dur <= 0:
            continue
        # 实际语速按**有说话时间**计: 优先用该段语音 wav 的有声时长, 退化用槽时长
        wav = it.get("voice_wav") or it.get("tts_wav")
        vdur = _voiced_duration(wav)
        actual = _cps(it.get("text"), min(vdur, dur) if vdur else dur)
        ref = float(it.get("ref_cps") or 0.0)
        prior = float(prior_speeds.get(it.get("slot_id")) or 1.0)
        seg_max = max(min_speed, max_speed / prior) if prior > 1.0 else max_speed
        # 保总时长模式下只提音频、画面不动, 因此**原声段必须不变速**, 否则声音比口型快、
        # 口型对不上(原声段的价值就是口型对得上)。原声段的语速本来就是用户真实语速。
        if keep_total and str(it.get("voice_source") or "") == "original":
            seg_max = min_speed
        if actual > 0 and ref > 0:
            # 目标语速 = min(参考该段 cps, 结果语速天花板): 参考很快时不硬追, 只提到自然上限
            target = min(ref, target_cps_max) if target_cps_max > 0 else ref
            speed = max(min_speed, min(seg_max, target / actual))
        else:
            speed = 1.0
        plan.append({
            "slot_id": it.get("slot_id"),
            "start": round(start, 3),
            "end": round(end, 3),
            "actual_cps": round(actual, 2),
            "ref_cps": ref,
            "prior_speed": round(prior, 3),
            "speed": round(speed, 3),
            "new_dur": round(dur / speed, 3),
            "result_cps": round(actual * speed, 2),
        })
    return plan


def apply_pace(video, speed_plan, out_video, keep_total=KEEP_TOTAL):
    """按 speed_plan 逐段变速后 concat, 一次 filter_complex 出片。

    keep_total=True: 只加速音频, 画面段长不变(段尾补静音) -> 成片总时长 = 参考总时长。
    keep_total=False: 音画同倍率加速 -> 总时长按倍率缩短(旧行为)。
    """
    parts, vlabels, alabels = [], [], []
    for i, seg in enumerate(speed_plan):
        r = seg["speed"]
        t0, t1 = seg["start"], seg["end"]
        dur = t1 - t0
        if keep_total:
            parts.append(
                "[0:v]trim=start={:.3f}:end={:.3f},setpts=PTS-STARTPTS[v{}]".format(t0, t1, i))
            # 语音提速后比槽短, 用 apad 补静音再裁回槽长, 保证音画等长、拼接不漂移
            parts.append(
                "[0:a]atrim=start={:.3f}:end={:.3f},asetpts=PTS-STARTPTS,atempo={:.4f},"
                "apad,atrim=duration={:.3f},asetpts=PTS-STARTPTS[a{}]".format(t0, t1, r, dur, i))
        else:
            parts.append(
                "[0:v]trim=start={:.3f}:end={:.3f},setpts=(PTS-STARTPTS)/{:.4f}[v{}]".format(t0, t1, r, i))
            # atempo 单滤镜支持 0.5~2.0, 倍率已钳在该范围内
            parts.append(
                "[0:a]atrim=start={:.3f}:end={:.3f},asetpts=PTS-STARTPTS,atempo={:.4f}[a{}]".format(t0, t1, r, i))
        vlabels.append("[v{}]".format(i))
        alabels.append("[a{}]".format(i))
    n = len(speed_plan)
    parts.append("{}concat=n={}:v=1:a=0[vout]".format("".join(vlabels), n))
    parts.append("{}concat=n={}:v=0:a=1[aout]".format("".join(alabels), n))
    cmd = [FFMPEG, "-y", "-i", video,
           "-filter_complex", ";".join(parts),
           "-map", "[vout]", "-map", "[aout]",
           "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-ar", "44100", "-ac", "2", out_video]
    print("[pace_match] {}| ".format("保总时长(仅提语速) " if keep_total else "") + " | ".join(
        "{} x{:.2f} ({:.1f}->{:.1f}cps)".format(s["slot_id"], s["speed"], s["actual_cps"], s["result_cps"])
        for s in speed_plan), flush=True)
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_video


def match_pace(video, tts_plan_path, out_video, max_speed=MAX_SPEED, min_speed=MIN_SPEED,
               prior_speeds=None, keep_total=KEEP_TOTAL):
    with open(tts_plan_path, encoding="utf-8") as f:
        items = json.load(f).get("items", [])
    speed_plan = plan_speeds(items, max_speed=max_speed, min_speed=min_speed,
                             prior_speeds=prior_speeds, keep_total=keep_total)
    if not speed_plan:
        raise ValueError("no valid segments in tts plan")
    apply_pace(video, speed_plan, out_video, keep_total=keep_total)
    total_old = sum(s["end"] - s["start"] for s in speed_plan)
    total_new = total_old if keep_total else sum(s["new_dur"] for s in speed_plan)
    w_cps = sum(s["result_cps"] * s["new_dur"] for s in speed_plan) / max(
        1e-6, sum(s["new_dur"] for s in speed_plan))
    print("[pace_match] 总时长 {:.1f}s -> {:.1f}s, 口播语速 -> {:.2f} 字/秒".format(
        total_old, total_new, w_cps), flush=True)
    return out_video, speed_plan


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tts-plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED)
    ap.add_argument("--min-speed", type=float, default=MIN_SPEED)
    ap.add_argument("--shrink-total", action="store_true",
                    help="音画同倍率加速(总时长会短于参考); 默认只提语速、保总时长")
    args = ap.parse_args(argv)
    match_pace(args.video, args.tts_plan, args.out, args.max_speed, args.min_speed,
               keep_total=not args.shrink_total)


if __name__ == "__main__":
    main()
