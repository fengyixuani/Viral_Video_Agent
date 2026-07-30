"""对已配音成片(_voiced.mp4)按参考语速逐段变速。

实际语速: 用成片自身 ASR 逐字时间戳, 每段(槽位窗口)内合并 token 区间得有声时长,
actual_cps = 窗内字符数 / 有声时长。
目标语速: min(该段 ref_cps(来自 manifest), TARGET_CPS_MAX), 倍率钳 [1.0, MAX_SPEED]。
复用 whq.pace_match.apply_pace 做逐段 trim+setpts/atempo+concat。

与 pace_match.match_pace 的区别: match_pace 依赖 _whq_work/tts/tts_overlay_plan.json
(只反映最近一次 run_clone 的产物); 本脚本对**任意已出片的 _voiced.mp4** 生效——
先对成片自身跑 ASR 拿逐字时间戳, 实际语速不依赖中间产物。

用法:
  1) ASR 成片(viral-split-asr 环境):
     env ASR_PYTHON=/root/miniconda3/envs/viral-split-asr/bin/python \
         FFMPEG=/root/miniconda3/envs/viral-split-tts/bin/ffmpeg \
         NARIS_SOURCE_GLOB=<voiced.mp4> NARIS_ASR_DIR=<workdir> \
         $ASR_PYTHON understanding/batch_qwen3_asr.py
  2) 变速出片:
     /root/miniconda3/envs/viral-split-tts/bin/python generation/whq/pace_voiced.py \
         <voiced.mp4> <workdir>/all_source_asr.json <slug>_manifest.json <out.mp4>
详见 whq/docs/pace_match.md。
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pace_match  # noqa: E402

FFPROBE = "/root/miniconda3/envs/viral-split-tts/bin/ffprobe"
_PUNCT = re.compile(r"[\s，。,.!？?、；;：:…~\-]")


def probe_duration(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def merged_voiced(tokens, w0, w1):
    """窗口 [w0,w1] 内 token 区间并集时长 + 字符数。"""
    ivs, chars = [], 0
    for t in tokens:
        s, e = float(t["start"]), float(t["end"])
        mid = (s + e) / 2
        if not (w0 <= mid < w1):
            continue
        if _PUNCT.sub("", t["text"]):
            chars += len(_PUNCT.sub("", t["text"]))
        ivs.append((max(s, w0), min(e, w1)))
    ivs.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e + 0.15:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total, chars


def build_plan(asr_json, manifest_json, video):
    tokens = json.load(open(asr_json))[0]["asr_items"]
    mani = json.load(open(manifest_json))
    vdur = probe_duration(video)
    plan = []
    for seg in mani["segments"]:
        t0, t1 = (float(x) for x in seg["target_time_range"].split("-"))
        t1 = min(t1, vdur)
        voiced, chars = merged_voiced(tokens, t0, t1)
        actual = chars / voiced if voiced > 0.2 else 0.0
        ref = float(seg.get("ref_cps") or 0.0)
        if actual > 0 and ref > 0:
            target = min(ref, pace_match.TARGET_CPS_MAX)
            speed = max(pace_match.MIN_SPEED, min(pace_match.MAX_SPEED, target / actual))
        else:
            speed = 1.0
        plan.append({
            "slot_id": seg["slot_id"], "start": round(t0, 3), "end": round(t1, 3),
            "actual_cps": round(actual, 2), "ref_cps": ref,
            "speed": round(speed, 3), "new_dur": round((t1 - t0) / speed, 3),
            "result_cps": round(actual * speed, 2),
        })
    return plan


def main():
    video, asr_json, manifest_json, out = sys.argv[1:5]
    plan = build_plan(asr_json, manifest_json, video)
    for p in plan:
        print("  {slot_id}: {actual_cps} cps -> x{speed} -> {result_cps} cps (ref {ref_cps})".format(**p))
    pace_match.apply_pace(video, plan, out)
    total_old = sum(p["end"] - p["start"] for p in plan)
    total_new = sum(p["new_dur"] for p in plan)
    print("total {:.2f}s -> {:.2f}s".format(total_old, total_new))


if __name__ == "__main__":
    main()
