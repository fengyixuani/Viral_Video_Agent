"""reference_shots — 构建参考视频的真实分镜时间线。

分镜来源合并两路:
  1. ffmpeg 场景检测 (select='gt(scene,T)') -> 真实切点/时长 (捕捉节奏)
  2. DNA key_beats -> 每个 beat 的语义描述 + 大致时间 (捕捉「这个镜头在讲什么」)

产出 shot 列表, 每个 shot:
    {index, start, end, duration, description, beat_desc, source}

若拿不到参考视频 (只有 DNA), 退化为直接用 key_beats 时间线。
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import FFMPEG


# ---------- DNA 解析 ----------

def _extract_json_block(text):
    """从 DNA md 的 '## 模型输出' 之后抓第一个平衡的 JSON 对象。"""
    marker = text.find("## 模型输出")
    scan = text[marker:] if marker != -1 else text
    start = scan.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(scan)):
        ch = scan[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return scan[start:i + 1]
    return None


def load_dna(dna_path):
    text = open(dna_path, encoding="utf-8").read()
    block = _extract_json_block(text)
    if not block:
        raise ValueError("DNA md 中未找到 JSON 模型输出: {}".format(dna_path))
    data = json.loads(block)
    return data.get("viral_dna_template", data)


_TS = re.compile(r"(\d{1,2}):(\d{2})(?:\.(\d+))?")


def _parse_ts(tok):
    m = _TS.match(tok.strip())
    if not m:
        return None
    mm, ss, frac = m.groups()
    val = int(mm) * 60 + int(ss)
    if frac:
        val += float("0." + frac)
    return float(val)


def parse_key_beats(dna):
    """key_beats: ['00:00-00:02：成品特写...', ...] -> [{start,end,desc}]。"""
    beats = (dna.get("content_structure") or {}).get("key_beats") or []
    out = []
    for raw in beats:
        s = str(raw)
        # 分隔符可能是全角冒号「：」或半角「:」后接文字
        head = s
        desc = s
        m = re.match(r"\s*(\d{1,2}:\d{2}(?:\.\d+)?)\s*[-~到]\s*(\d{1,2}:\d{2}(?:\.\d+)?)\s*[：:]?\s*(.*)", s)
        if m:
            start = _parse_ts(m.group(1))
            end = _parse_ts(m.group(2))
            desc = (m.group(3) or "").strip()
        else:
            start = end = None
            # 去掉可能的前缀编号
            desc = re.sub(r"^\s*[-\d、.]+\s*", "", s).strip()
        out.append({"start": start, "end": end, "desc": desc, "raw": head})
    return out


# ---------- 场景检测 ----------

def detect_scene_cuts(video_path, threshold=0.30):
    """返回场景切点时间戳列表 (秒), 不含 0 和结尾。"""
    if not video_path or not os.path.exists(video_path):
        return []
    cmd = [
        FFMPEG, "-hide_banner", "-i", video_path,
        "-filter_complex", "select='gt(scene,{})',metadata=print".format(threshold),
        "-an", "-f", "null", "-",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cuts = []
    for m in re.finditer(r"pts_time:([0-9.]+)", r.stdout):
        t = float(m.group(1))
        if t > 0.05:
            cuts.append(round(t, 3))
    # 去重排序
    return sorted(set(cuts))


def _video_duration(video_path):
    if not video_path or not os.path.exists(video_path):
        return 0.0
    cmd = [FFMPEG, "-hide_banner", "-i", video_path, "-f", "null", "-"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stdout)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


# ---------- 合并 ----------

def _overlap(a0, a1, b0, b1):
    if None in (a0, a1, b0, b1):
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_reference_shots(video_path=None, dna_path=None, duration=None,
                          scene_threshold=0.30, min_shot=0.6):
    """合并场景检测切点与 DNA key_beats -> shot 列表。"""
    dna = load_dna(dna_path) if dna_path else {}
    beats = parse_key_beats(dna) if dna else []
    if duration is None:
        duration = _video_duration(video_path) if video_path else None
    if not duration:
        duration = dna.get("duration_estimate") or (beats[-1]["end"] if beats and beats[-1]["end"] else 0.0)
    duration = float(duration or 0.0)

    cuts = detect_scene_cuts(video_path, scene_threshold) if video_path else []
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    bounds = sorted(set(round(b, 3) for b in bounds))
    # 合并过短镜头
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < min_shot and b != duration:
            continue
        merged.append(b)
    if merged[-1] != duration and duration:
        merged.append(duration)

    shots = []
    use_scene = len(merged) >= 3  # 场景检测有效
    if not use_scene and beats:
        # 退化: 直接用 key_beats 时间线
        for i, bt in enumerate(beats):
            start = bt["start"] if bt["start"] is not None else (i * duration / max(1, len(beats)))
            end = bt["end"] if bt["end"] is not None else ((i + 1) * duration / max(1, len(beats)))
            shots.append({
                "index": i + 1, "start": round(start, 3), "end": round(end, 3),
                "duration": round(end - start, 3), "description": bt["desc"],
                "beat_desc": bt["desc"], "source": "dna_key_beats",
            })
        return {"duration": duration, "shots": shots, "n_scene_cuts": len(cuts)}

    for i in range(len(merged) - 1):
        s0, s1 = merged[i], merged[i + 1]
        # 找与该镜头时间重叠最大的 beat 描述
        best_desc, best_ov = "", 0.0
        for bt in beats:
            ov = _overlap(s0, s1, bt["start"], bt["end"])
            if ov > best_ov:
                best_ov, best_desc = ov, bt["desc"]
        shots.append({
            "index": i + 1, "start": round(s0, 3), "end": round(s1, 3),
            "duration": round(s1 - s0, 3),
            "description": best_desc or (dna.get("topic_and_emotion", {}) or {}).get("topic", ""),
            "beat_desc": best_desc, "source": "scene_detect+dna",
        })
    return {"duration": duration, "shots": shots, "n_scene_cuts": len(cuts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="构建参考视频分镜时间线")
    ap.add_argument("--ref", help="参考视频路径")
    ap.add_argument("--dna", help="DNA md 路径")
    ap.add_argument("--scene-threshold", type=float, default=0.30)
    ap.add_argument("--out", help="输出 json 路径 (默认打印)")
    args = ap.parse_args(argv)
    result = build_reference_shots(args.ref, args.dna, scene_threshold=args.scene_threshold)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
