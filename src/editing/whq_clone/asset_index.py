"""asset_index — 把 understanding 阶段的用户素材理解产物拍平成候选片段库。

输入: user_understanding/.../all_user_assets.json  (或该目录, 或单个理解 json)。
输出: candidates 列表, 每个:
    {global_asset_id, source_video_id, source_path, start, end, duration,
     asset_type, text}   # text = 供匹配的合并描述文本
"""
import argparse
import glob
import json
import os
import re


def _parse_range(rng):
    if not rng:
        return 0.0, 0.0
    m = re.match(r"\s*([0-9.]+)\s*[-~]\s*([0-9.]+)", str(rng))
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def _seg_text(seg):
    parts = [
        seg.get("one_sentence_summary", ""),
        seg.get("visual_description", ""),
        seg.get("asset_type", ""),
        " ".join(seg.get("actions", []) or []),
        " ".join(seg.get("visible_objects", []) or []),
        " ".join(seg.get("visual_evidence_tags", []) or []),
        " ".join(seg.get("keywords", []) or []),
    ]
    return "  ".join(p for p in parts if p).strip()


def _iter_understanding_files(path):
    if os.path.isfile(path):
        base = os.path.basename(path)
        if base == "all_user_assets.json":
            yield path
            return
        yield path
        return
    if os.path.isdir(path):
        allf = os.path.join(path, "all_user_assets.json")
        if os.path.exists(allf):
            yield allf
            return
        for f in sorted(glob.glob(os.path.join(path, "*_understanding.json"))):
            yield f


def load_candidates_from_results(results):
    """从内存里的 understanding ``results`` 列表拍平成候选片段库并去重。

    ``results`` 每项形如 ``{source_path, source_video_id, asset_segments:[...]}``。
    Agent 集成时 ``AnalysisResult.material_understanding.values()`` 正是这个形状，
    可直接传入（见 whq_input.candidates_from_understanding），无需落盘。
    """
    candidates = []
    for res in results or []:
        if not isinstance(res, dict):
            continue
        src_path = res.get("source_path", "")
        src_id = res.get("source_video_id", os.path.splitext(os.path.basename(src_path))[0])
        for seg in res.get("asset_segments", []) or []:
            s, e = _parse_range(seg.get("source_time_range"))
            candidates.append({
                "global_asset_id": seg.get("global_asset_id") or "{}::{}".format(src_id, seg.get("asset_id", "")),
                "source_video_id": seg.get("source_video_id", src_id),
                "source_path": seg.get("source_path", src_path),
                "start": s, "end": e, "duration": round(max(0.0, e - s), 3),
                "asset_type": seg.get("asset_type", ""),
                "quality_score": seg.get("quality_score", 0.0),
                "keywords": seg.get("keywords", []) or [],
                "text": _seg_text(seg),
            })
    # 去重 (同一 global_asset_id)
    seen, uniq = set(), []
    for c in candidates:
        if c["global_asset_id"] in seen:
            continue
        seen.add(c["global_asset_id"])
        uniq.append(c)
    return uniq


def load_candidates(path):
    """path 可以是 all_user_assets.json / understanding 目录 / 单个理解 json。"""
    results = []
    for f in _iter_understanding_files(path):
        data = json.load(open(f, encoding="utf-8"))
        if isinstance(data, dict) and "results" in data:
            results.extend(data["results"])      # all_user_assets.json 形态
        elif isinstance(data, dict) and "asset_segments" in data:
            results.append(data)                 # 单个素材理解
        elif isinstance(data, list):
            results.extend(data)
    return load_candidates_from_results(results)


def main(argv=None):
    ap = argparse.ArgumentParser(description="拍平用户素材理解产物为候选片段库")
    ap.add_argument("--assets", required=True, help="all_user_assets.json / 目录 / 单个理解 json")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cands = load_candidates(args.assets)
    print("candidates:", len(cands))
    text = json.dumps(cands, ensure_ascii=False, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print("wrote", args.out)
    else:
        for c in cands:
            print("  {:28s} {:>5.2f}-{:<5.2f} {}".format(
                c["global_asset_id"], c["start"], c["end"], c["asset_type"]))


if __name__ == "__main__":
    main()
