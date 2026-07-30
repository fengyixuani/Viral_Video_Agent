"""shot_matcher — 为每个参考镜头挑选「最像」的用户素材片段。

两条路:
  - LLM (ask_qianfan): 逐镜把镜头描述 + 候选清单交给模型打分挑选 (默认, 需网络)
  - 确定性兜底 (--no-llm 或 LLM 不可达): 中文二元组 Jaccard + asset_type 关键词
    + quality_score 加权, 完全离线可复现。

产出 matches: 每个参考镜头 -> {shot, best_candidate, score, method, alternatives}
低于 gap_threshold 的镜头标 is_gap=True (供 seedance 补拍)。
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import COMMON  # noqa: F401  (确保 sys.path 就绪)

try:
    from pipeline_utils import ask_qianfan, loads_with_repair
    _HAS_LLM = True
except Exception:  # pragma: no cover
    _HAS_LLM = False


# ---------- 确定性打分 ----------

_STOP = set("的了在是和与及为对把被将从到并且或者一个这那有着我们你他她它")


def _bigrams(text):
    chars = re.sub(r"[\s，。、！？：；·\-—…()（）【】\"'0-9a-zA-Z.]+", "", text or "")
    chars = "".join(c for c in chars if c not in _STOP)
    if len(chars) < 2:
        return set(chars)
    return set(chars[i:i + 2] for i in range(len(chars) - 1))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def score_candidate_deterministic(shot_desc, cand):
    sg = _bigrams(shot_desc)
    cg = _bigrams(cand["text"])
    sim = _jaccard(sg, cg)
    # 关键词直接命中加成
    kw_hit = sum(1 for k in cand.get("keywords", []) if k and k in shot_desc)
    kw_bonus = min(0.2, 0.05 * kw_hit)
    quality = float(cand.get("quality_score") or 0.0)
    return round(0.7 * sim + kw_bonus + 0.1 * quality, 4)


def match_deterministic(shots, candidates, reuse_penalty=0.15):
    used = {}
    matches = []
    for shot in shots:
        scored = []
        for c in candidates:
            base = score_candidate_deterministic(shot["description"], c)
            pen = reuse_penalty * used.get(c["global_asset_id"], 0)
            scored.append((round(base - pen, 4), base, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        used[best[2]["global_asset_id"]] = used.get(best[2]["global_asset_id"], 0) + 1
        matches.append({
            "shot": shot,
            "best_candidate": best[2],
            "score": best[1],
            "effective_score": best[0],
            "method": "deterministic",
            "alternatives": [
                {"global_asset_id": s[2]["global_asset_id"], "score": s[1]}
                for s in scored[1:4]
            ],
        })
    return matches


# ---------- LLM 打分 ----------

_LLM_PROMPT = """你是短视频剪辑师。参考视频有一个镜头, 需要从候选用户素材片段里挑出**画面内容最相似**的一个来复刻它。

参考镜头(第 {idx} 个, 时长约 {dur:.1f}s):
{shot_desc}

候选用户素材片段:
{cand_block}

请只输出 JSON:
{{"best_id": "<最像的候选 global_asset_id>", "score": <0~1 相似度>, "reason": "<一句话>", "ranking": ["<次优 id>", ...]}}
score 表示该候选与参考镜头画面内容的相似程度; 若没有任何候选贴切, score 给低分(<0.4)。"""


def _llm_match_one(shot, candidates, model=None):
    cand_lines = []
    for c in candidates:
        cand_lines.append("- {} [{}] {:.1f}s: {}".format(
            c["global_asset_id"], c.get("asset_type", ""), c.get("duration", 0.0),
            c["text"][:160]))
    prompt = _LLM_PROMPT.format(
        idx=shot["index"], dur=shot["duration"],
        shot_desc=shot["description"] or "(无描述)",
        cand_block="\n".join(cand_lines))
    text, _ = ask_qianfan([{"role": "user", "content": prompt}], model=model, temperature=0.1)
    obj = loads_with_repair(text)
    return obj


def _norm_id(s):
    return re.sub(r"\s+", "", str(s or ""))


def match_llm(shots, candidates, model=None, reuse_penalty=0.10):
    by_id = {c["global_asset_id"]: c for c in candidates}
    # LLM 常把 "牛肉饼4-素材3" 渲染成 "牛肉饼 4-素材 3" (CJK/数字间插空格),
    # 故按去空格做一份归一化查表兜底。
    by_norm = {_norm_id(k): v for k, v in by_id.items()}

    def resolve(rid):
        if rid in by_id:
            return by_id[rid]
        return by_norm.get(_norm_id(rid))

    used = {}
    matches = []
    for shot in shots:
        try:
            obj = _llm_match_one(shot, candidates, model=model)
        except Exception as exc:
            print("[shot_matcher] LLM 第{}镜失败, 该镜回退确定性: {}".format(
                shot["index"], str(exc)[:160]), flush=True)
            obj = None
        cand = resolve(obj.get("best_id")) if obj else None
        if not cand:
            # 单镜回退确定性
            det = match_deterministic([shot], candidates, reuse_penalty=0)[0]
            det["method"] = "deterministic_fallback"
            matches.append(det)
            used[det["best_candidate"]["global_asset_id"]] = used.get(
                det["best_candidate"]["global_asset_id"], 0) + 1
            continue
        score = float(obj.get("score", 0.5))
        pen = reuse_penalty * used.get(cand["global_asset_id"], 0)
        used[cand["global_asset_id"]] = used.get(cand["global_asset_id"], 0) + 1
        matches.append({
            "shot": shot,
            "best_candidate": cand,
            "score": round(score, 4),
            "effective_score": round(score - pen, 4),
            "method": "llm",
            "reason": obj.get("reason", ""),
            "alternatives": [
                {"global_asset_id": rc["global_asset_id"], "score": None}
                for rc in (resolve(rid) for rid in (obj.get("ranking") or []))
                if rc is not None
            ][:3],
        })
    return matches


def match_shots(shots, candidates, use_llm=True, model=None,
                gap_threshold=0.4, det_gap_threshold=0.06):
    if use_llm and _HAS_LLM:
        matches = match_llm(shots, candidates, model=model)
    else:
        matches = match_deterministic(shots, candidates)
    for m in matches:
        # 确定性打分量纲被压缩(中文二元组 Jaccard 绝对值天然偏低), 用单独阈值。
        thr = det_gap_threshold if m["method"].startswith("deterministic") else gap_threshold
        m["is_gap"] = m["score"] < thr
    return matches


def main(argv=None):
    from reference_shots import build_reference_shots
    from asset_index import load_candidates
    ap = argparse.ArgumentParser(description="逐镜匹配用户素材")
    ap.add_argument("--ref")
    ap.add_argument("--dna")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--gap-threshold", type=float, default=0.4)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    ref = build_reference_shots(args.ref, args.dna)
    cands = load_candidates(args.assets)
    matches = match_shots(ref["shots"], cands, use_llm=not args.no_llm,
                          model=args.model, gap_threshold=args.gap_threshold)
    out = {"reference": ref, "n_candidates": len(cands), "matches": matches}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print("wrote", args.out)
    for m in matches:
        s = m["shot"]
        flag = " GAP" if m["is_gap"] else ""
        print("  镜{:>2} {:>5.2f}-{:<5.2f} <- {:28s} score={:.3f} [{}]{}".format(
            s["index"], s["start"], s["end"], m["best_candidate"]["global_asset_id"],
            m["score"], m["method"], flag))


if __name__ == "__main__":
    main()
