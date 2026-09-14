"""统一检索门面 Retriever（对齐 docs/retrieval_redesign.md，Phase 1-3）。

一处实现：入库 + 多 query 检索 + RRF 融合 + 千帆失败 lexical 降级 + embedding 内存缓存。

- 存储 per-task 隔离持久化：``uploads/vectors/{task_id}/store.json``；
- 语义索引 key 只用 ``one_sentence_summary + visual_description``（口播默认不进 key，
  ``RAG_INCLUDE_SPEECH=1`` 可选打开），其余 VLM 字段全部保留到 meta；
- 检索每个 shot 生成多条互补 query（L1 意图 + L2 视觉），各自余弦排序后 RRF 融合；
- 判定权交给上层 LLM 二判，这里不做相似度硬阈值，只保留「全空则空」兜底；
- 千帆 embedding 失败自动降级到本地 char-2gram Jaccard lexical，pipeline 不断链。

``RetrievalTool`` / ``VectorStore`` 退居兼容层，内部转发到本模块。
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

import obs

_log = obs.get_logger("retriever")

QIANFAN_BASE_URL = os.getenv("QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2")
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding-0.6b")
EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "60"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "500"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_FALLBACK_TO_LEXICAL = os.getenv("RAG_FALLBACK_TO_LEXICAL", "1") not in ("0", "false", "False")
RAG_INCLUDE_SPEECH = os.getenv("RAG_INCLUDE_SPEECH", "0") not in ("0", "false", "False")
RAG_INSTRUCTION = os.getenv("RAG_INSTRUCTION", "")

STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "uploads", "vectors",
)
os.makedirs(STORE_DIR, exist_ok=True)

_LOCK = threading.Lock()
# 进程内 embedding 缓存：key = f"{model}::{instruction}::{text}"，避免同一 pipeline 反复 embed
_EMBED_CACHE: dict[str, list[float]] = {}

# L2 视觉 query 白名单：只挑视觉/形式类维度拼句（情绪/节奏/结构进 L1 语境）
_VISUAL_DIMS = ("景别", "机位", "镜头", "运动", "主体", "展示重点", "交互动作",
                "动作", "光影", "构图", "素材类型", "视角", "画面")


# --------------------------------------------------------------------------- #
# 文本 / 相似度 / 融合 工具
# --------------------------------------------------------------------------- #
def _segment_text(seg: dict) -> str:
    """合并索引文本（供 lexical 兜底 + 展示）：summary + visual_description（口播默认排除）。"""
    if not isinstance(seg, dict):
        return str(seg)
    parts = [seg.get("one_sentence_summary") or "", seg.get("visual_description") or ""]
    if RAG_INCLUDE_SPEECH and seg.get("speech_or_text"):
        parts.append(seg.get("speech_or_text") or "")
    text = " ".join(p for p in parts if p).strip()
    return text or seg.get("asset_id", "")


def _join(v) -> str:
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v if x)
    return str(v or "")


def _segment_facets(seg: dict) -> dict:
    """把一个素材片段拆成两个语义面，各自单独 embedding：

    - visual（视觉面）：画面/景别/主体/动作/物体/关键词——匹配 L2 视觉 query。
    - intent（意图面）：一句话概括 + 适配角色/卖点（可选口播）——匹配 L1 意图 query。
    这样"视觉 query 只跟视觉面比、意图 query 只跟意图面比"，不再混在一个向量里互相稀释。
    """
    if not isinstance(seg, dict):
        return {"visual": str(seg), "intent": str(seg)}
    combined = _segment_text(seg)
    visual = " ".join(p for p in [
        seg.get("visual_description") or "",
        _join(seg.get("keywords")), _join(seg.get("visible_objects")),
        _join(seg.get("actions")), _join(seg.get("visual_evidence_tags")),
    ] if p).strip()
    intent_parts = [seg.get("one_sentence_summary") or "", _join(seg.get("suitable_roles"))]
    if RAG_INCLUDE_SPEECH and seg.get("speech_or_text"):
        intent_parts.append(seg.get("speech_or_text") or "")
    intent = " ".join(p for p in intent_parts if p).strip()
    fallback = combined or seg.get("asset_id", "")
    return {"visual": visual or fallback, "intent": intent or fallback}


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _char_ngrams(text: str, n: int = 2) -> set:
    s = str(text or "").replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def lexical_score(query: str, key_text: str, keywords: list) -> float:
    """千帆失败时的本地打分：char-2gram Jaccard + keyword 命中 boost。"""
    q, k = _char_ngrams(query), _char_ngrams(key_text)
    jaccard = len(q & k) / max(1, len(q | k))
    hits = sum(1 for kw in (keywords or []) if kw and kw in query)
    return jaccard + 0.1 * hits


def rrf_fuse(rankings: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion：rankings 为多条 query 各自的 id 排序（高分在前）。"""
    scores: dict = {}
    for ranking in rankings:
        for rank, asset_id in enumerate(ranking):
            scores[asset_id] = scores.get(asset_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _parse_duration(time_range: str) -> float:
    try:
        a, b = str(time_range).split("-")
        return round(max(0.0, float(b) - float(a)), 2)
    except (ValueError, AttributeError):
        return 0.0


def queries_for_shot(shot: dict) -> list:
    """为一个 shot 生成多条互补 query，每条带**语义面标签**：
    L1 意图（role·want，比意图面 intent）+ L2 视觉（visual breakdown，比视觉面 visual）。
    返回 [(query, facet), ...]，facet ∈ {"intent","visual"}。"""
    role = str(shot.get("role", "") or "").strip()
    want = str(shot.get("want", "") or "").strip()
    queries = []
    l1 = "·".join(p for p in (role, want) if p)
    if l1:
        queries.append((l1, "intent"))
    visual_vals = []
    for dim in shot.get("breakdown", []) or []:
        if not isinstance(dim, dict):
            continue
        name = str(dim.get("dim", "") or "")
        value = str(dim.get("value", "") or "").strip()
        if value and any(tok in name for tok in _VISUAL_DIMS):
            visual_vals.append(value)
    if visual_vals:
        queries.append((" ".join(visual_vals), "visual"))
    # 去重 + 去空
    seen, out = set(), []
    for q, facet in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append((q, facet))
    return out


# --------------------------------------------------------------------------- #
# Embedding（带进程内缓存 + 分批 + 截断）
# --------------------------------------------------------------------------- #
def _embed_qianfan(texts: list) -> list:
    payload = {"model": EMBED_MODEL, "input": texts}
    if RAG_INSTRUCTION:
        payload["instruction"] = RAG_INSTRUCTION
    req = urllib.request.Request(
        QIANFAN_BASE_URL.rstrip("/") + "/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + QIANFAN_API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"qianfan embeddings HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"qianfan embeddings request failed: {exc}") from exc
    data = sorted(body.get("data", []), key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def embed(texts: list) -> list:
    """返回每条文本的稠密向量；命中进程内缓存的不重复请求。失败抛 RuntimeError。"""
    texts = [str(t) for t in (texts or []) if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    clipped = [t[:EMBED_MAX_CHARS] for t in texts]
    misses = [t for t in clipped if f"{EMBED_MODEL}::{RAG_INSTRUCTION}::{t}" not in _EMBED_CACHE]
    unique_misses = list(dict.fromkeys(misses))
    for start in range(0, len(unique_misses), EMBED_BATCH):
        batch = unique_misses[start:start + EMBED_BATCH]
        vectors = _embed_qianfan(batch)
        if len(vectors) != len(batch):
            raise RuntimeError("embedding count mismatch")
        for text, vec in zip(batch, vectors):
            _EMBED_CACHE[f"{EMBED_MODEL}::{RAG_INSTRUCTION}::{text}"] = vec
    return [_EMBED_CACHE[f"{EMBED_MODEL}::{RAG_INSTRUCTION}::{t}"] for t in clipped]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """统一检索门面，per-task 持久化。"""

    def __init__(self, task_id: str = ""):
        self.task_id = str(task_id or "_shared")
        self._dir = os.path.join(STORE_DIR, self.task_id)
        self._path = os.path.join(self._dir, "store.json")

    # -- 持久化 --
    def _load(self) -> list:
        if not os.path.isfile(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, records: list):
        os.makedirs(self._dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)

    def embed(self, texts: list) -> list:
        return embed(texts)

    # -- 入库 --
    def index_segments(self, segments: list, source: str = "") -> dict:
        segs = [s for s in (segments or []) if isinstance(s, dict)]
        if not segs:
            return {"indexed": 0}
        texts = [_segment_text(s) for s in segs]
        facets = [_segment_facets(s) for s in segs]
        # 一批同时 embed 两个语义面（视觉面 + 意图面），各素材各存两条向量
        vis_texts = [f["visual"] for f in facets]
        int_texts = [f["intent"] for f in facets]
        try:
            vis_vecs = self.embed(vis_texts)
            int_vecs = self.embed(int_texts)
        except RuntimeError as exc:
            _log.warning("[%s] index embed failed: %s", self.task_id, exc)
            return {"indexed": 0, "error": str(exc)}
        if len(vis_vecs) != len(segs) or len(int_vecs) != len(segs):
            return {"indexed": 0, "error": "embedding count mismatch"}
        with _LOCK:
            records = self._load()
            existing = {r["id"] for r in records}
            added = skipped = 0
            for seg, text, vv, iv in zip(segs, texts, vis_vecs, int_vecs):
                gid = seg.get("global_asset_id") or f"{source}::{seg.get('asset_id', '')}"
                if gid in existing:
                    skipped += 1
                    continue
                records.append({
                    "id": gid,
                    "text": text,
                    "vector": iv,                       # 向后兼容：默认向量=意图面
                    "vectors": {"visual": vv, "intent": iv},
                    "meta": {
                        "asset_id": seg.get("asset_id", ""),
                        "source_video_id": seg.get("source_video_id") or source,
                        "source_path": seg.get("source_path", ""),
                        "source_time_range": seg.get("source_time_range", ""),
                        "duration": _parse_duration(seg.get("source_time_range", "")),
                        "asset_type": seg.get("asset_type", ""),
                        "one_sentence_summary": seg.get("one_sentence_summary", ""),
                        "visual_description": seg.get("visual_description", ""),
                        "speech_or_text": seg.get("speech_or_text", ""),
                        "keywords": seg.get("keywords", []) or [],
                        "actions": seg.get("actions", []) or [],
                        "visible_objects": seg.get("visible_objects", []) or [],
                        "visual_evidence_tags": seg.get("visual_evidence_tags", []) or [],
                        "quality_score": seg.get("quality_score", 0.0),
                        "limitations": seg.get("limitations", []) or [],
                        "indexed_at": int(time.time()),
                    },
                })
                existing.add(gid)
                added += 1
            self._save(records)
        _log.info("[%s] indexed %d segments (skipped %d) from %s (total=%d)",
                  self.task_id, added, skipped, source, len(records))
        return {"indexed": added, "skipped": skipped, "total": len(records)}

    # -- 检索 --
    def search(self, queries, top_k: int = None) -> dict:
        top_k = max(1, int(top_k or RAG_TOP_K))
        if isinstance(queries, str):
            queries = [queries]
        # 归一为 [(text, facet)]；facet ∈ {"visual","intent",None}。None=自由 query，跟两面取较大。
        norm, seen = [], set()
        for q in (queries or []):
            if isinstance(q, (list, tuple)) and len(q) == 2:
                text, facet = str(q[0] or "").strip(), q[1]
            else:
                text, facet = str(q or "").strip(), None
            if text and (text, facet) not in seen:
                seen.add((text, facet))
                norm.append((text, facet))
        query_list = [t for t, _ in norm]
        records = self._load()
        base = {"query_list": query_list, "matches": [], "count": 0,
                "store_size": len(records), "backend": "qianfan"}
        if not norm or not records:
            return base

        def _rec_vec(rec, facet):
            vecs = rec.get("vectors") or {}
            if facet in ("visual", "intent") and vecs.get(facet):
                return vecs[facet]
            return rec.get("vector") or vecs.get("intent") or vecs.get("visual") or []

        by_id = {r["id"]: r for r in records}
        rankings, cosine_by_query, backend = [], {}, "qianfan"
        try:
            qvecs = self.embed(query_list)
            qvec_by_text = dict(zip(query_list, qvecs))
            for text, facet in norm:
                qvec = qvec_by_text[text]
                scored = []
                for r in records:
                    if facet is None:
                        cos = max(_cosine(qvec, _rec_vec(r, "visual")),
                                  _cosine(qvec, _rec_vec(r, "intent")))
                    else:
                        cos = _cosine(qvec, _rec_vec(r, facet))
                    scored.append((r["id"], cos))
                scored.sort(key=lambda x: x[1], reverse=True)
                rankings.append([rid for rid, _ in scored])
                cosine_by_query[text] = {rid: round(sc, 4) for rid, sc in scored}
        except RuntimeError as exc:
            if not RAG_FALLBACK_TO_LEXICAL:
                _log.warning("[%s] embed failed, no lexical fallback: %s", self.task_id, exc)
                base["backend"] = "embed_failed"
                base["error"] = str(exc)
                return base
            _log.warning("[%s] embed failed, lexical fallback: %s", self.task_id, exc)
            backend = "lexical_fallback"
            rankings, cosine_by_query = [], {}
            for text, _facet in norm:
                scored = [
                    (r["id"], lexical_score(text, r.get("text", ""), r.get("meta", {}).get("keywords", [])))
                    for r in records
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                rankings.append([rid for rid, _ in scored])
                cosine_by_query[text] = {rid: round(sc, 4) for rid, sc in scored}

        fused = rrf_fuse(rankings)
        # 最终兜底：所有 query 都没召回任何素材（不可能到这，records 非空则总有排序）
        if not fused or fused[0][1] <= 0:
            base["backend"] = backend
            return base
        matches = []
        for rid, rrf_score in fused[:top_k]:
            rec = by_id.get(rid, {})
            matches.append({
                "id": rid,
                "rrf_score": round(rrf_score, 6),
                "cosine_scores": {q: cosine_by_query.get(q, {}).get(rid, 0.0) for q in query_list},
                "quality_score": rec.get("meta", {}).get("quality_score", 0.0),
                "meta": rec.get("meta", {}),
            })
        return {"query_list": query_list, "matches": matches, "count": len(matches),
                "store_size": len(records), "backend": backend}

    def retrieve_for_shot(self, shot: dict, top_k: int = None) -> dict:
        return self.search(queries_for_shot(shot), top_k=top_k)

    def size(self) -> int:
        return len(self._load())

    def clear(self):
        with _LOCK:
            self._save([])
