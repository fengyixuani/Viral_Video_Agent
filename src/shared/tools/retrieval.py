"""检索工具兼容层：``RetrievalTool`` 保留 ad-hoc 片段检索接口。

Embedding 与文本拼接统一走 ``tools.retriever``（带进程内缓存、分批、截断），
避免与向量库两处实现漂移。``retrieve`` 针对「传入的一批候选片段」即时算相似度，
不落库，供 react_agents 的 FunctionTool 使用。
"""
from __future__ import annotations

import obs
from . import retriever as _R

_log = obs.get_logger("retrieval")


class RetrievalTool:
    name = "检索"

    def embed(self, texts: list) -> list:
        """调用千帆 embedding（带缓存）。失败抛 RuntimeError。"""
        return _R.embed(texts)

    _cosine = staticmethod(_R._cosine)
    _segment_text = staticmethod(_R._segment_text)

    def retrieve(self, query: str, segments: list, top_k: int = 5) -> dict:
        """按语义相似度从一批候选片段里检索与 query 最相关的 top_k 个（不落库）。"""
        segments = [s for s in (segments or []) if isinstance(s, dict)]
        if not query or not segments:
            return {"query": query, "matches": [], "count": 0}
        seg_texts = [_R._segment_text(s) for s in segments]
        try:
            vectors = self.embed([query] + seg_texts)
        except RuntimeError as exc:
            _log.warning("retrieval embed failed: %s", exc)
            return {"query": query, "matches": [], "count": 0, "error": str(exc)}
        if len(vectors) != len(seg_texts) + 1:
            return {"query": query, "matches": [], "count": 0, "error": "embedding count mismatch"}
        qv, seg_vecs = vectors[0], vectors[1:]
        scored = []
        for seg, vec in zip(segments, seg_vecs):
            scored.append({
                "asset_id": seg.get("asset_id") or seg.get("global_asset_id", ""),
                "source_video_id": seg.get("source_video_id", ""),
                "source_time_range": seg.get("source_time_range", ""),
                "summary": seg.get("one_sentence_summary") or seg.get("visual_description", ""),
                "score": round(_R._cosine(qv, vec), 4),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[: max(1, int(top_k or 5))]
        return {"query": query, "matches": top, "count": len(top), "candidate_count": len(segments)}
