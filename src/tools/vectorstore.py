"""向量库兼容层：保留 ``VectorStore`` 接口，内部转发到统一门面 ``Retriever``。

历史调用方（react_agents 的 FunctionTool、结构编排等）继续用 ``VectorStore``；
新代码建议直接用 ``tools.retriever.Retriever(task_id)``。
"""
from __future__ import annotations

from .retriever import STORE_DIR, Retriever, _segment_text  # noqa: F401  (兼容再导出)


class VectorStore:
    """薄兼容层：按素材片段字段建索引 + 语义检索，per-task 隔离。"""

    def __init__(self, task_id: str = ""):
        self._r = Retriever(task_id)

    def index_segments(self, segments: list, source: str = "") -> dict:
        return self._r.index_segments(segments, source=source)

    def search(self, query, top_k: int = 5) -> dict:
        """兼容旧签名：接受单条 query 字符串（或多条列表）。"""
        return self._r.search(query, top_k=top_k)

    def retrieve_for_shot(self, shot: dict, top_k: int = 5) -> dict:
        return self._r.retrieve_for_shot(shot, top_k=top_k)

    def size(self) -> int:
        return self._r.size()

    def clear(self):
        self._r.clear()
