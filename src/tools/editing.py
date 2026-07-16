"""时间线剪辑工具（当前为 MOCK 占位）。"""
from .base import MockTool


class EditingTool(MockTool):
    name = "剪辑"

    def run(self, shots: list, profile=None) -> dict:
        """把镜头列表组装成时间线。

        Args:
            shots: 有序的镜头描述列表。
            profile: 行业 profile，可选。
        """
        self._log("assemble timeline")
        return {
            "timeline": [],
            "note": "剪辑能力未接入，返回空时间线",
            "shots": list(shots or []),
        }
