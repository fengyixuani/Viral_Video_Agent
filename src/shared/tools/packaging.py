"""视频包装工具（字幕/TTS/BGM/导出，当前为 MOCK 占位）。"""
from schema import VideoFile

from .base import MockTool


class PackagingTool(MockTool):
    name = "包装"

    def run(self, shots: list, duration_sec: float, profile=None) -> VideoFile:
        """把最终镜头列表包装成成片。

        Args:
            shots: 最终镜头列表。
            duration_sec: 成片总时长（秒）。
            profile: 行业 profile，可选。
        """
        self._log("subtitle, TTS, BGM and export placeholders")
        return VideoFile(
            uri="mock://output/final.mp4",
            duration_sec=float(duration_sec or 0.0),
            shots=list(shots or []),
        )
