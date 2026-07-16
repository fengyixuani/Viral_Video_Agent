"""镜头生成工具（当前为 MOCK 占位）。"""
from .base import MockTool


class GenerationTool(MockTool):
    name = "生成"

    def run(self, gen_prompt: str, duration: float, profile=None) -> dict:
        """按提示词生成一个镜头片段。

        Args:
            gen_prompt: 生成镜头的提示词。
            duration: 目标时长（秒）。
            profile: 行业 profile，可选。
        """
        prompt = gen_prompt or ""
        self._log(prompt)
        return {
            "uri": "mock://generated/placeholder.mp4",
            "duration": float(duration or 0.0),
            "prompt": prompt,
            "engine": "seedance-mock",
        }
