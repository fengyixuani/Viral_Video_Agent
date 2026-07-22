"""编排包（B 负责：编排 → 剪辑 的编排端）。

- OrchestrationAgent.replicate_stream：消费 A 的 AnalysisResult（契约①）→ 产出 strategy JSON（契约②）。
- orchestration.py：结构优先 / 镜头优先两种编排函数。
- scriptgen.py：把编排结果导出成 Split 兼容脚本 + strategy JSON。
- prompts.py：编排/决策 system prompt。
"""
from .pipeline import OrchestrationAgent

__all__ = ["OrchestrationAgent"]
