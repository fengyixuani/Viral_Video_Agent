"""规划相关的业务实现。

`PlanningTool` 负责把规划阶段（plan-execute 与逐镜决策）的输入整理成结构化 payload，
供 `react_agents.py` 中的 FunctionTool 薄封装以及 orchestrator 复用。

方法本身不做 LLM 调用，只做数据整形；真正驱动 LLM 的仍是 orchestrator 里的
`_llm_json`，这里保留业务边界，方便后续接真实规划服务时替换。
"""
from __future__ import annotations

from typing import Any


class PlanningTool:
    name = "规划"

    def plan_payload(
        self,
        scheme_name: str,
        strategy: str,
        dimensions: list,
        trends: list,
        materials_count: int,
    ) -> dict:
        """整理 plan_execute 的结构化 payload。

        Args:
            scheme_name: 选中的复刻方案名称。
            strategy: 复刻策略，取值 faithful / balanced / regenerate。
            dimensions: 用户勾选的可复刻维度。
            trends: 用户勾选的热门趋势短语。
            materials_count: 用户提供的素材数量。
        """
        return {
            "scheme_name": scheme_name or "",
            "strategy": strategy or "",
            "dimensions": list(dimensions or []),
            "trends": list(trends or []),
            "materials_count": int(materials_count or 0),
        }

    def decide_payload(self, slot_id: int, want: str, strategy: str) -> dict:
        """整理逐镜决策的结构化 payload。

        Args:
            slot_id: 分镜 slot id。
            want: 该分镜的目标描述。
            strategy: 复刻策略。
        """
        return {
            "slot_id": int(slot_id or 0),
            "want": want or "",
            "strategy": strategy or "",
        }
