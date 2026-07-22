"""AgentScope 运行时基类：理解(A)与编排(B)两个阶段共用的 LLM/工具基础设施。

`ReplicationAgentBase` 只放**通用**能力：
- 引用 `react_agents` 里的共享业务实例（understanding/planning/generation/editing/packaging）；
- 懒构建 AgentScope 的 understanding/planning Agent 与 Toolkit；
- `_llm_json` / `_run_tool` / `_last_metadata` 三个流式辅助。

理解阶段 `understanding.UnderstandingAgent`（analyze_stream）与编排阶段
`orchestration.OrchestrationAgent`（replicate_stream）分别继承它。放在 shared 是因为它不含
任何阶段专属逻辑，只是 AgentScope 的通用胶水。
"""
import json
import uuid

import as_core  # noqa: F401  (保持与旧模块一致的依赖可见性)
import react_agents
from agentscope.message import ToolCallBlock
from agentscope.state import AgentState


class ReplicationAgentBase:
    """理解/编排两阶段共用的 AgentScope 运行时基类。"""

    def __init__(self):
        # 直接引用 react_agents 中的共享业务实例，保证 orchestrator 与
        # FunctionTool 薄封装走同一份业务实现。
        self.understanding = react_agents.UNDERSTANDING
        self.planning = react_agents.PLANNING
        self.generation = react_agents.GENERATION
        self.editing = react_agents.EDITING
        self.packaging = react_agents.PACKAGING
        self._understanding_agent = None
        self._planning_agent = None
        self._understanding_toolkit = react_agents.build_understanding_toolkit()
        self._planning_toolkit = react_agents.build_planning_toolkit()

    @property
    def understanding_agent(self):
        if self._understanding_agent is None:
            self._understanding_agent = react_agents.build_understanding_agent()
        return self._understanding_agent

    @property
    def planning_agent(self):
        if self._planning_agent is None:
            self._planning_agent = react_agents.build_planning_agent()
        return self._planning_agent

    async def _llm_json(self, phase, system, user, *, vision=False, media=None):
        content = ""
        async for item in as_core.stream(system, user, vision=vision, media=media):
            if item.get("reasoning"):
                yield {"type": "reasoning", "phase": phase, "text": item["reasoning"]}
            elif "content" in item:
                content = item["content"]
        try:
            data = as_core.parse_json(content) if content.strip() else {}
        except (ValueError, TypeError, json.JSONDecodeError):
            data = {}
        yield {"type": "__json__", "data": data}

    async def _run_tool(self, toolkit, name, phase, *, title=None, **payload):
        """执行 Toolkit 注册的 FunctionTool，并发出成对（running→done）step 事件。"""
        call_id = uuid.uuid4().hex[:12]
        label = title or name
        block = ToolCallBlock(id=call_id, name=name, input=json.dumps(payload, ensure_ascii=False))
        yield {"type": "step", "phase": phase, "key": call_id, "state": "running",
               "title": label, "thought": f"正在{label}"}
        state = AgentState()
        observation = ""
        async for chunk in toolkit.call_tool(block, state):
            content = getattr(chunk, "content", None) or []
            for entry in content:
                text = entry.get("text") if isinstance(entry, dict) else getattr(entry, "text", "")
                if text:
                    observation = text
        yield {"type": "step", "phase": phase, "key": call_id, "state": "done",
               "title": label, "thought": f"{label}完成", "observation": observation[:400]}

    async def _last_metadata(self, toolkit, name, payload):
        """再跑一次工具只为拿到 metadata（AgentScope call_tool 的 ToolResponse.metadata）。"""
        call_id = uuid.uuid4().hex[:12]
        block = ToolCallBlock(id=call_id, name=name, input=json.dumps(payload, ensure_ascii=False))
        state = AgentState()
        metadata = {}
        async for chunk in toolkit.call_tool(block, state):
            meta = getattr(chunk, "metadata", None)
            if meta:
                metadata = meta
        return metadata or {}
