# Agent 层实现说明

本项目使用 **AgentScope 2.0.4** 承载 LLM 与 Agent 层。所有 LLM 调用都通过 wenchain
OpenAI-compatible 网关，AgentScope 仅提供 `OpenAIChatModel`、`Agent`、`Toolkit`、
`FunctionTool` 等原语。

## 版本与依赖

- `agentscope == 2.0.4`
- `python-frontmatter >= 1.1.0`

## 真实使用到的 AgentScope API

| API | 用途 |
| --- | --- |
| `agentscope.model.OpenAIChatModel(stream=True)` | 走 wenchain 网关的 OpenAI 协议流式模型 |
| `agentscope.credential.OpenAICredential` | 装 `api_key`/`base_url` |
| `agentscope.message.Msg` / `TextBlock` | 构造 system/user 输入 |
| `agentscope.message.ToolCallBlock` | 手动触发一个工具调用 |
| `agentscope.tool.Toolkit` | 组织每个 Agent 的工具集 |
| `agentscope.tool.FunctionTool` | 从 Python 函数自动推断 schema（依赖签名 + docstring） |
| `agentscope.tool.ToolResponse` | FunctionTool 的标准返回类型 |
| `agentscope.state.AgentState` | `Toolkit.call_tool` 需要传入的状态对象 |
| `agentscope.agent.Agent` | UnderstandingAgent / PlanningAgent 的基类 |

## 关键坑

- **`OpenAIChatModel(stream=True)` 返回的是 async generator**：每个 chunk 是
  `ChatResponse`，其 `.content` 是 `TextBlock`/`ThinkingBlock` 列表，**增量**
  给出；`ChatResponseBase` 会在最后再 yield 一个 `is_last=True` 的累计块，包含
  **完整正文**。所以 `as_core.stream` 里我们把非最后的 chunk 当作 reasoning
  转发（有 `thinking` 用 thinking，没有就把 text 增量当作 reasoning），并从最后
  一个 `is_last=True` 的 chunk 拿累计正文作为 `{"content": ...}`。
- **`Toolkit.get_tool_schemas()` 是 async**，测试里要 `asyncio.run` 才能拿到
  schema 列表。
- **`ToolCallBlock.input` 是字符串**，必须把 kwargs `json.dumps` 后传入。
- **`FunctionTool` 从函数签名 + docstring 推断 JSON schema**，所以工具函数
  的类型注解和 docstring 必须完整。
- 网关偶尔会拒绝 `video_url`/`image_url` 块，`as_core.stream` 在检测到这种错误
  信息（"unexpected item type in content" / "video_url" 等）时会自动回退成
  纯文本再调用一次。

## Agent 分工

| Agent | 模型 | 工具集 |
| --- | --- | --- |
| `UnderstandingAgent` | `ali-qwen3.7-plus`（视觉） | `parse_reference`, `assess_materials` |
| `PlanningAgent` | `ali-qwen3.7-max`（文本） | `plan_execute`, `decide_shot`, `generate_shot`, `edit_timeline`, `package_video` |

两个 Agent 都由 `src/agent/react_agents.py` 中的 `build_understanding_agent` /
`build_planning_agent` 构造。它们持有 `Toolkit` 与共享的 `OpenAIChatModel`
（`as_core._get_model` 单例，绑定 wenchain 网关）。

`ReplicationAgent`（`src/agent/orchestrator.py`）不直接调 `agent.reply`，而是：

1. 通过 `Toolkit.call_tool(ToolCallBlock, AgentState)` 触发工具执行，让 AgentScope
   官方管线负责调度、schema 校验和结果累计。
2. 结构化 JSON 决策仍由 `as_core.stream` 里的 `OpenAIChatModel` 直连产出（更容易
   保持 `reasoning/content` 契约）。

需要接真实 ReAct 循环时，改成 `await agent.reply(Msg(...))` 或
`async for msg in agent.reply_stream(...)` 即可。

## SSE 事件映射

| AgentScope 事件语义 | 我们发送的 SSE 事件 |
| --- | --- |
| `ToolCallStartEvent` | `{"type":"step","phase":..,"thought":"调用工具 X","action":"tool: X"}` |
| `ToolResultEndEvent` | `{"type":"step","phase":..,"thought":"工具 X 返回","observation":...}` |
| `ThinkingBlockDeltaEvent` / `TextBlockDeltaEvent` | `{"type":"reasoning","phase":..,"text":...}` |
| 累计 `is_last=True` chunk | 作为最终 JSON 内容驱动 `analysis` / `final` 事件 |

外部 SSE 输出契约不变：`reasoning` → `step` → `analysis` / `final` → `[DONE]`。

## 模型路由约定

- `as_core.pick_model(vision=True)` → `ali-qwen3.7-plus`
- `as_core.pick_model(vision=False)` → `ali-qwen3.7-max`

环境变量覆盖：`VISION_LLM_MODEL`、`TEXT_LLM_MODEL`、`WENCHAIN_BASE_URL`、
`WENCHAIN_API_KEY`（默认 `wangpantob_all_video_copy`）、`USE_WENCHAIN_OPENAI`、
`ALLOW_MOCK_LLM`。

## 接真实工具的替换点

`src/agent/react_agents.py` 里的每个 FunctionTool 现在都是**薄封装**（3~5 行），
真实/MOCK 业务全部落在 `src/tools/`：

- 理解层：`src/tools/understanding.py` 的 `UnderstandingTool.schema_hint /
  analyze_messages / build_template / profile_materials / _compat_breakdown`
- 规划层：`src/tools/planning.py` 的 `PlanningTool.plan_payload / decide_payload`
- 生成 / 剪辑 / 包装：`src/tools/generation.py`、`src/tools/editing.py`、
  `src/tools/packaging.py` 中对应类的 `run(...)` 方法

`react_agents` 顶部实例化了 `UNDERSTANDING / PLANNING / GENERATION / EDITING /
PACKAGING` 五个共享实例，orchestrator 与 FunctionTool 都通过这些实例调用业务，
接入真实能力时只替换 `src/tools/*.py` 的方法体，Toolkit 注册与 SSE 契约保持不变。

- `parse_reference` / `assess_materials` → `src/tools/understanding.py`
- `plan_execute` / `decide_shot` → `src/tools/planning.py`
- `generate_shot` / `edit_timeline` / `package_video` →
  `src/tools/generation.py` / `editing.py` / `packaging.py`

## SKILL 系统

保留原有 5 个 `skills_defs/*/SKILL.md`，`src/skills.py` 继续使用
`agentscope.skill.LocalSkillLoader`（在没有活跃事件循环时）+ frontmatter 手动
读取的 fallback，不做变更。
