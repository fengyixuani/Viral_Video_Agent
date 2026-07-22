# Agent 层实现说明

本项目基于 **AgentScope 2.0.4** 承载 LLM/Agent。所有 LLM/VLM 调用都走内网 wenchain
OpenAI 兼容网关；AgentScope 仅提供 `OpenAIChatModel` / `Agent` / `Toolkit` /
`FunctionTool` / `AgentState` 等原语。代码已按三人协作拆分（见 `README.md`）：
理解(A · `understanding/`) → 编排(B · `orchestration/`) → 剪辑(B · `editing/`)，
以及独立的 AI 短剧(C · `drama/`)。

## 依赖与共享运行时

- `agentscope==2.0.4`、`python-frontmatter`（见 `requirements.txt`）。
- 通用 AgentScope 胶水集中在 **`src/shared/agent_runtime.py::ReplicationAgentBase`**：
  - `__init__` 引用 `shared/react_agents.py` 的共享业务实例（UNDERSTANDING/PLANNING/
    GENERATION/EDITING/PACKAGING）并懒构建 understanding/planning 的 Agent+Toolkit；
  - `_llm_json(phase, system, user, *, vision, media)`：流式调 LLM，把增量当 reasoning
    转发，末尾解析 JSON；
  - `_run_tool(toolkit, name, phase, **payload)`：执行 FunctionTool 并发成对 step 事件；
  - `_last_metadata(...)`：取 `ToolResponse.metadata`。
- 理解阶段 `UnderstandingAgent`、编排阶段 `OrchestrationAgent` 都继承它。

## 各阶段的 Agent

### A · 理解（`src/understanding/`）
- **工具规划 Agent** `planner.py::run_understanding_planner`：ReAct 方式从 `TOOL_CATALOG`
  （`detect_shot_boundaries` / `detect_music_beats` / `transcribe_audio` / `assess_materials`）
  里决定本次要调用哪些结构化工具，受 `prompts` 之外的 planner 逻辑约束。
- **视觉理解** `orchestrator.py::UnderstandingAgent.analyze_stream`（模型 `ali-qwen3.7-plus`）：
  看参考视频、拆分镜（含 `source_time_range` 真实时间轴）、出方案，理解用户素材，
  做可行性验证 + 仲裁。产出 **AnalysisResult（契约①）**。

### B · 编排（`src/orchestration/`）
- **编排 Agent** `pipeline.py::OrchestrationAgent.replicate_stream`（规划/决策走 `ali-qwen3.7-max`）：
  吃 AnalysisResult，用 `PLAN_SYSTEM`/`DECIDE_SYSTEM` 做规划+逐镜决策，按 reproduce_mode
  调 `orchestration.py` 的 `orchestrate_structure_first` / `orchestrate_from_feasibility`，
  最后 `scriptgen.export_scripts` 落盘 **strategy JSON（契约②）**。

### B · 剪辑（`src/editing/`）
- **剪辑 Agent** `loop.py::_run_edit_agent`：真正的 ReAct 循环，工具见 `Tool.md`
  （retrieve / verify / place / finish / tts_clone），可从全量素材池自由召回。
- **审片 Agent** `loop.py` + `gemini_review.py`：看成片对照 DNA 打分/提问题，
  后端可选 `qwen`（画面）或 `gemini`（画面+声音）；不通过则增量重剪（默认最多 N 轮）。
- **仲裁 Agent** `arbiter.py`：多镜争抢同一素材/时间重叠时裁决归属。
- **AIGC 补镜 Agent** `aigc.py::generate_missing_shots`（内部每镜 `_run_aigc_slot` 跑 ReAct）：
  缺失镜头用 seedream+seedance 生成，工具见 `Tool.md`，按镜头数并行、上限 5 个子 Agent。

### C · AI 短剧（`src/drama/`）
`pipeline.py::run_drama_replication` 分阶段串联：①理解 `understand.py` → ②脚本
`script.py` → ③三视图/故事板/首帧 `storyboard.py` → ④分片段 i2v + 拼接 `video.py`
→ ⑤ASR+qwen 验证 `verify.py`，未达标迭代。VLM/LLM 直调（`as_core` + `tools/wenchain_media`），
不是 FunctionTool ReAct，而是确定性阶段编排。

## 关键坑（AgentScope）

- `OpenAIChatModel(stream=True)` 返回 async generator：chunk 增量给出，末尾一个
  `is_last=True` 的累计块含完整正文。`as_core.stream` 据此把非最后 chunk 当 reasoning、
  末块当 `{"content": ...}`。
- `Toolkit.get_tool_schemas()` 是 async；`ToolCallBlock.input` 必须是 `json.dumps` 后的字符串。
- `FunctionTool` 从函数签名 + docstring 推断 JSON schema，故工具函数的类型注解与
  docstring 必须完整。
- 网关偶尔拒绝 `video_url`/`image_url` 块，`as_core.stream` 检测到相关报错会自动回退纯文本重试。

## 模型路由

- `as_core.pick_model(vision=True)` → `ali-qwen3.7-plus`（视觉理解、审片、AIGC 回看）
- `as_core.pick_model(vision=False)` → `ali-qwen3.7-max`（规划/决策/编排等文本）
- 环境变量覆盖：`VISION_LLM_MODEL` / `TEXT_LLM_MODEL` / `WENCHAIN_BASE_URL` /
  `WENCHAIN_API_KEY`（默认 `wangpantob_all_video_copy`）/ `USE_WENCHAIN_OPENAI` / `ALLOW_MOCK_LLM`。

## SSE 事件契约

各 stream（`analyze_stream` / `replicate_stream` / `agent_edit_stream` /
`run_drama_replication`）统一产出：`reasoning` → `step`(running→done) →
`analysis`/`materials`/`feasibility`/`bgm`（理解）或 `final`（编排）或
`agent_edit_sample`/`agent_edit_done`（剪辑）或 `data`/`final`（短剧），最后 `[DONE]`。
前端按这些 `type` 字段对接。

## Skill 约束

各 Agent 的行为由 `skills_defs/*/SKILL.md` 约束（`src/shared/skills.py` 加载，
frontmatter + prompt_hint）：
- 理解：`viral_reference_understanding` / `user_material_understanding` / `ecom_understanding` / 场景 skill / `material_coarse_match` / `material_arbitration`
- 编排：`orchestration_script` / `orchestration_shot_replicate`
- 剪辑：`agent_edit_editor` / `agent_edit_reviewer` / `agent_edit_arbiter` / `aigc_generator`
- 短剧：`drama_replication`（场景识别/路由；生成 prompt 直接写在 `drama/*.py` 里）
