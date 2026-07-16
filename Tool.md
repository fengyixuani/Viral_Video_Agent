# Tool.md — 当前 Agent 可用的工具清单

> 定义位置：`src/agent/react_agents.py`（**薄封装**）+ `src/tools/`（**业务实现**）
> 注册方式：`FunctionTool(fn)` 自动从函数签名与 docstring 生成 schema
> 调用方式：`Toolkit.call_tool(...)`，返回 `ToolResponse(content=[TextBlock])`
> 分层原则：`react_agents.py` 的每个 FunctionTool 只做 3~5 行的薄封装——把入参交给
> `src/tools/` 下对应的业务类方法（`UNDERSTANDING` / `PLANNING` / `GENERATION` /
> `EDITING` / `PACKAGING` 共享实例），再用 `_text_response` 包装成 `ToolResponse`。
> 真实/MOCK 业务只允许出现在 `src/tools/` 里，orchestrator 与 FunctionTool 都不得
> 内嵌业务实现。

## FunctionTool → 业务方法映射

| FunctionTool（`react_agents.py`） | 对应业务方法（`src/tools/`） |
| --- | --- |
| `parse_reference` | `UnderstandingTool.schema_hint()`（附带 schema 提示，回显 payload） |
| `assess_materials` | `UnderstandingTool.profile_materials(materials)` |
| `plan_execute` | `PlanningTool.plan_payload(scheme_name, strategy, dimensions, trends, materials_count)` |
| `decide_shot` | `PlanningTool.decide_payload(slot_id, want, strategy)` |
| `generate_shot` | `GenerationTool.run(gen_prompt, duration, profile=None)` |
| `edit_timeline` | `EditingTool.run(shots, profile=None)` |
| `package_video` | `PackagingTool.run(shots, duration_sec, profile=None)` |

Orchestrator 侧（`src/agent/orchestrator.py`）复用同一批共享实例：
`self.understanding = react_agents.UNDERSTANDING` 等等，因此 `analyze_messages`、
`build_template`、`profile_materials` 与 `generation.run` / `editing.run` /
`packaging.run` 全部走业务层，避免任何业务副本。

---

## 1. UnderstandingAgent — 视觉理解（模型 `ali-qwen3.7-plus`）

Toolkit 构造：`build_understanding_toolkit()`

### 1.1 `parse_reference`

- 用途：把参考视频 + 描述 + 意图 + 用户素材 + 探测时长打包，供理解 LLM 拆解
- 参数
  - `video_uri: str` — 参考视频 URI/本地路径
  - `video_desc: str` — 视频描述（可选补充信息）
  - `intent: str` — 用户复刻意图
  - `materials: list` — 用户已提供的素材条目
  - `duration_sec: float` — ffmpeg 探测出的真实时长
- 返回：`ToolResponse` 内嵌回显 JSON
- 替换点：接入真实拆镜 / VLM 打标 / ASR / OCR 时，在函数内组装结构化观察结果

### 1.2 `assess_materials`

- 用途：对用户素材做粗盘点，供 LLM 判断哪些镜头可 match、哪些需 generate
- 参数：`materials: list`
- 返回：`{provided_count, materials}`
- 替换点：真实版应调用 shot_segment / vlm_tag / asr / ocr 生成 `MaterialProfile`

---

## 2. PlanningAgent — 规划与执行（模型 `ali-qwen3.7-max`）

Toolkit 构造：`build_planning_toolkit()`

### 2.1 `plan_execute`

- 用途：把复刻方案、策略、勾选维度、趋势、素材数打包给规划 LLM，产出 plan-execute 计划
- 参数
  - `scheme_name: str`
  - `strategy: str` — `faithful | balanced | regenerate`
  - `dimensions: list` — 用户勾选的可复刻维度
  - `trends: list` — 用户勾选的热门趋势短语
  - `materials_count: int`
- 返回：回显 JSON，LLM 据此产出 `{goal, granularity, reasoning, steps}`

### 2.2 `decide_shot`

- 用途：逐镜决策 `match` vs `generate`
- 参数
  - `slot_id: int`
  - `want: str` — 该镜头目标
  - `strategy: str`
- 返回：回显 JSON，LLM 输出 `{decisions:[{slot_id, action, reason}]}`

### 2.3 `generate_shot`（占位）

- 用途：镜头生成
- 参数
  - `gen_prompt: str`
  - `duration: float`
- 返回：`{gen_prompt, duration, uri: "mock://generated"}`
- 替换点：接入 Seedance / Wan 等 T2V 后返回真实 `video_url`、`duration`

### 2.4 `edit_timeline`（占位）

- 用途：把镜头列表组装成时间线
- 参数：`shots: list`
- 返回：`{shots, timeline: "mock://timeline"}`
- 替换点：接入剪辑引擎后返回真实 timeline JSON（含卡点、转场、素材切片）

### 2.5 `package_video`（占位）

- 用途：包装成片（字幕 / TTS / BGM / 转场 / 导出）
- 参数
  - `shots: list`
  - `duration: float`
- 返回：`{shots, duration, uri: "mock://output/final.mp4"}`
- 替换点：接入 TTS / 字幕烧录 / BGM 混音 / ffmpeg 导出后返回真实成片 URI

---

## 3. 结构化探测工具（UnderstandingAgent + PlanningAgent 共享）

来源：`src/tools/media_probe.py`，返回 AgentScope `ToolChunk`；人类可读摘要在 `content`，机器可读结果在 `metadata`。

### 3.1 `detect_shot_boundaries`

- 用途：用 ffmpeg 场景切分给出真实镜头切点
- 参数：`video_path: str`、`threshold=0.18`、`fallback_threshold=0.08`、`ffmpeg_bin=""`
- metadata：`{boundaries:[秒], boundary_count, threshold_used, fallback_used, ffmpeg_seconds}`
- Orchestrator 会把 `boundaries` 注入理解 prompt，使 `shot_slots` 与实际镜头对齐

### 3.2 `detect_music_beats`

- 用途：用 librosa 估计 BPM + beat 时间点（可选 onset），支持视频直接抽轨
- 参数：`audio_path: str`、`include_onsets=False`、`sample_rate=22050`、`hop_length=512`、`start_bpm=120.0`、`tightness=100.0`、`ffmpeg_bin=""`
- metadata：`{tempo_bpm, beats:[秒], beat_count, duration_seconds, sample_rate, hop_length, librosa_seconds}`；`include_onsets=true` 时附加 `onsets/onset_count`
- Orchestrator 会把 `tempo_bpm` 与 `beats_preview` 注入理解 prompt，帮助 LLM 卡点

---

## 4. 工具在两阶段中的调用顺序

阶段一（analyze_stream，UnderstandingAgent）

1. `parse_reference(video_uri, video_desc, intent, materials, duration_sec)`
2. `detect_shot_boundaries(video_path)`（本地视频时）
3. `detect_music_beats(audio_path)`（本地视频时）
4. `assess_materials(materials)`
5. LLM 输出结构化 JSON（`shot_slots` / `schemes` / `industry_guess` 等）

阶段二（replicate_stream，PlanningAgent）

1. `plan_execute(scheme_name, strategy, dimensions, trends, materials_count)`
2. 对每个 shot_slot 调 `decide_shot(slot_id, want, strategy)`
3. 对 `action=generate` 的镜头调 `generate_shot(gen_prompt, duration)`
4. `edit_timeline(shots)`
5. `package_video(shots, duration)`

每次工具调用都会生成两条 SSE `step`：`ToolCallStart` 与 `ToolResultEnd`（详见 `Agent.md`）。

---

## 4. 新增工具的方法

1. 在 `src/agent/react_agents.py` 里写一个纯函数，签名 + docstring 清晰（AgentScope 用它推断 schema）
2. 返回 `ToolResponse(content=[TextBlock(type="text", text=...)])`
3. 加入 `UNDERSTANDING_TOOLS` 或 `PLANNING_TOOLS`
4. 在 orchestrator 里通过 `toolkit.call_tool(tool_name, kwargs)` 调用；结果会自动出现在对应 SSE `step` 事件的 `observation` 字段
5. 无需修改前端契约

---

## 5. 未接入但预留的能力

- Operator 级：`shot_segment` / `vlm_tag` / `asr` / `ocr` / `emotion` / `scene_event`（`src/operators.py` 目前是 MOCK 类）
- Trend：`src/trends.py` 已经通过 `as_core.complete_json` 独立调用 LLM，与工具解耦，未来可再包一层 `fetch_trends_tool` 注册到 PlanningAgent
