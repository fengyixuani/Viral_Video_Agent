# Tool.md —— 各阶段 Agent 可用工具清单

工具用 AgentScope `FunctionTool(fn)` 声明：schema 由函数签名 + docstring 自动生成。
理解/编排的工具在 `src/shared/react_agents.py`（薄封装，业务落在 `src/tools/`）；
剪辑/AIGC 的工具在 `src/editing/`。下面按阶段列出**当前真实注册的工具**。

---

## A · 理解阶段（`understanding/`；Toolkit=`build_understanding_toolkit`）

`UNDERSTANDING_TOOLS = [parse_reference, assess_materials, transcribe_audio,
detect_shot_boundaries, detect_music_beats]`。工具规划 Agent
（`planner.run_understanding_planner`）按 `TOOL_CATALOG` 决定实际调用哪些。

- **parse_reference**(video_uri, video_desc, intent, materials, duration_sec) — 打包参考视频上下文供理解 LLM。
- **assess_materials**(materials) — 粗盘点用户素材（数量/时长）。
- **transcribe_audio**(media_path) — 语音识别转写（走独立 qwen3-asr 子进程环境）。
- **detect_shot_boundaries**(video_path, threshold=…) — ffmpeg 场景切分，`metadata.boundaries` 注入理解 prompt 对齐镜头。
- **detect_music_beats**(audio_path, …) — librosa 估计 BPM + beat 时间点，供卡点。

真实理解业务在 `src/tools/understanding.py`（`UnderstandingTool`：schema_hint /
analyze_messages / build_template / profile_materials）；探测工具在 `src/tools/media_probe.py`。

---

## B · 编排阶段（`orchestration/`；Toolkit=`build_planning_toolkit`）

`PLANNING_TOOLS = [plan_execute, decide_shot, generate_shot, edit_timeline,
package_video, detect_music_beats, retrieve_video_segments]`（生成/剪辑/包装为 MOCK 占位，
真实成片走 B 的 Agent 剪辑链路）。

- **plan_execute**(scheme_name, strategy, dimensions, trends, materials_count) — 产出 plan-execute 计划。
- **decide_shot**(slot_id, want, strategy) — 逐镜 match/generate 决策。
- **generate_shot / edit_timeline / package_video** — MOCK 占位（真实剪辑在 `editing/`）。
- 业务在 `src/tools/planning.py`。编排结果由 `orchestration/scriptgen.py` 导出 strategy JSON。

---

## B · 剪辑阶段 · 剪辑 Agent（`editing/tools.py`）

`EDIT_FUNCTION_TOOLS`（ReAct，剪辑 Agent 每次只输出一个动作）：

- **retrieve**(slot_id, query, top_k=6) — 从全量用户素材池按语义召回候选片段。
- **verify**(global_asset_id, question) — 对某片段按需 VLM 视觉核验。
- **place**(slot_id, global_asset_id, source_time_range="", target_duration=0, caption="", …) — 把某片段放入某 slot。
- **tts_clone**(slot_id, ref_global_asset_id, text) — 克隆音色为该镜配音（走 CosyVoice 子进程），台词需衔接上下文。
- **finish**() — 所有 slot 放好后结束本轮。

工具箱 `EditToolbox`（retrieve/verify）在 `editing/tools.py`；执行分发在
`editing/loop.py::_run_edit_agent`。审片 Agent（`gemini_review.py` / qwen）与仲裁
Agent（`arbiter.py`）不是 FunctionTool，而是 loop 内的评审/裁决步骤。

---

## B · 剪辑阶段 · AIGC 补镜 Agent（`editing/aigc.py`）

`AIGC_FUNCTION_TOOLS`（缺失镜头 seedream+seedance 生成，ReAct，≤5 子 Agent 并行）：

- **recall_product**(query, top_k=6) — 从用户素材池召回最能代表该商品的片段。
- **pick_product_frame**(global_asset_id, timestamp=-1) — 抽一帧作产品参考帧（i2i 用）。
- **gen_first_frame**(prompt, use_product_ref=False) — seedream 生成首帧（i2i 保真商品或 t2i）。
- **gen_storyboard**(motion_prompt) — 基于首帧生成脚本图指引运动（可选）。
- **gen_video**(prompt, duration_sec=5) — seedance 由首帧生视频并下载到本地。
- **review_and_extract**(start, dur, caption="") — VLM 回看生成视频并截取一段作最终片段。
- **finish**() — 该镜补齐后结束。

底层生成客户端 `src/tools/aigc_gen.py`（seedream `doubao-seedream-5-0` / seedance
`doubao-seedance-2-0`，走 wenchain `/incommonuserr`，i2i 支持 base64 参考图、免 BOS）。

---

## C · AI 短剧（`drama/`，非 FunctionTool）

短剧是确定性阶段编排，不走 ReAct 工具。媒体生成用 `src/tools/wenchain_media.py`
（seedream/seedance + ffmpeg 拼接），阶段实现在 `drama/{understand,script,storyboard,
video,verify}.py`。

---

## 新增工具的方法

- 理解/编排工具：在 `src/shared/react_agents.py` 写纯函数（签名+docstring 清晰），
  返回 `ToolResponse`，加入 `UNDERSTANDING_TOOLS` / `PLANNING_TOOLS`；业务落在 `src/tools/`。
- 剪辑/AIGC 工具：在 `src/editing/tools.py`（或 `aigc.py`）写函数加入 `EDIT_FUNCTION_TOOLS`
  / `AIGC_FUNCTION_TOOLS`，并在对应 ReAct 分发器里实现执行；schema 会自动渲染进 Agent prompt。
- 无需改前端 SSE 契约。
