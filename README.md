# 爆款视频复刻工作台（Viral Video Agent）

一个把「爆款短视频」拆解 → 复刻编排 → Agent 剪辑成片 的工作台，另含一条独立的
「AI 带货短剧复刻」链路。后端是 stdlib `ThreadingHTTPServer + SSE`，模型统一走内网 wenchain 网关。

本 README 的核心目的：**把整条链路按"流程段"拆成三个人可以并行开发、互不踩踏的结构**，
并固定各段之间的**数据契约**（见 [`docs/contracts.md`](docs/contracts.md)）。

---

## 一、整体架构：4+1 条"流程控制器"

每条链路都有一个**总控函数**（Python 控制流决定"每一步干什么、什么顺序、跑几轮"；
skill/prompt 只约束某一步里 Agent 内部怎么想）。点网页按钮 → 前端发 SSE 请求 →
`server/app.py` 路由 → 调对应总控函数：

- **理解** `understanding/orchestrator.py :: UnderstandingAgent.analyze_stream`（A）
  ← 「理解爆款」按钮 → `POST /api/analyze`
  工具规划 → 理解参考视频 → BGM 分析 → 用户素材理解 → 可行性验证/仲裁 → 抽缩略图；产出 **AnalysisResult（契约①）**
- **复刻编排** `orchestration/pipeline.py :: OrchestrationAgent.replicate_stream`（B）
  ← 「生成复刻方案」按钮 → `POST /api/replicate`
  吃 AnalysisResult → 按 reproduce_mode 选 `orchestrate_structure_first` / `orchestrate_from_feasibility` → 产出 **strategy JSON**
- **Agent 剪辑** `editing/loop.py :: agent_edit_stream`（B）
  ← 「Agent 剪辑」按钮 → `POST /api/agent_edit`
  （AIGC 补镜）→ 每轮「剪辑 Agent → 出片 → 审片 Agent → 增量重剪」，最多 N 轮
- **AI 短剧复刻** `drama/pipeline.py :: run_drama_replication`（C）
  ← 「生成 AI 短剧复刻」按钮 → `POST /api/drama_replicate`
  ①理解→②脚本→③三视图/故事板/首帧→④i2v+拼接→⑤验证→未达标迭代
- （旧·Legacy，先不管）**Split 剪辑** `shared/connector.py :: run_edit` ← 「剪辑合成」→ `POST /api/edit`
  遗留链路，已被 Agent 剪辑取代，当前不维护；代码暂保留。

通用 AgentScope 运行时（`_llm_json/_run_tool` 等）在 `shared/agent_runtime.py :: ReplicationAgentBase`，
理解(A)与编排(B)的 Agent 都继承它。

另有轻量路由决策 `POST /api/route`（`drama/router.route_reference`）：通用模式点「理解爆款」时
先判走电商还是短剧。

---

## 二、三人协作拆分

按"流程段"划分，靠**数据契约**解耦，不共享内部实现。

### A — 理解（产出 AnalysisResult）
负责"看懂爆款 + 理解用户素材 + 可行性验证"。**产物 = AnalysisResult（契约①）**。
- 拥有：
  - `src/understanding/`（`orchestrator.py::UnderstandingAgent.analyze_stream` + `planner / material_understanding / feasibility`）
  - `src/shared/tools/understanding.py`（VLM 理解工具，归属 A）
  - `server/routes_understanding.py`（/api/analyze、/api/route）
  - skills：`viral_reference_understanding / user_material_understanding / ecom_understanding / material_coarse_match / material_arbitration` + 各场景 skill
- 交付给 B：AnalysisResult（`template`(含 shot_slots) + `schemes` + `feasibility` + `material_understanding`），随 `/api/replicate` 请求体传入。

### B — 编排 + 剪辑（吃 AnalysisResult → strategy → 成片）
负责"把理解结果编排成复刻方案，并剪出成片"。**剪辑主职责 = Agent 剪辑**（`editing/loop.py`）。
- 拥有：
  - `src/orchestration/`（`pipeline.py::OrchestrationAgent.replicate_stream` + `orchestration.py` 两种编排 + `scriptgen.py` 导出 strategy + `prompts.py`）
  - `src/editing/`（`loop.py` Agent 剪辑总控 + `editor / tools / aigc / arbiter / gemini_review / tts / bgm_reuse / beats / caption_tool`）
  - `server/routes_orchestration.py`（/api/replicate）、`server/routes_editing.py`（/api/agent_edit、/api/debug/materials）
  - skills：`orchestration_script / orchestration_shot_replicate / agent_edit_* / aigc_generator`
- strategy JSON（契约②）现在是 **B 内部产物**（编排产出、Agent 剪辑消费），不再跨人交接。
- 复用 A 的两个「理解服务」函数 `understand_materials` / `verify_materials`（前端时序缺数据时的自愈重建）——B→A 的服务级依赖。
- **Legacy（先不管）**：旧的 Split 剪辑 `shared/connector.py::run_edit` + `/api/edit` 属遗留链路，当前不维护、不作为 B 的核心职责；Agent 剪辑取代它。代码暂保留。

### C — AI 短剧复刻（端到端独立）
几乎自成一体，只用共享底层。
- 拥有：`src/drama/`（`pipeline.py` 总控 + `understand / script / storyboard / video / verify / router / asr_util / run`）、`src/shared/tools/wenchain_media.py`、`skills_defs/drama_replication`、`server/routes_drama.py`（/api/drama_replicate）+ 前端 `🎬 AI短剧复刻` tab。

### 共享层（`src/shared/`，三人共用，改动需评审）
`as_core.py`（模型网关）、`agent_runtime.py`（AgentScope 运行时基类）、`react_agents.py`（Agent/Toolkit 构建）、
`obs / cache / asr_cache / skills / schema / serialize / thumbs / bgm / operators / profile / projects / trends`、
`tools/`（retriever / vlm / asr / media_probe / aigc_gen / wenchain_media / understanding …）、`inputs/`。
`src/shared/` 在 `sys.path` 上，故这些模块仍是扁平 import。

---

## 三、两个数据契约（解耦的关键）

详见 [`docs/contracts.md`](docs/contracts.md) 与 `src/shared/contracts.py`（校验函数）。要点：

- **契约①（A→B：理解→编排，现在的跨人边界）** = AnalysisResult：`template.shot_slots`(含 `source_time_range`) + `schemes` + `feasibility` + `material_understanding`。随 `/api/replicate` 请求体从 A 传给 B；B 在 `replicate_stream` 入口用 `contracts.validate_analysis_result` 自检。
- **契约②（B 内部：编排→剪辑）** = `*_selected_editing_strategy_*.json`（`metadata / user_asset_bank / slot_matching / editing_timeline / missing_assets`）+ `connector_context.json`。编排(scriptgen)产出后、剪辑(loop)消费前各用 `contracts.validate_strategy` 自检。

**规则：契约字段冻结，改字段必须改 `contracts.py` + `docs/contracts.md` + `CONTRACT_VERSION`。**

---

## 四、独立开发 / 测试（不互相等）

每段都能脱离网页、用 CLI + fixtures 单独跑：

- **A**：固定一组参考视频+素材，跑到"产出 strategy JSON"即可；把产物存进 `tests/fixtures/strategy_*.json` 当交给 B 的样例。
- **B**：直接吃 A 冻结的 `strategy_*.json + connector_context.json`，跑 `agent_edit_stream`（当前只有网页/`/api/agent_edit` 入口，**建议 B 首个小任务：补一个 `python -m editing.run <strategy.json>` CLI**），**不用重跑理解**。
- **C**：`PYTHONPATH=src python src/drama/run.py <video> --iters N [--product-* ...]`，完全独立。

约定：每人用独立 `task_id` / 输出前缀 / 各自端口，避免踩 `uploads/`、`outputs/`、单卡 GPU（ASR/TTS 串行）、网关 QPM。

---

## 五、冲突热点

1. ✅ `server/app.py` 路由**已去巨石化**：各模块在 `server/routes_{common,understanding,editing,drama}.py`
   里各自导出 `GET`/`POST` 路由表（`{路径: 处理函数(handler)}`），`app.py` 只聚合分发 + 保留
   跨模块基础设施（JSON/文件响应、上传解析、SSE 发送、路径解析）。路径/`sys.path` 在 `server/_paths.py`。
   - A 改理解路由只动 `routes_understanding.py`，B/C 各动自己的 `routes_editing.py` / `routes_drama.py`。
2. ⏳ `server/static/index.html` 仍是单文件巨石（HTML+JS）→ 待拆 `static/{understanding,editing,drama}.js` + 骨架 include。

归属见 `CODEOWNERS`。

---

## 六、快速开始（新用户从这里）

### 6.1 前置条件
- **Python 3.10+**（`asyncio`/`from __future__ import annotations` + PEP 604 类型）。
- **内网可达**：所有 LLM/VLM/seedream/seedance 都走内网 wenchain 网关，公网机器跑不了。
- **ffmpeg**：由 `imageio-ffmpeg` 内置，无需系统装。
- **GPU 可选**：主链路（理解/编排/Agent 剪辑主体/AIGC 补镜/AI 短剧）**不需要本地 GPU 或本地大模型**——生成/理解都走网关；只有 ASR（口播识别）和 TTS（声音克隆）需要 GPU，且**是可选功能**（缺失优雅降级，见 6.3）。

### 6.2 最小可运行（核心链路，5 分钟）
```bash
git clone <repo> Viral_Video_Agent && cd Viral_Video_Agent
python -m venv .venv && source .venv/bin/activate      # 或 conda 也行
pip install -r requirements.txt
# 可选：编辑 config.env 覆盖端口/网关/路径（缺省一切都有默认值）
python main.py                                         # PORT 默认 8000，被占用会自动+1
# 浏览器打开 http://<host>:8000
```

**所有可配项集中在项目根的 [`config.env`](config.env) 一个文件**——启动时
`server/_paths.py` 会自动读它，设进 `os.environ`（已 `export` 的环境变量优先）。想改端口 /
网关 / 模型 / ASR / TTS 路径，**只改这一份**就够。模型清单见 [`MODELS.md`](MODELS.md)。

至此可用：
- 「理解爆款」（视觉理解、素材理解、可行性验证、缩略图）
- 「生成复刻方案」（编排产出 strategy JSON）
- 「Agent 剪辑」（ReAct 剪辑+审片+仲裁；**默认不烧字幕、不启审片**——想审片就勾上，默认走 Gemini）
- 「AIGC 补镜」（seedream+seedance 生成缺失镜头）
- 「AI 短剧复刻」（🎬 tab）
- 顶栏「运行记录」「素材调试」（多 tab 查看各阶段中间产物）

`outputs/index.json`+`outputs/{时间戳}_{类型}_{项目}_{rid}/` 会自动记录每次运行。

### 6.3 可选扩展：ASR（口播识别）+ TTS（声音克隆）

**用来做什么**：ASR 让审片能判定"口播是否被截断"、给素材注入 ASR 文本供剪辑参考；TTS 是 Agent 剪辑给空镜/AIGC 镜的克隆配音工具（`tts_clone` 工具）。**不装也能跑主链路**（会自动跳过 tts_clone、审片改为只看画面）。

**模型清单** 见 [`MODELS.md`](MODELS.md)（Qwen3-ASR-0.6B / Qwen3-ForcedAligner-0.6B / CosyVoice-300M）。

**配置方式**：**只改项目根的 [`config.env`](config.env)** 一个文件——把里面 ASR/TTS 段的
`#` 去掉、路径换成你机器上的实际路径即可。启动时 `server/_paths.py` 会自动加载它，
所有下游读 `os.getenv(...)` 都能拿到（已 `export` 的环境变量优先，兼容 CI）。

需要 GPU + CUDA，用**独立 conda 子环境**跑（避免污染主 env）。推荐目录布局：
```
<parent>/
  Viral_Video_Agent/            # 本项目
  Viral_Video_Split/
    common/vendor/Viral_Video/run_qwen3_asr_test.py   # ASR 调用脚本
    generation/run_cosyvoice3_zero_shot.py            # TTS 调用脚本
    models/Qwen3-ASR-0.6B/
    models/Qwen3-ForcedAligner-0.6B/
    models/CosyVoice-300M/
```
`config.env` 里对应改：
```
VIRAL_VIDEO_SPLIT_ROOT=/absolute/path/to/Viral_Video_Split
ASR_PYTHON=/absolute/path/to/miniconda/envs/qwen3-asr-cu128/bin/python
# ASR_SCRIPT / QWEN3_ASR_MODEL / QWEN3_FORCED_ALIGNER 用 ${VIRAL_VIDEO_SPLIT_ROOT} 引用即可（模板已给好）
TTS_PYTHON=/absolute/path/to/miniconda/envs/cosyvoice/bin/python
# TTS_SCRIPT / TTS_MODEL_DIR / TTS_REPO 同理
```

建 conda 子环境（示例，具体依赖以 Qwen3-ASR / CosyVoice 官方 README 为准）：
```bash
conda create -n qwen3-asr-cu128 python=3.11 -y
conda activate qwen3-asr-cu128
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate soundfile librosa

conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
# 按 CosyVoice 官方仓库 requirements 装
```
子环境**不要**继承主 env 的 `PYTHONPATH`——本项目在启动 ASR/TTS 子进程前会自动 `pop` 掉
`PYTHONPATH/PYTHONHOME/PYTHONSTARTUP`（`src/shared/tools/asr.py`、`src/editing/tts.py`），避免 site-packages 串包。

### 6.4 常用环境变量速查（完整清单见 [`config.env`](config.env)）
- `PORT` / `HOST`：默认 8000 / 0.0.0.0；`PORT` 被占用自动 +1 并在命令行提示。
- `WENCHAIN_BASE_URL` / `WENCHAIN_API_KEY`：网关和通道。
- `VISION_LLM_MODEL` / `TEXT_LLM_MODEL`：默认 `ali-qwen3.7-plus` / `ali-qwen3.7-max`。
- `AIGC_T2I_MODEL` / `AIGC_I2V_MODEL` / `AIGC_ASPECT` / `AIGC_MAX_SUBAGENTS`（默认 5）。
- `VIRAL_VIDEO_SPLIT_ROOT` + `ASR_*` / `TTS_*` / `QWEN3_*`：见 6.3、`config.env`、`MODELS.md`。
- `ALLOW_MOCK_LLM=1`：网络不可达时走 MOCK（仅调试）。
- `RAG_INCLUDE_SPEECH=1`：检索意图 embedding 中带上口播文本。

### 6.5 常见问题
- 起服务提示 `[端口检查] 8000 端口被占用，自动改为 8001 端口`：正常，自动换端口。
- Agent 剪辑跑到 `tts_clone` 返回 "CosyVoice 未配置"：没装 TTS 子环境；把 tts 关掉，或按 6.3 配置。
- 审片"要求配音"但你没有 TTS：改用 Gemini 审片（默认），或临时把审片关掉。
- 「运行记录」全 missing：那次运行素材池为空（`materials=0` 或 `material_understanding` 未回填）——先「理解爆款」把素材理解跑完再复刻。
- 短剧生成很慢：seedance 单片段约几百秒，正常；进度看底部「Agent 思考过程」或「运行记录」。
ffmpeg 由 `imageio_ffmpeg` 提供，无需系统安装。

---

## 七、目录结构（已按三人拆分）

```
src/
  shared/          # 共享层（加入 sys.path，扁平 import）
    as_core, agent_runtime, react_agents, obs, cache, asr_cache, skills, schema,
    serialize, thumbs, bgm, operators, profile, projects, trends, connector, contracts,
    tools/  (retriever, vlm, asr, media_probe, aigc_gen, wenchain_media, understanding, ...)
    inputs/
  understanding/   # A：理解（analyze_stream）
    orchestrator(UnderstandingAgent), planner, material_understanding, feasibility
  orchestration/   # B：编排（replicate_stream）
    pipeline(OrchestrationAgent), orchestration, scriptgen, prompts
  editing/         # B：剪辑
    loop, tools, editor, aigc, arbiter, gemini_review, tts, bgm_reuse, beats, caption_tool
  drama/           # C：AI 短剧
    pipeline, understand, script, storyboard, video, verify, router, asr_util, run
server/
  _paths.py, app.py（路由聚合分发）
  routes_common.py / routes_understanding.py(A) / routes_orchestration.py(B) /
  routes_editing.py(B) / routes_drama.py(C)
```

路径约定：`src/` 与 `src/shared/` 都在 `sys.path` 上——owned 包按名 import
（`from understanding import ...` / `from orchestration import ...` / `from editing import ...` / `from drama import ...`），
共享模块扁平 import（`import as_core` / `from tools import ...`）。

尚未做（可选后续）：把 `server/static/index.html` 拆成 `understanding.js / editing.js / drama.js`。
