# MODELS.md —— 本项目使用的模型清单

按"部署方式"分两类：**走网关（无需本地安装）** 和 **本地 GPU 子环境（可选）**。
所有模型 ID / 路径都在 `config.env` 里可以覆盖（对应 env var 见每条最后的括号），
安装步骤见 `README.md §6.3`。

---

## 一、走 wenchain 网关（无需本地安装，主链路依赖）

统一入口：`WENCHAIN_BASE_URL` + `WENCHAIN_API_KEY`（channel）。默认已配好。
主链路（理解 / 编排 / Agent 剪辑主体 / AIGC 补镜 / AI 短剧）**只依赖这一段**。

- **通用 LLM（文本）**
  - 用途：编排/决策/规划、剪辑 Agent ReAct、审片、AIGC ReAct 等所有文本推理。
  - 模型：**`ali-qwen3.7-max`**（env `TEXT_LLM_MODEL`）
- **VLM（视觉理解）**
  - 用途：参考视频拆分镜、素材理解、可行性验证、审片（默认后端）、AIGC 回看截取。
  - 模型：**`ali-qwen3.7-plus`**（env `VISION_LLM_MODEL`）
- **T2I / i2i（首帧、故事板、卡通商品资产）**
  - 用途：AIGC 补镜首帧、短剧角色三视图/故事板/首帧/商品资产。
  - 模型：**`doubao-seedream-5-0-260128`**（env `AIGC_T2I_MODEL`）
  - 支持 base64 参考图 i2i，无需 BOS 上传。
- **T2V / i2v（视频生成）**
  - 用途：AIGC 补镜生成缺失镜头、短剧逐片段 i2v 生成。
  - 模型：**`doubao-seedance-2-0`**（env `AIGC_I2V_MODEL`）；单片段约几百秒。
- **Embedding（检索）**
  - 用途：素材片段/分镜的向量检索（RRF 融合 + 字符 2gram 词法兜底）。
  - 模型：**`qwen3-embedding-0.6b`**（env `EMBED_MODEL`），1024 维。
- **Gemini 审片（可选后端）**
  - 用途：Agent 剪辑的审片 Agent 备选后端（画面+声音），能查字幕/口播错位。
  - 模型：**`gemini-2.5-pro`**（env `GEMINI_REVIEW_MODEL`），走 oneapi-comate（env `ONEAPI_BASE_URL` / `ONEAPI_TOKEN`）。
  - 前端 UI 里"审片模型" radio 可切；本项目默认勾选 Gemini。

## 二、本地 GPU 子环境（可选功能）

**不装也能跑主链路。** 装了才有：ASR 口播识别、CosyVoice 声音克隆（`tts_clone` 工具）。
在**独立 conda 环境**里跑，主 env 不装它们的依赖，避免污染。

- **Qwen3-ASR-0.6B**
  - 用途：语音识别，输出带时间戳（供审片判"口播是否截断"、给素材注入 ASR 文本、TTS 参考）。
  - 大小：约 0.6B 参数。
  - 存放：`${VIRAL_VIDEO_SPLIT_ROOT}/models/Qwen3-ASR-0.6B`（env `QWEN3_ASR_MODEL`）
  - 运行环境：conda env `qwen3-asr-cu128`（含 torch+CUDA），入口脚本
    `${VIRAL_VIDEO_SPLIT_ROOT}/common/vendor/Viral_Video/run_qwen3_asr_test.py`（env `ASR_SCRIPT`）
  - Python 二进制：env `ASR_PYTHON`。
- **Qwen3-ForcedAligner-0.6B**
  - 用途：口播强制时间对齐，让审片能按句级时间戳判"是否被剪断"。
  - 大小：约 0.6B 参数。
  - 存放：`${VIRAL_VIDEO_SPLIT_ROOT}/models/Qwen3-ForcedAligner-0.6B`（env `QWEN3_FORCED_ALIGNER`）
  - 与 Qwen3-ASR 共用同一 conda env。
- **CosyVoice3（推荐用 `CosyVoice-300M` 或 `CosyVoice2-0.5B`）**
  - 用途：zero-shot 声音克隆（Agent 剪辑的 `tts_clone` 工具，为空镜/AIGC 镜配音）。
  - 大小：300M 参数（或 0.5B）。
  - 存放：`${VIRAL_VIDEO_SPLIT_ROOT}/models/CosyVoice-300M`（env `TTS_MODEL_DIR`）
  - 运行环境：独立 conda env（例如 `cosyvoice`），入口脚本
    `${VIRAL_VIDEO_SPLIT_ROOT}/generation/run_cosyvoice3_zero_shot.py`（env `TTS_SCRIPT`）
  - Python 二进制：env `TTS_PYTHON`；源码仓库：env `TTS_REPO`。

## 三、其它

- **librosa**（`requirements.txt` 可选依赖）：BGM 节拍检测（`detect_music_beats`），
  不是模型，纯数字信号处理；缺失会跳过卡点。
- **demucs**（可选、非默认）：仅在 Agent 剪辑「复用参考 BGM」且需要人声分离时用到；
  当前用它做 BGM/人声分离取 BGM 轨。可以不装，缺失时会退化为不分离直接用参考音轨。
- **ffmpeg**：由 `imageio-ffmpeg` 内置，无需单独装。

---

## 一句话决策

- 只用主链路？→ **只装 `pip install -r requirements.txt`**，模型全走网关，别的都不用管。
- 要 ASR / TTS？→ 按 `README.md §6.3` 建两个 conda 子 env、下三份模型，在 `config.env` 里
  把对应路径改为你的实际路径。
