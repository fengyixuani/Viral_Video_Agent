# Portable TTS_clone

这个目录把本项目的 `TTS_clone` 能力整理成可迁移组件：从素材池选择稳定的参考音片段，截取并转换为 16 kHz 单声道 prompt，调用外部零样本 TTS 后端生成 WAV，再做时长读取和响度归一。

它不依赖当前项目的 Web 服务、AgentScope、`obs` 或完整剪辑循环。调用方只需要提供素材片段 JSON 和一个兼容 CLI 的 TTS 后端。

## 目录

- `reference_selector.py`：参考音选择逻辑。
- `synthesizer.py`：参考音抽取、prompt JSON、TTS 子进程、响度归一。
- `constraints.py`：画面时长/字数/重复文案等可选检查。
- `cli.py`：一次选择并生成一条 WAV 的命令行入口。
- `.env.example`：配置模板。
- `requirements.txt`：迁移包依赖。

## 参考音选择规则

`pick_voice_reference()` 与原项目 `src/editing/loop.py::_pick_voice_ref` 对齐：

1. 遍历全部素材片段。
2. 取 `speech_or_text`，没有时取 `whq_speech.text`。
3. 过滤：文本至少 8 个字符、片段时长 1.5～20 秒、有 `source_path`。
4. 按 ASR 文本长度降序排列，只保留前 8 条。
5. 交给文本 LLM 从前 8 条中选择“真人正常讲产品、语义通顺、转写准确”的片段，排除现场口令、乱码和明显噪声误识别。
6. LLM 返回有效序号就使用该片段；LLM 调用失败时使用最长候选。
7. LLM 返回 `best=0` 或没有候选时，返回调用方提供的 `reference_video` 兜底对象（没有兜底则返回空）。

这里的 LLM 只看 ASR 文本，不能真正听出音频杂音。若要检测离机位低音量、底噪或人声质量，调用方需要另接音频检测器。

默认选择模型是 `ali-qwen3.7-max`，可用 `TEXT_LLM_MODEL` 或 `LLM_MODEL` 覆盖。网关使用 OpenAI 兼容的 `/chat/completions` 接口，`WENCHAIN_BASE_URL` 应填写你的实际网关地址。

## TTS 后端契约

`clone()` 调用外部脚本时传入：

```text
--prompt-wav  临时生成的 16 kHz 单声道参考音
--prompt-asr  {"results":[{"text":"..."}]} JSON
--text        要合成的新文案
--output      输出 WAV
```

当前项目的 VoxCPM2 后端脚本是 [src/editing/whq_clone/run_voxcpm2_zero_shot.py](../src/editing/whq_clone/run_voxcpm2_zero_shot.py)。迁移到其他项目时可以复制该脚本，或者使用任何实现相同参数契约的 CosyVoice3/VoxCPM2 包装脚本。

`AGENT_TTS_PROMPT_MODE=basic` 是当前推荐值：prompt JSON 中不放参考原话，只用参考音波形提取音色，输出集中在目标 `text`。`ultimate` 会同时传入参考音转写，音色相似度可能更高，但可能重复念参考原话或吞掉目标文案开头。

## 安装环境

### 1. 调用方环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r tts_clone_portable/requirements.txt
```

`imageio-ffmpeg` 自带 ffmpeg。若不使用它，也可以安装系统 ffmpeg，并设置 `FFMPEG=/absolute/path/to/ffmpeg`。

### 2. VoxCPM2 推理环境

建议与调用方环境隔离。当前已验证的组合是：

- Python 3.11
- `voxcpm==2.0.3`
- `torch==2.11.0+cu128`
- `torchaudio==2.11.0+cu128`
- `soundfile==0.14.0`
- NVIDIA GPU 和匹配的 CUDA 驱动；当前项目验证设备是 L20

示例：

```bash
conda create -n voxcpm python=3.11 -y
conda activate voxcpm
python -m pip install voxcpm==2.0.3 soundfile
python -m pip install --force-reinstall torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

如果你的驱动/CUDA 版本不同，应选择对应的 PyTorch wheel，不能直接照搬 `cu128`。

### 3. 下载模型

当前 VoxCPM2 后端首次运行时会从 Hugging Face 下载：

```text
openbmb/VoxCPM2
```

模型主页：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)

脚本通过 `VOXCPM_MODEL_ID` 覆盖模型标识，默认值为 `openbmb/VoxCPM2`。网络受限时，可在运行环境中设置 Hugging Face 镜像或提前下载到本地后把 `VOXCPM_MODEL_ID` 指向本地目录。

当前后端使用 `load_denoiser=False`，不需要额外下载 zipenhancer 去噪器；这是为了避免部分 CUDA/torchcodec 环境冲突。

### 4. 可选：素材 ASR 环境

参考音选择依赖素材已有 ASR 文本。如果没有 ASR JSON，需要先用自己的 ASR 生成 `speech_or_text`。当前项目使用 Qwen3-ASR：

- `Qwen3-ASR-0.6B`
- `Qwen3-ForcedAligner-0.6B`

它们不是 TTS 模型，不能替代 VoxCPM2。当前项目的本地 ASR 适配可参考 [src/shared/tools/asr.py](../src/shared/tools/asr.py)；ASR 运行通常需要独立 CUDA 环境。

## 配置

复制模板并填写：

```bash
cp tts_clone_portable/.env.example .env
```

最少需要：

```bash
export AGENT_TTS_PYTHON=/path/to/voxcpm/bin/python
export AGENT_TTS_SCRIPT=/path/to/run_voxcpm2_zero_shot.py
```

如果需要 LLM 自动从 Top 8 中挑选参考音，还需要：

```bash
export WENCHAIN_BASE_URL=http://your-real-gateway/v1
export WENCHAIN_API_KEY=your-real-key
export TEXT_LLM_MODEL=ali-qwen3.7-max
```

若不配置 LLM，传入单个候选或使用测试时显式注入 `llm`；CLI 在多个候选时会尝试网关，网关不可用时回退最长候选。

## 输入格式

`segments.json` 可以是数组，也可以是 `{ "segments": [...] }`：

```json
[
  {
    "global_asset_id": "asset_001",
    "source_path": "/data/materials/a.mp4",
    "source_time_range": "12.00-16.00",
    "speech_or_text": "这款产品日常使用非常方便"
  }
]
```

## 使用

```bash
PYTHONPATH=. python -m tts_clone_portable.cli \
  --segments segments.json \
  --product-name 商品名 \
  --text "这款产品日常使用非常方便" \
  --output output/tts_S01.wav \
  --reference-output output/tts_S01_reference.json
```

成功时输出包含：

```json
{
  "ok": true,
  "output": "/absolute/path/output/tts_S01.wav",
  "duration": 2.4,
  "backend": "run_voxcpm2_zero_shot.py",
  "reference": {
    "global_asset_id": "asset_001",
    "source_path": "/data/materials/a.mp4",
    "source_time_range": "12.00-16.00",
    "speech": "这款产品日常使用非常方便"
  }
}
```

## 与原项目的对应关系

- 参考音选择：[src/editing/loop.py:1059-1107](../src/editing/loop.py#L1059-L1107)
- `tts_clone` 执行与约束：[src/editing/loop.py:1110-1288](../src/editing/loop.py#L1110-L1288)
- 参考音截取、16 kHz prompt、外部 CLI：[src/editing/tts.py:86-102](../src/editing/tts.py#L86-L102) 和 [src/editing/tts.py:189-236](../src/editing/tts.py#L189-L236)
- VoxCPM2 后端实现：[src/editing/whq_clone/run_voxcpm2_zero_shot.py](../src/editing/whq_clone/run_voxcpm2_zero_shot.py)
- VoxCPM2 环境和调参记录：[src/editing/whq_clone/VOXCPM2_INTEGRATION.md](../src/editing/whq_clone/VOXCPM2_INTEGRATION.md)
