# whq × VoxCPM2 声音克隆接入（可复现存档）

把 whq 配音阶段的克隆模型从 **CosyVoice3** 换成 **VoxCPM2**，作为同 CLI 的可选后端。
本文记录**当前最佳配置**（产出 `..._whq_clone_voxcpm_same.mp4` / `_voiced.mp4`，实测最干净无杂音）、
改了哪些文件、如何复现，以及调参过程中的关键结论与坑。

## 最佳配置（结论）

> **VoxCPM2 + ultimate cloning + 流水线原生 16k「干净」prompt + cfg=2.0 + steps=30 + tts_clean_wrap 净化**

核心洞察：**参考音"干净"比"采样率高"更重要**。
- 流水线 16k `prompt.wav` 虽窄带（8kHz 封顶、略闷），但已经过降噪+高通+响度归一，**无伪影**，
  配合 `tts_clean_wrap` 的 clarity EQ 提 presence，整体最干净耐听。
- 44.1k demucs 分离人声虽采样率高，但**人声分离会在高频引入水声/音乐噪声伪影**，VoxCPM 会把这些
  伪影一起克隆 → 杂音。故**不推荐**默认用 demucs 音做参考。

## 改了哪些文件（最小、可回退、默认不影响 CosyVoice）

- 新增 `generation/whq/run_voxcpm2_zero_shot.py`：与 `run_cosyvoice3_zero_shot.py` **同 CLI**
  （`--prompt-wav/--prompt-asr/--text-file/--output`）。ultimate cloning（参考音+转写），
  `load_denoiser=False`（规避本机 torchcodec/cu128 冲突），soundfile 落盘。
  knob：`WHQ_VOXCPM_CFG`(2.0) / `WHQ_VOXCPM_STEPS`(30) / `WHQ_VOXCPM_NORMALIZE`(1) /
  可选高保真参考覆盖 `WHQ_VOXCPM_REF_WAV` + `WHQ_VOXCPM_REF_TEXT`。
- `generation/whq/tts_clean_wrap.py`：真实合成脚本改用 `WHQ_REAL_TTS_PYTHON` 指定解释器
  （默认沿用当前）——因为 VoxCPM 在独立 conda env。VoxCPM 输出**照样过起点伪声修复 + clarity EQ**。
- `generation/whq/voiceover.py`：`WHQ_REAL_TTS_SCRIPT` 改为可被环境变量覆盖，并透传
  `WHQ_REAL_TTS_PYTHON`。默认不变（CosyVoice3），零回归。

## 运行环境

- 新建隔离 env `voxcpm`（python3.11）：`pip install voxcpm soundfile`
- **torch 必须匹配本机 CUDA 12.8**：`pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
  （voxcpm 默认会拉 cu130，导致 `cuda=False`）。实测 torch 2.11.0+cu128、L20、`cuda=True` 可用。
- 权重：`openbmb/VoxCPM2`（HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`，首跑自动下载，~数 GB）
- **不加载去噪器**：VoxCPM 内置 zipenhancer 走 `torchaudio.load→torchcodec`，本机缺 `libnvrtc.so.13`
  会挂；故 `load_denoiser=False`，仅用生成核心。

## 复现命令（产出最佳版）

```bash
REPO=/home/wanghequan/Viral_Video_Split
SLUG=2_黑巧咖_ref_羽衣甘蓝_vlm37
env PATH="/root/miniconda3/envs/viral-split-tts/bin:$PATH" \
    PYTHONPATH="$REPO/generation" \
    TTS_PYTHON=/root/miniconda3/envs/viral-split-tts/bin/python \
    ASR_PYTHON=/root/miniconda3/envs/viral-split-asr/bin/python \
    FFMPEG=/root/miniconda3/envs/viral-split-tts/bin/ffmpeg \
    WHQ_CV_PYTHON=/root/miniconda3/bin/python3.13 \
    USE_WENCHAIN_OPENAI=1 HF_ENDPOINT=https://hf-mirror.com \
    WHQ_PREFER_ORIGINAL_VOICE=1 WHQ_PLAN_BEST_OF_K=1 \
    WHQ_SLOT_PIN="S01=2025-04-20 222615::A1,S02=2025-04-20 222615::A2,S03=2025-04-20 233842::A2,S04=2025-04-21 000540::A1,S05=2025-04-20 222615::A10" \
    WHQ_REAL_TTS_SCRIPT="$REPO/generation/whq/run_voxcpm2_zero_shot.py" \
    WHQ_REAL_TTS_PYTHON=/root/miniconda3/envs/voxcpm/bin/python \
    WHQ_VOXCPM_CFG=2.0 WHQ_VOXCPM_STEPS=30 \
    /root/miniconda3/envs/viral-split/bin/python -m whq.run_clone \
        --ref "$REPO/Resource/Ref_Video/羽衣甘蓝.mp4" --slug "$SLUG" \
        --out "$REPO/outputs/$SLUG/final_videos/${SLUG}_whq_clone_voxcpm_same.mp4" \
        --product-name 黑巧咖固体饮料 --model ali-qwen3.7-plus --asr \
        "$REPO/outputs/source_asr/$SLUG/all_source_asr.json" --mode hardcut
```

- 5 段全 pin 成与 `..._whq_clone_auto_pinned.mp4`（CosyVoice 版）完全相同的素材，便于纯 A/B：
  S01/S02/S05 真人原声，**S03/S04 由 VoxCPM2 克隆**。
- 换用 CosyVoice3：去掉 `WHQ_REAL_TTS_SCRIPT`/`WHQ_REAL_TTS_PYTHON`/`WHQ_VOXCPM_*` 即可。

## 成片对比清单（S03/S04 为克隆段，其余同为真人原声）

- `..._whq_clone_auto_pinned.mp4` — CosyVoice3（16k prompt）
- `..._whq_clone_voxcpm_same.mp4` — **VoxCPM2 + 16k 干净 prompt + steps30（当前最佳，无杂音）**
- `..._whq_clone_voxcpm_hiref.mp4` — VoxCPM2 + 44.1k demucs 参考 + steps30（高频有分离伪影/杂音）
- `..._whq_clone_voxcpm_hiref_s50.mp4` — VoxCPM2 + 44.1k demucs 参考 + steps50

## 调参过程结论（避免重踩）

- **参考音采样率不是越高越好**：demucs 44.1k 分离音带高频伪影 → 杂音；16k 干净 prompt 反而最净。
- **步数与"吞字"**：`inference_timesteps=30` 时短句可能吐字急促含糊（听感像吞字），50 步更舒展；
  但 ASR 复核显示 30/50 步**都没有真的丢字**，差异在节奏/清晰度而非缺字。按需在 clarity 与耗时间权衡。
- **cfg**：2.0（默认）贴音色；1.5 更松弛自然，可按素材试。
- **torchcodec/cu128**：去噪器与 `torchaudio.load` 不可用，必须 `load_denoiser=False`；音频读写用 soundfile/ffmpeg。
