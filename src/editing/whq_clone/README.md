# whq — 参考视频「结构级复刻」成片模块

用**用户自己的素材**，复刻一条参考爆款视频的**叙事结构**，产出带配音 + 文案字幕的带货成片。
不逐镜对齐，而是按参考视频 DNA 的叙事节拍做**结构级**段落（约 5~8 段），逻辑合理、尽量贴近参考节奏。

> 相关文档：
> - 配置项模板：[`whq.env.example`](whq.env.example)（所有可调 knob，可 `source` 后运行）
> - 能力/原理详解与二次开发指引：[`FEATURES.md`](FEATURES.md)

## 设计目标

1. **成片有文案 + 配音**：硬剪无声 base 后叠加 TTS 配音，并烧录同步字幕。
2. **音轨完全重建**：移除参考/素材原声，音轨由配音（原声保留段 + 克隆段）重建。
3. **镜头不重复**：用户素材片段 1:1 分配，每段素材最多用一次。
4. **尽量保留用户原声**：素材自带贴题口播且有人脸时保留真人原声，否则才克隆配音（见下）。
5. **贴近参考**：段落顺序/时长贴着参考总时长与叙事节奏铺满，结尾按参考语速加速。

代码只新增，复用 `common/`（pipeline_utils/config）、understanding 产物与主流水线收尾脚本
（`generation/build_tts_overlay.py`），不动上游。

## 端到端流程

```
参考视频 DNA (key_beats)
  → 结构级段落 (edit_planner: 按节拍比例铺满参考总时长, 每段带 ref_cps)
  → 用户素材 1:1 不重复分配 (LLM 主 + 确定性兜底; 可原声优先 + best-of-K 择优 + 连贯定向修复 + 人工 pin)
  → [可选 seedance 补缺口]
  → 原声/克隆决策 (voice_policy: 有贴题口播+人脸 → 保留原声, 否则克隆)
  → 硬剪无声 base (clone_builder: 每段 -an, 原声段视频严格跟随"有声区间", 末尾补静音音轨)
  → 配音 (voiceover: 原声段抽实录音频 + 克隆段 CosyVoice3 零样本克隆 + LLM 文案审查/错字字幕更正)
  → 烧录文案字幕 (+可选迁移参考 BGM) (finisher)
  → 逐段语速贴参考 (pace_match)
```

## 前置产物（消费 understanding 阶段，落在 `outputs/<slug>/`）

- `dna_understanding/*dna_template*.md` — 参考视频 DNA
- `user_understanding/*/all_user_assets.json` — 用户素材理解（候选片段库）
- `outputs/source_asr/<slug>/all_source_asr.json` — 用户素材 ASR（原声保留 & 配音底座；缺失则全克隆）

生成方式见主流水线 `Pipeline_Runner.py` / `understanding/`。

## 快速开始

```bash
# 推荐: source 配置模板后一键跑(复用某 slug 的理解产物, 自动发现 DNA/assets/ASR)
cp generation/whq/whq.env.example /tmp/whq.env      # 按需改 knob
set -a; source /tmp/whq.env; set +a
python -m whq.run_clone --slug <slug> --ref <参考视频> \
    --out outputs/<slug>/final_videos/<slug>_whq_clone.mp4 \
    --product-name <商品名> --model ali-qwen3.7-plus --mode hardcut
```

不 source 配置时，所有 knob 用内置默认值（等价旧行为）。关键运行环境（`PYTHONPATH`、
`TTS_PYTHON`、`FFMPEG` 等）必须正确，见 [`whq.env.example`](whq.env.example) 顶部「运行环境」。

### 常用姿势

```bash
# 尽量保留用户原声 + 3 选 1 自动择优(原声率优先) + 连贯定向修复
WHQ_PREFER_ORIGINAL_VOICE=1 WHQ_PLAN_BEST_OF_K=3 python -m whq.run_clone --slug ... --ref ...

# 人工锁定某段用哪条素材(硬约束, 不被自动修复改动); 逗号分隔多段
WHQ_SLOT_PIN="S02=2025-04-20 222615::A2" python -m whq.run_clone --slug ... --ref ...

# 离线确定性分配(不调 LLM), 完全可复现
python -m whq.run_clone --slug ... --ref ... --no-llm

# seedance 补缺口 + 迁移参考 BGM
python -m whq.run_clone --slug ... --ref ... --mode seedance --migrate-bgm
```

## CLI 参数（`run_clone.py`）

- `--ref` 参考视频；`--slug` 从 `outputs/<slug>` 自动发现产物
- `--dna` / `--assets` / `--asr` 显式指定产物（省略则按 slug 发现）
- `--out` 最终成片 mp4；`--mode hardcut|seedance`
- `--no-llm` 离线确定性分配；`--model` 文案/分配 LLM 模型名
- `--product-name` 商品名（TTS/文案上下文）
- `--gap-threshold` 缺口判定阈值；`--total-duration` 覆盖参考总时长
- `--no-stretch` 短素材不拉伸；`--migrate-bgm` 迁移参考 BGM（需 meishe 环境）
- `--no-pace-match` 关闭逐段语速贴参考；`--no-ref-speed` 不采集参考语速

## 产出（均以 `<out>` 去扩展名为前缀）

- `<out>.mp4` — 最终成片（无声 base + 配音 + 字幕）
- `<out>_base.mp4` / `<out>_voiced.mp4` / `<out>_bgm.mp4`（后者仅 `--migrate-bgm`）
- `<out>_plan.json` — 结构级段落 + 1:1 分配 + 原声对窗决策
- `<out>_manifest.json` — 每段源片段/时长/累计时间线
- `_whq_work/tts/tts_overlay_plan.json` — **最终逐段成片文本/配音**（每段 text/caption_text/
  voice_source/audio_take/wav；最贴近"成片念了什么"）。注意 `_whq_work/` 按输出目录共享，会被
  同目录下后续任意一次 whq 重跑覆盖。

## 各模块（均可单独运行）

| 文件 | 作用 |
|---|---|
| `reference_shots.py` | 参考 DNA 载入 + `parse_key_beats` |
| `asset_index.py` | understanding 产物 → 拍平候选片段库 |
| `edit_planner.py` | DNA→结构段落 + 1:1 分配（原声优先 / best-of-K / 连贯修复 / slot pin） |
| `voice_policy.py` | 逐段「原声 vs 克隆」决策 + 句子级对窗 |
| `clone_builder.py` | 裁剪+归一化(720x1280)+硬剪无声 base（原声段视频跟随有声区间） |
| `seedance_fill.py` | 缺口段 T2V 补拍 |
| `voiceover.py` | 原声抽取 + CosyVoice3 克隆 + LLM 文案审查/错字字幕更正（复用 `build_tts_overlay.py`） |
| `tts_clean_wrap.py` | CosyVoice3 薄封装：修每段起点低频哼鸣伪声 |
| `finisher.py` | 烧录字幕(libass) + 可选迁移参考 BGM |
| `pace_match.py` | 成片逐段语速贴参考（atempo 变速不变调，详见 `PACE_MATCH.md`） |
| `pace_voiced.py` | 对任意已出片 `_voiced.mp4` 事后调速（成片 ASR 算实际语速 + manifest ref_cps） |
| `run_clone.py` | 一键编排 |

## 环境

- 分配/编排/配音本体：`viral-split`（`ask_qianfan` 走 Wenchain/千帆）
- ffmpeg（含 libass）+ CosyVoice3 TTS：`viral-split-tts`
- 用户/参考 ASR：`viral-split-asr`（Qwen3-ASR）
- 人脸检测（YuNet）：`/root/miniconda3/bin/python3.13`（需 cv2）
- BGM 迁移：meishe 环境，不可达时优雅跳过

所有路径/开关均可用环境变量覆盖，见 [`whq.env.example`](whq.env.example)。
