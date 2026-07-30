# 语速贴参考（pace_match / pace_voiced）

把成片的口播语速逐段调整到贴近参考爆款视频。两个入口，核心变速逻辑共用。

## 原理

- **语速指标**：cps（字/秒），只按**有说话时间**计（去标点字符数 ÷ 有声时长），
  避免把段内留白摊进语速、低估真实语速。
- **每段倍率** `r = 目标cps / 实际cps`，其中 目标cps = `min(该段参考 ref_cps, TARGET_CPS_MAX)`，
  倍率钳制在 `[MIN_SPEED, MAX_SPEED]`。
- **逐段而非全局**：原声段可能本就达到参考语速，全局加速会把它推到失真快感。
- **变速实现**：一次 ffmpeg filter_complex，逐段
  `trim + setpts/(r)`（视频）+ `atrim + atempo=r`（音频，变速不变调）后 concat。
  字幕若已烧进画面则随段同步。**总时长会缩短**（加速的必然结果）。
- `ref_cps` 数据链见 `FEATURES.md`：run_clone step0 对参考视频跑 ASR →
  edit_planner 按段 `ref_time_range` 聚合窗内字数/时长 → 透传到 manifest / tts plan。

## 参数（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `WHQ_PACE_MAX_SPEED` | 1.4 | 每段最大加速倍率；再快就是"录音快放"式失真 |
| `WHQ_PACE_MIN_SPEED` | 1.0 | 每段最小倍率（只加速不减速） |
| `WHQ_PACE_TARGET_CPS` | 5.5 | 结果语速天花板（字/秒）；参考爆款常 7+ 字/秒，不硬追 |

即：即使参考段 8 字/秒，实际 3 字/秒的段最多也只提到 min(5.5, 3×1.4)=4.2 字/秒。

## 入口一：pace_match.py（run_clone 流水线内）

数据来自 `_whq_work/tts/tts_overlay_plan.json`（每 item 带 `ref_cps` 与
`voice_wav`/`tts_wav`，实际 cps 用该段语音 wav 的 silencedetect 有声时长算）。
run_clone 收尾自动调用（`--no-pace-match` 关闭），也可单跑：

```bash
/root/miniconda3/envs/viral-split-tts/bin/python generation/whq/pace_match.py \
    --video <成片.mp4> --tts-plan <tts_overlay_plan.json> --out <out.mp4> \
    [--max-speed 1.4 --min-speed 1.0]
```

限制：`tts_overlay_plan.json` 在共享的 `_whq_work/` 下，**只反映最近一次
run_clone**；对更早出的片不适用 → 用入口二。

## 入口二：pace_voiced.py（任意已出片的 _voiced.mp4，事后调）

不依赖中间产物：先对成片自身跑 ASR 拿逐字时间戳算每段实际 cps（窗口=manifest 的
`target_time_range`，token 区间取并集、>0.15s 间隙断开），`ref_cps` 取自该片的
`_manifest.json`，然后复用 `pace_match.apply_pace` 出片。

```bash
# 1) ASR 成片（viral-split-asr 环境, 产 <workdir>/all_source_asr.json）
env ASR_PYTHON=/root/miniconda3/envs/viral-split-asr/bin/python \
    FFMPEG=/root/miniconda3/envs/viral-split-tts/bin/ffmpeg \
    NARIS_SOURCE_GLOB=<voiced.mp4> NARIS_ASR_DIR=<workdir> \
    /root/miniconda3/envs/viral-split-asr/bin/python understanding/batch_qwen3_asr.py

# 2) 变速出片
/root/miniconda3/envs/viral-split-tts/bin/python generation/whq/pace_voiced.py \
    <voiced.mp4> <workdir>/all_source_asr.json <slug>_manifest.json <out.mp4>
```

## 实例（2026-07-22, 2_黑巧咖_ref_羽衣甘蓝_vlm37）

用入口二对 `..._voxcpm_same_voiced.mp4` 与 `..._auto_pinned_voiced.mp4` 原地调速
（参考各段 ref_cps 7.4~8.6）：

- voxcpm_same：S01 x1.40 (3.1→4.3 cps) / S02 x1.0 / S03 x1.10 / S04 x1.09 / S05 x1.0，
  27.93s → 25.37s
- auto_pinned：S01 x1.40 / S04 x1.17 (4.7→5.5) / 其余 x1.0，27.93s → 25.57s

S01 是原声段且只有 3.1 字/秒，顶到 1.4x 上限也到不了参考的 7.6——这是
MAX_SPEED 有意设的失真天花板，属预期行为。
