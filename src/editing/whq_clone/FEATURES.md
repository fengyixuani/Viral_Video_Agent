# whq 功能文档（二次开发指引）

面向想复用/扩展 whq 结构级复刻的开发者。逐能力讲**它解决什么、决策逻辑、涉及代码、可调 knob、
如何扩展与已知边界**。总体链路见 [`README.md`](README.md)，全部 knob 见 [`whq.env.example`](whq.env.example)。

数据在模块间的主干结构：

- `candidate`（`asset_index.load_candidates`）：`{global_asset_id, source_path, start, end, duration, asset_type, quality_score, keywords, text}`
- `segment`（`edit_planner`）：一个叙事节拍 + 分到的 `best_candidate` + `score/method/is_gap`
- `decision`（`voice_policy.decide`）：`{voice_source, window_text, audio_take, win_start, win_take, atempo, decision_basis, ...}`
- `manifest`（`clone_builder.build_base`）：每段实际裁剪信息 + 累计 `target_time_range`
- `tts_overlay_plan.json`（`build_tts_overlay`）：最终逐段 `text/caption_text/voice_source/audio_take/tts_wav`

---

## 1. 结构级段落规划

**做什么**：把参考视频 DNA 的 `key_beats`（叙事节拍）按时长比例铺满参考总时长，得到 5~8 段
`segment`，每段带 `ref_time_range` 与 `ref_cps`（该段参考真实语速，配音跟随）。

**代码**：`edit_planner.build_segments_from_dna` + `reference_shots.parse_key_beats`；`ref_cps`
由 `_ref_cps_for_range` 按"有说话时间并集"计（剔除停顿/BGM，避免低估）。CTA（催单）节拍由
`reorder_cta_last` 统一后移到片尾。

**扩展**：改 beat→时长的分配策略、CTA 识别正则 `_CTA_RE`。

---

## 2. 用户素材 1:1 不重复分配

**做什么**：给每个段落分配一条**互不重复**的用户素材。LLM 主（`assign_llm`）+ 确定性兜底
（`assign_deterministic`，中文双字 Jaccard + 关键词 + 质量分）。

**代码**：`edit_planner.plan_edit`。`--no-llm` 走纯确定性、完全可复现。

---

## 3. 原声优先匹配（`WHQ_PREFER_ORIGINAL_VOICE=1`）

**解决**：whq 会重建音轨，默认易全克隆；用户希望**尽量保留素材真实原声**。全克隆的根因通常是
匹配阶段挑了"视觉相关但音轨静音/跑题"的空镜，而非阈值太严——**靠放宽 voice_policy 阈值会把
跑题闲聊粘进成片，是错的**。正解在匹配阶段偏向自带贴题口播的素材。

**决策逻辑**：`build_speech_map` 用与 `voice_policy.window_speech` **完全一致**的口径，算每个候选
**自身窗口内**的用户口播字数/覆盖率（用候选自身 `[start,end]` 作窗，长素材里口播集中的 asset_segment
局部覆盖率高，不被整片低覆盖率淹没）。LLM 分配提示里标注这些候选的真实口播片段，并要求
"承接连贯优先、贴题口播优先、明显跑题闲聊宁可克隆"。

**代码**：`edit_planner.build_speech_map` / `assign_llm`（`_OV_GUIDANCE`）/ `assign_deterministic`
（`OV_DET_BONUS`）。需在 `run_clone` 把用户 ASR (`speech_records`) 提前加载并传入 `plan_edit`。

**边界**：口播度量识别的是"有没有人在说话"，不判"是否贴题"——贴题交给 LLM（它能看到口播原文）
和相关性打分共同把关。

---

## 4. best-of-K 择优（`WHQ_PLAN_BEST_OF_K`）

**解决**：LLM 分配是带温度的非确定性采样，单次结果不稳。

**决策逻辑**：采样 K 个分配方案，用 `score_plan` 打分取最优。**目标是约束/优先式，不是加权混合**：
`total = 原声率 + 0.01×连贯 + 0.001×贴合`——**原声率是绝对主目标**，连贯只当同原声率时的平局项。
（历史教训：早期用"连贯主导加权"会把"全克隆(连贯高/原声0)"顶上来，与保原声目标相反；连贯 LLM
判分本身噪声大且对口语粗糙有偏见，不适合当主目标。）

**代码**：`edit_planner.plan_edit` 的 K>1 分支 + `score_plan` + `_plan_coherence`。

**成本**：在线 LLM 调用约翻 K 倍（K 次分配 + 各 1 次连贯打分 + 修复若干次）。

---

## 5. 连贯定向修复 + 人工 pin 建议

**解决**：某段口播接不上上一段（叙事断裂），纯重采样常采不到能桥接的排列。

**决策逻辑**：对原声率最高的方案，取连贯 LLM 标出的最弱衔接 `SXX->SYY`，让 LLM 从**未用口播**里
挑一条最能承接 `SXX` 的换入 `SYY`（`_pick_bridge_candidate`，承接连贯优先于贴原定主题），
重评择优。修复不减原声才采纳。仍有断裂时，打印一条现成的 `WHQ_SLOT_PIN` 建议交人工一键锁定。

**人工 pin**（`WHQ_SLOT_PIN="S02=<id>[,S03=<id>]"`）：**硬约束**——被 pin 的槽 `pinned=True`，
连贯修复不得改动/交换它。displaced 的重复素材自动让位以保持 1:1。

**代码**：`edit_planner._repair_coherence` / `_suggest_pin` / `_apply_slot_pins`。

**已知边界**：连贯是主观项，自动判分在零碎口语素材上不可靠——设计上**自动保原声 + 连贯交人工 pin**，
不追求纯全自动复现人工审美。

---

## 6. 原声 vs 克隆决策（`voice_policy`）

**决策**：逐段看三证据——窗口口播（字数≥`WHQ_VOICE_MIN_CHARS` 且覆盖≥`WHQ_VOICE_MIN_COVERAGE`）、
人脸（≥`WHQ_FACE_MIN_RATIO`）、变速可行（atempo∈[`MIN`,`MAX`]）：
- 有口播 + 有人脸 + 变速可行 → `original`
- 有口播 + 无人脸 → `llm_decide`（脚本 LLM 判原声是否贴合整体文案）
- 无口播 / 变速不可行 → `clone`

**句子级对窗**（`align_window`）：把窗口终点落在句尾停顿（`full_sentence`），说完整句、音画同倍率
伸缩；塞不进弹性时 `clause_cut`——**只截音频到最后一个完整分句**。

---

## 7. 视频严格跟随"有声区间"（无声尾根治）

**解决的 bug**："扒了配料表才发现"有口型无声——`clause_cut` 只把**音频**收口到分句停顿，
**视频窗口没跟着裁**，画面继续放下一分句的口型却没有声音。

**修法**：`run_clone` 对**所有原声段**用 `audio_take`（有声时长）当视频 `source_take`（不再用整窗
`win_take`）。有声段 = `[win_start, win_start+audio_take]`，视频与音频同窗同倍率，`clone_builder`
setpts 拉伸填满槽位。`full_sentence` 尾部停顿一并裁掉（无害），`clause_cut` 的无声口型被裁掉（根治）。

**代码**：`run_clone.run`（decisions 应用处）+ `clone_builder._build_segment`（`source_take` 驱动
setpts）。校验：`<out>_plan.json` 里原声段 `source_take` 应 ≈ overlay plan 的 `audio_take`。

---

## 8. 配音与字幕错字更正（`voiceover`）

- **原声段**：抽实录音频，atempo 与视频同倍率对齐。
- **克隆段**：CosyVoice3 零样本克隆用户音色念 LLM 文案；`tts_clean_wrap` 修每段起点低频哼鸣伪声。
- **LLM 整体审查**（`review_script`）：原声段实录 ASR 若有同音错字（如「壁垒」实为「避雷」），在
  `caption_text` 给**等长逐字更正版**（音频不动，字幕优先烧录）；克隆段可合规改写（须重过事实核验）。
- 事实核验：克隆文案里的数字/参考卖点断言必须能在用户口播里找到出处，防编造/搬参考。

---

## 9. 收尾：字幕 + BGM + 语速贴参考

`finisher` 烧录字幕（优先 `caption_text`）、可选迁移参考 BGM；`pace_match` 按参考语速逐段加速
（atempo 变速不变调，字幕已烧进画面随段走），总时长随之缩短。

---

## 校验 / 自测建议

- 单模块可独立跑（见 README 表格），如 `python asset_index.py --assets ...`、
  `python edit_planner.py --dna ... --assets ...`。
- 改分配/择优逻辑后，建议用现成 slug 的 `all_user_assets.json` + `all_source_asr.json` 离线构造
  `out_segments`，mock `_plan_coherence` 打分，验证 `score_plan` / `_repair_coherence` 的择优方向，
  再跑端到端（省 LLM/GPU 成本）。
- 端到端产物看 `_whq_work/tts/tts_overlay_plan.json`（逐段 text/voice_source/audio_take）与
  `<out>_plan.json`（分配 + 原声对窗）。
