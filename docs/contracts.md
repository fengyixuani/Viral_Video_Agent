# 数据契约（模块间接口）

三人协作的解耦靠的是这两个契约。**字段冻结**；需要改动时，改字段的人负责：加版本号、
同步本文件、通知上下游评审。只要契约稳定，A/B 各自改内部实现互不影响。

---

## 契约①：理解 → 编排（A → B，**当前的跨人边界**）

`UnderstandingAgent.analyze_stream`（A）通过 SSE 产出，前端聚合后随 `/api/replicate` 回传给
`OrchestrationAgent.replicate_stream`（B）。B 在入口用 `contracts.validate_analysis_result` 自检。

来源事件：
- `{"type":"analysis","result": <AnalysisResult>}`
- `{"type":"materials","result": <material_understanding>}`
- `{"type":"feasibility","result": <feasibility>}`

组合后的 `AnalysisResult`（`understanding/orchestrator.py` 构造）：
```jsonc
{
  "template": {
    "shot_slots": [
      { "id": 1, "role": "开场钩子", "want": "一句话镜头意图",
        "duration": 3.0, "source_time_range": "0.00-3.50",   // ★ 参考视频里该镜真实起止
        "breakdown": [ { "dim": "景别", "value": "近景特写" } ] }
    ],
    "total_duration_sec": 14.66, "hook": {}, "narrative_structure": [],
    "rhythm": {}, "cta": {}, "packaging": {}, "dropped_trailing_slots": {}
  },
  "industry_guess": "ecom|live|drama|knowledge",
  "schemes": [ { "id": "faithful", "name": "", "strategy": "faithful|balanced|regenerate",
                 "dimensions": [ { "id": "", "name": "", "level": "coarse|fine",
                                  "replace": { "enabled": true, "types": ["image|text|video"] } } ] } ],
  "aspect": { "width": 1080, "height": 1920, "ratio": 0.5625 },
  "material_understanding": { /* {cache_key: 单素材理解 payload}，见契约②的 segment 结构 */ },
  "feasibility": {
    "<shot_id>": {
      "status": "direct|partial|none",
      "matched_asset_id": "gid", "matched_source_path": "", "matched_time_range": "",
      "score": 0.0, "candidates": [ { "asset_id": "", "summary": "", "speech_or_text": "",
                                      "source_time_range": "", "source_path": "" } ]
    }
  }
}
```

---

## 契约②：编排 → 剪辑（**B 内部**）

编排 `orchestration/`（B）产出、剪辑 `editing/`（B）消费，落盘到 `outputs/<方案名>/`。
虽为 B 内部产物，仍用 `contracts.validate_strategy` 在产出/消费两端自检。

### 2a. `*_selected_editing_strategy_*.json`
由 `shared/scriptgen.build_selected_editing_strategy` 生成。B 的 `editing/loop.py::_load_inputs` 消费。
```jsonc
{
  "metadata": { "scheme": "", "strategy": "faithful|balanced|regenerate",
                "project_name": "", "dna_path": "", "generated_at": "" },
  "user_asset_bank": [
    { "asset_id": "asset_S01_<gid>", "source_video_id": "", "source_path": "uploads/x.mp4",
      "source_time_range": "0.00-3.50", "visual_description": "", "speech_or_text": "",
      "suitable_roles": [], "strengths": [], "quality_score": 0.0,
      "concrete_editing_plan": { "slot_id": "S01", "source_path": "", "source_time_range": "",
                                 "target_duration": 3.0, "final_caption_text": "" },
      "alternate_assets": [ { "asset_id": "", "source_path": "", "source_time_range": "",
                              "summary": "", "speech_or_text": "" } ] }
  ],
  "slot_matching": [
    { "slot_id": "S01", "status": "matched|partial_matched|missing",
      "matched_asset_id": "", "fit_score": 0.0, "reason": "" }
  ],
  "editing_timeline": [                       // ★ B 的主输入：每个 slot 一条
    { "slot_id": "S01", "target_time_range": "0.00-3.00",
      "source_path": "uploads/x.mp4", "source_time_range": "0.00-3.50",
      "caption_text": "", "transition_to_next": "hard_cut",
      "action": "use_user_asset|generate|reshoot" }   // use_user_asset=有素材；generate/reshoot=缺失
  ],
  "missing_assets": [                         // ★ AIGC 补镜的待办（action != use_user_asset 的 slot）
    { "slot_id": "S07", "role": "", "goal": "该镜意图", "action": "generate",
      "generation_prompt": "", "target_duration": 3.0 }
  ],
  "overall_editing_strategy": { "structure_fit_score": 0.0, "material_sufficiency_score": 0.0 }
}
```

### 2b. `connector_context.json`（同目录）
提供全量素材池 + 分镜结构，B 用于全池召回、AIGC 补镜。
```jsonc
{
  "shots": [ { "slot_id": "S01", "role": "", "want": "", "duration": 3.0,
               "source_time_range": "0.00-3.50",
               "breakdown": [ { "dim": "景别", "value": "" } ] } ],
  "segments": [                               // 用户素材池（每个可检索片段）
    { "global_asset_id": "video::A1", "source_path": "uploads/x.mp4",
      "source_time_range": "0.00-3.50", "one_sentence_summary": "", "visual_description": "",
      "speech_or_text": "", "keywords": [], "actions": [], "visible_objects": [],
      "visual_evidence_tags": [], "quality_score": 0.0 } ]
}
```

---

## 契约③（C 独立，仅供参考）：AI 短剧
C 自成一体，产物在 `outputs/drama_<ts>/`：`01_core.json`（理解）→ `iter1/02_script.json`（换皮脚本）
→ `sheets/ boards/ first_frames/ product_asset/ clips/ final.mp4` + `05_verify.json`（打分）。
不与契约①②交互，只共享 `shared` 层（as_core / tools / wenchain_media）。
