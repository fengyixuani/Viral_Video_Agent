---
name: 直播切片复刻
skill_id: live_clip
icon: 直播
description: 从直播内容中复刻高光话术、情绪峰值与转化节奏
industry: live
scheme_hint: faithful
operator_pipeline: [asr, emotion, shot_segment, ocr]
preset_dims: [coarse-structure, fine-hook, fine-emotion, fine-cta]
---
这是直播带货高光切片场景。重点检测高光话术、情绪峰值、前 3 秒 Hook、商品利益点与转化 CTA；优先复用真实直播素材，声音允许使用 TTS Clone 占位。
