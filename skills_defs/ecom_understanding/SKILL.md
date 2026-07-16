---
name: 电商理解增强
skill_id: ecom_understanding
icon: 电商
description: 叠加在通用理解之上，聚焦电商带货的痛点、商品、卖点、CTA 与转化结构
industry: ecom
scheme_hint: faithful
operator_pipeline: [shot_segment, vlm_tag, asr, ocr]
preset_dims: [coarse-structure, fine-product, fine-hook, fine-order, fine-cta]
overlay_of: viral_reference_understanding
---
【电商理解增强叠加层】在通用爆款拆解的基础上，额外聚焦电商带货要素：
- 识别痛点场景 → 商品出现 → 多视角展示 → 使用演示 → 卖点罗列 → CTA 的转化链路。
- breakdown 重点标注：商品位置/多视角镜头、卖点顺序、贴片文字、价格标、优惠信息、CTA 形式。
- 方案维度里把「商品镜头」「人物口播」「卖点字幕」标为可替换（replace.enabled=true），便于用户补自己的商品素材。
- 判断该视频的转化目标（下单/引流/涨粉），并让 industry_reason 说明电商属性证据。
