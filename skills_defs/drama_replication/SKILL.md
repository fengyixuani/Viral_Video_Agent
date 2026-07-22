---
name: 长带货短剧复刻
skill_id: drama_replication
icon: 短剧
description: 复刻爆款 AI 带货短剧的剧情套路，换皮换商品、避免一眼抄袭，端到端生成新短剧
industry: drama
scheme_hint: balanced
operator_pipeline: [shot_segment, scene_event, asr, vlm_tag]
preset_dims: [coarse-structure, fine-conflict, fine-character, fine-style]
entry: /api/drama_replicate
---
这是「创意短剧」场景下的长 AI 带货短剧复刻能力。输入一条爆款 AI 漫剧（先支持 ~1 分钟），端到端产出剧情套路一致、但表层重构、不被一眼看出抄袭的新短剧，并把商品替换为指定商品。

链路（详见 docs/drama_replication.md）：
1. 理解：qwen3.7-plus 观看参考短剧，提取核心成分（套路/人物/场景/分镜/节奏/卖点）。
2. 写脚本：qwen3.7-max 保留套路、重构表层、替换商品，产出分段(<=15s)剧本与人物设定。
3. 角色三视图：seedream 为每个人物生成三视图，作为人物一致性锚点。
4. 分镜图：seedream i2i 以三视图(+商品图)为参考，每段生成一张关键帧图。
5. 生成视频：seedance 2.0 逐段图生视频（首帧=分镜图），ffmpeg 拼接成片。
6. 验证：qwen3.7-plus 观看成片对照目标打分，未达标按建议迭代。

风格约定：AI 漫剧统一走 3D 动画/国漫风格（非写实），既贴合漫剧调性，也规避 seedance 真人首帧风控。
