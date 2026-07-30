"""whq — 参考视频「逐镜复刻」增补模块 (additive, 不改动原始 source)。

目标: 在用户素材中裁剪出一个「最像参考视频」的成片。
与主流水线的区别:
  - 主流水线走抽象 DNA -> 5 个语义 slot -> 匹配, 丢掉了参考视频真实的
    分镜时序/节奏。
  - whq 保留参考视频的**真实分镜时间线**(ffmpeg 场景检测 + DNA key_beats
    语义描述), 逐镜为每个参考镜头挑选最像的用户素材片段, 按参考顺序与时长
    硬剪拼接 -> 成片在结构/节奏上贴近原视频。

两个方向:
  1. hardcut  — 纯用户素材硬剪 (默认)
  2. seedance — 对没有合适用户素材的镜头, 可选调用 T2V 补拍, 用户素材仍为主

所有代码只新增, 复用 common/ 的 pipeline_utils / config 与既有 understanding
产物 (all_user_assets.json + DNA md)。
"""
