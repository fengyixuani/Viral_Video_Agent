"""阶段②：根据核心成分写新剧本 + 人物设定（qwen3.7-max 文本）。

对齐用户的分镜模型：
- 「片段(segment)」= 一次 seedance i2v 输出（≤15s）。参考片 ~28s → 2 个片段。
- 每个片段内设计 6~9 个「分镜格(panel)」——这些格子只是**给同一段 15s 视频的设计蓝图**，
  不会各自变成独立 seedance 调用。整段 15s 由一次 i2v 生成，prompt 汇总各 panel 的画面推进 + 台词。
"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import as_core

SYSTEM = """你是顶尖带货短剧编剧 + 分镜师。给你一份「爆款短剧核心成分（含 ASR 口播）」和「目标商品」。
产出**新**剧本：套路/结构/情绪曲线/卖点编排与原片一致，但台词、人物、场景、构图、道具全部重构，
商品替换为目标商品，**不要与原片镜头一一对应**（避免一眼抄袭）。

关键概念——片段(segment) vs 分镜格(panel)：
- 因为 seedance 单次最多生 15 秒，所以按 15 秒切成若干个「片段」，每个片段对应一次 seedance i2v 调用；
- 每个片段内再设计 6~9 个「分镜格(panel)」，格子是**同一段 15s 视频里镜头/动作的推进设计**（画故事板用），
  它们不会各自成为独立视频，是用来**汇总喂给 seedance 的一段 prompt**。

只输出一个 JSON 对象：
{
  "title": "短剧标题",
  "product": {"name":"","key_features":[],"appearance":""},
  "logline": "",
  "viral_formula_kept": "",
  "differentiation": "",
  "characters": [
    {"id":"c1","name":"","role":"",
     "appearance":"极具体外观：性别/年龄/脸型/发型发色/身材/肤色/妆容",
     "wardrobe":"服装","vibe":""}
  ],
  "segments": [
    {"idx":1,"duration_sec":14,"purpose":"hook+conflict|proof+cta",
     "scene":"场景与环境(地点/光线/布景)",
     "characters":["c1","c2"],
     "product_in_frame":true,
     "opening_frame_desc":"该片段首帧的整幅画面描述(用于文生图生成 seedance i2v 首帧)",
     "panels":[
       {"idx":1,"shot_type":"WIDE|MEDIUM|CLOSE_UP|EXTREME_CLOSE_UP",
        "label":"该格用途标签(如 建立镜头/角色出场/冲突点/证据/CTA)",
        "visual":"该格具体画面描述(含人物固定外观、构图、动作)",
        "action":"1-2 秒内的动作/表情/运镜",
        "dialogue":"该格里角色说的话(会写进 seedance prompt 让其发声，可空)",
        "approx_sec":1.6}
     ],
     "voiceover_line":"把所有 panels 的 dialogue 按顺序拼成一整段(用于 seedance 语音提示)"
    }
  ]
}
硬约束：
- 这是 **AI 漫剧**（3D 动画/国漫风格，非真人写实）。
- segments 数量 = ceil(参考片时长 / 15)；单个 segment 的 duration_sec ∈ [10,15]；两段合计接近原片时长。
- 每个 segment 内 panels 数量 6~9；panel.approx_sec 之和 ≈ segment.duration_sec。
- 所有 panel 的 visual 里都要复述该片段出场人物的固定外观（与 characters.appearance 一致）以保持一致性。
- 商品段的 visual/opening_frame_desc 里必须描述商品外观。
- **只要片段内任一 panel 出现商品，该 segment 的 product_in_frame 必须 = true**（不要漏，
  否则下游生图不会把商品图作为参考）。
- 竖屏 9:16 电商短剧。"""


def write_script(core: dict, product: dict) -> dict:
    user = json.dumps(
        {
            "reference_core": core,
            "target_product": {
                "name": product.get("name", ""),
                "features": product.get("features", []),
                "appearance_notes": product.get("notes", ""),
            },
            "instruction": "按 15 秒/片段切分，每段设计 6~9 个分镜格 panel；输出 JSON。",
        },
        ensure_ascii=False,
    )
    import asyncio

    return asyncio.run(as_core.complete_json(SYSTEM, user, vision=False))


def iter_segments(script: dict):
    for seg in script.get("segments", []) or []:
        yield seg


def build_seedance_prompt(seg: dict, style: str) -> str:
    """把一个片段的所有 panels 汇总成 seedance 的单次 i2v prompt（含台词发声）。"""
    parts = [f"{style}。竖屏 9:16。场景：{seg.get('scene','')}。",
             f"该 {seg.get('duration_sec',12):.0f} 秒片段内按顺序演绎以下分镜推进："]
    for p in seg.get("panels", []) or []:
        chunk = (f"[镜{p.get('idx')} {p.get('shot_type','')} {p.get('label','')}] "
                 f"{p.get('visual','')}；动作：{p.get('action','')}")
        if p.get("dialogue"):
            chunk += f"；台词：“{p.get('dialogue')}”"
        parts.append(chunk + "。")
    voice = seg.get("voiceover_line") or "".join(
        (p.get("dialogue") or "").strip() for p in (seg.get("panels") or []))
    if voice:
        parts.append(f"画面中的角色开口说中文台词并配音（有清晰人声，口型对齐）：“{voice}”。")
    # 片段内人物一致性：seedance 只锚定首帧，长片段易中途漂移——显式要求全程与首帧一致，
    # 且台词出现时说话人须在画面内、不要凭空替换成另一个人。
    parts.append("全程保持画面中人物的外观、发型、服装、配色与首帧完全一致，"
                 "不要中途更换服装或改变脸型发型；有台词时说话的角色必须在画面内对口型，"
                 "不要凭空出现或替换成外观不同的另一个人。")
    return "\n".join(parts)


if __name__ == "__main__":
    core_path = sys.argv[1]
    with open(core_path, encoding="utf-8") as f:
        core = json.load(f)
    product = {
        "name": "理然 MAKE SENSE 去黑头泥膜棒",
        "features": ["瓦晶矿物泥+三重植物精油", "有效清洁毛孔/去黑头", "棒状膏体直接涂抹，方便"],
        "notes": "绿色膏体棒状，黑色底座，瓶身有 理然 / MAKE SENSE logo，膏体为灰紫色",
    }
    print(json.dumps(write_script(core, product), ensure_ascii=False, indent=2))
