"""阶段⑤：用 qwen3.7-plus 观看成片，对照目标打分并给出修改建议。"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import as_core
from drama import asr_util

SYSTEM = """你是带货短剧质检官。给你「复刻目标(原片核心成分)」「新剧本」「成片视频」以及「成片的 ASR 口播文本」。
请结合画面与 ASR 文本评估成片是否达标。只输出一个 JSON：
{
  "scores": {
    "story_formula": 0-10,   // 是否复用了原片套路/结构/情绪曲线
    "not_plagiarism": 0-10,  // 表层是否与原片明显不同(不被一眼看出抄袭)
    "character_consistency": 0-10, // 同一人物跨段外观是否一致
    "product_correct": 0-10, // 商品是否为目标商品且外观正确
    "coherence": 0-10,       // 拼接后是否连贯像一条完整短剧
    "audio_script_match": 0-10, // ASR 口播是否契合剧本台词/卖点，且完成转化引导
    "selling_effect": 0-10   // 带货说服力
  },
  "overall": 0-10,
  "pass": true/false,        // overall>=7 且无单项<5 视为通过
  "problems": ["具体问题"],
  "fix_suggestions": [{"stage":"script|storyboard|video","segment":序号或null,"suggestion":"如何改"}]
}"""


def verify_final(video_path: str, core: dict, script: dict) -> dict:
    asr = asr_util.transcribe(video_path)
    asr_text = asr_util.format_for_prompt(asr)
    user = json.dumps(
        {"reference_core": core, "new_script": script,
         "final_video_asr": asr_text,
         "instruction": "结合画面与 final_video_asr 口播文本，按 schema 打分，判断是否达标(pass)。"},
        ensure_ascii=False,
    )
    import asyncio

    report = asyncio.run(
        as_core.complete_json(SYSTEM, user, vision=True,
                              media=[{"type": "video", "url": video_path}])
    )
    report["final_video_asr"] = {"text": asr.get("text", ""), "engine": asr.get("engine", "")}
    return report
