"""阶段①：理解爆款 AI 带货短剧，提取核心成分（qwen3.7-plus 视觉）。

输出结构化 JSON：剧情套路 / 人物 / 场景 / 分镜 / 节奏 / 卖点 / 爆点成分。
"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import as_core
from drama import asr_util

SYSTEM = """你是资深短视频/带货短剧策略分析师。请观看给定的爆款 AI 带货短剧，拆解出它「为什么会爆」的核心成分与剧情套路。
只输出一个 JSON 对象，不要多余文字。字段如下：
{
  "genre": "短剧类型（如：反转带货/情景剧/测评剧）",
  "product_category": "原片主推商品的品类",
  "total_duration_sec": 数字,
  "logline": "一句话概括剧情",
  "story_beats": [{"beat":"该段情节","purpose":"hook|setup|conflict|turn|proof|cta","approx_sec":数字}],
  "characters": [{"id":"c1","name":"称呼","role":"主角/配角/路人","appearance":"外观(性别年龄发型服装配色等，尽量具体)","personality":"性格"}],
  "scenes": ["出现的场景列表"],
  "hook": {"type":"钩子类型","desc":"前3秒如何抓人"},
  "selling_points_order": ["卖点出现顺序"],
  "emotional_arc": "情绪曲线（如 好奇→焦虑→惊喜→认同）",
  "cta": {"type":"引导类型","text":"结尾行动引导"},
  "rhythm": {"avg_shot_sec":数字,"cut_density":"fast|medium|slow"},
  "shots": [{"idx":1,"start_sec":数字,"end_sec":数字,"desc":"镜头画面内容","camera":"景别与运镜","characters":["c1"],"dialogue":"台词/旁白"}],
  "viral_formula": "用一句话总结这条片子的可复用套路公式",
  "replicable_core": ["可复刻的结构性要素(与具体商品无关)"]
}
要求：shots 要覆盖全片、按时间顺序、颗粒度到「一个镜头」。characters.appearance 要足够具体以便后续生成一致的人物形象。"""


def understand_reference(video_path: str, product_hint: str = "") -> dict:
    """观看参考短剧，返回核心成分 JSON（含 ASR 口播文本）。"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    # ① 先做 ASR，把口播文本作为理解的额外证据（比纯视觉更准地拿到台词/卖点话术）
    asr = asr_util.transcribe(video_path)
    asr_text = asr_util.format_for_prompt(asr)
    user = (
        "请拆解这条爆款 AI 带货短剧的核心成分与剧情套路，严格按 system 里的 JSON schema 输出。"
        + (f" 我后续要把商品替换为：{product_hint}，因此请把与原商品强绑定的要素和可复用的结构要素分开。" if product_hint else "")
        + f"\n\n【ASR 口播文本（辅助你更准地还原台词与卖点话术）】\n{asr_text}\n"
        + "请把每个 shot 的 dialogue 尽量对齐 ASR 文本。"
    )
    import asyncio

    core = asyncio.run(
        as_core.complete_json(SYSTEM, user, vision=True,
                              media=[{"type": "video", "url": video_path}])
    )
    core["asr"] = {"text": asr.get("text", ""), "segments": asr.get("segments", []),
                   "engine": asr.get("engine", "")}
    return core


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    hint = sys.argv[2] if len(sys.argv) > 2 else ""
    out = understand_reference(path, hint)
    print(json.dumps(out, ensure_ascii=False, indent=2))
