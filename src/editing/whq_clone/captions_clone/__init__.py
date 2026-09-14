"""captions_clone —— 参考视频【字幕特效】复刻(移植自 Split 仓 copy_zimu/v2)。

whq_clone 原来的字幕是 finisher.burn_captions 的固定白字黑边句级 SRT。本包把 Split 仓
`copy_zimu/v2` 的「参考字幕特效复刻」原生移植进来, 让 reproduce_mode=whq_clone 的成片
字幕也跟着参考视频走(颜色/字号/位置/特效/逐字揭示)。

三段链路(与 copy_zimu/v2 同构, 但去掉硬编码路径、LLM/VLM 改走 Agent 的 wenchain 网关):

  1. 参考风格分析  ref_analyzer(VLM 抽帧读字幕) + color_calib(原生 PNG 像素级取色校准)
                   -> style_profile.distill -> <ref>_style_profile.json (按参考视频指纹缓存)
  2. 目标编排      target_match(LLM 按参考风格把成片念白拆块配色配位配特效) -> 字幕清单.md
  3. 烧录          ass_burn(逐字揭示 ASS + ffmpeg/libass)   —— WHQ_CAPTION_CLONE=ass
                   meishe(花字 styleId 云端渲染)            —— WHQ_CAPTION_CLONE=meishe

逐字时间来自对**成片**跑一次词级 ASR(asr_tokens.run_asr), 见 charstream.py。

对外只暴露 caption_clone.burn_styled_captions(), 由 finisher.burn_captions 调用。
"""
import os
import sys

# whq_clone 目录(父目录)必须在 sys.path: 本包要 import pipeline_utils / asr_tokens / _common。
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import _common  # noqa: F401,E402  (副作用: 挂 sys.path + 解析 REPO/FFMPEG)

from .caption_clone import burn_styled_captions  # noqa: E402

__all__ = ["burn_styled_captions"]
