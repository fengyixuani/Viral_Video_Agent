"""共享引导 (Agent 版) — 把 whq_clone 自身、Agent 的 src/shared、以及 Split 兄弟仓
的 vendor 目录挂到 sys.path，并解析 REPO/FFMPEG。

移植说明 (Viral_Video_Split.generation.whq -> Viral_Video_Agent.src.editing.whq_clone):
- 原 whq 依赖 Split 仓的 ``common/pipeline_utils`` (内含 ``ask_qianfan``)。在 Agent 里
  改为本目录下的 ``pipeline_utils`` 薄封装，转发到 Agent 的 ``as_core`` 网关（同一个
  wenchain 网关、同一个 key），因此**不再**把 Split 的 ``common/`` 挂上 sys.path。
- 仍有少量子进程脚本 (``build_tts_overlay.py`` / ``batch_qwen3_asr.py`` / 模型权重) 物理
  存放在 Split 仓，通过 ``REPO`` (= ``VIRAL_VIDEO_SPLIT_ROOT``) 定位；Agent 的 config.env
  本就引用该根目录放 ASR/TTS 模型，属既有约定。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# whq_clone 自身目录必须在 sys.path，保证模块内扁平 import (from shot_matcher import ...,
# import voice_policy, from pipeline_utils import ...) 解析到本目录（而非 Split 的同名模块）。
if _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)

# Agent 的 src/ 与 src/shared/ —— 供本目录的 pipeline_utils 薄封装 import as_core 等。
_AGENT_SRC = os.path.dirname(os.path.dirname(_HERE))  # .../src
for _p in (_AGENT_SRC, os.path.join(_AGENT_SRC, "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# Split 兄弟仓根目录：存放 build_tts_overlay.py / understanding/batch_qwen3_asr.py / models/。
REPO = os.getenv("VIRAL_VIDEO_SPLIT_ROOT", "/home/wanghequan/Viral_Video_Split")
COMMON = os.path.join(REPO, "common")
VENDOR = os.path.join(COMMON, "vendor", "Viral_Video")

# seedance_fill 需要 vendor 里的 qianfan_t2v_client；仅在 --mode seedance 时用到。
if os.path.isdir(VENDOR) and VENDOR not in sys.path:
    sys.path.append(VENDOR)

os.environ.setdefault("VIRAL_VIDEO_SOURCE_ROOT", VENDOR)

# ffmpeg 需含 libass（字幕烧录）。系统 PATH 常缺，imageio 静态 ffmpeg 也常缺 libass；
# 默认用 viral-split-tts 环境的 ffmpeg，可用 FFMPEG 环境变量覆盖。
FFMPEG = os.getenv("FFMPEG", "/root/miniconda3/envs/viral-split-tts/bin/ffmpeg")
if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"
os.environ.setdefault("FFMPEG", FFMPEG)


def repo_path(*parts):
    return os.path.join(REPO, *parts)


def legacy():
    """WHQ_LEGACY=1 时回到迁移前（Split 7/22 版）的编排/文案行为。

    迁移进 Agent 后新加的几条约束（商品展示下限、原声完整句护栏、文案字数放宽 +4、
    锚定本段素材/禁重复/催单只在末段的 prompt 硬约束、结尾 CTA 用原声）会互相压制，
    实测成片不如迁移前。这个开关用于一键对齐旧行为，便于 A/B 定位。
    """
    return os.getenv("WHQ_LEGACY", "0") not in ("0", "false", "False", "")
