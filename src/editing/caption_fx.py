"""字幕特效模仿：对任意成片烧上「参考视频同风格」的字幕。

前端「字幕特效模仿」按钮走这里。与 `editor.py` + `caption_tool.py` 那套固定样式(底部半透明
黑底白字、逐镜 PNG overlay)是两回事：这里复用 `whq_clone/captions_clone`(移植自 Split 仓
`copy_zimu/v2`)——分析参考视频的字幕风格(VLM 抽帧 + 像素级取色校准), 再把成片念白按该风格
拆块配色配位配特效, 整块弹出烧上去(WHQ_CAPTION_REVEAL=char 可切回逐字揭示;
WHQ_CAPTION_ANIM=on 可加回缩放弹入动画, 默认直接展示)。

为什么是「成片后处理」而不是接进 editor: 块位置交替与整条时间轴的互斥排布需要一条**全局时间轴**, 逐镜
各烧各的做不出来; 而且 editor 用的 imageio-ffmpeg 常缺 libass, captions_clone 自带
`viral-split-tts` 的 ffmpeg。做成独立一步也让它能对任何一条成片重跑, 不用重剪。

字幕文本来源: 有配音 plan 时用 plan 文本(准), 没有就用成片词级 ASR 的识别结果(可能有同音错字)。
"""
from __future__ import annotations

import os
import sys
import time

import obs

_log = obs.get_logger("caption_fx")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(AGENT_ROOT, "uploads", "final")
_WHQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whq_clone")


def _resolve(uri: str) -> str:
    """uri/相对路径 -> 本地绝对路径; 非本地(http/data)或不存在返回 ""。"""
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for c in (uri, os.path.join(AGENT_ROOT, uri.lstrip("/"))):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def _load_burner():
    """import captions_clone.burn_styled_captions（需要把 whq_clone 目录挂上 sys.path）。"""
    if _WHQ_DIR not in sys.path:
        sys.path.insert(0, _WHQ_DIR)
    from captions_clone import burn_styled_captions
    return burn_styled_captions


def run_caption_fx(video_uri: str, reference_video: str = "", tts_items=None):
    """对成片跑字幕特效模仿，逐步 yield step 事件，最后 yield caption_fx_done。

    video_uri:       成片（uploads/final/xxx.mp4 这类相对 uri 或绝对路径）。
    reference_video: 参考视频 uri；缺失/非本地时用内置回退风格（白字口播 + 红橙斜排大字）。
    tts_items:       可选的配音 plan items（[{start,end,text}]），缺省则用成片 ASR 文本。
    """
    rid = time.strftime("%H%M%S")

    def step(key, title, thought, state="done", observation=None):
        ev = {"type": "step", "phase": "字幕特效模仿", "key": f"{key}-{rid}",
              "state": state, "title": title, "thought": thought}
        if observation is not None:
            ev["observation"] = observation
        return ev

    src = _resolve(video_uri)
    if not src:
        yield {"type": "error", "message": "成片不存在或不是本地文件：{}".format(video_uri)}
        return
    ref = _resolve(reference_video)
    if reference_video and not ref:
        _log.warning("[%s] 参考视频非本地文件, 将用回退风格: %s", rid, reference_video)

    yield step("prepare", "准备", "成片 {}；参考视频 {}".format(
        os.path.basename(src), os.path.basename(ref) if ref else "（无，用回退风格）"),
        state="running")

    try:
        burn = _load_burner()
    except Exception as exc:  # noqa: BLE001
        _log.error("[%s] captions_clone 不可用: %s", rid, exc, exc_info=True)
        yield {"type": "error", "message": "字幕特效模块不可用：{}".format(str(exc)[:200])}
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(OUT_DIR, "{}_capfx.mp4".format(stem))
    work_dir = os.path.join(AGENT_ROOT, "uploads", "caption_fx", stem)

    yield step("analyze", "分析参考字幕风格 + 编排",
               "VLM 抽帧读参考字幕(配色/字号/位置/特效) + 像素级取色校准 → 成片词级 ASR 对齐 "
               "→ LLM 按该风格拆块 → 整块弹出烧录。首次分析一条新参考视频较慢, 之后按视频指纹命中缓存。",
               state="running")
    try:
        done = burn(src, tts_items or [], out_path, work_dir, ref_video=ref)
    except Exception as exc:  # noqa: BLE001
        _log.error("[%s] caption_fx failed: %s", rid, exc, exc_info=True)
        yield {"type": "error", "message": "字幕特效模仿失败：{}".format(str(exc)[:200])}
        return
    if not done or not os.path.isfile(done):
        yield {"type": "error",
               "message": "未能生成字幕特效成片（可能 WHQ_CAPTION_CLONE=0、无念白或 ASR 不可用），详见服务日志"}
        return

    rel = os.path.relpath(done, AGENT_ROOT).replace(os.sep, "/")
    md = os.path.join(work_dir, "目标字幕清单.md")
    _log.info("[%s] caption_fx done %s", rid, rel)
    yield step("analyze", "分析参考字幕风格 + 编排", "完成，已按参考风格烧录", state="done")
    yield {"type": "caption_fx_done", "video_uri": rel, "final_path": done,
           "inventory_md": md if os.path.isfile(md) else ""}
