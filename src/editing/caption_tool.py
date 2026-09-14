"""字幕烧录工具（剪辑 Agent 可调用的能力）。

把一段文字渲染成整帧透明 PNG（底部半透明黑底 + 白字，按竖屏宽度自动换行），
再由剪辑器用 ffmpeg overlay 叠加到画面上——因为随包的 imageio-ffmpeg 静态构建
**没有 drawtext（缺 libfreetype）**，但都带 overlay。

剪辑 Agent 通过在某镜 clip 里给出 caption 文本来"使用"这个工具；把 burn_caption
设为 false 则跳过该镜字幕。字幕文本应使用所选素材片段自己的口播原话（speech），
保证与成片保留的原声一致。
"""
from __future__ import annotations

import glob
import os
import subprocess

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

# PIL 的 truetype() 只认**字体文件路径**(不像 libass 能问 fontconfig 要字体名), 所以这里必须
# 自己找到一个能显示中文的字体文件。按机器上常见的 CJK 字体包依次找, 再退化到 fc-match。
# CAPTION_FONT 环境变量可直接指定字体文件, 优先级最高。
_FONT_GLOBS = (
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.tt[cf]",
    "/usr/share/fonts/**/NotoSansSC*.tt[fc]",
    "/usr/share/fonts/**/SourceHanSans*.tt[cf]",
    "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    "/usr/share/fonts/**/DroidSansFallback.ttf",
    "/usr/share/fonts/**/wqy-*.tt[cf]",
)


def _fc_match_zh():
    """问 fontconfig 要一个能显示中文的字体文件路径(最后兜底)。取不到返回 ""。"""
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", ":lang=zh"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return out if out and os.path.isfile(out) else ""


def _find_cjk_font():
    """按候选路径 + fc-match 找一个可用的中文字体文件; 找不到返回 ""。"""
    env = os.getenv("CAPTION_FONT", "").strip()
    if env:
        if os.path.isfile(env):
            return env
        print("[caption_tool] CAPTION_FONT 指定的字体不存在: {}".format(env), flush=True)
    for pat in _FONT_GLOBS:
        if any(ch in pat for ch in "*?["):
            hits = sorted(glob.glob(pat, recursive=True))
            if hits:
                return hits[0]
        elif os.path.isfile(pat):
            return pat
    return _fc_match_zh()


_CJK_FONT = _find_cjk_font() if _PIL_OK else ""

# 是否具备烧字幕能力（PIL 可用且找到 CJK 字体）
CAPTION_AVAILABLE = _PIL_OK and bool(_CJK_FONT)
if not CAPTION_AVAILABLE:
    # 以前这里是静默的: 缺字体时 render_caption 直接返回 False, 成片无字幕且不报错, ops 里
    # 只留一句"字幕未烧", 排查要翻到最底层。改成启动即告警。
    print("[caption_tool] 警告: 无法烧字幕(PIL={} 中文字体={}); 请装 CJK 字体或用 "
          "CAPTION_FONT 指定字体文件".format(_PIL_OK, _CJK_FONT or "未找到"), flush=True)
else:
    print("[caption_tool] 字幕字体: {}".format(_CJK_FONT), flush=True)


def render_caption(text: str, width: int, height: int, out_png: str) -> bool:
    """把字幕渲染成 width×height 的透明 PNG（底部半透明黑底 + 白字，自动按宽度换行）。

    返回是否成功渲染（文本为空或缺字体/PIL 时返回 False）。
    """
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return False
    if not CAPTION_AVAILABLE:
        print("[caption_tool] 跳过烧字幕(无可用中文字体): {}".format(text[:20]), flush=True)
        return False
    font_size = max(28, int(width * 0.055))
    try:
        font = ImageFont.truetype(_CJK_FONT, font_size)
    except Exception as exc:  # noqa: BLE001  字体文件损坏/ttc 索引不可用等
        print("[caption_tool] 字体加载失败({}): {}".format(_CJK_FONT, str(exc)[:120]), flush=True)
        return False
    max_w = int(width * 0.88)
    # 按像素宽度换行（CJK 无空格，逐字累加）
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        w = font.getbbox(trial)[2]
        if w > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = lines[:4]
    line_h = int(font_size * 1.35)
    block_h = line_h * len(lines)
    pad = int(font_size * 0.5)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    box_top = height - block_h - pad * 2 - int(height * 0.06)
    draw.rectangle([int(width * 0.04), box_top, int(width * 0.96), box_top + block_h + pad * 2],
                   fill=(0, 0, 0, 150))
    y = box_top + pad
    for ln in lines:
        w = font.getbbox(ln)[2]
        draw.text(((width - w) // 2, y), ln, font=font, fill=(255, 255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0, 220))
        y += line_h
    img.save(out_png)
    return True
