"""字幕烧录工具（剪辑 Agent 可调用的能力）。

把一段文字渲染成整帧透明 PNG（底部半透明黑底 + 白字，按竖屏宽度自动换行），
再由剪辑器用 ffmpeg overlay 叠加到画面上——因为随包的 imageio-ffmpeg 静态构建
**没有 drawtext（缺 libfreetype）**，但都带 overlay。

剪辑 Agent 通过在某镜 clip 里给出 caption 文本来"使用"这个工具；把 burn_caption
设为 false 则跳过该镜字幕。字幕文本应使用所选素材片段自己的口播原话（speech），
保证与成片保留的原声一致。
"""
from __future__ import annotations

import os

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

_CJK_FONT = next((f for f in [
    "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
] if os.path.isfile(f)), "")

# 是否具备烧字幕能力（PIL 可用且找到 CJK 字体）
CAPTION_AVAILABLE = _PIL_OK and bool(_CJK_FONT)


def render_caption(text: str, width: int, height: int, out_png: str) -> bool:
    """把字幕渲染成 width×height 的透明 PNG（底部半透明黑底 + 白字，自动按宽度换行）。

    返回是否成功渲染（文本为空或缺字体/PIL 时返回 False）。
    """
    text = str(text or "").replace("\n", " ").strip()
    if not text or not CAPTION_AVAILABLE:
        return False
    font_size = max(28, int(width * 0.055))
    font = ImageFont.truetype(_CJK_FONT, font_size)
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
