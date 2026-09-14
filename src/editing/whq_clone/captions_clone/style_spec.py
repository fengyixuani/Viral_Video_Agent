"""字幕样式常量(移植自 Split 仓 copy_zimu/style_spec.py 的可达子集)。

只保留 style_profile / ass_burn 真正用到的项: ASS 颜色、主字体、斜排角度、full 模式
口播/大字预设(VLM 不可用时的回退风格)。ASS 颜色格式 &HAABBGGRR, alpha=00 为不透明。
"""
import os

# 基准画布(ASS PlayRes): 720x1280 竖屏, libass 会等比缩放到实际分辨率。
PLAY_RES_X = 720
PLAY_RES_Y = 1280

WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
# 内联金黄关键词高亮 (#FFDC1E -> BGR 1EDCFF)
HIGHLIGHT_YELLOW = "&H001EDCFF"
# 顶部红橙大字填充 / 暗红描边(参考视频常见的彩色斜排强调)
FULL_BIG_FILL = "&H001737EA"
FULL_BIG_OUTLINE = "&H00102A6E"

# 中文主字体。libass 按 fontconfig 名字查找, 需系统已装(Agent 侧与 finisher 一致)。
FONT_MAIN = os.getenv("WHQ_CAPTION_FONT", "Noto Sans CJK SC")

# 斜排(\frz)角度
SLANT_DEG = 8

# 回退预设: 底部口播白字 / 顶部红橙斜排大字
FULL_NRM = dict(color=WHITE, outline="&H00101010", ow=5.0, sh=2.0, size=72, an=2, slant=0)
FULL_BIG = dict(color=FULL_BIG_FILL, outline=FULL_BIG_OUTLINE, ow=6.0, sh=4.0, size=128,
                an=8, slant=5)
