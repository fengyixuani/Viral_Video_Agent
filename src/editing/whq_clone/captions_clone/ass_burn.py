"""ASS 逐字揭示烧录(本机 ffmpeg + libass) —— 移植 copy_zimu/v2/burn_from_inventory.py
的 ASS 生成部分 + copy_zimu/ass_render.burn 的 ffmpeg 调用。

与 Split 版差别: 不再解析《字幕清单.md》拿块(那是 CLI 的中间态), 直接吃 target_match
产出的内存 seq(每块已带 els 逐字时间/颜色/字号/位置/特效), 少一层 md round-trip。

坐标/字号以 720x1280 基准, libass 等比缩放到实际分辨率。放射线特效不支持(需集中线素材,
target_match 已剔除)。

揭示方式(WHQ_CAPTION_REVEAL): block=整块一次弹出(默认), char=逐字揭示。整块弹出时块内不再
逐字控 alpha —— 短块节奏更像抖音花字, 也不受成片词级 ASR 逐字时间偏差影响; 块的出现时刻仍是
首字语音起点。

入场动画(WHQ_CAPTION_ANIM): off=直接展示(默认, 无缩放/淡入, 到点即整块出现),
on=保留弹入/放大镜/缩放回弹。

字号与配色: 正文字号**全片统一**(target_match.base_style 取参考口播档), 不再按角色跳大字;
参考的彩色只用在**块内关键词**上 —— els 里 hot=True 的字按 WHQ_CAPTION_HOT_SCALE 放大并换成
accent 色, 同块其余字保持正文样式。
"""
import os
import subprocess

from _common import FFMPEG

from . import style_spec as S

# 位置(中文) -> (\an, x, y) on 720x1280 基准。中部不用(挡脸)。
_POS = {
    "顶部居中": (8, 360, 200), "顶部偏左": (7, 90, 200), "顶部偏右": (9, 630, 200),
    "中部偏左": (4, 90, 620), "中部偏右": (6, 630, 620), "中部居中": (5, 360, 620),
    "底部居中": (2, 360, 1180), "底部偏左": (1, 90, 1180), "底部偏右": (3, 630, 1180),
}
_BIG_ROLES = ("emphasis", "highlight", "hook")   # 兼容旧数据用, 字号/配色已不再按角色分档
# 块末字结束后额外停留(秒)
TAIL = 0.30
# 边缘安全: 基准画布宽 720, 两侧留白, 文字缩放/换行不超此宽度(不出画面)。
BASE_W = 720
SIDE_MARGIN = 44
SAFE_W = BASE_W - 2 * SIDE_MARGIN
# 折行阈值: 比 SAFE_W 略宽。超 SAFE_W 一点点(9 字带放大关键词 ≈ 643px)就折行会让大半块变两行,
# 而 643px 距画面边缘还有 38px, 并不出画。真正超过 WRAP_W 才折。
WRAP_W = BASE_W - 2 * 24
# 为「尽可能一行」允许压到的最小字号(见 uniform_size)。压到这个还排不下的块, 先在词边界
# 拆成前后两条单行字幕(_split_overwide); 拆不动(找不到断点)才退回折两行。
MIN_SIZE = int(os.getenv("WHQ_CAPTION_MIN_SIZE", "52"))


def reveal_mode():
    """揭示方式: "block"(整块一次弹出, 默认) / "char"(逐字揭示)。"""
    v = (os.getenv("WHQ_CAPTION_REVEAL") or "block").strip().lower()
    return "char" if v in ("char", "chars", "逐字", "字") else "block"


def hot_scale():
    """关键词相对正文的放大倍数(WHQ_CAPTION_HOT_SCALE, 默认 1.25; 1.0=只换色不变大)。"""
    try:
        v = float(os.getenv("WHQ_CAPTION_HOT_SCALE", "1.25"))
    except ValueError:
        v = 1.25
    return min(2.0, max(1.0, v))



def anim_enabled():
    """是否要入场动画(缩放弹入/淡入)。默认关: 用户要「直接展示字幕」, 不要飞入。"""
    v = (os.getenv("WHQ_CAPTION_ANIM") or "off").strip().lower()
    return v in ("1", "on", "true", "yes", "开")


def hex_to_ass(hexstr):
    """#RRGGBB -> ASS &H00BBGGRR&。"""
    h = (hexstr or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        return "&H00FFFFFF&"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return "&H00{}{}{}&".format(b, g, r).upper()


def _cs(t):
    """秒 -> ASS 时间戳 h:mm:ss.cc。"""
    t = max(0.0, t)
    return "{:d}:{:02d}:{:05.2f}".format(int(t // 3600), int((t % 3600) // 60), t % 60)


def _header():
    """单条流 ASS 头(720x1280 基准)。位置/字号/颜色全走内联覆盖。"""
    style = ("Style: cap,{font},72,&H00FFFFFF,&H000000FF,&H00101010,&H90000000,"
             "-1,0,0,0,100,100,1,0,1,5.0,2.0,2,40,40,0,1").format(font=S.FONT_MAIN)
    return "\n".join([
        "[Script Info]", "Title: whq_clone_captions", "ScriptType: v4.00+",
        "PlayResX: {}".format(S.PLAY_RES_X), "PlayResY: {}".format(S.PLAY_RES_Y),
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
         "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
         "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"),
        style, "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]) + "\n"


def _char_units(e):
    """CJK 约占 1 个字身宽, 拉丁/数字约 0.55; 关键词按放大倍数加宽(用于边缘安全估宽)。"""
    ch = e["ch"] if isinstance(e, dict) else e
    u = 1.0 if ord(ch) >= 0x2E80 else 0.55
    return u * hot_scale() if isinstance(e, dict) and e.get("hot") else u


def _units(els, lo, hi):
    return sum(_char_units(e) for e in els[lo:hi]) or 1.0


def _wrap_at(els):
    """选换行点: 只在词边界(els 的 brk)下刀、**不切开关键词**, 并让较宽那半最窄。

    两个坑都踩过: ① 取"离中点最近的词边界"会切出 9+6 的偏斜两行, 宽的那半仍超宽, 又得缩字号
    (66 -> 63), 字号就不一致了; ② 词边界可能落在关键词内部(「致敏|信息」), 把要突出的词劈成
    两行。所以按宽度均衡选点, 并排除落在 hot 连续段内部的候选。无可用候选时退回中点。
    """
    n = len(els)
    if n < 2:
        return -1
    cands = [i for i in range(1, n)
             if els[i - 1].get("brk") and not (els[i - 1].get("hot") and els[i].get("hot"))]
    if not cands:
        cands = [(n + 1) // 2]
    return min(cands, key=lambda i: max(_units(els, 0, i), _units(els, i, n)))


def _fit(b, base):
    """返回 (字号, 换行下标 brk)。字号一律用全片统一的 base。

    超过 WRAP_W 的块**折两行**(在词边界), 而不是把字号缩小 —— 字号一致优先于行数一致;
    折完两行里较宽那半**仍**超 WRAP_W, 才对这一块缩字号(缩到 SAFE_W 以内, 极长块才会发生)。
    关键词那几个字按 hot_scale 放大, 估宽时已计入(见 ``_char_units``), 不会因放大而出画面。
    """
    els = b["els"]
    n = len(els)
    if _units(els, 0, n) * base <= WRAP_W:
        return base, -1
    brk = _wrap_at(els)
    per = max(_units(els, 0, brk), _units(els, brk, n))
    return (base if per * base <= WRAP_W else max(MIN_SIZE, int(SAFE_W / per))), brk


def _split_overwide(blocks):
    """把「压到最小字号也排不成一行」的块在词边界拆成两条单行块, 替代折两行。

    拆出的两条各带自己的逐字时间(els), 前后相继弹出, 视觉上就是两条单行字幕;
    role/颜色/位置/特效原样继承, 关键词不会被切开(_wrap_at 已排除 hot 内部断点)。
    极长块拆一次可能仍超宽, 递归再拆。找不到可用断点的块原样保留, 交给 _fit 折两行兜底。
    额外收益: 全片统一字号由最宽块决定, 拆掉超宽块后统一字号能从下限回升, 字更大。
    """
    out = []
    for b in blocks:
        els = b.get("els") or []
        n = len(els)
        if n < 2 or _units(els, 0, n) * MIN_SIZE <= WRAP_W:
            out.append(b)
            continue
        brk = _wrap_at(els)
        if brk <= 0 or brk >= n:
            out.append(b)
            continue
        first, second = dict(b), dict(b)
        first["els"], second["els"] = els[:brk], els[brk:]
        out.extend(_split_overwide([first, second]))
    return out


def uniform_size(blocks):
    """全片统一的正文字号: 取「每块都能排成一行」的最大字号, 下限 MIN_SIZE。

    用户要求「字幕尽可能一行」+「大小保持一致」, 两者一起满足只有这一种解: 字号由**最宽的那块**
    决定、全片共用。所以最长块会把字号压下来一些(参考口播档 66 -> 常见 52~60), 换来的是绝大多数
    块都是单行、且没有忽大忽小。压到 MIN_SIZE 还排不下的块才交给 ``_fit`` 折两行。
    下限可用 WHQ_CAPTION_MIN_SIZE 调: 调大 -> 更多块换行; 调小 -> 字更小但都是一行。
    """
    base = max((int(b.get("size") or 66) for b in blocks), default=66)
    if not blocks:
        return base
    floor = min(base, MIN_SIZE)
    need = min(int(WRAP_W / _units(b["els"], 0, len(b["els"]))) for b in blocks)
    return max(floor, min(base, need))


def _lead_tags(b, tilt, size):
    """块首内联覆盖: 位置 + 正文字号/填充/描边 + 静态样式(斜排/描边发光) + 可选入场动画。"""
    an, x, y = _POS.get(b.get("position"), _POS["底部居中"])
    eff = b.get("effects") or []
    p = ["\\an{}".format(an), "\\pos({},{})".format(x, y),
         "\\fs{}".format(size), "\\1c{}".format(hex_to_ass(b.get("color"))),
         "\\3c{}".format(hex_to_ass(b.get("outline") or "#101010")),
         "\\bord5.0", "\\shad2.0"]
    if "斜排" in eff and b["role"] != "narration":  # 口播一律正字
        p.append("\\frz{}".format(tilt))
    if "描边发光" in eff:
        p.append("\\blur2")
    if not anim_enabled():  # 直接展示: 不加任何缩放/淡入, 到点整块出现
        return "{" + "".join(p) + "}"
    if "弹入" in eff:
        p.append("\\fscx45\\fscy45\\t(0,170,\\fscx112\\fscy112)"
                 "\\t(170,300,\\fscx100\\fscy100)\\fad(0,120)")
    elif "放大镜" in eff:
        p.append("\\fscx70\\fscy70\\t(0,150,\\fscx120\\fscy120)\\t(150,280,\\fscx100\\fscy100)")
    else:
        p.append("\\fscx72\\fscy72\\t(0,160,\\fscx105\\fscy105)\\t(160,260,\\fscx100\\fscy100)")
    return "{" + "".join(p) + "}"


def _render_block(b, disp_end, tilt, base):
    """单块字幕: 正文统一字号/白字, 只有关键词(els 的 hot)变大换色。

    block 模式整块一次弹出(块内不控 alpha); char 模式逐字揭示。
    """
    els = b["els"]
    cstart = els[0]["start"]
    size, brk = _fit(b, base)
    hot_fs = int(round(size * hot_scale()))
    base_c = hex_to_ass(b.get("color"))
    hot_c = hex_to_ass(b.get("accent") or b.get("color"))
    parts = [_lead_tags(b, tilt, size)]
    by_char = reveal_mode() == "char"
    hot_on = False
    for i, e in enumerate(els):
        if i == brk:
            parts.append("\\N")
        if bool(e.get("hot")) != hot_on:
            hot_on = not hot_on
            parts.append("{{\\fs{fs}\\1c{c}}}".format(fs=hot_fs if hot_on else size,
                                                      c=hot_c if hot_on else base_c))
        if by_char:
            r0 = int(max(0.0, e["start"] - cstart) * 1000)
            parts.append("{{\\alpha&HFF&\\t({r0},{r1},\\alpha&H00&)}}{ch}".format(
                r0=r0, r1=r0 + 60, ch=e["ch"]))
        else:
            parts.append(e["ch"])
    return "Dialogue: 0,{s},{e},cap,,0,0,0,,{t}".format(
        s=_cs(cstart), e=_cs(disp_end), t="".join(parts))



def build_ass(seq):
    """单条流 ASS: 按 start 排序, disp_end 不越过下一块 start; 斜排块左右交替倾斜。

    正文字号先按最长块算一次(``uniform_size``), 全片共用 —— 保证字幕大小一致。
    """
    blocks = sorted((b for b in seq if b.get("els")), key=lambda b: b["els"][0]["start"])
    n_before = len(blocks)
    blocks = sorted(_split_overwide(blocks), key=lambda b: b["els"][0]["start"])
    if len(blocks) > n_before:
        print("[captions_clone] 超宽块拆分: {} -> {} 块(超宽的改为两条单行)".format(
            n_before, len(blocks)), flush=True)
    base = uniform_size(blocks)
    out = []
    tilt_sign = 1
    wrapped = 0
    for i, b in enumerate(blocks):
        els = b["els"]
        cstart = els[0]["start"]
        disp_end = els[-1]["end"] + TAIL
        if i + 1 < len(blocks):
            disp_end = min(disp_end, blocks[i + 1]["els"][0]["start"] - 0.02)
        disp_end = max(disp_end, cstart + 0.45)
        tilt = 0
        if "斜排" in (b.get("effects") or []) and b["role"] != "narration":
            tilt = 6 * tilt_sign
            tilt_sign *= -1
        wrapped += 1 if _fit(b, base)[1] >= 0 else 0
        out.append(_render_block(b, disp_end, tilt, base))
    print("[captions_clone] 统一字号 {}px(关键词 x{}), 单行 {} 块 / 折两行 {} 块".format(
        base, hot_scale(), len(blocks) - wrapped, wrapped), flush=True)
    return _header() + "\n".join(out) + "\n"


def burn(seq, video_in, video_out, work_dir):
    """生成 ASS 并用 ffmpeg/libass 烧进视频。返回输出路径; 无块可烧时返回 None。"""
    if not seq:
        return None
    os.makedirs(work_dir, exist_ok=True)
    ass_path = os.path.join(work_dir, "captions_clone.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(build_ass(seq))
    ass_esc = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)
    subprocess.run([
        FFMPEG, "-y", "-i", video_in,
        "-map", "0:v:0", "-vf", "subtitles='{}'".format(ass_esc),
        "-map", "0:a?", "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart", video_out,
    ], check=True)
    print("[captions_clone] ASS 烧录 -> {}".format(video_out), flush=True)
    return video_out
