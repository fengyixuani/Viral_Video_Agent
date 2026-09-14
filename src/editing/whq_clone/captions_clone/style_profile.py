"""蒸馏样式档案 style_profile(校准色优先) —— 合并移植 copy_zimu v1+v2 的 style_profile。

把 ref_analyzer + color_calib 的逐条清单归纳成【命名样式类 + 密度】。role 映射到 4 个桶:

  narration -> narration   (逐句口播, 通常白字底部)
  emphasis/hook -> emphasis (强调大字/钩子, 顶部, 可斜排, 常红橙)
  highlight -> highlight    (句内高亮关键词)
  label -> label            (标签/角标)

颜色优先取【像素校准色】(fill_hex_calib), 低置信/无校准回退【VLM 目测色】。
颜色统一转 ASS &HAABBGGRR。密度 = 有字幕时长/总时长, >= FULL_DENSITY 记 full。
VLM/像素都不可用时 fallback() 回退到 style_spec 预设(白字口播 + 红橙斜排大字 + 金黄高亮)。
"""
import re

from . import style_spec as S
from .color_calib import best_fill_hex as _best_fill, best_outline_hex as _best_outline

FULL_DENSITY = 0.55

_ROLE_BUCKET = {
    "narration": "narration",
    "emphasis": "emphasis",
    "hook": "emphasis",
    "highlight": "highlight",
    "label": "label",
}

# size 关键字 -> 基准字号(相对 720x1280)
_SIZE_PX = {"small": 40, "normal": 66, "big": 120}

_HEX_RE = re.compile(r"#?([0-9a-fA-F]{6})")


def hex_to_ass(hex_str, default="&H00FFFFFF"):
    """#RRGGBB -> ASS &H00BBGGRR(不透明)。解析失败返回 default。"""
    m = _HEX_RE.search(str(hex_str or ""))
    if not m:
        return default
    rr, gg, bb = m.group(1)[0:2], m.group(1)[2:4], m.group(1)[4:6]
    return "&H00{}{}{}".format(bb, gg, rr).upper()


def _an_from_pos(v, h):
    """竖直/水平位置 -> ASS \\an(数字键盘布局)。"""
    row = {"top": 6, "middle": 3, "bottom": 0}.get(v, 0)
    col = {"left": 1, "center": 2, "right": 3}.get(h, 2)
    return row + col


def _covered_seconds(captions):
    """字幕时间段的并集时长(秒), 用于估密度。"""
    spans = sorted((c["t_start"], c["t_end"]) for c in captions if c["t_end"] > c["t_start"])
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _mode(caps, key, default):
    counts = {}
    for c in caps:
        v = c.get(key)
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else default


def _union(caps, key):
    out = []
    for c in caps:
        for v in c.get(key) or []:
            if v not in out:
                out.append(v)
    return out


def _dominant_fill(caps):
    """桶内众数填充色(优先校准色) -> (fill_hex, outline_hex, calib_share)。"""
    fill_counts, out_counts, calib_hits = {}, {}, 0
    for c in caps:
        fh, src = _best_fill(c)
        oh, _ = _best_outline(c)
        fill_counts[fh] = fill_counts.get(fh, 0) + 1
        out_counts[oh] = out_counts.get(oh, 0) + 1
        if src == "calib":
            calib_hits += 1
    fill = max(fill_counts, key=lambda k: fill_counts[k]) if fill_counts else "#FFFFFF"
    outline = max(out_counts, key=lambda k: out_counts[k]) if out_counts else "#101010"
    share = round(calib_hits / len(caps), 3) if caps else 0.0
    return fill, outline, share


def _build_bucket(bucket, caps, total):
    v = _mode(caps, "v", "bottom")
    h = _mode(caps, "h", "center")
    size_kw = _mode(caps, "size", "normal")
    slant = sum(1 for c in caps if c.get("slant")) * 2 >= len(caps)
    fill, outline, calib_share = _dominant_fill(caps)
    an = _an_from_pos(v, h)
    return {
        "role": bucket,
        "color": hex_to_ass(fill, S.WHITE),
        "outline": hex_to_ass(outline, S.BLACK),
        "fill_hex": fill,
        "outline_hex": outline,
        # 目测 hint 众数(供清单双列对照)
        "fill_hex_hint": _mode(caps, "fill_hex", "#FFFFFF"),
        "outline_hex_hint": _mode(caps, "outline_hex", "#101010"),
        "calib_share": calib_share,
        "size": _SIZE_PX.get(size_kw, 66),
        "size_kw": size_kw,
        "an": an,
        "position": {8: "top", 5: "middle", 2: "bottom"}.get(
            an, "top" if v == "top" else "bottom"),
        "slant": (S.SLANT_DEG if slant else 0),
        "effects": _union(caps, "effect"),
        "annotations": _union(caps, "annotation"),
        "count": len(caps),
        "share": round(len(caps) / total, 3) if total else 0.0,
    }


def distill(inventory):
    """ref_analyzer + color_calib 的产物 -> profile dict。"""
    caps = inventory.get("captions") or []
    duration = float(inventory.get("duration") or 0.0)
    if not caps:
        return fallback(inventory.get("video"), duration, reason="empty_inventory")

    by_bucket = {}
    for c in caps:
        by_bucket.setdefault(_ROLE_BUCKET.get(c.get("role"), "narration"), []).append(c)

    total = len(caps)
    styles = {name: _build_bucket(name, cs, total) for name, cs in by_bucket.items()}
    if "narration" not in styles:  # 保证至少有 narration(迁移时的底色)
        styles["narration"] = {
            "role": "narration", "color": S.WHITE, "outline": S.BLACK,
            "fill_hex": "#FFFFFF", "outline_hex": "#101010",
            "fill_hex_hint": "#FFFFFF", "outline_hex_hint": "#101010", "calib_share": 0.0,
            "size": 66, "size_kw": "normal", "an": 2, "position": "bottom", "slant": 0,
            "effects": [], "annotations": [], "count": 0, "share": 0.0,
        }

    covered = _covered_seconds(caps)
    density = "full" if duration > 0 and covered / duration >= FULL_DENSITY else "sparse"
    hl = styles.get("highlight") or {}
    return {
        "source": inventory.get("video"),
        "duration": round(duration, 2),
        "covered_seconds": round(covered, 2),
        "density": density,
        "styles": styles,
        "highlight_color": hl.get("color", S.HIGHLIGHT_YELLOW),
        "calibrated": True,
        "note": "像素级取色校准; 校准色优先、VLM 目测兜底。",
    }


def fallback(video=None, duration=0.0, reason="fallback"):
    """VLM 不可用/清单为空: 回退预设(白字口播 + 红橙斜排大字 + 金黄高亮)。"""
    return {
        "source": video,
        "duration": round(float(duration or 0.0), 2),
        "covered_seconds": 0.0,
        "density": "full",
        "styles": {
            "narration": {
                "role": "narration", "color": S.FULL_NRM["color"],
                "outline": S.FULL_NRM["outline"],
                "fill_hex": "#FFFFFF", "outline_hex": "#101010",
                "fill_hex_hint": "#FFFFFF", "outline_hex_hint": "#101010", "calib_share": 0.0,
                "size": S.FULL_NRM["size"], "size_kw": "normal",
                "an": S.FULL_NRM["an"], "position": "bottom", "slant": S.FULL_NRM["slant"],
                "effects": [], "annotations": [], "count": 0, "share": 0.0,
            },
            "emphasis": {
                "role": "emphasis", "color": S.FULL_BIG["color"],
                "outline": S.FULL_BIG["outline"],
                "fill_hex": "#EA3717", "outline_hex": "#6E2A10",
                "fill_hex_hint": "#EA3717", "outline_hex_hint": "#6E2A10", "calib_share": 0.0,
                "size": S.FULL_BIG["size"], "size_kw": "big",
                "an": S.FULL_BIG["an"], "position": "top", "slant": S.FULL_BIG["slant"],
                "effects": ["斜排", "弹入"], "annotations": [], "count": 0, "share": 0.0,
            },
            "highlight": {
                "role": "highlight", "color": S.HIGHLIGHT_YELLOW, "outline": S.BLACK,
                "fill_hex": "#FFDC1E", "outline_hex": "#101010",
                "fill_hex_hint": "#FFDC1E", "outline_hex_hint": "#101010", "calib_share": 0.0,
                "size": 120, "size_kw": "big", "an": 8, "position": "top", "slant": 0,
                "effects": ["放大镜"], "annotations": [], "count": 0, "share": 0.0,
            },
        },
        "highlight_color": S.HIGHLIGHT_YELLOW,
        "note": "回退预设({}): 无 VLM/清单为空。".format(reason),
    }
