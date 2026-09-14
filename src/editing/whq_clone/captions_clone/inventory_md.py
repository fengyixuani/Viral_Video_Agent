"""渲染参考视频《字幕清单.md》(两表) —— 移植 copy_zimu/v2/inventory_md.py。

  表① 样式与颜色规范: 样式 | 位置 | 目测色 | 取色校准值 | ASS填充 | ASS描边 | 特效
  表② 完整时间轴:     时间 | 内容 | 位置 | 颜色 | 目测色 | 样式/特效

「目测色」= VLM 目测 hint; 「取色校准值」= 像素级聚类校准值(修正红橙/金黄丢失)。
这份 md 只作留档/人读, 机器消费走同名 style_profile.json。
"""
import os

from .color_calib import best_fill_hex as _best_fill

_ROLE_CN = {
    "narration": "口播", "emphasis": "强调大字", "highlight": "句内高亮",
    "hook": "标题钩子", "label": "标签/角标",
}
_POS_CN = {"top": "顶部", "middle": "中部", "bottom": "底部"}
_H_CN = {"left": "偏左", "center": "居中", "right": "偏右"}


def _cap_fx(c):
    """样式/特效列: 角色 + 斜排 + effect[] + annotation[](去重)。"""
    parts = [_ROLE_CN.get(c.get("role"), c.get("role"))]
    if c.get("slant"):
        parts.append("斜排")
    for e in c.get("effect") or []:
        if e not in parts:
            parts.append(e)
    annos = c.get("annotation") or []
    if annos:
        parts.append("标注:" + "/".join(annos))
    return " ".join(parts)


def _style_table(profile):
    rows = [
        "| 样式类 | 角色 | 位置(an) | 目测色(填/描) | 取色校准值(填/描) | ASS 填充 | ASS 描边 | 字号 | 斜排 | 特效 | 占比 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, st in profile.get("styles", {}).items():
        calib_fill = st.get("fill_hex", "-")
        calib_out = st.get("outline_hex", "-")
        rows.append(
            "| {name} | {role} | {pos}({an}) | {hf}/{ho} | {cf}/{co}(校{cs:.0%}) | `{c}` | `{o}` | {size} | {slant} | {fx} | {share} |".format(
                name=name, role=_ROLE_CN.get(st.get("role"), st.get("role")),
                pos=_POS_CN.get(st.get("position"), st.get("position")), an=st.get("an"),
                hf=st.get("fill_hex_hint", calib_fill), ho=st.get("outline_hex_hint", calib_out),
                cf=calib_fill, co=calib_out, cs=st.get("calib_share", 0.0),
                c=st.get("color", "-"), o=st.get("outline", "-"), size=st.get("size"),
                slant=("是 %d°" % st["slant"]) if st.get("slant") else "否",
                fx="、".join(st.get("effects") or []) or "-", share=st.get("share", 0.0)))
    return "\n".join(rows)


def _timeline_table(inventory):
    rows = [
        "| 时间(s) | 内容 | 位置 | 颜色(校准优先) | 目测色 | 样式/特效 |",
        "|---|---|---|---|---|---|",
    ]
    for c in inventory.get("captions", []):
        rows.append("| {ts:.1f}–{te:.1f} | {text} | {pos}{h} | {col} | {hint} | {fx} |".format(
            ts=c["t_start"], te=c["t_end"], text=c["text"],
            pos=_POS_CN.get(c.get("v"), c.get("v")), h=_H_CN.get(c.get("h"), ""),
            col=_best_fill(c)[0], hint=c.get("fill_hex", "-"), fx=_cap_fx(c)))
    return "\n".join(rows)


def render_md(inventory, profile):
    """-> 参考《字幕清单》markdown 文本(两表, 含目测/校准双列)。"""
    caps = inventory.get("captions", [])
    calib_n = sum(1 for c in caps if c.get("fill_hex_calib"))
    return "\n".join([
        "# 参考字幕清单（自动分析·像素校准）: {}".format(
            os.path.basename(inventory.get("video") or "")),
        "",
        "> 由 whq_clone/captions_clone 生成: VLM 抽帧读字/位置/特效 + 原生分辨率 PNG 像素级取色校准。",
        "> 「目测色」为 VLM 目测 hint; 「取色校准值」为像素聚类校准值(修正目测偏色, 如红橙/金黄)。",
        "",
        "- 时长: {:.1f}s  抽帧数: {}  抽帧率: {}fps  校准命中: {}/{} 条".format(
            float(inventory.get("duration") or 0), inventory.get("frame_count", 0),
            inventory.get("fps"), calib_n, len(caps)),
        "- 字幕密度: **{}** (有字幕 {:.1f}s / 总 {:.1f}s)".format(
            profile.get("density"), profile.get("covered_seconds", 0.0),
            float(inventory.get("duration") or 0)),
        "",
        "## 一、样式与颜色规范",
        "",
        _style_table(profile),
        "",
        "## 二、完整时间轴",
        "",
        _timeline_table(inventory),
        "",
    ]) + "\n"
