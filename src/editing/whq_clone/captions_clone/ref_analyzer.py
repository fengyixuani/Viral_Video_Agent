"""参考视频字幕识别(VLM 抽帧, 带归一 bbox + 特效 + 标注) —— 移植 copy_zimu/v2/ref_analyzer.py。

与 Split 版的唯一差别: VLM 不再走 Split 仓 ``common/pipeline_utils`` 的千帆多图, 而是走
Agent 的 wenchain 网关(``pipeline_utils.ask_qianfan`` + ``as_core.VISION_MODEL``), 帧图以
``image_url`` + ``data:image/jpeg;base64`` content block 传入(同 as_core 既有约定)。

每条字幕输出:
  - bbox: [x0,y0,x1,y1] 归一坐标(0~1), 供 color_calib 像素取色裁剪。
  - effect[] / annotation[]: 特效与画面标注标签。
  - fill_hex/outline_hex: VLM 目测色(hint), 真值由 color_calib 校准。
"""
import base64
import os
import re

import as_core
from pipeline_utils import ask_qianfan, loads_with_repair, parallel_map

from .frames import extract_frames, duration_seconds

_PROMPT = (
    "你在分析一条竖屏带货短视频的【字幕特效风格】。下面按时间顺序给出若干视频帧, 每帧前标注"
    "了它的时间戳(秒)。请【只关注叠加在画面上的文字字幕/标注元素】(不要描述人脸、产品、背景)。\n"
    "对每一帧, 列出该帧上可见的每一条字幕文字, 逐条给出:\n"
    "  - text: 该条字幕的文字(原样, 去掉多余空格)\n"
    "  - bbox: 该条字幕文字在整帧中的包围盒, 归一坐标 [x0,y0,x1,y1], 0~1, 原点在左上,"
    " x 向右 y 向下; 尽量贴合文字外缘(供精确取色, 这是最重要的字段之一)\n"
    "  - v: 竖直位置, 只能是 top | middle | bottom\n"
    "  - h: 水平位置, 只能是 left | center | right\n"
    "  - fill_hex: 文字填充色, 十六进制如 #FFFFFF(白)/#EA3717(红橙)/#FFDC1E(金黄), 尽量准确目测\n"
    "  - outline_hex: 文字描边色, 十六进制(如 #101010 黑)\n"
    "  - size: 相对字号, 只能是 small | normal | big\n"
    "  - slant: 是否斜排(倾斜), true | false\n"
    "  - effect: 字符串数组, 从 [斜排,弹入,逐字揭示,放射线,爆闪,对比标签,放大镜,描边发光] 里"
    "选出适用的(可为空数组)\n"
    "  - annotation: 字符串数组, 画面上伴随该字幕的标注元素, 从 [红X,对勾,箭头,放大镜,emoji,"
    "注释小字,角标] 里选(可为空数组)\n"
    "  - role: 该字幕的作用, 只能是 narration(逐句口播) | emphasis(强调大字/punch词) | "
    "highlight(句内高亮关键词) | hook(标题钩子) | label(标签/角标)\n"
    "若某帧没有任何字幕, captions 给空数组。严格只返回 JSON:\n"
    '{"frames": [{"t": <帧时间戳数字>, "captions": [{"text":"..","bbox":[0,0,0,0],"v":"..",'
    '"h":"..","fill_hex":"#..","outline_hex":"#..","size":"..","slant":false,"effect":[],'
    '"annotation":[],"role":".."}]}]}'
)


def default_model():
    """字幕分析用视觉模型: WHQ_CAPTION_VLM_MODEL 优先, 否则复用 Agent 的 VISION_MODEL。"""
    return os.getenv("WHQ_CAPTION_VLM_MODEL") or as_core.VISION_MODEL


def _image_data_url(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def _norm_text(t):
    return re.sub(r"\s+", "", str(t or ""))


def _norm_bbox(raw):
    """把 VLM 给的 bbox 归一到 [x0,y0,x1,y1] 且 0~1、有序; 解析失败返回 None。"""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        vals = [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    # 若模型误用 0~100 或像素, 按最大值缩放归一
    m = max(abs(v) for v in vals)
    if m > 1.5:
        vals = [v / m for v in vals]
    x0, y0, x1, y1 = vals
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _norm_list(raw):
    out = []
    for v in raw or []:
        s = _norm_text(v)
        if s and s not in out:
            out.append(s)
    return out


def _vlm_batch(frames, model):
    """frames: [(t_sec, jpg_path)] -> [{'t':float,'captions':[...]}]。单批调用 VLM。"""
    content = [{"type": "text", "text": _PROMPT}]
    for t, path in frames:
        content.append({"type": "text", "text": "帧 t={:.2f}s:".format(t)})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    text, _raw = ask_qianfan([{"role": "user", "content": content}], model=model,
                             max_tokens=6144, temperature=0.1)
    data = loads_with_repair(text)
    out = []
    for fr in data.get("frames") or []:
        try:
            t = float(fr.get("t"))
        except (TypeError, ValueError):
            continue
        caps = []
        for c in fr.get("captions") or []:
            txt = _norm_text(c.get("text"))
            if not txt:
                continue
            caps.append({
                "text": txt,
                "bbox": _norm_bbox(c.get("bbox")),
                "v": (c.get("v") or "bottom").lower(),
                "h": (c.get("h") or "center").lower(),
                "fill_hex": (c.get("fill_hex") or "").strip(),
                "outline_hex": (c.get("outline_hex") or "").strip(),
                "size": (c.get("size") or "normal").lower(),
                "slant": bool(c.get("slant")),
                "effect": _norm_list(c.get("effect")),
                "annotation": _norm_list(c.get("annotation")),
                "role": (c.get("role") or "narration").lower(),
            })
        out.append({"t": t, "captions": caps})
    return out


def _mode(values, default=None):
    vals = [v for v in values if v]
    if not vals:
        return default
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _median_bbox(bboxes):
    bbs = [b for b in bboxes if b]
    if not bbs:
        return None

    def med(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    cols = list(zip(*bbs))
    return [round(med(cols[i]), 4) for i in range(4)]


def _finalize_run(text, run, step):
    ts = [t for t, _ in run]
    caps = [c for _, c in run]
    effects, annos = [], []
    for c in caps:
        for e in c.get("effect") or []:
            if e not in effects:
                effects.append(e)
        for a in c.get("annotation") or []:
            if a not in annos:
                annos.append(a)
    return {
        "text": text,
        "t_start": round(min(ts), 2),
        "t_end": round(max(ts) + step, 2),
        "bbox": _median_bbox([c.get("bbox") for c in caps]),
        "v": _mode([c["v"] for c in caps], "bottom"),
        "h": _mode([c["h"] for c in caps], "center"),
        "fill_hex": _mode([c["fill_hex"] for c in caps], "#FFFFFF"),
        "outline_hex": _mode([c["outline_hex"] for c in caps], "#101010"),
        "size": _mode([c["size"] for c in caps], "normal"),
        "slant": _mode([("slant" if c["slant"] else "flat") for c in caps], "flat") == "slant",
        "effect": effects,
        "annotation": annos,
        "role": _mode([c["role"] for c in caps], "narration"),
        "frames": len(run),
    }


def _aggregate(observations, step):
    """逐帧观察合并成逐条字幕: 同一文本相邻帧连续出现 -> 一条(t_start~t_end)。"""
    by_text = {}
    for fr in observations:
        for cap in fr["captions"]:
            by_text.setdefault(cap["text"], []).append((fr["t"], cap))
    gap = max(step * 2.2, 1.2)
    items = []
    for text, seq in by_text.items():
        seq.sort(key=lambda x: x[0])
        run, prev_t = [], None
        for t, cap in seq:
            if prev_t is not None and t - prev_t > gap:
                items.append(_finalize_run(text, run, step))
                run = []
            run.append((t, cap))
            prev_t = t
        if run:
            items.append(_finalize_run(text, run, step))
    items.sort(key=lambda c: c["t_start"])
    return items


def analyze(video, work_dir, model=None, fps=1.0, batch=8):
    """参考视频 -> 原始字幕清单 dict(每条含 bbox/effect/annotation)。

    返回 {'video','duration','fps','frame_count','captions':[逐条]}。
    单批 VLM 失败不阻断(该批丢弃), 全部失败则 captions 为空, 由上层回退预设。
    """
    model = model or default_model()
    frames = extract_frames(video, os.path.join(work_dir, "frames"), fps=fps, prefix="ref")
    step = 1.0 / fps if fps and fps > 0 else 1.0
    chunks = [frames[i:i + batch] for i in range(0, len(frames), batch)]
    observations = []
    for i, res in enumerate(parallel_map(lambda ch: _vlm_batch(ch, model), chunks)):
        if isinstance(res, Exception):
            print("[captions_clone] VLM 批 {} 失败(跳过): {}".format(i, str(res)[:160]), flush=True)
            continue
        observations.extend(res)
    return {
        "video": video,
        "duration": round(duration_seconds(video), 2),
        "fps": fps,
        "frame_count": len(frames),
        "captions": _aggregate(observations, step),
    }
