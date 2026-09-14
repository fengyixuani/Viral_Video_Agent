"""像素级取色校准(PIL + numpy) —— 原样移植 copy_zimu/v2/color_calib.py。

VLM 目测取色会把红橙误判成白。这里用 VLM 给的归一 bbox 在【原生分辨率 PNG】上裁剪字幕
区域, 颜色量化聚类后按启发式挑出【填充色】与【描边色】, 修正目测偏色。

启发式(以 VLM 目测 hint 为锚):
  - hint 偏白(低饱和高亮) -> 填充取最亮的高频簇; 但区域若存在明显高饱和簇则认定文字是彩色。
  - hint 有彩(高饱和)     -> 填充取与 hint 色相最近且饱和度足够的高频簇。
  - 描边色恒取亮度最低的高频簇(通常黑/暗色描边)。
  - 只在校准结果【明显彩色】且置信足够时才覆盖 hint(见 _trust), 否则信 VLM 目测。

每条字幕在其存活区间取若干原生帧采样, 逐帧取色后中位聚合, 抗单帧噪声。
"""
import colorsys

try:
    import numpy as np
    from PIL import Image
    _PIL_OK = True
except Exception:  # noqa: BLE001
    _PIL_OK = False

# 选簇阈值
_MIN_SHARE = 0.04          # 簇像素占比低于此不作为候选
_HINT_HUE_TOL = 0.12       # 色相差(0~1)容忍, 超过视为与 hint 不同色系
_WHITE_S_MAX = 0.18        # hint 饱和度低于此认定为"偏白/灰"
_WHITE_L_MIN = 0.60        # 且亮度高于此
_VIVID_S = 0.55            # 簇饱和度高于此视为"明显彩色文字"(排除肤色, 保留红橙)


def hex_to_rgb(hex_str):
    """#RRGGBB -> (r,g,b) 0~255; 解析失败返回 None。"""
    s = str(hex_str or "").lstrip("#").strip()
    if len(s) < 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def rgb_to_hex(rgb):
    """(r,g,b) -> #RRGGBB(大写)。"""
    r, g, b = (int(max(0, min(255, round(v)))) for v in rgb)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def _hls(rgb):
    r, g, b = (v / 255.0 for v in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, l, s


def _hue_dist(h1, h2):
    """环形色相差 0~0.5。"""
    d = abs(h1 - h2) % 1.0
    return min(d, 1.0 - d)


def _crop_pixels(png_path, bbox_norm, inset=0.12):
    """归一 bbox -> 原生像素裁剪, 向内收缩 inset 取中心带(避开边缘背景) -> (ndarray(N,3), ok)。"""
    try:
        im = Image.open(png_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None, False
    W, H = im.size
    x0, y0, x1, y1 = bbox_norm
    bw, bh = (x1 - x0), (y1 - y0)
    x0 += bw * inset
    x1 -= bw * inset
    y0 += bh * inset
    y1 -= bh * inset
    px0, py0 = int(round(x0 * W)), int(round(y0 * H))
    px1, py1 = int(round(x1 * W)), int(round(y1 * H))
    px0, px1 = max(0, min(px0, W - 1)), max(1, min(px1, W))
    py0, py1 = max(0, min(py0, H - 1)), max(1, min(py1, H))
    if px1 - px0 < 2 or py1 - py0 < 2:
        return None, False
    arr = np.asarray(im.crop((px0, py0, px1, py1)), dtype=np.uint8).reshape(-1, 3)
    if arr.shape[0] < 8:
        return None, False
    return arr, True


def _clusters(pixels, n_colors=6):
    """颜色量化聚类 -> [(rgb, count)] 按 count 降序(PIL quantize 中位切分)。"""
    h = int(np.sqrt(pixels.shape[0])) or 1
    w = int(np.ceil(pixels.shape[0] / h))
    pad = h * w - pixels.shape[0]
    buf = pixels
    if pad:
        buf = np.vstack([pixels, np.repeat(pixels[-1:], pad, axis=0)])
    img = Image.fromarray(buf.reshape(h, w, 3).astype(np.uint8), "RGB")
    q = img.quantize(colors=n_colors, method=Image.MEDIANCUT)
    pal = q.getpalette() or []
    counts = q.getcolors() or []
    out = []
    for cnt, idx in counts:
        base = idx * 3
        if base + 2 < len(pal):
            out.append(((pal[base], pal[base + 1], pal[base + 2]), int(cnt)))
    out.sort(key=lambda x: -x[1])
    return out


def _pick_fill(clusters, total, hint_rgb):
    """按 hint 从簇里挑填充色 -> (rgb, share)。

    关键: 即便 hint 目测成白, 只要区域里存在明显高饱和簇, 就认定字幕填充是那抹彩色。
    """
    cands = [(rgb, cnt / total) for rgb, cnt in clusters if cnt / total >= _MIN_SHARE]
    if not cands:
        cands = [(clusters[0][0], clusters[0][1] / total)] if clusters else []
    if not cands:
        return None, 0.0
    hint_hls = _hls(hint_rgb) if hint_rgb else None
    hint_is_white = bool(hint_hls and hint_hls[2] <= _WHITE_S_MAX and hint_hls[1] >= _WHITE_L_MIN)
    vivid = [c for c in cands if _hls(c[0])[2] >= _VIVID_S and 0.18 <= _hls(c[0])[1] <= 0.85]

    if hint_is_white:
        if vivid:
            return max(vivid, key=lambda c: (_hls(c[0])[2] * 0.55 + c[1] * 0.45))
        return max(cands, key=lambda c: _hls(c[0])[1] * (1.0 - _hls(c[0])[2]))

    if hint_hls:
        same_hue = [c for c in cands if _hue_dist(_hls(c[0])[0], hint_hls[0]) <= _HINT_HUE_TOL
                    and _hls(c[0])[2] >= 0.25]
        pool = same_hue or vivid or cands
        return max(pool, key=lambda c: (_hls(c[0])[2] * 0.6 + c[1] * 0.4))

    pool = vivid or cands
    return max(pool, key=lambda c: (_hls(c[0])[2], c[1]))


def _pick_outline(clusters, total, fill_rgb):
    """描边色 = 亮度最低的高频簇, 且需与填充色明显不同 -> (rgb, share)。"""
    cands = [(rgb, cnt / total) for rgb, cnt in clusters if cnt / total >= _MIN_SHARE]
    if not cands:
        return None, 0.0
    fill_l = _hls(fill_rgb)[1] if fill_rgb else 1.0
    darker = [c for c in cands if _hls(c[0])[1] < fill_l - 0.12]
    return min(darker or cands, key=lambda c: _hls(c[0])[1])


def sample_region(png_path, bbox_norm, fill_hint_hex=None, outline_hint_hex=None):
    """裁剪 bbox 区域取色 -> (fill_hex, outline_hex, confidence 0~1)。失败位返回 None。"""
    if not (_PIL_OK and bbox_norm):
        return None, None, 0.0
    pixels, ok = _crop_pixels(png_path, bbox_norm)
    if not ok:
        return None, None, 0.0
    total = int(pixels.shape[0])
    clusters = _clusters(pixels)
    if not clusters:
        return None, None, 0.0
    hint_rgb = hex_to_rgb(fill_hint_hex)
    fill_rgb, fill_share = _pick_fill(clusters, total, hint_rgb)
    out_rgb, _ = _pick_outline(clusters, total, fill_rgb)

    fill_hex = rgb_to_hex(fill_rgb) if fill_rgb else None
    outline_hex = rgb_to_hex(out_rgb) if out_rgb else None

    conf = min(1.0, fill_share * 2.2)
    if hint_rgb and fill_rgb:
        hd = _hue_dist(_hls(hint_rgb)[0], _hls(fill_rgb)[0])
        hint_hls, fill_hls = _hls(hint_rgb), _hls(fill_rgb)
        # 目测偏白但校准取到彩色属于"修正", 不扣分; 二者都有彩且色相偏差大才扣分
        if hint_hls[2] > _WHITE_S_MAX and fill_hls[2] > 0.25 and hd > _HINT_HUE_TOL:
            conf *= 0.5
    return fill_hex, outline_hex, round(conf, 3)


def _nearest_frames(t_start, t_end, hires_frames, k=3):
    """区间内(含端点)最近的 k 个原生帧; 区间无帧则取全局最近。"""
    inside = [(t, p) for t, p in hires_frames if t_start - 0.01 <= t <= t_end + 0.01]
    if inside:
        return inside[:k]
    if not hires_frames:
        return []
    mid = (t_start + t_end) / 2.0
    return sorted(hires_frames, key=lambda tp: abs(tp[0] - mid))[:k]


def _median_hex(hexes):
    """一组 #RRGGBB 的逐通道中位 -> #RRGGBB; 空返回 None。"""
    rgbs = [hex_to_rgb(h) for h in hexes if h]
    rgbs = [r for r in rgbs if r]
    if not rgbs:
        return None
    return rgb_to_hex(tuple(np.median(np.asarray(rgbs), axis=0)))


def calibrate(inventory, hires_frames):
    """对 inventory 每条字幕做像素校准, 就地写回校准字段并返回 inventory。"""
    caps = inventory.get("captions") or []
    if not _PIL_OK or not hires_frames:
        for c in caps:
            c.setdefault("fill_hex_calib", None)
            c.setdefault("outline_hex_calib", None)
            c.setdefault("calib_confidence", 0.0)
            c.setdefault("calib_trust", False)
        return inventory
    for c in caps:
        bbox = c.get("bbox")
        frames = _nearest_frames(c["t_start"], c["t_end"], hires_frames)
        fills, outs, confs = [], [], []
        for _t, path in frames:
            fh, oh, conf = sample_region(path, bbox, c.get("fill_hex"), c.get("outline_hex"))
            if fh:
                fills.append(fh)
                confs.append(conf)
            if oh:
                outs.append(oh)
        c["fill_hex_calib"] = _median_hex(fills)
        c["outline_hex_calib"] = _median_hex(outs)
        c["calib_confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
        c["calib_trust"] = _trust(c["fill_hex_calib"], c["calib_confidence"])
    return inventory


# 采信校准的门槛: VLM 目测对【白/浅色】文字很可靠, 像素校准的价值在于抓【彩色】文字
# (红橙/金黄)。故只在校准结果明显彩色且置信足够时才覆盖 hint。
_TRUST_S = 0.45
_TRUST_CONF = 0.40


def _saturation(hex_str):
    rgb = hex_to_rgb(hex_str)
    return _hls(rgb)[2] if rgb else 0.0


def _trust(calib_hex, conf):
    if not calib_hex:
        return False
    return _saturation(calib_hex) >= _TRUST_S and float(conf or 0.0) >= _TRUST_CONF


def best_fill_hex(cap):
    """对外统一取色策略: 采信校准则用校准色, 否则用 VLM 目测 hint。"""
    if cap.get("calib_trust") and cap.get("fill_hex_calib"):
        return cap["fill_hex_calib"], "calib"
    return (cap.get("fill_hex") or "#FFFFFF"), "hint"


def best_outline_hex(cap):
    if cap.get("calib_trust") and cap.get("outline_hex_calib"):
        return cap["outline_hex_calib"], "calib"
    return (cap.get("outline_hex") or "#101010"), "hint"
