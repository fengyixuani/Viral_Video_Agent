"""阶段③：角色三视图 + 每片段的故事板整图 + 每片段的 seedance 首帧。

- 角色三视图：人物一致性锚点。
- 故事板整图：**一个片段一张图**（多格漫画式，格子里是设计好的分镜、格下方是镜头类型/说明/台词），
  参考「DUSTLINE RESCUE STORYBOARD」的排版：3×3 或 2×3 网格 + 编号 + shot type + caption。
  用一次文生图直接产出整张（不是逐格）。
- 片段首帧：**一个片段一张**，用 i2i（参考角色三视图 + 商品图）生成 seedance i2v 的首帧关键画面。
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tools import wenchain_media as media

STYLE = os.getenv(
    "DRAMA_STYLE",
    "3D动画电影风格（国漫/皮克斯质感），卡通渲染，明确非真人、非写实，夸张卡通造型，色彩鲜明",
)


# ---------------- 角色三视图 ----------------

def gen_character_sheets(characters, outdir: str, emit=None,
                          concurrency: int = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    n = concurrency or int(os.getenv("DRAMA_SHEET_CONCURRENCY", "4"))

    def _one(ch):
        cid = ch.get("id") or "c?"
        prompt = (
            "角色设定三视图，同一个人物的正面/侧面/背面全身像并排，纯白背景，无文字，"
            f"{STYLE}，光线均匀。人物特征：{ch.get('appearance','')}；服装：{ch.get('wardrobe','')}；"
            f"气质：{ch.get('vibe','')}。三个视图外观、发型、服装、配色完全一致。"
        )
        if emit:
            emit(f"生成三视图：{ch.get('name', cid)}")
        url = media.gen_image(prompt, size="2048x2048")
        local = os.path.join(outdir, f"sheet_{cid}.jpg")
        try:
            media.download(url, local)
        except Exception:
            local = ""
        return cid, {"url": url, "local": local, "name": ch.get("name", cid)}

    sheets = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        for fut in as_completed({pool.submit(_one, c): c for c in (characters or [])}):
            cid, data = fut.result()
            sheets[cid] = data
    return sheets


# ---------------- 故事板整图（每片段一张，多格漫画） ----------------

_PRODUCT_HINTS = ("理然", "make sense", "泥膜棒", "MAKE SENSE", "泥膜")


def _seg_has_product(seg):
    """段级或任一 panel 涉及商品都算——避免脚本只在 panel 级提到商品导致丢参考图。"""
    if seg.get("product_in_frame"):
        return True
    for p in seg.get("panels", []) or []:
        if p.get("product_in_frame"):
            return True
        text = f"{p.get('visual','')} {p.get('label','')} {p.get('action','')}"
        low = text.lower()
        if any(h in text or h.lower() in low for h in _PRODUCT_HINTS):
            return True
    return False


def gen_product_asset(product_images, product_desc: str, outdir: str,
                      emit=None) -> dict:
    """把真人拍摄的商品图**预先卡通化**成一张"3D 动画风格的商品资产图"，
    后续故事板/首帧都用这张卡通商品图当参考——避免 seedream i2i 直接把
    写实商品照贴进画面（就是你看到的"06 直接放了原图"的根因）。"""
    os.makedirs(outdir, exist_ok=True)
    product_images = [p for p in (product_images or []) if p and os.path.isfile(p)]
    prompt = (
        f"{STYLE}。把参考图里的商品重新绘制成 3D 动画/国漫风格的商品资产图（product asset sheet），"
        "纯白背景，无文字，正面主视图 + 45° 侧视图 + 顶部旋出灰紫色膏体的特写，三视图并排。"
        f"必须严格保留商品的外观、形态、配色和 logo：{product_desc}（棒状膏体，非罐装/瓶装/泵头）。"
        "画面必须是卡通渲染，绝对不要沿用参考图的写实照片质感。"
    )
    if emit:
        emit("卡通化商品资产图（一次性预生成）")
    url = media.gen_image(prompt, size="2048x2048", ref_images=product_images or None)
    local = os.path.join(outdir, "product_asset.jpg")
    try:
        media.download(url, local)
    except Exception:
        local = ""
    return {"url": url, "local": local}


def _seg_refs(seg, sheets, product_ref_urls):
    """product_ref_urls: 卡通化后的商品资产图 URL 列表（不是原始写实照片）。"""
    refs = []
    for cid in seg.get("characters", []) or []:
        s = sheets.get(cid)
        if s and s.get("url"):
            refs.append(s["url"])
    if _seg_has_product(seg) and product_ref_urls:
        refs.extend(product_ref_urls)
    return refs


def gen_story_board(seg, sheets: dict, product_refs, outdir: str,
                    product_desc: str = "", emit=None) -> dict:
    """为一个片段生成一张多格故事板整图。product_refs 传**卡通化后**的商品资产 URL 列表。"""
    os.makedirs(outdir, exist_ok=True)
    idx = seg.get("idx", 0)
    panels = seg.get("panels", []) or []
    n = len(panels)
    grid = "3x3" if n >= 7 else ("2x3" if n >= 5 else "2x2")
    lines = []
    for p in panels:
        cap = (p.get("visual", "") or "")[:36]
        line = (f"{p.get('idx',0):02d} · {p.get('shot_type','')} · {p.get('label','')} — {cap}")
        if p.get("dialogue"):
            line += f"　台词：{p.get('dialogue')}"
        lines.append(line)
    refs = _seg_refs(seg, sheets, product_refs or [])
    prompt = (
        f"{STYLE}。一张专业影视分镜故事板（storyboard sheet）整图，"
        f"顶部有片名/场景/PAGE 标题栏；主体是 {grid} 网格，共 {n} 格，"
        "每格里是一个电影镜头画面，格子左上角有编号 01/02/…，右上角标 shot type（WIDE/MEDIUM/CLOSE_UP…），"
        "格子正下方有一条文字说明栏写该镜头的 label 与画面描述，若有台词也写在下方。"
        "整体白底黑框，专业分镜脚本风格。分镜内容如下：\n" + "\n".join(lines) +
        "\n保持同一人物跨格外观一致；商品外观/配色/logo 一致。"
        "\n重要：每一格必须是 3D 动画/国漫风格的绘制画面，绝对不要复制粘贴参考图（尤其是任何写实照片），"
        "参考图只作为角色/商品的外观参照，实际画面必须重绘为卡通渲染。"
    )
    if product_desc and _seg_has_product(seg):
        prompt += f" 商品严格还原为：{product_desc}（棒状膏体，非罐装/瓶装/泵头）。"
    if emit:
        emit(f"生成片段 {idx} 故事板（{n} 格 {grid}）")
    url = media.gen_image(prompt, size="2048x2048", ref_images=refs or None)
    local = os.path.join(outdir, f"board_seg{idx:02d}.jpg")
    try:
        media.download(url, local)
    except Exception:
        local = ""
    return {"idx": idx, "url": url, "local": local, "panels": n}


def gen_story_boards(script, sheets: dict, product_images, outdir: str, emit=None,
                     product_desc: str = "", concurrency: int = None) -> list:
    n = concurrency or int(os.getenv("DRAMA_BOARD_CONCURRENCY", "4"))
    segs = list(script.get("segments", []) or [])
    boards = [None] * len(segs)
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(gen_story_board, s, sheets, product_images, outdir,
                             product_desc, emit): i for i, s in enumerate(segs)}
        for fut in as_completed(futs):
            i = futs[fut]
            boards[i] = fut.result()
    return boards


# ---------------- 每片段首帧（seedance i2v 的首帧） ----------------

def gen_segment_first_frame(seg, sheets: dict, product_refs, outdir: str,
                             extra: str = "", product_desc: str = "") -> dict:
    os.makedirs(outdir, exist_ok=True)
    idx = seg.get("idx", 0)
    refs = _seg_refs(seg, sheets, product_refs or [])
    # 该段出场角色（用于强制首帧里出现人物——否则商品段的 opening_frame_desc 常是纯商品空镜，
    # 导致 i2v 无人物锚点、seedance 中途凭空新画角色 → 跨段人物不一致）。
    char_names = [(sheets.get(cid) or {}).get("name") or cid
                  for cid in (seg.get("characters", []) or []) if sheets.get(cid)]
    has_product = _seg_has_product(seg)
    prompt = (
        f"{STYLE}。{seg.get('opening_frame_desc','')} 场景：{seg.get('scene','')}。"
        "竖屏 9:16 电商短剧首帧，画面干净，无字幕无水印。"
        "重要：必须是 3D 动画/国漫风格的绘制画面，绝对不要沿用参考图的写实照片质感，"
        "参考图只是外观参照，实际画面必须重绘为卡通渲染。"
    )
    if char_names:
        # 强约束：本段出场角色必须清晰出现在首帧里，且与参考三视图一致
        prompt += (f" 本片段的出场角色（{('、'.join(char_names))}）**必须清晰出现在首帧画面中**，"
                   "为可辨认的半身/全身人物，外观/发型/服装/配色严格与对应参考三视图一致；"
                   "即使本段侧重商品，也**不要画成没有人物的纯商品空镜**——商品应作为角色手中/桌上的道具与人物同框。")
    if refs:
        prompt += " 严格保持参考图中人物的外观/发型/服装/配色一致；若含商品，保持商品外观、配色、logo 与参考图一致。"
    if has_product and product_desc:
        prompt += f" 商品必须严格还原为：{product_desc}（棒状膏体，非罐装/瓶装/泵头），清晰可见。"
    if extra:
        prompt += " " + extra
    url = media.gen_image(prompt, size="1440x2560", ref_images=refs or None)
    local = os.path.join(outdir, f"ff_seg{idx:02d}.jpg")
    try:
        media.download(url, local)
    except Exception:
        local = ""
    return {"idx": idx, "url": url, "local": local, "prompt": prompt}


def gen_segment_first_frames(script, sheets: dict, product_refs, outdir: str,
                              emit=None, product_desc: str = "",
                              concurrency: int = None) -> dict:
    n = concurrency or int(os.getenv("DRAMA_FF_CONCURRENCY", "4"))
    segs = list(script.get("segments", []) or [])
    frames = {}
    if emit:
        emit(f"并发生成 {len(segs)} 张片段首帧")

    def _one(seg):
        return seg.get("idx"), gen_segment_first_frame(
            seg, sheets, product_refs, outdir, product_desc=product_desc)

    with ThreadPoolExecutor(max_workers=n) as pool:
        for fut in as_completed({pool.submit(_one, s): s for s in segs}):
            idx, data = fut.result()
            frames[idx] = data
            if emit:
                emit(f"片段 {idx} 首帧完成")
    return frames
