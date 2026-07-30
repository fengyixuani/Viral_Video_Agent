"""seedance_fill — 对「缺口镜头」(没有贴切用户素材)可选调用 T2V 补拍。

以用户素材为主的做法:
  - 只对 is_gap=True 的镜头补拍, 其余镜头一律用用户素材。
  - 补拍时默认抽取该镜头「次优用户候选」的一帧作 reference_image, 让生成画面
    在产品/风格上贴近用户真实素材(而非凭空生成)。
  - T2V 不可达/失败时**优雅降级**: 保留原用户素材匹配, 绝不让流程硬失败。

复用 source 的 qianfan_t2v_client.generate_video(prompt, out, neg, ref_image)。
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import FFMPEG, VENDOR  # noqa: F401

try:
    # VENDOR 已由 _common 挂在 sys.path (排在 COMMON 之后)
    from qianfan_t2v_client import generate_video
    _HAS_T2V = True
except Exception as _exc:  # pragma: no cover
    _HAS_T2V = False
    _T2V_IMPORT_ERR = str(_exc)


_NEG = os.getenv("T2V_NEGATIVE_PROMPT",
                 "low quality, blurry, distorted hands, text, watermark, logo, extra fingers, deformed food")


def _grab_reference_frame(cand, out_png):
    """从候选片段中点抽一帧, 作 T2V reference_image。"""
    src = cand.get("source_path")
    if not src or not os.path.exists(src):
        return None
    mid = cand.get("start", 0.0) + max(0.1, cand.get("duration", 0.0) / 2.0)
    cmd = [FFMPEG, "-y", "-ss", "{:.3f}".format(mid), "-i", src,
           "-frames:v", "1", "-q:v", "3", out_png]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_png if os.path.exists(out_png) else None
    except Exception:
        return None


def build_t2v_prompt(shot, product_name=""):
    desc = shot.get("beat_desc") or shot.get("description") or ""
    parts = []
    if product_name:
        parts.append(product_name)
    parts.append(desc)
    parts.append("竖屏 9:16, 高清美食特写, 暖色调, 无文字水印")
    return ", ".join(p for p in parts if p)


def fill_gaps(segments, out_dir, product_name="", use_reference_image=True,
              max_fill=None):
    os.makedirs(out_dir, exist_ok=True)
    if not _HAS_T2V:
        print("[seedance] T2V 客户端不可用, 全部保留用户素材: {}".format(
            globals().get("_T2V_IMPORT_ERR", "")), flush=True)
        return segments, {"filled": 0, "reason": "t2v_unavailable"}
    filled = 0
    for seg in segments:
        if not seg.get("is_gap"):
            continue
        if max_fill is not None and filled >= max_fill:
            break
        idx = seg.get("index")
        prompt = build_t2v_prompt(seg, product_name)
        out_mp4 = os.path.join(out_dir, "t2v_shot_{:02d}.mp4".format(idx))
        ref_img = None
        if use_reference_image and seg.get("best_candidate"):
            ref_img = _grab_reference_frame(
                seg["best_candidate"], os.path.join(out_dir, "ref_{:02d}.png".format(idx)))
        try:
            print("[seedance] 补拍 段{} prompt={}".format(idx, prompt[:80]), flush=True)
            generate_video(prompt, out_mp4, _NEG, ref_img or "")
            if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
                seg["is_t2v"] = True
                seg["t2v_path"] = out_mp4
                seg["t2v_prompt"] = prompt
                seg["t2v_reference_image"] = ref_img
                seg["is_gap"] = False
                filled += 1
            else:
                print("[seedance] 段{} 输出为空, 保留用户素材".format(idx), flush=True)
        except Exception as exc:
            print("[seedance] 段{} 补拍失败, 保留用户素材: {}".format(
                idx, str(exc)[:160]), flush=True)
    return segments, {"filled": filled}


def main(argv=None):
    ap = argparse.ArgumentParser(description="对缺口段落 T2V 补拍(用户素材为主)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--product-name", default="")
    ap.add_argument("--no-reference-image", action="store_true")
    ap.add_argument("--max-fill", type=int)
    ap.add_argument("--out", help="写回带 t2v 标记的 plan json")
    args = ap.parse_args(argv)
    data = json.load(open(args.plan, encoding="utf-8"))
    wrapper = data if isinstance(data, dict) and "segments" in data else {"segments": data}
    segments, stat = fill_gaps(
        wrapper["segments"], args.out_dir, product_name=args.product_name,
        use_reference_image=not args.no_reference_image, max_fill=args.max_fill)
    print("[seedance] filled:", stat)
    out = args.out or args.plan
    wrapper["segments"] = segments
    json.dump(wrapper, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
