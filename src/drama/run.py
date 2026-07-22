"""长 AI 带货短剧复刻 —— 命令行入口（用于本地跑通/验证）。

用法：
    python -m drama.run <参考视频> [--iters N] [--out DIR]
默认商品=理然去黑头泥膜棒，商品图取 Res/ 下两张截图。
"""
import argparse
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_SRC, "shared"), _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from drama.pipeline import run_drama_replication, DEFAULT_PRODUCT

_ROOT = os.path.dirname(_SRC)
_RES = os.path.join(_ROOT, "Res")
DEFAULT_PRODUCT_IMAGES = [
    os.path.join(_RES, "Screenshot 2026-07-20 at 00.47.14.png"),
    os.path.join(_RES, "Screenshot 2026-07-20 at 00.47.33.png"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--iters", type=int, default=4,
                    help="最多迭代轮数（每轮 verify 通过即提前结束，实现“未达标就一直改”）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--core", default=None, help="已算好的理解结果 JSON，跳过①")
    ap.add_argument("--product-images", nargs="*", default=None)
    ap.add_argument("--product-name", default=None)
    ap.add_argument("--product-features", nargs="*", default=None)
    ap.add_argument("--product-notes", default=None)
    args = ap.parse_args()

    core = None
    if args.core and os.path.isfile(args.core):
        with open(args.core, encoding="utf-8") as f:
            core = json.load(f)

    pimgs = args.product_images if args.product_images is not None else DEFAULT_PRODUCT_IMAGES
    pimgs = [p for p in pimgs if os.path.isfile(p)]

    product = dict(DEFAULT_PRODUCT)
    if args.product_name:
        product["name"] = args.product_name
    if args.product_features:
        product["features"] = args.product_features
    if args.product_notes:
        product["notes"] = args.product_notes

    for ev in run_drama_replication(args.video, product=product,
                                    product_images=pimgs, outdir=args.out,
                                    max_iters=args.iters, core=core):
        print(json.dumps(ev, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
