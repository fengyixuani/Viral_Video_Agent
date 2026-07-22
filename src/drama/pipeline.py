"""长 AI 带货短剧复刻编排器（串起 ①理解 → ②脚本 → ③三视图/分镜 → ④生成/拼接 → ⑤验证/迭代）。

以同步生成器 yield 事件，供 server SSE 或 CLI 消费。事件格式：
    {"type":"step"|"reasoning"|"data"|"error"|"final", ...}
"""
import json
import os
import sys
import time

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_SRC, "shared"), _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runs
from drama import understand as _understand
from drama import script as _script
from drama import storyboard as _storyboard
from drama import video as _video
from drama import verify as _verify
from drama import router as _router

DEFAULT_PRODUCT = {
    "name": "理然 MAKE SENSE 去黑头泥膜棒",
    "features": ["瓦晶矿物泥+三重植物精油", "有效清洁毛孔/去黑头", "棒状膏体直接涂抹，便携"],
    "notes": "绿色膏体棒状，黑色底座，瓶身有 理然 / MAKE SENSE logo，膏体为灰紫色",
}


def _ev(type_, **kw):
    d = {"type": type_}
    d.update(kw)
    return d


def run_drama_replication(video_path, product=None, product_images=None,
                          outdir=None, max_iters=2, core=None):
    """主流程生成器。逐阶段 yield 事件，最后 yield 一个 final 事件。

    core: 可传入已算好的理解结果，跳过①（省时省额度）。
    """
    product = product or DEFAULT_PRODUCT
    product_images = product_images or []
    # 每次短剧复刻 = 一个可读命名的 run 目录（outputs/{时间}_drama_{商品}_{rid}/），并登记索引
    _run = None
    if outdir:
        outdir = os.path.abspath(outdir)
    else:
        _run = runs.new_run("drama", (product or {}).get("name", ""))
        outdir = _run["dir"]
    os.makedirs(outdir, exist_ok=True)

    steps = []

    def log_step(stage, msg):
        steps.append({"stage": stage, "msg": msg})

    try:
        # ---- ⓪ 商品信息推断（商品名/卖点/外观任一缺失时用视觉模型补全）----
        _need_infer = not (product.get("name") or "").strip() or not product.get("features") \
            or not (product.get("notes") or "").strip()
        if _need_infer:
            yield _ev("step", stage="product_infer", thought="商品信息不全，AI 从商品图/视频自动补全名称/卖点/外观…")
            product = _router.infer_product(product, product_images, video_path)
            yield _ev("data", stage="product_infer",
                      name=product.get("name"),
                      features=product.get("features", []),
                      inferred=bool(product.get("_inferred")))

        # ---- ① 理解 ----
        if core is None:
            yield _ev("step", stage="understand", thought="观看参考短剧，提取核心成分…")
            core = _understand.understand_reference(video_path, product.get("name", ""))
        else:
            yield _ev("step", stage="understand", thought="复用已有理解结果，跳过重新观看。")
        with open(os.path.join(outdir, "01_core.json"), "w", encoding="utf-8") as f:
            json.dump(core, f, ensure_ascii=False, indent=2)
        yield _ev("data", stage="understand",
                  summary={"logline": core.get("logline"),
                           "viral_formula": core.get("viral_formula"),
                           "shots": len(core.get("shots", [])),
                           "characters": len(core.get("characters", []))})

        best = None
        feedback = None
        for it in range(1, max_iters + 1):
            iterdir = os.path.join(outdir, f"iter{it}")
            os.makedirs(iterdir, exist_ok=True)
            yield _ev("step", stage="iterate", thought=f"第 {it}/{max_iters} 轮生成")

            # ---- ② 脚本 ----
            yield _ev("step", stage="script", thought="写新剧本 + 人物设定…")
            prod_for_script = dict(product)
            if feedback:
                prod_for_script["notes"] = (product.get("notes", "") +
                                            f"\n[上一轮质检修改建议]{feedback}")
            scr = _script.write_script(core, prod_for_script)
            with open(os.path.join(iterdir, "02_script.json"), "w", encoding="utf-8") as f:
                json.dump(scr, f, ensure_ascii=False, indent=2)
            segs = scr.get("segments", [])
            yield _ev("data", stage="script",
                      summary={"title": scr.get("title"), "segments": len(segs),
                               "characters": len(scr.get("characters", [])),
                               "differentiation": scr.get("differentiation")})

            # ---- ③a 角色三视图 ----
            yield _ev("step", stage="character_sheets", thought="生成角色三视图（人物一致性锚点）…")
            sheets = _storyboard.gen_character_sheets(
                scr.get("characters", []), os.path.join(iterdir, "sheets"),
                emit=lambda m: log_step("character_sheets", m))
            yield _ev("data", stage="character_sheets",
                      sheets={k: v.get("url") for k, v in sheets.items()})

            # ---- ③b 故事板整图（每片段一张，多格漫画式设计蓝图）----
            yield _ev("step", stage="storyboard", thought="为每个片段生成一张多格故事板整图…")
            product_desc = f"{product.get('name','')}，{product.get('notes','')}"

            # ③a-2：**先把真实商品照卡通化**成动画风格的"商品资产图"（一次性）——
            # 之后所有 board/首帧都用这张卡通商品图作参考，避免 seedream 把写实照片直接贴进画面。
            yield _ev("step", stage="product_asset", thought="卡通化商品资产图（避免写实照片贴回）…")
            asset = _storyboard.gen_product_asset(
                product_images, product_desc, os.path.join(iterdir, "product_asset"),
                emit=lambda m: log_step("product_asset", m))
            product_refs = [asset.get("url")] if asset.get("url") else []
            yield _ev("data", stage="product_asset", url=asset.get("url"))

            boards = _storyboard.gen_story_boards(
                scr, sheets, product_refs, os.path.join(iterdir, "boards"),
                emit=lambda m: log_step("storyboard", m), product_desc=product_desc)
            yield _ev("data", stage="storyboard",
                      boards=[{"idx": b["idx"], "url": b["url"], "panels": b.get("panels")}
                              for b in boards if b])

            # ---- ③c 片段首帧（seedance i2v 首帧）----
            yield _ev("step", stage="first_frames", thought="为每个片段生成 seedance i2v 首帧…")
            first_frames = _storyboard.gen_segment_first_frames(
                scr, sheets, product_refs, os.path.join(iterdir, "first_frames"),
                emit=lambda m: log_step("first_frames", m), product_desc=product_desc)
            yield _ev("data", stage="first_frames", count=len(first_frames))

            # ---- ④ 逐片段生成视频（一次 i2v 生 15s，台词进 prompt 发声）+ 拼接 ----
            yield _ev("step", stage="video",
                      thought="seedance 逐片段图生视频（每片段一次 15s i2v，含台词）…")

            def _regen_ff(seg, _sheets=sheets, _iterdir=iterdir, _pdesc=product_desc,
                          _prefs=product_refs):
                kf = _storyboard.gen_segment_first_frame(
                    seg, _sheets, _prefs,
                    os.path.join(_iterdir, "first_frames_regen"),
                    extra="强烈卡通化、明确非真人、扁平动画质感，避免任何写实人像。",
                    product_desc=_pdesc)
                return kf.get("url")

            clips = _video.gen_segment_clips(
                scr, first_frames, os.path.join(iterdir, "clips"),
                emit=lambda m: log_step("video", m),
                regen_first_frame=_regen_ff)
            yield _ev("data", stage="video",
                      clips=[{"idx": c["idx"], "mode": c.get("mode")} for c in clips])

            final_path = os.path.join(iterdir, "final.mp4")
            yield _ev("step", stage="concat", thought="拼接成片…")
            _video.concat_final(clips, final_path)
            yield _ev("data", stage="concat", video=final_path)

            # ---- ⑤ 验证 ----
            yield _ev("step", stage="verify", thought="qwen3.7-plus 观看成片打分…")
            report = _verify.verify_final(final_path, core, scr)
            # 由代码按分数硬判定是否达标（overall>=7 且无单项<5），不盲信 LLM 的 pass 标志
            scores = report.get("scores", {}) or {}
            overall = report.get("overall", 0)
            min_score = min(scores.values()) if scores else 0
            passed = bool(overall >= 7 and min_score >= 5)
            report["pass"] = passed
            report["_gate"] = {"overall": overall, "min_score": min_score,
                               "rule": "overall>=7 and min_score>=5"}
            with open(os.path.join(iterdir, "05_verify.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            yield _ev("data", stage="verify", overall=overall, passed=passed,
                      scores=scores, problems=report.get("problems", []))

            cand = {"iter": it, "video": final_path, "script": scr,
                    "report": report, "overall": overall}
            if best is None or overall > best["overall"]:
                best = cand
            if passed:
                if _run:
                    runs.record(_run, project=(product or {}).get("name", ""),
                                status="pass", iter=it, overall=overall, video=final_path)
                yield _ev("final", status="pass", iter=it, video=final_path,
                          overall=overall, outdir=outdir)
                return
            feedback = json.dumps(report.get("fix_suggestions", []), ensure_ascii=False)
            yield _ev("step", stage="iterate",
                      thought=f"第 {it} 轮未达标(overall={overall})，据建议迭代…")

        if _run and best:
            runs.record(_run, project=(product or {}).get("name", ""),
                        status="best_effort", iter=best["iter"],
                        overall=best["overall"], video=best["video"])
        yield _ev("final", status="best_effort",
                  iter=best["iter"], video=best["video"],
                  overall=best["overall"], outdir=outdir)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        yield _ev("error", message=str(exc))


def yield_hint(msg):
    # storyboard emit 只记录日志（生成器内不便直接 yield），此处占位。
    pass
