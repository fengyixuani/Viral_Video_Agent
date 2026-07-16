"""纯 Agent 剪辑链路：剪辑 Agent + 审片 Agent + 重剪循环（agent_cut 分支）。

流程（最多 MAX_LOOPS 轮）：
  1. 剪辑 Agent（LLM）：据爆款 DNA + 每镜候选片段 + 上一轮审片反馈，产出/修订 edit_plan
     （每镜：选哪个候选、trim 区间、时长、字幕、变速）。
  2. 剪辑器（editor.build_video，纯 ffmpeg）：执行 edit_plan → 样片 + 逐操作记录 ops。
  3. 审片 Agent（LLM + ali-qwen3.7-plus 视觉）：直接看样片 + 对照 DNA + 看本轮 ops 记录
     + 历轮简要（执行/思考/问题），判定是否符合预期；不符合就给出重剪命令。
  4. 通过则输出；否则带反馈回到 1 重剪。达到轮次上限仍不达标 → 输出历轮最佳样片并标注
     「素材受限」。
"""
from __future__ import annotations

import copy
import json
import os
import time

import as_core
import obs
from agent_edit import editor

_log = obs.get_logger("agent_edit_loop")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BGM = os.path.join(os.path.dirname(AGENT_ROOT), "Viral_Video", "带货素材", "dataset", "可商用bgm",
                           "时尚动感放克律动 Funk Caravan Main_爱给网_aigei_com.mp3")
MAX_LOOPS = int(os.getenv("AGENT_EDIT_MAX_LOOPS", "5"))


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(AGENT_ROOT, path))


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _load_inputs(strategy: dict) -> dict:
    """从 strategy 抽出：DNA 概要 + 每镜（want/角色/目标时长/字幕 + 候选片段 C1..）。"""
    bank_by_slot = {}
    for asset in strategy.get("user_asset_bank", []) or []:
        plan = asset.get("concrete_editing_plan") or {}
        sid = plan.get("slot_id") or asset.get("slot_id")
        if sid:
            bank_by_slot[sid] = asset

    shots = []
    for item in strategy.get("editing_timeline", []) or []:
        if item.get("action") != "use_user_asset":
            continue
        sid = item.get("slot_id", "")
        asset = bank_by_slot.get(sid) or {}
        t0, t1 = _parse_range(item.get("target_time_range", ""))
        target_dur = round(t1 - t0, 2) if t1 > t0 else 3.0
        # 候选：首选（editing_timeline 的 source）+ 继承 alternates
        cands = []
        primary_path = _abspath(item.get("source_path", ""))
        if primary_path and os.path.isfile(primary_path):
            cands.append({"cid": "C1", "source_path": item.get("source_path", ""),
                          "source_time_range": item.get("source_time_range", ""),
                          "summary": asset.get("visual_description", "") or item.get("caption_text", "")})
        for alt in (asset.get("alternate_assets", []) or []):
            ap = _abspath(alt.get("source_path", ""))
            if ap and os.path.isfile(ap):
                cands.append({"cid": f"C{len(cands) + 1}", "source_path": alt.get("source_path", ""),
                              "source_time_range": alt.get("source_time_range", ""),
                              "summary": alt.get("summary", "")})
        if not cands:
            continue
        shots.append({
            "slot_id": sid, "role": item.get("role", ""),
            "want": asset.get("suitable_roles", "") or item.get("caption_text", ""),
            "target_duration": target_dur,
            "reference_caption": item.get("caption_text", ""),
            "candidates": cands,
        })
    meta = strategy.get("metadata", {}) or {}
    return {
        "product_name": meta.get("scheme", "") or meta.get("project_name", "") or "目标商品",
        "narrative_structure": strategy.get("overall_editing_strategy", {}) if isinstance(strategy.get("overall_editing_strategy"), dict) else {},
        "shots": shots,
    }


_EDIT_PROMPT = (
    "你是短视频剪辑 Agent。目标：用给定的用户素材候选，剪出一条符合爆款 DNA 的竖屏成片。\n"
    "输入：dna（爆款结构/节奏），shots（每个镜头的角色/意图/目标时长/参考字幕 + 候选片段 candidates，"
    "每个候选有 cid/source_time_range/summary），review_feedback（上一轮审片的问题与重剪命令，首轮为空）。\n"
    "规则：\n"
    "- 为每个镜头从它的 candidates 里选一个 cid；可在候选的 source_time_range 内进一步收窄 trim 区间。\n"
    "- caption 用简体中文口播/字幕，简短有力，贴合该镜头意图；可基于参考字幕改写，别照抄参考视频。\n"
    "- target_duration 参考给定值，可按节奏微调（0.8~1.4 倍）；需要时可设 speed（0.8~1.6）加快节奏。\n"
    "- 若有 review_feedback，必须针对性修正（换候选/改 trim/改字幕/调时长/调速度/删镜头）。\n"
    "- 只用候选里存在的 cid，不要编造。\n"
    "严格只输出 JSON：{\"clips\":[{\"slot_id\":\"S01\",\"cid\":\"C1\",\"source_time_range\":\"0.00-3.20\","
    "\"target_duration\":3.0,\"caption\":\"...\",\"speed\":1.0}],\"note\":\"本轮剪辑思路一句话\"}"
)

_REVIEW_PROMPT = (
    "你是短视频审片 Agent。你会看到一条**剪好的成片**（视频）以及它对照的爆款 DNA、本轮剪辑操作记录、"
    "以及历轮简要。你的任务：判断这条成片是否符合爆款 DNA 的预期（结构/节奏/卖点表达/字幕与画面匹配/"
    "整体观感），并指出问题、给出可执行的重剪命令。\n"
    "评估要点：镜头顺序与叙事是否贴合 DNA；单镜时长/节奏是否合适；字幕是否与画面匹配、有无错位；"
    "有无明显重复/突兀/黑屏/空镜；整体是否像一条完整的带货短视频。\n"
    "严格只输出 JSON：{\n"
    '  "pass": true/false,            // 是否已符合预期\n'
    '  "score": 0-100,                // 综合评分\n'
    '  "problems": [{"slot_id":"S0x","issue":"具体问题","fix":"该怎么改"}],\n'
    '  "commands": ["给剪辑 Agent 的重剪指令（具体到镜头/操作）"],\n'
    '  "thought": "一句话总体判断",\n'
    '  "material_limited": true/false // 若判断是"素材本身撑不起 DNA、再剪也只能这样"，置 true\n'
    "}"
)


async def _run_llm(system: str, user: str, *, media=None, vision=False):
    content = ""
    async for item in as_core.stream(system, user, vision=vision, media=media):
        if "content" in item:
            content = item["content"]
    return as_core.parse_json(content) if content.strip() else {}


def _resolve_clips(edit_plan: dict, shots_by_slot: dict) -> list:
    """把剪辑 Agent 的 edit_plan（用 cid 引用候选）解析成 editor 需要的 clips（真实路径）。"""
    clips = []
    for c in edit_plan.get("clips", []) or []:
        sid = c.get("slot_id")
        shot = shots_by_slot.get(sid)
        if not shot:
            continue
        cand = next((x for x in shot["candidates"] if x["cid"] == c.get("cid")), shot["candidates"][0])
        rng = c.get("source_time_range") or cand["source_time_range"]
        clips.append({
            "slot_id": sid, "source_path": cand["source_path"], "source_time_range": rng,
            "target_duration": float(c.get("target_duration") or shot["target_duration"] or 3.0),
            "caption_text": c.get("caption") or shot.get("reference_caption", ""),
            "speed": float(c.get("speed") or 1.0),
        })
    return clips


def _ops_brief(result: dict) -> list:
    out = []
    for o in result.get("ops", []):
        if o.get("op") == "trim":
            out.append(f"{o.get('slot_id')}: {o.get('source')} {o.get('source_time_range')} {o.get('target_duration')}s "
                       f"speed={o.get('speed')} 字幕{'已烧' if o.get('caption_burned') else '未烧'} {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
        else:
            out.append(f"{o.get('op')}: {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
    return out


async def agent_edit_stream(strategy_path: str, *, enable_bgm: bool = True, max_loops: int = None):
    """产出 step / edit_log / agent_edit_done / error 事件。"""
    rid = time.strftime("%H%M%S")
    max_loops = max_loops or MAX_LOOPS

    def step(key, title, thought, state="done", observation=None):
        ev = {"type": "step", "phase": "Agent 剪辑", "key": f"{key}-{rid}", "state": state,
              "title": title, "thought": thought}
        if observation is not None:
            ev["observation"] = observation
        return ev

    strategy_abs = _abspath(strategy_path)
    if not strategy_abs or not os.path.isfile(strategy_abs):
        yield {"type": "error", "message": f"找不到编排脚本：{strategy_path}"}
        return
    with open(strategy_abs, "r", encoding="utf-8") as fh:
        strategy = json.load(fh)
    inputs = _load_inputs(strategy)
    shots = inputs["shots"]
    if not shots:
        yield {"type": "error", "message": "没有可剪辑的镜头（都是补拍/AIGC 生成或候选缺失）。"}
        return
    shots_by_slot = {s["slot_id"]: s for s in shots}
    dna = {"product_name": inputs["product_name"], "narrative_structure": inputs["narrative_structure"],
           "shots": [{"slot_id": s["slot_id"], "role": s["role"], "want": s["want"],
                      "target_duration": s["target_duration"]} for s in shots]}
    bgm = DEFAULT_BGM if enable_bgm and os.path.isfile(DEFAULT_BGM) else ""

    final_dir = os.path.join(AGENT_ROOT, "uploads", "final")
    os.makedirs(final_dir, exist_ok=True)
    slug = "agentcut_" + time.strftime("%Y%m%d_%H%M%S")

    history = []          # 每轮简要（供审片看历轮）
    best = None           # {score, video_uri, path, loop}
    review_feedback = {}

    yield step("prep", "读取编排脚本", f"命中 {len(shots)} 个可剪辑镜头，进入 Agent 剪辑-审片循环（最多 {max_loops} 轮）",
               observation="\n".join(f"{s['slot_id']} {s['role']} 候选{len(s['candidates'])}个 目标{s['target_duration']}s" for s in shots))

    for loop in range(1, max_loops + 1):
        # 1) 剪辑 Agent
        yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 规划", "据 DNA + 候选 + 上一轮反馈决定每镜选片/字幕/时长", state="running")
        edit_user = json.dumps({"dna": dna, "shots": shots, "review_feedback": review_feedback,
                                "history": history[-3:]}, ensure_ascii=False)
        edit_plan = await _run_llm(_EDIT_PROMPT, edit_user)
        clips = _resolve_clips(edit_plan, shots_by_slot)
        if not clips:
            # 兜底：直接用每镜首选
            clips = [{"slot_id": s["slot_id"], "source_path": s["candidates"][0]["source_path"],
                      "source_time_range": s["candidates"][0]["source_time_range"],
                      "target_duration": s["target_duration"],
                      "caption_text": s.get("reference_caption", ""), "speed": 1.0} for s in shots]
        yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 规划",
                   edit_plan.get("note", "") or f"排定 {len(clips)} 个镜头",
                   observation=json.dumps([{"slot": c["slot_id"], "src_range": c["source_time_range"],
                                            "dur": c["target_duration"], "speed": c["speed"],
                                            "cap": c["caption_text"]} for c in clips], ensure_ascii=False))

        # 2) 执行剪辑
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"ffmpeg 逐镜 trim/字幕/拼接{'/BGM' if bgm else ''}", state="running")
        out_path = os.path.join(final_dir, f"{slug}_loop{loop}.mp4")
        result = editor.build_video(clips, out_path, bgm_path=bgm, width=720, height=1080, fps=30)
        ops_brief = _ops_brief(result)
        if result.get("error") or not result.get("output"):
            yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"剪辑失败：{result.get('error','')}", state="done",
                       observation="\n".join(ops_brief))
            history.append({"loop": loop, "edit_note": edit_plan.get("note", ""), "ops": ops_brief,
                            "review": "本轮剪辑失败", "problems": [result.get("error", "")]})
            review_feedback = {"commands": [f"上一轮剪辑失败：{result.get('error','')}，请调整选片/时长后重试"]}
            continue
        video_uri = f"uploads/final/{os.path.basename(out_path)}"
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"成片 {result['clip_count']} 镜已生成",
                   observation="\n".join(ops_brief))
        yield {"type": "agent_edit_sample", "loop": loop, "video_uri": video_uri}

        # 3) 审片 Agent（看视频）
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 看片", "对照 DNA 审阅成片，定位问题", state="running")
        review_user = json.dumps({
            "dna": dna, "this_loop_ops": ops_brief,
            "history": [{"loop": h["loop"], "note": h.get("edit_note", ""), "problems": h.get("problems", [])} for h in history[-4:]],
            "instruction": "请观看视频，对照 DNA 判定是否符合预期。",
        }, ensure_ascii=False)
        review = await _run_llm(_REVIEW_PROMPT, review_user, vision=True,
                                media=[{"type": "video", "url": video_uri}])
        score = float(review.get("score") or 0)
        passed = bool(review.get("pass"))
        problems = review.get("problems", []) or []
        if best is None or score > best["score"]:
            best = {"score": score, "video_uri": video_uri, "path": out_path, "loop": loop}
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 判定",
                   f"{'通过' if passed else '未通过'} · {score:.0f} 分 · {review.get('thought','')}",
                   observation=json.dumps(review, ensure_ascii=False)[:1500])

        history.append({"loop": loop, "edit_note": edit_plan.get("note", ""), "ops": ops_brief,
                        "review": review.get("thought", ""), "score": score,
                        "problems": [f"{p.get('slot_id','')}: {p.get('issue','')}" for p in problems]})

        if passed:
            yield {"type": "agent_edit_done", "video_uri": video_uri, "final_path": out_path,
                   "loops": loop, "score": score, "material_limited": False,
                   "verdict": review.get("thought", "符合预期"), "history": history}
            return
        if review.get("material_limited"):
            yield step("stop", "审片判定素材受限", "审片 Agent 认为再剪也只能这样，直接输出当前最佳样片", state="done")
            break
        review_feedback = {"problems": problems, "commands": review.get("commands", []),
                           "last_score": score}

    # 达到上限 / 素材受限：输出最佳样片
    b = best or {"video_uri": "", "path": "", "score": 0, "loop": max_loops}
    yield {"type": "agent_edit_done", "video_uri": b["video_uri"], "final_path": b["path"],
           "loops": len(history), "score": b["score"], "material_limited": True,
           "verdict": "已达轮次上限/素材受限，输出历轮最佳样片", "history": history}
