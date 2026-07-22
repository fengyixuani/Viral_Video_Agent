"""AIGC 补镜 Agent（agent_cut 链路）：为缺失镜头用 seedream + seedance 生成片段。

设计对齐 agent_edit 的范式：用 skill（``aigc_generator``）约束 Agent、把功能做成 tool
（AgentScope 原生 FunctionTool 声明 schema），ReAct 循环里手动分发执行。可按镜头数并行
起多个子 Agent（每镜一个），并发上限 5。

单镜工作流（见 SKILL.md）：看爆款分镜 → 决策是否要产品参考图 → 需要则从用户素材池召回、
取产品参考帧 → 据爆款分镜写 prompt → seedream 出首帧 →（脚本图指引运动）→ seedance 出视频 →
回看成片、按 slot 要求截取一段作为最终片段。

结束时每镜产出一个可直接进 editor 的 clip：
``{"slot_id","source_path"(本地 mp4),"source_time_range","target_duration","caption","aigc":True}``。
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import as_core
import obs
from skills import get as get_skill
from agentscope.tool import FunctionTool
from editing import tools as edit_tools
from tools.retriever import Retriever
from tools.vlm import VLMTool
from tools import aigc_gen

_log = obs.get_logger("aigc_agent")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AIGC_MAX_SUBAGENTS = int(os.getenv("AIGC_MAX_SUBAGENTS", "5"))
AIGC_MAX_STEPS = int(os.getenv("AIGC_MAX_STEPS", "16"))


# --------------------------------------------------------------------------- #
# 工具声明（schema 由 docstring + type hint 自动生成；执行在 _run_aigc_slot 里分发）
# --------------------------------------------------------------------------- #
def _tool_recall_product(query: str, top_k: int = 6):
    """从用户素材池按语义召回最能代表该商品的片段（仅当这一镜需要展示具体商品、要保证商品外观与真实一致时才用）。返回候选 global_asset_id / 画面描述 / 时间区间。

    Args:
        query: 检索词，围绕"这一镜要展示的商品/主体"描述
        top_k: 返回候选数，默认 6
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_pick_product_frame(global_asset_id: str, timestamp: float = -1.0):
    """从某召回片段里抽取一帧作为**产品参考帧**（后续首帧走图生图时保留真实商品外观）。选商品清晰、正面、占主体的时间点。

    Args:
        global_asset_id: recall_product 召回的候选片段 id
        timestamp: 抽帧时间点（秒，相对源视频）；<0 时自动取该片段中点
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_gen_first_frame(prompt: str, use_product_ref: bool = False):
    """用 seedream 生成这一镜的**首帧**（竖屏 9:16）。use_product_ref=true 且已取到产品参考帧时走图生图（保留真实商品外观），否则纯文生图。

    Args:
        prompt: 首帧画面描述（结合爆款分镜的景别/主体/展示重点/光影/风格，中文、具体）
        use_product_ref: 是否用已取的产品参考帧做图生图
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_gen_storyboard(motion_prompt: str):
    """基于已生成的首帧再画一张**脚本图/分镜指引图**，用一句话描述这镜的运动/动作/镜头调度，指引视频怎么动。可选步骤。

    Args:
        motion_prompt: 运动/动作描述，如"镜头缓推，手拿起商品转向镜头，轻微景深变化"
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_gen_video(prompt: str, duration_sec: float = 5.0):
    """用 seedance 由首帧（或脚本图）生成视频（图生视频）。较慢，别重复生成。

    Args:
        prompt: 视频动作/节奏描述（承接首帧，写清楚这镜怎么动）
        duration_sec: 目标时长（秒）；会夹到 4–15s，可略长于 slot 目标，后面再截
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_review_and_extract(start: float, dur: float, caption: str = ""):
    """回看已生成的视频并从中截取最贴合、最稳定的一段作为最终片段（放进成片）。截取前建议先让视觉模型看一遍。

    Args:
        start: 截取起点（秒，相对生成视频）
        dur: 截取时长（秒，贴近 slot 目标时长即可）
        caption: 该镜字幕（可留空；纯音乐/无字幕模式一律留空）
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_finish():
    """这一镜已生成并截好片段后结束。"""
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


AIGC_FUNCTION_TOOLS = [FunctionTool(fn) for fn in
                       (_tool_recall_product, _tool_pick_product_frame, _tool_gen_first_frame,
                        _tool_gen_storyboard, _tool_gen_video, _tool_review_and_extract, _tool_finish)]


_AIGC_FALLBACK = (
    "你是 AIGC 补镜 Agent，用 ReAct 方式为一个缺失镜头生成竖屏视频片段：看该镜对应的爆款分镜，"
    "决策是否需要产品参考图，需要则从用户素材池召回并取产品参考帧，据爆款分镜写 prompt，"
    "seedream 出首帧（必要时图生图保留真实商品），seedance 出视频，最后回看成片截取一段。"
    "每次只输出一个动作 JSON。可用工具：\n{{tools}}\n所有步骤完成后输出 {\"action\":\"finish\"}。"
)


def aigc_system_prompt() -> str:
    sk = get_skill("aigc_generator")
    base = sk.prompt_hint if (sk and sk.prompt_hint) else _AIGC_FALLBACK
    return base.replace("{{tools}}", edit_tools.render_tools_spec(AIGC_FUNCTION_TOOLS))


def tool_schemas() -> list:
    return edit_tools.tool_schemas(AIGC_FUNCTION_TOOLS)


async def _run_llm(system: str, user: str, tag: str = "") -> dict:
    content = ""
    async for item in as_core.stream(system, user):
        if "content" in item:
            content = item["content"]
    if tag:
        _log.info("%s output: %s", tag, (content or "").strip()[:1500])
    return as_core.parse_json(content) if content.strip() else {}


class AigcToolbox:
    """AIGC 生成用工具箱：用户素材池召回（Retriever）+ 视觉核验（VLM）+ 生成客户端。"""

    def __init__(self, task_id: str, segments: list, work_dir: str):
        self.retriever = Retriever(task_id)
        self.retriever.index_segments([s for s in (segments or []) if isinstance(s, dict)])
        self.by_gid = {s.get("global_asset_id"): s for s in (segments or [])
                       if isinstance(s, dict) and s.get("global_asset_id")}
        self.vlm = VLMTool()
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def recall(self, query: str, top_k: int = 6) -> list:
        res = self.retriever.search(query, top_k=top_k)
        out = []
        for m in res.get("matches", []):
            meta = m.get("meta", {})
            out.append({"global_asset_id": m.get("id"), "source_path": meta.get("source_path", ""),
                        "source_time_range": meta.get("source_time_range", ""),
                        "summary": meta.get("one_sentence_summary", ""),
                        "visual_description": meta.get("visual_description", "")})
        return out


def _mid_of(time_range: str) -> float:
    try:
        a, b = str(time_range).split("-")
        return (float(a) + float(b)) / 2.0
    except (ValueError, AttributeError):
        return 0.0


async def _run_aigc_slot(rid: str, slot: dict, toolbox: "AigcToolbox", *,
                         burn_on: bool = True):
    """为单个缺失镜头跑 AIGC ReAct，产出一个本地 clip；yield step 事件 + 最终 result。"""
    sid = slot["slot_id"]
    tag = f"[{rid}] AIGC {sid}"
    system = aigc_system_prompt()
    reference_shot = {"slot_id": sid, "role": slot.get("role", ""), "want": slot.get("want", ""),
                      "breakdown": slot.get("breakdown", []), "caption": slot.get("caption", ""),
                      "target_duration": slot.get("target_duration", 3.0),
                      "generation_prompt": slot.get("generation_prompt", "")}
    ctx = {"product_ref_path": "", "first_frame_url": "", "storyboard_url": "",
           "video_path": "", "video_dur": 0.0, "final_clip": None, "recalled": []}
    scratch = []
    step = 0
    yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid}",
                f"为缺失镜头 {sid}（{slot.get('role','')}）生成片段", state="running")

    while step < AIGC_MAX_STEPS:
        step += 1
        agent_user = json.dumps({
            "reference_shot": reference_shot,
            "state": {"has_product_ref": bool(ctx["product_ref_path"]),
                      "has_first_frame": bool(ctx["first_frame_url"]),
                      "has_storyboard": bool(ctx["storyboard_url"]),
                      "has_video": bool(ctx["video_path"]),
                      "video_duration": round(ctx["video_dur"], 2)},
            "mode": {"burn_caption": burn_on},
            "recent_steps": scratch[-6:],
            "instruction": ("只输出一个动作 JSON。按工作流推进：先决策是否要产品参考图 → "
                            "（需要则 recall_product + pick_product_frame）→ gen_first_frame → "
                            "（可选 gen_storyboard）→ gen_video → review_and_extract → finish。"
                            + ("本次不烧字幕，review_and_extract 的 caption 留空。" if not burn_on else "")),
        }, ensure_ascii=False)
        action = await _run_llm(system, agent_user, tag=(tag if step == 1 else ""))
        act = str(action.get("action", "")).lower().replace("_tool_", "")

        if act == "finish" or (not act and ctx["final_clip"]):
            break

        try:
            obs_res = await _dispatch(act, action, ctx, toolbox, slot, rid, sid, burn_on)
        except Exception as exc:  # noqa: BLE001
            _log.warning("%s tool '%s' failed: %s", tag, act, exc)
            obs_res = {"ok": False, "error": str(exc)[:200]}
        # 生成类动作对外冒个泡，方便前端看进度
        if act in ("gen_first_frame", "gen_video", "review_and_extract") and obs_res.get("ok"):
            yield _step(rid, f"aigc-{sid}-{act}", f"AIGC {sid} · {_ACT_LABEL.get(act, act)}",
                        obs_res.get("note", ""), state="done")
        scratch.append({"action": act, "observation": obs_res})
        if ctx["final_clip"] and act == "review_and_extract":
            # 有了最终片段，允许 Agent 再 finish；但也直接可结束
            pass

    result = ctx["final_clip"]
    if result:
        _log.info("%s done clip=%s %s", tag, os.path.basename(result["source_path"]), result["source_time_range"])
        yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid} 完成",
                    f"生成片段 {result['source_time_range']}", state="done")
    else:
        _log.warning("%s produced no clip", tag)
        yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid} 未产出", "未能生成可用片段", state="done")
    yield {"__aigc_slot_result__": True, "slot_id": sid, "clip": result}


_ACT_LABEL = {"gen_first_frame": "seedream 首帧", "gen_storyboard": "脚本图",
              "gen_video": "seedance 出片", "review_and_extract": "回看并截取"}


async def _dispatch(act: str, action: dict, ctx: dict, toolbox: "AigcToolbox",
                    slot: dict, rid: str, sid: str, burn_on: bool) -> dict:
    if act == "recall_product":
        query = action.get("query", "") or slot.get("want", "")
        cands = await asyncio.to_thread(toolbox.recall, query, int(action.get("top_k") or 6))
        ctx["recalled"] = cands
        return {"ok": True, "candidates": cands}

    if act == "pick_product_frame":
        gid = action.get("global_asset_id", "")
        seg = toolbox.by_gid.get(gid)
        if not seg:
            return {"ok": False, "error": f"未知 global_asset_id：{gid}"}
        ts = float(action.get("timestamp") if action.get("timestamp") not in (None, "") else -1.0)
        if ts < 0:
            ts = _mid_of(seg.get("source_time_range", ""))
        out_jpg = os.path.join(toolbox.work_dir, f"{sid}_prod_{int(time.time()*1000)%100000}.jpg")
        ok = await asyncio.to_thread(aigc_gen.extract_frame,
                                     _abspath(seg.get("source_path", "")), ts, out_jpg)
        if not ok:
            return {"ok": False, "error": "抽帧失败"}
        ctx["product_ref_path"] = out_jpg
        return {"ok": True, "note": f"已取产品参考帧 @{ts:.2f}s", "product_ref": os.path.basename(out_jpg)}

    if act == "gen_first_frame":
        prompt = (action.get("prompt") or "").strip() or slot.get("want", "")
        use_ref = bool(action.get("use_product_ref")) and bool(ctx["product_ref_path"])
        refs = [ctx["product_ref_path"]] if use_ref else None
        url = await asyncio.to_thread(aigc_gen.gen_image, prompt, ref_image_paths=refs)
        ctx["first_frame_url"] = url
        return {"ok": True, "note": f"首帧已生成（{'图生图' if use_ref else '文生图'}）", "first_frame_url": url}

    if act == "gen_storyboard":
        if not ctx["first_frame_url"]:
            return {"ok": False, "error": "请先 gen_first_frame"}
        motion = (action.get("motion_prompt") or "").strip()
        prompt = f"基于参考图的同一场景与主体，生成一张分镜指引图：{motion}。保持竖屏 9:16、风格一致。"
        url = await asyncio.to_thread(aigc_gen.gen_image, prompt, ref_image_urls=[ctx["first_frame_url"]])
        ctx["storyboard_url"] = url
        return {"ok": True, "note": "脚本图已生成", "storyboard_url": url}

    if act == "gen_video":
        first = ctx["storyboard_url"] or ctx["first_frame_url"]
        prompt = (action.get("prompt") or "").strip() or slot.get("want", "")
        dur = float(action.get("duration_sec") or slot.get("target_duration") or 5.0)
        if first:
            url = await asyncio.to_thread(aigc_gen.gen_video_i2v, prompt, first, dur)
        else:
            url = await asyncio.to_thread(aigc_gen.gen_video_t2v, prompt, dur)
        out_mp4 = os.path.join(toolbox.work_dir, f"{sid}_aigc_{int(time.time()*1000)%100000}.mp4")
        if not await asyncio.to_thread(aigc_gen.download, url, out_mp4):
            return {"ok": False, "error": "生成视频下载失败", "video_url": url}
        ctx["video_path"] = out_mp4
        ctx["video_dur"] = await asyncio.to_thread(aigc_gen.probe_duration, out_mp4)
        return {"ok": True, "note": f"seedance 出片 {ctx['video_dur']:.1f}s", "video_duration": ctx["video_dur"]}

    if act == "review_and_extract":
        if not ctx["video_path"]:
            return {"ok": False, "error": "还没有生成视频"}
        vdur = ctx["video_dur"] or aigc_gen.probe_duration(ctx["video_path"])
        start = max(0.0, float(action.get("start") or 0.0))
        dur = float(action.get("dur") or slot.get("target_duration") or 3.0)
        if start + dur > vdur > 0:
            dur = max(0.5, vdur - start)
        # 回看一眼（失败不阻断）
        try:
            r = await toolbox.vlm.inspect(
                f"这段生成视频是否贴合该镜意图：{slot.get('want','')}？描述画面主体、动作与是否稳定。",
                targets=[{"source_path": ctx["video_path"],
                          "source_time_range": f"{start:.2f}-{start+dur:.2f}", "asset_id": sid}])
            review = (r.get("observation") or "")[:400]
        except Exception:  # noqa: BLE001
            review = ""
        cap = (action.get("caption") or "").strip() if burn_on else ""
        rel = os.path.relpath(ctx["video_path"], AGENT_ROOT)
        ctx["final_clip"] = {
            "slot_id": sid, "source_path": rel,
            "source_time_range": f"{start:.2f}-{start+dur:.2f}",
            "target_duration": round(dur, 2), "caption": cap,
            "burn_caption": bool(cap), "speed": 1.0, "aigc": True,
        }
        return {"ok": True, "note": f"已截取 {start:.2f}-{start+dur:.2f}", "review": review}

    return {"ok": False, "error": "未知动作，请用 recall_product/pick_product_frame/gen_first_frame/gen_storyboard/gen_video/review_and_extract/finish"}


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(AGENT_ROOT, path))


def _step(rid, key, title, thought="", state="done"):
    return {"type": "step", "phase": "AIGC 补镜", "key": f"{key}-{rid}", "state": state,
            "title": title, "thought": thought}


async def generate_missing_shots(rid: str, aigc_slots: list, segments: list, task_id: str, *,
                                 burn_on: bool = True, max_concurrency: int = None):
    """按镜头数并行起子 Agent（上限 5）为所有缺失镜头生成片段。

    作为异步生成器：过程中 yield step 事件；最后 yield
    ``{"__aigc_result__": True, "clips_by_slot": {slot_id: clip}}``。
    """
    slots = [s for s in (aigc_slots or []) if s.get("slot_id")]
    clips_by_slot: dict = {}
    if not slots:
        yield {"__aigc_result__": True, "clips_by_slot": clips_by_slot}
        return
    if not aigc_gen.available():
        _log.warning("[%s] AIGC 网关不可用，跳过补镜", rid)
        yield _step(rid, "aigc-head", "AIGC 补镜不可用", "生成网关未配置，缺失镜头将被跳过", state="done")
        yield {"__aigc_result__": True, "clips_by_slot": clips_by_slot}
        return

    work_dir = os.path.join(AGENT_ROOT, "uploads", "aigc", task_id)
    toolbox = AigcToolbox(task_id, segments, work_dir)
    limit = max(1, min(max_concurrency or AIGC_MAX_SUBAGENTS, AIGC_MAX_SUBAGENTS))
    yield _step(rid, "aigc-head", "AIGC 补镜启动",
                f"{len(slots)} 个缺失镜头，最多 {limit} 个子 Agent 并行生成", state="running")

    out_q: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(limit)

    async def worker(slot):
        async with sem:
            async for ev in _run_aigc_slot(rid, slot, toolbox, burn_on=burn_on):
                await out_q.put(ev)
        await out_q.put({"__worker_done__": slot["slot_id"]})

    tasks = [asyncio.create_task(worker(s)) for s in slots]
    done = 0
    while done < len(slots):
        ev = await out_q.get()
        if ev.get("__worker_done__") is not None:
            done += 1
            continue
        if ev.get("__aigc_slot_result__"):
            if ev.get("clip"):
                clips_by_slot[ev["slot_id"]] = ev["clip"]
            continue
        yield ev
    await asyncio.gather(*tasks, return_exceptions=True)
    _log.info("[%s] AIGC 补镜完成 %d/%d", rid, len(clips_by_slot), len(slots))
    yield _step(rid, "aigc-head", "AIGC 补镜完成",
                f"成功生成 {len(clips_by_slot)}/{len(slots)} 个镜头", state="done")
    yield {"__aigc_result__": True, "clips_by_slot": clips_by_slot}
