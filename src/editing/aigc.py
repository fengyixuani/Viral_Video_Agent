"""AIGC 补镜 Agent（agent_cut 链路）：为缺失镜头用 seedream + seedance 生成片段。

设计对齐 agent_edit 的范式：用 skill（``aigc_generator``）约束 Agent、把功能做成 tool
（AgentScope 原生 FunctionTool 声明 schema），ReAct 循环里手动分发执行。可按镜头数并行
起多个子 Agent（每镜一个），并发上限 5。

单镜工作流（见 SKILL.md）：看爆款分镜 → 决策是否要产品参考图 → 需要则从用户素材池召回、
取产品参考帧 → 据爆款分镜写 prompt → seedream 出首帧 → seedance 由该首帧出视频 →
回看成片、按 slot 要求截取一段作为最终片段。

结束时每镜产出一个可直接进 editor 的 clip：
``{"slot_id","source_path"(本地 mp4),"source_time_range","target_duration","caption","aigc":True}``。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
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
# 单镜最多出片次数：VLM 判定不贴合时允许重生成，但要有上限（seedance 每次数百秒）
AIGC_MAX_GEN_VIDEO = int(os.getenv("AIGC_MAX_GEN_VIDEO", "2"))
# 首帧图生图最多带几张产品参考图（多角度/多细节的同一件商品，2–3 张效果最好；
# 太多会互相干扰，且必须是同一件商品——混进别款会直接把外观带偏）
AIGC_MAX_REF_IMAGES = int(os.getenv("AIGC_MAX_REF_IMAGES", "3"))
# 逐帧核验商品是否在画面里的抽帧间隔（秒）与最多帧数
AIGC_VIS_STEP = float(os.getenv("AIGC_VIS_STEP", "0.4"))
AIGC_VIS_MAX_FRAMES = int(os.getenv("AIGC_VIS_MAX_FRAMES", "16"))
# 多参考图直喂 seedance（跳过 seedream 首帧那次重绘）。实测 seedance 2.0 接受 base64 data URL，
# 多图时每张须带 role=reference_image。置 0 回退到"seedream 首帧 → i2v"的老路径。
AIGC_DIRECT_MULTIREF = os.getenv("AIGC_DIRECT_MULTIREF", "1") not in ("0", "false", "False")


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


def _tool_pick_product_frame(global_asset_id: str = "", timestamp: float = -1.0):
    """从某召回片段里抽取一帧作为**产品参考帧**（后续首帧走图生图时保留真实商品外观）。

    ``global_asset_id`` 留空即**自动挑选**：对 recall_product 的候选逐个抽帧，交给视觉模型
    比对后选商品最清晰、最正面、占画面主体的那一帧（推荐，因为你看不到画面）。

    Args:
        global_asset_id: recall_product 召回的候选片段 id；留空则自动从候选里挑最佳一帧
        timestamp: 抽帧时间点（秒，相对源视频）；<0 时自动取该片段中点；自动挑选时忽略
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_gen_first_frame(prompt: str, use_product_ref: bool = False):
    """用 seedream 生成这一镜的**首帧**（竖屏 9:16）。use_product_ref=true 且已取到产品参考帧时走图生图（保留真实商品外观），否则纯文生图。

    Args:
        prompt: 首帧画面描述（结合爆款分镜的景别/主体/展示重点/光影/风格，中文、具体）
        use_product_ref: 是否用已取的产品参考帧做图生图
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_gen_video(prompt: str, duration_sec: float = 5.0):
    """用 seedance 生成这一镜的视频。已取到产品参考帧时**直接把参考图喂给 seedance**（不需要先 gen_first_frame，少一次重绘、商品更像真的）；已经生成过首帧则用首帧图生视频。较慢，别重复生成。

    Args:
        prompt: 视频画面 + 动作/节奏描述（直喂参考图时要把这镜的画面也写清楚，不只是动作）
        duration_sec: 目标时长（秒）；会夹到 4–15s，可略长于 slot 目标，后面再截
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_review_and_extract(start: float, dur: float, caption: str = ""):
    """从已生成的视频里截取最贴合、最稳定的一段作为最终片段（放进成片）。

    `gen_video` 已经自动回看过整段并给出了带时间点的画面描述与建议截取区间，直接按那个
    建议给 start / dur 即可，不用自己猜。

    Args:
        start: 截取起点（秒，相对生成视频）——你只需要挑"从哪一段起"
        dur: 截取时长（秒）；会按编排给的 slot 目标时长自动校正，成片节奏要对齐参考爆款
        caption: 该镜字幕（可留空；纯音乐/无字幕模式一律留空）
    """
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


def _tool_finish():
    """这一镜已生成并截好片段后结束。"""
    raise NotImplementedError("schema 声明；由 AIGC ReAct 分发器执行")


AIGC_FUNCTION_TOOLS = [FunctionTool(fn) for fn in
                       (_tool_recall_product, _tool_pick_product_frame, _tool_gen_first_frame,
                        _tool_gen_video, _tool_review_and_extract, _tool_finish)]


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


def _range_dur(time_range: str) -> float:
    try:
        a, b = str(time_range).split("-")
        return max(0.0, float(b) - float(a))
    except (ValueError, AttributeError):
        return 0.0


_FACTS_PROMPT = (
    "这是用户手里**这件**商品的真实素材。请客观描述它的**结构事实**：品类形态、材质与表面工艺、"
    "有无缝线/拼接/鞋带/织物/五金等结构部件、开口或穿戴方式、以及可以做哪些展示动作"
    "（能否按压、弯折、撑开、翻转等）。只写画面里确实看得到的，看不清就说看不清。"
    "不要评价、不要营销语、不要推测参数。"
)


_PICK_PROMPT = (
    "下面是若干候选帧，按顺序编号 1..N。请挑出用于**商品图生图**的参考图。\n"
    "**首要标准是遮挡**：图生图只能照抄参考图里看得见的部分，被手/身体/其他物体挡住的部位"
    "模型只能凭空编，编出来就和真实商品不符。所以宁可要一张平平无奇但商品完整露出的帧，"
    "也不要一张构图好看却被手挡掉大半的帧。\n"
    "1) best：遮挡最少、商品最完整清晰的一帧（其次才看正面/占画面主体/无动态模糊）；\n"
    "2) occlusion：best 这一帧里商品被遮挡的程度，只能填「无」「轻微」「严重」之一"
    "（严重 = 商品主体有三分之一以上看不到）；\n"
    "3) hidden_parts：best 里**看不到或看不清**的部位（如\"鞋底\"\"后跟\"\"内里\"），没有就给 []；\n"
    "4) same_item：其余候选里**和 best 是同一件商品**、且能补上 hidden_parts 或提供不同角度的"
    "序号列表；不同款式/不同颜色/不同商品一律不要放进来，宁缺勿滥；\n"
    "只输出 JSON：{\"best\": 序号, \"occlusion\": \"无|轻微|严重\", \"hidden_parts\": [\"...\"], "
    "\"same_item\": [序号...], \"reason\": \"一句话理由\"}。"
)


def _probe_times(rng: str) -> list:
    """一个候选片段里取几个抽帧时间点：太短只取中点，够长就取 20%/50%/80%。

    只抽中点帧是之前"参考图被手挡掉大半"的直接原因——展示类素材的中段几乎必然是手在
    操作商品，而头尾常有商品静置的干净画面。多抽两帧几乎不额外花钱（仍是同一次视觉比对），
    却能让选帧有"没被遮住的帧"可选。
    """
    a, b = 0.0, 0.0
    try:
        _a, _b = str(rng).split("-")
        a, b = float(_a), float(_b)
    except (ValueError, AttributeError):
        return [_mid_of(rng)]
    dur = max(b - a, 0.0)
    if dur <= 0:
        return [_mid_of(rng)]
    if dur < 1.2:
        return [round(a + dur / 2, 2)]
    return [round(a + dur * p, 2) for p in (0.2, 0.5, 0.8)]


async def _auto_pick_frame(rid: str, sid: str, ctx: dict, toolbox: "AigcToolbox",
                           max_cands: int = 4) -> dict:
    """对 recall 候选各抽几帧，交给视觉模型挑参考图：一张主参考 + 若干同款补充参考。

    Agent 看不到画面（recall 只返回文字），让它自己指定 timestamp 等于盲猜；这里用一次视觉
    比对代替。挑选以**遮挡最小**为首要标准：图生图只能照抄看得见的部分，被手挡住的部位
    seedream 只会凭空编（实测编出来的鞋面样式和真实商品不符）。多张参考图能提升保真，但
    **必须是同一件商品**——混进别款会直接把外观带偏，所以由 VLM 判定同款，判不出就退化成
    单张。失败时退回第一帧，保证链路不断。
    """
    cands = [c for c in (ctx.get("recalled") or []) if c.get("global_asset_id")][:max_cands]
    if not cands:
        return {"ok": False, "error": "还没有召回候选，请先 recall_product"}
    frames = []
    for ci, c in enumerate(cands, 1):
        seg = toolbox.by_gid.get(c["global_asset_id"]) or c
        src = _abspath(seg.get("source_path", ""))
        for ts in _probe_times(seg.get("source_time_range", "")):
            idx = len(frames) + 1
            out_jpg = os.path.join(toolbox.work_dir,
                                   f"{sid}_cand{ci}_{idx}_{int(time.time()*1000)%100000}.jpg")
            if await asyncio.to_thread(aigc_gen.extract_frame, src, ts, out_jpg):
                frames.append({"idx": idx, "path": out_jpg, "gid": c["global_asset_id"],
                               "ts": ts, "cand": ci})
    if not frames:
        return {"ok": False, "error": "候选抽帧全部失败"}
    best, reason, same = frames[0], "", []
    occ, hidden = "", []
    if len(frames) > 1:
        try:
            res = await toolbox.vlm.inspect(_PICK_PROMPT, targets=[
                {"source_path": f["path"], "asset_id": str(f["idx"])} for f in frames],
                max_targets=len(frames))
            data = as_core.parse_json(res.get("observation") or "") or {}
            pick = int(data.get("best") or 0)
            reason = str(data.get("reason") or "")[:120]
            occ = str(data.get("occlusion") or "").strip()
            hidden = [str(x).strip() for x in (data.get("hidden_parts") or []) if str(x).strip()][:4]
            for f in frames:
                if f["idx"] == pick:
                    best = f
                    break
            want = {int(x) for x in (data.get("same_item") or []) if str(x).strip().isdigit()}
            # 补充参考优先给不同候选片段的帧：同一片段相邻时刻的帧信息高度重复，
            # 占了名额反而挤掉真正能补上被遮部位的角度。
            sib = [f for f in frames if f["idx"] in want and f is not best]
            same = sorted(sib, key=lambda f: (f["cand"] == best["cand"], f["idx"]))
        except (ValueError, TypeError) as exc:
            _log.warning("[%s] %s 选帧结果解析失败，用首个候选: %s", rid, sid, exc)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] %s 选帧视觉比对失败，用首个候选: %s", rid, sid, exc)
    picked = ([best] + same)[:AIGC_MAX_REF_IMAGES]   # 主参考在前（重要素材前置）
    ctx["product_ref_paths"] = [f["path"] for f in picked]
    ctx["product_ref_path"] = best["path"]
    ctx["ref_occlusion"] = occ
    ctx["ref_hidden_parts"] = hidden
    _log.info("[%s] AIGC %s 自动选帧: 候选%d帧 → 主参考第%d张 @%.2fs 遮挡=%s 未露出=%s，"
              "同款补充%d张 %s", rid, sid, len(frames), best["idx"], best["ts"],
              occ or "未知", "/".join(hidden) or "无", len(picked) - 1, reason)
    # 参考图被遮住的部位，图生图只能靠"文字事实"兜住，否则模型会自由发挥。把这句回给
    # Agent，让它在写首帧 prompt 时照商品事实档案描述这些部位，而不是任模型编。
    warn = ""
    if occ == "严重" or (hidden and occ != "无"):
        warn = ("；注意：主参考里 " + ("、".join(hidden) if hidden else "商品主体")
                + " 被遮住/看不清，写首帧 prompt 时必须按【商品事实】明确描述这些部位"
                  "（如无走线、一体成型），不要让模型自由发挥")
    return {"ok": True,
            "note": (f"已从 {len(frames)} 个候选帧中选出参考图："
                     f"主参考第{best['idx']}张 @{best['ts']:.2f}s"
                     + (f"（遮挡{occ}）" if occ else "")
                     + (f"，另带 {len(picked)-1} 张同款不同角度作补充参考" if len(picked) > 1 else "")
                     + (f"：{reason}" if reason else "") + warn),
            "product_ref": os.path.basename(best["path"]),
            "product_ref_count": len(picked),
            "occlusion": occ,
            "hidden_parts": hidden,
            "_media": [{"kind": "image", "url": "/" + _rel(f["path"]),
                        "label": ("主参考" if i == 0 else f"补充参考{i}")}
                       for i, f in enumerate(picked)]
                      + [{"kind": "image", "url": "/" + _rel(f["path"]), "label": f"未选用{f['idx']}"}
                         for f in frames if f not in picked]}



async def _build_product_facts(rid: str, toolbox: "AigcToolbox", segments: list) -> dict:
    """本商品事实档案：素材文字描述 + 对真实素材的一次视觉核验。

    补镜 Agent 拿到的 ``reference_shot`` 全部来自**参考爆款**（另一件商品），照 breakdown
    写 prompt 会画出本商品根本不具备的结构（洞洞鞋画出缝线、一体拖鞋"撑开鞋腔"）。文案侧
    早有事实核验（见 loop.py 的 _copy_violations），画面侧此前没有——这份档案就是画面侧的
    事实来源。整个 run 只算一次，所有子 Agent 共用。
    """
    briefs, seen = [], set()
    for seg in (segments or []):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("visual_description") or seg.get("one_sentence_summary") or "").strip()
        if text and text not in seen:
            seen.add(text)
            briefs.append(text[:120])
        if len(briefs) >= 8:
            break
    # 视觉核验：挑时长最长的两段（太短的窗口 VLM 会以 too short 拒掉）
    cands = sorted([s for s in (segments or []) if isinstance(s, dict) and s.get("source_path")],
                   key=lambda s: _range_dur(s.get("source_time_range", "")), reverse=True)[:2]
    visual = ""
    if cands:
        try:
            res = await toolbox.vlm.inspect(_FACTS_PROMPT, targets=[
                {"source_path": c.get("source_path", ""),
                 "source_time_range": c.get("source_time_range", ""),
                 "asset_id": c.get("global_asset_id", "")} for c in cands])
            visual = (res.get("observation") or "").strip()[:1500]
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] 商品事实视觉核验失败: %s", rid, exc)
    facts = {"from_materials": briefs, "visual_facts": visual}
    _log.info("[%s] 商品事实档案：素材描述 %d 条，视觉核验 %d 字", rid, len(briefs), len(visual))
    if visual:
        _log.info("[%s] 商品结构事实: %s", rid, visual[:400])
    return facts


async def _run_aigc_slot(rid: str, slot: dict, toolbox: "AigcToolbox", *,
                         burn_on: bool = True, product_facts: dict = None):
    """为单个缺失镜头跑 AIGC ReAct，产出一个本地 clip；yield step 事件 + 最终 result。"""
    sid = slot["slot_id"]
    tag = f"[{rid}] AIGC {sid}"
    system = aigc_system_prompt()
    reference_shot = {"slot_id": sid, "role": slot.get("role", ""), "want": slot.get("want", ""),
                      "breakdown": slot.get("breakdown", []), "caption": slot.get("caption", ""),
                      "target_duration": slot.get("target_duration", 3.0),
                      "generation_prompt": slot.get("generation_prompt", "")}
    ctx = {"product_ref_path": "", "product_ref_paths": [],
           "first_frame_url": "",
           "video_path": "", "video_dur": 0.0, "final_clip": None, "recalled": [],
           # 生成不可复现：prompt / 首帧 / 回看结论不记就永久丢失，事后无从判断"哪一步不好"
           "first_frame_path": "", "prompts": {}, "review": "",
           "gen_video_count": 0, "verdict": "", "reason_kind": "", "unusable": False,
           "usable_range": None, "visible_mask": ""}
    scratch = []
    step = 0
    yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid}",
                f"为缺失镜头 {sid}（{slot.get('role','')}）生成片段", state="running")

    while step < AIGC_MAX_STEPS:
        step += 1
        agent_user = json.dumps({
            "reference_shot": reference_shot,
            # 本商品的真实结构事实：reference_shot 描述的是**参考爆款里的另一件商品**，
            # 两者冲突时以这里为准（详见 _build_product_facts）
            "product_facts": product_facts or {},
            "state": {"has_product_ref": bool(ctx["product_ref_path"]),
                      "has_first_frame": bool(ctx["first_frame_url"]),
                      "has_video": bool(ctx["video_path"]),
                      "video_duration": round(ctx["video_dur"], 2)},
            "mode": {"burn_caption": burn_on},
            "recent_steps": scratch[-6:],
            "instruction": ("只输出一个动作 JSON。按工作流推进：先决策是否要产品参考图 → "
                            "（需要则 recall_product + pick_product_frame）→ gen_first_frame → "
                            "gen_video → review_and_extract → finish。"
                            "写 prompt 前先对照 product_facts：reference_shot 的 breakdown 描述的是"
                            "参考爆款里的另一件商品，凡是本商品不具备的结构（缝线/拼接/鞋带/织物/"
                            "五金等）一律不得出现，不成立的动作（如一体成型的鞋不能'撑开鞋腔'）"
                            "要换成本商品真实可做的等效展示。"
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
        # 生成类动作对外冒个泡：把参考图/首帧/成片和当次 prompt 一并推给前端
        media = obs_res.pop("_media", None) if isinstance(obs_res, dict) else None
        if act in _ACT_LABEL and obs_res.get("ok"):
            yield _step(rid, f"aigc-{sid}-{act}", f"AIGC {sid} · {_ACT_LABEL[act]}",
                        obs_res.get("note", ""), state="done", media=media)
        # 存**完整**动作（含 prompt 等参数），不只动作名：否则写视频 prompt 那步看不到自己
        # 上一步写的首帧 prompt，两段 prompt 只能各自对着 breakdown 猜，容易前后不一致。
        scratch.append({"action": act,
                        "params": {k: v for k, v in action.items() if k != "action"},
                        "observation": obs_res})
        if ctx["final_clip"] and act == "review_and_extract":
            # 有了最终片段，允许 Agent 再 finish；但也直接可结束
            pass

    result = ctx["final_clip"]
    _save_slot_meta(toolbox.work_dir, sid, slot, ctx, result, product_facts)
    if result:
        _log.info("%s done clip=%s %s", tag, os.path.basename(result["source_path"]), result["source_time_range"])
        yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid} 完成",
                    f"生成片段 {result['source_time_range']}", state="done")
    else:
        _log.warning("%s produced no clip", tag)
        yield _step(rid, f"aigc-{sid}", f"AIGC 补镜 · {sid} 未产出", "未能生成可用片段", state="done")
    yield {"__aigc_slot_result__": True, "slot_id": sid, "clip": result}


_ACT_LABEL = {"pick_product_frame": "产品参考帧", "gen_first_frame": "seedream 首帧",
              "gen_video": "seedance 出片",
              "review_and_extract": "回看并截取"}


_INSPECT_TMPL = (
    "这是刚生成的一段候选视频，本镜的意图是：{want}（承担的角色：{role}）。请：\n"
    "1) 按时间顺序描述画面里发生了什么，标出大致时间点（如 0-1s / 1-2.5s）；\n"
    "2) 商品是否全程在画面内？凡是商品消失、飞出画面、被完全遮挡、或形变到认不出的时段都要指出，"
    "然后给出**商品清晰完整可见**的最长一段连续区间，**单独一行、格式固定**：\n"
    "   `可用区间：起点-终点`（单位秒，例：`可用区间：0.0-2.6`）；全程都可见就写 `可用区间：全程`。\n"
    "3) 在可用区间内，指出哪一段最稳定、最贴合上述意图，给出建议截取起点；\n"
    "4) 判定是否贴合意图。单独一行只写「判定：贴合」或「判定：不贴合」；\n"
    "   若不贴合，下一行只写「原因类别：生成质量」或「原因类别：意图不可达」——\n"
    "   生成质量 = 画面模糊/主体形变或消失/动作僵硬穿模等，换个描述重新生成有机会解决；\n"
    "   意图不可达 = 这个意图需要画面里根本没有、也无法凭这件商品拍出的东西（如需要真人试穿、"
    "需要多件对比、需要商品不具备的部件），重新生成也不会变好。\n"
    "然后再写理由。\n"
    "注意：**字幕、文案与配音都由后期叠加，不属于本次生成范围**，不要因为画面里没有文字、"
    "字幕或听不到口播就判不贴合；只评估画面本身能否承载这一镜的视觉意图。\n"
    "只依据画面回答，不要臆测画面外的内容。"
)

_VIS_PROMPT = (
    "下面是同一段视频按时间顺序抽出的 {n} 张帧，编号 1..{n}。逐张判断：**目标商品是否完整、"
    "清晰地出现在这一帧里**。被手部小面积遮挡算出现；商品只剩局部边缘、被完全遮挡、飞出画面、"
    "形变到认不出、或这一帧里根本没有商品，都算未出现。\n"
    "只输出 JSON：{{\"visible\": [1或0, ...]}}，数组长度必须等于 {n}，顺序与编号一致。"
)


async def _visible_mask(rid: str, sid: str, ctx: dict, toolbox: "AigcToolbox"):
    """按固定间隔抽帧，用一次视觉核验逐帧判断"商品在不在画面里"，返回 [(时间点, 0/1)]。

    整段回看给的"可用区间"是模型自己读时间轴的结果，时间精度粗、容易漏掉短暂消失；
    逐帧核验把它变成可计算的掩码，成片"每帧都有商品"这条要求才能真正落地。
    """
    vdur = float(ctx.get("video_dur") or 0.0)
    if vdur <= 0.6:
        return None
    times, t = [], 0.0
    while t < vdur - 0.05 and len(times) < AIGC_VIS_MAX_FRAMES:
        times.append(round(t, 2))
        t += AIGC_VIS_STEP
    frames = []
    for i, ts in enumerate(times, 1):
        out = os.path.join(toolbox.work_dir, f"{sid}_vis{i}_{int(time.time()*1000)%100000}.jpg")
        if await asyncio.to_thread(aigc_gen.extract_frame, ctx["video_path"], ts, out):
            frames.append((ts, out))
    if len(frames) < 2:
        return None
    vis = []
    try:
        res = await toolbox.vlm.inspect(
            _VIS_PROMPT.format(n=len(frames)),
            targets=[{"source_path": p, "asset_id": str(i)} for i, (_, p) in enumerate(frames, 1)],
            max_targets=len(frames))
        data = as_core.parse_json(res.get("observation") or "") or {}
        vis = [1 if int(x) else 0 for x in (data.get("visible") or [])]
    except (ValueError, TypeError) as exc:
        _log.warning("[%s] AIGC %s 逐帧核验解析失败: %s", rid, sid, exc)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] AIGC %s 逐帧核验失败: %s", rid, sid, exc)
    finally:
        for _, p in frames:      # 探针帧是一次性的，别留在产物目录里
            try:
                os.remove(p)
            except OSError:
                pass
    if len(vis) != len(frames):
        return None
    return [(ts, v) for (ts, _), v in zip(frames, vis)]


def _best_visible_window(mask, vdur: float):
    """从可见掩码里取最长的连续可见区间；不足 0.5s 返回 None。"""
    best = None
    i, n = 0, len(mask)
    while i < n:
        if not mask[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1][1]:
            j += 1
        a = mask[i][0]
        b = min(vdur, mask[j][0] + AIGC_VIS_STEP) if j + 1 < n else vdur
        if best is None or (b - a) > (best[1] - best[0]):
            best = (a, b)
        i = j + 1
    return best if best and best[1] - best[0] >= 0.5 else None


_RANGE_RE = re.compile(r"可用区间[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*[-~到至]\s*([0-9]+(?:\.[0-9]+)?)")


async def _inspect_generated(rid: str, sid: str, slot: dict, ctx: dict,
                             toolbox: "AigcToolbox") -> str:
    """对**整段**生成视频做一次回看，产出带时间点的描述 + 贴合判定 + 建议截取区间。

    必须看整段而不是截取窗口：一是短窗口会被 VLM 以 "video too short" 拒掉（快切镜目标
    时长常不足 1s），二是只有先看到画面，Agent 才有依据决定重生成还是截哪一段——之前
    回看跑在截取之后，结论无人可用，等于开环。

    判定同时要一个**原因类别**：只有"生成质量"类才值得重生成，"意图不可达"类重生成纯烧时间
    （实测三镜全判不贴合、全重生成一次、全没翻盘）。
    """
    prompt = _INSPECT_TMPL.format(want=slot.get("want", "") or "（未给定）",
                                  role=slot.get("role", "") or "（未给定）")
    try:
        res = await toolbox.vlm.inspect(prompt, targets=[
            {"source_path": ctx["video_path"], "asset_id": sid}])
        review = (res.get("observation") or "").strip()
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] AIGC %s 回看失败: %s", rid, sid, exc)
        return ""
    ctx["review"] = review[:1500]
    ctx["verdict"] = "不贴合" if "判定：不贴合" in review else ("贴合" if "判定：贴合" in review else "")
    # 原因类别只在"不贴合"时才有意义：VLM 在判定贴合时也可能在正文里提到"意图不可达"这个词
    # 来解释判据，早先无条件按关键词解析，导致判定贴合的片段被误标成不可达、进而被换掉。
    ctx["reason_kind"] = ""
    if ctx["verdict"] == "不贴合":
        if "意图不可达" in review:
            ctx["reason_kind"] = "意图不可达"
        elif "生成质量" in review:
            ctx["reason_kind"] = "生成质量"
    # 只有"画面本身坏了"才算不可用（形变/模糊/主体消失这类）。意图对不上但画面可用的照样进成片——
    # 参考爆款要脚、要对比、要人物时静物生成永远"不可达"，照那个标准 AIGC 会被全部替换掉。
    ctx["unusable"] = ctx["reason_kind"] == "生成质量"
    # 商品清晰可见的可用区间：seedance 常在片段中段把商品搞消失/飞出画面，
    # 光靠 Agent 自觉避开不可靠，这里解析出硬边界，截取时强制夹在里面。
    ctx["usable_range"] = None
    m = _RANGE_RE.search(review)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b - a >= 0.5:
            ctx["usable_range"] = (a, b)
    _log.info("[%s] AIGC %s 回看判定=%s/%s%s 可用区间=%s %s", rid, sid, ctx["verdict"] or "未明确",
              ctx["reason_kind"] or "-", "（画面不可用）" if ctx["unusable"] else "",
              ("%.2f-%.2f" % ctx["usable_range"]) if ctx["usable_range"] else "全程",
              review[:180].replace("\n", " "))
    return ctx["review"]


def _clip_desc(ctx: dict, slot: dict, limit: int = 300) -> str:
    """生成片段的画面描述：首帧 prompt（长什么样）+ 视频 prompt（怎么动）。

    给下游剪辑 Agent 当"这一镜实际拍到了什么"用，所以要的是画面本身，不是回看的评价。
    """
    ff = (((ctx.get("prompts") or {}).get("first_frame") or {}).get("prompt", "")).rstrip("。，")
    vp = (((ctx.get("prompts") or {}).get("video") or {}).get("prompt", "")).rstrip("。，")
    text = "；".join(x for x in (ff, vp) if x) or slot.get("want", "")
    return text[:limit]


async def _dispatch(act: str, action: dict, ctx: dict, toolbox: "AigcToolbox",
                    slot: dict, rid: str, sid: str, burn_on: bool) -> dict:
    if act == "recall_product":
        query = action.get("query", "") or slot.get("want", "")
        cands = await asyncio.to_thread(toolbox.recall, query, int(action.get("top_k") or 6))
        ctx["recalled"] = cands
        return {"ok": True, "candidates": cands}

    if act == "pick_product_frame":
        gid = (action.get("global_asset_id") or "").strip()
        if not gid or gid.lower() in ("auto", "best"):
            return await _auto_pick_frame(rid, sid, ctx, toolbox)
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
        ctx["product_ref_paths"] = [out_jpg]
        return {"ok": True, "note": f"已取产品参考帧 @{ts:.2f}s",
                "product_ref": os.path.basename(out_jpg),
                "_media": [{"kind": "image", "url": "/" + _rel(out_jpg), "label": "产品参考帧"}]}

    if act == "gen_first_frame":
        prompt = (action.get("prompt") or "").strip() or slot.get("want", "")
        use_ref = bool(action.get("use_product_ref")) and bool(ctx["product_ref_path"])
        refs = (ctx.get("product_ref_paths") or [ctx["product_ref_path"]])[:AIGC_MAX_REF_IMAGES] \
            if use_ref else None
        url = await asyncio.to_thread(aigc_gen.gen_image, prompt, ref_image_paths=refs)
        ctx["first_frame_url"] = url
        ctx["prompts"]["first_frame"] = {"prompt": prompt, "use_product_ref": use_ref,
                                        "ref_count": len(refs or [])}
        _log.info("[%s] AIGC %s 首帧 prompt(%s, 参考图%d张): %s", rid, sid,
                  "i2i" if use_ref else "t2i", len(refs or []), prompt[:300])
        # bos_url 是临时直链，过期后无从复盘"这镜为什么生成成这样"，落一份本地
        ctx["first_frame_path"] = await _save_image(toolbox.work_dir, f"{sid}_first", url)
        return {"ok": True,
                "note": (f"首帧已生成（{'图生图' if use_ref else '文生图'}"
                         + (f"，参考图 {len(refs)} 张" if use_ref else "") + f"）｜prompt：{prompt}"),
                "first_frame_url": url,
                "_media": ([{"kind": "image", "url": "/" + ctx["first_frame_path"], "label": "seedream 首帧"}]
                           if ctx["first_frame_path"] else [])}

    if act == "gen_video":
        if ctx["gen_video_count"] >= AIGC_MAX_GEN_VIDEO:
            return {"ok": False, "error": f"已出片 {ctx['gen_video_count']} 次（上限 "
                                         f"{AIGC_MAX_GEN_VIDEO}），不再重生成；"
                                         f"请用 review_and_extract 从现有视频里截最好的一段"}
        # 意图不可达（缺真人试穿/缺对比/商品不具备该部件）重生成不会变好，直接拦掉省时间
        if ctx["gen_video_count"] >= 1 and ctx["reason_kind"] == "意图不可达":
            return {"ok": False,
                    "error": "上一次回看判定为『意图不可达』——这一镜要的东西无法凭本商品的"
                             "画面拍出来，重新生成也不会变好，不再出片；请用 review_and_extract "
                             "从现有视频里截最稳定的一段收尾"}
        first = ctx["first_frame_url"]
        refs = [p for p in (ctx.get("product_ref_paths") or []) if p and os.path.isfile(p)]
        # 优先"多参考图直喂 seedance"：真实商品帧直接进 seedance，省掉 seedream 那次重绘。
        # 商品外观原来要在「真实帧 → seedream 首帧 → 视频」上被重构两次，是不像真实商品的
        # 主要来源；实测直喂后 VLM 比对结论为"一致"。首帧已经生成过就沿用（Agent 的显式选择）。
        direct = AIGC_DIRECT_MULTIREF and not first and bool(refs)
        prompt = (action.get("prompt") or "").strip() or slot.get("want", "")
        dur = float(action.get("duration_sec") or slot.get("target_duration") or 5.0)
        mode = "multiref" if direct else ("i2v" if first else "t2v")
        ctx["prompts"]["video"] = {"prompt": prompt, "duration_sec": dur, "mode": mode,
                                   "ref_images": len(refs) if direct else 0}
        _log.info("[%s] AIGC %s 视频 prompt(%s%s, %.1fs): %s", rid, sid, mode,
                  f", 参考图{len(refs)}张" if direct else "", dur, prompt[:300])
        if direct:
            url = await asyncio.to_thread(aigc_gen.gen_video_multiref, prompt, refs, dur)
        elif first:
            url = await asyncio.to_thread(aigc_gen.gen_video_i2v, prompt, first, dur)
        else:
            url = await asyncio.to_thread(aigc_gen.gen_video_t2v, prompt, dur)
        out_mp4 = os.path.join(toolbox.work_dir, f"{sid}_aigc_{int(time.time()*1000)%100000}.mp4")
        if not await asyncio.to_thread(aigc_gen.download, url, out_mp4):
            return {"ok": False, "error": "生成视频下载失败", "video_url": url}
        # seedance 会自带背景音乐/音效：删掉音轨，成片的声音只来自 BGM + 配音 + 用户原声
        await asyncio.to_thread(aigc_gen.strip_audio, out_mp4)
        ctx["video_path"] = out_mp4
        ctx["video_dur"] = await asyncio.to_thread(aigc_gen.probe_duration, out_mp4)
        ctx["gen_video_count"] += 1
        # 出片后立刻回看整段：给出带时间点的画面描述 + 贴合判定 + 建议截取区间，
        # Agent 据此决定"再生成一次"还是"按建议截取"，不再盲猜。
        review = await _inspect_generated(rid, sid, slot, ctx, toolbox)
        # 逐帧核验商品是否在画面里：把"每帧都有商品"变成可计算的掩码，覆盖整段回看给的粗区间
        mask = await _visible_mask(rid, sid, ctx, toolbox)
        if mask:
            ctx["visible_mask"] = "".join(str(v) for _, v in mask)
            win = _best_visible_window(mask, ctx["video_dur"])
            if win:
                ctx["usable_range"] = win
            else:
                ctx["unusable"] = True     # 整段都没有完整商品，靠生成救不回来
            _log.info("[%s] AIGC %s 商品可见掩码 %s（步长%.1fs）→ 可用区间=%s", rid, sid,
                      ctx["visible_mask"], AIGC_VIS_STEP,
                      ("%.2f-%.2f" % win) if win else "无（整段无完整商品）")
        left = AIGC_MAX_GEN_VIDEO - ctx["gen_video_count"]
        can_regen = ctx["verdict"] == "不贴合" and ctx["reason_kind"] != "意图不可达" and left > 0
        return {"ok": True,
                "note": (f"seedance 出片 {ctx['video_dur']:.1f}s"
                         + (f"｜回看判定：{ctx['verdict']}"
                            + (f"（{ctx['reason_kind']}）" if ctx["reason_kind"] else "")
                            if ctx["verdict"] else "")
                         + f"｜prompt：{prompt}"),
                "video_duration": ctx["video_dur"],
                "verdict": ctx["verdict"], "reason_kind": ctx["reason_kind"], "review": review,
                "regenerate_left": left if can_regen else 0,
                "hint": ("回看判定不贴合、原因是生成质量，可再 gen_video 一次（换动作/运镜思路）。"
                         if can_regen else
                         "不要再 gen_video 了：意图不可达或已无配额。按回看给出的建议区间调用 "
                         "review_and_extract，截最稳定的一段收尾。"
                         if ctx["verdict"] == "不贴合" else
                         "按回看给出的建议区间调用 review_and_extract。"),
                "_media": ([{"kind": "image", "url": "/" + _rel(p),
                             "label": ("直喂参考图%d" % (i + 1))} for i, p in enumerate(refs)]
                           if direct else
                           ([{"kind": "image", "url": "/" + ctx["first_frame_path"],
                              "label": "本次输入首帧"}] if ctx["first_frame_path"] else []))
                          + [{"kind": "video", "url": "/" + _rel(out_mp4),
                              "label": f"seedance 出片{ctx['gen_video_count']}"}]}

    if act == "review_and_extract":
        if not ctx["video_path"]:
            return {"ok": False, "error": "还没有生成视频"}
        vdur = ctx["video_dur"] or aigc_gen.probe_duration(ctx["video_path"])
        start = max(0.0, float(action.get("start") or 0.0))
        dur = float(action.get("dur") or slot.get("target_duration") or 3.0)
        # 时长以编排给的目标为准（成片节奏是对齐参考爆款的），Agent 只负责挑"从哪一段起"。
        # 否则它会在判定不贴合、挑不出好段时直接截整段——4.5s 的静止空镜压在结尾，节奏就垮了。
        target = float(slot.get("target_duration") or 0.0)
        if target > 0 and abs(dur - target) > 0.05:
            _log.info("[%s] AIGC %s 截取时长按目标校正 %.2fs -> %.2fs", rid, sid, dur, target)
            dur = target
        # 强制夹进"商品清晰可见"的可用区间：中段商品消失/飞出画面的那截绝不能进成片
        ur = ctx.get("usable_range")
        if ur and ur[1] - ur[0] >= 0.5:
            a, b = ur
            old = (start, dur)
            start = min(max(start, a), max(a, b - dur))
            if start + dur > b:
                dur = max(0.5, b - start)
            if (round(old[0], 2), round(old[1], 2)) != (round(start, 2), round(dur, 2)):
                _log.info("[%s] AIGC %s 截取夹进可用区间 %.2f-%.2f：%.2f+%.2f -> %.2f+%.2f",
                          rid, sid, a, b, old[0], old[1], start, dur)
        if vdur > 0 and start + dur > vdur:      # 越界优先往前挪起点，保住时长
            start = max(0.0, vdur - dur)
            if start + dur > vdur:
                dur = max(0.5, vdur - start)
        # 回看已在 gen_video 时对整段做过（见 _inspect_generated），这里直接复用结论，
        # 不再对截取窗口二次调用 VLM——短窗口会被以 "video too short" 拒掉。
        review = ctx.get("review", "")
        cap = (action.get("caption") or "").strip() if burn_on else ""
        rel = os.path.relpath(ctx["video_path"], AGENT_ROOT)
        ctx["final_clip"] = {
            "slot_id": sid, "source_path": rel,
            "source_time_range": f"{start:.2f}-{start+dur:.2f}",
            "target_duration": round(dur, 2), "caption": cap,
            "burn_caption": bool(cap), "speed": 1.0, "aigc": True,
            "prompts": dict(ctx["prompts"]),
            # aigc:: 是合成 id、生成的 mp4 也不在素材池里，下游 _material_brief 两条路都查不到
            # 内容，剪辑 Agent 会按"素材撑不起这一段"把整镜 skip 掉。用两段 prompt 拼出画面描述
            # （比回看结论更适合当素材描述——回看是带"判定：不贴合"的评价，不是画面本身）。
            "visual_description": _clip_desc(ctx, slot),
            # 回看判定透出去：unusable=画面本身坏了（上层会换真实素材顶上）；
            # intent_unreachable 只作信息记录——意图对不上但画面可用的仍然进成片
            "verdict": ctx.get("verdict", ""),
        "unusable": bool(ctx.get("unusable")),
        "usable_range": ctx.get("usable_range"),
        "visible_mask": ctx.get("visible_mask", ""),
            "intent_unreachable": ctx.get("reason_kind") == "意图不可达",
        }
        return {"ok": True, "note": f"已截取 {start:.2f}-{start+dur:.2f}", "review": review,
                "_media": [{"kind": "video", "url": "/" + _rel(ctx["video_path"]),
                            "label": f"最终片段 {start:.2f}-{start+dur:.2f}s"}]}

    return {"ok": False, "error": "未知动作，请用 recall_product/pick_product_frame/gen_first_frame/gen_video/review_and_extract/finish"}


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(AGENT_ROOT, path))


def _rel(path: str) -> str:
    """绝对路径 → 相对项目根的 ``uploads/...``（前端按 /uploads/... 直接取用）。"""
    if not path:
        return ""
    try:
        return os.path.relpath(path, AGENT_ROOT) if os.path.isabs(path) else path
    except ValueError:
        return path


async def _save_image(work_dir: str, stem: str, url: str) -> str:
    """把生成图（seedream 返回的临时 bos_url）落一份本地，失败不阻断生成流程。"""
    if not url:
        return ""
    out_jpg = os.path.join(work_dir, f"{stem}_{int(time.time()*1000)%100000}.jpg")
    ok = await asyncio.to_thread(aigc_gen.download, url, out_jpg)
    if not ok:
        _log.warning("生成图落盘失败 %s", stem)
        return ""
    return os.path.relpath(out_jpg, AGENT_ROOT)


def _save_slot_meta(work_dir: str, sid: str, slot: dict, ctx: dict, result, product_facts=None):
    """这一镜的 prompt / 参考图 / 首帧 / 回看结论落盘，供事后复盘。

    生成结果不可复现，且 clip 只带得走少数字段（loop 拿它拼 preset 时会丢弃其余），
    所以单独写一份 sidecar，文件名跟着 mp4 走以便配对。
    """
    stem = (os.path.splitext(os.path.basename(ctx.get("video_path") or ""))[0]
            or f"{sid}_{int(time.time()*1000)%100000}")
    meta = {
        "slot_id": sid,
        "role": slot.get("role", ""),
        "want": slot.get("want", ""),
        "breakdown": slot.get("breakdown", []),
        "target_duration": slot.get("target_duration"),
        "product_facts": product_facts or {},
        "prompts": ctx.get("prompts", {}),
        "product_ref_path": ctx.get("product_ref_path", ""),
        "product_ref_paths": ctx.get("product_ref_paths", []),
        "first_frame_url": ctx.get("first_frame_url", ""),
        "first_frame_path": ctx.get("first_frame_path", ""),
        "video_path": ctx.get("video_path", ""),
        "video_duration": round(float(ctx.get("video_dur") or 0.0), 2),
        "review": ctx.get("review", ""),
        "verdict": ctx.get("verdict", ""),
        "reason_kind": ctx.get("reason_kind", ""),
        "unusable": bool(ctx.get("unusable")),
        "gen_video_count": ctx.get("gen_video_count", 0),
        "clip": result,
    }
    try:
        with open(os.path.join(work_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        _log.warning("AIGC %s meta 落盘失败: %s", sid, exc)


def _step(rid, key, title, thought="", state="done", media=None):
    ev = {"type": "step", "phase": "AIGC 补镜", "key": f"{key}-{rid}", "state": state,
          "title": title, "thought": thought}
    if media:
        ev["media"] = media
    return ev


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
    # 先建本商品事实档案（整个 run 一次），再起子 Agent —— 否则每镜都照参考爆款的 breakdown
    # 写 prompt，会生成本商品不存在的结构/不成立的动作
    yield _step(rid, "aigc-facts", "商品事实核对", "看真实素材，确认本商品的形态/材质/结构与可做动作",
                state="running")
    product_facts = await _build_product_facts(rid, toolbox, segments)
    yield _step(rid, "aigc-facts", "商品事实核对",
                (product_facts.get("visual_facts") or "；".join(product_facts.get("from_materials") or [])
                 or "未取到商品事实，将只依据参考分镜生成")[:300], state="done")
    yield _step(rid, "aigc-head", "AIGC 补镜启动",
                f"{len(slots)} 个缺失镜头，最多 {limit} 个子 Agent 并行生成", state="running")

    out_q: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(limit)

    async def worker(slot):
        async with sem:
            async for ev in _run_aigc_slot(rid, slot, toolbox, burn_on=burn_on,
                                           product_facts=product_facts):
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
