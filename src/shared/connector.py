"""Connector: 把 Agent 导出的 Split 兼容脚本接到 Viral_Video_Split 编排层之后。

点击「剪辑」后，本模块以 Agent 已导出的 ``selected_editing_strategy.json`` 为输入，
合成 Split 后链路（rebuild_asr_edit → execute_tool_plan → TTS → 字幕 → BGM）所需的
四个契约文件，然后子进程调用 ``common/run_asr_remake_main_chain.py`` 完成真实剪辑，
把日志与最终成片路径流式回传。

设计要点（只改 Agent 侧，不动 Split 代码）：
- Split 的最终链路强约束四个入参：BASE_PLAN / BASE_STRATEGY / SCORES_PATH / ASR_PATH，
  全部用 ``require_existing_env_path`` 校验。本模块负责把 Agent 的一份 strategy 展开成
  这四份文件。
- Agent 的 editing_timeline 里 source_path 是相对 ``uploads/...``；Split 以自己的
  仓库根为 CWD，因此这里统一解析成绝对路径。
- **选片对齐复刻分镜（方案 A）**：SCORES 只用编排 Agent 的选择——每镜首选 = 复刻分镜
  展示的那个片段（最高分 1.0），备选 = 可行性验证/编排继承下来的 alternate_assets
  （0.9 递减）。连接器**不再自己对全量素材池重新检索**，因此 Split 首选就是复刻分镜
  那个片段，成片=复刻分镜；只有当首选与别的镜头重叠时，Split 才从继承的备选里换片。
- **保留 Split 的不重叠/去尾约束默认开启**（不关闭 ENFORCE_FINAL_NON_OVERLAP 等），
  用继承的备选做去重换片，避免重复或丢镜头。
- 字幕相关性：对首选/备选涉及的素材跑本机离线 ASR，Split 的字幕跟随所选片段的真实
  口播（asr 语义足够时），因此字幕与画面对齐、不再复读。
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time

import obs
import cache  # noqa: F401  # 保留供未来子模块使用；ASR 命中共享 asr_cache
import asr_cache
from tools import ASRTool, Retriever

_log = obs.get_logger("connector")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_ROOT = os.getenv("VIRAL_VIDEO_SPLIT_ROOT", "/root/chengzhiyang/Viral_Video_Split")
MAIN_CHAIN = os.path.join(SPLIT_ROOT, "common", "run_asr_remake_main_chain.py")
DEFAULT_BGM = os.path.join(
    os.path.dirname(SPLIT_ROOT), "Viral_Video", "带货素材", "dataset", "可商用bgm",
    "时尚动感放克律动 Funk Caravan Main_爱给网_aigei_com.mp3",
)
RICH_TOP_K = int(os.getenv("CONNECTOR_TOP_K_PER_SLOT", "8"))

_ASR = ASRTool()


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _abspath(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(AGENT_ROOT, path))


def _slug(text: str) -> str:
    ascii_slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(text or "")).strip("_")
    return ascii_slug or "agent_edit"


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _fmt_range(start, end):
    return "{:.2f}-{:.2f}".format(float(start), float(end))


def _load_split_env() -> dict:
    """读取 Split 的 .env（机器本地解释器/模型路径），os.environ 优先。"""
    env = {}
    path = os.path.join(SPLIT_ROOT, ".env")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _kept_timeline(strategy: dict) -> list:
    """挑出可直接剪辑的镜头：动作为 use_user_asset 且有合法源片段。"""
    kept = []
    for item in strategy.get("editing_timeline", []) or []:
        if item.get("action") != "use_user_asset":
            continue
        source_path = _abspath(item.get("source_path", ""))
        start, end = _parse_range(item.get("source_time_range", ""))
        if not source_path or not os.path.isfile(source_path) or end <= start:
            continue
        kept.append({
            "slot_id": item.get("slot_id", ""),
            "source_path": source_path,
            "source_time_range": _fmt_range(start, end),
            "target_time_range": item.get("target_time_range", ""),
            "caption_text": item.get("caption_text", ""),
            "source_video_id": os.path.splitext(os.path.basename(source_path))[0],
        })
    return kept


def _asset_bank_by_slot(strategy: dict) -> dict:
    by_slot = {}
    for asset in strategy.get("user_asset_bank", []) or []:
        plan = asset.get("concrete_editing_plan") or {}
        slot_id = plan.get("slot_id") or asset.get("slot_id")
        if slot_id:
            by_slot[slot_id] = asset
    return by_slot


def _load_context(strategy_abs: str) -> dict:
    """读取 export 时落盘的 connector_context.json（含镜头结构 + 全量素材片段池）。"""
    path = os.path.join(os.path.dirname(strategy_abs), "connector_context.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream) or {}
    except (OSError, json.JSONDecodeError):
        return {}



# --------------------------------------------------------------------------- #
# 语音识别（复用 Agent 的本机离线 ASRTool，按素材去重 + 缓存）
# --------------------------------------------------------------------------- #
def _asr_for_source(source_path: str) -> dict:
    """共享 ASR 缓存优先——理解阶段跑过的素材直接复用，不再重复转写。"""
    cached = asr_cache.get(source_path)
    if cached:
        return {
            "text": cached.get("text", ""),
            "segments": cached.get("segments", []),
            "duration": float(cached.get("duration_seconds") or 0.0),
            "error": "",
        }
    result = _ASR.transcribe(source_path)
    payload = {
        "text": result.get("text", ""),
        "segments": [
            {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": s.get("text", "")}
            for s in (result.get("segments") or [])
            if isinstance(s, dict)
        ],
        "duration": float(result.get("duration_seconds") or 0.0),
        "error": result.get("error", ""),
    }
    if not payload["error"]:
        asr_cache.set(source_path, {
            "text": payload["text"], "segments": payload["segments"],
            "duration_seconds": payload["duration"],
        })
    return payload

def build_asr_path(sources: list, out_dir: str) -> str:
    """对候选池里出现的所有去重素材跑 ASR，写出 Split 需要的 all_source_asr.json。

    ``sources``: ``[{"source_video_id","source_path"}, ...]``。字幕/口播跟随所选片段的
    真实 ASR，因此候选池里任何可能被选中的素材都要有 ASR 记录。
    """
    records = []
    seen = set()
    for s in sources or []:
        source_path = _abspath(s.get("source_path", "") if isinstance(s, dict) else "")
        if not source_path or source_path in seen or not os.path.isfile(source_path):
            continue
        seen.add(source_path)
        svid = (s.get("source_video_id") if isinstance(s, dict) else "") or \
            os.path.splitext(os.path.basename(source_path))[0]
        asr = _asr_for_source(source_path)
        records.append({
            "source_video_id": svid,
            "source_path": source_path,
            "duration": asr.get("duration", 0.0),
            "audio_path": source_path,  # Split 用 ffmpeg 直接从视频抽 TTS prompt 音频
            "asr_text": asr.get("text", ""),
            "asr_items": asr.get("segments", []),
            "asr_segments": asr.get("segments", []),
        })
    path = os.path.join(out_dir, "all_source_asr.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
# 合成 SCORES_PATH（all_scores.json）——方案 A：只用编排 Agent 的选择。
# 每个 slot：首选 = 复刻分镜展示的那个片段（最高分 1.0，Split 非冲突时直接用它）；
# 备选 = 可行性验证/编排继承下来的 alternate_assets（0.9 递减，仅当首选与别的镜头
# 重叠时供 Split 换片）。不再让连接器自己重新检索全量池，保证成片=复刻分镜。
# --------------------------------------------------------------------------- #
def build_scores_path(kept: list, bank_by_slot: dict, context: dict, out_dir: str, work_slug: str):
    results_by_source = {}
    pool_sources = {}

    def add_candidate(svid, source_path, source_time_range, slot_id, slot_role, cand_id, score, extra=None):
        source_path_abs = _abspath(source_path)
        if not source_path_abs or not os.path.isfile(source_path_abs) or not source_time_range:
            return
        key = svid or source_path_abs
        result = results_by_source.setdefault(key, {
            "source_video_id": svid, "source_path": source_path_abs,
            "candidate_segments": [], "slot_fit_scores": []})
        if not any(cs["candidate_id"] == cand_id for cs in result["candidate_segments"]):
            seg = {"candidate_id": cand_id, "source_video_id": svid,
                   "source_path": source_path_abs, "source_time_range": source_time_range}
            seg.update(extra or {})
            result["candidate_segments"].append(seg)
        result["slot_fit_scores"].append({
            "slot_id": slot_id, "slot_role": slot_role, "candidate_id": cand_id,
            "source_video_id": svid, "source_path": source_path_abs,
            "source_time_range": source_time_range, "score": round(float(score), 4)})
        pool_sources[source_path_abs] = svid

    # 首选：编排 Agent 为每个 slot 选定的片段（= 复刻分镜展示的），给最高分 1.0
    for item in kept:
        asset = bank_by_slot.get(item["slot_id"]) or {}
        cand_id = asset.get("asset_id") or f"{item['source_video_id']}::{item['slot_id']}"
        add_candidate(item["source_video_id"], item["source_path"], item["source_time_range"],
                      item["slot_id"], "", cand_id, 1.0,
                      {"visual_description": asset.get("visual_description", ""),
                       "asset_type": asset.get("asset_type", ""),
                       "speech_or_text": asset.get("speech_or_text", "")})

    # 备选：可行性验证/编排为该 slot 继承的 alternate_assets——仅在首选与别的镜头重叠时，
    # Split 会从这里换一个仍契合本镜头的不同片段（避免重复或丢镜头）。
    for item in kept:
        asset = bank_by_slot.get(item["slot_id"]) or {}
        for rank, alt in enumerate(asset.get("alternate_assets", []) or []):
            if not isinstance(alt, dict):
                continue
            add_candidate(alt.get("source_video_id", ""), alt.get("source_path", ""),
                          alt.get("source_time_range", ""), item["slot_id"], "",
                          alt.get("asset_id") or f"{alt.get('source_video_id', '')}::alt{rank}",
                          max(0.1, 0.9 - rank * 0.05),
                          {"visual_description": alt.get("summary", "")})

    payload = {"results": list(results_by_source.values())}
    path = os.path.join(out_dir, "all_scores.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    asr_sources = [{"source_video_id": v, "source_path": p} for p, v in pool_sources.items()]
    return path, asr_sources



# --------------------------------------------------------------------------- #
# 合成 BASE_PLAN（simple_concat_plan.json）：trim 每镜 + 末尾 hard_cut
# 结构对齐 Split 的 write_simple_concat_plan 输出，Split 后链路原生消费。
# --------------------------------------------------------------------------- #
def build_base_plan(kept: list, bank_by_slot: dict, out_dir: str) -> str:
    clip_assets = []
    tool_calls = []
    input_refs = []
    current = 0.0
    for index, item in enumerate(kept, 1):
        slot_id = item["slot_id"]
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(slot_id))
        clip_id = "clip_{:02d}_{}".format(index, safe)
        out_ref = "out_{:02d}_{}".format(index, safe)
        t_start, t_end = _parse_range(item["target_time_range"])
        s_start, s_end = _parse_range(item["source_time_range"])
        duration = round((t_end - t_start) if t_end > t_start else (s_end - s_start), 2)
        duration = max(0.2, duration)
        asset = bank_by_slot.get(slot_id) or {}
        clip_assets.append({
            "clip_id": clip_id,
            "slot_id": slot_id,
            "source_video_id": item["source_video_id"],
            "source_path": item["source_path"],
            "source_time_range": item["source_time_range"],
            "target_time_range": _fmt_range(current, current + duration),
            "target_duration": duration,
            "source_asset_type": "video",
            "visual_description": asset.get("visual_description", ""),
            "speech_or_text": asset.get("speech_or_text", ""),
            "caption_text": item.get("caption_text", ""),
            "notes": "agent_connector_concat",
        })
        tool_calls.append({
            "call_id": "trim_{:02d}_{}".format(index, safe),
            "slot_id": slot_id,
            "stage": "trim",
            "tool_name": "no_transition",
            "input_ref": clip_id,
            "input_refs": [],
            "output_ref": out_ref,
            "duration": duration,
            "params": {},
            "reason": "connector concat: trim selected source range only",
        })
        input_refs.append(out_ref)
        current += duration
    tool_calls.append({
        "call_id": "tc_final_concat",
        "slot_id": "GLOBAL",
        "stage": "final_concat",
        "tool_name": "hard_cut",
        "input_ref": "",
        "input_refs": input_refs,
        "output_ref": "final_video_output",
        "duration": 0.0,
        "params": {},
        "reason": "connector concat: hard cut selected clips in slot order",
    })
    payload = {
        "tool_plan_version": "simple_concat_v1",
        "source_strategy_summary": "Connector concat plan generated from Agent editing_timeline.",
        "global_settings": {"video_size": "720x1280", "fps": 30, "output_dir": os.path.join(SPLIT_ROOT, "outputs")},
        "clip_assets": clip_assets,
        "tool_calls": tool_calls,
        "final_output_ref": "final_video_output",
        "unsupported_effects": [],
        "validation_notes": ["Connector concat plan: only trim/no_transition plus final hard_cut are used."],
    }
    path = os.path.join(out_dir, "simple_concat_plan.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
# 合成 BASE_STRATEGY：只保留可剪辑镜头，source_path 绝对化，target 连续化
# --------------------------------------------------------------------------- #
def build_base_strategy(strategy: dict, kept: list, out_dir: str) -> str:
    kept_slots = {item["slot_id"] for item in kept}
    cleaned = copy.deepcopy(strategy)

    timeline = []
    current = 0.0
    kept_by_slot = {item["slot_id"]: item for item in kept}
    for slot_id in [item["slot_id"] for item in kept]:
        src = kept_by_slot[slot_id]
        s_start, s_end = _parse_range(src["source_time_range"])
        t_start, t_end = _parse_range(src["target_time_range"])
        duration = max(0.2, round((t_end - t_start) if t_end > t_start else (s_end - s_start), 2))
        timeline.append({
            "slot_id": slot_id,
            "target_time_range": _fmt_range(current, current + duration),
            "source_video_id": src["source_video_id"],
            "source_path": src["source_path"],
            "source_time_range": src["source_time_range"],
            "caption_text": src.get("caption_text", ""),
            "transition_to_next": "hard_cut",
            "action": "use_user_asset",
        })
        current += duration
    cleaned["editing_timeline"] = timeline

    bank = []
    for asset in cleaned.get("user_asset_bank", []) or []:
        plan = asset.get("concrete_editing_plan") or {}
        slot_id = plan.get("slot_id") or asset.get("slot_id")
        if slot_id not in kept_slots:
            continue
        if asset.get("source_path"):
            asset["source_path"] = _abspath(asset["source_path"])
        if plan.get("source_path"):
            plan["source_path"] = _abspath(plan["source_path"])
        bank.append(asset)
    cleaned["user_asset_bank"] = bank
    cleaned["slot_matching"] = [
        entry for entry in cleaned.get("slot_matching", []) or []
        if entry.get("slot_id") in kept_slots
    ]
    path = os.path.join(out_dir, "selected_editing_strategy.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(cleaned, stream, ensure_ascii=False, indent=2)
    return path


def _target_product_name(strategy: dict) -> str:
    dna_path = (strategy.get("metadata") or {}).get("dna_path", "")
    if dna_path and os.path.isfile(dna_path):
        try:
            with open(dna_path, "r", encoding="utf-8") as stream:
                template = (json.load(stream) or {}).get("viral_dna_template", {})
            name = str(template.get("target_product_name") or "").strip()
            if name:
                return name
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return str((strategy.get("metadata") or {}).get("scheme") or "").strip() or "目标商品"


# --------------------------------------------------------------------------- #
# whq 结构级复刻的出片：直接用 whq 自己的出片链路，不走 Split 主链路
# --------------------------------------------------------------------------- #
def _run_whq_edit(strategy_abs: str, strategy: dict, *, enable_bgm: bool, bgm_path: str,
                  rid: str, step):
    """「workflow 剪辑」在 whq_clone 编排产物上的出片实现（生成器，事件与 Split 分支一致）。

    Split 主链路的 ``rebuild_asr_edit`` 会把每镜窗口扩到 ASR 整句边界、并按扩窗长度
    retime 整条时间线，顶开 whq 锁好的槽时长（实测参考 28s 被撑到 65s），扩窗还会把别的
    段的口播吞进来（听起来同一句话说两遍）。whq 编排阶段已经做完选片/对窗/原声决策/文案，
    所以这里直接调 whq 的 render（硬剪 → 配音 → 字幕 → 语速贴参考），产物时长 = 参考时长。
    """
    whq_dir = os.path.join(AGENT_ROOT, "src", "editing", "whq_clone")
    if whq_dir not in sys.path:
        sys.path.insert(0, whq_dir)
    import runner as whq_runner  # noqa: PLC0415

    meta = strategy.get("metadata") or {}
    product_name = str(meta.get("product_name") or "").strip() or _target_product_name(strategy)
    ref_dur = ((strategy.get("whq_meta") or {}).get("duration_estimate")) or 0
    yield step("whq", "whq 出片", "复用 whq 编排（硬剪 + 原声/克隆配音 + 字幕 + 语速贴参考），"
                                 "不走 Split 重建（它会把槽时长顶开、口播重复）", state="running")

    final_dir = os.path.join(AGENT_ROOT, "uploads", "final")
    os.makedirs(final_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slug((meta.get("project_name") or "whq_clone"))
    final_name = f"{slug}_{ts}_final.mp4"
    final_abs = os.path.join(final_dir, final_name)

    resolved_bgm = bgm_path or _load_split_env().get("BGM_PATH", "") or DEFAULT_BGM
    migrate_bgm = bool(enable_bgm and resolved_bgm and os.path.isfile(resolved_bgm))
    try:
        info = whq_runner.edit_from_strategy(
            strategy_abs, final_abs, product_name=product_name, migrate_bgm=migrate_bgm)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] whq edit failed: %s", rid, exc)
        yield {"type": "error", "message": f"whq 出片失败：{exc!s}"}
        return
    final_out = info.get("final") or ""
    if final_out and os.path.abspath(final_out) != final_abs and os.path.isfile(final_out):
        try:
            shutil.copy2(final_out, final_abs)
        except OSError:
            final_abs = final_out
    if not os.path.isfile(final_abs):
        yield {"type": "error", "message": "whq 出片结束但未找到成片文件。"}
        return

    final_script = []
    try:
        with open(info["plan_path"], "r", encoding="utf-8") as stream:
            plan = json.load(stream)
        decisions = plan.get("decisions") or {}
        for seg in plan.get("segments") or []:
            sid = seg.get("slot_id", "")
            dec = decisions.get(sid) or {}
            cand = seg.get("best_candidate") or {}
            final_script.append({
                "slot_id": sid,
                "target_time_range": "",
                "source": os.path.basename(str(cand.get("source_path") or "")),
                "source_time_range": "{}-{}".format(cand.get("start"), cand.get("end")),
                "caption_text": dec.get("window_text", "") or seg.get("beat_desc", ""),
                "caption_source": "user_original" if dec.get("voice_source") == "original" else "whq_script",
                "source_asr_text": dec.get("window_text", ""),
            })
    except (OSError, KeyError, json.JSONDecodeError, ValueError):
        pass

    yield step("whq", "whq 出片", "成片已生成", state="done",
               observation=(f"{info.get('n_segments')} 段，其中 {info.get('n_original_voice')} 段保留用户原声\n"
                            f"参考总时长 {ref_dur}s（成片按槽时长锁定，不做整体加速缩短）\n{final_abs}"))
    yield {"type": "edit_done", "video_uri": f"uploads/final/{final_name}",
           "final_path": final_abs, "slots": info.get("n_segments", 0), "asr_hits": 0,
           "product_name": product_name, "final_script": final_script}


# --------------------------------------------------------------------------- #
# 主流程：合成四份文件 → 子进程跑 Split 后链路 → 流式回传
# --------------------------------------------------------------------------- #
def run_edit(strategy_path: str, *, enable_tts: bool = True, enable_bgm: bool = True,
             enable_t2v: bool = False, bgm_path: str = ""):
    """生成器：产出 step / edit_log / edit_done / error 事件。"""
    rid = time.strftime("%H%M%S")

    def step(key, title, thought, state="done", observation=None):
        ev = {"type": "step", "phase": "剪辑合成", "key": f"{key}-{rid}", "state": state,
              "title": title, "thought": thought}
        if observation is not None:
            ev["observation"] = observation
        return ev

    strategy_abs = _abspath(strategy_path)
    if not strategy_abs or not os.path.isfile(strategy_abs):
        yield {"type": "error", "message": f"找不到编排脚本：{strategy_path}"}
        return

    yield step("prep", "读取编排脚本", "解析 selected_editing_strategy，抽取可剪辑镜头", state="running")
    with open(strategy_abs, "r", encoding="utf-8") as stream:
        strategy = json.load(stream)
    # whq 结构级复刻的编排产物 -> 用 whq 自己的出片链路（Split 重建会顶开槽时长 + 口播重复）
    if (strategy.get("metadata") or {}).get("reproduce_mode") == "whq_clone" \
            and os.getenv("WHQ_EDIT_VIA_SPLIT", "0") in ("0", "false", "False"):
        yield from _run_whq_edit(strategy_abs, strategy, enable_bgm=enable_bgm,
                                 bgm_path=bgm_path, rid=rid, step=step)
        return
    kept = _kept_timeline(strategy)
    if not kept:
        yield {"type": "error",
               "message": "没有可直接剪辑的镜头：所有镜头都是补拍/AIGC 生成。请开启 T2V 或补充用户素材后重试。"}
        return
    bank_by_slot = _asset_bank_by_slot(strategy)
    context = _load_context(strategy_abs)
    product_name = _target_product_name(strategy)
    project_name = (strategy.get("metadata") or {}).get("project_name") or "agent_edit"
    slug = _slug(project_name)
    pool_note = f"，素材候选池 {len(context.get('segments') or [])} 段" if context.get("segments") else "（无候选池上下文，降级为交叉授粉）"
    yield step("prep", "读取编排脚本",
               f"命中 {len(kept)} 个可剪辑镜头，目标商品「{product_name}」{pool_note}",
               observation="\n".join(f"{k['slot_id']} {k['source_time_range']} {os.path.basename(k['source_path'])}" for k in kept))

    work_dir = os.path.join(os.path.dirname(strategy_abs), "connector")
    os.makedirs(work_dir, exist_ok=True)

    # 用编排 Agent 每镜的选片（复刻分镜展示的）作首选 + 继承的 alternates 作备选，拿到涉及素材再跑 ASR
    yield step("synth", "合成 Split 契约", "以编排 Agent 每镜选片为首选 + 继承备选，生成 SCORES / simple_concat_plan / strategy", state="running")
    scores_path, asr_sources = build_scores_path(kept, bank_by_slot, context, work_dir, slug)
    base_plan_path = build_base_plan(kept, bank_by_slot, work_dir)
    base_strategy_path = build_base_strategy(strategy, kept, work_dir)
    yield step("synth", "合成 Split 契约",
               f"{len(kept)} 个镜头，每镜首选=复刻分镜片段、备选来自继承候选，涉及 {len(asr_sources)} 个素材",
               observation=f"BASE_PLAN: {base_plan_path}\nBASE_STRATEGY: {base_strategy_path}\nSCORES_PATH: {scores_path}")

    # 预扫共享 asr 缓存，让前端看到「多少命中 / 多少需要新跑」，而不是黑盒一整批
    cache_hits = sum(1 for s in asr_sources if asr_cache.get(s.get("source_path", "")) is not None)
    cache_misses = len(asr_sources) - cache_hits
    if cache_misses == 0:
        yield step("asr", "语音识别（全部命中缓存）",
                   f"候选池 {len(asr_sources)} 个素材全部命中共享 ASR 缓存，无需重跑",
                   state="running")
    else:
        yield step("asr", "语音识别",
                   f"候选池 {len(asr_sources)} 个素材：{cache_hits} 个命中共享缓存跳过、{cache_misses} 个需要重新转写",
                   state="running")
    asr_path = build_asr_path(asr_sources, work_dir)
    with open(asr_path, "r", encoding="utf-8") as stream:
        asr_records = json.load(stream)
    asr_hits = sum(1 for r in asr_records if r.get("asr_segments"))
    yield step("asr", "语音识别",
               f"{len(asr_records)} 个素材，其中 {asr_hits} 个有有效口播（{cache_hits} 命中缓存 + {cache_misses} 新转写）",
               observation="\n".join(f"{os.path.basename(r['source_path'])}: {r.get('asr_text','')[:60]}" for r in asr_records))

    # 最终成片直接落到 uploads/final，走已有 /uploads 静态路由，前端可直接播放
    final_dir = os.path.join(AGENT_ROOT, "uploads", "final")
    os.makedirs(final_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_name = f"{slug}_{ts}_final.mp4"
    final_abs = os.path.join(final_dir, final_name)
    video_uri = f"uploads/final/{final_name}"

    split_env = _load_split_env()

    def pick(key, default=""):
        return os.environ.get(key) or split_env.get(key) or default

    main_python = pick("MAIN_PYTHON", "python")
    pythonpath = os.pathsep.join([
        os.path.join(SPLIT_ROOT, "common"),
        os.path.join(SPLIT_ROOT, "common", "vendor", "Viral_Video"),
    ])
    if os.environ.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + os.environ["PYTHONPATH"]

    resolved_bgm = bgm_path or split_env.get("BGM_PATH", "") or DEFAULT_BGM
    if enable_bgm and not os.path.isfile(resolved_bgm):
        resolved_bgm = ""  # 缺 BGM 时 Split 自动跳过混音，不报错

    env = dict(os.environ)
    env.update({
        "PIPELINE_ROOT": SPLIT_ROOT,
        "PYTHONPATH": pythonpath,
        "VIRAL_VIDEO_SOURCE_ROOT": os.path.join(SPLIT_ROOT, "common", "vendor", "Viral_Video"),
        "OUTPUT_DIR": os.path.join(SPLIT_ROOT, "outputs"),
        "PROJECT_SLUG": slug,
        "PLAN_BASENAME": slug,
        "OUTPUT_NAME_PREFIX": slug,
        "TARGET_PRODUCT_NAME": product_name,
        "BASE_PLAN": base_plan_path,
        "BASE_STRATEGY": base_strategy_path,
        "SCORES_PATH": scores_path,
        "ASR_PATH": asr_path,
        "FINAL_OUTPUT": final_abs,
        # —— 对齐 Split 原生防重复：保留其不重叠/去尾/slot 选择默认（不再关闭），
        #    配合多候选池，Split 会给每个镜头分到不同片段，成片不再复读 ——
        # —— 生成开关 ——
        "ENABLE_T2V": "1" if enable_t2v else "0",
        "DISABLE_T2V": "0" if enable_t2v else "1",
        "ENABLE_TTS": "1" if enable_tts else "0",
        "ENABLE_BGM": "1" if (enable_bgm and resolved_bgm) else "0",
        "BGM_PATH": resolved_bgm if (enable_bgm and resolved_bgm) else "",
        "USE_WENCHAIN_OPENAI": pick("USE_WENCHAIN_OPENAI", "1"),
        # —— 解释器 / 模型（来自 Split .env）——
        "ASR_PYTHON": pick("ASR_PYTHON", main_python),
        "TTS_PYTHON": pick("TTS_PYTHON"),
        "TTS_SCRIPT": pick("TTS_SCRIPT"),
        "TTS_REPO": pick("TTS_REPO"),
        "TTS_MODEL_DIR": pick("TTS_MODEL_DIR"),
        "QWEN3_ASR_MODEL": pick("QWEN3_ASR_MODEL"),
        "QWEN3_FORCED_ALIGNER": pick("QWEN3_FORCED_ALIGNER"),
    })
    env = {k: v for k, v in env.items() if v is not None and str(v) != ""}

    yield step("run", "执行 Split 后链路",
               f"rebuild → 剪辑{'（含 T2V）' if enable_t2v else ''} → {'TTS → ' if enable_tts else ''}字幕 → {'BGM' if (enable_bgm and resolved_bgm) else '不加 BGM'}",
               state="running")
    _log.info("[%s] launch main chain slug=%s python=%s", rid, slug, main_python)

    proc = subprocess.Popen(
        [main_python, MAIN_CHAIN],
        cwd=SPLIT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    final_from_log = ""
    tail = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        for marker in ("FINAL_OUTPUT", "LATEST_OUTPUT"):
            if line.startswith(marker):
                candidate = line[len(marker):].strip()
                if candidate:
                    final_from_log = candidate.split()[0]
        yield {"type": "edit_log", "line": line}
    proc.wait()

    if proc.returncode != 0:
        yield step("run", "执行 Split 后链路", "剪辑链路失败", state="done",
                   observation="\n".join(tail[-20:]))
        yield {"type": "error", "message": f"Split 后链路退出码 {proc.returncode}，详见日志末尾。"}
        return

    if not os.path.isfile(final_abs) and final_from_log and os.path.isfile(final_from_log):
        try:
            shutil.copy2(final_from_log, final_abs)
        except OSError:
            final_abs = final_from_log
            video_uri = ""

    if not os.path.isfile(final_abs):
        yield {"type": "error", "message": "剪辑链路结束但未找到最终成片文件。"}
        return

    # 读回 Split 重建后的时间线：每镜实际选中的用户片段 + 跟随用户口播的字幕
    # （区别于「复刻方案」里参考视频的文案），作为独立「成片脚本」tab 展示
    final_script = []
    mode = "t2v_on" if enable_t2v else "t2v_off"
    rebuilt = os.path.join(SPLIT_ROOT, "outputs", slug, f"asr_rebuild_{mode}", f"{slug}_asr_strategy.json")
    if os.path.isfile(rebuilt):
        try:
            with open(rebuilt, "r", encoding="utf-8") as stream:
                rebuilt_strategy = json.load(stream)
            for it in rebuilt_strategy.get("editing_timeline", []) or []:
                final_script.append({
                    "slot_id": it.get("slot_id", ""),
                    "target_time_range": it.get("target_time_range", ""),
                    "source": os.path.basename(str(it.get("source_path", "") or "")),
                    "source_time_range": it.get("source_time_range", ""),
                    "caption_text": it.get("caption_text", ""),
                    "caption_source": it.get("caption_source", ""),
                    "source_asr_text": it.get("source_asr_text", ""),
                })
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    yield step("run", "执行 Split 后链路", "成片已生成", state="done", observation=final_abs)
    yield {"type": "edit_done", "video_uri": video_uri, "final_path": final_abs,
           "slots": len(kept), "asr_hits": asr_hits, "product_name": product_name,
           "final_script": final_script}
