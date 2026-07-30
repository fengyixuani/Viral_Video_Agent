"""纯 Agent 剪辑链路：剪辑 Agent + 审片 Agent + 重剪循环（agent_cut 分支）。

流程（最多 MAX_LOOPS 轮）：
  1. 剪辑 Agent（ReAct）：不再局限于每个 slot 的预分配候选，而是用召回工具从**全量素材池**
     自由检索、按需 VLM 验证、再放入；放入时后台检测重叠（同片段/同口播被多槽占用），
     重叠触发审核 Agent 裁决归属，输家 slot 同轮重新召回另选。字幕用所放片段自己的口播 speech。
  2. 剪辑器（editor.build_video，纯 ffmpeg）：执行 placements → 样片 + 逐操作记录 ops。
  3. 审片 Agent（默认 qwen 视觉 / 可选 Gemini 画面+声音）：直接看样片 + 对照 DNA + 看本轮 ops
     + 历轮简要，判定是否符合预期；不符合就给出重剪命令。
  4. 通过则输出；否则带反馈回到 1 重剪。达到轮次上限仍不达标 → 输出历轮最佳样片并标注
     「素材受限」。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time

import as_core
import asr_cache
import obs
import contracts
from skills import get as get_skill
from editing import editor
from editing import gemini_review
from editing import arbiter
from editing import tts as edit_tts
from editing import bgm_reuse
from editing import beats as beats_tool
from editing import tools as edit_tools
from editing import aigc as aigc_agent
from editing.tools import EditToolbox, ranges_overlap, normalize_speech, render_tools_spec

_log = obs.get_logger("agent_edit_loop")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _save_edit_trace(task_id, strategy_abs, *, video_uri, score, verdict, history, review_model):
    """把 Agent 剪辑每轮中间输出（剪辑note/ops/审片评语/评分/问题）落盘，供素材调试页「Agent剪辑」tab 查看。"""
    if not task_id:
        return
    import time
    from datetime import datetime
    trace = {
        "task_id": task_id, "strategy_path": strategy_abs,
        "created_at": time.time(),
        "created_at_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_uri": video_uri, "final_score": score, "verdict": verdict,
        "review_model": review_model, "loops": len(history or []),
        "history": history or [],
    }
    try:
        d = os.path.join(AGENT_ROOT, "uploads", "debug", "edit")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{task_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(trace, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        _log.warning("[%s] edit trace save failed: %s", task_id, exc)

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


def _load_context(strategy_abs: str, strategy: dict) -> dict:
    """加载全量素材池 + DNA 分镜结构。

    优先读 strategy 同目录的 connector_context.json（含 shots 结构 + 全部 segments 素材池）；
    缺失时退化用 strategy.user_asset_bank 里的 primary+alternates 拼一个小池，保证不断链。
    """
    ctx_path = os.path.join(os.path.dirname(strategy_abs), "connector_context.json")
    if os.path.isfile(ctx_path):
        try:
            with open(ctx_path, "r", encoding="utf-8") as fh:
                ctx = json.load(fh)
            if isinstance(ctx, dict) and ctx.get("segments"):
                return ctx
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("connector_context 读取失败(%s)，退化用 strategy 素材池", exc)
    # 退化：从 user_asset_bank 收集片段（primary + alternates）
    segs, seen = [], set()
    for asset in strategy.get("user_asset_bank", []) or []:
        for src in [asset] + list(asset.get("alternate_assets", []) or []):
            gid = src.get("asset_id") or src.get("global_asset_id") or ""
            path = src.get("source_path", "")
            key = gid or f"{path}::{src.get('source_time_range','')}"
            if not path or key in seen:
                continue
            seen.add(key)
            segs.append({
                "global_asset_id": gid or key,
                "source_path": path,
                "source_time_range": src.get("source_time_range", ""),
                "one_sentence_summary": src.get("summary", "") or src.get("visual_description", ""),
                "visual_description": src.get("visual_description", ""),
                "speech_or_text": src.get("speech_or_text", ""),
                "keywords": src.get("strengths", []) or [],
            })
    shots = [{"slot_id": (a.get("concrete_editing_plan") or {}).get("slot_id") or a.get("slot_id"),
              "role": (a.get("suitable_roles") or [""])[0] if isinstance(a.get("suitable_roles"), list) else a.get("suitable_roles", ""),
              "want": a.get("visual_description", "")}
             for a in strategy.get("user_asset_bank", []) or []]
    return {"shots": shots, "segments": segs}


def _load_inputs(strategy: dict, context: dict) -> dict:
    """抽出 DNA 概要 + 待填 slot（slot_id/角色/意图/目标时长/breakdown）。

    slot 结构来自 connector_context 的 shots（DNA 角色/意图更完整），按 slot_id 对齐；
    候选片段不再预分配——剪辑 Agent 会用召回工具从全量 segments 池自己找。
    """
    shots_meta = {s.get("slot_id"): s for s in (context.get("shots") or []) if isinstance(s, dict)}
    # 复刻方案里编排 Agent 已为每个 slot 选好的片段（primary）——直接复用，避免剪辑 Agent 重复召回
    bank_by_slot = {}
    for a in strategy.get("user_asset_bank", []) or []:
        bsid = (a.get("concrete_editing_plan") or {}).get("slot_id") or a.get("slot_id")
        if bsid:
            bank_by_slot[bsid] = a
    # editing_timeline 里每个 slot 的字幕/目标时长（供 AIGC 缺失镜头用）
    tl_by_slot = {it.get("slot_id"): it for it in (strategy.get("editing_timeline", []) or [])
                  if isinstance(it, dict) and it.get("slot_id")}
    slots = []
    presets = {}
    for item in strategy.get("editing_timeline", []) or []:
        if item.get("action") != "use_user_asset":
            continue
        sid = item.get("slot_id", "")
        meta = shots_meta.get(sid, {})
        t0, t1 = _parse_range(item.get("target_time_range", ""))
        target_dur = round(t1 - t0, 2) if t1 > t0 else float(meta.get("duration") or 3.0)
        slots.append({
            "slot_id": sid,
            "role": meta.get("role", "") or item.get("role", ""),
            "want": meta.get("want", "") or item.get("caption_text", ""),
            "breakdown": meta.get("breakdown", []) or [],
            "target_duration": target_dur,
            # whq_clone 链路：编排阶段定好的原声/克隆基线 + 参考语速（其他链路为空）
            "whq_voice": meta.get("whq_voice") or item.get("whq_voice") or {},
        })
        # 复刻方案预填：source 来自 editing_timeline，口播来自 user_asset_bank
        asset = bank_by_slot.get(sid) or {}
        sp = item.get("source_path", "") or asset.get("source_path", "")
        srng = item.get("source_time_range", "") or asset.get("source_time_range", "")
        if sp and _abspath(sp) and os.path.isfile(_abspath(sp)):
            speech = asset.get("speech_or_text", "")
            presets[sid] = {
                "global_asset_id": (asset.get("asset_id", "") or item.get("global_asset_id", "")),
                "source_path": sp, "source_time_range": srng,
                "target_duration": target_dur,
                "caption": speech, "speech": speech,
                "burn_caption": bool(speech), "speed": 1.0,
                # 原声段：source_time_range 是 whq「说完整句」的对窗结果，成片必须播完整个
                # 窗口，不能被节拍目标时长截短（否则最后一句话说一半就切）
                "voice_source": (item.get("whq_voice") or {}).get("voice_source"),
            }
    meta = strategy.get("metadata", {}) or {}
    # 缺失镜头（需 AIGC 生成）：从 missing_assets 收集，补上 breakdown（爆款分镜维度）与字幕
    aigc_slots = []
    for miss in strategy.get("missing_assets", []) or []:
        sid = miss.get("slot_id", "")
        if not sid or sid in presets:
            continue
        smeta = shots_meta.get(sid, {})
        tl = tl_by_slot.get(sid, {})
        t0, t1 = _parse_range(tl.get("target_time_range", ""))
        target_dur = round(t1 - t0, 2) if t1 > t0 else float(miss.get("target_duration") or smeta.get("duration") or 3.0)
        aigc_slots.append({
            "slot_id": sid,
            "role": miss.get("role", "") or smeta.get("role", ""),
            "want": miss.get("goal", "") or smeta.get("want", ""),
            "breakdown": smeta.get("breakdown", []) or [],
            "caption": tl.get("caption_text", ""),
            "generation_prompt": miss.get("generation_prompt", ""),
            "target_duration": target_dur,
        })
    return {
        "product_name": (meta.get("product_name", "") or meta.get("scheme", "")
                         or meta.get("project_name", "") or "目标商品"),
        "narrative_structure": strategy.get("overall_editing_strategy", {}) if isinstance(strategy.get("overall_editing_strategy"), dict) else {},
        "slots": slots,
        "presets": presets,
        "aigc_slots": aigc_slots,
    }


_EDIT_FALLBACK = (
    "你是短视频剪辑 Agent，用 ReAct 方式工作：每次只输出**一个** JSON 动作，我执行后把结果反馈给你，"
    "你再决定下一步，直到所有 slot 都放好片、输出 finish。\n"
    "目标：为爆款 DNA 的每个 slot 从**全量用户素材池**里挑最贴合其角色/意图的片段，剪出一条竖屏带货成片。\n"
    "你**不再被限制在某个预分配候选池**——用召回工具自己从全池找素材、（拿不准时）验证、再放入。\n"
    "可用工具（每次只输出其中一个动作的 JSON）：\n{{tools}}\n"
    "硬规则：\n"
    "- **成片保留素材原声，字幕必须用你所放那个片段自己的 speech（口播原话）**，别用参考爆款字幕、别照抄；"
    "speech 为空则 burn_caption=false。caption 可轻微精简断句但语义须与 speech 一致。\n"
    "- 一个片段/同一句口播只能用于一个 slot，不要重复占用。\n"
    "- target_duration 参考 slot 目标时长，可按节奏微调；必要时 speed 0.8~1.6。\n"
    "- 有 review_feedback（上一轮审片问题/命令）时，必须针对性地重新召回替换、改 trim/字幕/时长。\n"
    "- 优先把关键转化节点（痛点/产品登场等靠前 slot）配到最贴合的素材。"
)

_REVIEW_FALLBACK = (
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


def _skill_prompt(skill_id: str, fallback: str) -> str:
    """读 skills_defs 里该 Agent 的 SKILL.md 作为 system prompt；缺失时用内置兜底。"""
    sk = get_skill(skill_id)
    return sk.prompt_hint if (sk and sk.prompt_hint) else fallback


def edit_system_prompt() -> str:
    """剪辑 Agent 的 system prompt = SKILL.md 角色/约束 + 工具注册表渲染出的工具清单。"""
    return _skill_prompt("agent_edit_editor", _EDIT_FALLBACK).replace("{{tools}}", render_tools_spec())


def review_prompt() -> str:
    return _skill_prompt("agent_edit_reviewer", _REVIEW_FALLBACK)


async def _run_llm(system: str, user: str, *, media=None, vision=False, tag: str = ""):
    """跑一次 LLM，捕获思考(reasoning)+输出(content)。传 tag 时把两者都写进 log，方便事后查原因。"""
    content = ""
    reasoning_parts = []
    async for item in as_core.stream(system, user, vision=vision, media=media):
        if item.get("reasoning"):
            reasoning_parts.append(str(item["reasoning"]))
        if "content" in item:
            content = item["content"]
    reasoning = "".join(reasoning_parts).strip()
    if tag:
        if reasoning:
            _log.info("%s reasoning: %s", tag, reasoning[:2000])
        _log.info("%s output: %s", tag, (content or "").strip()[:4000])
    return as_core.parse_json(content) if content.strip() else {}


def _extend_range_for_audio(source_path, rng, need_dur):
    """把 [s,e] 往后延到至少 need_dur 秒（不超过源片总长）。延不动就原样返回。"""
    s, e = _parse_range(rng)
    if e <= s or need_dur <= (e - s) + 0.05:
        return rng
    total = _video_duration(_abspath(source_path))
    if total <= 0:
        return rng
    new_e = min(total, s + need_dur)
    if new_e - s <= (e - s) + 0.05:
        return rng
    return f"{s:.2f}-{new_e:.2f}"


def _video_duration(path):
    """源片时长（秒）；取不到返回 0。用 ffmpeg 读，本机无 ffprobe。"""
    if not path or not os.path.isfile(path):
        return 0.0
    try:
        import subprocess
        out = subprocess.run([editor._FFMPEG, "-hide_banner", "-i", path],
                             capture_output=True, text=True, timeout=20).stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception as exc:  # noqa: BLE001
        _log.warning("读取源片时长失败 %s: %s", path, str(exc)[:120])
    return 0.0


_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")
_NUM_TOKEN_RE = re.compile(
    r"百分之[零一二三四五六七八九十百千点两0-9]+"
    r"|[0-9]+(?:\.[0-9]+)?%?"
    r"|[一二三四五六七八九十百千两]{1,6}(?:点[一二三四五六七八九十]+)?"
    r"(?:块|元|折|盒|袋|包|条|斤|克|毫升|升|倍|周|天|年|个月|小时)"
)


def _norm_text(s):
    return _PUNCT_RE.sub("", str(s or ""))


def _copy_violations(text, user_corpus, ref_corpus, n=4):
    """文案里「抄了参考爆款、但用户素材里没有」的片段 + 无出处的数字断言。

    机制化拦截 whq 的老坑：DNA 的节拍描述来自另一条爆款，模型很容易把参考商品的
    成分/含量/价格/喝法（如"抹茶奶绿""51.7%膳食纤维""29.9 元"）写进本商品的文案。
    光靠 prompt 约束不住，这里按 n-gram + 数字 token 做客观核验。
    """
    t = _norm_text(text)
    bad = []
    for tok in _NUM_TOKEN_RE.findall(t):
        if tok not in user_corpus:
            bad.append(tok)
    if ref_corpus:
        for i in range(0, max(0, len(t) - n + 1)):
            gram = t[i:i + n]
            if gram in ref_corpus and gram not in user_corpus and gram not in bad:
                bad.append(gram)
    return bad[:6]


def _fact_corpora(context, toolbox):
    """(user_corpus, ref_corpus)：用户素材语料 / 参考爆款语料，供 _copy_violations 核验。

    user = 素材池的画面描述 + 各片段自带原声（本商品真实存在的东西）
    ref  = DNA 各节拍描述（参考爆款那个商品的说法）
    """
    user_parts, ref_parts = [], []
    for seg in (getattr(toolbox, "by_gid", {}) or {}).values():
        user_parts.append(seg.get("visual_description") or seg.get("one_sentence_summary") or "")
        user_parts.append((seg.get("whq_speech") or {}).get("text") or seg.get("speech_or_text") or "")
    for sh in (context.get("shots") or []):
        ref_parts.append(sh.get("want") or "")
        ref_parts.append(sh.get("narrative") or "")
    return _norm_text("".join(user_parts)), _norm_text("".join(ref_parts))


def _placements_to_clips(placements: dict, slots: list, tts_by_slot: dict = None,
                         burn_captions: bool = True) -> list:
    """把每个 slot 的最终 placement 按 slot 顺序转成 editor 需要的 clips。

    若该 slot 用了 tts_clone（克隆配音），则该镜改用 TTS 音频替换原声、字幕=配音文案。
    burn_captions=False（无人声/纯音乐参考）时**全程不烧任何字幕**。
    """
    tts_by_slot = tts_by_slot or {}
    clips = []
    for s in slots:
        p = placements.get(s["slot_id"])
        if not p:
            continue
        tts = tts_by_slot.get(s["slot_id"])
        if tts and tts.get("audio_path"):
            # 克隆配音：字幕=改写文案，音轨=TTS wav，时长以配音为准
            cap = (tts.get("text") or "").strip() if burn_captions else ""
            dur = float(tts.get("duration") or p.get("target_duration") or s.get("target_duration") or 3.0)
            clips.append({
                "slot_id": s["slot_id"],
                "source_path": p["source_path"],
                "global_asset_id": p.get("global_asset_id", ""),
                # 配音比所选片段长时，先从**源片里往后多截**补足画面；源片不够长才让
                # editor 冻结末帧。否则会出现"画面静止好几秒、只有配音在响"。
                "source_time_range": _extend_range_for_audio(p["source_path"],
                                                             p.get("source_time_range", ""), dur),
                "target_duration": dur,
                "caption_text": cap,
                "burn_caption": bool(cap),
                "tts_audio_path": tts["audio_path"],
                "speed": 1.0,
            })
            continue
        caption = (p.get("caption") or "").strip() or (p.get("speech") or "").strip()
        burn = burn_captions and (p.get("burn_caption", True) is not False)
        dur = float(p.get("target_duration") or s.get("target_duration") or 3.0)
        # 保留原声的镜头：source_time_range 是「说完整句」的对窗窗口，必须整段播完。
        # 否则 editor 会按节拍目标时长 min(seg_len, target) 截短，最后一句说一半就切（结尾截断）。
        if p.get("voice_source") == "original" or (s.get("whq_voice") or {}).get("voice_source") == "original":
            a, b = _parse_range(p.get("source_time_range", ""))
            if b > a:
                dur = max(dur, round(b - a, 3))
        clips.append({
            "slot_id": s["slot_id"],
            "source_path": p["source_path"],
            "global_asset_id": p.get("global_asset_id", ""),
            "source_time_range": p.get("source_time_range", ""),
            "target_duration": dur,
            "caption_text": caption if burn else "",
            "burn_caption": burn,
            "speed": float(p.get("speed") or 1.0),
        })
    return clips


def _in_window_speech(toolbox, placement):
    """该镜窗口内**实际会听到**的原声文本（中心落窗的逐字 ASR 拼起来）。"""
    seg = (getattr(toolbox, "by_gid", {}) or {}).get((placement or {}).get("global_asset_id")) or {}
    s, e = _parse_range((placement or {}).get("source_time_range", ""))
    if e <= s:
        return ""
    out = ""
    for it in ((seg.get("whq_speech") or {}).get("asr_items") or []):
        try:
            mid = (float(it.get("start")) + float(it.get("end"))) / 2.0
        except (TypeError, ValueError):
            continue
        if s <= mid <= e:
            out += str(it.get("text") or "")
    return _norm_text(out)


def _mute_slots(placements, slots, tts_by_slot, toolbox):
    """既没有原声、也没有克隆配音的「哑巴段」slot_id 列表。

    带货成片不该出现整段没人声的空档。Agent 偶尔会漏配，这里客观检测后触发补配音。
    """
    out = []
    for s in slots or []:
        sid = s["slot_id"]
        p = (placements or {}).get(sid)
        if not p:
            continue                              # 已被 skip_slot 跳过
        if sid in (tts_by_slot or {}):
            continue                              # 有克隆配音
        if p.get("voice_source") == "original":
            continue                              # 保留了原声
        if len(_in_window_speech(toolbox, p)) >= 3:
            continue                              # 窗口内本来就有人在说话（原声可听）
        out.append(sid)
    return out


def _snap_clips_to_sentences(clips, toolbox, gap=0.35, extend=3.0):
    """保原声的镜头：结束点必须落在自然停顿上，别"话说一半就切"。

    优先**往后延到那句说完**（最多 extend 秒，且不超过源片长度）——素材切片边界经常把
    一句话切两半，往前回收会丢内容；延不到句尾时才退而回收到窗口内最后一个停顿。
    克隆配音镜不动（音轨是 TTS，与素材说话无关）。
    """
    by_gid = getattr(toolbox, "by_gid", {}) or {}
    # 同一源片的逐字 ASR 合并：单个候选下发的 items 只带自身窗口附近的上下文，合起来才是
    # 整条源片的字轨，才能判断"这句到哪里才算说完"。
    by_path = {}
    for seg in by_gid.values():
        items = (seg.get("whq_speech") or {}).get("asr_items") or []
        if not items:
            continue
        bucket = by_path.setdefault(seg.get("source_path", ""), {})
        for it in items:
            try:
                bucket[(round(float(it["start"]), 3), round(float(it["end"]), 3))] = str(it.get("text") or "")
            except (TypeError, ValueError, KeyError):
                continue
    for c in clips or []:
        if c.get("tts_audio_path"):
            continue
        s, e = _parse_range(c.get("source_time_range", ""))
        if e <= s:
            continue
        toks = sorted((a, b, t) for (a, b), t in (by_path.get(c.get("source_path", "")) or {}).items())
        if not toks:
            continue
        # 判定"话被切一半"：结束点落在某个字中间，或紧接结束点之后 gap 内还有字开口
        cut_mid = any(a < e < b for a, b, _ in toks)
        cont = any(e - 0.05 < a < e + gap for a, b, _ in toks)
        if not (cut_mid or cont):
            continue
        # ① 往后延到该句说完：沿着字间隔 < gap 的连续串一直走到停顿处
        limit = e + extend
        total = _video_duration(_abspath(c.get("source_path", "")))
        if total > 0:
            limit = min(limit, total)
        new_e, prev_end = e, None
        for a, b, _t in toks:
            if b <= e:
                prev_end = b
                continue
            if a > new_e + gap or b > limit:
                break
            new_e, prev_end = b, b
        if new_e > e + 0.05:
            new_e = round(min(new_e + 0.12, limit), 3)
            c["source_time_range"] = f"{s:.2f}-{new_e:.2f}"
            c["target_duration"] = max(float(c.get("target_duration") or 0.0), new_e - s)
            continue
        # ② 延不动（源片到头了）→ 回收到窗口内最后一个自然停顿，至少留 0.6s 画面
        inside = [(a, b) for a, b, _ in toks if s <= (a + b) / 2.0 <= e]
        cut = None
        for i in range(len(inside) - 1, 0, -1):
            if inside[i][0] - inside[i - 1][1] >= gap:
                cut = inside[i - 1][1] + 0.12
                break
        if cut is None or cut - s < 0.6:
            continue
        new_e = min(e, round(cut, 3))
        if new_e < e - 0.05:
            c["source_time_range"] = f"{s:.2f}-{new_e:.2f}"
            c["target_duration"] = min(float(c.get("target_duration") or 0.0) or (new_e - s), new_e - s)


def _snap_clips_to_beats(clips: list, beats: list) -> list:
    """卡点剪辑：**只延长、不缩短**——把每镜结束点延到自然结束点之后的下一个鼓点。

    在 BGM 时间轴上累加**实际剪辑后的时长**推进：clip i 从累计位置 T 开始、编排时长 d，
    自然结束点 T+d，取**该点之后（含极近）的下一个鼓点 B** 作实际结束点，实际时长 = B-T ≥ d
    （绝不短于原时长）；再令 T=B 看下一镜。配音镜（TTS）时长以配音为准、不卡点，仅推进时间轴。
    """
    bs = sorted(b for b in (beats or []) if isinstance(b, (int, float)) and b > 0)
    if not bs or not clips:
        return clips
    T = 0.0
    for c in clips:
        d = float(c.get("target_duration") or 3.0)
        if c.get("tts_audio_path"):
            T = round(T + d, 3)
            continue
        natural_end = T + d
        # 只延长不缩短：取自然结束点之后（容差 0.08s 内视为对齐）的第一个鼓点
        nxt = [b for b in bs if b >= natural_end - 0.08]
        B = nxt[0] if nxt else natural_end
        dur = round(max(d, B - T), 3)      # 保证 ≥ 原时长
        c["force_duration"] = dur          # 让 editor 按此时长出片（不足则冻结末帧补足）
        c["target_duration"] = dur
        c["beat_synced"] = True
        T = round(T + dur, 3)
    return clips


def _dedup_clips(clips: list, toolbox, slots: list) -> list:
    """去重：多个镜头用了**同一素材的同一段**（同源+时间相交）会造成成片重复镜头。
    对重复的后续镜头，优先从全池召回一个**没用过的素材**替换；召回不到再退化为同源不同片段。

    复刻方案（编排）在素材少或 LLM 未去重时会把同一段分给多个 slot；而首轮 seeded 直接采用
    复刻方案、跳过了 ReAct 里的重叠仲裁，所以这里补一道成片级去重。
    """
    if not clips:
        return clips
    slot_meta = {s["slot_id"]: s for s in (slots or [])}
    used = []  # [(source_path, range)]
    used_paths = set()
    for c in clips:
        if c.get("tts_audio_path"):
            used.append((c["source_path"], c.get("source_time_range", "")))
            used_paths.add(c["source_path"])
            continue
        path, rng = c["source_path"], c.get("source_time_range", "")
        dup = any(ranges_overlap(path, rng, up, ur) for up, ur in used)
        if dup and toolbox is not None:
            sm = slot_meta.get(c["slot_id"], {})
            query = sm.get("want") or sm.get("role") or ""
            try:
                res = toolbox.retrieve(query, top_k=12)
            except Exception:  # noqa: BLE001
                res = {"candidates": []}
            repl = next((x for x in res.get("candidates", [])
                         if x.get("source_path") and x["source_path"] not in used_paths), None)
            if repl:
                _log.info("[dedup] %s 重复(%s)，替换为 %s", c["slot_id"],
                          os.path.basename(path), os.path.basename(repl["source_path"]))
                c["source_path"] = repl["source_path"]
                c["source_time_range"] = repl.get("source_time_range", "") or rng
                path, rng = c["source_path"], c["source_time_range"]
        used.append((path, rng))
        used_paths.add(path)
    return clips


@edit_tools.edit_tool_handler("tts_clone")
async def _handle_tts_clone(ctx: dict, action: dict) -> dict:
    """tts_clone 工具执行器：用参考片段音色把改写文案念出来，产出 wav 记到该 slot。"""
    slot_id = action.get("slot_id", "")
    ref_gid = action.get("ref_global_asset_id", "")
    text = (action.get("text") or "").strip()
    if not ctx.get("enable_tts", True):
        return {"ok": False, "error": "本次已关闭 TTS 配音，不能使用 tts_clone；空镜保留原声/静音即可"}
    toolbox = ctx.get("toolbox")
    seg = toolbox.by_gid.get(ref_gid) if (toolbox and ref_gid) else None
    if not slot_id or slot_id not in ctx.get("slot_meta", {}):
        return {"ok": False, "error": "slot_id 无效"}
    if not seg:
        return {"ok": False, "error": f"参考片段无效：{ref_gid}"}
    if not text:
        return {"ok": False, "error": "缺少要配音的文案 text"}
    # 该 slot 必须先有画面：配音是配给某个已选好的镜头的（工具说明里也写了先 place）。
    # 否则 Agent 会陷入 retrieve→tts_clone→retrieve 的空转（配了音但画面一直没定）。
    if slot_id not in (ctx.get("placements") or {}):
        return {"ok": False, "error": f"{slot_id} 还没有画面：请先用 place/place_original 为它选好片段，再 tts_clone 配音"}
    # 口型一致性硬约束：该镜画面里的人本来就在说话，就必须保留他的原声。换成克隆配音
    # 会让口型和听到的话对不上（用户明确要求：有人脸且开口说话 -> 保原声，没说话才克隆）。
    # 注意只统计**落在该镜窗口内**的字：素材池下发的 asr_items 为了对窗带了窗口外 ±1.5s 的
    # 上下文，若把它们算进来，"有人脸但没说话"的镜头会被邻镜的说话声误判成在说话，
    # 于是该配音的段也被拦下 -> 成片整段没配音。
    _p = (ctx.get("placements") or {}).get(slot_id) or {}
    _seg = (getattr(toolbox, "by_gid", {}) or {}).get(_p.get("global_asset_id")) or {}
    _ws = _seg.get("whq_speech") or {}
    _ws_start, _ws_end = _parse_range(_p.get("source_time_range", ""))
    _items = _ws.get("asr_items") or []
    _spoken = ""
    if _items and _ws_end > _ws_start:
        for _it in _items:
            try:
                _mid = (float(_it.get("start")) + float(_it.get("end"))) / 2.0
            except (TypeError, ValueError):
                continue
            if _ws_start <= _mid <= _ws_end:
                _spoken += str(_it.get("text") or "")
    elif not _items:
        # 老编排产物没给逐字 ASR（只有整段原声文本）：宁可保原声也不要口型错位，
        # 用整段原声文本判定"这段有人在说话"。
        _spoken = _ws.get("text") or ""
    if _p.get("voice_source") != "original" and len(_norm_text(_spoken)) >= 3:
        return {"ok": False, "error": (
            "这一镜的素材里人正在说话（原声：「{}」），换成克隆配音会导致**口型和话对不上**。"
            "请改用 place_original（同一个 global_asset_id={}，不换素材）保留用户原声；"
            "确实要换掉这段说话画面，就先 place 一个没有人说话的片段再配音。"
        ).format(_norm_text(_spoken)[:30], _p.get("global_asset_id", ""))}
    # 同一 slot 重复提交同一条文案 = 空转（每次 TTS 要几十秒）。直接拦住并提示下一步。
    prev = (ctx.get("tts_by_slot") or {}).get(slot_id) or {}
    if prev.get("text", "").strip() == text and prev.get("audio_path"):
        return {"ok": False, "error": (
            f"{slot_id} 已经用这条文案配过音了（不要重复提交同一条）。若还有别的 slot 没配音就去配，"
            "都配完了就输出 finish；若想改这一镜的文案，请给一条**不同的** text。")}
    # 字数硬约束：文案过长 -> TTS 比画面长得多，成片只能冻结末帧补足（画面静止数秒）。
    # 上限 = ref_cps(参考该段语速) × 该镜时长；没有 ref_cps 时按 5 字/秒兜底。
    meta = ctx.get("slot_meta", {}).get(slot_id) or {}
    target = float(meta.get("target_duration") or 0.0)
    cps = float((meta.get("whq_voice") or {}).get("ref_cps") or 0) or 5.0
    limit = int(max(8, target * cps)) if target > 0 else 0
    n = len(re.sub(r"[^\w\u4e00-\u9fff]+", "", text))
    if limit and n > limit:
        return {"ok": False, "error": (
            "文案过长：{} 字 > 本镜上限 {} 字（该镜 {:.1f}s × 参考语速 {:.1f} 字/秒）。"
            "配音比画面长会导致成片冻结末帧、画面静止。请压缩到 {} 字以内重试。"
        ).format(n, limit, target, cps, limit)}
    # 事实核验：文案不能出现"参考爆款有、用户素材里没有"的说法/数字（抹茶奶绿、51.7%、29.9元…）
    corpora = ctx.get("fact_corpora") or ()
    if len(corpora) == 2:
        bad = _copy_violations(text, corpora[0], corpora[1])
        if bad:
            return {"ok": False, "error": (
                "文案里这些内容来自**参考爆款那个商品**、用户素材里查不到出处：{}。"
                "请只讲本镜 material（该镜实际画面 + 该片段原声）里真实存在的东西重写；"
                "本商品没有对应卖点时，就只描述画面/使用感受，不要下事实断言。"
            ).format("、".join("「%s」" % b for b in bad))}
    # 文案不许和别的镜头重复：同一个卖点/说法讲两遍是最刺耳的"逻辑不通"。按 6 字连续
    # 重叠判定（含上一轮补配音生成的文案、以及原声段的字幕）。
    _mine = _norm_text(text)
    for _sid, _it in (ctx.get("tts_by_slot") or {}).items():
        if _sid == slot_id:
            continue
        _other = _norm_text((_it or {}).get("text", ""))
        _dup = next((_mine[i:i + 6] for i in range(0, max(0, len(_mine) - 5))
                     if _mine[i:i + 6] in _other), "")
        if _dup:
            return {"ok": False, "error": (
                "这条文案和 {} 的配音重复了（都出现「{}」）。同一个说法全片只能讲一次，"
                "请改成**本镜画面自己**该讲的内容（按它在叙事里的位置：痛点/产品登场/"
                "用法演示/效果呈现/催单），并和前后镜自然衔接。"
            ).format(_sid, _dup)}
    for _sid, _pp in (ctx.get("placements") or {}).items():
        if _sid == slot_id:
            continue
        _other = _norm_text((_pp or {}).get("caption", ""))
        if not _other:
            continue
        _dup = next((_mine[i:i + 6] for i in range(0, max(0, len(_mine) - 5))
                     if _mine[i:i + 6] in _other), "")
        if _dup:
            return {"ok": False, "error": (
                "这条文案和 {} 的原声字幕重复了（都出现「{}」）。请换成本镜画面自己该讲的内容。"
            ).format(_sid, _dup)}
    tts_dir = os.path.join(AGENT_ROOT, "uploads", "tts")
    os.makedirs(tts_dir, exist_ok=True)
    out_wav = os.path.join(tts_dir, f"tts_{slot_id}_{int(time.time() * 1000) % 1000000}.wav")
    res = await asyncio.to_thread(edit_tts.clone, seg.get("source_path", ""), seg.get("source_time_range", ""),
                                  seg.get("speech_or_text", ""), text, out_wav)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "TTS 失败")}
    # 生成后按**实际配音时长**兜一道：字数上限只是估算，语速慢时仍可能超出画面可用长度，
    # 超了成片会冻结末帧（画面卡住不动）。此时丢弃这段配音，让 Agent 按可用秒数重写。
    dur = float(res.get("duration") or 0.0)
    p = (ctx.get("placements") or {}).get(slot_id) or {}
    a, b = _parse_range(p.get("source_time_range", ""))
    total = _video_duration(_abspath(p.get("source_path", "")))
    avail = max(b - a, (total - a) if total > a else 0.0)   # 可从源片往后延到片尾
    if avail > 0.5 and dur > avail + 0.4:
        try:
            os.remove(out_wav)
        except OSError:
            pass
        return {"ok": False, "error": (
            "这段配音 {:.1f}s，但该镜画面最多只有 {:.1f}s（源片到片尾就这么长），"
            "成片会冻结末帧、画面卡住不动。请把文案压到约 {} 字以内重写；"
            "或先用 place 换一段更长的素材、或用 skip_slot 跳过这一段。"
        ).format(dur, avail, max(6, int(avail * cps)))}
    ctx.setdefault("tts_by_slot", {})[slot_id] = {
        "audio_path": out_wav, "text": text, "duration": res.get("duration", 0.0)}
    return {"ok": True, "slot_id": slot_id, "duration": res.get("duration", 0.0),
            "note": "已生成克隆配音；该镜将改用此配音、字幕=该文案"}


def _material_brief(toolbox, gid, limit=160):
    """某片段的实际内容摘要：画面描述 +（若有）片段自带口播。

    写 tts_clone 文案时必须**只讲这段素材里真实有的东西**——DNA 的 role/want 来自另一条
    参考爆款，照它写会把参考商品的成分/喝法/价格搬到本商品头上（whq 的老坑）。
    """
    seg = (getattr(toolbox, "by_gid", {}) or {}).get(gid) or {}
    if not seg:
        return ""
    desc = (seg.get("visual_description") or seg.get("one_sentence_summary") or "")[:limit]
    speech = ((seg.get("whq_speech") or {}).get("text") or seg.get("speech_or_text") or "")[:80]
    return desc + ("｜该片段原声：" + speech if speech else "")


async def _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_system,
                          base_placements=None, base_tts=None, revise_slots=None,
                          enable_tts=True, pure_music=False, voice_only_slots=None,
                          fact_corpora=None):
    """剪辑 Agent 的 ReAct 循环：自由召回 → 按需验证 → 放入（后台查重叠→仲裁）→ finish。

    增量重剪：base_placements/base_tts 是上一轮的成片基线（继承过来，不重排）；
    revise_slots 是审片点名要改的 slot——只有这些进入待办，其余 slot 默认保留上一轮结果，
    避免"为修一个镜头把其它已 OK 的镜头也重排坏了"的回归。首轮 base 为空则填全部 slot。

    voice_only_slots：**只配音不改画面**模式（whq 复刻用）。画面已由编排定好，这些 slot
    缺配音（whq 判定该段要克隆配音），Agent 只需为它们调 tts_clone 写文案。

    作为异步生成器：过程中 yield step 事件；最后 yield
    ``{"__edit_result__": True, "placements": {...}, "notes": [...], "tts_by_slot": {...}}``。
    """
    placements: dict = dict(base_placements or {})   # 继承上一轮成片基线
    tts_by_slot: dict = dict(base_tts or {})
    used: list = []                # [{gid, source_path, range, speech_norm, slot_id}]
    slot_meta = {s["slot_id"]: s for s in slots}
    scratch: list = []             # 最近若干步 {action, observation}
    notes: list = []
    max_steps = len(slots) * 6 + 12
    tag = f"[{rid}] loop{loop} 剪辑Agent"

    def _register(slot_id, seg):
        used[:] = [u for u in used if u["slot_id"] != slot_id]
        used.append({"gid": seg.get("global_asset_id", ""), "source_path": seg.get("source_path", ""),
                     "range": seg.get("source_time_range", ""),
                     "speech_norm": normalize_speech(seg.get("speech", "")), "slot_id": slot_id})

    # 把继承来的基线 placements 登记进 used，保证重叠检测/召回去重对"保留的镜头"也生效
    for _sid, _p in placements.items():
        _register(_sid, _p)

    if voice_only_slots:
        # 只配音模式：画面已定好，待办 = 缺配音的 slot（Agent 只对它们调 tts_clone）
        unfilled = [s for s in voice_only_slots if s in slot_meta]
    elif base_placements and revise_slots:
        # 增量重剪：只把审片点名的 slot 列为待办；其余保留基线
        unfilled = [s for s in revise_slots if s in slot_meta]
    elif base_placements:
        # 已用复刻方案预填（seeded）：待办 = 尚未有片段的 slot；已有的默认采用，不重新召回
        unfilled = [s["slot_id"] for s in slots if s["slot_id"] not in placements]
    else:
        # 无基线（无复刻方案预填）：填全部 slot
        unfilled = [s["slot_id"] for s in slots]

    def _overlap_slot(slot_id, path, rng, speech):
        sn = normalize_speech(speech)
        for u in used:
            if u["slot_id"] == slot_id:
                continue
            if ranges_overlap(path, rng, u["source_path"], u["range"]):
                return u["slot_id"]
            if sn and sn == u["speech_norm"]:
                return u["slot_id"]
        return None

    incremental = bool(base_placements and revise_slots)
    seeded = bool(base_placements) and not revise_slots and unfilled != [s["slot_id"] for s in slots]
    step = 0
    while step < max_steps:
        step += 1
        # 按 slot 顺序给出「叙事上下文」：每镜的目的(role/want) + 当前这镜的语音内容
        # （tts 配音文案 > 原声口播/字幕 > 未定）。tts_clone 写配音时要据此衔接前后镜头。
        narrative = []
        for s in slots:
            sid = s["slot_id"]
            p = placements.get(sid)
            voice = ""
            if sid in tts_by_slot:
                voice = tts_by_slot[sid].get("text", "")
            elif p:
                voice = p.get("caption") or p.get("speech") or ""
            narrative.append({"slot_id": sid, "role": s["role"], "want": s["want"],
                              "voice": voice or "(未定)"})
        mode_bits = []
        if not enable_tts:
            mode_bits.append("本次关闭 TTS 配音：**不要调用 tts_clone**；无口播的空镜保留原声/静音即可，不要给它配音。")
        if pure_music:
            mode_bits.append("参考是纯音乐无人声：选片要**贴合每镜 target_duration、保持与参考相同的剪辑节奏**"
                             "（每镜时长、切换快慢尽量一致）；不做配音；**全程不烧任何字幕——place 一律不要给 caption，"
                             "系统也不提供字幕烧录能力**。")
        agent_user = json.dumps({
            "dna": dna,
            "slots": [dict({"slot_id": s["slot_id"], "role": s["role"], "want": s["want"],
                            "target_duration": s["target_duration"]},
                           **({"whq_voice": s["whq_voice"]} if s.get("whq_voice") else {}))
                      for s in slots],
            "narrative": narrative,
            "mode": {"enable_tts": enable_tts, "pure_music": pure_music},
            "review_feedback": review_feedback,
            "history_brief": [{"loop": h["loop"], "problems": h.get("problems", [])} for h in history[-2:]],
            "baseline_placements": [{"slot_id": k, "gid": v.get("global_asset_id"),
                                     "caption": v.get("caption") or v.get("speech", ""),
                                     # 该镜**实际画面内容**（写配音文案的唯一依据；DNA 的 want
                                     # 来自另一条参考爆款，只表示这一段承担的叙事功能）
                                     "material": _material_brief(toolbox, v.get("global_asset_id")),
                                     "tts": k in tts_by_slot} for k, v in placements.items()],
            "slots_to_revise": unfilled,
            "recent_steps": scratch[-8:],
            "instruction": (" ".join(mode_bits) + " " if mode_bits else "") + (
                ("【只配音模式】所有镜头的**画面已由编排定好**（见 baseline_placements），"
                 "**不要**用 retrieve 重新召回、也不要 place 换别的素材改动画面。"
                 "对 slots_to_revise 里的每个 slot，先看它 baseline_placements 里的 material：\n"
                 "① 如果 material 显示**该片段自带原声口播**（有「该片段原声：…」），"
                 "**优先用 place_original 保留用户真声**（同一个 global_asset_id，不换素材，"
                 "它只会把窗口吸附到自然停顿、把字幕设成那句原话）——真人口型对得上，比克隆配音好；"
                 "编排把这段标成 clone 只是因为它按「拉伸填满槽位」估算过，Agent 出片不拉伸，"
                 "所以有真声就该用真声。\n"
                 "② 只有 material 里**确实没有原声**时，才对它调 tts_clone。写 text 的依据是"
                 "**该 slot 的 material（这一镜的实际画面内容）**——只讲这段素材画面里真实有的东西；"
                 "DNA 的 role/want 来自另一条参考爆款，**只用来判断这一段承担什么叙事功能**"
                 "（痛点/产品登场/配料特写/使用演示/价格/催单），里面的商品名/成分/含量/价格/喝法"
                 "一律不许写进文案。再结合 narrative 让前后自然衔接，"
                 "字数不超过 whq_voice.ref_cps × 该镜时长（配音比画面长会导致画面卡住不动）。\n"
                 "若某个 slot 的画面内容与它要承担的叙事功能完全对不上（如参考要求"
                 "「改数量下单演示」但素材里没有任何下单/价格画面），用 skip_slot 跳过该段，"
                 "**不要编造素材里没有的价格/活动/成分**。"
                 "结束前逐个确认：slots_to_revise 里每个 slot 要么已 place_original 保留原声、"
                 "要么已 tts_clone 配上音、要么已 skip_slot——**不要留下既没原声又没配音的哑巴段**"
                 "（画面里有人但没说话的镜头必须 tts_clone，不能就这么静着过去）。"
                 "全部处理完后输出 {\"action\":\"finish\"}。")
                if voice_only_slots else
                ("增量重剪模式：baseline_placements 是上一轮已成片的镜头，**只修订 slots_to_revise 里点名的 slot**，"
                 "其余镜头保持不动、不要重新召回或改动。改完点名的 slot 就输出 {\"action\":\"finish\"}。")
                if incremental else
                ("复刻方案已给出每镜选片（baseline_placements 就是编排 Agent 定好的复刻分镜）——"
                 "**默认直接采用，不要重新召回**；只有当某镜的选片明显不合适时，才对那一镜用 retrieve 换片，"
                 "然后输出 {\"action\":\"finish\"}。unfilled_slots 里若有未填的 slot 才需要召回补上。")
                if seeded else
                "只输出一个动作 JSON。所有 slot 都放好后输出 {\"action\":\"finish\"}。"),
        }, ensure_ascii=False)
        action = await _run_llm(edit_system, agent_user, tag=(tag if step == 1 else ""))
        act = str(action.get("action", "")).lower()

        if act == "finish" or (not act and not unfilled):
            break

        if act == "retrieve":
            sid = action.get("slot_id", "")
            query = action.get("query", "") or slot_meta.get(sid, {}).get("want", "")
            res = toolbox.retrieve(query, top_k=int(action.get("top_k") or 6),
                                   exclude_gids=[u["gid"] for u in used])
            cands = res["candidates"]
            obs = {"ok": True, "slot_id": sid, "backend": res.get("backend"), "pool_size": res.get("pool_size"),
                   "candidates": [{"global_asset_id": c["global_asset_id"], "summary": c["summary"],
                                   "speech": c["speech"], "source_time_range": c["source_time_range"],
                                   "duration": c["duration"], "keywords": c["keywords"]} for c in cands]}
            _log.info("%s retrieve slot=%s query=%s -> %d cand", tag, sid, query, len(cands))
            scratch.append({"action": f"retrieve[{sid}] {query}", "observation": obs})

        elif act == "verify":
            gid = action.get("global_asset_id", "")
            vr = await toolbox.verify(gid, action.get("question", ""))
            obs = {"ok": not vr.get("error"), "global_asset_id": gid,
                   "observation": vr.get("observation", "")[:600], "error": vr.get("error", "")}
            _log.info("%s verify gid=%s -> %s", tag, gid, (vr.get("observation") or vr.get("error"))[:120])
            scratch.append({"action": f"verify {gid}", "observation": obs})

        elif act == "place":
            sid = action.get("slot_id", "")
            gid = action.get("global_asset_id", "")
            seg = toolbox.by_gid.get(gid)
            if not sid or sid not in slot_meta or not seg:
                scratch.append({"action": f"place {sid}", "observation": {"ok": False, "error": "slot_id 或 global_asset_id 无效"}})
                continue
            path = seg.get("source_path", "")
            rng = action.get("source_time_range") or seg.get("source_time_range", "")
            speech = seg.get("speech_or_text", "")
            conflict_slot = _overlap_slot(sid, path, rng, speech)
            if conflict_slot:
                # 后台重叠 → 触发仲裁 Agent
                yield _estep(rid, f"arb{loop}-{sid}", "重叠检测触发审核", state="running",
                             thought=f"{sid} 想放的素材/口播与 {conflict_slot} 重叠，交审核 Agent 裁决归属")
                verdict = await arbiter.arbitrate_overlap(
                    {"summary": seg.get("one_sentence_summary", ""), "speech": speech, "source_time_range": rng},
                    [{"slot_id": sid, **{k: slot_meta[sid][k] for k in ("role", "want")}},
                     {"slot_id": conflict_slot, **{k: slot_meta[conflict_slot][k] for k in ("role", "want")}}])
                winner = verdict["winner_slot_id"]
                _log.info("%s overlap %s vs %s -> winner=%s (%s)", tag, sid, conflict_slot, winner, verdict.get("reason"))
                yield _estep(rid, f"arb{loop}-{sid}", "审核裁决重叠归属",
                             thought=f"审核判定 {winner} 保留该素材（{verdict.get('reason','')}），{'; '.join(verdict.get('losers',[]))} 需另选",
                             observation=json.dumps(verdict, ensure_ascii=False))
                if winner != sid:
                    # 本次放入的是输家 → 拒绝，要求另选
                    scratch.append({"action": f"place {sid}", "observation": {
                        "ok": False, "overlap_with": conflict_slot,
                        "arbitration": f"该素材判归 {conflict_slot}；{sid} 请用 retrieve 另选一个不冲突的片段",
                        "used_gids": [u["gid"] for u in used]}})
                    continue
                # 本次是赢家 → 抢占，把输家 conflict_slot 的 placement 移除并重新列入待填
                placements.pop(conflict_slot, None)
                tts_by_slot.pop(conflict_slot, None)
                used[:] = [u for u in used if u["slot_id"] != conflict_slot]
                if conflict_slot not in unfilled:
                    unfilled.append(conflict_slot)
                scratch.append({"action": f"place {sid}", "observation": {
                    "ok": True, "won_over": conflict_slot,
                    "note": f"{sid} 抢到该素材；{conflict_slot} 需重新召回另选"}})
            else:
                scratch.append({"action": f"place {sid}", "observation": {"ok": True}})
            # 重新放片 = 该 slot 的画面变了，清掉旧的克隆配音（如需要 Agent 会再调 tts_clone）
            tts_by_slot.pop(sid, None)
            placements[sid] = {
                "global_asset_id": gid, "source_path": path, "source_time_range": rng,
                "target_duration": action.get("target_duration") or slot_meta[sid]["target_duration"],
                "caption": action.get("caption", ""), "speech": speech,
                "burn_caption": action.get("burn_caption", True), "speed": action.get("speed", 1.0),
            }
            _register(sid, {"global_asset_id": gid, "source_path": path, "source_time_range": rng, "speech": speech})
            if sid in unfilled:
                unfilled.remove(sid)
            if action.get("note"):
                notes.append(str(action["note"]))
        else:
            # 扩展工具的执行插件点：别人用 @edit_tool_handler 注册的新工具走这里
            handler = edit_tools.get_tool_handler(act)
            if handler:
                ctx = {"toolbox": toolbox, "used": used, "placements": placements,
                       "slot_meta": slot_meta, "notes": notes, "tts_by_slot": tts_by_slot,
                       "enable_tts": enable_tts, "pure_music": pure_music,
                       # 放片类扩展工具（如 whq_clone 的 place_original）需要这两个才能
                       # 把 slot 标记为已填 + 参与重叠检测
                       "unfilled": unfilled, "register": _register,
                       # (用户素材语料, 参考爆款语料)：tts_clone 用它核验文案没抄参考商品的事实
                       "fact_corpora": fact_corpora}
                try:
                    obs = await handler(ctx, action)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("%s tool '%s' handler failed: %s", tag, act, exc)
                    obs = {"ok": False, "error": str(exc)[:200]}
                scratch.append({"action": str(act), "observation": obs if isinstance(obs, dict) else {"ok": True, "result": obs}})
            else:
                scratch.append({"action": str(act), "observation": {"ok": False, "error": "未知动作，请用 retrieve/verify/place/finish"}})

    yield {"__edit_result__": True, "placements": placements, "notes": notes, "tts_by_slot": tts_by_slot}


def _voice_plan_brief(slots, placements, tts_by_slot):
    """逐 slot 的声音方案，交给审片 Agent 判「该用原声的段是不是被换成了克隆配音」。

    whq 结构级复刻里 whq_voice.voice_source 是编排阶段定的基线（expected），actual 是本轮
    剪辑 Agent 实际的选择：original=保留用户真声、clone=克隆配音、none=未配音。
    """
    out = []
    for s in slots or []:
        sid = s.get("slot_id")
        p = placements.get(sid) or {}
        if sid in (tts_by_slot or {}):
            actual = "clone"
            text = (tts_by_slot[sid] or {}).get("text", "")
        elif p.get("voice_source") == "original" or p.get("speech"):
            actual = "original"
            text = p.get("caption") or p.get("speech", "")
        else:
            actual = "none"
            text = p.get("caption", "")
        item = {"slot_id": sid, "actual_voice": actual, "text": text}
        wv = s.get("whq_voice") or {}
        if wv:
            item["expected_voice"] = wv.get("voice_source")
            item["ref_cps"] = wv.get("ref_cps")
        out.append(item)
    return out


def _estep(rid, key, title, *, thought="", state="done", observation=None):
    ev = {"type": "step", "phase": "Agent 剪辑", "key": f"{key}-{rid}", "state": state,
          "title": title, "thought": thought}
    if observation is not None:
        ev["observation"] = observation
    return ev


def _ops_brief(result: dict) -> list:
    out = []
    for o in result.get("ops", []):
        if o.get("op") == "trim":
            out.append(f"{o.get('slot_id')}: {o.get('source')} {o.get('source_time_range')} {o.get('target_duration')}s "
                       f"speed={o.get('speed')} 字幕{'已烧' if o.get('caption_burned') else '未烧'}"
                       f"{' 克隆配音' if o.get('tts') else ''} {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
        else:
            out.append(f"{o.get('op')}: {'OK' if o.get('ok') else '失败:'+str(o.get('error',''))[:40]}")
    return out


def _flagged_slots(review: dict, all_slot_ids: list) -> list:
    """从审片结果里解析出「被点名要改」的 slot（problems.slot_id + commands 文本里的 Sxx）。

    解析不到（如只写"全局"）时返回全部 slot——安全兜底为全量重排。
    """
    bits = [str(p.get("slot_id", "")) for p in (review.get("problems", []) or [])]
    bits += [str(c) for c in (review.get("commands", []) or [])]
    combined = " ".join(bits)
    flagged = []
    for sid in all_slot_ids:
        num = sid.lstrip("S").lstrip("0") or "0"
        if sid in combined or re.search(rf"S0*{num}\b", combined):
            flagged.append(sid)
    return flagged or list(all_slot_ids)


def _asr_clip_check(clips: list) -> list:
    """用缓存的 ASR 句级时间戳，逐镜判断结束点是否切在句子中间（口播被截断），并给出该镜内的 ASR 文本。

    供审片 Agent 参考：ASR 时间戳比"凭听感"更客观地判断截断；ASR 文本也帮审片对比"听到的"
    是否与"ASR 以为说的"一致（识别成杂音/指导语时两者会对不上）。仅对有 ASR 缓存的源片段产出。
    """
    rows = []
    for c in clips or []:
        sp = c.get("source_path", "")
        if not sp:
            continue
        asr = asr_cache.get(sp)
        segs = (asr or {}).get("segments") or []
        if not segs:
            continue
        try:
            a, b = str(c.get("source_time_range", "")).split("-")
            s, e = float(a), float(b)
        except (ValueError, AttributeError):
            continue
        played = [g for g in segs if float(g.get("end", 0)) > s + 0.05 and float(g.get("start", 0)) < e - 0.05]
        row = {"slot_id": c.get("slot_id"), "clip_range": c.get("source_time_range", ""),
               "asr_in_clip": "".join(g.get("text", "") for g in played)[:140]}
        # 结束截断：有句子在 e 之前开始、却在 e 之后才结束 → 这句被切断
        for g in segs:
            gs, ge = float(g.get("start", 0)), float(g.get("end", 0))
            if gs < e - 0.15 and ge > e + 0.25 and ge > s:
                row["truncated_at_end"] = True
                row["cut_sentence"] = g.get("text", "")
                row["suggest_extend_end_to"] = round(ge + 0.15, 2)
                break
        rows.append(row)
    return rows


async def agent_edit_stream(strategy_path: str, *, enable_bgm: bool = True, max_loops: int = None,
                            review_model: str = "qwen", enable_review: bool = True,
                            enable_tts: bool = True, pure_music: bool = False,
                            reuse_bgm: bool = False, reference_video: str = "",
                            beat_sync: bool = False, burn_subtitle: bool = True,
                            missing_shot_mode: str = "aigc"):
    """产出 step / edit_log / agent_edit_done / error 事件。

    review_model: 审片后端 ``qwen``（只看画面）或 ``gemini``（画面+声音）。
    enable_review: 是否启用 Agent 审片-重剪循环；False 则只剪一轮直接输出。
    enable_tts:   是否允许剪辑 Agent 用 tts_clone 配音（纯音乐参考默认关）。
    pure_music:   参考视频为纯音乐无口播——选片按时长/节奏对齐、成片静音、不做 TTS。
    reuse_bgm:    是否复用参考视频的 BGM（从 reference_video 分离/抽取）作成片配乐。
    reference_video: 参考视频 uri（reuse_bgm 时需要）。
    missing_shot_mode: 缺失镜头处理——``aigc`` 时启用 AIGC 补镜 Agent 为缺失镜头生成片段。
    """
    rid = time.strftime("%H%M%S")
    max_loops = max_loops or MAX_LOOPS
    if not enable_review:
        max_loops = 1
    review_model = (review_model or "qwen").lower()
    if review_model not in ("qwen", "gemini"):
        review_model = "qwen"
    # 无人声/纯音乐 → 强制不烧字幕（保证）；否则按用户的「烧字幕」开关
    burn_on = bool(burn_subtitle) and not pure_music

    def step(key, title, thought, state="done", observation=None):
        ev = {"type": "step", "phase": "Agent 剪辑", "key": f"{key}-{rid}", "state": state,
              "title": title, "thought": thought}
        if observation is not None:
            ev["observation"] = observation
        return ev

    strategy_abs = _abspath(strategy_path)
    if not strategy_abs or not os.path.isfile(strategy_abs):
        _log.warning("[%s] agent_edit abort: strategy not found %s", rid, strategy_path)
        yield {"type": "error", "message": f"找不到编排脚本：{strategy_path}"}
        return
    with open(strategy_abs, "r", encoding="utf-8") as fh:
        strategy = json.load(fh)
    # 契约②边界自检（消费端）：A 给的 strategy 若字段不符只告警，方便定位是不是上游改坏了接口
    _cp = contracts.validate_strategy(strategy)
    if _cp:
        _log.warning("[%s] [contract] 收到的 EditingStrategy 不符契约(v%s)：%s",
                     rid, contracts.CONTRACT_VERSION, contracts.summarize(_cp))
    context = _load_context(strategy_abs, strategy)
    inputs = _load_inputs(strategy, context)
    # whq_clone 链路：挂上 place_original 工具（句子级对窗保留用户原声）。工具是链路专属的，
    # 只在方案确实来自 whq 编排时注册，其他链路的 prompt 不受影响。
    if (strategy.get("metadata") or {}).get("reproduce_mode") == "whq_clone" \
            or context.get("source") == "whq_clone":
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "whq_clone"))
            import agent_tools as _whq_agent_tools  # noqa: F401  import 即注册
            _log.info("[%s] whq_clone: place_original 工具已挂载", rid)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] whq_clone place_original 挂载失败(退化为 place+tts_clone)：%s", rid, exc)
    slots = inputs["slots"]
    aigc_slots = inputs.get("aigc_slots", []) or []
    do_aigc = (missing_shot_mode or "").lower() == "aigc" and bool(aigc_slots)
    if not slots and not do_aigc:
        _log.warning("[%s] agent_edit abort: no editable slots (strategy=%s)", rid, strategy_abs)
        yield {"type": "error", "message": "没有可剪辑的镜头（都是补拍/AIGC 生成或候选缺失）。"}
        return
    segments = [s for s in (context.get("segments") or []) if isinstance(s, dict)]
    if not segments:
        _log.warning("[%s] agent_edit abort: empty material pool", rid)
        yield {"type": "error", "message": "素材池为空，无法召回剪辑。"}
        return
    task_id = "agentedit_" + hashlib.sha1(strategy_abs.encode("utf-8")).hexdigest()[:12]
    toolbox = EditToolbox(task_id, segments)
    # 文案事实核验语料：用户素材说了什么 vs 参考爆款说了什么（见 _copy_violations）
    fact_corpora = _fact_corpora(context, toolbox)

    # 缺失镜头 AIGC 补齐：missing_shot_mode=aigc 时，为需生成的镜头起 AIGC 子 Agent（并发≤5）
    # 生成本地片段，并入 slots + presets，后续与用户素材镜头一起进剪辑/审片。
    aigc_presets = {}
    if do_aigc:
        yield step("aigc", "缺失镜头 AIGC 生成", f"{len(aigc_slots)} 个镜头无匹配素材，启用 AIGC 补镜 Agent", state="running")
        async for ev in aigc_agent.generate_missing_shots(rid, aigc_slots, segments, task_id, burn_on=burn_on):
            if ev.get("__aigc_result__"):
                clips_by_slot = ev.get("clips_by_slot", {})
                by_meta = {s["slot_id"]: s for s in aigc_slots}
                for asid, clip in clips_by_slot.items():
                    m = by_meta.get(asid, {})
                    slots.append({"slot_id": asid, "role": m.get("role", ""), "want": m.get("want", ""),
                                  "breakdown": m.get("breakdown", []), "target_duration": m.get("target_duration", 3.0)})
                    aigc_presets[asid] = {
                        "global_asset_id": f"aigc::{asid}", "source_path": clip["source_path"],
                        "source_time_range": clip["source_time_range"],
                        "target_duration": clip.get("target_duration", m.get("target_duration", 3.0)),
                        "caption": clip.get("caption", ""), "speech": "",
                        "burn_caption": bool(clip.get("caption")), "speed": clip.get("speed", 1.0),
                    }
                continue
            yield ev
        # 按 slot_id 顺序稳定排序（AIGC 生成是并发的，顺序可能乱）
        slots.sort(key=lambda s: s["slot_id"])
        yield step("aigc", "缺失镜头 AIGC 生成", f"补齐 {len(aigc_presets)}/{len(aigc_slots)} 个镜头", state="done")
    if not slots:
        _log.warning("[%s] agent_edit abort: no slots after AIGC", rid)
        yield {"type": "error", "message": "缺失镜头 AIGC 生成未产出可用片段，且无其它可剪辑镜头。"}
        return
    edit_sys = edit_system_prompt()   # SKILL.md 角色/约束 + 工具注册表渲染的工具清单
    review_p = review_prompt()        # 审片 Agent 的 SKILL.md
    dna = {"product_name": inputs["product_name"], "narrative_structure": inputs["narrative_structure"],
           "slots": [{"slot_id": s["slot_id"], "role": s["role"], "want": s["want"],
                      "target_duration": s["target_duration"]} for s in slots]}
    bgm = DEFAULT_BGM if enable_bgm and os.path.isfile(DEFAULT_BGM) else ""
    if reuse_bgm and reference_video:
        yield step("bgm-reuse", "复用参考 BGM", "从参考视频分离/抽取背景音乐用于成片", state="running")
        bg = await asyncio.to_thread(bgm_reuse.extract_reference_bgm, reference_video, pure_music=pure_music)
        if bg.get("ok"):
            bgm = bg["output"]
            yield step("bgm-reuse", "复用参考 BGM", f"已取得参考 BGM（{bg.get('mode')}），成片将用它作配乐", state="done")
        else:
            yield step("bgm-reuse", "复用参考 BGM", f"失败：{bg.get('error','')}，回退默认 BGM", state="done")
    # 卡点剪辑：对成片用的 BGM 检测鼓点，供逐镜卡点吸附
    beats = []
    if beat_sync and bgm:
        yield step("beats", "BGM 鼓点分析", "检测 BGM 节拍点用于卡点剪辑", state="running")
        bt = await asyncio.to_thread(beats_tool.detect_beats, bgm)
        if bt.get("ok") and bt.get("beats"):
            beats = bt["beats"]
            yield step("beats", "BGM 鼓点分析", f"检测到 {len(beats)} 个鼓点（tempo≈{round(bt.get('tempo') or 0)}），成片将卡点剪辑", state="done")
        else:
            yield step("beats", "BGM 鼓点分析", f"未能检测鼓点：{bt.get('error','')}，本次不卡点", state="done")
    elif beat_sync and not bgm:
        yield step("beats", "BGM 鼓点分析", "未启用/未取得 BGM，卡点剪辑需要 BGM，跳过", state="done")
    _log.info("[%s] agent_edit start strategy=%s slots=%d pool=%d max_loops=%d bgm=%s review_model=%s pure_music=%s tts=%s review=%s",
              rid, os.path.basename(strategy_abs), len(slots), toolbox.pool_size(), max_loops, bool(bgm),
              review_model, pure_music, enable_tts, enable_review)

    final_dir = os.path.join(AGENT_ROOT, "uploads", "final")
    os.makedirs(final_dir, exist_ok=True)
    slug = "agentcut_" + time.strftime("%Y%m%d_%H%M%S")

    history = []          # 每轮简要（供审片看历轮）
    best = None           # {score, video_uri, path, loop}
    review_feedback = {}
    # 首轮用复刻方案（编排 Agent 已定好的每镜选片）预填，剪辑 Agent 默认采用、不重复全池召回
    prev_placements = dict(inputs.get("presets") or {})
    prev_placements.update(aigc_presets)   # AIGC 补镜生成的片段也作为已定选片种入
    prev_tts = {}
    revise_slots = None   # 审片点名要改的 slot；None=全量
    all_slot_ids = [s["slot_id"] for s in slots]

    yield step("prep", "读取编排脚本 + 建全池召回",
               f"{len(slots)} 个 DNA 槽位待填，素材池 {toolbox.pool_size()} 段可自由召回，进入 Agent 剪辑-审片循环（最多 {max_loops} 轮）",
               observation="\n".join(f"{s['slot_id']} {s['role']} 目标{s['target_duration']}s" for s in slots))

    for loop in range(1, max_loops + 1):
        # 1) 出片计划
        incr = bool(prev_placements and revise_slots)
        seeded_all = bool(prev_placements) and not revise_slots and all(s["slot_id"] in prev_placements for s in slots)
        # whq 复刻：编排把某些段判为"该走克隆配音"（无可用原声）。这些段只有画面、没有口播，
        # 直接采用基线会得到「画面对但全程没配音没字幕」的成片 —— 必须让 Agent 为它们配音。
        voice_slots = []
        if seeded_all and enable_tts and not pure_music:
            voice_slots = [s["slot_id"] for s in slots
                           if (s.get("whq_voice") or {}).get("voice_source") == "clone"
                           and not (prev_placements.get(s["slot_id"], {}) or {}).get("speech")
                           and s["slot_id"] not in prev_tts]
        if seeded_all and not voice_slots:
            # 复刻方案已为每个 slot 选好片段 → **直接采用，不跑剪辑 Agent、不重新召回**，
            # 保证成片片段与复刻分镜完全一致（后续审片再按需增量改）。
            yield step(f"edit{loop}", f"第 {loop} 轮 · 采用复刻分镜选片",
                       "直接使用编排 Agent 定好的复刻分镜每镜选片（不重新召回）", state="done",
                       observation="\n".join(f"{sid} {os.path.basename(p.get('source_path',''))} {p.get('source_time_range','')}"
                                             for sid, p in prev_placements.items()))
            placements = dict(prev_placements)
            tts_by_slot = dict(prev_tts)
            notes = ["直接采用复刻方案选片"]
        else:
            voice_only = bool(seeded_all and voice_slots)
            if voice_only:
                title = "（只配音：" + "、".join(voice_slots) + "）"
                desc = "画面沿用复刻分镜不动，只为无原声的镜头生成克隆配音 + 字幕文案"
            else:
                title = ("（增量修订：" + "、".join(revise_slots) + "）") if incr else "自由召回选片"
                desc = ("只重剪审片点名的镜头，其余保留上一轮" if incr else
                        "从全池按 DNA 角色召回、按需验证、放入并后台查重叠")
            yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent{title}", desc, state="running")
            placements, notes = {}, []
            tts_by_slot = {}
            async for ev in _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_sys,
                                            base_placements=(prev_placements or None), base_tts=(prev_tts or None),
                                            revise_slots=revise_slots, enable_tts=enable_tts, pure_music=pure_music,
                                            voice_only_slots=(voice_slots if voice_only else None),
                                            fact_corpora=fact_corpora):
                if ev.get("__edit_result__"):
                    placements, notes = ev["placements"], ev["notes"]
                    tts_by_slot = ev.get("tts_by_slot", {})
                    continue
                yield ev
        # 兜底补配音：Agent 有时会漏掉"既没原声、也没配音"的镜头（哑巴段）。这里客观检测
        # 出来，再跑一轮**只配音**把它们补上（最多补一次，避免无限循环）。带货成片不能有
        # 整段没人声的空档。
        if enable_tts and not pure_music:
            mute = _mute_slots(placements, slots, tts_by_slot, toolbox)
            if mute:
                yield step(f"mute{loop}", f"第 {loop} 轮 · 补配音（{len(mute)} 段哑巴）",
                           "检测到既无原声又无配音的镜头：" + "、".join(mute) + "，补生成克隆配音",
                           state="running")
                async for ev in _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_sys,
                                                base_placements=placements, base_tts=tts_by_slot,
                                                enable_tts=enable_tts, pure_music=pure_music,
                                                voice_only_slots=mute, fact_corpora=fact_corpora):
                    if ev.get("__edit_result__"):
                        placements = ev["placements"]
                        tts_by_slot = ev.get("tts_by_slot", tts_by_slot)
                        notes += ev.get("notes", [])
                        continue
                    yield ev
                still = _mute_slots(placements, slots, tts_by_slot, toolbox)
                yield step(f"mute{loop}", f"第 {loop} 轮 · 补配音",
                           ("已补齐全部哑巴段" if not still else "仍有无声段：" + "、".join(still)),
                           state="done")
        # 本轮的成片计划成为下一轮的基线（下一轮据本轮审片只改被点名的 slot）
        prev_placements, prev_tts = placements, tts_by_slot
        clips = _placements_to_clips(placements, slots, tts_by_slot, burn_captions=burn_on)
        _dedup_clips(clips, toolbox, slots)  # 成片级去重：同段素材被多镜复用 → 换未用过的素材
        _snap_clips_to_sentences(clips, toolbox)  # 保原声的镜头：结束点吸到自然停顿，别把话切一半
        if beats:
            _snap_clips_to_beats(clips, beats)  # 卡点：结束点吸附到 BGM 鼓点
        note = "；".join(notes[:3])
        if not clips:
            _log.warning("[%s] loop%d 剪辑Agent 未产出任何 placement", rid, loop)
            yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 选片", "本轮未放入任何镜头，跳过", state="done")
            history.append({"loop": loop, "edit_note": note, "ops": [], "review": "本轮未产出", "problems": ["剪辑 Agent 未放入镜头"]})
            review_feedback = {"commands": ["上一轮没有放入任何镜头，请用 retrieve 召回后 place 每个 slot"]}
            revise_slots = None  # 全量重排
            continue
        _log.info("[%s] loop%d 剪辑Agent placed=%d note=%s detail=%s", rid, loop, len(clips), note,
                  json.dumps([{"slot": c["slot_id"], "src": os.path.basename(c["source_path"]),
                               "range": c["source_time_range"], "dur": c["target_duration"],
                               "speed": c["speed"], "cap": c["caption_text"]} for c in clips], ensure_ascii=False)[:2000])
        yield step(f"edit{loop}", f"第 {loop} 轮 · 剪辑 Agent 选片完成",
                   note or f"放入 {len(clips)} 个镜头",
                   observation=json.dumps([{"slot": c["slot_id"], "src_range": c["source_time_range"],
                                            "dur": c["target_duration"], "speed": c["speed"],
                                            "cap": c["caption_text"]} for c in clips], ensure_ascii=False))

        # 2) 执行剪辑
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"ffmpeg 逐镜 trim/字幕/拼接{'/BGM' if bgm else ''}", state="running")
        out_path = os.path.join(final_dir, f"{slug}_loop{loop}.mp4")
        result = editor.build_video(clips, out_path, bgm_path=bgm, width=720, height=1080, fps=30,
                                    mute_source=pure_music)
        ops_brief = _ops_brief(result)
        _log.info("[%s] loop%d 剪辑执行 clip_count=%s error=%s ops=%s", rid, loop,
                  result.get("clip_count"), result.get("error", ""),
                  json.dumps(ops_brief, ensure_ascii=False)[:3000])
        if result.get("error") or not result.get("output"):
            yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"剪辑失败：{result.get('error','')}", state="done",
                       observation="\n".join(ops_brief))
            history.append({"loop": loop, "edit_note": note, "ops": ops_brief,
                            "review": "本轮剪辑失败", "problems": [result.get("error", "")]})
            review_feedback = {"commands": [f"上一轮剪辑失败：{result.get('error','')}，请调整选片/时长后重试"]}
            revise_slots = None  # 剪辑失败 → 全量重排
            continue
        video_uri = f"uploads/final/{os.path.basename(out_path)}"
        yield step(f"cut{loop}", f"第 {loop} 轮 · 执行剪辑", f"成片 {result['clip_count']} 镜已生成",
                   observation="\n".join(ops_brief))
        yield {"type": "agent_edit_sample", "loop": loop, "video_uri": video_uri}

        if not enable_review:
            # 未启用 Agent 审片：剪一轮直接出片，不进入审片-重剪循环
            _log.info("[%s] agent_edit done (审片关闭) loop=%d video=%s", rid, loop, video_uri)
            _save_edit_trace(task_id, strategy_abs, video_uri=video_uri, score=0,
                             verdict="已按剪辑 Agent 结果直接输出（未开启审片）", history=history, review_model=review_model)
            yield {"type": "agent_edit_done", "video_uri": video_uri, "final_path": out_path,
                   "loops": loop, "score": 0, "material_limited": False, "reviewed": False,
                   "review_model": review_model, "verdict": "已按剪辑 Agent 结果直接输出（未开启审片）",
                   "history": history}
            return

        # 3) 审片 Agent（看视频）
        model_label = "Gemini（画面+声音）" if review_model == "gemini" else "默认视觉模型"
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 看片", f"用 {model_label} 对照 DNA 审阅成片，定位问题", state="running")
        # 审片用的 DNA 去掉 target_duration —— 避免审片盯着"单镜差几秒"扣分/要求降速凑时长（优先看效果）
        review_dna = dict(dna)
        review_dna["slots"] = [{k: v for k, v in s.items() if k != "target_duration"} for s in dna.get("slots", [])]
        asr_check = _asr_clip_check(clips)  # ASR 句级时间戳：客观判断口播是否被截断 + 该镜 ASR 文本
        base_instruction = (
            "请观看视频，对照 DNA 判定是否符合预期；镜头时长以内容表达自然为准，不要要求与目标秒数一致。"
            "参考 asr_check 里的 ASR 时间戳判断口播是否被截断；并判断每镜声音是否为真实产品口播（而非环境杂音/拍摄现场指导语）。"
            "【跨品类复刻·最高优先级】DNA 来自**另一条参考爆款**，只借它的叙事结构/节奏/镜头功能；"
            f"本片实际带货的商品是「{inputs['product_name']}」（若此处未给出明确商品名，"
            "则以**用户素材画面里实际出现的那个商品**为准），用户素材拍的就是这个商品。"
            "DNA 的 role/want 里出现的**参考商品品类名、成分、价格、喝法**都属于参考视频那个商品，"
            "**不属于本片**。严禁因为「素材不是 DNA 里提到的那个商品」而列为问题、扣分或判 material_limited；"
            "请把 DNA 的每条 want 理解成它的**叙事功能**（痛点铺垫/产品登场/配料表特写/使用演示/价格/催单），"
            "只评判本片这一镜有没有承担起该功能。"
        )
        if pure_music:
            # 空镜/纯音乐复刻：成片本就静音、无口播、无字幕，审片不得据此扣分或要求配音
            base_instruction = (
                "【本片为空镜剪辑（纯音乐参考，无人声）】成片按设计为静音、无口播解说、无字幕，"
                "复用参考 BGM 卡点。**严禁**把「无配音/无口播/全程静音/未加字幕」列为问题或据此扣分，"
                "也**不要**建议调用 tts_clone 配音。请只从画面本身评审：镜头是否贴合 DNA 卖点、"
                "剪辑节奏是否与参考一致、镜头之间是否有重复/拼凑、时长节奏是否协调。"
            )
        review_user = json.dumps({
            "dna": review_dna, "this_loop_ops": ops_brief,
            "product_name": inputs["product_name"],
            "asr_check": asr_check,
            "voice_plan": _voice_plan_brief(slots, placements, tts_by_slot),
            "mode": {"pure_music": pure_music, "空镜剪辑": pure_music,
                     "expect_speech": (not pure_music)},
            "history": [{"loop": h["loop"], "note": h.get("edit_note", ""), "problems": h.get("problems", [])} for h in history[-4:]],
            "instruction": base_instruction,
        }, ensure_ascii=False)
        if review_model == "gemini":
            review = await asyncio.to_thread(gemini_review.review_video, review_p, review_user, out_path)
            if review.get("_error"):
                # Gemini 审片失败 → 回退默认视觉模型，避免整轮卡死
                _log.warning("[%s] loop%d Gemini 审片失败(%s)，回退默认视觉模型", rid, loop, review.get("_error"))
                yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 看片",
                           f"Gemini 审片失败（{review.get('_error')}），回退默认视觉模型", state="running")
                review = await _run_llm(review_p, review_user, vision=True,
                                        media=[{"type": "video", "url": video_uri}],
                                        tag=f"[{rid}] loop{loop} 审片Agent(回退qwen)")
            else:
                _log.info("[%s] loop%d 审片Agent(gemini) output: %s", rid, loop,
                          json.dumps(review, ensure_ascii=False)[:4000])
        else:
            review = await _run_llm(review_p, review_user, vision=True,
                                    media=[{"type": "video", "url": video_uri}],
                                    tag=f"[{rid}] loop{loop} 审片Agent")
        score = float(review.get("score") or 0)
        passed = bool(review.get("pass"))
        problems = review.get("problems", []) or []
        if best is None or score > best["score"]:
            best = {"score": score, "video_uri": video_uri, "path": out_path, "loop": loop}
        _log.info("[%s] loop%d 审片Agent pass=%s score=%.0f material_limited=%s thought=%s problems=%s commands=%s",
                  rid, loop, passed, score, bool(review.get("material_limited")),
                  review.get("thought", ""),
                  json.dumps([{"slot": p.get("slot_id", ""), "issue": p.get("issue", ""), "fix": p.get("fix", "")}
                              for p in problems], ensure_ascii=False)[:2000],
                  json.dumps(review.get("commands", []), ensure_ascii=False)[:1500])
        yield step(f"review{loop}", f"第 {loop} 轮 · 审片 Agent 判定",
                   f"{'通过' if passed else '未通过'} · {score:.0f} 分 · {review.get('thought','')}",
                   observation=json.dumps(review, ensure_ascii=False)[:1500])

        history.append({"loop": loop, "edit_note": note, "ops": ops_brief,
                        "review": review.get("thought", ""), "score": score,
                        "problems": [f"{p.get('slot_id','')}: {p.get('issue','')}" for p in problems]})

        if passed:
            _log.info("[%s] agent_edit done PASS loop=%d score=%.0f video=%s", rid, loop, score, video_uri)
            _save_edit_trace(task_id, strategy_abs, video_uri=video_uri, score=score,
                             verdict=review.get("thought", "符合预期"), history=history, review_model=review_model)
            yield {"type": "agent_edit_done", "video_uri": video_uri, "final_path": out_path,
                   "loops": loop, "score": score, "material_limited": False,
                   "review_model": review_model,
                   "verdict": review.get("thought", "符合预期"), "history": history}
            return
        if review.get("material_limited"):
            _log.info("[%s] loop%d 审片判定素材受限，输出当前最佳样片", rid, loop)
            yield step("stop", "审片判定素材受限", "审片 Agent 认为再剪也只能这样，直接输出当前最佳样片", state="done")
            break
        review_feedback = {"problems": problems, "commands": review.get("commands", []),
                           "last_score": score}
        revise_slots = _flagged_slots(review, all_slot_ids)  # 下一轮只增量修订被点名的 slot
        _log.info("[%s] loop%d 下一轮增量修订 slots=%s", rid, loop, revise_slots)

    # 达到上限 / 素材受限：输出最佳样片
    b = best or {"video_uri": "", "path": "", "score": 0, "loop": max_loops}
    _log.info("[%s] agent_edit done BEST loops=%d best_loop=%s score=%s video=%s",
              rid, len(history), b.get("loop"), b.get("score"), b.get("video_uri"))
    _save_edit_trace(task_id, strategy_abs, video_uri=b["video_uri"], score=b["score"],
                     verdict="已达轮次上限/素材受限，输出历轮最佳样片", history=history, review_model=review_model)
    yield {"type": "agent_edit_done", "video_uri": b["video_uri"], "final_path": b["path"],
           "loops": len(history), "score": b["score"], "material_limited": True,
           "review_model": review_model,
           "verdict": "已达轮次上限/素材受限，输出历轮最佳样片", "history": history}
