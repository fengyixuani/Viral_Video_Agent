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
import subprocess
import sys
import tempfile
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
from editing import bgm_library
from editing import beats as beats_tool
from editing import tools as edit_tools
from editing import aigc as aigc_agent
from editing.tools import EditToolbox, ranges_overlap, normalize_speech, render_tools_spec
from tools import aigc_gen

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
                # 实际念出来的文案（与是否烧字幕无关）：字幕特效模仿要用它当字幕文本，
                # 避免退化成对成片跑 ASR（会有同音错字）。
                "spoken_text": (tts.get("text") or "").strip(),
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
            "spoken_text": caption,
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


def _mute_slots(placements, slots, tts_by_slot, toolbox, min_chars=6, min_dur=2.0):
    """需要补克隆配音的 slot：既没有可用原声、也没有配音的镜头。

    带货成片不该出现整段没人声的空档。两类都算："完全没声音"，以及**原声撑不起这一段**
    （窗口太短或只有零星几个字，如 1.9s 的现场杂音"哥四妹把脸闭住啊"）——后者听起来就是
    "这段没配音"，实测就是用户看到的那段没人说话的画面。
    """
    out = []
    for s in slots or []:
        sid = s["slot_id"]
        p = (placements or {}).get(sid)
        if not p:
            continue                              # 已被 skip_slot 跳过
        if sid in (tts_by_slot or {}):
            continue                              # 有克隆配音
        spoken = _in_window_speech(toolbox, p)
        a, b = _parse_range(p.get("source_time_range", ""))
        if len(spoken) >= min_chars and (b - a) >= min_dur:
            continue                              # 原声够长够多字，撑得起这一段
        out.append(sid)
    return out


async def _unusable_original_slots(slots, placements, toolbox, product_name=""):
    """LLM 判断哪些"原声"其实不能用（拍摄现场杂音/闲聊/ASR 乱识别）。

    ASR 会把现场杂音识别成话（实测「哥四妹把脸闭住啊」被当成口播保留下来），成片里就是
    一段听不懂的杂音。长度阈值挡不住这种，只能看内容——这里一次性把各段原声文本给模型，
    让它标出"不能作为带货成片口播"的段，交给上层改走克隆配音。失败时返回空集（不拦）。
    """
    items = []
    for s in slots or []:
        sid = s["slot_id"]
        p = (placements or {}).get(sid) or {}
        text = _norm_text(_in_window_speech(toolbox, p)) or _norm_text(p.get("caption", ""))
        if len(text) >= 3:
            items.append((sid, text))
    if not items:
        return set()
    listing = "\n".join("- {}：「{}」".format(sid, t[:60]) for sid, t in items)
    system = (
        "你在质检一条带货短视频里各镜头保留下来的**用户原声**。有些"
        "并不是真正的产品讲解，而是拍摄现场的杂音/口误/与商品无关的闲聊，或 ASR 把噪音"
        "误识别成的乱码句子（特点：语义不通、像在跟旁人说话、与带货完全无关）。"
        "把这类**不能留在成片里**的段挑出来。真正在讲产品、使用感受、痛点、活动的都算可用，"
        "口语化、有语病但意思通顺的也算可用。只输出 JSON："
        '{"unusable": ["S0x", ...], "reason": {"S0x": "一句话"}}'
    )
    user = json.dumps({"product_name": product_name or "（未指定）", "segments": listing},
                      ensure_ascii=False)
    try:
        obj = await _run_llm(system, user, tag="")
        bad = {str(x).strip() for x in (obj.get("unusable") or []) if str(x).strip()}
        reason = obj.get("reason") or {}
        valid = {sid for sid, _ in items}
        bad &= valid
        if bad:
            _log.info("原声合理性核验：%s 不可用（%s）", "、".join(sorted(bad)),
                      "；".join(f"{k}:{v}" for k, v in reason.items() if k in bad))
        return bad
    except Exception as exc:  # noqa: BLE001
        _log.warning("原声合理性核验失败(不拦)：%s", str(exc)[:160])
        return set()


def _drop_stutter(clip, toks, s, e, splits, min_pause=0.45, max_residual=5):
    """去掉口误重说的残尾：把该镜拆成两段，跳过"说了一半又重说"的那几个字。

    实测最后一段延展到完整句后是「…很累了 / 睡个 / (停 0.56s) / 睡个好觉它真的很重要」，
    连着播就是"睡个 睡个好觉"，割裂感很重。这里检测"停顿前的残尾是停顿后那句的前缀"，
    命中就在停顿处切开、丢掉残尾：成片听到「…很累了」+「睡个好觉它真的很重要」。
    ``splits`` 收集 (原 clip, 第二段 clip)，由调用方插回 clips 列表。
    """
    inside = [(a, b, t) for a, b, t in toks if s <= (a + b) / 2.0 <= e]
    if len(inside) < 3:
        return
    # 找窗口内最后一个"明显停顿"——重说通常发生在这里
    pause_i = None
    for i in range(len(inside) - 1, 0, -1):
        if inside[i][0] - inside[i - 1][1] >= min_pause:
            pause_i = i
            break
    if pause_i is None:
        return
    before = [(a, b, _norm_text(t)) for a, b, t in inside[:pause_i]]
    before_text = "".join(t for _, _, t in before)
    after_text = _norm_text("".join(t for _, _, t in inside[pause_i:]))
    if not before_text or not after_text:
        return
    # 停顿前的最后 n 个字 == 停顿后那句的开头 n 个字 → 前面那 n 个字是没说完的重说残尾
    n = 0
    for k in range(min(max_residual, len(before_text), len(after_text)), 0, -1):
        if before_text[-k:] == after_text[:k]:
            n = k
            break
    if n == 0:
        return
    # 定位残尾对应的第一个 token（从后往前累计 n 个字）
    acc, j = 0, len(before)
    while j > 0 and acc < n:
        j -= 1
        acc += len(before[j][2])
    cut_end = round(before[j][0] - 0.05, 3)
    part2_start = round(max(0.0, inside[pause_i][0] - 0.08), 3)
    if cut_end - s < 1.0 or e - part2_start < 0.8:
        return
    residual = before_text[-n:]
    kept = before_text[:-n]
    had_cap = bool(clip.get("caption_text"))
    clip["source_time_range"] = f"{s:.2f}-{cut_end:.2f}"
    clip["target_duration"] = round(cut_end - s, 3)
    part2 = dict(clip)
    part2["source_time_range"] = f"{part2_start:.2f}-{e:.2f}"
    part2["target_duration"] = round(e - part2_start, 3)
    # 两段的念白也要跟着切开：part2 只念停顿后那句，原 clip 只剩停顿前（去掉重说残尾）的话。
    # 漏了这一步，字幕特效模仿拿 spoken_text 当字幕就会把整段原话同时贴到两段上——实测成片
    # 末尾 2.84s 人在说「睡个好觉它真的很重要」，字幕却是上一句「…很累了睡个」。
    if kept:
        clip["spoken_text"] = kept
        if had_cap:
            clip["caption_text"] = kept
    part2["spoken_text"] = after_text
    part2["caption_text"] = after_text if had_cap else ""
    splits.append((clip, part2))
    _log.info("去口误重说 %s：残尾「%s」剪除，拆成 %s + %s", clip.get("slot_id"), residual,
              clip["source_time_range"], part2["source_time_range"])


def _asr_tokens_by_path(toolbox):
    """源片路径 -> 逐字 ASR {(start, end): text}。

    同一源片的逐字 ASR 要合并：单个候选下发的 items 只带自身窗口附近的上下文，合起来才是
    整条源片的字轨，才能判断"这句到哪里才算说完"。
    """
    by_path = {}
    for seg in (getattr(toolbox, "by_gid", {}) or {}).values():
        items = (seg.get("whq_speech") or {}).get("asr_items") or []
        if not items:
            continue
        bucket = by_path.setdefault(seg.get("source_path", ""), {})
        for it in items:
            try:
                bucket[(round(float(it["start"]), 3), round(float(it["end"]), 3))] = str(it.get("text") or "")
            except (TypeError, ValueError, KeyError):
                continue
    return by_path


def _sorted_toks(by_path, source_path):
    """某条源片的字轨排序成 [(start, end, text)]。"""
    return sorted((a, b, t) for (a, b), t in (by_path.get(source_path or "") or {}).items())


def _speech_cut_at(toks, start: float, end: float, gap: float = 0.8) -> bool:
    """窗口 [start, end] 里有口播、且 end 之后 gap 内话还在继续 → 话没说完就切。"""
    if not toks:
        return False
    if not any(b > start + 0.05 and a < end - 0.05 for a, b, _ in toks):
        return False           # 窗口内本来就没人说话，随便切
    return any(a < end + gap and b > end - 0.05 for a, b, _ in toks)


def _snap_clips_to_sentences(clips, toolbox, gap=0.8, extend=6.0):
    """保原声的镜头：结束点必须落在自然停顿上，别"话说一半就切"。

    优先**往后延到那句说完**（最多 extend 秒，且不超过源片长度）——素材切片边界经常把
    一句话切两半，往前回收会丢内容；延不到句尾时才退而回收到窗口内最后一个停顿。
    克隆配音镜不动（音轨是 TTS，与素材说话无关）。
    """
    by_path = _asr_tokens_by_path(toolbox)
    splits = []          # [(原 clip, 拆出的第二段)]，循环结束后插回 clips
    for c in clips or []:
        if c.get("tts_audio_path"):
            continue
        s, e = _parse_range(c.get("source_time_range", ""))
        if e <= s:
            continue
        toks = _sorted_toks(by_path, c.get("source_path", ""))
        if not toks:
            continue
        # 判定"话被切一半"：结束点落在某个字中间，或紧接结束点之后 gap 内还有字开口
        cut_mid = any(a < e < b for a, b, _ in toks)
        cont = any(e - 0.05 < a < e + gap for a, b, _ in toks)
        # 数据边界：结束点已经到了这条源片 ASR 覆盖的末尾，后面有没有字**看不到**（素材池下发
        # 的逐字 ASR 只带窗口附近若干秒）。这种情况多半正好卡在句子中间，往后放一点收尾。
        at_data_edge = bool(toks) and e >= toks[-1][1] - 0.05
        if not (cut_mid or cont or at_data_edge):
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
        # 延展后必须真的落在停顿上；若这句在 extend 秒内还没说完，就别延（改走回收分支），
        # 否则只是把"半句"换成"更长的半句"。
        ends_ok = not any(new_e - 0.05 < a < new_e + gap for a, b, _ in toks)
        if new_e <= e + 0.05 and at_data_edge and not (cut_mid or cont):
            # 到了 ASR 数据边界、延不出新的字：多留一点让这句话有收尾空间（上限 1.5s）
            pad_e = round(min(e + 1.5, limit), 3)
            if pad_e > e + 0.05:
                c["source_time_range"] = f"{s:.2f}-{pad_e:.2f}"
                c["target_duration"] = max(float(c.get("target_duration") or 0.0), pad_e - s)
            continue
        if new_e > e + 0.05 and ends_ok:
            new_e = round(min(new_e + 0.12, limit), 3)
            c["source_time_range"] = f"{s:.2f}-{new_e:.2f}"
            c["target_duration"] = max(float(c.get("target_duration") or 0.0), new_e - s)
            _drop_stutter(c, toks, s, new_e, splits)
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
    # 去口误重说时拆出来的第二段插回原位（紧跟原 clip），保持镜头顺序
    for origin, part2 in splits:
        try:
            clips.insert(clips.index(origin) + 1, part2)
        except ValueError:
            clips.append(part2)


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


SAME_SRC_MIN_GAP = float(os.getenv("AGENT_SAME_SRC_MIN_GAP", "0.5"))
# 成片级"每帧都有商品"核验：抽帧间隔（秒）与单镜最多抽帧数
PROD_VIS_STEP = float(os.getenv("AGENT_PROD_VIS_STEP", "0.4"))
PROD_VIS_MAX_FRAMES = int(os.getenv("AGENT_PROD_VIS_MAX_FRAMES", "12"))
# 避空镜时允许把镜头压到目标时长的几成、以及绝对下限（秒）：低于这个就宁可留一点空镜也不改窗口。
# 实测 S01 被压成 1.04s（目标 3.24s），开场第一镜一闪而过，比画面里少个商品刺眼得多。
PROD_VIS_MIN_KEEP = float(os.getenv("AGENT_PROD_VIS_MIN_KEEP", "0.75"))
PROD_VIS_MIN_SEC = float(os.getenv("AGENT_PROD_VIS_MIN_SEC", "1.5"))



def _same_src_gap(path: str, rng: str, other_path: str, other_rng: str):
    """同一源片时两个窗口之间的间隙秒数（<=0 表示相接或重叠）；不同源片返回 None。"""
    if not path or not other_path or _abspath(path) != _abspath(other_path):
        return None
    a0, a1 = _parse_range(rng)
    b0, b1 = _parse_range(other_rng)
    if a1 <= 0 or b1 <= 0:
        return None
    return max(a0, b0) - min(a1, b1)


def _too_close(path: str, rng: str, other_path: str, other_rng: str) -> bool:
    """同源片且两窗相接/几乎相接——观众看到的是同一个连续动作，算重复镜头。

    ranges_overlap 只判"相交"，而 0.00-1.50 与 1.50-3.50 严格不相交却是同一个上摇动作的
    前后半段，之前会被当成两段不同素材分给两个 slot（成片里就是同一个画面播两遍）。
    """
    gap = _same_src_gap(path, rng, other_path, other_rng)
    return gap is not None and gap <= SAME_SRC_MIN_GAP


def _order_same_source(clips: list) -> list:
    """同一源片被多镜使用时，让靠前的 slot 播源时间靠前的窗口。

    编排可能把 C2140 的 1.50-3.50 分给 S05、0.00-1.50 分给 S07，成片里就先播后半段再播前半段，
    同一个上摇动作被倒着放，观感既重复又跳。这里只重排"哪个窗口给哪个 slot"，不改选片本身。
    带原声或已配音的镜头跳过——它们的文本和窗口绑定，换窗口会音画不符。
    """
    by_src = {}
    for i, c in enumerate(clips or []):
        if c.get("tts_audio_path") or c.get("speech") or c.get("voice_source") == "original":
            continue
        by_src.setdefault(_abspath(c.get("source_path", "")), []).append(i)
    for _src, idxs in by_src.items():
        if len(idxs) < 2:
            continue
        ordered = sorted((clips[i].get("source_time_range", "") for i in idxs),
                         key=lambda r: _parse_range(r)[0])
        for i, rng in zip(idxs, ordered):   # idxs 天然按 clips（= slot）顺序
            if clips[i].get("source_time_range") != rng:
                _log.info("[order] %s 同源片窗口重排 %s -> %s", clips[i].get("slot_id"),
                          clips[i].get("source_time_range"), rng)
                clips[i]["source_time_range"] = rng
    return clips


_PROD_VIS_PROMPT = (
    "下面是同一条素材按时间顺序抽出的 {n} 张帧，编号 1..{n}。逐张判断：**{product}是否出现在"
    "这一帧里**。被手部小面积遮挡、只拍到局部特写都算出现；画面里根本没有这件商品（纯人脸、"
    "纯环境、纯背景板）、或商品被完全遮挡、飞出画面，都算未出现。\n"
    "只输出 JSON：{{\"visible\": [1或0, ...]}}，数组长度必须等于 {n}，顺序与编号一致。"
)


async def _clip_visible_mask(path: str, a: float, b: float, tmp_dir: str, toolbox,
                             product_name: str):
    """在 [a, b] 里按固定间隔抽帧，一次视觉核验逐帧判断商品在不在画面里。

    返回 ``([(时间点, 0/1)], 抽帧间隔)``；核验失败返回 ``(None, 0)``（不拦，宁可不动窗口）。
    """
    span = b - a
    if span <= 0.3:
        return None, 0.0
    step = max(PROD_VIS_STEP, span / PROD_VIS_MAX_FRAMES)
    times, t = [], a
    while t < b - 0.05 and len(times) < PROD_VIS_MAX_FRAMES:
        times.append(round(t, 2))
        t += step
    frames = []
    for i, ts in enumerate(times, 1):
        out = os.path.join(tmp_dir, f"v{i}_{int(ts * 100)}.jpg")
        if await asyncio.to_thread(aigc_gen.extract_frame, path, ts, out):
            frames.append((ts, out))
    if len(frames) < 2:
        return None, 0.0
    vis = []
    try:
        res = await toolbox.vlm.inspect(
            _PROD_VIS_PROMPT.format(n=len(frames), product=product_name or "目标商品"),
            targets=[{"source_path": p, "asset_id": str(i)} for i, (_, p) in enumerate(frames, 1)],
            max_targets=len(frames))
        data = as_core.parse_json(res.get("observation") or "") or {}
        vis = [1 if int(x) else 0 for x in (data.get("visible") or [])]
    except Exception as exc:  # noqa: BLE001
        _log.warning("[prodvis] %s 逐帧核验失败(不拦): %s", os.path.basename(path), str(exc)[:160])
    finally:
        for _, p in frames:      # 探针帧一次性使用
            try:
                os.remove(p)
            except OSError:
                pass
    if len(vis) != len(frames):
        return None, 0.0
    return [(ts, v) for (ts, _), v in zip(frames, vis)], step


def _pick_visible_window(mask, step: float, need: float, hard_end: float, orig_start: float):
    """从可见掩码里选一个长度尽量为 need、且离原窗口最近的连续可见区间。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1][1]:
            j += 1
        a = mask[i][0]
        # 末尾只敢延到"下一张（已判无商品）帧"的一半：延满一个 step 会把空镜的前半截带进来
        b = min(hard_end, mask[j][0] + (step if j + 1 >= n else step / 2))
        if b - a >= 0.8:
            runs.append((a, b))
        i = j + 1
    if not runs:
        return None
    # 先要够长的（够长 = 能装下目标时长），都不够长就取最长的；同等条件下离原起点最近
    long_enough = [r for r in runs if r[1] - r[0] >= need - 0.05]
    cands = long_enough or [max(runs, key=lambda r: r[1] - r[0])]
    a, b = min(cands, key=lambda r: abs(r[0] - orig_start))
    if b - a >= need:
        start = min(max(orig_start, a), b - need)
        return round(start, 2), round(start + need, 2)
    return round(a, 2), round(b, 2)


async def _drop_product_absent(rid: str, clips: list, toolbox, product_name: str = "") -> list:
    """逐帧核验**用户素材**镜头窗口里商品在不在画面里，把窗口挪到最长连续可见区间。

    AIGC 片段在 aigc.py 的 gen_video 里已经做过同样的核验，真实素材却一直没有——实测成片
    16.20-17.40s 整整 1.2s 没有商品，元凶是 S07 用的用户素材 C2144 的 2.00-3.20 本身就是
    空镜，"每帧都有商品"这条要求只落地了一半。这里在执行剪辑前补上：只挪窗口不换素材；
    带原声/配音的镜头跳过（挪窗口会音画错位）。返回改动说明，供 trace 展示。
    空镜是次要瑕疵，**把镜头砍短/把话切一半是主要瑕疵**，所以挪窗口有两条下限：
      - 新窗口短于目标时长的 PROD_VIS_MIN_KEEP（或绝对下限 PROD_VIS_MIN_SEC）→ 放弃改动；
      - 新窗口的结束点把源片里的一句话切一半 → 放弃改动（交给后面的吸句尾逻辑）。
    """
    notes = []
    if not clips or toolbox is None:
        return notes
    by_path = _asr_tokens_by_path(toolbox)
    tmp_dir = tempfile.mkdtemp(prefix="prodvis_")
    try:
        for c in clips:
            gid = str(c.get("global_asset_id") or "")
            if c.get("aigc") or gid.startswith("aigc::"):
                continue                       # 生成片段已逐帧核验过
            if c.get("tts_audio_path") or c.get("speech") or c.get("voice_source") == "original":
                continue                       # 音画绑定，窗口不能动
            path = c.get("source_path") or ""
            s, e = _parse_range(c.get("source_time_range", ""))
            need = float(c.get("target_duration") or 0.0) or max(e - s, 0.0)
            if need <= 0.3 or e <= s or not os.path.exists(path):
                continue
            vdur = _video_duration(path) or 0.0
            # 搜索范围两侧各放宽 1.5s：空镜常只占窗口一头，稍微挪一下就能避开
            a = max(0.0, s - 1.5)
            b = (min(vdur, e + 1.5) if vdur > 0 else e + 1.5)
            mask, step = await _clip_visible_mask(path, a, b, tmp_dir, toolbox, product_name)
            if not mask:
                continue
            in_win = [v for ts, v in mask if s - 0.05 <= ts <= e + 0.05]
            if in_win and all(in_win):
                continue                       # 原窗口本来就全程有商品
            win = _pick_visible_window(mask, step, need, b, s)
            if not win:
                _log.warning("[prodvis] %s %s 整段搜不到有商品的区间，保持原窗口",
                             c.get("slot_id"), os.path.basename(path))
                notes.append(f"{c.get('slot_id')} 素材内找不到有商品的区间，未改动")
                continue
            new_rng = f"{win[0]:.2f}-{win[1]:.2f}"
            if new_rng == c.get("source_time_range"):
                continue
            # 下限①：避空镜不能把镜头砍到撑不起节奏（编排时长是对齐参考爆款的）
            floor = max(need * PROD_VIS_MIN_KEEP, min(need, PROD_VIS_MIN_SEC))
            if win[1] - win[0] < floor - 0.05:
                _log.info("[prodvis] %s 有商品的区间只有 %.2fs（目标 %.2fs，下限 %.2fs），"
                          "宁可留空镜也不砍镜头，保持原窗口 %s", c.get("slot_id"),
                          win[1] - win[0], need, floor, c.get("source_time_range"))
                notes.append(f"{c.get('slot_id')} 有商品的区间只有 {win[1] - win[0]:.2f}s"
                             f"（目标 {need:.2f}s），保持原窗口不砍镜头")
                continue
            # 下限②：新窗口的结束点不能把源片里的一句话切一半
            if _speech_cut_at(_sorted_toks(by_path, path), win[0], win[1]):
                _log.info("[prodvis] %s 新窗口 %s 会把话切一半，保持原窗口 %s",
                          c.get("slot_id"), new_rng, c.get("source_time_range"))
                notes.append(f"{c.get('slot_id')} 避空镜的窗口 {new_rng} 会把口播切一半，未改动")
                continue
            _log.info("[prodvis] %s 窗口内有空镜（掩码 %s），%s -> %s", c.get("slot_id"),
                      "".join(str(v) for _, v in mask), c.get("source_time_range"), new_rng)
            notes.append(f"{c.get('slot_id')} 避开空镜：{c.get('source_time_range')} → {new_rng}")
            c["source_time_range"] = new_rng
            c["target_duration"] = round(min(need, win[1] - win[0]), 2)
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    return notes


def _fallback_for_unreachable(toolbox, meta: dict, used_paths: set, target_dur: float):
    """给"意图不可达"的 AIGC 镜位找一个用户素材顶上；找不到返回 None。

    回看判定意图不可达 = 这一镜要的东西凭本商品的画面拍不出来（缺真人试穿、缺对比等），
    生成出来的往往是"手在静止商品旁悬停"这类近似空镜，拉满目标时长后在成片里非常突兀。
    与其硬塞，不如换一段真实素材——内容会偏离原意图，但至少是能看的画面。
    """
    if toolbox is None:
        return None
    query = meta.get("want") or meta.get("role") or ""
    try:
        res = toolbox.retrieve(query, top_k=12)
    except Exception:  # noqa: BLE001
        return None
    cand = next((x for x in res.get("candidates", [])
                 if x.get("source_path") and _abspath(x["source_path"]) not in used_paths), None)
    if not cand:
        return None
    speech = cand.get("speech", "") or ""
    return {"global_asset_id": cand.get("global_asset_id", ""),
            "source_path": cand["source_path"],
            "source_time_range": cand.get("source_time_range", ""),
            "target_duration": target_dur,
            "caption": speech, "speech": speech,
            "burn_caption": bool(speech), "speed": 1.0,
            "visual_description": cand.get("visual_description") or cand.get("summary", "")}


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
        dup = any(ranges_overlap(path, rng, up, ur) or _too_close(path, rng, up, ur)
                  for up, ur in used)
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


async def _asr_window(rid: str, path: str, segs: list, want_sec: float = None):
    """从长口播里截一小段做参考音，并**对这一小段单独转写**，返回 (start, end, text)。

    zero-shot 克隆要的是「几秒参考音 + 与它严格对应的转写」。整条 24s 的口播当参考音又长又
    贵，而按比例切分文本又对不准；所以截完窗口就地再跑一次 ASR，让转写天然对应这段音频。
    """
    want = float(want_sec or float(os.getenv("AGENT_TTS_REF_SEC", "8")))
    start = 0.0
    for g in segs or []:
        try:
            start = max(0.0, float(g.get("start")) )
            break
        except (TypeError, ValueError):
            continue
    tmp_wav = os.path.join(tempfile.gettempdir(), f"refvoice_{int(time.time()*1000)%1000000}.wav")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}",
           "-t", f"{want:.2f}", "-i", path, "-vn", "-ac", "1", "-ar", "16000", tmp_wav]
    try:
        r = await asyncio.to_thread(subprocess.run, cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=120)
        if r.returncode != 0 or not os.path.isfile(tmp_wav):
            return None
        from tools.asr import ASRTool

        res = await asyncio.to_thread(ASRTool().transcribe, tmp_wav, True)
        text = _norm_text(res.get("text", ""))
        if res.get("error") or len(text) < 8:
            _log.warning("[%s] 参考音窗口转写不可用：%s", rid, res.get("error") or text[:30])
            return None
        return (round(start, 2), round(start + want, 2), text)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] 参考音窗口转写失败：%s", rid, str(exc)[:160])
        return None
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


async def _reference_voice_ref(rid: str, reference_video: str):
    """退路：用**参考爆款视频**里的人声当克隆音色参考。

    用户素材是纯拍摄画面、只有现场杂音时（实测那套鞋素材整段 ASR 只有「走。」「好。」），
    池子里挑不出可靠参考音，克隆必然跑偏。参考视频本身是带口播的爆款，它的 ASR 分句既长
    又准，拿它当 zero-shot 的「参考音 + 转写」至少能念对文案。代价是音色不是用户本人的声音——
    只在没有更好选择时用，并在前端明确标注。
    """
    path = _abspath(reference_video or "")
    if not path or not os.path.isfile(path):
        return None
    asr = asr_cache.get(path)
    if not asr:
        try:
            from tools.asr import ASRTool

            res = await asyncio.to_thread(ASRTool().transcribe, path, True)
            if not res.get("error"):
                asr_cache.set(path, res)
                asr = res
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] 参考视频 ASR 失败：%s", rid, str(exc)[:160])
    best = None
    segs = (asr or {}).get("segments") or []
    for g in segs:
        try:
            s, e = float(g.get("start")), float(g.get("end"))
        except (TypeError, ValueError):
            continue
        text = _norm_text(g.get("text", ""))
        if len(text) < 8 or not 1.5 <= (e - s) <= 20.0:
            continue
        if best is None or len(text) > len(best[2]):
            best = (s, e, text)
    if best is None:
        # 没有句级时间戳（实测参考视频整条口播被 ASR 归成一个 23.7s 的段）：截一小段
        # 再单独转写一次，保证「参考音」和「转写」严格对应——错开一点点模型就会跑偏。
        best = await _asr_window(rid, path, segs)
    if not best:
        return None
    s, e, text = best
    _log.info("[%s] 用参考视频人声做音色参考：%.2f-%.2f「%s」", rid, s, e, text[:40])
    return {"global_asset_id": "reference_video", "source_path": path,
            "source_time_range": f"{s:.2f}-{e:.2f}", "speech": text, "from_reference": True}



async def _pick_voice_ref(rid: str, toolbox, product_name: str = "", reference_video: str = ""):
    """全片统一的克隆音色参考：挑一段**转写可靠、真人在讲产品**的原声。

    zero-shot 克隆是把「参考音 + 它的转写」一起喂给 VoxCPM 的；一旦转写和音频对不上，
    模型就不念给定文案、而是顺着参考音胡念。实测踩到过：Agent 挑了 C2143 4.50-9.50 当音色
    参考，那段 ASR 是「你一兔充电 好」（现场杂音误识别），生成出来的 wav 用 ASR 回读是
    「跟去，不去。」——成片里就是一串听不懂的怪声，等于没配音。
    所以参考音不能交给 Agent 随便挑：这里全 run 只挑一次、全片共用（音色也统一），
    与 whq 结构级复刻 select_prompt 的做法一致。挑不到返回 None（退回 Agent 指定的参考）。
    """
    cands = []
    for gid, seg in (getattr(toolbox, "by_gid", {}) or {}).items():
        text = _norm_text(seg.get("speech_or_text", "") or
                          ((seg.get("whq_speech") or {}).get("text", "")))
        a, b = _parse_range(seg.get("source_time_range", ""))
        if len(text) >= 8 and 1.5 <= (b - a) <= 20.0 and seg.get("source_path"):
            cands.append((gid, seg, text))
    if not cands:
        return await _reference_voice_ref(rid, reference_video)
    cands.sort(key=lambda x: -len(x[2]))
    cands = cands[:8]
    pick = 1
    if len(cands) > 1:
        listing = "\n".join("{}. 「{}」".format(i, t[:60]) for i, (_g, _s, t) in enumerate(cands, 1))
        system = (
            "下面是同一条带货视频素材里几段原声的 ASR 转写。要挑一段作为**声音克隆的参考音**，"
            "标准是：真人在正常说话讲产品、语义通顺、**转写看起来准确**（不是把现场杂音/噪音"
            "误识别成的乱码短句）。转写不准的段会让克隆模型跑偏、念出不相干的话，必须排除。"
            "**如果这些段没有一个像正常人话（全是杂音误识别），best 就填 0**。"
            '只输出 JSON：{"best": 序号或0, "reason": "一句话"}'
        )
        try:
            obj = await _run_llm(system, json.dumps(
                {"product_name": product_name or "（未指定）", "segments": listing},
                ensure_ascii=False), tag="")
            n = int(obj.get("best") or 0)
            if n == 0:
                _log.info("[%s] 素材原声全是杂音误识别（%s），改用参考视频人声", rid,
                          str(obj.get("reason", ""))[:60])
                return await _reference_voice_ref(rid, reference_video)
            if 1 <= n <= len(cands):
                pick = n
            _log.info("[%s] 克隆音色参考：第%d段「%s」（%s）", rid, pick, cands[pick - 1][2][:30],
                      str(obj.get("reason", ""))[:60])
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] 音色参考挑选失败，用最长的一段：%s", rid, str(exc)[:120])
    gid, seg, text = cands[pick - 1]
    return {"global_asset_id": gid, "source_path": seg.get("source_path", ""),
            "source_time_range": seg.get("source_time_range", ""), "speech": text}


@edit_tools.edit_tool_handler("tts_clone")
async def _handle_tts_clone(ctx: dict, action: dict) -> dict:
    """tts_clone 工具执行器：用参考片段音色把改写文案念出来，产出 wav 记到该 slot。"""
    slot_id = action.get("slot_id", "")
    ref_gid = action.get("ref_global_asset_id", "")
    text = (action.get("text") or "").strip()
    if not ctx.get("enable_tts", True):
        return {"ok": False, "error": "本次已关闭 TTS 配音，不能使用 tts_clone；空镜保留原声/静音即可"}
    # 配音服务连续失败（如显存不足）时短路：否则 Agent 会一直换文案重试，每次都要几十秒，
    # 实测能空转 20 次以上。直接告诉它别再调，把剩下的事情做完。
    if int(ctx.get("tts_fail", 0)) >= 3:
        return {"ok": False, "error": (
            "配音服务当前不可用（已连续失败 {} 次，多为 GPU 显存不足）。**本轮不要再调用 tts_clone**："
            "有原声的镜头用 place_original 保留真声，其余 slot 处理完直接输出 finish。"
        ).format(int(ctx.get("tts_fail", 0)))}
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
    # 拦不拦的三条依据（实测漏拦的两种情况都在这里）：
    #   1) 窗口里有**成句**真实口播（≥ AGENT_LIPSYNC_HARD_CHARS 字）-> 一律保原声，
    #      连"原声不可用"白名单也不许覆盖。实测 S05 窗口里人在说「小分子柠檬酸特工负责钻…」，
    #      因为编排把它判成 clone 就进了白名单，配上「成分温和不刺激」后口型完全不对。
    #   2) 编排已判「有人脸(N/N 帧, 口型需对上) -> 保留用户原声」(voice_source=original)
    #      -> 不许再拿克隆配音覆盖。原来这一项是**跳过检查**的前置条件，最该拦的反而放过了：
    #      实测 S02 编排写着保原声（原声「该抛光了」），Agent 主动 tts_clone 就没人拦。
    #   3) 窗口里有零星原声（≥3 字）且没被判定"原声不可用" -> 保原声。
    _spoken_n = len(_norm_text(_spoken))
    _tts_allowed = slot_id in (ctx.get("allow_tts_slots") or ())
    _hard_n = int(os.getenv("AGENT_LIPSYNC_HARD_CHARS", "8"))
    # 例外：原声是**拍摄现场的废话**（"行行行行往上走这个都够了"、"OK然后捏一捏那个泡沫"）时
    # 不拦。这种段字数够、也真有人在说，但保留它等于成片里放一段现场口令 —— 用户明确说过
    # "原声是没有用的啊 都是废话"。宁可口型略有出入，也不能把废话留在成片里。
    _filler, _filler_why = edit_tools.is_filler_speech(_spoken) if _spoken_n else (False, "")
    if not _filler and _spoken_n:
        # 文本规则没兜住的离机位口令（如「各位姐妹儿直接闭眼冲啊」）：字面像真实口播，
        # 但取窗内电平极低、成片里听不见，保原声等于"有字幕没声音"。按音量放行克隆配音。
        _filler, _filler_why = edit_tools.is_offmic_quiet(
            _p.get("source_path", ""), _p.get("source_time_range", ""))
    if _filler:
        _log.info("%s 原声是现场废话（%s），允许克隆配音：「%s」",
                  slot_id, _filler_why, _norm_text(_spoken)[:30])
    elif _spoken_n >= _hard_n or (not _tts_allowed
                                  and (_p.get("voice_source") == "original" or _spoken_n >= 3)):
        return {"ok": False, "error": (
            "这一镜的素材里人正在说话（原声：「{}」），换成克隆配音会导致**口型和话对不上**。"
            "请改用 place_original（同一个 global_asset_id={}，不换素材）保留用户原声；"
            "确实要换掉这段说话画面，就先 place 一个没有人说话的片段再配音。"
        ).format(_norm_text(_spoken)[:30] or "画面里的人在说话", _p.get("global_asset_id", ""))}

    # 同一 slot 重复提交同一条文案 = 空转（每次 TTS 要几十秒）。直接拦住并提示下一步。
    prev = (ctx.get("tts_by_slot") or {}).get(slot_id) or {}
    if prev.get("text", "").strip() == text and prev.get("audio_path"):
        return {"ok": False, "error": (
            f"{slot_id} 已经用这条文案配过音了（不要重复提交同一条）。若还有别的 slot 没配音就去配，"
            "都配完了就输出 finish；若想改这一镜的文案，请给一条**不同的** text。")}
    # 字数硬约束：文案过长 -> TTS 比画面长得多，成片只能冻结末帧补足（画面静止数秒）。
    # 上限 = ref_cps(参考该段语速) × 该镜时长；没有 ref_cps 时按 5 字/秒兜底。
    # 时长优先取**本镜实际放入的片段时长**：AIGC 补镜片段常比 slot 目标短（回看只截了商品
    # 全程可见的那一段，2.4s 目标能只剩 1.6s），按 slot 目标算上限会放过念不完的文案。
    meta = ctx.get("slot_meta", {}).get(slot_id) or {}
    _pl = (ctx.get("placements") or {}).get(slot_id) or {}
    target = float(_pl.get("target_duration") or meta.get("target_duration") or 0.0)
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
    # 音色参考用全片统一挑好的那一段（见 _pick_voice_ref），不用 Agent 随手指定的：
    # 参考音的转写一旦不准（现场杂音被 ASR 识别成短句），克隆模型会不念文案、顺着参考音胡念。
    ref = ctx.get("voice_ref") or {}
    if ref.get("source_path") and ref.get("speech"):
        if ref.get("global_asset_id") != ref_gid:
            _log.info("[tts] %s 音色参考换成全片统一的 %s（Agent 指定的是 %s）", slot_id,
                      ref.get("global_asset_id"), ref_gid)
        ref_path, ref_rng, ref_text = ref["source_path"], ref.get("source_time_range", ""), ref["speech"]
    else:
        ref_path, ref_rng = seg.get("source_path", ""), seg.get("source_time_range", "")
        ref_text = seg.get("speech_or_text", "")
    res = await asyncio.to_thread(edit_tts.clone, ref_path, ref_rng, ref_text, text, out_wav)
    if not res.get("ok"):
        ctx["tts_fail"] = int(ctx.get("tts_fail", 0)) + 1
        return {"ok": False, "error": res.get("error", "TTS 失败")}
    ctx["tts_fail"] = 0
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


async def _fill_missing_voice(rid, slots, placements, tts_by_slot, toolbox, voice_ref,
                             product_name="", enable_tts=True):
    """兜底补齐剩余无声镜头的配音：**一次 LLM 写完所有缺口文案，再逐条合成**，不走 Agent。

    只配音那一轮和哑巴补配音那一轮都靠剪辑 Agent 自己迭代，实测它配了前几镜就输出 finish
    （142238 那次 S01–S06 有配音、S07/S08 全程无声，用户听到的就是"后半段没口播"）。
    这里不再指望 Agent 把清单走完：客观算出还缺哪些镜，一次性把这几条文案写出来（同一次调用
    里看到全片已定的口播，衔接和去重都比逐条更好），然后逐条克隆合成。返回新增的 slot 列表。
    """
    if not enable_tts:
        return []
    missing = [s for s in (slots or [])
               if s["slot_id"] in (placements or {}) and s["slot_id"] not in (tts_by_slot or {})
               and not _norm_text(_in_window_speech(toolbox, placements[s["slot_id"]]))]
    if not missing:
        return []
    # 全片已定的口播（原声 + 已配音），按镜头顺序给模型，让缺口文案接得上、不重复
    timeline = []
    for s in slots or []:
        sid = s["slot_id"]
        p = (placements or {}).get(sid) or {}
        if sid in (tts_by_slot or {}):
            timeline.append({"slot_id": sid, "voice": (tts_by_slot[sid] or {}).get("text", ""), "fixed": True})
        elif _norm_text(_in_window_speech(toolbox, p)):
            timeline.append({"slot_id": sid, "voice": _in_window_speech(toolbox, p), "fixed": True})
        else:
            dur = float(p.get("target_duration") or s.get("target_duration") or 3.0)
            cps = float((s.get("whq_voice") or {}).get("ref_cps") or 0) or 5.0
            timeline.append({"slot_id": sid, "role": s.get("role", ""),
                             "material": _material_brief(toolbox, p.get("global_asset_id"), placement=p),
                             "max_chars": max(6, int(dur * cps)), "fixed": False})
    system = (
        "你在给一条带货短视频补口播。timeline 是按镜头顺序排好的全片口播：fixed=true 的已经定了"
        "（用户原声或已配好的音），**不要改**；fixed=false 的这几镜现在完全没声音，请为它们各写"
        "一句口播。要求：只讲该镜 material（这一镜的实际画面）里真实有的东西，不许编造价格/活动/"
        "成分；每条不超过该镜 max_chars 字；和 timeline 里已有的说法不重复；读下来整条片子要连贯"
        "（承接上一镜、给下一镜留话头，最后一镜收口引导下单）。"
        '只输出 JSON：{"lines": {"S0x": "文案", ...}}'
    )
    try:
        obj = await _run_llm(system, json.dumps(
            {"product_name": product_name or "（未指定）", "timeline": timeline}, ensure_ascii=False), tag="")
        lines = {str(k): str(v).strip() for k, v in (obj.get("lines") or {}).items() if str(v).strip()}
    except Exception as exc:  # noqa: BLE001
        _log.warning("[%s] 补配音文案生成失败：%s", rid, str(exc)[:160])
        return []
    tts_dir = os.path.join(AGENT_ROOT, "uploads", "tts")
    os.makedirs(tts_dir, exist_ok=True)
    done = []
    for s in missing:
        sid = s["slot_id"]
        text = lines.get(sid, "")
        if not text:
            continue
        ref = voice_ref or {}
        if not (ref.get("source_path") and ref.get("speech")):
            _log.warning("[%s] 补配音缺音色参考，跳过 %s", rid, sid)
            break
        out_wav = os.path.join(tts_dir, f"tts_{sid}_{int(time.time() * 1000) % 1000000}.wav")
        res = await asyncio.to_thread(edit_tts.clone, ref["source_path"],
                                      ref.get("source_time_range", ""), ref["speech"], text, out_wav)
        if not res.get("ok"):
            _log.warning("[%s] 补配音 %s 合成失败：%s", rid, sid, str(res.get("error"))[:160])
            continue
        tts_by_slot[sid] = {"audio_path": out_wav, "text": text,
                            "duration": res.get("duration", 0.0)}
        done.append(sid)
        _log.info("[%s] 补配音 %s（%.1fs）：%s", rid, sid, float(res.get("duration") or 0), text)
    return done


def _material_brief(toolbox, gid, limit=160, placement=None):

    """某片段的实际内容摘要：画面描述 +（若有）片段自带口播。

    写 tts_clone 文案时必须**只讲这段素材里真实有的东西**——DNA 的 role/want 来自另一条
    参考爆款，照它写会把参考商品的成分/喝法/价格搬到本商品头上（whq 的老坑）。

    gid 查不到时（编排给出的是 ``cta_original`` / ``aigc::`` 这类**合成 id**，不在素材池里）
    按 placement 的源片 + 时间窗反查素材池，再退回 placement 自带的画面描述（AIGC 生成的
    mp4 根本不在池里，反查也查不到，只有补镜 Agent 带过来的描述可用），最后退回该镜口播。
    曾经这里直接返回空串，剪辑 Agent 看到 material 为空就按"素材撑不起这一段"把整个节拍
    skip 掉（CTA 段消失、AIGC 补的镜头被丢弃）。
    """
    by_gid = getattr(toolbox, "by_gid", {}) or {}
    seg = by_gid.get(gid) or {}
    if not seg and placement:
        seg = _seg_by_window(by_gid, placement.get("source_path", ""),
                             placement.get("source_time_range", ""))
    if not seg:
        pl = placement or {}
        desc = (pl.get("visual_description") or "")[:limit]
        speech = (pl.get("speech") or pl.get("caption") or "")[:80]
        if desc:
            return desc + ("｜该片段原声：" + speech if speech else "")
        return ("（无画面描述）｜该片段原声：" + speech) if speech else ""
    desc = (seg.get("visual_description") or seg.get("one_sentence_summary") or "")[:limit]
    speech = ((seg.get("whq_speech") or {}).get("text") or seg.get("speech_or_text") or "")[:80]
    return desc + ("｜该片段原声：" + speech if speech else "")


def _seg_by_window(by_gid, source_path, source_time_range):
    """按 (源片, 时间窗重叠最大) 在素材池里找那一段; 找不到返回 {}。"""
    if not source_path:
        return {}
    a, b = _parse_range(source_time_range)
    src = _abspath(source_path)
    best, best_ov = {}, 0.0
    for seg in by_gid.values():
        if _abspath(seg.get("source_path", "")) != src:
            continue
        s, e = _parse_range(seg.get("source_time_range", ""))
        ov = min(b, e) - max(a, s) if b > a and e > s else 0.0
        if ov > best_ov:
            best, best_ov = seg, ov
    return best


async def _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_system,
                          base_placements=None, base_tts=None, revise_slots=None,
                          enable_tts=True, pure_music=False, voice_only_slots=None,
                          fact_corpora=None, voice_ref=None):
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
    # 扩展工具共享的执行上下文（跨步复用，用于累积 tts 连续失败计数等状态）
    tool_ctx: dict = {}

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
            if ranges_overlap(path, rng, u["source_path"], u["range"]) \
                    or _too_close(path, rng, u["source_path"], u["range"]):
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
                                     "material": _material_brief(toolbox, v.get("global_asset_id"),
                                                                 placement=v),
                                     "tts": k in tts_by_slot} for k, v in placements.items()],
            "slots_to_revise": unfilled,
            "recent_steps": scratch[-8:],
            "instruction": (" ".join(mode_bits) + " " if mode_bits else "") + (
                ("【只配音模式】所有镜头的**画面已由编排定好**（见 baseline_placements），"
                 "**不要**用 retrieve 重新召回、也不要 place 换别的素材改动画面。"
                 "slots_to_revise 里这些 slot 的**原声已判定不可用**（太短撑不起这一段，或只是"
                 "现场杂音/ASR 乱识别的话），所以**默认对它们逐个 tts_clone 配音**。"
                 "但**口型优先**：若 tts_clone 被拒绝并提示「素材里人正在说话」，说明该镜窗口里"
                 "有成句真实口播、画面里的人在说这句话——这时改用 place_original（同一个 gid、"
                 "不换素材）保住原声，不要换文案硬配音。"
                 "反过来，若 place_original 被拒绝并提示「原声是拍摄现场的废话」（现场口令/"
                 "口水话，如「行行行往上走这个都够了」），就照常 tts_clone 配音——那种原声"
                 "没有信息量，留在成片里比口型略有出入更糟。"
                 "写 text 的依据是"
                 "**该 slot 的 material（这一镜的实际画面内容）**——只讲这段素材画面里真实有的东西；"
                 "DNA 的 role/want 来自另一条参考爆款，**只用来判断这一段承担什么叙事功能**"
                 "（痛点/产品登场/配料特写/使用演示/价格/催单），里面的商品名/成分/含量/价格/喝法"
                 "一律不许写进文案。"
                 "**整片口播要当成一条连贯脚本来写**：narrative 是按镜头顺序排好的全片口播"
                 "（含保留用户原声那些镜的原话），按 slot_id 从小到大依次配音，每条都要承接上一镜"
                 "说完的话、给下一镜留出话头；同一个卖点全片只讲一次，别各镜各说一套、也别"
                 "重复上一镜的说法；最后一镜要收口（引导下单/总结），不要说半句就断。"
                 "字数不超过 whq_voice.ref_cps × 该镜时长（配音比画面长会导致画面卡住不动）。"
                 "若某个 slot 的画面内容与它要承担的叙事功能完全对不上（如参考要求"
                 "「改数量下单演示」但素材里没有任何下单/价格画面），用 skip_slot 跳过该段，"
                 "**不要编造素材里没有的价格/活动/成分**。"
                 "结束前逐个确认 slots_to_revise 里每个 slot 要么已配上音、要么已 skip_slot——"
                 "**不要留下既没原声又没配音的哑巴段**。"
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
                # 注意 ctx 用同一个 dict（tool_ctx）跨步复用：像 tts 连续失败计数这类状态
                # 需要在多步之间累积，每步新建会丢掉，导致 Agent 反复重试已经不可用的服务。
                ctx = tool_ctx
                ctx.update({"toolbox": toolbox, "used": used, "placements": placements,
                            "slot_meta": slot_meta, "notes": notes, "tts_by_slot": tts_by_slot,
                            "enable_tts": enable_tts, "pure_music": pure_music,
                            # 放片类扩展工具（如 whq_clone 的 place_original）需要这两个才能
                            # 把 slot 标记为已填 + 参与重叠检测
                            "unfilled": unfilled, "register": _register,
                            # (用户素材语料, 参考爆款语料)：tts_clone 用它核验文案没抄参考商品的事实
                            "fact_corpora": fact_corpora,
                            # 全片统一的克隆音色参考（转写可靠的一段），见 _pick_voice_ref
                            "voice_ref": voice_ref,
                            # 这些 slot 的"原声"撑不起该段（太短/只有零星杂音），允许克隆配音
                            # 覆盖它，不受"有人说话就必须保原声"的口型护栏限制
                            "allow_tts_slots": set(voice_only_slots or ())})
                try:
                    obs = await handler(ctx, action)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("%s tool '%s' handler failed: %s", tag, act, exc)
                    obs = {"ok": False, "error": str(exc)[:200]}
                scratch.append({"action": str(act), "observation": obs if isinstance(obs, dict) else {"ok": True, "result": obs}})
            else:
                scratch.append({"action": str(act), "observation": {"ok": False, "error": "未知动作，请用 retrieve/verify/place/finish"}})

    yield {"__edit_result__": True, "placements": placements, "notes": notes,
           "tts_by_slot": tts_by_slot, "tts_broken": int(tool_ctx.get("tts_fail", 0)) >= 3}


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


def _caption_items(result: dict, clips: list) -> list:
    """成片时间轴上的「每镜实际念白」[{slot_id,start,end,text}]。

    给「字幕特效模仿」当字幕文本用：那边默认对成片跑 ASR 取文本，会带同音错字（实测把
    「侧颜」听成「侧眼」），而这里的文案是写出来的字，是准的。时间轴按 editor 实际用的
    每镜时长累加（ops 的 target_duration 就是实际截取长度，顺序与 concat 一致）。

    一个 slot 可能对应**多个 clip**（``_drop_stutter`` 去口误重说时会把一镜拆成两段），所以
    按 slot 排队逐个取，不能用 {slot_id: clip} —— 那样两段会共用同一条文案，末尾那段的字幕
    就变成上一句的话。
    """
    by_slot = {}
    for c in clips or []:
        by_slot.setdefault(c.get("slot_id"), []).append(c)
    items, t = [], 0.0
    for op in result.get("ops", []) or []:
        if op.get("op") != "trim" or not op.get("ok"):
            continue
        dur = float(op.get("target_duration") or 0.0)
        if dur <= 0:
            continue
        bucket = by_slot.get(op.get("slot_id")) or []
        c = bucket.pop(0) if bucket else {}
        text = str(c.get("spoken_text") or op.get("caption") or "").strip()
        if text and c and not c.get("tts_audio_path"):
            # 原声镜头：取窗电平低到听不见的（离机位口令/环境人声），字幕也不出——
            # 否则成片"有字幕没声音"（实测 S04「各位姐妹儿直接闭眼冲啊」峰值 -29dB）。
            _quiet, _qwhy = edit_tools.is_offmic_quiet(c.get("source_path", ""),
                                                       c.get("source_time_range", ""))
            if _quiet:
                _log.info("%s 原声听不见（%s），该镜不出字幕", op.get("slot_id"), _qwhy)
                text = ""
        if text:
            items.append({"slot_id": op.get("slot_id"), "start": round(t, 3),
                          "end": round(t + dur, 3), "text": text})
        t = round(t + dur, 3)
    return items


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
                # 已被基线占用的源片，换素材时避开，免得又撞重复
                used_paths = {_abspath(p.get("source_path", ""))
                              for p in (inputs.get("presets") or {}).values()
                              if p.get("source_path")}
                for asid, clip in clips_by_slot.items():
                    m = by_meta.get(asid, {})
                    target_dur = clip.get("target_duration", m.get("target_duration", 3.0))
                    slots.append({"slot_id": asid, "role": m.get("role", ""), "want": m.get("want", ""),
                                  "breakdown": m.get("breakdown", []), "target_duration": m.get("target_duration", 3.0)})
                    # 只有画面本身坏了（形变/模糊/主体消失）才换真实素材；意图对不上但画面可用的
                    # 保留生成片段——否则参考爆款一旦要脚/要对比，AIGC 会被全部替换掉，白跑一场
                    if clip.get("unusable"):
                        alt = _fallback_for_unreachable(toolbox, m, used_paths, target_dur)
                        if alt:
                            _log.info("[%s] %s AIGC 画面不可用，改用素材 %s %s", rid, asid,
                                      os.path.basename(alt["source_path"]), alt["source_time_range"])
                            aigc_presets[asid] = alt
                            used_paths.add(_abspath(alt["source_path"]))
                            continue
                        _log.warning("[%s] %s AIGC 画面不可用但池里没有可替代素材，仍用生成片段",
                                     rid, asid)
                    aigc_presets[asid] = {
                        "global_asset_id": f"aigc::{asid}", "source_path": clip["source_path"],
                        "source_time_range": clip["source_time_range"],
                        "target_duration": clip.get("target_duration", m.get("target_duration", 3.0)),
                        "caption": clip.get("caption", ""), "speech": "",
                        "burn_caption": bool(clip.get("caption")), "speed": clip.get("speed", 1.0),
                        # 生成片段天生没有人声（seedance 出的是无声画面），只能走克隆配音：
                        # 标成 clone 后和 whq 复刻判定"该配音"的镜头走同一条只配音链路
                        "voice_source": "clone",
                        # 生成片段不在素材池里，画面描述只能由补镜 Agent 带过来（见 _material_brief）
                        "visual_description": clip.get("visual_description", ""),
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
    # BGM 取谁：勾了「复用参考 BGM」就**直接用参考视频自己的伴奏**（最贴参考，不必再去曲库挑一首
    # 像的，也省掉一次选曲+下载）；没勾、或扒不出来，才走美摄资产库按参考配乐风格选曲。
    bgm, bgm_beats = "", []
    # 成片下方要展示"这条片子的 BGM 是怎么来的"，所以选取过程留一份摘要跟着 done 事件回前端
    bgm_info = {"source": "未加 BGM", "name": "", "bpm": 0, "reason": "", "beats": 0}
    if enable_bgm and reuse_bgm and reference_video:
        yield step("bgm-reuse", "复用参考 BGM", "直接用参考视频自己的 BGM（分离/抽取其伴奏）",
                   state="running")
        bg = await asyncio.to_thread(bgm_reuse.extract_reference_bgm, reference_video, pure_music=pure_music)
        if bg.get("ok"):
            bgm = bg["output"]
            bgm_info = {"source": "参考视频原伴奏", "name": os.path.basename(bgm), "bpm": 0,
                        "reason": "复用参考 BGM（{}）".format(bg.get("mode") or ""), "beats": 0}
            yield step("bgm-reuse", "复用参考 BGM", f"已取得参考 BGM（{bg.get('mode')}），成片直接用它作配乐", state="done")
        else:
            bgm_info["reason"] = "复用参考 BGM 失败：{}".format(bg.get("error", ""))
            yield step("bgm-reuse", "复用参考 BGM", f"失败：{bg.get('error','')}，改走美摄曲库选曲", state="done")
    # 曲库选曲的依据分两档：参考视频的音频（最贴参考）＞剪辑后的成片内容。参考没配乐信息
    # （无音频/纯口播/分析失败）时不能预先选，延后到首轮成片出来后按成片内容匹配。
    bgm_defer_lib = False
    if enable_bgm and not bgm:
        ref_music_ok = bool(reference_video) and await asyncio.to_thread(
            bgm_library.reference_music_available, reference_video)
        if ref_music_ok:
            yield step("bgm-lib", "美摄曲库选曲", "按参考视频的配乐风格从美摄 BGM 资产库里挑一首",
                       state="running")
            pick = await bgm_library.choose_bgm(reference_video, inputs["product_name"])
            if pick.get("ok"):
                bgm, bgm_beats = pick["path"], pick.get("beats") or []
                bgm_info = {"source": "美摄曲库（对齐参考配乐）",
                            "name": pick["name"], "bpm": round(pick.get("bpm") or 0, 1),
                            "reason": pick.get("reason") or "", "beats": 0}
                yield step("bgm-lib", "美摄曲库选曲",
                           "选中《{}》（bpm≈{:.0f}，{} 个拍点，对齐参考配乐）：{}".format(
                               pick["name"], pick.get("bpm") or 0, len(bgm_beats),
                               pick.get("reason") or ""), state="done")
            else:
                bgm = DEFAULT_BGM if os.path.isfile(DEFAULT_BGM) else ""
                prev = bgm_info.get("reason") or ""
                bgm_info = {"source": "本地默认 BGM" if bgm else "未加 BGM",
                            "name": os.path.basename(bgm) if bgm else "", "bpm": 0,
                            "reason": "；".join(x for x in (prev, "曲库选曲失败：{}".format(
                                pick.get("error", ""))) if x), "beats": 0}
                yield step("bgm-lib", "美摄曲库选曲",
                           "失败：{}；{}".format(pick.get("error", ""),
                                              "回退本地默认 BGM" if bgm else "本地默认 BGM 也不存在，本次不加 BGM"),
                           state="done")
        else:
            bgm_defer_lib = True
            yield step("bgm-lib", "美摄曲库选曲",
                       "参考视频没有可用的配乐信息（无音频/纯口播/分析失败），"
                       "待首轮成片生成后按成片内容匹配选曲", state="done")
    # 卡点剪辑：曲库自带逐拍时间就直接用；参考原伴奏没有拍点数据，跑 librosa 检测
    beats = []
    if beat_sync and bgm and bgm_beats:
        beats = bgm_beats
        yield step("beats", "BGM 鼓点分析",
                   f"美摄曲库自带 {len(beats)} 个拍点，直接用于卡点（不必再跑节拍检测）", state="done")
    elif beat_sync and bgm:
        yield step("beats", "BGM 鼓点分析", "检测 BGM 节拍点用于卡点剪辑", state="running")
        bt = await asyncio.to_thread(beats_tool.detect_beats, bgm)
        if bt.get("ok") and bt.get("beats"):
            beats = bt["beats"]
            yield step("beats", "BGM 鼓点分析", f"检测到 {len(beats)} 个鼓点（tempo≈{round(bt.get('tempo') or 0)}），成片将卡点剪辑", state="done")
        else:
            yield step("beats", "BGM 鼓点分析", f"未能检测鼓点：{bt.get('error','')}，本次不卡点", state="done")
    elif beat_sync and not bgm:
        yield step("beats", "BGM 鼓点分析",
                   "曲库选曲延后到首轮成片后，首轮不卡点" if bgm_defer_lib
                   else "未启用/未取得 BGM，卡点剪辑需要 BGM，跳过", state="done")
    bgm_info["beats"] = len(beats)
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
    tts_broken = False    # 配音服务连续失败（显存不足等）→ 本次不再尝试补配音
    # 克隆音色参考全 run 只挑一次：交给 Agent 随手指定会挑到转写不准的杂音段，模型就跑偏
    voice_ref = None
    if enable_tts and not pure_music:
        voice_ref = await _pick_voice_ref(rid, toolbox, inputs["product_name"], reference_video)
        if voice_ref and voice_ref.get("from_reference"):
            yield step("voiceref", "改用参考视频人声做音色参考",
                       "素材里没有语义通顺的真人口播（ASR 只识别到零星杂音），"
                       "改用参考爆款视频里的人声："
                       f"{os.path.basename(voice_ref['source_path'])} {voice_ref['source_time_range']}"
                       "；音色不是用户本人，但能保证配音念对文案。",
                       observation="「%s」" % voice_ref["speech"][:60])
        elif voice_ref:
            yield step("voiceref", "选定克隆音色参考",
                       "全片配音统一用这段用户原声做音色参考："
                       f"{os.path.basename(voice_ref['source_path'])} {voice_ref['source_time_range']}",
                       observation="「%s」" % voice_ref["speech"][:60])
        elif not os.getenv("WHQ_VOXCPM_REF_WAV"):
            # 素材池和参考视频都没有可靠的真人口播：zero-shot 克隆缺参考音 + 转写，模型会不念
            # 文案、顺着杂音胡念（实测生成的 wav 回读 ASR 是「跟去，不去。」）。必须让用户知道，
            # 否则成片里只是一串听不懂的怪声，看起来像"没有配音"。
            _log.warning("[%s] 素材与参考视频都无可靠真人口播，克隆配音可能跑偏", rid)
            yield step("voiceref", "缺少可用的克隆音色参考",
                       "用户素材和参考视频里都没有语义通顺的真人口播，"
                       "zero-shot 克隆缺少可靠参考音，配音可能念出不相干的内容。"
                       "建议在 config.env 里用 WHQ_VOXCPM_REF_WAV / WHQ_VOXCPM_REF_TEXT "
                       "指定一段干净口播作为音色参考，或本次关掉 TTS 配音。")
    all_slot_ids = [s["slot_id"] for s in slots]

    yield step("prep", "读取编排脚本 + 建全池召回",
               f"{len(slots)} 个 DNA 槽位待填，素材池 {toolbox.pool_size()} 段可自由召回，进入 Agent 剪辑-审片循环（最多 {max_loops} 轮）",
               observation="\n".join(f"{s['slot_id']} {s['role']} 目标{s['target_duration']}s" for s in slots))

    for loop in range(1, max_loops + 1):
        # 1) 出片计划
        incr = bool(prev_placements and revise_slots)
        seeded_all = bool(prev_placements) and not revise_slots and all(s["slot_id"] in prev_placements for s in slots)
        # 配音覆盖整片：勾了 TTS 配音就不该只有 AIGC 镜或 whq 点名的镜有人声。四类都要配——
        # whq 编排判为 clone 的、AIGC 生成的（seedance 出的画面天生无人声）、原声撑不起这一段的
        # （窗口太短/只有零星几个字，见 _mute_slots），以及**原声内容不可用**的（现场杂音、闲聊、
        # ASR 把噪音识别成的乱码句——长度够也不能留，见 _unusable_original_slots）。
        # 只有"完整、听得懂、在讲产品"的原声才保原声；这也是 whq 结构级复刻的取舍。
        voice_slots = []
        if seeded_all and enable_tts and not pure_music:
            need = set(_mute_slots(prev_placements, slots, prev_tts, toolbox))
            need |= await _unusable_original_slots(slots, prev_placements, toolbox,
                                                   inputs["product_name"])
            voice_slots = [s["slot_id"] for s in slots
                           if ((s.get("whq_voice") or {}).get("voice_source") == "clone"
                               or s["slot_id"] in aigc_presets
                               or s["slot_id"] in need)
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
                                            fact_corpora=fact_corpora, voice_ref=voice_ref):
                if ev.get("__edit_result__"):
                    placements, notes = ev["placements"], ev["notes"]
                    tts_by_slot = ev.get("tts_by_slot", {})
                    tts_broken = bool(ev.get("tts_broken"))
                    continue
                yield ev
        # 兜底补配音：Agent 有时会漏掉"既没原声、也没配音"的镜头（哑巴段）。这里客观检测
        # 出来，再跑一轮**只配音**把它们补上（最多补一次，避免无限循环）。带货成片不能有
        # 整段没人声的空档。配音服务本身挂了（显存不足等）就别补了，否则又是几十秒空转。
        if enable_tts and not pure_music and not tts_broken:
            mute = _mute_slots(placements, slots, tts_by_slot, toolbox)
            # 再加一道内容合理性核验：ASR 把现场杂音识别成话时，长度够也不能留
            bad = await _unusable_original_slots(slots, placements, toolbox, inputs["product_name"])
            mute = [s["slot_id"] for s in slots
                    if s["slot_id"] in set(mute) | bad and s["slot_id"] in placements]
            if mute:
                yield step(f"mute{loop}", f"第 {loop} 轮 · 补配音（{len(mute)} 段哑巴）",
                           "检测到既无原声又无配音的镜头：" + "、".join(mute) + "，补生成克隆配音",
                           state="running")
                async for ev in _run_edit_agent(rid, loop, toolbox, slots, dna, review_feedback, history, edit_sys,
                                                base_placements=placements, base_tts=tts_by_slot,
                                                enable_tts=enable_tts, pure_music=pure_music,
                                                voice_only_slots=mute, fact_corpora=fact_corpora,
                                                voice_ref=voice_ref):
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
        # 最后一道：Agent 常常配了前几镜就 finish（实测 S01–S06 有声、S07/S08 全程无声）。
        # 这里客观算出还缺哪些镜，一次 LLM 写完这几条文案再逐条合成，不再依赖 Agent 走完清单。
        if enable_tts and not pure_music and not tts_broken:
            added = await _fill_missing_voice(rid, slots, placements, tts_by_slot, toolbox,
                                              voice_ref, inputs["product_name"], enable_tts)
            if added:
                yield step(f"fillvoice{loop}", f"第 {loop} 轮 · 补齐剩余配音（{len(added)} 段）",
                           "Agent 未覆盖到的无声镜头由系统统一补配音：" + "、".join(added),
                           observation="\n".join(f"{sid}：{(tts_by_slot[sid] or {}).get('text','')}"
                                                 for sid in added))
        # 本轮的成片计划成为下一轮的基线（下一轮据本轮审片只改被点名的 slot）
        prev_placements, prev_tts = placements, tts_by_slot
        clips = _placements_to_clips(placements, slots, tts_by_slot, burn_captions=burn_on)
        _dedup_clips(clips, toolbox, slots)  # 成片级去重：同段素材被多镜复用 → 换未用过的素材
        _order_same_source(clips)            # 同源片多窗口按源时间顺序播，避免动作被倒放
        vis_notes = await _drop_product_absent(rid, clips, toolbox, inputs["product_name"])
        if vis_notes:
            yield step(f"prodvis{loop}", f"第 {loop} 轮 · 空镜核验（{len(vis_notes)} 处）",
                       "逐帧核验真实素材镜头里商品是否在画面里，把窗口挪开空镜",
                       observation="\n".join(vis_notes))
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
        caption_items = _caption_items(result, clips)   # 每镜实际文案 + 成片时间轴位置
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

        if bgm_defer_lib and not bgm:
            # 参考没配乐信息 → 成片已出：Gemini 看这轮成片（画面+声音）从曲库匹配最相关的一首，
            # 补混进成片（uri 不变）；选中的曲子后续轮次直接在 build_video 里混
            yield step("bgm-lib2", "美摄曲库选曲（按成片内容）",
                       "参考视频没有配乐信息，按这轮成片的画面/节奏/口播情绪匹配最相关的 BGM",
                       state="running")
            pick = await bgm_library.choose_bgm(reference_video, inputs["product_name"],
                                                edited_video=out_path)
            if pick.get("ok"):
                mix = await asyncio.to_thread(editor.mix_bgm, out_path, pick["path"])
                bgm, bgm_beats = pick["path"], pick.get("beats") or []
                if beat_sync and bgm_beats:
                    beats = bgm_beats   # 后续轮次重剪时卡点用
                matched = {"edited_video": "按成片内容匹配", "reference_audio": "对齐参考配乐"}.get(
                    pick.get("matched_by"), "按商品调性")
                bgm_info = {"source": "美摄曲库（{}）".format(matched), "name": pick["name"],
                            "bpm": round(pick.get("bpm") or 0, 1),
                            "reason": pick.get("reason") or "", "beats": len(beats)}
                if mix.get("ok"):
                    yield step("bgm-lib2", "美摄曲库选曲（按成片内容）",
                               "选中《{}》（bpm≈{:.0f}，{}）已混入本轮成片：{}".format(
                                   pick["name"], pick.get("bpm") or 0, matched,
                                   pick.get("reason") or ""), state="done")
                else:
                    bgm_info["reason"] = "；".join(x for x in (
                        bgm_info["reason"], "本轮混音失败：{}".format(mix.get("error", ""))) if x)
                    yield step("bgm-lib2", "美摄曲库选曲（按成片内容）",
                               "选中《{}》但本轮混音失败：{}（后续轮次会在剪辑时混入）".format(
                                   pick["name"], mix.get("error", "")), state="done")
            else:
                bgm = DEFAULT_BGM if os.path.isfile(DEFAULT_BGM) else ""
                if bgm:
                    await asyncio.to_thread(editor.mix_bgm, out_path, bgm)
                bgm_info = {"source": "本地默认 BGM" if bgm else "未加 BGM",
                            "name": os.path.basename(bgm) if bgm else "", "bpm": 0,
                            "reason": "按成片内容选曲失败：{}".format(pick.get("error", "")),
                            "beats": 0}
                yield step("bgm-lib2", "美摄曲库选曲（按成片内容）",
                           "失败：{}；{}".format(pick.get("error", ""),
                                              "回退本地默认 BGM" if bgm else "本次不加 BGM"),
                           state="done")
            bgm_defer_lib = False

        yield {"type": "agent_edit_sample", "loop": loop, "video_uri": video_uri}

        if not enable_review:
            # 未启用 Agent 审片：剪一轮直接出片，不进入审片-重剪循环
            _log.info("[%s] agent_edit done (审片关闭) loop=%d video=%s", rid, loop, video_uri)
            _save_edit_trace(task_id, strategy_abs, video_uri=video_uri, score=0,
                             verdict="已按剪辑 Agent 结果直接输出（未开启审片）", history=history, review_model=review_model)
            yield {"type": "agent_edit_done", "video_uri": video_uri, "final_path": out_path,
                   "loops": loop, "score": 0, "material_limited": False, "reviewed": False,
                   "review_model": review_model, "verdict": "已按剪辑 Agent 结果直接输出（未开启审片）",
                   "history": history, "bgm": bgm_info, "caption_items": caption_items}
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
            best = {"score": score, "video_uri": video_uri, "path": out_path, "loop": loop,
                    "caption_items": caption_items}
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
                   "verdict": review.get("thought", "符合预期"), "history": history,
                   "bgm": bgm_info, "caption_items": caption_items}
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
           "verdict": "已达轮次上限/素材受限，输出历轮最佳样片", "history": history,
           "bgm": bgm_info, "caption_items": b.get("caption_items") or []}
