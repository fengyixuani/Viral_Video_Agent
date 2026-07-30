"""strategy_out — 把 whq 产物映射成 Agent 的 selected_editing_strategy.json（契约②）。

按团队约定(a)：尽量复用现有 strategy schema，不新增并列契约。whq 特有的原声决策/对窗/
参考语速等字段挂到每条 timeline 的扩展子对象 ``whq_voice`` / 顶层 ``whq_meta``，不进入
``contracts.validate_strategy`` 的必填集，避免污染 A/C 共用的契约。

注意：whq_clone 有两种形态。
- workflow 形态（run_clone 直接产 mp4）：strategy JSON 是**产物记录 + 契约兼容**用途。
- Agent 形态（reproduce_mode=whq_clone 走 editing/loop.py 的剪辑-审片循环）：strategy JSON
  是**真正的编排产出**，editing_timeline 作剪辑 Agent 第 1 轮的基线预填，配套的
  connector_context.json 提供 DNA 槽位 + 全量素材池（见本文件下半部分 build_context）。
"""
import json
import os

import _common  # noqa: F401  确保 src/shared 挂上 sys.path（供下面 import contracts）
import contracts  # src/shared/contracts.py


def _src_range(seg_manifest):
    start = float(seg_manifest.get("source_start") or 0.0)
    take = seg_manifest.get("source_take")
    if not take:
        avail = float(seg_manifest.get("source_avail") or 0.0)
        target = float(seg_manifest.get("target_duration") or 0.0)
        take = min(avail, target) if avail > 0 else target
    return "{:.2f}-{:.2f}".format(start, start + float(take or 0.0))


def _tts_by_slot(tts_items):
    """slot_id -> tts_overlay_plan item（配音阶段的**最终**逐段决策，权威来源）。"""
    out = {}
    for it in tts_items or []:
        sid = it.get("slot_id") or it.get("slot")
        if sid:
            out[sid] = it
    return out


def build_strategy(segments, manifest, tts_items=None, *, dna=None,
                   product_name="", reproduce_mode="whq_clone"):
    """返回 selected_editing_strategy.json 结构（dict）+ 校验问题列表。"""
    tts_by_slot = _tts_by_slot(tts_items)
    seg_by_slot = {s.get("slot_id"): s for s in (segments or [])}

    timeline, bank, missing = [], [], []
    non_gap = 0
    for m in manifest or []:
        slot_id = m.get("slot_id") or "S{:02d}".format(m.get("segment_index", 0))
        seg = seg_by_slot.get(slot_id, {})
        tts_it = tts_by_slot.get(slot_id, {})
        cand = seg.get("best_candidate") or {}
        is_generated = bool(m.get("is_t2v")) or seg.get("is_gap")
        caption = (tts_it.get("caption_text") or tts_it.get("text")
                   or m.get("window_text") or m.get("beat_desc") or "")
        action = "generate" if is_generated else "use_user_asset"

        # 原声/克隆的**最终**判定以 tts_overlay_plan(配音阶段产出) 为准；voiceover 内部
        # 的 LLM 决策会写进 item.voice_source，故这里优先读它，段内预判仅作兜底。
        # Agent 形态(只编排不出片)没有 tts_overlay_plan，取 manifest 里的 voice_policy 预判。
        voice_source = tts_it.get("voice_source") or m.get("voice_source") or seg.get("voice_align")
        item = {
            "slot_id": slot_id,
            "target_time_range": m.get("target_time_range", ""),
            "source_path": m.get("source_path", ""),
            "source_time_range": _src_range(m),
            # 素材 id 也写进 timeline：下游（loop._load_inputs）据它把该镜与素材池对齐，
            # 从而拿到「本镜实际画面内容」喂给剪辑 Agent 写文案。只放在 bank 里不够稳。
            "global_asset_id": cand.get("global_asset_id", ""),
            "caption_text": caption,
            "transition_to_next": "hard_cut",
            "action": action,
            # whq 特有：原声/克隆决策 + 对窗 + 参考语速（挂扩展位，不进必填契约）
            "whq_voice": {
                "voice_source": voice_source,
                "audio_take": tts_it.get("audio_take") if tts_it.get("audio_take") is not None else m.get("source_take"),
                "ref_cps": tts_it.get("ref_cps") if tts_it.get("ref_cps") is not None else m.get("ref_cps"),
                "ref_time_range": m.get("ref_time_range", ""),
                "basis": tts_it.get("basis", ""),
            },
        }
        timeline.append(item)

        if is_generated:
            missing.append({
                "slot_id": slot_id,
                "role": m.get("beat_desc", ""),
                "goal": m.get("beat_desc", ""),
                "action": "generate",
                "target_duration": m.get("target_duration"),
            })
        else:
            non_gap += 1
            bank.append({
                # slot_id 必须给：loop.py 的 _load_inputs 用它把 bank 与 slot 对齐，
                # 从而把 caption/口播种进第 1 轮的预填（缺了就变成"画面有、字幕空"）。
                "slot_id": slot_id,
                "asset_id": cand.get("global_asset_id") or slot_id,
                "source_video_id": cand.get("source_video_id", ""),
                "source_path": m.get("source_path", ""),
                "source_time_range": _src_range(m),
                "visual_description": m.get("beat_desc", ""),
                # 只有**原声段**才把对窗后的原话作为口播/字幕；克隆段留空——它的字幕要由
                # 剪辑 Agent 的 tts_clone 文案决定，塞素材画面描述会烧出错误字幕。
                "speech_or_text": (m.get("window_text", "")
                                   if voice_source == "original" else ""),
                "fit_score": m.get("score"),
                "final_caption_text": caption,
            })

    total = len(manifest or []) or 1
    strategy = {
        "metadata": {
            "matching_mode": "whq_structure_clone",
            "template_source": "reconstructed_key_beats" if (dna or {}).get("_reconstructed") else "template_fallback",
            "reproduce_mode": reproduce_mode,
            "contract_version": contracts.CONTRACT_VERSION,
            "product_name": product_name,
        },
        "user_asset_bank": bank,
        "editing_timeline": timeline,
        "missing_assets": missing,
        "overall_editing_strategy": {
            "structure_fit_score": round(non_gap / total, 3),
            "material_sufficiency_score": round(non_gap / total, 3),
            "main_risks": (["部分镜头需生成/补拍"] if missing else []),
        },
        "whq_meta": {
            "duration_estimate": (dna or {}).get("duration_estimate"),
            "n_segments": len(manifest or []),
            "n_gap": len(missing),
        },
    }
    problems = contracts.validate_strategy(strategy)
    return strategy, problems


def write_strategy(segments, manifest, out_path, tts_items=None, **kw):
    """构建并落盘 strategy JSON，返回 (path, problems)。"""
    strategy, problems = build_strategy(segments, manifest, tts_items=tts_items, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(strategy, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_path, problems


# ---------------------------------------------------------------------------
# Agent 形态：只编排不出片时，把 whq 的规划映射成 Agent 剪辑循环的两个输入文件
#   selected_editing_strategy.json（契约②，editing_timeline 作第 1 轮基线预填）
#   connector_context.json（shots 结构 + **全量**素材池，供剪辑 Agent 自主召回）
# 见 src/editing/loop.py 的 _load_context / _load_inputs。
# ---------------------------------------------------------------------------

def manifest_from_segments(segments, decisions=None):
    """把 plan 阶段的 segments 拍成 build_base 那套 manifest 形态（不真的剪画面）。

    build_strategy 原本吃 clone_builder.build_base 的产物 manifest；Agent 形态下画面由
    editing/loop.py 剪，所以这里按 segments + 原声对窗结果合成等价结构。
    """
    decisions = decisions or {}
    out = []
    cursor = 0.0
    for i, seg in enumerate(segments or []):
        sid = seg.get("slot_id") or "S{:02d}".format(i + 1)
        cand = seg.get("best_candidate") or {}
        target = float(seg.get("target_duration") or 0.0)
        avail = float(cand.get("duration") or 0.0)
        start = seg.get("source_start_override")
        start = float(start) if start is not None else float(cand.get("start") or 0.0)
        take = seg.get("source_take") or (min(avail, target) if avail > 0 else target)
        d = decisions.get(sid) or {}
        out.append({
            "slot_id": sid,
            "segment_index": i + 1,
            "target_time_range": "{:.2f}-{:.2f}".format(cursor, cursor + target),
            "target_duration": target,
            "source_path": (seg.get("t2v_path") if seg.get("is_t2v") else cand.get("source_path", "")),
            "source_start": start,
            "source_take": float(take or 0.0),
            "source_avail": avail,
            "beat_desc": seg.get("beat_desc", ""),
            "is_t2v": bool(seg.get("is_t2v")),
            "score": seg.get("score"),
            "ref_cps": seg.get("ref_cps"),
            "ref_time_range": seg.get("ref_time_range", ""),
            "voice_source": d.get("voice_source"),
            "window_text": d.get("window_text", ""),
        })
        cursor += target
    return out


def _window_asr_items(speech_records, source_path, start, end, pad=3.5):
    """取某源片 [start-pad, end+pad] 内的逐字 ASR（用于 Agent 侧句子级对窗，无需重跑 ASR）。

    pad 要够大：素材切片边界常把一句话切两半，对窗/延展需要看到窗口外那半句才能把话补完
    （实测 pad=1.5 时最后两个字仍落在 pad 之外，成片还是半句被截断）。
    """
    out = []
    for rec in speech_records or []:
        if rec.get("source_path") != source_path:
            continue
        for it in rec.get("asr_items") or []:
            try:
                s, e = float(it.get("start")), float(it.get("end"))
            except (TypeError, ValueError):
                continue
            if start - pad <= s and e <= end + pad:
                out.append({"start": round(s, 3), "end": round(e, 3),
                            "text": str(it.get("text") or "")})
    return out


def _pool_item(cand, speech_map=None, speech_records=None):
    """候选片段 -> connector_context.segments 的一条（Retriever 索引 + 召回展示用）。

    ``whq_speech`` 是 whq 独有的原声标注：该候选窗口内是否自带可用口播、原话文本，以及
    窗口附近的逐字 ASR。剪辑 Agent 召回时即可见"这条素材有真声"，并能用 place_original
    在不重跑 ASR 的前提下做句子级对窗（见 whq_clone/agent_tools.py）。
    """
    gid = cand.get("global_asset_id") or ""
    sm = (speech_map or {}).get(gid) or {}
    text = cand.get("text") or ""
    start, end = float(cand.get("start") or 0.0), float(cand.get("end") or 0.0)
    ws = {
        "has_speech": bool(sm.get("has_speech")),
        "chars": sm.get("chars"),
        "coverage": sm.get("coverage"),
        "text": sm.get("text", ""),
    }
    # 逐字 ASR 对**所有**候选都下发（不只 has_speech 的）：Agent 侧的 place_original 靠它做
    # 句子级对窗。编排的 has_speech 判定偏保守，若只给"判定有口播"的候选带 asr_items，
    # 那些被判 clone 的段即使画面里人在说话，Agent 也没法把原声救回来 -> 口型对不上配音。
    items = _window_asr_items(speech_records, cand.get("source_path", ""), start, end)
    if items:
        ws["asr_items"] = items
        if not ws["text"]:
            ws["text"] = "".join(it.get("text", "") for it in items)[:80]
    return {
        "global_asset_id": gid,
        "source_video_id": cand.get("source_video_id", ""),
        "source_path": cand.get("source_path", ""),
        "source_time_range": "{:.2f}-{:.2f}".format(start, end),
        "duration": cand.get("duration"),
        "one_sentence_summary": text[:120],
        "visual_description": text,
        "speech_or_text": sm.get("text", ""),
        "keywords": cand.get("keywords", []) or [],
        "quality_score": cand.get("quality_score"),
        "whq_speech": ws,
    }


def build_context(segments, candidates, *, dna=None, decisions=None, speech_map=None,
                  speech_records=None):
    """返回 connector_context.json 结构：shots（DNA 节拍槽位）+ segments（全量素材池）。"""
    decisions = decisions or {}
    shots = []
    for i, seg in enumerate(segments or []):
        sid = seg.get("slot_id") or "S{:02d}".format(i + 1)
        d = decisions.get(sid) or {}
        beat = seg.get("beat_desc", "")
        shots.append({
            "slot_id": sid,
            "role": "节拍{}".format(i + 1),
            "want": beat,
            "breakdown": [{"dim": "叙事节拍", "value": beat}] if beat else [],
            "duration": seg.get("target_duration"),
            "narrative": beat,
            # whq 的原声基线：该段第 1 轮用原声还是克隆配音，以及对窗后的原话
            "whq_voice": {"voice_source": d.get("voice_source"),
                          "window_text": d.get("window_text", ""),
                          "ref_cps": seg.get("ref_cps"),
                          "basis": d.get("decision_basis", "")},
        })
    pool = [_pool_item(c, speech_map, speech_records) for c in (candidates or [])
            if isinstance(c, dict) and c.get("global_asset_id")]
    return {
        "source": "whq_clone",
        "dna_topic": (dna or {}).get("topic_and_emotion") if isinstance(dna, dict) else "",
        "shots": shots,
        "segments": pool,
    }


def write_context(segments, candidates, out_path, **kw):
    """落盘 connector_context.json（必须与 strategy JSON 同目录，loop.py 按同目录找）。"""
    ctx = build_context(segments, candidates, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(ctx, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out_path
