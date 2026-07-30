"""whq_input — 把 Agent 的 AnalysisResult(契约①) 适配成 whq 的三件套输入。

whq 的 ``run_clone.run`` 原本吃三个文件（DNA md / all_user_assets.json / all_source_asr.json）。
Agent 集成后改为吃内存对象，本模块负责转换：

- ``dna``          <- reference_dna.build_dna_from_reference(参考视频, 参考ASR)
                      （按用户决策：对参考视频跑场景检测+ASR 重建 key_beats）
                      无参考视频时退化为 template.shot_slots 的时长序列近似。
- ``candidates``   <- material_understanding（asset_segments 拍平；schema 与 whq 对齐）
- ``asr``(路径)    <- 对用户素材源视频跑词级 ASR（asr_tokens），供原声保留/语速用
- ``ref_asr_items``<- 参考视频词级 ASR

字段来源不做猜测：由调用方（orchestration.pipeline）从 bundle 显式取出
material_understanding / reference_template / 参考视频路径后传入。
"""
import os

import asset_index
import asr_tokens
import reference_dna
import reference_shots

# Agent 仓根目录：whq_input.py 在 .../Viral_Video_Agent/src/editing/whq_clone/ 下
_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _abs_source(p):
    """把素材 source_path 解析成存在的绝对路径。

    Agent 的 material_understanding 里 source_path 常是相对 Agent 根的 'uploads/xxx.mov'。
    下游 ASR(cwd=Split根)/ffmpeg 用相对路径会找不到文件，这里统一转绝对。
    """
    if not p or (os.path.isabs(p) and os.path.exists(p)):
        return p
    for base in (_AGENT_ROOT, os.getcwd()):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand
    return p


def _resolve_candidates(candidates):
    for c in candidates or []:
        c["source_path"] = _abs_source(c.get("source_path"))
    return candidates


def _dna_from_template(template):
    """无参考视频时的兜底：用 shot_slots 的 duration 序列近似 key_beats（节奏贴合略降）。"""
    tpl = template or {}
    slots = tpl.get("shot_slots") or []
    total = float(tpl.get("total_duration_sec") or 0.0)
    key_beats = []
    cursor = 0.0
    for i, shot in enumerate(slots):
        dur = float(shot.get("duration") or 0.0) or (total / max(1, len(slots)) if total else 3.0)
        s, e = cursor, cursor + dur
        cursor = e
        desc = ""
        for d in shot.get("breakdown", []) or []:
            if isinstance(d, dict) and ("字幕" in (d.get("dim") or "") or "口播" in (d.get("dim") or "")):
                desc = d.get("value") or ""
                break
        desc = desc or shot.get("want") or shot.get("role") or ""
        m1, s1 = int(s // 60), s - 60 * int(s // 60)
        m2, s2 = int(e // 60), e - 60 * int(e // 60)
        beat = "{:d}:{:05.2f}-{:d}:{:05.2f}".format(m1, s1, m2, s2)
        key_beats.append("{}：{}".format(beat, desc) if desc else beat)
    return {
        "duration_estimate": round(cursor or total, 3),
        "content_structure": {"key_beats": key_beats},
        "topic_and_emotion": {"topic": tpl.get("industry", "")},
        "_reconstructed": False,
        "_from_template": True,
    }


def _consolidate_beats(dna, max_beats):
    """把过多的叙事节拍合并成 max_beats 个核心节拍（保序、保开场钩子+结尾催单），避免段太碎。"""
    cs = (dna.get("content_structure") or {}) if isinstance(dna, dict) else {}
    beats = cs.get("key_beats") or []
    if len(beats) <= max_beats:
        return dna
    try:
        from pipeline_utils import ask_qianfan, loads_with_repair
    except Exception:
        return dna
    prompt = (
        "下面是一条带货短视频的叙事节拍(有序)。请合并成 **{} 个核心节拍**：保持原有顺序，"
        "**必须保留开场钩子和结尾催单**，把相邻且功能相近的节拍合并，每个节拍一句话描述其"
        "画面内容+叙事功能。只返回 JSON：{{\"key_beats\":[\"...\", ...]}}\n{}"
    ).format(max_beats, "\n".join("{}. {}".format(i + 1, b) for i, b in enumerate(beats)))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], temperature=0.2)
        nb = [str(x).strip() for x in (loads_with_repair(text).get("key_beats") or []) if str(x).strip()]
        if 3 <= len(nb) <= max_beats:
            dna = dict(dna)
            cs = dict(cs)
            cs["key_beats"] = nb
            dna["content_structure"] = cs
            print("[whq_input] 节拍收敛: {} -> {} 段".format(len(beats), len(nb)), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[whq_input] 节拍收敛失败(保留原节拍): {}".format(str(exc)[:120]), flush=True)
    return dna


def _shared_asr_dir(paths, kind):
    """词级 ASR 的**跨 run 共享**缓存目录：outputs/whq_asr_cache/<kind>_<素材指纹>。

    原先 cache_dir 落在 outputs/whq_clone/<run_id>/_whq_work 下，run_id 每次都新，
    asr_tokens 里的指纹缓存永远命中不了 —— 同一批素材每跑一次方案就重跑一遍 ASR
    （实测 32 个源片约 10s/个，5 分钟以上）。按素材指纹放到共享目录即可复用。
    """
    from pipeline_utils import fingerprint, file_fingerprint
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not real:
        return None
    fp = fingerprint(file_fingerprint(sorted(real)))[:16]
    return os.path.join(_AGENT_ROOT, "outputs", "whq_asr_cache", "{}_{}".format(kind, fp))


def build_whq_inputs(material_understanding, reference_template, ref_video,
                     work_dir, *, run_asr=True, topic="",
                     dna_path=None, assets_path=None, asr_path=None, use_vlm=True,
                     max_beats=None):
    """返回 dict(dna, candidates, asr, ref_asr_items)。

    **优先吃真实 understanding 产物**（与原版 whq 一致，最忠实）：
    - dna_path:    真实 DNA md（VLM 撰写的语义节拍）-> reference_shots.load_dna
    - assets_path: all_user_assets.json -> asset_index.load_candidates
    - asr_path:    all_source_asr.json（用户素材词级 ASR）-> voiceover 直接读

    未提供某项产物时才回退到 Agent 侧的重建/内存对象：
    - dna:        参考视频 VLM 重建 key_beats；再不行用 template 近似
    - candidates: 从 AnalysisResult.material_understanding 拍平
    - asr:        对用户素材源视频现跑词级 ASR（asr_tokens）

    ref_asr_items（参考视频语速，用于 ref_cps）始终对参考视频跑一遍 ASR 得到，与
    上面的用户素材 asr 无关。
    """
    os.makedirs(work_dir, exist_ok=True)

    # ---- 候选素材：真实 all_user_assets.json 优先 ----
    if assets_path and os.path.exists(assets_path):
        candidates = asset_index.load_candidates(assets_path)
    else:
        results = [v for v in (material_understanding or {}).values() if isinstance(v, dict)]
        candidates = asset_index.load_candidates_from_results(results)
    # source_path 统一转绝对，供下游 ASR / ffmpeg 正确定位（Agent 素材常是相对 uploads/ 路径）
    candidates = _resolve_candidates(candidates)

    # ---- 参考视频语速（ref_cps 用）----
    ref_asr_items = []
    if run_asr and ref_video and os.path.exists(ref_video):
        ref_asr_items = asr_tokens.reference_asr_items(
            ref_video,
            _shared_asr_dir([ref_video], "ref") or os.path.join(work_dir, "ref_asr"))

    # ---- 用户素材 ASR：真实 all_source_asr.json 优先 ----
    if asr_path and os.path.exists(asr_path):
        asr_json = asr_path
    elif run_asr:
        source_paths = [c.get("source_path") for c in candidates if c.get("source_path")]
        asr_json = asr_tokens.source_asr_json(
            source_paths,
            _shared_asr_dir(source_paths, "src") or os.path.join(work_dir, "source_asr"))
    else:
        asr_json = None

    # ---- DNA：真实 DNA md 优先 ----
    if dna_path and os.path.exists(dna_path):
        dna = reference_shots.load_dna(dna_path)
        dna.setdefault("_reconstructed", False)
        dna["_source"] = "real_dna_md"
        print("[whq_input] 使用真实 DNA md: {}".format(dna_path), flush=True)
    elif ref_video and os.path.exists(ref_video):
        dna = reference_dna.build_dna_from_reference(ref_video, ref_asr_items,
                                                     topic=topic, use_vlm=use_vlm)
    else:
        dna = _dna_from_template(reference_template)

    # 节拍收敛：段太碎会让叙事跳、难连贯；合并到 ~max_beats 个核心节拍（默认 6）
    mb = max_beats or int(os.getenv("WHQ_MAX_BEATS", "6"))
    dna = _consolidate_beats(dna, mb)

    return {"dna": dna, "candidates": candidates, "asr": asr_json,
            "ref_asr_items": ref_asr_items}
