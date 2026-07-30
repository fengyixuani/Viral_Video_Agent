"""runner — whq_clone 的 Agent 集成入口（AnalysisResult 驱动，落契约②）。

两个入口共用 ``_prepare_inputs``（契约①→whq 三件套）：

- ``plan_whq_clone``（Agent 形态，推荐）：只编排不出片
    build_whq_inputs → run_clone.plan_from_objects（结构段落/1:1 分配/原声决策/对窗）
      → strategy_out.write_strategy + write_context
    出片交 src/editing/loop.py 的剪辑-审片循环：第 1 轮用这里的确定性分配作基线，
    审片不通过则剪辑 Agent 自主从全量素材池换段重剪。

- ``run_whq_clone``（workflow 形态，legacy）：一步出成片
    …→ run_clone.run_from_objects（+ build_base/voiceover/finisher/pace_match）

- ``edit_from_strategy``（前端「workflow 剪辑」按钮）：只出片
    读编排落的 ``*_plan.json`` → run_clone.render_from_plan，不走 Split 主链路。

供 orchestration/pipeline.py 在 reproduce_mode=="whq_clone" 时调用；
``edit_from_strategy`` 由 src/shared/connector.py 在 whq_clone 编排产物上调用。
"""
import json
import os

import whq_input
import strategy_out
import run_clone


def _get(bundle, key, default=None):
    """兼容 dict / dataclass / 普通对象的取值。"""
    if isinstance(bundle, dict):
        return bundle.get(key, default)
    return getattr(bundle, key, default)


def run_whq_clone(bundle, out_path, ref_video=None, *, product_name="",
                  use_llm=True, model=None, mode="hardcut", run_asr=True,
                  migrate_bgm=False, pace_match=True, prefer_original_voice=True,
                  dna_md=None, assets_json=None, asr_json=None, slug=None,
                  use_seedance=False, voice_model="cosyvoice",
                  deep_understanding=False, material_paths=None):
    """从 AnalysisResult(bundle) 跑完整 whq 结构级复刻，返回产物信息。

    - use_seedance: True=缺镜头/弱匹配段用 seedance 补拍（mode=seedance）；
      False(默认)=不补拍(hardcut)，改由 weak_arbiter 做 LLM 判定+VLM 二次校验，
      跑题段按连贯优先删段/保留（见 weak_arbiter）。
    其余产物路径参数见下（真实 understanding 产物优先）。
    """
    prep = _prepare_inputs(
        bundle, out_path, ref_video, product_name=product_name, run_asr=run_asr,
        prefer_original_voice=prefer_original_voice, dna_md=dna_md,
        assets_json=assets_json, asr_json=asr_json, slug=slug, mode=mode,
        use_seedance=use_seedance, voice_model=voice_model,
        deep_understanding=deep_understanding, material_paths=material_paths)
    inp = prep["inp"]

    details = run_clone.run_from_objects(
        inp["dna"], inp["candidates"], inp["asr"], inp["ref_asr_items"], out_path,
        ref=prep["ref_video"], mode=prep["effective_mode"], use_llm=use_llm, model=model,
        product_name=prep["product_name"], migrate_bgm=migrate_bgm, pace_match=pace_match,
        work_dir=prep["work_dir"], return_details=True, arbitrate_weak=prep["arbitrate_weak"])

    strategy_path = prep["stem"] + "_selected_editing_strategy.json"
    _, problems = strategy_out.write_strategy(
        details["segments"], details["manifest"], strategy_path,
        tts_items=details["tts_items"], dna=inp["dna"],
        product_name=prep["product_name"], reproduce_mode="whq_clone")

    process = _build_process(inp["dna"], details, prep["dna_md"])

    return {
        "final": details["final"],
        "strategy_path": strategy_path,
        "strategy_problems": problems,
        "plan_path": details["plan_path"],
        "manifest_path": details["manifest_path"],
        "n_gap": details["n_gap"],
        "dna_reconstructed": bool(inp["dna"].get("_reconstructed")),
        "dna_source": inp["dna"].get("_source", "reconstructed"),
        "process": process,
    }


def plan_whq_clone(bundle, out_path, ref_video=None, *, product_name="",
                   use_llm=True, model=None, mode="hardcut", run_asr=True,
                   prefer_original_voice=True, dna_md=None, assets_json=None,
                   asr_json=None, slug=None, use_seedance=False,
                   deep_understanding=False, material_paths=None):
    """**只编排不出片**（Agent 形态）：产出 strategy JSON + connector_context.json。

    出片交给 src/editing/loop.py 的 Agent 剪辑循环：第 1 轮直接用这里的确定性分配作基线，
    审片不通过时剪辑 Agent 自主从全量素材池重新召回/换段（见 loop.agent_edit_stream）。
    返回 {strategy_path, context_path, plan_path, n_gap, process, ...}，无 final。
    """
    prep = _prepare_inputs(
        bundle, out_path, ref_video, product_name=product_name, run_asr=run_asr,
        prefer_original_voice=prefer_original_voice, dna_md=dna_md,
        assets_json=assets_json, asr_json=asr_json, slug=slug, mode=mode,
        use_seedance=use_seedance, voice_model="cosyvoice",
        deep_understanding=deep_understanding, material_paths=material_paths)
    inp = prep["inp"]

    plan = run_clone.plan_from_objects(
        inp["dna"], inp["candidates"], inp["asr"], inp["ref_asr_items"], prep["stem"],
        ref=prep["ref_video"], mode=prep["effective_mode"], use_llm=use_llm, model=model,
        product_name=prep["product_name"], work_dir=prep["work_dir"],
        arbitrate_weak=prep["arbitrate_weak"],
        # Agent 形态成片不拉伸素材去填满节拍槽位，原声判定按实际会用到的时长来算，
        # 否则短素材会被误判成"要拉伸 4x、原声得放慢到 0.25 倍"而全降级成克隆配音
        no_stretch=True)

    # 原声标注（哪条候选自带可用口播）随素材池下发，剪辑 Agent 召回时即可见 -> 优先保留原声
    speech_map = {}
    try:
        from edit_planner import build_speech_map
        speech_map = build_speech_map(inp["candidates"], plan["speech"]) or {}
    except Exception as exc:  # noqa: BLE001
        print("[runner] 原声标注生成失败(素材池不带原声提示): {}".format(str(exc)[:120]), flush=True)

    manifest = strategy_out.manifest_from_segments(plan["segments"], plan["decisions"])
    strategy_path = prep["stem"] + "_selected_editing_strategy.json"
    _, problems = strategy_out.write_strategy(
        plan["segments"], manifest, strategy_path, dna=inp["dna"],
        product_name=prep["product_name"], reproduce_mode="whq_clone")
    # loop.py 只认 strategy 同目录的 connector_context.json
    context_path = strategy_out.write_context(
        plan["segments"], inp["candidates"],
        os.path.join(os.path.dirname(strategy_path), "connector_context.json"),
        dna=inp["dna"], decisions=plan["decisions"], speech_map=speech_map,
        speech_records=plan["speech"])

    details = dict(plan, manifest=manifest, tts_items=[], final="")
    process = _build_process(inp["dna"], details, prep["dna_md"])
    n_original = sum(1 for d in (plan["decisions"] or {}).values()
                     if (d or {}).get("voice_source") == "original")
    warnings = _plan_warnings(plan, inp, speech_map, n_original)
    for w in warnings:
        print("[runner] 告警: {}".format(w), flush=True)
    return {
        "strategy_path": strategy_path,
        "context_path": context_path,
        "warnings": warnings,
        "strategy_problems": problems,
        "plan_path": plan["plan_path"],
        "n_gap": plan["n_gap"],
        "n_segments": len(plan["segments"]),
        "n_original_voice": n_original,
        "n_pool": len(inp["candidates"] or []),
        "dna_reconstructed": bool(inp["dna"].get("_reconstructed")),
        "dna_source": inp["dna"].get("_source", "reconstructed"),
        "process": process,
    }


def edit_from_strategy(strategy_path, out_path=None, *, product_name="", model=None,
                       migrate_bgm=False, pace_match=True, stretch=True, asr_json=None,
                       ref_video=None):
    """从已落盘的编排产物**只出片**（前端「workflow 剪辑」按钮走这里）。

    读 strategy 旁边的 ``*_plan.json``（plan_from_objects 落的编排 checkpoint，含
    segments + decisions + reference_dna），直接跑 whq 自己的出片链路：
    硬剪 → 配音（原声段保留原声/克隆段 TTS）→ 字幕 → 语速贴参考。

    **为什么不走 Split 主链路**：Split 的 ``orchestration/rebuild_asr_edit.py`` 会把每镜
    窗口扩到 ASR 整句边界、并按扩窗后的长度 retime 时间线（ASR_CAPTION_DURATION_FOLLOWS_SOURCE），
    把 whq 锁好的槽时长顶开——实测参考 28s 的片子被撑到 65s，且扩窗把别的段的口播吞进来，
    听感上同一句话说两遍。whq 编排阶段已经做完选片/对窗/原声决策/文案，Split 再重建一次
    等于两套逻辑打架，因此这条按钮直接复用 run_clone 的出片后半段。

    返回 {final, base, voiced, manifest_path, plan_path, n_segments, n_original_voice}。
    """
    strategy_path = os.path.abspath(strategy_path)
    stem = strategy_path[:-len("_selected_editing_strategy.json")] \
        if strategy_path.endswith("_selected_editing_strategy.json") \
        else os.path.splitext(strategy_path)[0]
    plan_path = stem + "_plan.json"
    if not os.path.isfile(plan_path):
        raise FileNotFoundError(
            "找不到编排产物 {}：workflow 剪辑需要 whq 编排阶段落的 plan.json".format(plan_path))
    with open(plan_path, encoding="utf-8") as stream:
        plan = json.load(stream)
    segments = plan.get("segments") or []
    decisions = plan.get("decisions") or {}
    dna = plan.get("reference_dna") or {}
    if not segments:
        raise ValueError("编排产物里没有 segments，无法出片：{}".format(plan_path))
    if not decisions:
        raise ValueError("编排产物里没有原声/克隆决策（plan.json 由旧版本生成）：{}".format(plan_path))

    out_path = out_path or (stem + ".mp4")
    work_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_whq_work")
    # 词级 ASR：voiceover 靠它取声音克隆参考 + 原声窗口文本 + 文案取材。缺它会静默降级成
    # 「无配音 + 空字幕」（实测），所以按 编排记录 -> 本地产物 -> 现跑 三级兜底找齐。
    if not asr_json:
        asr_json = plan.get("asr_path") or ""
    if not (asr_json and os.path.isfile(asr_json)):
        cand = os.path.join(work_dir, "source_asr", "all_source_asr.json")
        asr_json = cand if os.path.isfile(cand) else ""
    if not asr_json:
        import asr_tokens
        sources = []
        for seg in segments:
            src = ((seg.get("best_candidate") or {}).get("source_path") or "")
            if src and src not in sources:
                sources.append(whq_input._abs_source(src))
        print("[runner] 编排产物没带词级 ASR, 对 {} 个源素材现跑一遍".format(len(sources)), flush=True)
        asr_json = asr_tokens.source_asr_json(sources, os.path.join(work_dir, "source_asr"))
    if not asr_json:
        print("[runner] 词级 ASR 不可用: 本次出片将无配音/无字幕(检查 ASR_PYTHON 与模型路径)",
              flush=True)

    rendered = run_clone.render_from_plan(
        segments, decisions, dna, asr_json, [], out_path,
        ref=whq_input._abs_source(ref_video) if ref_video else None,
        stretch=stretch, product_name=product_name, model=model,
        migrate_bgm=migrate_bgm, pace_match=pace_match, work_dir=work_dir)

    n_original = sum(1 for d in decisions.values()
                     if (d or {}).get("voice_source") == "original")
    return {
        "final": rendered["final"],
        "base": rendered["base"],
        "voiced": rendered["voiced"],
        "manifest_path": rendered["manifest_path"],
        "plan_path": plan_path,
        "n_segments": len(segments),
        "n_original_voice": n_original,
        "ref_duration": dna.get("duration_estimate"),
    }


def _plan_warnings(plan, inp, speech_map, n_original):
    """编排产物的健康检查：把「静默降级」变成显式告警。

    词级 ASR 失败（如显存不足）时，speech 记录全空 -> voice_policy 全判克隆、素材池没有
    原声标注、ref_cps 也拿不到。此时 Agent 形态下第 1 轮基线既没原声也没字幕，成片会像
    「什么都没做」。这类问题必须在编排阶段就喊出来，而不是等看完片才发现。
    """
    out = []
    recs = plan.get("speech") or []
    n_rec_with_items = sum(1 for r in recs if isinstance(r, dict) and (r.get("asr_items") or []))
    if recs and not n_rec_with_items:
        out.append("用户素材词级 ASR 全部为空（{} 个源片都没出 token）：原声保留/字幕全失效，"
                   "成片将只有素材原声、无字幕。常见原因是 ASR 子进程显存不足或模型路径缺失，"
                   "查日志里的 ASR_RUN_FAILED。".format(len(recs)))
    elif not recs:
        out.append("没有加载到任何用户素材 ASR：原声保留/字幕不可用（run_asr 关闭或 ASR 产物缺失）。")
    n_has_speech = sum(1 for v in (speech_map or {}).values() if (v or {}).get("has_speech"))
    if (inp.get("candidates") or []) and not n_has_speech:
        out.append("素材池 {} 段候选全部没有可用口播标注：剪辑 Agent 无法用 place_original 保留"
                   "用户原声，只能走克隆配音。".format(len(inp["candidates"])))
    if plan.get("segments") and not n_original:
        out.append("没有任何段判定为保留原声（original=0）：与 whq 复刻「优先保留用户原声」的目标"
                   "相悖，请先确认 ASR 与人脸检测是否正常。")
    if not (inp.get("ref_asr_items") or []):
        out.append("参考视频没拿到逐字 ASR：ref_cps（贴参考语速）缺失，配音字数只能用默认语速估。")
    return out


def _prepare_inputs(bundle, out_path, ref_video=None, *, product_name="", run_asr=True,
                    prefer_original_voice=True, dna_md=None, assets_json=None,
                    asr_json=None, slug=None, mode="hardcut", use_seedance=False,
                    voice_model="cosyvoice", deep_understanding=False,
                    material_paths=None):
    """契约①→whq 三件套输入 + 路径/环境准备（两种形态共用）。"""
    material_understanding = _get(bundle, "material_understanding") or {}
    template = _get(bundle, "template") or {}
    ref_video = ref_video or _get(bundle, "reference_video") or _get(bundle, "ref_video")
    # 参考视频统一转绝对路径：素材/参考常是相对 'uploads/xxx'，深度理解(cwd=Split根)/ffmpeg
    # 用相对路径会 FileNotFound（实测 understand_reference 因此失败回退）。
    ref_video = whq_input._abs_source(ref_video)
    product_name = product_name or _get(bundle, "product_name") or ""
    # bundle 里也可带产物路径（A 的理解阶段回填）
    dna_md = dna_md or _get(bundle, "dna_md") or _get(bundle, "dna_path")
    assets_json = assets_json or _get(bundle, "assets_json") or _get(bundle, "assets_path")
    asr_json = asr_json or _get(bundle, "asr_json") or _get(bundle, "source_asr_path")
    slug = slug or _get(bundle, "slug")

    # slug -> 自动发现真实产物（缺项才补），与原版 run_clone.discover_from_slug 同源
    if slug:
        d, a, r = run_clone.discover_from_slug(slug)
        dna_md = dna_md or d
        assets_json = assets_json or a
        asr_json = asr_json or r

    # 深度理解：调 Split 的 understand_reference + understand_user_segments 产出 Split 级
    # DNA md + all_user_assets.json（全模型），让自动/网页链路也达到手工版质量。缺产物才跑。
    if deep_understanding and not (dna_md and assets_json):
        import whq_understand as whq_understanding
        if material_paths is None:
            material_paths = []
            for v in (material_understanding or {}).values():
                if isinstance(v, dict) and v.get("source_path"):
                    material_paths.append(whq_input._abs_source(v["source_path"]))
        prefix = slug or "whq_" + os.path.basename(os.path.splitext(out_path)[0])
        d, a = whq_understanding.deep_understand(ref_video, material_paths, prefix,
                                                 gpu=os.getenv("CUDA_VISIBLE_DEVICES", "1"))
        dna_md = dna_md or d
        assets_json = assets_json or a

    # 接不接 seedance 是个选项
    effective_mode = "seedance" if use_seedance else mode
    arbitrate_weak = not use_seedance  # 不补拍时才做弱匹配仲裁

    if prefer_original_voice:
        os.environ.setdefault("WHQ_PREFER_ORIGINAL_VOICE", "1")
    # 原声优先靠 best-of-K 采样择"原声率最高"的分配，K=1 会关掉该优化 -> 原声变少;
    # 这里默认 3, 让"有贴题原声的段"尽量用真声（可用 WHQ_PLAN_BEST_OF_K 覆盖）。
    os.environ.setdefault("WHQ_PLAN_BEST_OF_K", "3")

    # 配音克隆模型可选：cosyvoice(默认) / voxcpm。voxcpm 走独立 conda env + HF 镜像。
    if (voice_model or "cosyvoice").lower() == "voxcpm":
        _here = os.path.dirname(os.path.abspath(__file__))
        os.environ["WHQ_REAL_TTS_SCRIPT"] = os.path.join(_here, "run_voxcpm2_zero_shot.py")
        os.environ["WHQ_REAL_TTS_PYTHON"] = os.getenv(
            "VOXCPM_PYTHON", "/root/miniconda3/envs/voxcpm/bin/python")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("WHQ_VOXCPM_CFG", "2.0")
        os.environ.setdefault("WHQ_VOXCPM_STEPS", "30")
        print("[runner] 配音模型: VoxCPM2 ({})".format(os.environ["WHQ_REAL_TTS_PYTHON"]), flush=True)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.abspath(out_path))[0]
    work_dir = os.path.join(out_dir, "_whq_work")

    inp = whq_input.build_whq_inputs(
        material_understanding, template, ref_video, work_dir,
        run_asr=run_asr, topic=product_name,
        dna_path=dna_md, assets_path=assets_json, asr_path=asr_json)

    return {"inp": inp, "work_dir": work_dir, "out_dir": out_dir, "stem": stem,
            "ref_video": ref_video, "product_name": product_name, "dna_md": dna_md,
            "effective_mode": effective_mode, "arbitrate_weak": arbitrate_weak}


def _build_process(dna, details, dna_md):
    """把复刻链路的"思考过程 + 选素材过程"整理成给前端展示的结构。"""
    if dna_md:
        dna_source = "Split 深度理解（真实 DNA）"
    elif isinstance(dna, dict) and dna.get("_vlm"):
        dna_source = "VLM 重建"
    else:
        dna_source = "模板兜底"
    beats = []
    if isinstance(dna, dict):
        beats = list((dna.get("content_structure") or {}).get("key_beats") or [])

    arb_by_slot = {a.get("slot_id"): a for a in (details.get("arbitration") or [])}
    cap_by_slot = {}
    for it in details.get("tts_items") or []:
        sid = it.get("slot_id") or it.get("slot")
        if sid:
            cap_by_slot[sid] = it.get("caption_text") or it.get("text") or ""

    steps = []
    for s in details.get("segments") or []:
        sid = s.get("slot_id")
        cand = s.get("best_candidate") or {}
        va = s.get("voice_align")
        decision = "原声保留" if va else ("AIGC 补拍" if s.get("is_t2v") else "克隆配音")
        reason = va or ""
        arb = arb_by_slot.get(sid)
        if arb:
            act = {"swap": "换素材", "drop": "删段", "keep_mismatch": "保留(略跑题)"}.get(arb.get("action"), arb.get("action"))
            reason = "【{}】{}".format(act, arb.get("reason", "")) + (("；" + reason) if reason else "")
        mat = cand.get("source_path", "").split("/")[-1]
        if mat and (cand.get("start") is not None):
            mat = "{} @ {:.1f}-{:.1f}s".format(mat, cand.get("start", 0.0), cand.get("end", 0.0))
        steps.append({
            "slot_id": sid,
            "beat": s.get("beat_desc", ""),
            "decision": decision,
            "material": mat,
            "reason": reason,
            "caption": cap_by_slot.get(sid, ""),
        })

    dropped = [{"slot_id": a.get("slot_id"), "reason": a.get("reason", "")}
               for a in (details.get("arbitration") or []) if a.get("action") == "drop"]
    return {"dna_source": dna_source, "beats": beats, "steps": steps, "dropped": dropped}
