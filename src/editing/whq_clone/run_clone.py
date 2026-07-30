"""run_clone — whq 结构级复刻一键编排(带配音+文案, 移除原声, 不重复镜头)。

流程:
  参考视频 DNA -> 结构级段落(不逐镜) -> 用户素材 1:1 不重复分配(edit_planner)
  -> [可选 seedance 补缺口] -> 硬剪无声 base(clone_builder, 移除原声)
  -> 两者结合配音(voiceover: 用户ASR+LLM改写+声音克隆)
  -> 烧录文案字幕(+可选迁移参考 BGM)(finisher)

用法:
  # 复用某个 slug 的 understanding+ASR 产物(自动发现 DNA/all_user_assets/all_source_asr)
  python -m whq.run_clone --slug 牛肉饼_ref_牛肉饼_ref_vlm3.7_full6 \
      --ref Resource/Ref_Video/牛肉饼_Ref.mov --product-name 牛肉饼 \
      --out whq/video/牛肉饼_clone.mp4

  # 离线确定性分配(不调 LLM)
  python -m whq.run_clone --slug ... --ref ... --no-llm

  # seedance 补缺口 + 迁移参考 BGM
  python -m whq.run_clone --slug ... --ref ... --mode seedance --migrate-bgm
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import REPO, legacy
from reference_shots import load_dna
from asset_index import load_candidates
from edit_planner import plan_edit
from clone_builder import build_base
from seedance_fill import fill_gaps
from voiceover import run_voiceover, load_user_speech
import voice_policy
import pipeline_utils
from finisher import finish


# 影响「编排结果」的 env：变了就让 plan 阶段缓存失效（否则改了阈值重跑却复用旧编排）。
_PLAN_ENV = (
    "WHQ_LEGACY", "WHQ_CTA_ORIGINAL", "WHQ_PREFER_ORIGINAL_VOICE", "WHQ_ORIGINAL_VOICE_BONUS",
    "WHQ_PLAN_BEST_OF_K", "WHQ_PLAN_COHERENCE_REPAIR", "WHQ_MIN_PRODUCT_SHOTS",
    "WHQ_PRODUCT_SHOT_PENALTY", "WHQ_SLOT_PIN", "WHQ_MAX_BEATS", "WHQ_ARB_MAX_SWAP",
    "WHQ_VOICE_MIN_CHARS", "WHQ_VOICE_MIN_COVERAGE", "WHQ_VOICE_MIN_ATEMPO",
    "WHQ_VOICE_MAX_ATEMPO", "WHQ_FACE_MIN_RATIO", "WHQ_SENT_GAP", "WHQ_ORIGINAL_SENT_GUARD",
)


def _plan_fingerprint(dna_obj, cands, asr, ref_asr_items, mode, total_duration,
                      use_llm, model, gap_threshold, arbitrate_weak, ref, no_stretch=False):
    """编排阶段的输入指纹：DNA 节拍 + 候选素材(id/源/区间) + 参考语速 + 参数 + 相关 env。"""
    beats = ((dna_obj or {}).get("content_structure") or {}).get("key_beats") or []
    cand_key = [[c.get("global_asset_id"), c.get("source_path"), c.get("start"), c.get("end")]
                for c in (cands or [])]
    return pipeline_utils.fingerprint(
        beats, (dna_obj or {}).get("duration_estimate"), cand_key, len(ref_asr_items or []),
        [mode, total_duration, bool(use_llm), model, gap_threshold, bool(arbitrate_weak),
         bool(no_stretch)],
        pipeline_utils.file_fingerprint([p for p in (asr, ref) if p]),
        pipeline_utils.env_fingerprint(_PLAN_ENV))


# 影响「文案 + 配音」的 env：变了就重跑 TTS（否则改了文案审查/克隆模型却复用旧配音）。
_VOICE_ENV = (
    "WHQ_LEGACY", "WHQ_SCRIPT_REVIEW_APPLY_REWRITE", "WHQ_SCRIPT_TEXT_OVERRIDE",
    "WHQ_STRICT_SCRIPT", "WHQ_TARGET_CHARS_SLACK", "WHQ_REAL_TTS_SCRIPT",
    "WHQ_REAL_TTS_PYTHON", "TTS_CHARS_PER_SECOND", "TTS_CPS_MIN", "TTS_CPS_MAX",
    "TTS_MAX_ATEMPO", "WHQ_TTS_MAX_ATEMPO", "TTS_RESYNTH_ROUNDS", "TTS_OVERLAY_VOLUME",
    "WHQ_VOXCPM_CFG", "WHQ_VOXCPM_STEPS", "WHQ_VOXCPM_REF_WAV",
)


def _prior_speeds(manifest):
    """每段在硬剪阶段**已经**施加的音画倍率: source_take/target_duration。

    原声段为对齐完整句会把源窗口(take)压进槽时长(target)，clone_builder/voice_policy 已
    用同倍率 atempo 变速。若 pace_match 再乘一次，最坏是 1.35×1.4≈1.9 倍的"快放"失真。
    把已用倍率传给 pace_match 做封顶。克隆段无 source_take -> 1.0。
    """
    out = {}
    for m in manifest or []:
        sid = m.get("slot_id")
        take = m.get("source_take")
        target = m.get("target_duration")
        if sid and take and target:
            try:
                out[sid] = round(float(take) / float(target), 4)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return out


def _latest(paths):
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def _exact_window_text(speech, source_path, start, dur, sent_gap=0.35):
    """返回某源片 [start, start+dur] 音频窗口内的逐字 ASR 文本 + 吸附到句首后的实际起点。

    - token 中心落窗即计入；
    - 若窗口开头是"上一句的残尾"（起点后不久就有 >sent_gap 的停顿，且停顿前只有很短几个字），
      把起点吸附到该停顿之后的句首，去掉半句残尾（如 S02 开头的"贵"）。
    返回 (text, snapped_start)。
    """
    end = start + dur
    toks = []
    for rec in speech or []:
        if rec.get("source_path") != source_path:
            continue
        for it in rec.get("asr_items") or []:
            try:
                s, e = float(it.get("start") or 0), float(it.get("end") or 0)
            except (TypeError, ValueError):
                continue
            if start - 0.15 <= (s + e) / 2.0 <= end + 0.15:
                toks.append((s, e, str(it.get("text") or "")))
    if not toks:
        return "", start, start + dur
    # 吸附句首：若前 <=3 个字后就有个大停顿，说明开头是残尾，从停顿后开始
    for i in range(len(toks) - 1):
        gap = toks[i + 1][0] - toks[i][1]
        if gap >= sent_gap and (i + 1) <= 3:
            toks = toks[i + 1:]
            break
    # 吸附句尾：若结尾是"半句残尾"(最后 <=3 个字前有个大停顿)，收回到那个停顿(说完整句)
    for k in range(len(toks) - 1, 0, -1):
        gap = toks[k][0] - toks[k - 1][1]
        if gap >= sent_gap and (len(toks) - k) <= 3:
            toks = toks[:k]
            break
    if not toks:
        return "", start, start + dur
    snapped_start = round(max(start, toks[0][0] - 0.05), 3)
    snapped_end = round(min(start + dur + 0.2, toks[-1][1] + 0.12), 3)
    text = "".join(t for _, _, t in toks).strip()
    return text, snapped_start, snapped_end


def _utterances_from_speech(speech, min_dur=0.8, max_dur=9.0, gap=0.6):
    """把用户素材的逐字 ASR 切成"句子级"口播窗口 [(source_path, start, end, text)]。"""
    outs = []
    for rec in speech or []:
        sp = rec.get("source_path")
        items = rec.get("asr_items") or []
        if not sp or not items:
            continue
        cur = []
        last_e = None
        for it in items:
            try:
                s, e = float(it.get("start") or 0), float(it.get("end") or 0)
            except (TypeError, ValueError):
                continue
            if cur and last_e is not None and s - last_e > gap:
                st, en = cur[0][0], cur[-1][1]
                txt = "".join(t for _, _, t in cur).strip()
                if min_dur <= en - st <= max_dur and len(txt) >= 3:
                    outs.append((sp, st, en, txt))
                cur = []
            cur.append((s, e, str(it.get("text") or "")))
            last_e = e
        if cur:
            st, en = cur[0][0], cur[-1][1]
            txt = "".join(t for _, _, t in cur).strip()
            if min_dur <= en - st <= max_dur and len(txt) >= 3:
                outs.append((sp, st, en, txt))
    return outs


def _cta_original():
    """结尾段是否用「从用户原声里 LLM 选一句购买号召」。

    关闭时回到迁移前行为: plan_edit(cta_last=True) 由编排确定性安排结尾段, 结尾走克隆配音。
    WHQ_CTA_ORIGINAL=0 / WHQ_LEGACY=1 关闭。
    """
    if legacy():
        return False
    return os.getenv("WHQ_CTA_ORIGINAL", "1") not in ("0", "false", "False")


def _assign_cta_original(last_seg, decisions, speech):
    """结尾 CTA 段：让 **LLM** 从用户所有原声句子里挑一句真正的"购买号召/催单"用作结尾原声
    （与参考对齐、尽量保留用户真声）。找不到则不动（回退克隆，review_script 的 cta_ending 生成号召）。
    不使用写死的关键词正则，纯 LLM 判断。"""
    sid = last_seg.get("slot_id")
    utts = _utterances_from_speech(speech)
    if not utts:
        return
    try:
        from pipeline_utils import ask_qianfan, loads_with_repair
    except Exception:
        return
    beat = last_seg.get("beat_desc", "")
    lines = ["{}. 「{}」".format(i, u[3][:50]) for i, u in enumerate(utts)]
    prompt = (
        "这是短视频结尾段，作用是「{}」——需要一句**明确的购买号召/催单**（如叫大家下单、"
        "买、上链接、赶紧入手之类的号召性语气）。下面是用户素材里的真实口播句子，请选出**最像"
        "购买号召/催单**的一句用作结尾；如果**没有任何一句是购买号召**，返回 -1。\n{}\n"
        "只返回 JSON：{{\"index\": <句子编号或 -1>, \"reason\": \"<一句话>\"}}"
    ).format(beat, "\n".join(lines))
    try:
        text, _ = ask_qianfan([{"role": "user", "content": prompt}], temperature=0.1)
        obj = loads_with_repair(text)
        idx = int(obj.get("index", -1))
    except Exception as exc:  # noqa: BLE001
        print("[run_clone] 结尾CTA LLM 选句失败(回退克隆): {}".format(str(exc)[:120]), flush=True)
        return
    if idx < 0 or idx >= len(utts):
        print("[run_clone] 结尾CTA: 用户原声里无购买号召, 回退克隆生成", flush=True)
        return
    sp, st, en, txt = utts[idx]
    last_seg["best_candidate"] = {"source_path": sp, "start": st, "end": en,
                                  "duration": round(en - st, 2),
                                  "global_asset_id": "cta_original", "text": txt}
    last_seg["is_gap"] = False
    decisions[sid] = {"voice_source": "original", "window_text": txt, "win_start": st,
                      "audio_take": round(en - st, 2), "decision_basis": "cta_original"}
    print("[run_clone] 结尾CTA用用户原声购买号召(LLM选): {} @{:.1f}-{:.1f} '{}'".format(
        sp.split("/")[-1], st, en, txt[:24]), flush=True)


def discover_from_slug(slug):
    base = os.path.join(REPO, "outputs", slug)
    dna = _latest(glob.glob(os.path.join(base, "dna_understanding", "*dna_template*.md")))
    assets = _latest(glob.glob(os.path.join(base, "user_understanding", "*", "all_user_assets.json")))
    asr = _latest([os.path.join(REPO, "outputs", "source_asr", slug, "all_source_asr.json")]
                  + glob.glob(os.path.join(REPO, "outputs", "source_asr", slug, "all_source_asr.json")))
    return dna, assets, asr


def collect_reference_asr(ref_video, cache_dir):
    """对参考视频跑 Qwen3-ASR, 返回逐字 asr_items(参考语速用)。

    让配音每段语速跟随参考对应段落: edit_planner 按 ref_time_range 聚合出每段真实字/秒。
    实调委托 asr_tokens(同一套 batch_qwen3_asr + **输入指纹**缓存)——旧实现只看
    all_source_asr.json 是否存在, 同一 work_dir 换了参考视频会静默复用上次的 ASR,
    ref_cps 全错且无任何报错。无参考/无环境/失败 -> [], 上游回退全局默认 cps。
    """
    import asr_tokens
    return asr_tokens.reference_asr_items(ref_video, cache_dir)


def run(ref=None, dna=None, assets=None, asr=None, out=None, slug=None,
        mode="hardcut", use_llm=True, model=None, product_name="",
        gap_threshold=0.4, stretch=True, use_reference_image=True, max_fill=None,
        total_duration=None, migrate_bgm=False, bgm_endpoint="online", ref_speed=True, pace_match=True):
    if slug and (not dna or not assets or not asr):
        d, a, r = discover_from_slug(slug)
        dna, assets, asr = dna or d, assets or a, asr or r
    if not assets or not dna:
        raise SystemExit("必须提供 --dna + --assets 或可发现产物的 --slug")
    out = out or os.path.join(REPO, "outputs", slug or "whq_clone", "{}_clone.mp4".format(slug or "whq"))
    out_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.abspath(out))[0]
    work_dir = os.path.join(out_dir, "_whq_work")
    os.makedirs(work_dir, exist_ok=True)

    dna_obj = load_dna(dna)
    cands = load_candidates(assets)
    print("[run_clone] 候选素材片段: {}".format(len(cands)))

    # 0) 参考语速采集(可选): 跑参考视频 ASR, 供 edit_planner 按段算 ref_cps,
    #    让配音每段语速跟随参考对应段落(快段配音也快, 慢段也慢); 失败则全段回退默认语速。
    ref_asr_items = []
    if ref_speed:
        ref_asr_items = collect_reference_asr(ref, os.path.join(work_dir, "ref_asr"))

    return run_from_objects(
        dna_obj, cands, asr, ref_asr_items, out, ref=ref, mode=mode, use_llm=use_llm,
        model=model, product_name=product_name, gap_threshold=gap_threshold,
        stretch=stretch, use_reference_image=use_reference_image, max_fill=max_fill,
        total_duration=total_duration, migrate_bgm=migrate_bgm,
        bgm_endpoint=bgm_endpoint, pace_match=pace_match, work_dir=work_dir)


def plan_from_objects(dna_obj, cands, asr, ref_asr_items, stem, *, ref=None,
                      mode="hardcut", use_llm=True, model=None, product_name="",
                      gap_threshold=0.4, use_reference_image=True, max_fill=None,
                      total_duration=None, work_dir=None, arbitrate_weak=False,
                      no_stretch=False):
    """**只做编排**：结构段落 → 1:1 分配 → 原声/克隆决策 → 句子级对窗 → 落 plan.json。

    从 run_from_objects 里抽出来的前半段，两个消费方共用：
    - run_from_objects：继续往下 build_base → voiceover → finisher，一步出成片（workflow 形态）；
    - runner.plan_whq_clone：到此为止，把 segments/decisions 映射成 strategy + connector_context，
      交给 editing/loop.py 的 Agent 剪辑循环出片（Agent 形态）。

    返回 dict(segments, decisions, speech, n_gap, fill_stat, arbitration, plan_path)。

    可续跑：编排阶段最贵（best-of-K LLM 分配 + VLM 画面仲裁 + 人脸探测），因此结果按
    「输入指纹」缓存在 plan.json 里；同一 stem 重跑且输入/相关 env 未变时直接复用，
    只跑下游剪辑/配音。``WHQ_RESUME=0`` 可强制重新编排。
    """
    work_dir = work_dir or os.path.join(os.path.dirname(os.path.abspath(stem)), "_whq_work")
    os.makedirs(work_dir, exist_ok=True)
    plan_path = stem + "_plan.json"

    # 1) 结构级规划 + 1:1 不重复分配
    #    用户口播 ASR 提前加载: 原声优先(WHQ_PREFER_ORIGINAL_VOICE=1)时, 分配阶段
    #    偏向选到窗口内自带贴题口播的素材, 使下游 voice_policy 能保留原声。
    speech = load_user_speech(asr)

    fp = _plan_fingerprint(dna_obj, cands, asr, ref_asr_items, mode, total_duration,
                           use_llm, model, gap_threshold, arbitrate_weak, ref, no_stretch)
    cached = pipeline_utils.load_stage(plan_path, fp)
    if cached and cached.get("segments") and cached.get("decisions"):
        print("[run_clone] 复用已有编排(输入未变): {}".format(plan_path), flush=True)
        return {"segments": cached["segments"], "decisions": cached["decisions"],
                "speech": speech, "n_gap": cached.get("n_gap", 0),
                "fill_stat": cached.get("fill_stat") or {"filled": 0},
                "arbitration": cached.get("arbitration") or [],
                "plan_path": plan_path}

    plan = plan_edit(dna_obj, cands, total_duration=total_duration,
                     use_llm=use_llm, model=model, gap_threshold=gap_threshold,
                     ref_asr_items=ref_asr_items, speech_records=speech,
                     cta_last=not _cta_original())
    segments = plan["segments"]
    n_gap = sum(1 for s in segments if s.get("is_gap"))
    plan_total = sum(float(s.get("target_duration") or 0.0) for s in segments)
    ref_total = float((dna_obj or {}).get("duration_estimate") or 0.0)
    print("[run_clone] 结构段落 {} 段, 缺口 {}, 计划总时长 {:.2f}s (参考 {:.2f}s{})".format(
        len(segments), n_gap, plan_total, ref_total,
        ", 偏差 {:+.2f}s".format(plan_total - ref_total) if ref_total else ""))

    # 2) seedance 补缺口(可选)
    fill_stat = {"filled": 0}
    if mode == "seedance" and n_gap:
        segments, fill_stat = fill_gaps(
            segments, os.path.join(work_dir, "t2v"), product_name=product_name,
            use_reference_image=use_reference_image, max_fill=max_fill)
        print("[run_clone] seedance 补拍: {}".format(fill_stat))

    # 2.5) 原声/克隆决策 + 句子级对窗。定义为局部函数，因换素材仲裁后需重跑一次。
    def _decide_voices(segs):
        pre_manifest = []
        for seg in segs:
            cand = seg.get("best_candidate") or {}
            target = float(seg.get("target_duration") or 0.0)
            avail = float(cand.get("duration") or 0.0)
            # Agent 形态（no_stretch）：成片不会把短素材拉伸去填满节拍槽位，实际用到的时长
            # 就是 min(槽位, 素材可用)。若仍按槽位判定，voice_policy 会算出"素材要拉伸 4x、
            # 原声得放慢到 0.25 倍"而把本来有真声的段全降级成克隆（实测 original 只剩 1 段）。
            if no_stretch and avail > 0 and target > avail:
                target = avail
            pre_manifest.append({
                "slot_id": seg.get("slot_id"),
                "target_duration": target or seg.get("target_duration"),
                "source_path": (seg.get("t2v_path") if seg.get("is_t2v") else cand.get("source_path")),
                "source_start": cand.get("start", 0.0),
                "source_avail": cand.get("duration", 0.0),
            })
        return voice_policy.decide(pre_manifest, speech)

    decisions = _decide_voices(segments)

    # 2.6) 画面仲裁（不补拍时·保原声不删段）：原声段锁定保护；非原声段 VLM 比画面是否贴近参考，
    #      不贴则从未用候选换更贴的素材，换不到就保留原分配(降级克隆)。任何段都不删，保住结构。
    dropped_arb = []
    if arbitrate_weak and mode != "seedance":
        try:
            import weak_arbiter
            # 视觉贴近参考：从参考视频抽各节拍的中帧，供仲裁做 VLM 画面比对
            ref_frames = None
            if ref:
                try:
                    import visual_match
                    ref_frames = visual_match.reference_frames(
                        ref, len(segments), os.path.join(work_dir, "ref_frames"))
                except Exception as exc:  # noqa: BLE001
                    print("[run_clone] 参考帧抽取失败(退化为文本仲裁): {}".format(str(exc)[:160]), flush=True)
                    ref_frames = None
            # 不删段后单轮即可：换素材 -> 换不到保留(降级克隆)，之后重跑 voice_policy 定原声/克隆。
            segments, arb = weak_arbiter.arbitrate(
                segments, cands, decisions, work_dir, ref_frames=ref_frames)
            decisions = _decide_voices(segments)  # 换素材后重定原声/克隆
            dropped_arb = arb or []
        except Exception as exc:  # noqa: BLE001
            print("[run_clone] 仲裁失败(保留原分配): {}".format(str(exc)[:160]), flush=True)

    # 结尾 CTA 段优先用**用户原声里的购买号召**（如"赶紧吧/链接放下面"）——与参考视频对齐、
    # 尽量保留用户原声；用户没有这类原声时才回退克隆生成 CTA（review_script cta_ending 兜底）。
    if segments and _cta_original():
        _assign_cta_original(segments[-1], decisions, speech)

    apply_voice_windows(segments, decisions)

    # 原声段字幕对齐：把 window_text 改成"实际抽取的音频窗口 [start, start+take] 内的逐字文本"，
    # 而不是 voice_policy 的整句组文本——否则字幕(整段)远长于音频(只截了 take 秒) -> 字幕≠语音。
    for seg in segments:
        d = decisions.get(seg.get("slot_id")) or {}
        if d.get("voice_source") != "original":
            continue
        sp = (seg.get("best_candidate") or {}).get("source_path")
        st = seg.get("source_start_override")
        tk = seg.get("source_take")
        if sp and st is not None and tk:
            wt, snapped_s, snapped_e = _exact_window_text(speech, sp, float(st), float(tk))
            if wt:
                d["window_text"] = wt
                # 吸附句首/句尾后同步平移音频/画面窗口，字幕=音频=画面一致，且原声说完整句
                seg["source_start_override"] = snapped_s
                seg["source_take"] = round(max(0.3, snapped_e - snapped_s), 3)

    plan_path = stem + "_plan.json"
    pipeline_utils.save_stage(plan_path, fp, {
        "reference_dna": dna_obj, "n_candidates": len(cands), "mode": mode,
        "fill_stat": fill_stat, "arbitration": dropped_arb, "n_gap": n_gap,
        # 记下这轮用的词级 ASR 产物: 「只出片」入口(runner.edit_from_strategy)靠它拿原声窗口
        # 文本与文案取材, 否则会因为找不到 ASR 而降级成无配音+空字幕。
        "asr_path": os.path.abspath(asr) if asr and os.path.exists(asr) else "",
        "segments": segments, "decisions": decisions,
    })
    return {"segments": segments, "decisions": decisions, "speech": speech,
            "n_gap": n_gap, "fill_stat": fill_stat, "arbitration": dropped_arb,
            "plan_path": plan_path}


def apply_voice_windows(segs, decisions):
    """把原声决策的对窗结果写回 segment（clone_builder / Agent 剪辑都按它切画面）。

    原声段视频窗口跟随有声时长 audio_take，音画同倍率回填，句子在停顿处收束。
    换素材后可重复调用（先清旧值，因为可能从原声变克隆）。
    """
    for seg in segs:
        d = decisions.get(seg.get("slot_id")) or {}
        seg.pop("source_start_override", None)
        seg.pop("source_take", None)
        seg.pop("voice_align", None)
        if d.get("voice_source") != "original":
            continue
        if d.get("win_start") is not None:
            seg["source_start_override"] = d.get("win_start")
        take = d.get("audio_take") or d.get("win_take")
        if take:
            seg["source_take"] = float(take)
        seg["voice_align"] = d.get("decision_basis")


def render_from_plan(segments, decisions, dna_obj, asr, ref_asr_items, out, *, ref=None,
                     stretch=True, product_name="", model=None, migrate_bgm=False,
                     bgm_endpoint="online", pace_match=True, work_dir=None):
    """**只做出片**：硬剪无声 base → 配音 → 字幕/BGM → 语速贴参考。

    从 run_from_objects 里抽出来的后半段，两个消费方共用：
    - run_from_objects：编排完接着出片（workflow 一步流）；
    - runner.edit_from_strategy：从已落盘的 selected_editing_strategy.json 复原
      segments/decisions 后出片（前端「workflow 剪辑」按钮，不再走 Split 主链路）。

    返回 dict(final, base, voiced, manifest, manifest_path, tts_items)。
    """
    out_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.abspath(out))[0]
    work_dir = work_dir or os.path.join(out_dir, "_whq_work")
    os.makedirs(work_dir, exist_ok=True)

    # 3) 硬剪无声 base(移除原声)。可续跑: 段落窗口未变且 base/段文件都还在则跳过重编码。
    base_video = stem + "_base.mp4"
    manifest_path = stem + "_manifest.json"
    seg_dir = os.path.join(work_dir, "_segments")
    base_fp = pipeline_utils.fingerprint(
        [[s.get("index"), s.get("slot_id"), s.get("target_duration"),
          s.get("source_start_override"), s.get("source_take"), s.get("is_t2v"),
          s.get("t2v_path"), (s.get("best_candidate") or {}).get("source_path"),
          (s.get("best_candidate") or {}).get("start")] for s in segments], bool(stretch))
    cached_base = pipeline_utils.load_stage(manifest_path, base_fp)
    if (cached_base and os.path.exists(base_video)
            and all(os.path.exists(m.get("segment_file") or "") for m in cached_base["segments"])):
        manifest = cached_base["segments"]
        print("[run_clone] 复用已有硬剪 base(段落未变): {}".format(base_video), flush=True)
    else:
        base_video, manifest = build_base(segments, base_video, stretch=stretch,
                                          work_dir=seg_dir)
        pipeline_utils.save_stage(manifest_path, base_fp,
                                  {"output": base_video, "segments": manifest})

    # 4) 两者结合配音。TTS 是全链路最贵的一步(克隆逐段合成), 同样按指纹续跑。
    voiced = stem + "_voiced.mp4"
    tts_dir = os.path.join(work_dir, "tts")
    overlay_plan = os.path.join(tts_dir, "tts_overlay_plan.json")
    # 参考口播全文只用于「防照搬」核验(voiceover 里 n-gram 对照), 不作为文案素材
    ref_text = "".join(str(it.get("text") or "") for it in (ref_asr_items or []))
    voice_fp = pipeline_utils.fingerprint(
        base_fp, decisions, product_name, model, len(ref_text),
        pipeline_utils.env_fingerprint(_VOICE_ENV))
    voice_stage_path = stem + "_voice_stage.json"
    cached_voice = pipeline_utils.load_stage(voice_stage_path, voice_fp)
    if cached_voice and os.path.exists(voiced) and os.path.exists(overlay_plan):
        voice_video = voiced
        print("[run_clone] 复用已有配音(文案/决策未变): {}".format(voiced), flush=True)
    else:
        voiced_out, vstat = run_voiceover(manifest, dna_obj, asr, base_video, voiced,
                                          tts_dir, product_name=product_name, model=model,
                                          ref_text=ref_text, decisions=decisions)
        print("[run_clone] 配音: {}".format({k: v for k, v in vstat.items() if k != "script"}))
        voice_video = voiced_out or base_video
        if voiced_out:
            pipeline_utils.save_stage(voice_stage_path, voice_fp, {"voiced": voiced_out})

    # 5) 文案字幕 (+可选参考 BGM)
    # 字幕文本必须取 build_tts_overlay 的**产出** tts_overlay_plan.json: 它反映了
    # compress_overlong_tts 对超长段的压缩改写(=实际念出来的文本); 输入 tts_plan_input.json
    # 是压缩前原文, 用它会导致字幕(全句)与配音(压缩短句)不同步。缺失时才回退输入 plan。
    tts_plan_path = overlay_plan if os.path.exists(overlay_plan) else os.path.join(tts_dir, "tts_plan_input.json")
    tts_items = json.load(open(tts_plan_path, encoding="utf-8")).get("items", []) if os.path.exists(tts_plan_path) else []
    final = finish(voice_video, tts_items, out, os.path.join(work_dir, "cap"),
                   ref_video=ref, out_final=stem + "_bgm.mp4",
                   migrate_bgm=migrate_bgm, endpoint=bgm_endpoint)

    # 5.5) 语速贴参考(逐段): 成片每段实际语速(受字数预算/atempo cap 限制, 常 ~3-4 字/秒)
    # 追不上参考爆款(~7 字/秒)。默认只提**口播语速**、画面段长不变(pace_match.KEEP_TOTAL),
    # 因此成片总时长 = 参考总时长; 原声段锁 1.0x 保口型。
    if pace_match and tts_items and os.path.exists(overlay_plan):
        try:
            from pace_match import match_pace
            paced_tmp = stem + "_paced_tmp.mp4"
            match_pace(final, overlay_plan, paced_tmp,
                       prior_speeds=_prior_speeds(manifest))
            os.replace(paced_tmp, final)
        except Exception as exc:
            print("[run_clone] 语速贴参考失败(保留原速成片): {}".format(str(exc)[:200]))

    print("=" * 60)
    print("[run_clone] SUCCESS")
    print("  base(无声)  : {}".format(base_video))
    print("  配音        : {}".format(voice_video))
    print("  最终成片    : {}".format(final))
    print("=" * 60)
    return {"final": final, "base": base_video, "voiced": voice_video,
            "manifest": manifest, "manifest_path": manifest_path, "tts_items": tts_items}


def run_from_objects(dna_obj, cands, asr, ref_asr_items, out, *, ref=None,
                     mode="hardcut", use_llm=True, model=None, product_name="",
                     gap_threshold=0.4, stretch=True, use_reference_image=True,
                     max_fill=None, total_duration=None, migrate_bgm=False,
                     bgm_endpoint="online", pace_match=True, work_dir=None,
                     return_details=False, arbitrate_weak=False):
    """内存驱动的编排主体（Agent 集成入口 runner.py 复用此函数）。

    dna_obj/cands/ref_asr_items 已是内存对象（不再从文件 load）；``asr`` 仍是
    all_source_asr.json 路径（voiceover 内部按路径读原声）。CLI 的 run() 与 Agent 的
    runner 都调用它，保证编排逻辑单一事实来源。``return_details=True`` 时返回 dict
    （含 segments/manifest/tts_items/各产物路径），供 strategy_out 生成契约②。
    """
    out_dir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.abspath(out))[0]
    work_dir = work_dir or os.path.join(out_dir, "_whq_work")
    os.makedirs(work_dir, exist_ok=True)

    # 1~2.6) 编排（结构段落/分配/原声决策/对窗/plan.json）——与 Agent 形态共用同一实现
    plan_info = plan_from_objects(
        dna_obj, cands, asr, ref_asr_items, stem, ref=ref, mode=mode, use_llm=use_llm,
        model=model, product_name=product_name, gap_threshold=gap_threshold,
        use_reference_image=use_reference_image, max_fill=max_fill,
        total_duration=total_duration, work_dir=work_dir, arbitrate_weak=arbitrate_weak)
    segments = plan_info["segments"]
    decisions = plan_info["decisions"]

    # 3~5.5) 出片
    rendered = render_from_plan(
        segments, decisions, dna_obj, asr, ref_asr_items, out, ref=ref, stretch=stretch,
        product_name=product_name, model=model, migrate_bgm=migrate_bgm,
        bgm_endpoint=bgm_endpoint, pace_match=pace_match, work_dir=work_dir)

    print("  规划        : {}".format(plan_info["plan_path"]))
    if return_details:
        return {
            "final": rendered["final"], "base": rendered["base"], "voiced": rendered["voiced"],
            "plan_path": plan_info["plan_path"], "manifest_path": rendered["manifest_path"],
            "segments": segments, "manifest": rendered["manifest"],
            "tts_items": rendered["tts_items"], "n_gap": plan_info["n_gap"],
            "fill_stat": plan_info["fill_stat"], "arbitration": plan_info["arbitration"],
        }
    return rendered["final"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="whq 结构级复刻一键编排(配音+文案)")
    ap.add_argument("--ref", help="参考视频")
    ap.add_argument("--dna", help="DNA md (省略则从 --slug 发现)")
    ap.add_argument("--assets", help="all_user_assets.json / understanding 目录")
    ap.add_argument("--asr", help="all_source_asr.json (省略则从 --slug 发现)")
    ap.add_argument("--slug", help="从 outputs/<slug> 自动发现 DNA/assets/ASR")
    ap.add_argument("--out", help="最终成片 mp4")
    ap.add_argument("--mode", choices=["hardcut", "seedance"], default="hardcut")
    ap.add_argument("--no-llm", action="store_true", help="用离线确定性分配")
    ap.add_argument("--model", help="LLM 模型名")
    ap.add_argument("--product-name", default="")
    ap.add_argument("--gap-threshold", type=float, default=0.4)
    ap.add_argument("--total-duration", type=float, help="覆盖参考总时长")
    ap.add_argument("--no-stretch", action="store_true", help="短素材不拉伸")
    ap.add_argument("--no-reference-image", action="store_true", help="seedance 补拍不用素材帧作参考图")
    ap.add_argument("--max-fill", type=int, help="seedance 最多补拍几段")
    ap.add_argument("--migrate-bgm", action="store_true", help="迁移参考视频真实 BGM(需 meishe 环境)")
    ap.add_argument("--bgm-endpoint", default="online", choices=["online", "direct"])
    ap.add_argument("--no-pace-match", action="store_true",
                    help="关闭成片整体语速贴参考(逐段加速)")
    ap.add_argument("--no-ref-speed", action="store_true",
                    help="不采集参考语速(跳过参考 ASR), 配音全段用默认语速 TTS_CHARS_PER_SECOND")
    args = ap.parse_args(argv)
    run(ref=args.ref, dna=args.dna, assets=args.assets, asr=args.asr, out=args.out,
        slug=args.slug, mode=args.mode, use_llm=not args.no_llm, model=args.model,
        product_name=args.product_name, gap_threshold=args.gap_threshold,
        total_duration=args.total_duration, stretch=not args.no_stretch,
        use_reference_image=not args.no_reference_image, max_fill=args.max_fill,
        migrate_bgm=args.migrate_bgm, bgm_endpoint=args.bgm_endpoint,
        ref_speed=not args.no_ref_speed, pace_match=not args.no_pace_match)


if __name__ == "__main__":
    main()
