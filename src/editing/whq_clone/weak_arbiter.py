"""weak_arbiter — 画面级仲裁：LLM 判定 + VLM 二次校验，跑题段先换镜、换不到再删。

背景：whq 的匹配用素材"画面描述+关键词"对 beat 打分，但开原声优先(WHQ_PREFER_ORIGINAL_VOICE)
后会偏向"自带口播"的候选，可能压过"画面更贴"的候选（实测：配料表特写 beat 选了"手持盒子
讲话"的口播镜，而真正的"配料表特写"空镜没被选中）。LLM 文本分对"画面是否贴题"也不可靠。

策略（用户决策·选项2）：对**所有段**（含原声段）——
  1) VLM 二次校验：抽当前候选帧，判是否贴合 beat 语义。
  2) 跑题 → 先**换镜**：从未使用候选里按画面相关性排序，逐个 VLM 校验，选第一个贴合的换上
     （画面优先；换上的段若无口播，下游 voice_policy 自然改克隆配音）。
  3) 换不到贴题素材 → LLM 判定 drop(删段，连贯优先) / keep。
本模块在 voice_policy 决策**之前**跑，换镜后再由 voice_policy 重新定原声/克隆。
"""
import os
import subprocess

from _common import FFMPEG
import as_core
import visual_match
from pipeline_utils import parallel_map
from reference_dna import _run_async
from shot_matcher import score_candidate_deterministic

MAX_SWAP_TRIES = int(os.getenv("WHQ_ARB_MAX_SWAP", "3"))


def _extract_frame(src, start, dur, out_png):
    if not src or not os.path.exists(src):
        return False
    t = max(0.0, float(start) + max(0.0, float(dur)) / 2.0)
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", "{:.3f}".format(t),
           "-i", src, "-frames:v", "1", "-vf", "scale=480:-2", out_png]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_png)
    except Exception:
        return False


def _vlm_matches(frame_png, beat_desc):
    """VLM 看帧判断是否**明显跑题**（按叙事功能/镜头类型判，无视商品差异）。返回 (matches, reason)。"""
    system = ("你是短视频复刻的画面审校专家。复刻的是**参考视频的叙事结构**，用户商品与参考"
              "**通常不是同一个**，这很正常。判断标准要**宽松**且**只看结构功能**，以 JSON 返回。")
    user = ("这个节拍的作用是：「{}」。上图是分配到该节拍的用户素材画面。\n"
            "判断原则：**复刻的是结构，不是商品**——画面里没有出现参考视频那个商品/成分/动作，"
            "**不算**跑题。但要判断这段画面**承担的叙事功能/镜头类型**是否 = 该节拍要求的功能：\n"
            "- 若一致（都是开场钩子/成品展示/包装卖点特写/使用演示/手机下单/催单等同一类）→ matches=true;\n"
            "- 若**功能明显不同**（如节拍要「手机下单/价格展示」而画面是「人物讲解/配料特写」，"
            "或节拍要「使用演示」而画面是无关空镜）→ matches=false（需要换更贴合该功能的素材）。\n"
            "只返回 JSON：{{\"matches\": true/false, \"reason\": \"...\"}}").format(beat_desc)
    try:
        obj = _run_async(as_core.complete_json(
            system, user, vision=True, media=[{"type": "image", "url": frame_png}]))
        return bool(obj.get("matches", True)), obj.get("reason", "")
    except Exception as exc:  # noqa: BLE001
        print("[weak_arbiter] VLM 校验失败(放过): {}".format(str(exc)[:120]), flush=True)
        return True, "vlm_error"


def _best_swap(beat, candidates, used, work_dir, sid, info_len, verify_visual, ref_frame=None):
    """从未用候选里按'贴近参考 beat'的画面相关性排序，取最贴的一个。

    - ref_frame 给定时：优先按**和参考帧的视觉相似度**校验换上的候选（画面贴近参考镜头）。
    - 否则 verify_visual=True 用 beat 文字校验画面；False（口播跑题）只按 beat 相关性取最高。
    """
    ranked = sorted(
        (c for c in candidates if c.get("global_asset_id") not in used and c.get("source_path")),
        key=lambda c: score_candidate_deterministic(beat, c), reverse=True)
    for alt in ranked[:MAX_SWAP_TRIES]:
        if not verify_visual and not ref_frame:
            return alt
        apng = os.path.join(work_dir, "arb_frames", "{}_alt_{}.png".format(sid, info_len))
        if not _extract_frame(alt.get("source_path"), alt.get("start", 0.0),
                              alt.get("duration", 0.0), apng):
            continue
        if ref_frame:
            ok, _ = visual_match.visual_match(ref_frame, apng, beat)
        else:
            ok, _ = _vlm_matches(apng, beat)
        if ok:
            return alt
    return None


def arbitrate(segments, candidates, decisions, work_dir, ref_frames=None):
    """画面仲裁（保原声·不删段策略）。

    - **原声段**：锁定保护——用户真实口播+人脸是复刻的最高优先，画面即本人出镜，
      不参与换素材、不删除（即便口播/画面与参考 beat 不完全对齐也保留原声）。
    - **非原声段**：有参考帧则 VLM 比「画面是否视觉贴近参考镜头」，否则比 beat 文字功能；
      不贴 → 从未用候选换更贴的素材（若换到自带口播的候选，下游 voice_policy 自然恢复原声）。
    - **任何段都不删**：换不到更贴题的素材就保留原分配（降级为克隆配音），保住段数与结构。

    检测（抽帧+VLM 判是否跑题）逐段互不影响，故并发跑；换素材阶段串行执行，
    因为它要维护「候选已被占用」的全局状态。返回 (segments, info)，段数恒不变。
    """
    decisions = decisions or {}
    frame_dir = os.path.join(work_dir, "arb_frames")
    os.makedirs(frame_dir, exist_ok=True)
    used = {(s.get("best_candidate") or {}).get("global_asset_id")
            for s in segments if s.get("best_candidate")}

    info = []
    to_check = []
    for idx, s in enumerate(segments):
        cand = s.get("best_candidate") or {}
        if s.get("is_t2v") or not cand.get("source_path"):
            continue
        if (decisions.get(s.get("slot_id")) or {}).get("voice_source") == "original":
            # 原声段锁定：不换不删（保原声优先于画面贴近）
            info.append({"slot_id": s.get("slot_id"), "action": "keep_original",
                         "reason": "原声段锁定保护(禁换禁删)"})
            continue
        to_check.append(idx)

    def _check(idx):
        s = segments[idx]
        sid = s.get("slot_id")
        beat = s.get("beat_desc", "")
        cand = s.get("best_candidate") or {}
        ref_frame = ref_frames[idx] if (ref_frames and idx < len(ref_frames)) else None
        png = os.path.join(frame_dir, "{}.png".format(sid))
        if not _extract_frame(cand.get("source_path"), cand.get("start", 0.0),
                              cand.get("duration", 0.0), png):
            return (idx, False, "")
        if ref_frame:
            m, r = visual_match.visual_match(ref_frame, png, beat)
            r = "画面不贴近参考镜头: " + r
        else:
            m, r = _vlm_matches(png, beat)
        return (idx, not m, r)

    checked = []
    for r in parallel_map(_check, to_check):
        if isinstance(r, Exception):
            print("[weak_arbiter] 画面校验失败(放过): {}".format(str(r)[:120]), flush=True)
            continue
        checked.append(r)

    # 换素材串行：要按「已占用」逐个排他分配，且换镜自身还要 VLM 校验候选
    for idx, problem, reason in checked:
        if not problem:
            continue
        s = segments[idx]
        sid = s.get("slot_id")
        beat = s.get("beat_desc", "")
        cand = s.get("best_candidate") or {}
        ref_frame = ref_frames[idx] if (ref_frames and idx < len(ref_frames)) else None
        swapped = _best_swap(beat, candidates, used, work_dir, sid, len(info), True,
                             ref_frame=ref_frame)
        if swapped:
            used.discard(cand.get("global_asset_id"))
            used.add(swapped.get("global_asset_id"))
            s["best_candidate"] = swapped
            s.pop("source_start_override", None)
            s.pop("source_take", None)
            s.pop("voice_align", None)
            info.append({"slot_id": sid, "action": "swap", "from": cand.get("global_asset_id"),
                         "to": swapped.get("global_asset_id"), "reason": reason})
            print("[weak_arbiter] {} {} -> 换素材 {} ".format(
                sid, reason[:40], swapped.get("global_asset_id")), flush=True)
        else:
            info.append({"slot_id": sid, "action": "keep_mismatch",
                         "reason": "无更贴题素材可换, 保留原分配(降级克隆): " + reason})
            print("[weak_arbiter] {} 画面欠贴近但无更好素材可换 -> 保留(降级克隆): {}".format(
                sid, reason[:50]), flush=True)
    return segments, info
