"""并行用户素材理解：默认 8 路并发，每路一个独立的素材理解 Agent。

每个用户素材（视频/图片）由一个独立 lane 的 Agent 处理，加载隐藏 skill
``user_material_understanding`` 作为 system prompt，调用视觉模型产出可检索的
asset 片段结构。所有 lane 的事件通过 ``asyncio.Queue`` 汇流到单条 SSE 流，事件
携带 ``lane`` / ``lane_label`` / ``lane_skill`` 字段，前端据此竖排分组、每路横向
排自己的步骤方块。每个素材的理解结果单独缓存。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import AsyncGenerator

import as_core
import asr_cache
import cache
import obs
from skills import get as get_skill
from tools import ASRTool, Retriever

_log = obs.get_logger("material")
_ASR = ASRTool()
_ASR_SEM = asyncio.Semaphore(int(os.getenv("ASR_CONCURRENCY", "1")))

CONCURRENCY = int(os.getenv("MATERIAL_CONCURRENCY", "8"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

SKILL_ID = "user_material_understanding"


def _classify(uri: str) -> str:
    if not isinstance(uri, str):
        return "text"
    if uri.startswith(("http://", "https://")):
        return "video"  # 远端一律按视频/图片尝试
    ext = os.path.splitext(uri)[1].lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    # 本地文件但非音视频图片，或纯文本内容
    for cand in (uri, os.path.join(PROJECT_ROOT, uri)):
        if os.path.isfile(cand):
            return "video" if ext in _VIDEO_EXTS else "image" if ext in _IMAGE_EXTS else "text"
    return "text"


def _label(uri: str) -> str:
    if isinstance(uri, str) and ("/" in uri or "\\" in uri):
        return os.path.basename(uri)
    return (str(uri)[:24] + "…") if len(str(uri)) > 24 else str(uri)


def _ensure_global_ids(parsed: dict, source: str, source_path: str = "") -> dict:
    """兑现 SKILL.md 的承诺：为每个片段补 source_video_id / source_path / global_asset_id。"""
    if not isinstance(parsed, dict):
        return parsed
    svid = parsed.get("source_video_id") or source
    src_path = source_path or parsed.get("source_path") or ""
    for seg in parsed.get("asset_segments", []) or []:
        if not isinstance(seg, dict):
            continue
        seg.setdefault("source_video_id", svid)
        if src_path and not seg.get("source_path"):
            seg["source_path"] = src_path
        if not seg.get("global_asset_id"):
            seg["global_asset_id"] = f"{seg.get('source_video_id') or source}::{seg.get('asset_id', '')}"
    return parsed


def _index_segments(parsed: dict, source: str, retriever: Retriever) -> dict:
    """把理解产出的片段按字段写入 per-task 向量库，返回入库统计。"""
    segs = parsed.get("asset_segments", []) or []
    if not segs:
        return {"indexed": 0, "skipped": 0}
    try:
        return retriever.index_segments(segs, source=source)
    except Exception as exc:  # noqa: BLE001
        _log.warning("index segments failed for %s: %s", source, exc)
        return {"indexed": 0, "skipped": 0, "error": str(exc)}


async def _understand_one(lane: str, lane_label: str, key: str, uri: str,
                          skill_prompt: str, skill_name: str,
                          use_cache: bool, queue: "asyncio.Queue", results: dict,
                          retriever: Retriever):
    label = _label(uri)
    base = {"lane": lane, "lane_label": lane_label, "lane_skill": skill_name, "phase": "素材理解"}
    kind = _classify(uri)

    def step(state, title, thought, observation=None, cached=False):
        ev = {"type": "step", "state": state, "title": title, "thought": thought, "key": key, "cached": cached}
        ev.update(base)
        if observation is not None:
            ev["observation"] = observation
        return ev

    cache_input = {"uri": uri, "skill": SKILL_ID}
    if use_cache:
        cached = cache.get("material", cache_input)
        if cached and isinstance(cached.get("payload"), dict):
            # 视觉理解命中缓存 → 立即复用，不在这里跑 ASR（ASR 是重活，放在这里会让"理解"
            # 每次都变慢、看起来像没命中缓存）。ASR 缓存由「首次理解」和「剪辑连接器」按需落盘。
            await queue.put(step("done", label, "命中缓存，复用上次结果",
                                 observation=json.dumps(cached["payload"].get("segment_summary", {}), ensure_ascii=False),
                                 cached=True))
            results[key] = cached["payload"]
            _ensure_global_ids(cached["payload"], label, uri)
            _index_segments(cached["payload"], label, retriever)
            return

    if kind == "text":
        parsed = {"source_video_id": label, "asset_type": "text", "text": uri,
                  "segment_summary": {"speech_summary": str(uri)[:200]}, "asset_segments": []}
        await queue.put(step("done", f"{label}（文本）", "文本素材无需视觉理解", observation=str(uri)[:200]))
        results[key] = parsed
        cache.set("material", cache_input, parsed)
        return

    await queue.put(step("running", f"理解 {label}", f"加载 skill「{skill_name}」，视觉拆解该素材"))
    # ASR 前置：对视频类素材先跑一次本机离线语音识别，把 transcript + segments 注入视觉理解 prompt
    # 命中共享 asr 缓存则跳过——同一素材在剪辑连接器阶段也会用同一份，避免重复转写。
    asr_result = None
    if kind == "video":
        asr_key = f"{key}-asr"
        cached_asr = asr_cache.get(uri)
        if cached_asr:
            asr_data = {"text": cached_asr.get("text", ""), "segments": cached_asr.get("segments", []),
                        "duration_seconds": cached_asr.get("duration_seconds", 0.0)}
            await queue.put({"type": "step", "state": "done", "title": f"{label} 语音识别（缓存）",
                              "thought": f"命中 ASR 缓存，{len(asr_data['text'])} 字、{len(asr_data['segments'])} 句",
                              "observation": asr_data["text"][:200], "cached": True,
                              "key": asr_key, **base})
        else:
            await queue.put({"type": "step", "state": "running", "title": f"{label} 语音识别",
                              "thought": "本机离线 Qwen3-ASR 提取该素材的口播/台词",
                              "key": asr_key, **base})
            try:
                async with _ASR_SEM:
                    asr_data = await asyncio.to_thread(_ASR.transcribe, uri)
            except Exception as exc:  # noqa: BLE001
                _log.warning("material ASR lane=%s uri=%s failed: %s", lane, uri, exc)
                asr_data = {"error": repr(exc)}
            if isinstance(asr_data, dict) and not asr_data.get("error"):
                asr_cache.set(uri, asr_data)  # 成功即缓存（含"无口播"空结果），下次不再重转
        if isinstance(asr_data, dict) and (asr_data.get("text") or asr_data.get("segments")):
            asr_result = {
                "text": asr_data.get("text", ""),
                "segments": [
                    {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
                    for s in (asr_data.get("segments") or [])[:60]
                    if isinstance(s, dict)
                ],
            }
            await queue.put({"type": "step", "state": "done", "title": f"{label} 语音识别完成",
                              "thought": f"{len(asr_result['text'])} 字，{len(asr_result['segments'])} 句",
                              "observation": asr_result["text"][:200],
                              "key": asr_key, **base})
        else:
            await queue.put({"type": "step", "state": "done", "title": f"{label} 语音识别跳过",
                              "thought": (asr_data or {}).get("error", "无有效口播或权限受限"),
                              "key": asr_key, **base})
    user_payload = {"source_video_id": label, "source_path": uri}
    if asr_result:
        user_payload["asr_transcript"] = asr_result["text"]
        user_payload["asr_segments"] = asr_result["segments"]
    user = json.dumps(user_payload, ensure_ascii=False)
    media = [{"type": kind, "url": uri}]
    content = ""
    try:
        async for item in as_core.stream(skill_prompt, user, vision=True, media=media):
            if item.get("reasoning"):
                ev = {"type": "reasoning", "text": item["reasoning"]}
                ev.update(base)
                await queue.put(ev)
            elif "content" in item:
                content = item["content"]
        parsed = as_core.parse_json(content) if content.strip() else {}
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        _log.warning("material lane=%s uri=%s failed: %s", lane, uri, exc)
        parsed = {"source_video_id": label, "understanding_error": str(exc), "asset_segments": []}
    # 为每个片段补 source 信息与 global_asset_id，再入 per-task 向量库（按字段建索引）
    _ensure_global_ids(parsed, label, uri)
    indexed = _index_segments(parsed, label, retriever)
    seg_count = len(parsed.get("asset_segments", []) or [])
    added, skipped = indexed.get("indexed", 0), indexed.get("skipped", 0)
    index_note = f"入库 {added} 条" + (f"（{skipped} 条已在库）" if skipped else "")
    if indexed.get("error"):
        index_note = f"入库失败：{indexed['error'][:60]}"
    await queue.put(step("done", f"{label} 完成",
                         f"产出 {seg_count} 个片段，{index_note}",
                         observation=json.dumps(parsed.get("segment_summary", {}), ensure_ascii=False)))
    results[key] = parsed
    try:
        cache.set("material", cache_input, parsed)
    except OSError:
        pass


async def understand_materials(materials, *, use_cache=True, concurrency=None, task_id="") -> AsyncGenerator[dict, None]:
    """用固定大小的并行 Agent 池理解所有用户素材，流式产出带 lane 的事件。

    每个 worker Agent 占一个 lane（``agent-{w}``），从共享任务队列串行领取素材；
    素材多于并发度时，同一个 Agent 会依次处理多个素材，其步骤方块都排在自己那
    一行。结束时 yield ``{"__materials_result__": True, "results": {...}}``。

    ``task_id`` 用于 per-task 向量库隔离（同一素材的 VLM 理解结果仍走全局 cache 复用）。
    """
    materials = [m for m in (materials or []) if m]
    if not materials:
        yield {"__materials_result__": True, "results": {}}
        return
    skill = get_skill(SKILL_ID)
    skill_prompt = skill.prompt_hint if skill else "客观拆解用户素材，输出可检索片段的 JSON。"
    skill_name = skill.name if skill else SKILL_ID
    limit = concurrency or CONCURRENCY
    num_workers = max(1, min(limit, len(materials)))
    _log.info("material understanding start count=%d workers=%d (limit=%d)", len(materials), num_workers, limit)

    # 主流程行里的父方块：用户素材理解（先置为进行中，等所有子 Agent 完成再收尾）
    yield {"type": "step", "phase": "素材理解", "key": "materials-head", "state": "running",
           "lane_skill": skill_name,
           "title": "用户素材理解",
           "thought": f"启动 {num_workers} 个并行 Agent 理解 {len(materials)} 个素材，等待子进程完成…"}

    out_q: "asyncio.Queue" = asyncio.Queue()
    results: dict = {}
    # 静态 round-robin 分配：worker w 处理 materials[w], materials[w+num_workers], ...
    # 保证每个 Agent lane 都有稳定的素材，不受调度抢占影响。
    assignments = {w: [(i, materials[i]) for i in range(w, len(materials), num_workers)]
                   for w in range(num_workers)}

    retriever = Retriever(task_id)

    async def worker(w):
        lane = f"agent-{w}"
        lane_label = f"Agent #{w + 1}"
        for index, uri in assignments.get(w, []):
            await _understand_one(lane, lane_label, f"{lane}-{index}", uri,
                                  skill_prompt, skill_name, use_cache, out_q, results, retriever)
        await out_q.put({"__worker_done__": w})

    tasks = [asyncio.create_task(worker(w)) for w in range(num_workers)]
    done = 0
    while done < num_workers:
        ev = await out_q.get()
        if ev.get("__worker_done__") is not None:
            done += 1
            continue
        yield ev
    await asyncio.gather(*tasks, return_exceptions=True)
    _log.info("material understanding done: %d results", len(results))
    # 所有子 Agent 完成后，父方块收尾
    seg_total = sum(len(v.get("asset_segments", []) or []) for v in results.values())
    # 落一份「本次任务清单」：一次上传+理解 = 一次任务，debug 页据此按任务展示各素材理解 JSON
    _save_task_manifest(task_id, results, seg_total)
    yield {"type": "step", "phase": "素材理解", "key": "materials-head", "state": "done",
           "lane_skill": skill_name,
           "title": "用户素材理解完成",
           "thought": f"{num_workers} 个 Agent 完成 {len(materials)} 个素材，累计 {seg_total} 个可检索片段"}
    yield {"__materials_result__": True, "results": results}


def _save_task_manifest(task_id: str, results: dict, seg_total: int) -> None:
    """把本次任务理解出的每个素材（可播放视频 + 完整理解 JSON）写成任务清单。

    一次「上传素材 → 点击理解」即一次任务；debug 页默认按最近一次任务展示。
    """
    if not task_id:
        return
    materials = []
    for key, payload in (results or {}).items():
        if not isinstance(payload, dict):
            continue
        materials.append({
            "cache_key": key,
            "source_video_id": payload.get("source_video_id", ""),
            "source_path": payload.get("source_path", ""),
            "segment_count": len(payload.get("asset_segments", []) or []),
            "payload": payload,
        })
    manifest = {
        "task_id": task_id,
        "created_at": time.time(),
        "created_at_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "material_count": len(materials),
        "segment_count": seg_total,
        "materials": materials,
    }
    try:
        manifest_dir = os.path.join(PROJECT_ROOT, "uploads", "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        with open(os.path.join(manifest_dir, f"{task_id}.json"), "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        _log.info("[%s] task manifest saved: %d materials, %d segments", task_id, len(materials), seg_total)
    except OSError as exc:
        _log.warning("[%s] task manifest save failed: %s", task_id, exc)

