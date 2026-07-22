"""理解阶段（A 负责：理解 → 编排 的理解端）。

`UnderstandingAgent.analyze_stream`：看懂爆款参考视频 + 理解用户素材 + 可行性验证/仲裁，
产出 AnalysisResult（契约①：template + schemes + feasibility + material_understanding），
交给编排阶段 `orchestration.OrchestrationAgent`。

通用 AgentScope 运行时（LLM/工具胶水）在 `shared/agent_runtime.ReplicationAgentBase`。
"""
import asyncio
import json
import os

import bgm
import cache
import contracts
import obs
import projects
import react_agents  # noqa: F401  (保持与旧模块一致的可见性；agent_runtime 亦引用)
from agent_runtime import ReplicationAgentBase
from profile import INDUSTRY_OPTIONS
from serialize import to_jsonable
from skills import get as get_skill
from thumbs import extract_shot_thumbs, probe_aspect_ratio, probe_duration_seconds
from tools import Retriever

from .feasibility import arbitrate_conflicts, verify_materials
from .material_understanding import understand_materials
from .planner import TOOL_CATALOG, run_understanding_planner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_log = obs.get_logger("orchestrator")


class UnderstandingAgent(ReplicationAgentBase):
    """理解 Agent：拆解爆款 + 理解用户素材 + 可行性验证，产出 AnalysisResult（契约①）。"""

    async def analyze_stream(self, bundle):
        rid = obs.new_request_id()
        _log.info("[%s] analyze start uri=%s skill=%s materials=%d use_cache=%s enabled_tools=%s",
                  rid, bundle.video_uri or "(none)", bundle.skill_id or "-",
                  len(bundle.materials or []), bundle.use_cache, bundle.enabled_tools)
        # 记录/更新历史项目：下次可从左侧直接选此项目、自动载入素材，无需重新上传
        try:
            projects.save_project(
                video_uri=bundle.video_uri or "",
                video_desc=getattr(bundle, "video_desc", "") or "",
                intent=bundle.intent or "",
                skill_id=bundle.skill_id or "",
                materials=bundle.materials or [],
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] save project failed: %s", rid, exc)
        skill = get_skill(bundle.skill_id) if bundle.skill_id else None
        # 通用理解 skill 作为基础层，场景 skill 作为叠加层（两者可叠加）
        base_skill = get_skill("viral_reference_understanding")
        base_prompt = base_skill.prompt_hint if base_skill else None
        base_name = base_skill.name if base_skill else "通用理解"
        overlays = [(skill.name, skill.prompt_hint)] if skill else []
        duration_sec = probe_duration_seconds(bundle.video_uri)
        _log.info("[%s] probed duration=%.2fs", rid, duration_sec)

        # 缓存 key 只依赖会影响理解输出的输入字段
        cache_input = {
            "video_uri": bundle.video_uri,
            "video_desc": bundle.video_desc,
            "intent": bundle.intent,
            "skill_id": bundle.skill_id,
            "enabled_tools": sorted(bundle.enabled_tools) if bundle.enabled_tools else None,
            "materials": sorted(bundle.materials or []),
        }
        cache_key = cache.key_for("analyze", cache_input)
        cache_hit_payload = None
        if bundle.use_cache:
            cached = cache.get("analyze", cache_input)
            if cached and isinstance(cached.get("payload"), dict):
                _log.info("[%s] analyze cache HIT key=%s (仍走后续验证/审核)", rid, cache_key)
                cache_hit_payload = cached["payload"]
        if cache_hit_payload is None:
            _log.info("[%s] analyze cache MISS key=%s", rid, cache_key)

        local_video = None
        if bundle.video_uri and not bundle.video_uri.startswith(("http://", "https://", "data:")):
            for candidate in (bundle.video_uri, os.path.join(PROJECT_ROOT, bundle.video_uri)):
                if os.path.isfile(candidate):
                    local_video = candidate
                    break

        # 归纳素材元数据（数量、平均时长），一并交给编排 Agent 做工具选择。
        material_count = len(bundle.materials or [])
        material_durations = []
        for uri in bundle.materials or []:
            if isinstance(uri, str) and not uri.startswith(("http://", "https://", "data:")):
                for cand in (uri, os.path.join(PROJECT_ROOT, uri)):
                    if os.path.isfile(cand):
                        d = probe_duration_seconds(cand)
                        if d > 0:
                            material_durations.append(d)
                        break
        avg_material_duration = round(sum(material_durations) / len(material_durations), 2) if material_durations else 0.0

        metadata = {
            "reference_video_uri": bundle.video_uri,
            "reference_duration_sec": round(duration_sec, 2),
            "local_video": local_video,
            "video_desc": bundle.video_desc,
            "intent": bundle.intent,
            "skill_id": bundle.skill_id,
            "material_count": material_count,
            "material_avg_duration_sec": avg_material_duration,
            "materials": bundle.materials,
            "tool_catalog": list(TOOL_CATALOG.keys()),
        }
        yield {"type": "step", "phase": "调度", "key": f"meta-{rid}", "state": "done",
               "title": "接收任务元数据",
               "thought": "汇总参考视频时长、素材数量等元数据",
               "observation": json.dumps({k: v for k, v in metadata.items() if k != "materials"}, ensure_ascii=False)}

        findings: list = []
        planner_reason = ""
        data = {}
        plan_key = f"plan-{rid}"
        yield {"type": "step", "phase": "调度", "key": plan_key, "state": "running",
               "title": "规划要调用的工具",
               "cached": bool(cache_hit_payload),
               "thought": ("命中缓存，跳过工具规划" if cache_hit_payload else "Agent 正在决定调用哪些工具")}
        if cache_hit_payload:
            yield {"type": "step", "phase": "调度", "key": plan_key, "state": "done",
                   "title": "工具规划完成",
                   "cached": True,
                   "thought": f"复用理解缓存 {cache_key}，跳过 planner 与视觉理解"}
        else:
            async for event in run_understanding_planner(
                metadata=metadata, toolkit=self._understanding_toolkit,
                allowed_tools=bundle.enabled_tools,
            ):
                if event.get("__planner_result__"):
                    findings = event.get("findings", [])
                    planner_reason = event.get("reason", "")
                    continue
                yield event
            yield {"type": "step", "phase": "调度", "key": plan_key, "state": "done",
                   "title": "工具规划完成", "thought": planner_reason or "工具规划完成"}
            _log.info("[%s] planner done: %d tool findings=%s reason=%s", rid, len(findings),
                      [f["tool"] for f in findings], planner_reason[:80])

        understand_key = f"understand-{rid}"
        stacked = base_name + ("＋" + skill.name if skill else "")
        yield {"type": "step", "phase": "理解", "key": understand_key, "state": "running",
               "lane_skill": stacked,
               "cached": bool(cache_hit_payload),
               "title": "理解参考视频",
               "thought": ("命中缓存，复用上次视觉理解结果"
                            if cache_hit_payload else f"多模态模型正在观看并拆解参考视频（skill：{stacked}）")}

        if cache_hit_payload:
            template_dict = cache_hit_payload.get("template", {}) or {}
            industry = cache_hit_payload.get("industry_guess", "ecom")
            schemes = cache_hit_payload.get("schemes", []) or []
            industry_reason = cache_hit_payload.get("industry_reason", "")
            aspect = cache_hit_payload.get("aspect")
            template = self.understanding.build_template(template_dict, industry)
            data = {"industry_reason": industry_reason}
        else:
            shot_meta = next((f["metadata"] for f in findings if f["tool"] == "detect_shot_boundaries"), {})
            beat_meta = next((f["metadata"] for f in findings if f["tool"] == "detect_music_beats"), {})
            asr_finding = next((f for f in findings if f["tool"] == "transcribe_audio"), None)

            system, user = self.understanding.analyze_messages(
                bundle.video_uri, bundle.video_desc, bundle.intent, bundle.materials,
                duration_sec=duration_sec, base_prompt=base_prompt, overlays=overlays,
            )
            extra = {
                "planner_reason": planner_reason,
                "material_count": material_count,
                "material_avg_duration_sec": avg_material_duration,
            }
            if shot_meta:
                extra["shot_boundaries"] = shot_meta.get("boundaries", [])
            if beat_meta:
                extra["tempo_bpm"] = beat_meta.get("tempo_bpm")
                extra["beats_preview"] = (beat_meta.get("beats", []) or [])[:32]
            if asr_finding and asr_finding.get("summary"):
                try:
                    asr_data = json.loads(asr_finding["summary"])
                except (ValueError, TypeError):
                    asr_data = {}
                if isinstance(asr_data, dict) and (asr_data.get("text") or asr_data.get("segments")):
                    extra["asr_transcript"] = asr_data.get("text", "")
                    extra["asr_segments"] = [
                        {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
                        for s in (asr_data.get("segments") or [])[:80]
                        if isinstance(s, dict)
                    ]
                else:
                    extra["asr_transcript"] = asr_finding["summary"][:1500]
            user = user + "\n\n[任务元数据与工具观察]\n" + json.dumps(extra, ensure_ascii=False)
            media = []
            if bundle.video_uri:
                media.append({"type": "video", "url": bundle.video_uri})
            async for event in self._llm_json("理解", system, user, vision=True, media=media):
                if event["type"] == "__json__":
                    data = event["data"]
                else:
                    yield event
            industry = skill.industry if skill else data.get("industry_guess", "ecom")
            template = self.understanding.build_template(data, industry)
            schemes = data.get("schemes", []) if isinstance(data.get("schemes", []), list) else []
            aspect = probe_aspect_ratio(bundle.video_uri)
            template_dict = to_jsonable(template)
            extract_shot_thumbs(bundle.video_uri, template_dict.get("shot_slots", []))

        dimension_count = sum(len(s.get("dimensions", [])) for s in schemes if isinstance(s, dict))
        _log.info("[%s] understanding done (cache_hit=%s): industry=%s shots=%d schemes=%d dims=%d",
                  rid, bool(cache_hit_payload), industry, len(template.shot_slots), len(schemes), dimension_count)
        yield {"type": "step", "phase": "理解", "key": understand_key, "state": "done",
               "lane_skill": stacked,
               "cached": bool(cache_hit_payload),
               "title": "参考视频理解完成",
               "thought": f"识别行业 {industry}，拆出 {len(template.shot_slots)} 个分镜、{len(schemes)} 个方案",
               "observation": f"行业 {industry}；{len(template.shot_slots)} 个分镜；{len(schemes)} 个方案；{dimension_count} 个维度"}

        # 先展示拆片分析
        result = {
            "template": template_dict, "industry_guess": industry,
            "industry_reason": data.get("industry_reason", ""),
            "industry_options": INDUSTRY_OPTIONS, "schemes": schemes,
            "skill_id": bundle.skill_id, "aspect": aspect,
            "material_understanding": {},
            "cache_key": cache_key,
        }
        _ap = contracts.validate_analysis_result(result)
        if _ap:
            _log.warning("[%s] [contract] AnalysisResult 不符契约(v%s)：%s",
                         rid, contracts.CONTRACT_VERSION, contracts.summarize(_ap))
        yield {"type": "analysis", "result": result}

        # 真实 BGM 理解：抽参考视频音频交给 Gemini（oneapi-comate）分析配乐；结果异步补到 BGM 卡片。
        if bundle.video_uri:
            bgm_key = f"bgm-{rid}"
            yield {"type": "step", "phase": "拆片分析", "key": bgm_key, "state": "running",
                   "title": "BGM 理解", "thought": "抽取参考视频音频，交给 Gemini 分析配乐（曲风/情绪/BPM/卡点）"}
            try:
                bgm_info = await asyncio.to_thread(bgm.analyze_bgm, bundle.video_uri)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[%s] bgm analyze failed: %s", rid, exc)
                bgm_info = {"available": False, "reason": str(exc)}
            result["bgm"] = bgm_info
            if bgm_info.get("available"):
                summary = bgm_info.get("summary", "") or ("检测到背景音乐" if bgm_info.get("has_bgm") else "未检测到背景音乐")
                yield {"type": "step", "phase": "拆片分析", "key": bgm_key, "state": "done",
                       "title": "BGM 理解完成", "thought": summary,
                       "observation": json.dumps(bgm_info, ensure_ascii=False)}
            else:
                yield {"type": "step", "phase": "拆片分析", "key": bgm_key, "state": "done",
                       "title": "BGM 理解跳过", "thought": bgm_info.get("reason", "无法分析音频")}
            yield {"type": "bgm", "result": bgm_info}

        # 并行理解用户素材（默认 8 路），在拆片分析展示之后后台继续
        material_results = {}
        if bundle.materials:
            if bundle.refresh_vectors:
                try:
                    Retriever(rid).clear()
                    yield {"type": "step", "phase": "素材理解", "key": f"vclear-{rid}", "state": "done",
                           "title": "刷新向量库", "thought": "已清空本任务向量库，本次素材理解将全量重新入库"}
                    _log.info("[%s] vector store cleared (refresh_vectors)", rid)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("[%s] vector clear failed: %s", rid, exc)
            async for event in understand_materials(bundle.materials, use_cache=bundle.use_cache, task_id=rid):
                if event.get("__materials_result__"):
                    material_results = event.get("results", {})
                    continue
                yield event
            _log.info("[%s] material understanding done: %d results", rid, len(material_results))
            result["material_understanding"] = material_results
            yield {"type": "materials", "result": material_results}

        # 素材可行性验证：按 shot 数启动多子 Agent（≤10），在用户素材中查找可复刻片段
        feasibility = {}
        shot_dicts = template_dict.get("shot_slots", [])
        if bundle.materials and shot_dicts:
            async for event in verify_materials(shot_dicts, task_id=rid):
                if event.get("__feasibility_result__"):
                    feasibility = event.get("results", {})
                    continue
                yield event
            _log.info("[%s] feasibility done: %d shots", rid, len(feasibility))
            result["feasibility"] = feasibility
            yield {"type": "feasibility", "result": feasibility}

            # 审核 Agent：检测多个镜头争抢同一素材/时间重叠，裁决片段归属
            async for event in arbitrate_conflicts(shot_dicts, feasibility):
                if event.get("__arbitration_result__"):
                    if event.get("conflicts"):
                        feasibility = event.get("results", feasibility)
                        result["feasibility"] = feasibility
                        yield {"type": "feasibility", "result": feasibility}
                        _log.info("[%s] arbitration resolved %d conflict group(s)", rid, event.get("conflicts"))
                    continue
                yield event

        try:
            cache.set("analyze", cache_input, result)
            _log.info("[%s] analyze cache stored key=%s", rid, cache_key)
        except OSError as exc:
            _log.warning("[%s] cache write failed: %s", rid, exc)
