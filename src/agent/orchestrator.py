"""Two-stage viral-video replication orchestration on top of AgentScope 2.0.4."""
import asyncio
import json
import os
import uuid

import as_core
import bgm
import cache
import obs
import projects
from agentscope.message import ToolCallBlock
from agentscope.state import AgentState
from profile import INDUSTRY_OPTIONS, load_profile
from schema import ReplicationPlan, SlotAssignment
from serialize import to_jsonable
from skills import get as get_skill
from thumbs import extract_remake_thumbs, extract_shot_thumbs, probe_aspect_ratio, probe_duration_seconds
from scriptgen import export_scripts
from tools import Retriever

from . import react_agents
from .feasibility import arbitrate_conflicts, verify_materials
from .material_understanding import understand_materials
from .orchestration import orchestrate_from_feasibility, orchestrate_structure_first
from .planner import TOOL_CATALOG, run_understanding_planner
from .prompts import DECIDE_SYSTEM, PLAN_SYSTEM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_log = obs.get_logger("orchestrator")


class ReplicationAgent:
    """Coordinates UnderstandingAgent + PlanningAgent stages via AgentScope.

    业务逻辑集中在 `src/tools/`，这里通过 `react_agents` 提供的共享实例引用。
    """

    def __init__(self):
        # 直接引用 react_agents 中的共享业务实例，保证 orchestrator 与
        # FunctionTool 薄封装走同一份业务实现。
        self.understanding = react_agents.UNDERSTANDING
        self.planning = react_agents.PLANNING
        self.generation = react_agents.GENERATION
        self.editing = react_agents.EDITING
        self.packaging = react_agents.PACKAGING
        # AgentScope Agents (and their toolkits) are what actually drive the
        # LLM interactions. Building them lazily keeps unit tests light-weight.
        self._understanding_agent = None
        self._planning_agent = None
        self._understanding_toolkit = react_agents.build_understanding_toolkit()
        self._planning_toolkit = react_agents.build_planning_toolkit()

    @property
    def understanding_agent(self):
        if self._understanding_agent is None:
            self._understanding_agent = react_agents.build_understanding_agent()
        return self._understanding_agent

    @property
    def planning_agent(self):
        if self._planning_agent is None:
            self._planning_agent = react_agents.build_planning_agent()
        return self._planning_agent

    async def _llm_json(self, phase, system, user, *, vision=False, media=None):
        content = ""
        async for item in as_core.stream(system, user, vision=vision, media=media):
            if item.get("reasoning"):
                yield {"type": "reasoning", "phase": phase, "text": item["reasoning"]}
            elif "content" in item:
                content = item["content"]
        try:
            data = as_core.parse_json(content) if content.strip() else {}
        except (ValueError, TypeError, json.JSONDecodeError):
            data = {}
        yield {"type": "__json__", "data": data}

    async def _run_tool(self, toolkit, name, phase, *, title=None, **payload):
        """Execute a Toolkit-registered FunctionTool and emit paired SSE step events.

        发出一对带相同 ``key`` 的 step（running → done），前端据此渲染成
        一个"进行中→已完成"的步骤方块。
        """
        call_id = uuid.uuid4().hex[:12]
        label = title or name
        block = ToolCallBlock(id=call_id, name=name, input=json.dumps(payload, ensure_ascii=False))
        yield {"type": "step", "phase": phase, "key": call_id, "state": "running",
               "title": label, "thought": f"正在{label}"}
        state = AgentState()
        observation = ""
        async for chunk in toolkit.call_tool(block, state):
            content = getattr(chunk, "content", None) or []
            for entry in content:
                text = entry.get("text") if isinstance(entry, dict) else getattr(entry, "text", "")
                if text:
                    observation = text
        yield {"type": "step", "phase": phase, "key": call_id, "state": "done",
               "title": label, "thought": f"{label}完成", "observation": observation[:400]}

    async def _last_metadata(self, toolkit, name, payload):
        """再跑一次工具只为拿到 metadata（AgentScope call_tool 的 ToolResponse.metadata）。"""
        call_id = uuid.uuid4().hex[:12]
        block = ToolCallBlock(id=call_id, name=name, input=json.dumps(payload, ensure_ascii=False))
        state = AgentState()
        metadata = {}
        async for chunk in toolkit.call_tool(block, state):
            meta = getattr(chunk, "metadata", None)
            if meta:
                metadata = meta
        return metadata or {}

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

        findings: list[dict] = []
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

    async def replicate_stream(self, bundle):
        rid = obs.new_request_id()
        _log.info("[%s] replicate start industry=%s strategy=%s dims=%s trends=%d materials=%d",
                  rid, bundle.industry_id, bundle.material_strategy,
                  bundle.selected_dimensions, len(bundle.selected_trends or []), len(bundle.materials or []))
        profile = load_profile(bundle.industry_id)
        template = self.understanding.build_template(bundle.template, bundle.industry_id)
        selected = bundle.selected_dimensions

        async for event in self._run_tool(
            self._planning_toolkit, "plan_execute", "规划", title="规划复刻步骤",
            scheme_name=bundle.scheme_name, strategy=bundle.material_strategy,
            dimensions=selected, trends=bundle.selected_trends,
            materials_count=len(bundle.materials),
        ):
            yield event

        plan_user = json.dumps({
            "industry": bundle.industry_id, "intent": bundle.intent,
            "selected_dimensions": selected, "selected_trends": bundle.selected_trends,
            "materials_count": len(bundle.materials), "material_strategy": bundle.material_strategy,
            "scheme_name": bundle.scheme_name,
        }, ensure_ascii=False)
        plan_spec = {}
        plan_key = f"planllm-{rid}"
        yield {"type": "step", "phase": "规划", "key": plan_key, "state": "running",
               "title": "生成复刻计划", "thought": "规划 Agent 正在生成 plan-execute 计划"}
        async for event in self._llm_json("规划", PLAN_SYSTEM, plan_user):
            if event["type"] == "__json__":
                plan_spec = event["data"]
            else:
                yield event
        if not plan_spec:
            plan_spec = {"goal": "保留爆款结构并替换用户内容", "granularity": "action_scene", "reasoning": "采用稳定的逐镜规划。", "steps": []}
        yield {"type": "step", "phase": "规划", "key": plan_key, "state": "done",
               "title": "复刻计划完成", "thought": plan_spec.get("goal", "复刻计划完成"),
               "observation": plan_spec.get("reasoning", "")}

        materials = self.understanding.profile_materials(bundle.materials)
        yield {"type": "step", "phase": "解析", "key": f"mat-{rid}", "state": "done",
               "title": "盘点用户素材", "thought": "盘点用户素材能力",
               "observation": f"收到 {len(bundle.materials)} 项素材"}

        slots_brief = [{"slot_id": shot.id, "want": shot.want} for shot in template.shot_slots]
        decide_user = json.dumps({
            "strategy": bundle.material_strategy, "selected_dimensions": selected,
            "material_clip_count": len(materials.clips), "shot_slots": slots_brief,
        }, ensure_ascii=False)
        decision_data = {}
        decide_key = f"decide-{rid}"
        yield {"type": "step", "phase": "决策", "key": decide_key, "state": "running",
               "title": "逐镜决策", "thought": "决定每个镜头用素材还是生成"}
        async for event in self._llm_json("决策", DECIDE_SYSTEM, decide_user):
            if event["type"] == "__json__":
                decision_data = event["data"]
            else:
                yield event
        decisions = decision_data.get("decisions", []) if isinstance(decision_data, dict) else []
        # 两种复刻模式，归一成同一份 per-shot 决策(effective_feasibility)驱动脚本与逐镜标签
        shot_feasibility = (bundle.template or {}).get("feasibility") if isinstance(bundle.template, dict) else {}
        material_understanding = (bundle.template or {}).get("material_understanding") if isinstance(bundle.template, dict) else {}
        mode = (bundle.reproduce_mode or "structure").lower()
        # 自愈：前端可能因时序（素材理解事件未到）或选历史工程未回填而传空 material_understanding，
        # 结构优先编排一旦拿到空素材池就会退化成「按参考镜头全量补拍」。这里用素材缓存重建素材池，
        # 让复刻链路对前端时序不敏感。understand_materials 命中每素材缓存，通常很快。
        if not material_understanding and bundle.materials:
            rebuild_key = f"mu-rebuild-{rid}"
            yield {"type": "step", "phase": "解析", "key": rebuild_key, "state": "running",
                   "title": "重建素材理解", "thought": f"复刻请求未携带素材理解结果，从缓存重建 {len(bundle.materials)} 项素材的画像"}
            rebuilt = {}
            try:
                async for event in understand_materials(bundle.materials, use_cache=True, task_id=rid):
                    if event.get("__materials_result__"):
                        rebuilt = event.get("results", {})
                        continue
                    # 重建阶段不重复推送逐素材 step，保持 trace 简洁
                material_understanding = rebuilt or {}
            except Exception as exc:  # noqa: BLE001
                _log.warning("[%s] material_understanding rebuild failed: %s", rid, exc)
            _log.info("[%s] material_understanding rebuilt: %d results", rid, len(material_understanding))
            yield {"type": "step", "phase": "解析", "key": rebuild_key, "state": "done",
                   "title": "重建素材理解完成",
                   "thought": (f"重建出 {len(material_understanding)} 项素材画像"
                               if material_understanding else "缓存中无素材理解结果，素材池为空")}
        effective_feasibility = {}
        structure_template = None
        reference_dict = to_jsonable(template)
        if mode == "structure":
            # 结构优先：只参考爆款的结构 DNA（叙事阶段/节奏/Hook/CTA），用用户素材池自由编排，
            # **镜头数由 Agent 按 DNA 结构 + 素材实际情况决定，不与参考镜头数对齐**。
            orchestration_stream = orchestrate_structure_first(
                reference_dict, material_understanding,
                scheme_name=bundle.scheme_name, material_strategy=bundle.material_strategy,
                selected_dimensions=selected, selected_trends=bundle.selected_trends,
                intent=bundle.intent,
            )
        elif shot_feasibility:
            # 镜头优先：逐镜 1:1 复刻参考镜头——把可行性验证为每个参考镜头找到的候选交给编排 Agent
            # 改写字幕/选首选，并把多候选继承给 Split（保持参考镜头结构）。
            orchestration_stream = orchestrate_from_feasibility(
                reference_dict, shot_feasibility,
                scheme_name=bundle.scheme_name, material_strategy=bundle.material_strategy,
                selected_dimensions=selected, selected_trends=bundle.selected_trends,
                intent=bundle.intent,
            )
        else:
            orchestration_stream = None
            effective_feasibility = shot_feasibility or {}
        if orchestration_stream is not None:
            async for event in orchestration_stream:
                if event.get("__orchestration_result__"):
                    effective_feasibility = event.get("decisions", {})
                    structure_template = event.get("template")
                    continue
                yield event
        # 结构优先编排若产出空（素材池为空或编排 Agent 无有效输出），此前会静默退回参考模板的
        # 全量补拍，导致成片链路/Agent 剪辑「没有可剪辑的镜头」。这里显式告警，避免问题被吞掉。
        if mode == "structure" and not (structure_template and structure_template.get("shot_slots")):
            pool_empty = not material_understanding
            yield {"type": "step", "phase": "编排", "key": f"orch-empty-{rid}", "state": "done",
                   "title": "结构优先编排未产出可用镜头",
                   "thought": ("素材池为空（material_understanding 缺失），无法编排出用素材的镜头；"
                               "请重新「理解爆款」以生成素材理解，或检查素材上传。"
                               if pool_empty else
                               "编排 Agent 未返回有效镜头，将退回参考结构做全量补拍。")}
            _log.warning("[%s] structure-first produced no usable slots (pool_empty=%s)", rid, pool_empty)
        # 复刻分镜缩略图：在 structure_template（dict）上抽帧，再 build_template 带进 ShotSlot。
        # 用线程池并行 ffmpeg，避免同步子进程阻塞事件循环导致后续步骤迟迟不推进。
        if structure_template and effective_feasibility and structure_template.get("shot_slots"):
            shot_dicts = structure_template["shot_slots"]
            thumbs_key = f"remake-thumbs-{rid}"
            yield {"type": "step", "phase": "编排", "key": thumbs_key, "state": "running",
                   "title": "抽复刻分镜缩略图",
                   "thought": f"为 {len(shot_dicts)} 个镜头分别从「选中的用户片段」并行抽一帧"}
            try:
                await asyncio.to_thread(extract_remake_thumbs, shot_dicts, effective_feasibility)
                got = sum(1 for s in shot_dicts if s.get("remake_thumb"))
                yield {"type": "step", "phase": "编排", "key": thumbs_key, "state": "done",
                       "title": "抽复刻分镜缩略图完成",
                       "thought": f"共 {got}/{len(shot_dicts)} 个镜头拿到缩略图"}
            except Exception as exc:  # noqa: BLE001
                _log.warning("[%s] extract_remake_thumbs failed: %s", rid, exc)
                yield {"type": "step", "phase": "编排", "key": thumbs_key, "state": "done",
                       "title": "抽复刻分镜缩略图跳过", "thought": f"抽帧失败：{exc!s}"}
        if structure_template and structure_template.get("shot_slots"):
            # 用编排 Agent 输出/更新后的 template 替换（含缩略图/收尾信息），供后续 _build_plan / scriptgen 使用
            template = self.understanding.build_template(structure_template, bundle.industry_id)
        if effective_feasibility:
            feas_decisions = []
            for shot in template.shot_slots:
                fs = effective_feasibility.get(str(shot.id)) or effective_feasibility.get(shot.id) or {}
                status = fs.get("status")
                if status in ("direct", "partial"):
                    feas_decisions.append({"slot_id": shot.id, "action": "match",
                                            "reason": fs.get("reason", "可复刻")})
                elif status == "none":
                    feas_decisions.append({"slot_id": shot.id, "action": "generate",
                                            "reason": fs.get("reason", "需生成")})
            if feas_decisions:
                decisions = feas_decisions
        plan = self._build_plan(template, materials, decisions, bundle.material_strategy)
        counts = {"match": 0, "generate": 0}
        for assignment in plan.slot_assignments:
            counts[assignment.action] += 1
        yield {"type": "step", "phase": "决策", "key": decide_key, "state": "done",
               "title": "逐镜决策完成", "thought": f"用素材 {counts['match']} 镜，生成 {counts['generate']} 镜"}

        shots = []
        for assignment in plan.slot_assignments:
            shot = next((item for item in template.shot_slots if item.id == assignment.slot_id), None)
            if assignment.action == "generate":
                generated = self.generation.run(gen_prompt=assignment.gen_prompt or "", duration=shot.duration if shot else 0.0, profile=profile)
                shots.append({"slot_id": assignment.slot_id, "action": "generate", **generated})
            else:
                shots.append({"slot_id": assignment.slot_id, "action": "match", "uri": assignment.material_clip_id or "mock://material/unassigned"})
        if counts["generate"]:
            yield {"type": "step", "phase": "生成", "key": f"gen-{rid}", "state": "done",
                   "title": "生成缺失镜头", "thought": f"生成 {counts['generate']} 个占位片段"}
        edited = self.editing.run(shots=shots, profile=profile)
        yield {"type": "step", "phase": "剪辑", "key": f"edit-{rid}", "state": "done",
               "title": "组装时间线", "thought": "按爆款节奏组装时间线", "observation": "MOCK 时间线完成"}
        video = self.packaging.run(shots=edited.get("shots", shots), duration_sec=template.total_duration_sec, profile=profile)
        yield {"type": "step", "phase": "包装", "key": f"pack-{rid}", "state": "done",
               "title": "包装成片", "thought": "添加字幕、TTS、BGM 并导出", "observation": video.uri}
        report = {"plan_id": plan.plan_id, "structure_score": round(0.6 + 0.4 * plan.confidence, 2), "badcases": ["MOCK: 生成、剪辑与包装能力待替换为真实服务"]}
        yield {"type": "step", "phase": "评估", "key": f"eval-{rid}", "state": "done",
               "title": "复刻评估", "thought": "检查结构覆盖和素材利用率", "observation": f"结构分 {report['structure_score']}"}

        # 生成 Split 兼容的编排脚本（asset_guided_edit_plan + selected_editing_strategy）
        script_export = {}
        try:
            template_dict = bundle.template if isinstance(bundle.template, dict) and bundle.template.get("shot_slots") else to_jsonable(template)
            script_export = export_scripts(
                project_name=bundle.scheme_name or bundle.skill_id or "reproject",
                template=template_dict,
                schemes=(bundle.template or {}).get("schemes", []) if isinstance(bundle.template, dict) else [],
                scheme_name=bundle.scheme_name,
                selected_dimensions=bundle.selected_dimensions,
                selected_trends=bundle.selected_trends,
                material_strategy=bundle.material_strategy,
                feasibility=effective_feasibility or ((bundle.template or {}).get("feasibility") if isinstance(bundle.template, dict) else {}),
                material_understanding=(bundle.template or {}).get("material_understanding") if isinstance(bundle.template, dict) else {},
                target_product_name=bundle.intent,
            )
            yield {"type": "step", "phase": "包装", "key": f"script-{rid}", "state": "done",
                   "title": "生成 Split 兼容脚本",
                   "thought": "已写入 asset_guided_edit_plan 与 selected_editing_strategy",
                   "observation": f"strategy: {script_export['strategy_path']}\nplan: {script_export['edit_plan_path']}"}
            _log.info("[%s] scripts exported strategy=%s plan=%s", rid,
                      script_export["strategy_path"], script_export["edit_plan_path"])
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] script export failed: %s", rid, exc)

        _log.info("[%s] replicate done plan=%s confidence=%.2f match=%d generate=%d video=%s",
                  rid, plan.plan_id, plan.confidence, counts["match"], counts["generate"], video.uri)
        yield {"type": "final", "result": {
            "profile_industry": profile.industry, "plan_spec": plan_spec,
            "template": to_jsonable(template), "selected_dimensions": selected,
            "plan": to_jsonable(plan), "video": to_jsonable(video),
            "report": report, "trace": [],
            "script": {
                "strategy_path": script_export.get("strategy_path", ""),
                "edit_plan_path": script_export.get("edit_plan_path", ""),
                "strategy": script_export.get("strategy", {}),
                "edit_plan": script_export.get("plan", {}),
            } if script_export else {},
        }}

    @staticmethod
    def _build_plan(template, material_profile, decisions, strategy):
        indexed = {str(item.get("slot_id")): item for item in decisions if isinstance(item, dict)}
        assignments = []
        for shot in template.shot_slots:
            action = indexed.get(str(shot.id), {}).get("action")
            if action not in ("match", "generate"):
                action = "generate" if not material_profile.clips or strategy != "faithful" else "match"
            assignments.append(SlotAssignment(
                slot_id=shot.id, action=action,
                material_clip_id=None if action == "match" else None,
                gen_prompt=shot.want if action == "generate" else None,
            ))
        matches = sum(item.action == "match" for item in assignments)
        confidence = round(matches / len(assignments), 2) if assignments else 0.0
        return ReplicationPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:8]}",
            granularity="action_scene", confidence=confidence,
            source="from-default", slot_assignments=assignments,
            packaging=template.packaging, voice_strategy={}, material_strategy=strategy,
        )
