"""编排阶段（B 负责：编排 → 剪辑 的编排端）。

`OrchestrationAgent.replicate_stream` 消费 A（理解）产出的 AnalysisResult（契约①，随 bundle
传入：template + schemes + feasibility + material_understanding），编排出 strategy JSON
（契约②，B 内部产物），供剪辑链路 `editing/` 使用。

复用理解阶段（A）的两个「理解服务」函数：`understand_materials` / `verify_materials`
（前端时序缺素材理解/可行性时的自愈重建）——这是 B→A 的服务级依赖，非循环。
"""
import asyncio
import json
import uuid

import contracts
import obs
from agent_runtime import ReplicationAgentBase
from profile import load_profile
from schema import ReplicationPlan, SlotAssignment
from serialize import to_jsonable
from thumbs import extract_remake_thumbs

from understanding.feasibility import verify_materials
from understanding.material_understanding import understand_materials

from .orchestration import orchestrate_from_feasibility, orchestrate_structure_first
from .prompts import DECIDE_SYSTEM, PLAN_SYSTEM
from .scriptgen import export_scripts

_log = obs.get_logger("orchestration")


class OrchestrationAgent(ReplicationAgentBase):
    """编排 Agent：把理解结果编排成复刻方案（strategy JSON）。"""

    async def replicate_stream(self, bundle):
        rid = obs.new_request_id()
        _log.info("[%s] replicate start industry=%s strategy=%s dims=%s trends=%d materials=%d",
                  rid, bundle.industry_id, bundle.material_strategy,
                  bundle.selected_dimensions, len(bundle.selected_trends or []), len(bundle.materials or []))
        # 契约①边界自检（消费端）：A 给的 AnalysisResult（随 bundle.template）若缺字段只告警
        _ap = contracts.validate_analysis_result({
            "template": bundle.template if isinstance(bundle.template, dict) else {},
            "schemes": (bundle.template or {}).get("schemes", []) if isinstance(bundle.template, dict) else [],
        })
        if _ap:
            _log.warning("[%s] [contract] 收到的 AnalysisResult 不符契约(v%s)：%s",
                         rid, contracts.CONTRACT_VERSION, contracts.summarize(_ap))
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
        pure_music = bool(getattr(bundle, "pure_music", False))
        # 自愈：前端可能因时序或选历史工程未回填而传空 material_understanding / feasibility，
        # 导致编排拿到空池 → 退化成「全量补拍」。这里用素材缓存重建素材池（并按本次 rid 建向量）。
        need_feas_rebuild = (mode == "shot" and not shot_feasibility)
        if bundle.materials and (not material_understanding or need_feas_rebuild):
            rebuild_key = f"mu-rebuild-{rid}"
            yield {"type": "step", "phase": "解析", "key": rebuild_key, "state": "running",
                   "title": "重建素材理解", "thought": f"复刻请求缺素材理解/可行性，从缓存重建 {len(bundle.materials)} 项素材画像"}
            rebuilt = {}
            try:
                async for event in understand_materials(bundle.materials, use_cache=True, task_id=rid):
                    if event.get("__materials_result__"):
                        rebuilt = event.get("results", {})
                        continue
            except Exception as exc:  # noqa: BLE001
                _log.warning("[%s] material_understanding rebuild failed: %s", rid, exc)
            material_understanding = material_understanding or rebuilt or {}
            _log.info("[%s] material_understanding rebuilt: %d results", rid, len(rebuilt))
            yield {"type": "step", "phase": "解析", "key": rebuild_key, "state": "done",
                   "title": "重建素材理解完成",
                   "thought": (f"素材池就绪（{len(material_understanding)} 项）"
                               if material_understanding else "缓存中无素材理解结果，素材池为空")}
        # whq 结构级复刻：确定性一步出成片（硬剪+配音+字幕+语速贴参考）。不走下面的
        # structure/shot 编排与 MOCK 出片，直接调 editing/whq_clone.runner 并 return。
        if mode == "whq_clone":
            async for event in self._run_whq_clone(rid, bundle, template, material_understanding):
                yield event
            return
        effective_feasibility = {}
        structure_template = None
        reference_dict = to_jsonable(template)
        # 镜头模式且 feasibility 为空 → 用（已建好向量的）素材池重算逐镜可行性，避免全 reshoot
        if need_feas_rebuild and bundle.materials:
            ref_shots = reference_dict.get("shot_slots", []) if isinstance(reference_dict, dict) else []
            if ref_shots:
                fkey = f"feas-rebuild-{rid}"
                yield {"type": "step", "phase": "可行性验证", "key": fkey, "state": "running",
                       "title": "重建可行性验证", "thought": f"镜头优先模式缺可行性结果，为 {len(ref_shots)} 个参考镜头重新匹配用户素材"}
                try:
                    async for event in verify_materials(ref_shots, task_id=rid):
                        if event.get("__feasibility_result__"):
                            shot_feasibility = event.get("results", {})
                except Exception as exc:  # noqa: BLE001
                    _log.warning("[%s] feasibility rebuild failed: %s", rid, exc)
                hit = sum(1 for v in (shot_feasibility or {}).values()
                          if isinstance(v, dict) and v.get("status") in ("direct", "partial"))
                _log.info("[%s] feasibility rebuilt: %d shots, %d 命中", rid, len(shot_feasibility or {}), hit)
                yield {"type": "step", "phase": "可行性验证", "key": fkey, "state": "done",
                       "title": "重建可行性验证完成", "thought": f"{hit}/{len(ref_shots)} 个镜头匹配到用户素材"}
        if mode == "structure":
            orchestration_stream = orchestrate_structure_first(
                reference_dict, material_understanding,
                scheme_name=bundle.scheme_name, material_strategy=bundle.material_strategy,
                selected_dimensions=selected, selected_trends=bundle.selected_trends,
                intent=bundle.intent,
                missing_shot_mode=(getattr(bundle, "missing_shot_mode", "aigc") or "aigc"),
            )
        elif shot_feasibility:
            orchestration_stream = orchestrate_from_feasibility(
                reference_dict, shot_feasibility,
                scheme_name=bundle.scheme_name, material_strategy=bundle.material_strategy,
                selected_dimensions=selected, selected_trends=bundle.selected_trends,
                intent=bundle.intent, shot_replicate=pure_music,
                missing_shot_mode=(getattr(bundle, "missing_shot_mode", "aigc") or "aigc"),
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
        if mode == "structure" and not (structure_template and structure_template.get("shot_slots")):
            pool_empty = not material_understanding
            yield {"type": "step", "phase": "编排", "key": f"orch-empty-{rid}", "state": "done",
                   "title": "结构优先编排未产出可用镜头",
                   "thought": ("素材池为空（material_understanding 缺失），无法编排出用素材的镜头；"
                               "请重新「理解爆款」以生成素材理解，或检查素材上传。"
                               if pool_empty else
                               "编排 Agent 未返回有效镜头，将退回参考结构做全量补拍。")}
            _log.warning("[%s] structure-first produced no usable slots (pool_empty=%s)", rid, pool_empty)
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

    async def _run_whq_clone(self, rid, bundle, template, material_understanding):
        """reproduce_mode=='whq_clone' 分支：调 editing/whq_clone.runner。

        默认 **Agent 形态**（WHQ_AGENT_EDIT != 0）：这里只做编排，产出
        selected_editing_strategy.json + connector_context.json，出片交给
        editing/loop.py 的「剪辑 Agent 出片 → 审片 Agent 看片 → 不满意自动重剪」循环
        （前端 Agent 剪辑面板 / POST /api/agent_edit）。第 1 轮用 whq 的确定性分配作基线。

        WHQ_AGENT_EDIT=0 时回到 legacy 的 workflow 形态：whq 一步直接出成片。
        whq 全链路是同步重任务（ffmpeg/ASR/TTS），用 asyncio.to_thread 丢线程池跑，
        避免阻塞 SSE 事件循环。
        """
        import os
        import sys

        whq_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "editing", "whq_clone")
        if whq_dir not in sys.path:
            sys.path.insert(0, whq_dir)
        key = f"whq-{rid}"
        try:
            import runner as whq_runner
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] whq_clone import failed: %s", rid, exc)
            yield {"type": "step", "phase": "复刻", "key": key, "state": "done",
                   "title": "whq 结构级复刻不可用", "thought": f"模块导入失败：{exc!s}"}
            yield {"type": "final", "result": {}}
            return

        # 参考视频路径解析：video_uri 绝对存在则用之，否则相对项目根。
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ref_uri = bundle.video_uri or ""
        ref_video = ref_uri if os.path.isfile(ref_uri) else os.path.join(project_root, ref_uri)
        if not os.path.isfile(ref_video):
            ref_video = ref_uri  # 交给 runner 兜底（无参考视频→template 近似 key_beats）

        mu = material_understanding or (
            (bundle.template or {}).get("material_understanding") if isinstance(bundle.template, dict) else {})
        out_dir = os.path.join(project_root, "outputs", "whq_clone", rid)
        out_path = os.path.join(out_dir, "whq_clone.mp4")
        tpl = bundle.template if isinstance(bundle.template, dict) else {}
        agent_edit = os.getenv("WHQ_AGENT_EDIT", "1") not in ("0", "false", "False")
        if tpl.get("whq_agent_edit") is not None:
            agent_edit = bool(tpl.get("whq_agent_edit"))

        common = dict(
            product_name=bundle.intent or "",
            # 优先吃真实 understanding 产物（A 可在 template 里回填这些路径 / slug）
            dna_md=tpl.get("dna_md") or tpl.get("dna_path"),
            assets_json=tpl.get("assets_json") or tpl.get("assets_path"),
            asr_json=tpl.get("asr_json") or tpl.get("source_asr_path"),
            slug=tpl.get("slug"),
            # 无现成产物时，跑 Split 深度理解生成 Split 级 DNA+素材理解（全模型，达手工版质量）
            deep_understanding=bool(tpl.get("deep_understanding", True)),
        )
        arg = {"material_understanding": mu, "template": to_jsonable(template)}

        if agent_edit:
            yield {"type": "step", "phase": "复刻", "key": key, "state": "running",
                   "title": "whq 结构级复刻（编排）",
                   "thought": "重建参考 key_beats → 1:1 分配 → 原声/克隆决策 + 句子级对窗 → 产出剪辑方案，交 Agent 剪辑出片"}
        else:
            yield {"type": "step", "phase": "复刻", "key": key, "state": "running",
                   "title": "whq 结构级复刻", "thought": "重建参考 key_beats → 1:1 分配 → 硬剪 → 配音 → 字幕 → 语速贴参考（可能耗时数分钟）"}
        try:
            fn = whq_runner.plan_whq_clone if agent_edit else whq_runner.run_whq_clone
            info = await asyncio.to_thread(fn, arg, out_path, ref_video, **common)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[%s] whq_clone run failed: %s", rid, exc)
            yield {"type": "step", "phase": "复刻", "key": key, "state": "done",
                   "title": "whq 结构级复刻失败", "thought": f"{exc!s}"}
            yield {"type": "final", "result": {}}
            return

        probs = info.get("strategy_problems") or []
        if agent_edit:
            warns = info.get("warnings") or []
            if warns:
                _log.warning("[%s] whq_clone plan warnings: %s", rid, "; ".join(warns))
                yield {"type": "step", "phase": "复刻", "key": f"{key}-warn", "state": "done",
                       "title": "whq 编排告警（会明显影响成片）",
                       "thought": warns[0],
                       "observation": "\n".join("- " + w for w in warns)}
            _log.info("[%s] whq_clone plan done strategy=%s segs=%s original=%s pool=%s gap=%s contract=%s",
                      rid, info.get("strategy_path"), info.get("n_segments"),
                      info.get("n_original_voice"), info.get("n_pool"), info.get("n_gap"),
                      "OK" if not probs else contracts.summarize(probs))
            yield {"type": "step", "phase": "复刻", "key": key, "state": "done",
                   "title": "whq 剪辑方案已生成",
                   "thought": (f"{info.get('n_segments')} 个结构段落，其中 {info.get('n_original_voice')} 段保留用户原声；"
                               f"素材池 {info.get('n_pool')} 段可供 Agent 重新召回"),
                   "observation": (f"strategy: {info.get('strategy_path')}\n"
                                   f"缺口镜头: {info.get('n_gap')}；key_beats 重建: {info.get('dna_reconstructed')}\n"
                                   "下一步：点「Agent 剪辑」出片（剪辑→审片→自动重剪）")}
            yield {"type": "final", "result": {
                "template": to_jsonable(template),
                "whq_process": info.get("process") or {},
                "report": {"reproduce_mode": "whq_clone", "n_gap": info.get("n_gap"),
                           "n_original_voice": info.get("n_original_voice"),
                           "dna_reconstructed": info.get("dna_reconstructed")},
                "trace": [],
                "script": {
                    "strategy_path": info.get("strategy_path", ""),
                    "edit_plan_path": info.get("plan_path", ""),
                },
            }}
            return

        # 成片在 project_root/outputs/ 下 -> 转成可被 /outputs/ 服务的相对 uri 供前端播放
        final_abs = info.get("final", "")
        try:
            _rel = os.path.relpath(os.path.realpath(final_abs), os.path.realpath(project_root))
            video_uri = _rel.replace(os.sep, "/") if not _rel.startswith("..") else final_abs
        except Exception:  # noqa: BLE001
            video_uri = final_abs
        _log.info("[%s] whq_clone done final=%s strategy=%s gap=%s reconstructed=%s contract=%s",
                  rid, info.get("final"), info.get("strategy_path"), info.get("n_gap"),
                  info.get("dna_reconstructed"), "OK" if not probs else contracts.summarize(probs))
        yield {"type": "step", "phase": "复刻", "key": key, "state": "done",
               "title": "whq 结构级复刻完成",
               "thought": f"成片：{info.get('final')}",
               "observation": (f"strategy: {info.get('strategy_path')}\n"
                               f"缺口镜头: {info.get('n_gap')}；key_beats 重建: {info.get('dna_reconstructed')}")}
        yield {"type": "final", "result": {
            "template": to_jsonable(template),
            "video": {"uri": video_uri},
            "whq_process": info.get("process") or {},
            "report": {"reproduce_mode": "whq_clone", "n_gap": info.get("n_gap"),
                       "dna_reconstructed": info.get("dna_reconstructed")},
            "trace": [],
            "script": {
                "strategy_path": info.get("strategy_path", ""),
                "edit_plan_path": info.get("plan_path", ""),
            },
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
