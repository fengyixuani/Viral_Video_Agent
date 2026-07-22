"""Autonomous tool-selection loop for the UnderstandingAgent.

给理解阶段一个轻量的 ReAct 循环：Agent 先拿到本次任务的元数据（参考视频时长、
用户素材数量与平均时长、可用工具目录），然后自己决定要不要调用工具、调用哪个、
用什么参数。每一次思考、每一次工具调用都会通过异步生成器 yield 出来，供
`orchestrator.analyze_stream` 转成 SSE 事件在前端"Agent 思考过程"里可视化。
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncGenerator

from agentscope.message import ToolCallBlock
from agentscope.state import AgentState

import as_core
import obs

_log = obs.get_logger("planner")

# 每个工具的语义 + 参数模板；Agent 用它做决策，也用来构造 ToolCallBlock。
TOOL_CATALOG = {
    "detect_shot_boundaries": {
        "desc": "对参考视频做真实镜头切分，返回每个 cut 的时间点（秒）。适用于确认真实分镜数量、判断快剪/长镜头。",
        "args": {"video_path": "本地参考视频绝对路径", "threshold": "可选，默认 0.18"},
        "needs_local_video": True,
    },
    "detect_music_beats": {
        "desc": "对参考视频或音频提取 BPM + beat 时间点。适合判断卡点节奏、是否需要 BGM 卡点。",
        "args": {"audio_path": "本地视频或音频绝对路径", "include_onsets": "可选，false"},
        "needs_local_video": True,
    },
    "transcribe_audio": {
        "desc": "对参考视频做语音识别（ASR），返回口播/台词文本。适合有大量口播、需要理解叙事结构与卖点顺序时。",
        "args": {"media_path": "本地视频/音频绝对路径"},
        "needs_local_video": True,
    },
    "assess_materials": {
        "desc": "汇总用户提供的素材数量与画像。素材数为 0 或只关心生成时可跳过。",
        "args": {"materials": "素材列表（由系统自动传入）"},
        "needs_local_video": False,
    },
}


TOOL_TITLE = {
    "detect_shot_boundaries": "镜头切分检测",
    "detect_music_beats": "音乐节奏检测",
    "transcribe_audio": "语音识别转写",
    "assess_materials": "盘点用户素材",
}


def _build_system_prompt() -> str:
    lines = [
        "你是爆款视频复刻的理解阶段编排 Agent。",
        "你会先看到一次任务的元数据（参考视频时长、素材数量、素材平均时长等），",
        "然后从下面的工具目录里自主决定要调用哪些工具、按什么顺序调用。",
        "允许一次不调用任何工具（如果元数据足以让下一步 LLM 直接理解），",
        "也允许连续调用多个工具，每步都要说明 thought。",
        "",
        "重要约束：",
        "1. 如果参考视频时长 `reference_duration_sec` 小于 120 秒，不要调用 detect_shot_boundaries；",
        "   直接把视频原样交给下游 VLM 让它自己看画面，无需先做机械切镜。",
        "2. detect_music_beats 只有在需要 BGM 卡点或想验证节奏时才必要，其余可跳过。",
        "3. assess_materials 只有素材数量 > 0 时才有意义。",
        "",
        "工具目录：",
    ]
    for name, spec in TOOL_CATALOG.items():
        lines.append(f"- {name}: {spec['desc']} 参数: {json.dumps(spec['args'], ensure_ascii=False)}")
    lines += [
        "",
        "输出规则：每一步严格只输出一个 JSON 对象，形如：",
        '{"thought":"简述你的判断", "action":"call_tool|finish", "tool":"工具名(仅 call_tool 需要)", "args":{...}, "reason":"finish 时的收束原因"}',
        "只有当你决定不再调用工具时，才输出 action=finish。",
        "禁止输出多余文本。",
    ]
    return "\n".join(lines)


def _tool_available(name: str, has_local_video: bool, has_materials: bool) -> bool:
    spec = TOOL_CATALOG.get(name)
    if not spec:
        return False
    if spec["needs_local_video"] and not has_local_video:
        return False
    if name == "assess_materials" and not has_materials:
        return False
    return True


async def _ask_llm_json(system: str, user: str) -> tuple[dict, AsyncGenerator]:
    """跑一次 LLM 拿到 JSON 决策；同时 yield reasoning。"""
    # 由 caller 使用。这里保留占位（实际实现见 run_understanding_planner）。
    raise NotImplementedError


async def run_understanding_planner(
    *,
    metadata: dict,
    toolkit,
    max_iters: int = 4,
    allowed_tools: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """驱动理解阶段的 ReAct 循环。

    Parameters
    ----------
    allowed_tools:
        用户勾选允许的工具白名单。``None`` 表示不限，任何 ``TOOL_CATALOG`` 里的
        工具都可以调用；给一个列表则严格过滤，超出的工具被视为不可用。

    yields
    ------
    reasoning: 模型的 thought 增量
    step:      每次工具调用的开始/结束
    __planner_result__: 循环结束时的收束记录（不发前端）
    """
    system = _build_system_prompt()
    has_local_video = bool(metadata.get("local_video"))
    has_materials = metadata.get("material_count", 0) > 0
    allow_set = None if allowed_tools is None else {str(t) for t in allowed_tools}

    def _allowed(name: str) -> bool:
        if not _tool_available(name, has_local_video, has_materials):
            return False
        return allow_set is None or name in allow_set

    history = [
        {"role": "元数据", "content": metadata},
    ]
    if allow_set is not None:
        history.append({"role": "工具白名单", "content": sorted(allow_set)})
    findings: list[dict] = []
    finish_reason = ""

    for step in range(max_iters):
        user = json.dumps({
            "step": step + 1,
            "max_iters": max_iters,
            "history": history,
            "available_tools": [name for name in TOOL_CATALOG if _allowed(name)],
        }, ensure_ascii=False)

        content = ""
        async for item in as_core.stream(system, user):
            if item.get("reasoning"):
                yield {"type": "reasoning", "phase": "调度", "text": item["reasoning"]}
            elif "content" in item:
                content = item["content"]
        try:
            decision = as_core.parse_json(content) if content.strip() else {}
        except (ValueError, TypeError, json.JSONDecodeError):
            decision = {}

        action = decision.get("action") or ("finish" if not decision else "call_tool")
        thought = decision.get("thought") or ""
        # 决策本身只作为 reasoning 进入当前调度卡的详情，不单独成块
        if thought:
            yield {"type": "reasoning", "phase": "调度", "text": f"\n决策：{thought}\n"}

        if action == "finish" or not decision:
            finish_reason = decision.get("reason") or "元数据充足，直接进入结构化理解"
            break

        tool_name = decision.get("tool") or ""
        if tool_name not in TOOL_CATALOG or not _allowed(tool_name):
            yield {"type": "reasoning", "phase": "调度",
                   "text": f"\n忽略无效或未开启的工具 {tool_name}\n"}
            history.append({"role": "工具错误", "content": f"invalid tool {tool_name}"})
            continue

        args = decision.get("args") or {}
        # 自动补齐必要参数
        if tool_name in ("detect_shot_boundaries",) and metadata.get("local_video"):
            args.setdefault("video_path", metadata["local_video"])
        if tool_name in ("detect_music_beats",) and metadata.get("local_video"):
            args.setdefault("audio_path", metadata["local_video"])
        if tool_name == "transcribe_audio" and metadata.get("local_video"):
            args.setdefault("media_path", metadata["local_video"])
        if tool_name == "assess_materials":
            args = {"materials": metadata.get("materials", [])}

        # 调工具，采集 content + metadata；工具调用作为一个带状态的步骤方块
        call_id = uuid.uuid4().hex[:12]
        title = TOOL_TITLE.get(tool_name, tool_name)
        block = ToolCallBlock(id=call_id, name=tool_name, input=json.dumps(args, ensure_ascii=False))
        yield {"type": "step", "phase": "调度", "key": call_id, "state": "running",
               "title": title, "thought": f"正在{title}"}
        observation_text = ""
        tool_metadata: dict = {}
        state = AgentState()
        async for chunk in toolkit.call_tool(block, state):
            content_blocks = getattr(chunk, "content", None) or []
            for entry in content_blocks:
                text = entry.get("text") if isinstance(entry, dict) else getattr(entry, "text", "")
                if text:
                    observation_text = text
            meta = getattr(chunk, "metadata", None)
            if meta:
                tool_metadata = meta

        yield {"type": "step", "phase": "调度", "key": call_id, "state": "done",
               "title": title, "thought": f"{title}完成", "observation": observation_text[:400]}
        _log.info("planner tool=%s args=%s -> %s", tool_name, args, observation_text[:120])
        history.append({
            "role": "工具观察",
            "tool": tool_name,
            "summary": observation_text[:400],
            "metadata_keys": list(tool_metadata.keys()),
        })
        findings.append({"tool": tool_name, "summary": observation_text, "metadata": tool_metadata})
    else:
        finish_reason = f"达到最大迭代 {max_iters} 次"

    yield {"__planner_result__": True, "findings": findings, "reason": finish_reason}
