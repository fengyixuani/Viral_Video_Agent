"""AgentScope 2.0.4 backed agents for the viral-video replication workbench.

`react_agents.py` 中的 FunctionTool 现在只是薄封装：真实业务落在 `src/tools/` 下
（`UnderstandingTool` / `PlanningTool` / `GenerationTool` / `EditingTool` /
`PackagingTool`）。这里保持对外 FunctionTool 名称与函数签名稳定，方便
AgentScope Toolkit 通过 docstring + type hint 生成 schema。
"""
from __future__ import annotations

import json
from typing import Any

from agentscope.agent import Agent
from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, Toolkit, ToolResponse

from as_core import _get_model  # re-use the wenchain-bound OpenAIChatModel
from tools import (
    ASRTool,
    EditingTool,
    GenerationTool,
    PackagingTool,
    PlanningTool,
    RetrievalTool,
    UnderstandingTool,
    VectorStore,
    detect_music_beats,
    detect_shot_boundaries,
)


def _text_response(payload: Any) -> ToolResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    return ToolResponse(content=[TextBlock(type="text", text=text)])


# ---------------------------------------------------------------------------
# 共享的业务层实例（真实/MOCK 业务只走这些实例）
# ---------------------------------------------------------------------------

UNDERSTANDING = UnderstandingTool()
PLANNING = PlanningTool()
GENERATION = GenerationTool()
EDITING = EditingTool()
PACKAGING = PackagingTool()
RETRIEVAL = RetrievalTool()
VECTOR_STORE = VectorStore()
ASR = ASRTool()


# ---------------------------------------------------------------------------
# UnderstandingAgent 的 FunctionTool 薄封装
# ---------------------------------------------------------------------------


def parse_reference(
    video_uri: str,
    video_desc: str,
    intent: str,
    materials: list,
    duration_sec: float,
) -> ToolResponse:
    """回显参考视频输入并附带理解 schema，供理解 Agent 拆解。

    Args:
        video_uri: 参考视频 URI/本地路径。
        video_desc: 视频文字描述。
        intent: 用户的复刻意图。
        materials: 用户提供的素材条目。
        duration_sec: ffmpeg 探测得到的时长（秒）。
    """
    payload = {
        "video_uri": video_uri,
        "video_desc": video_desc,
        "intent": intent,
        "materials": materials,
        "duration_sec": duration_sec,
        "schema_hint": UNDERSTANDING.schema_hint(),
    }
    return _text_response(payload)


def transcribe_audio(media_path: str) -> ToolResponse:
    """对音频或视频做本机离线语音识别（Qwen3-ASR，GPU），返回转写文本与句级时间戳。

    业务在 ASRTool.transcribe（本地模型，不走网络）。常用于理解阶段获取参考视频的
    口播/台词，辅助拆解叙事结构与卖点顺序。

    Args:
        media_path: 音频或视频的 URI/本地路径。
    """
    result = ASR.transcribe(media_path)
    return _text_response({
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "engine": result.get("engine", "qwen3-asr-0.6b"),
        "error": result.get("error", ""),
    })


def assess_materials(materials: list) -> ToolResponse:
    """产出用户素材的粗盘点，供理解 Agent 判断能否匹配镜头。

    盘点时加载隐藏 skill ``user_material_understanding`` 的 schema 与拆分规则，
    指导 Agent 按统一结构客观拆解用户素材。

    Args:
        materials: 用户提供的素材条目列表。
    """
    from skills import get as get_skill

    profile = UNDERSTANDING.profile_materials(materials)
    guide = get_skill("user_material_understanding")
    return _text_response({
        "provided_count": len(materials or []),
        "material_id": profile.material_id,
        "capability": profile.capability,
        "materials": materials,
        "understanding_schema_guide": guide.prompt_hint if guide else "",
    })


# ---------------------------------------------------------------------------
# PlanningAgent 的 FunctionTool 薄封装
# ---------------------------------------------------------------------------


def plan_execute(
    scheme_name: str,
    strategy: str,
    dimensions: list,
    trends: list,
    materials_count: int,
) -> ToolResponse:
    """把规划输入交给业务层整理成结构化 payload。

    Args:
        scheme_name: 选中的复刻方案名称。
        strategy: 复刻策略（faithful/balanced/regenerate）。
        dimensions: 用户勾选的可复刻维度。
        trends: 用户勾选的趋势短语。
        materials_count: 用户素材数量。
    """
    payload = PLANNING.plan_payload(scheme_name, strategy, dimensions, trends, materials_count)
    return _text_response(payload)


def decide_shot(slot_id: int, want: str, strategy: str) -> ToolResponse:
    """逐镜决策入参整形。

    Args:
        slot_id: 分镜 slot id。
        want: 分镜目标描述。
        strategy: 复刻策略。
    """
    payload = PLANNING.decide_payload(slot_id, want, strategy)
    return _text_response(payload)


def generate_shot(gen_prompt: str, duration: float) -> ToolResponse:
    """镜头生成薄封装（业务在 GenerationTool.run）。

    Args:
        gen_prompt: 生成镜头的提示词。
        duration: 目标时长（秒）。
    """
    result = GENERATION.run(gen_prompt=gen_prompt, duration=duration)
    return _text_response(result)


def edit_timeline(shots: list) -> ToolResponse:
    """时间线组装薄封装（业务在 EditingTool.run）。

    Args:
        shots: 有序的镜头描述列表。
    """
    result = EDITING.run(shots=shots)
    return _text_response(result)


def package_video(shots: list, duration: float) -> ToolResponse:
    """成片包装薄封装（业务在 PackagingTool.run）。

    Args:
        shots: 最终镜头列表。
        duration: 成片时长（秒）。
    """
    video = PACKAGING.run(shots=shots, duration_sec=duration)
    return _text_response({"uri": video.uri, "duration_sec": video.duration_sec, "shots": video.shots})


def retrieve_video_segments(query: str, segments: list, top_k: int = 5) -> ToolResponse:
    """按语义相似度从用户素材片段里检索与 query 最相关的片段（业务在 RetrievalTool.retrieve）。

    用千帆 qwen3-embedding-0.6b 文本向量计算余弦相似度，常用于逐镜决策时为某个
    镜头意图匹配最合适的用户素材片段。

    Args:
        query: 检索语句，通常是某个镜头的意图/描述。
        segments: 候选素材片段列表（素材理解产出的 asset_segments），每项含
            one_sentence_summary / visual_description / keywords 等字段。
        top_k: 返回最相关的前 k 个片段，默认 5。
    """
    result = RETRIEVAL.retrieve(query, segments, top_k=top_k)
    return _text_response(result)


def index_segments(segments: list, source: str = "") -> ToolResponse:
    """把素材理解产出的片段按字段写入向量库，供后续检索（业务在 VectorStore.index_segments）。

    Args:
        segments: 素材理解产出的 asset_segments。
        source: 素材来源标识（如素材文件名），写入 meta 便于回溯。
    """
    return _text_response(VECTOR_STORE.index_segments(segments, source=source))


def search_user_materials(query: str, top_k: int = 5) -> ToolResponse:
    """只在用户素材向量库里按语义检索 top_k 个最相关片段（业务在 VectorStore.search）。

    素材可行性验证阶段用它为某个镜头在用户素材中查找对应镜头。

    Args:
        query: 检索语句，通常是某个参考镜头的意图/画面描述。
        top_k: 返回最相关的前 k 个片段，默认 5。
    """
    return _text_response(VECTOR_STORE.search(query, top_k=top_k))


# ---------------------------------------------------------------------------
# Toolkits + agents
# ---------------------------------------------------------------------------


UNDERSTANDING_TOOLS = [parse_reference, assess_materials, transcribe_audio, detect_shot_boundaries, detect_music_beats]
PLANNING_TOOLS = [plan_execute, decide_shot, generate_shot, edit_timeline, package_video, detect_music_beats, retrieve_video_segments]


def build_understanding_toolkit() -> Toolkit:
    return Toolkit(tools=[FunctionTool(fn) for fn in UNDERSTANDING_TOOLS])


def build_planning_toolkit() -> Toolkit:
    return Toolkit(tools=[FunctionTool(fn) for fn in PLANNING_TOOLS])


UNDERSTANDING_SYSTEM = (
    "你是爆款视频的多模态理解 Agent。工具：parse_reference / assess_materials 汇总输入；"
    "detect_shot_boundaries 用 ffmpeg 场景切分给出真实镜头切点；"
    "detect_music_beats 用 librosa 给出 BPM 与节拍点，可辅助节奏拆解；"
    "transcribe_audio 用本机离线 ASR（Qwen3-ASR, GPU）转写参考视频口播/台词，辅助拆解叙事与卖点。"
    "请综合视觉、镜头切点、节奏、口播与文本，产出严格 JSON（industry_guess、shot_slots、schemes 等），字段值使用简体中文。"
)

PLANNING_SYSTEM = (
    "你是爆款视频复刻的规划 Agent。工具：plan_execute / decide_shot 组织复刻计划，"
    "generate_shot / edit_timeline / package_video 执行镜头生成、剪辑与包装；"
    "retrieve_video_segments 用向量检索为某个镜头意图匹配最相关的用户素材片段；"
    "detect_music_beats 可获取 BGM/参考视频的 BPM 与节拍点用于卡点。"
    "所有回复以简体中文 JSON 输出。"
)


def build_understanding_agent(model_name: str = "ali-qwen3.7-plus") -> Agent:
    return Agent(
        name="UnderstandingAgent",
        system_prompt=UNDERSTANDING_SYSTEM,
        model=_get_model(model_name),
        toolkit=build_understanding_toolkit(),
    )


def build_planning_agent(model_name: str = "ali-qwen3.7-max") -> Agent:
    return Agent(
        name="PlanningAgent",
        system_prompt=PLANNING_SYSTEM,
        model=_get_model(model_name),
        toolkit=build_planning_toolkit(),
    )


__all__ = [
    "UNDERSTANDING",
    "PLANNING",
    "GENERATION",
    "EDITING",
    "PACKAGING",
    "RETRIEVAL",
    "VECTOR_STORE",
    "UNDERSTANDING_TOOLS",
    "PLANNING_TOOLS",
    "build_understanding_toolkit",
    "build_planning_toolkit",
    "build_understanding_agent",
    "build_planning_agent",
    "parse_reference",
    "assess_materials",
    "plan_execute",
    "decide_shot",
    "generate_shot",
    "edit_timeline",
    "package_video",
    "retrieve_video_segments",
    "index_segments",
    "search_user_materials",
    "detect_shot_boundaries",
    "detect_music_beats",
]
