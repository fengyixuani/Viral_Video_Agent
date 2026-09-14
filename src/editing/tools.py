"""剪辑 Agent 的工具箱（agent_cut「纯 Agent 剪辑」链路）。

剪辑 Agent 不再被限制在每个 slot 预分配的候选池里，而是可以对**全量素材池**
（connector_context.json 的 segments）自由召回。工具用 AgentScope 原生 ``FunctionTool``
声明（schema 由 docstring + type hint 自动生成，与 src/agent/react_agents.py 一致）。

内置工具：retrieve（语义召回）/ verify（按需 VLM 验证）/ place（放入，后台查重叠→仲裁）/
finish（结束）。重叠检测/去重/仲裁的编排在 loop.py。

────────────────────────────────────────────────────────────────────────
如何新增一个 tool（让别人写 tool 也能兼容）：
  1) 写一个带 **docstring（含 Args:）+ type hint** 的函数 ``_tool_xxx(...)``，加进
     ``EDIT_FUNCTION_TOOLS``——schema 会自动生成、自动渲染进剪辑 Agent 的 prompt，
     也会自动出现在 ``tool_schemas()`` / ``GET /api/agent_edit/tools`` 的展示里。
  2) 用 ``@edit_tool_handler("xxx")`` 注册它的执行处理器：
        @edit_tool_handler("xxx")
        async def _handle_xxx(ctx, action) -> dict:
            # ctx = {"toolbox","used","placements","slot_meta","notes"}
            # action = 剪辑 Agent 输出的动作 JSON（含它给的参数）
            return {"ok": True, ...}   # 返回值会作为 observation 反馈给 Agent
  内置的 place/finish 是流程控制动作，特殊处理不走 handler；其余新工具都走上面这套。
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import re
import subprocess

import obs

try:
    import imageio_ffmpeg
    _FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG_BIN = os.getenv("FFMPEG", "ffmpeg")

_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentscope.tool import FunctionTool, Toolkit
from tools.retriever import Retriever
from tools.vlm import VLMTool

_log = obs.get_logger("agent_edit_tools")


# --------------------------------------------------------------------------- #
# 工具声明（AgentScope 原生：FunctionTool 由 docstring + type hint 生成 schema，
# 与 src/agent/react_agents.py 的做法一致，单一事实来源）
# --------------------------------------------------------------------------- #
# 说明：本项目和它自己的 AgentScope agent 一样，**手动分发**工具（见
# orchestrator._run_tool 用 toolkit.call_tool 手动构造调用），不依赖 wenchain 的
# native function-calling。所以下面这些函数是「工具契约声明」——schema 从它们的
# 签名+docstring 自动生成、渲染进剪辑 Agent 的 prompt；真正的执行由 loop.py 的
# ReAct 分发器调用 EditToolbox / 放入逻辑完成（行为与之前完全一致）。


def _tool_retrieve(slot_id: str, query: str, top_k: int = 6):
    """从全量用户素材池按语义召回候选片段（不受任何预分配候选限制）。为某个 slot 按它的 DNA 角色/意图自拟检索词；返回候选的 global_asset_id / 画面描述 / 口播 speech / 关键词。

    Args:
        slot_id: 该次召回服务的槽位 id，如 S03
        query: 检索词：围绕该 slot 的角色/意图/期望画面自由描述
        top_k: 返回候选数，默认 6
    """
    raise NotImplementedError("schema 声明；由剪辑 ReAct 分发器执行")


def _tool_verify(global_asset_id: str, question: str):
    """按需 VLM 视觉验证：仅当你对某候选画面拿不准时，让视觉模型直接看该片段确认。较慢，不确定才用。

    Args:
        global_asset_id: 要验证的候选片段 id
        question: 想确认的画面问题，如：画面里是否有敷面膜动作
    """
    raise NotImplementedError("schema 声明；由剪辑 ReAct 分发器执行")


def _tool_place(slot_id: str, global_asset_id: str, source_time_range: str = "",
                target_duration: float = 0.0, caption: str = "", burn_caption: bool = True,
                speed: float = 1.0, note: str = ""):
    """把某片段放入某 slot。放入后系统会后台检测重叠：若该片段（同源+时间相交）或同一句口播已被别的 slot 占用，会触发审核 Agent 裁决归属，输家 slot 需你另选。

    Args:
        slot_id: 目标槽位 id
        global_asset_id: 要放入的片段 id
        source_time_range: 片段截取区间，如 0.00-3.80；**这就是该镜在成片里的实际时长**，想更长就选长一点
        target_duration: 仅供参考的目标时长（秒），不必严格对齐；成片时长以 source_time_range 为准
        caption: 烧录字幕；须用该片段自己的口播 speech（可轻微精简），语义须与原声一致
        burn_caption: 是否烧字幕；该片段无口播时置 false
        speed: 变速，只在需要加快节奏时用（1.0~1.6）；不要用 <1 去拉慢凑时长
        note: 可选，本步思路一句话
    """
    raise NotImplementedError("schema 声明；由剪辑 ReAct 分发器执行")


def _tool_finish():
    """所有 slot 都放好后结束本轮剪辑。"""
    raise NotImplementedError("schema 声明；由剪辑 ReAct 分发器执行")


def _tool_tts_clone(slot_id: str, ref_global_asset_id: str, text: str):
    """用某素材片段的音色做零样本声音克隆（CosyVoice），把改写后的文案念出来，生成配音替换该镜原声。用于「改写口播文案 + 保留原说话人音色」的带货重配音。需先对该 slot 用 place 选好画面片段。

    Args:
        slot_id: 要配音的槽位 id（须已 place 选好画面）
        ref_global_asset_id: 作为音色参考的素材片段 id（通常用该镜自己或目标说话人的片段，需该片段有口播 speech）
        text: 要让克隆音色念出来的文案（改写后的口播）；**须衔接上下文**——结合本镜 role/want 和前后镜头的语音(见输入 narrative)，自然承接、不与前后重复；该镜字幕会随之改成这段文案
    """
    raise NotImplementedError("schema 声明；由剪辑 ReAct 分发器执行")


# FunctionTool 列表 = 剪辑 Agent 可用工具的唯一声明处；Toolkit 与 react_agents.py 一致
EDIT_FUNCTION_TOOLS = [FunctionTool(fn) for fn in
                       (_tool_retrieve, _tool_verify, _tool_place, _tool_finish, _tool_tts_clone)]
EDIT_TOOLKIT = Toolkit(tools=EDIT_FUNCTION_TOOLS)


def register_edit_tool(fn):
    """把一个新工具的**契约声明**追加进剪辑 Agent 的工具清单（幂等，按函数名去重）。

    与 ``edit_tool_handler``（执行处理器）配套：声明进 prompt、执行走 handler。链路专属
    工具（如 whq_clone 的 place_original）用它在 import 时挂上，不必改动本文件。
    """
    name = getattr(fn, "__name__", "")
    if any(getattr(ft, "name", "") == name for ft in EDIT_FUNCTION_TOOLS):
        return
    ft = FunctionTool(fn)
    EDIT_FUNCTION_TOOLS.append(ft)
    EDIT_TOOLKIT.add_tools([ft]) if hasattr(EDIT_TOOLKIT, "add_tools") else None


def render_tools_spec(tools=None) -> str:
    """把 FunctionTool（AgentScope 自动生成的 schema）渲染成 prompt 里的工具清单 + 调用模板。"""
    tools = tools or EDIT_FUNCTION_TOOLS
    lines = []
    for ft in tools:
        name = ft.name.replace("_tool_", "") if ft.name.startswith("_tool_") else ft.name
        props = (ft.input_schema or {}).get("properties", {}) if isinstance(ft.input_schema, dict) else {}
        parts = [f'"action":"{name}"']
        for pname, pspec in props.items():
            parts.append(f'"{pname}":<{(pspec or {}).get("description", pname)}>')
        call = "{" + ", ".join(parts) + "}"
        lines.append(f"- {name}：{ft.description}\n  调用：{call}")
    return "\n".join(lines)


def tool_schemas(tools=None) -> list:
    """导出所有工具的 OpenAI 标准 function schema（用于 UI/接口展示、给别人写 tool 参考）。"""
    tools = tools or EDIT_FUNCTION_TOOLS
    out = []
    for ft in tools:
        name = ft.name.replace("_tool_", "") if ft.name.startswith("_tool_") else ft.name
        out.append({"type": "function", "function": {
            "name": name, "description": ft.description, "parameters": ft.input_schema or {},
        }})
    return out


# 执行处理器注册表：工具「怎么执行」的插件点（schema 在上面的 FunctionTool，执行在这里）。
_TOOL_HANDLERS: dict = {}


def edit_tool_handler(name: str):
    """装饰器：注册某工具的执行处理器 ``async def handler(ctx, action) -> dict``。"""
    def deco(fn):
        _TOOL_HANDLERS[name] = fn
        return fn
    return deco


def get_tool_handler(name: str):
    return _TOOL_HANDLERS.get(name)


def normalize_speech(text: str) -> str:
    """口播归一：去标点/空白/大小写，用于判定两段字幕是否实为同一句。"""
    return re.sub(r"[\s，。！？、,.!?~…\-—:：;；\"'“”‘’()（）]+", "", str(text or "")).lower()


# 拍摄现场的口令/语气词/口水话。这些词去掉后没剩下实词的原声 = 废话（见 is_filler_speech）。
# 按长度倒序剥离，避免"一下"被"一"之类的短词提前吃掉。
_FILLER_TOKENS = tuple(sorted((
    "ok", "okay", "然后", "这个", "那个", "就是说", "就是", "等一下", "一下", "等等",
    "够了", "可以了", "好了", "行了", "齐了", "成了", "展示", "开始", "预备", "准备",
    "往上", "往下", "往左", "往右", "过来", "过去", "再来", "来", "再", "停",
    "对对", "对", "好", "行", "嗯", "啊", "哦", "呃", "诶", "唉", "哎",
), key=len, reverse=True))
# 剥掉口令后至少要剩这么多实词字，且实词占比不低于此比例，才算"有信息量的口播"
_FILLER_MIN_CONTENT = 4
_FILLER_MIN_RATIO = 0.5


def is_filler_speech(text: str) -> tuple:
    """这段原声是不是**拍摄现场的废话**（导演口令/口水话/催促声）。返回 (是不是, 判据)。

    "窗口里有真实口播"不能只看字数：实测成片末两镜保下来的原声是
    「OK然后捏一捏那个泡沫」和「行行行行行行行行行行往上走这个都够了够了够了」——
    有真人在说、字数也远超口型护栏的 8 字门槛，但放进成片没有任何信息量。
    这类段应当换成克隆配音（或换一个没人说话的片段），而不是"保留用户原声"。

    两条判据（都命中现场废话的典型形态）：
      1) 同一个字/词连着念 3 次以上（"行行行行"、"够了够了够了"）= 现场催促；
      2) 剥掉口令/语气词后剩下的实词太少（<4 字或不到原文一半）= 只是现场指挥。
    """
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "")).lower()
    if not s:
        return True, "窗口内没有原声"
    run = best = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 3:
        return True, "同一个字连着念了 {} 次（现场催促声）".format(best)
    bigrams = [s[i:i + 2] for i in range(len(s) - 1)]
    for bg in set(bigrams):
        if bigrams.count(bg) >= 3:
            return True, "「{}」重复了 {} 次（现场催促声）".format(bg, bigrams.count(bg))
    body = s
    for w in _FILLER_TOKENS:
        body = body.replace(w, "")
    if len(body) < _FILLER_MIN_CONTENT or len(body) < _FILLER_MIN_RATIO * len(s):
        return True, "剥掉现场口令/语气词只剩「{}」（{}/{} 字）".format(body, len(body), len(s))
    return False, ""


# 离机位的现场口令/环境人声：字面上像"真实口播"（文本规则兜不住，如「各位姐妹儿直接闭眼冲啊」），
# 但录进来的电平极低（实测峰值 -29dB，相邻镜头人声峰值 ~0dB），放进成片人耳听不见——
# ASR 却能识别出文字，于是"有字幕没声音"。峰值与均值**都**低于阈值才判定，避免误伤录得偏轻但可用的口播。
QUIET_VOICE_MAX_DB = float(os.getenv("AGENT_QUIET_VOICE_MAX_DB", "-18"))
QUIET_VOICE_MEAN_DB = float(os.getenv("AGENT_QUIET_VOICE_MEAN_DB", "-30"))


def window_audio_level(source_path: str, time_range: str):
    """测某素材取窗内的音频电平（volumedetect），返回 (mean_db, max_db)；测不出返回 (None, None)。"""
    a, b = _parse_range(time_range)
    src = source_path or ""
    if src and not os.path.isabs(src):
        cand = os.path.join(_AGENT_ROOT, src)
        src = cand if os.path.isfile(cand) else src
    if not (src and os.path.isfile(src)):
        return None, None
    cmd = [_FFMPEG_BIN, "-hide_banner", "-ss", "{:.3f}".format(max(0.0, a))]
    if b > a:
        cmd += ["-t", "{:.3f}".format(max(0.3, b - a))]
    cmd += ["-i", src, "-vn", "-af", "volumedetect", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return None, None
    text = r.stderr.decode("utf-8", "ignore")
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", text)
    mx = re.search(r"max_volume:\s*(-?[\d.]+) dB", text)
    return (float(mean.group(1)) if mean else None,
            float(mx.group(1)) if mx else None)


def is_offmic_quiet(source_path: str, time_range: str) -> tuple:
    """取窗内的说话声是否小到成片里听不见（离机位口令/环境人声）。返回 (是不是, 判据)。"""
    mean, mx = window_audio_level(source_path, time_range)
    if mean is None or mx is None:
        return False, ""
    if mx <= QUIET_VOICE_MAX_DB and mean <= QUIET_VOICE_MEAN_DB:
        return True, ("取窗内原声电平极低（峰值 {:.1f}dB / 均值 {:.1f}dB），"
                      "是离机位口令/环境人声，成片里听不见").format(mx, mean)
    return False, ""


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def ranges_overlap(path_a: str, range_a: str, path_b: str, range_b: str) -> bool:
    """同一源文件且时间区间相交 → 视为重叠（同一段素材被用到两处）。"""
    if not path_a or path_a != path_b:
        return False
    a0, a1 = _parse_range(range_a)
    b0, b1 = _parse_range(range_b)
    if a1 <= a0 or b1 <= b0:
        return a0 == b0  # 退化：无有效时长时按起点判等
    return max(a0, b0) < min(a1, b1)


class EditToolbox:
    """封装全池召回 + 按需 VLM 验证，供剪辑 Agent 的 ReAct 循环调用。"""

    def __init__(self, task_id: str, segments: list):
        self.retriever = Retriever(task_id)
        # 幂等入库：已建过的 task 会 skip，不重复 embed
        self.retriever.index_segments([s for s in (segments or []) if isinstance(s, dict)])
        self.by_gid = {s.get("global_asset_id"): s for s in (segments or [])
                       if isinstance(s, dict) and s.get("global_asset_id")}
        self.vlm = VLMTool()

    def pool_size(self) -> int:
        return len(self.by_gid)

    def retrieve(self, query: str, top_k: int = 6, exclude_gids=None) -> list:
        """按 query 从全池召回候选；exclude_gids 里的（已被别的 slot 占用）不返回。"""
        exclude = set(exclude_gids or [])
        res = self.retriever.search(query, top_k=max(top_k * 2, top_k + len(exclude) + 2))
        out = []
        for m in res.get("matches", []):
            gid = m.get("id")
            if gid in exclude:
                continue
            meta = m.get("meta", {})
            item = {
                "global_asset_id": gid,
                "source_path": meta.get("source_path", ""),
                "source_time_range": meta.get("source_time_range", ""),
                "duration": meta.get("duration", 0.0),
                "summary": meta.get("one_sentence_summary", ""),
                "visual_description": meta.get("visual_description", ""),
                "speech": meta.get("speech_or_text", ""),
                "keywords": (meta.get("keywords", []) or [])[:6],
                "quality": meta.get("quality_score", 0.0),
                "rrf_score": m.get("rrf_score", 0.0),
            }
            # whq_clone 链路：素材池带「该片段窗口内是否自带用户真声」的标注（见
            # whq_clone/strategy_out.build_context），召回时透出去让 Agent 优先保留原声。
            # 现场废话（拍摄口令/口水话）不算「自带口播」：标 true 的话 Agent 会去
            # place_original，再被拦回来白跑几轮，成片也可能留下一段没信息量的现场录音。
            ws = (self.by_gid.get(gid) or {}).get("whq_speech")
            if isinstance(ws, dict):
                filler, why = is_filler_speech(ws.get("text", ""))
                item["has_original_voice"] = bool(ws.get("has_speech")) and not filler
                item["speech"] = item["speech"] or ws.get("text", "")
                if filler and ws.get("has_speech"):
                    item["original_voice_note"] = "原声是拍摄现场的废话（{}），不要保留".format(why)
            out.append(item)
            if len(out) >= top_k:
                break
        return {"query": query, "candidates": out, "backend": res.get("backend", ""),
                "pool_size": res.get("store_size", 0)}

    async def verify(self, gid: str, question: str) -> dict:
        """按需 VLM 验证某候选片段画面是否符合期望，返回 {observation, error}。"""
        seg = self.by_gid.get(gid)
        if not seg:
            return {"observation": "", "error": f"未知 global_asset_id：{gid}"}
        q = (question or "").strip() or "请如实描述该片段画面里的主体、动作、景别与产品。"
        try:
            r = await self.vlm.inspect(q, targets=[{
                "source_path": seg.get("source_path", ""),
                "source_time_range": seg.get("source_time_range", ""),
                "asset_id": gid,
            }])
        except Exception as exc:  # noqa: BLE001
            _log.warning("verify failed for %s: %s", gid, exc)
            return {"observation": "", "error": str(exc)[:200]}
        return {"observation": r.get("observation", ""), "error": r.get("error", "")}
