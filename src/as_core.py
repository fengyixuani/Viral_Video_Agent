"""AgentScope 2.0.4 backed text/vision model adapter.

Routing:

- Vision-grounded tasks (reference video/image understanding) default to
  ``ali-qwen3.7-plus`` and are sent directly to the wenchain gateway with a
  native ``video_url`` block; large sources are auto-downscaled *in resolution
  only* (frame rate and audio are preserved).
- Text-only tasks (planning, per-shot decision, trend mining) default to
  ``ali-qwen3.7-max`` and go through ``agentscope.model.OpenAIChatModel`` with
  ``stream=True`` so reasoning is streamed to the SSE UI.

When no credential is configured, deterministic mock payloads keep the
workbench runnable end-to-end.
"""
import asyncio
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import time

import obs

_log = obs.get_logger("as_core")

try:
    import imageio_ffmpeg  # type: ignore

    _DEFAULT_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _DEFAULT_FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

try:  # 视觉调用用 requests 直连 wenchain
    import requests
except ImportError:  # pragma: no cover
    requests = None

VISION_MODEL = os.getenv("VISION_LLM_MODEL", "ali-qwen3.7-plus")
TEXT_MODEL = os.getenv("TEXT_LLM_MODEL", os.getenv("LLM_MODEL", "ali-qwen3.7-max"))
# 默认用较大的输出上限，避免长视频（多分镜 + 多维度 + 多方案）的结构化 JSON 被截断
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

# 视频 base64 上限 (MB)。wenchain 网关单个 data URI 上限 20MB，因此这里默认
# 18 MB 留 10% 余量；超过则按 720p/540p/360p 逐级降分辨率，帧率与音频保持不变。
VISION_MAX_MEDIA_MB = float(os.getenv("VISION_MAX_MEDIA_MB", "18"))
_DOWNSCALE_WIDTHS = [1080, 960, 720, 540, 480, 360]

WENCHAIN_BASE_URL = os.getenv("WENCHAIN_BASE_URL", "http://wenku-openai.baidu-int.com/wenchain/strategy")
WENCHAIN_API_KEY = os.getenv("WENCHAIN_API_KEY", os.getenv("QIANFAN_API_KEY", "wangpantob_all_video_copy"))
USE_WENCHAIN = os.getenv("USE_WENCHAIN_OPENAI", "1") not in ("0", "false", "False")
ALLOW_MOCK = os.getenv("ALLOW_MOCK_LLM", "0") not in ("0", "false", "False")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MODEL_CACHE = {}


def pick_model(vision: bool = False) -> str:
    return VISION_MODEL if vision else TEXT_MODEL


def parse_json(txt: str) -> dict:
    txt = (txt or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1]
        if txt.startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
    index = txt.find("{")
    if index < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(txt[index:])
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object")
    return obj


def _build_media_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "video/mp4"
    with open(path, "rb") as stream:
        encoded = base64.b64encode(stream.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_media_path(uri: str) -> str:
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for candidate in (uri, os.path.join(PROJECT_ROOT, uri)):
        if os.path.isfile(candidate):
            return candidate
    return ""


def _resolve_media_url(uri: str) -> str:
    if not uri:
        return ""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    path = _resolve_media_path(uri)
    if path:
        return _build_media_data_url(path)
    return uri


def _downscale_video(src: str, width: int) -> str:
    """把视频按目标宽度等比缩放输出到临时 mp4，帧率与音频保持不变。"""
    fd, dst = tempfile.mkstemp(prefix="vision_", suffix=".mp4")
    os.close(fd)
    cmd = [
        _DEFAULT_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", dst,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return dst


def _probe_duration(path: str) -> float:
    import re
    result = subprocess.run(
        [_DEFAULT_FFMPEG, "-hide_banner", "-i", path, "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stdout)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


def _transcode_to_budget(src: str, max_mb: float, width: int = 720) -> str:
    """按目标大小一次性转码：根据时长算目标码率，缩放到 width，只编码一次。

    比逐级重编码快得多，尤其适合长视频。
    """
    fd, dst = tempfile.mkstemp(prefix="vision_", suffix=".mp4")
    os.close(fd)
    duration = _probe_duration(src) or 1.0
    # base64 会膨胀 ~33%，原始字节预算取 max_mb 的 72% 再留 5% 安全余量
    target_bytes = max_mb * 1024 * 1024 * 0.72 * 0.95
    audio_bps = 96_000
    total_bps = max(300_000, target_bytes * 8 / duration)
    video_bps = int(max(200_000, total_bps - audio_bps))
    bufsize = video_bps * 2
    cmd = [
        _DEFAULT_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", str(video_bps), "-maxrate", str(video_bps), "-bufsize", str(int(bufsize)),
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", dst,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return dst


def _encode_video_for_vision(path: str, max_mb: float):
    """返回 (base64_str, info)。比较 base64 后大小（wenchain 限单个 data URI 字节数）。

    源文件已在预算内则直接编码；否则用目标码率单次转码（720p），仍超限再降到
    540p / 360p 各转一次，避免逐级重复编码整段长视频。
    """
    def _b64(src: str) -> tuple[str, int]:
        with open(src, "rb") as stream:
            encoded = base64.b64encode(stream.read()).decode("ascii")
        return encoded, len(encoded)

    limit = int(max_mb * 1024 * 1024)
    encoded, size = _b64(path)
    if size <= limit:
        return encoded, {"downscaled": False, "width": None,
                          "base64_mb": round(size / 1024 / 1024, 2)}
    tmp_files: list[str] = []
    try:
        for width in (720, 540, 360):
            dst = _transcode_to_budget(path, max_mb, width=width)
            tmp_files.append(dst)
            enc, sz = _b64(dst)
            if sz <= limit:
                return enc, {"downscaled": True, "width": width,
                              "base64_mb": round(sz / 1024 / 1024, 2)}
        enc, sz = _b64(tmp_files[-1])
        return enc, {"downscaled": True, "width": 360,
                      "base64_mb": round(sz / 1024 / 1024, 2), "over_budget": True}
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


def _encode_vision_blocks(user: str, media):
    """编码媒体为 OpenAI 多模态 content blocks，返回 (blocks, info)。"""
    content_blocks: list = [{"type": "text", "text": user}]
    info_summary: list[dict] = []
    for item in media or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("url", "")
        if not raw:
            continue
        if raw.startswith(("http://", "https://", "data:")):
            url = raw
            info_summary.append({"kind": item.get("type"), "external": True})
        else:
            path = _resolve_media_path(raw)
            if not path:
                continue
            mime_type = mimetypes.guess_type(path)[0] or "video/mp4"
            if item.get("type") == "video":
                encoded, info = _encode_video_for_vision(path, VISION_MAX_MEDIA_MB)
                info_summary.append({"kind": "video", **info})
                url = f"data:{mime_type};base64,{encoded}"
            else:
                with open(path, "rb") as stream:
                    encoded = base64.b64encode(stream.read()).decode("ascii")
                info_summary.append({"kind": "image", "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2)})
                url = f"data:{mime_type};base64,{encoded}"
        if item.get("type") == "video":
            content_blocks.append({"type": "video_url", "video_url": {"url": url}})
        else:
            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
    return content_blocks, info_summary


def _mentions_json(system: str, content_blocks) -> bool:
    """wenchain 要求：启用 response_format=json_object 时 messages 必须含 'json' 字样。
    仅当调用方明确要 JSON（system 或文本块出现 json）才启用，否则自由文本，避免 400。"""
    if "json" in (system or "").lower():
        return True
    if isinstance(content_blocks, str):
        return "json" in content_blocks.lower()
    for block in content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            if "json" in str(block.get("text", "")).lower():
                return True
    return False


def _post_wenchain_vision(system: str, content_blocks, model_name: str) -> str:
    if requests is None:
        raise RuntimeError("vision path requires the `requests` package")
    if not WENCHAIN_API_KEY:
        raise RuntimeError("WENCHAIN_API_KEY not configured")
    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content_blocks},
        ],
    }
    if _mentions_json(system, content_blocks):
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        WENCHAIN_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {WENCHAIN_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"wenchain vision HTTP {response.status_code}: {response.text[:400]}")
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _get_model(model_name: str):
    """Lazily build an OpenAIChatModel bound to the wenchain gateway."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    from agentscope.model import OpenAIChatModel
    from agentscope.credential import OpenAICredential
    from pydantic import SecretStr

    credential = OpenAICredential(
        api_key=SecretStr(WENCHAIN_API_KEY or "unset"),
        base_url=WENCHAIN_BASE_URL,
    )
    model = OpenAIChatModel(
        credential=credential,
        model=model_name,
        stream=True,
        parameters=OpenAIChatModel.Parameters(max_tokens=MAX_TOKENS),
        client_kwargs={"base_url": WENCHAIN_BASE_URL},
    )
    _MODEL_CACHE[model_name] = model
    return model


def _build_messages(system: str, user: str, media):
    """Build AgentScope ``Msg`` objects, embedding video/image blocks if given."""
    from agentscope.message import Msg, TextBlock

    user_blocks = [TextBlock(type="text", text=user)]
    for item in media or []:
        if not isinstance(item, dict):
            continue
        url = _resolve_media_url(item.get("url", ""))
        if not url:
            continue
        kind = item.get("type") or "image"
        # OpenAI-compatible video_url / image_url block, expressed as an
        # extended TextBlock-like dict so the underlying formatter can pass it
        # through unchanged. The wenchain gateway accepts the same schema as
        # OpenAI multimodal.
        if kind == "video":
            user_blocks.append({"type": "video_url", "video_url": {"url": url}})
        else:
            user_blocks.append({"type": "image_url", "image_url": {"url": url}})
    return [
        Msg(name="system", role="system", content=[TextBlock(type="text", text=system)]),
        Msg(name="user", role="user", content=user_blocks),
    ]


def _extract_delta(chunk):
    """Return ``(thinking_delta, text_delta, is_last, acc_text)`` for a chunk."""
    thinking = ""
    text = ""
    for block in chunk.content or []:
        # Blocks are TypedDict-like objects with a "type" key.
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
        block_text = (block.get("text") if isinstance(block, dict) else getattr(block, "text", "")) or ""
        block_thinking = (block.get("thinking") if isinstance(block, dict) else getattr(block, "thinking", "")) or ""
        if block_type == "thinking":
            thinking += block_thinking
        elif block_type == "text":
            text += block_text
    return thinking, text, bool(getattr(chunk, "is_last", False))


async def _model_stream(model_name: str, system: str, user: str, media):
    """Iterate AgentScope streaming chunks, yielding reasoning/content events.

    AgentScope 2.0.4 emits incremental ``TextBlock`` chunks followed by a
    terminal chunk with ``is_last=True`` that carries the fully accumulated
    text. We treat the deltas as reasoning (using ``thinking`` when the model
    exposes it, otherwise reusing the text delta) and emit the final content
    from the accumulated block.
    """
    model = _get_model(model_name)
    messages = _build_messages(system, user, media)
    response = await model(messages)
    accumulated = ""
    async for chunk in response:
        thinking, text, is_last = _extract_delta(chunk)
        if is_last:
            # Prefer the accumulated text from the terminal chunk when present.
            if text:
                accumulated = text
            yield {"content": accumulated}
            return
        if thinking:
            yield {"reasoning": thinking}
        if text:
            accumulated += text
            yield {"reasoning": text}
    if accumulated:
        yield {"content": accumulated}


def _is_media_rejected(exc: BaseException) -> bool:
    body = str(exc).lower()
    tokens = ("unexpected item type in content", "video_url", "image_url", "unsupported")
    return any(tok in body for tok in tokens)


def _mock_payload(system: str, user: str):
    if '"decisions"' in system:
        try:
            brief = parse_json(user).get("shot_slots", [])
        except ValueError:
            brief = []
        return {"decisions": [{"slot_id": s.get("slot_id", s.get("id", i + 1)), "action": "generate", "reason": "离线模式无素材画像，生成补足"} for i, s in enumerate(brief)]}
    if '"goal"' in system and '"steps"' in system:
        return {"goal": "保留爆款结构并替换为用户内容", "granularity": "action_scene", "reasoning": "先解析结构，再逐镜生成或匹配，最后完成剪辑包装。该计划兼顾用户选择的维度和趋势。", "steps": [{"tool": "解析", "purpose": "确认模板与素材"}, {"tool": "生成", "purpose": "补齐缺失镜头"}, {"tool": "剪辑", "purpose": "按节奏组装"}, {"tool": "包装", "purpose": "添加字幕、声音和导出"}]}
    if '"trends"' in system:
        phrases = ["前3秒直接抛痛点", "结果前置制造反差", "高密度字幕强化信息", "真实体验建立信任", "评论区问题作开场", "节奏卡点快速切镜", "数字化利益点", "结尾明确行动指令"]
        sources = ["平台趋势", "爆款库沉淀", "网络热点"]
        return {"trends": [{"keyword": p[:6], "phrase": p, "source": sources[i % 3], "reason": "提升注意力、完播或转化效率"} for i, p in enumerate(phrases)]}
    dims = [
        {"id": "coarse-structure", "name": "叙事结构", "level": "coarse", "desc": "痛点到转化的结构", "recommended": True, "replace": {"enabled": False, "types": [], "hint": "系统自动处理"}},
        {"id": "fine-hook", "name": "Hook形式", "level": "fine", "desc": "开头痛点钩子", "recommended": True, "replace": {"enabled": True, "types": ["text", "video"], "hint": "补充痛点文案或视频"}},
        {"id": "fine-product", "name": "商品展示", "level": "fine", "desc": "主体多角度展示", "recommended": True, "replace": {"enabled": True, "types": ["image", "video"], "hint": "补充主体素材"}},
        {"id": "fine-cta", "name": "CTA形式", "level": "fine", "desc": "明确行动指令", "recommended": True, "replace": {"enabled": True, "types": ["text"], "hint": "补充行动文案"}},
    ]
    shots = [
        {"id": 1, "want": "前3秒痛点钩子", "duration": 3.0, "role": "hook", "breakdown": [{"dim": "Hook形式", "value": "直接抛用户痛点"}, {"dim": "景别", "value": "主体特写"}, {"dim": "字幕贴片", "value": "高对比大字"}, {"dim": "BGM节奏", "value": "快速起势"}, {"dim": "叙事结构", "value": "问题前置"}]},
        {"id": 2, "want": "展示核心主体与价值", "duration": 4.0, "role": "product_intro", "breakdown": [{"dim": "商品展示", "value": "多角度快速展示"}, {"dim": "镜头运动", "value": "推进与环绕"}, {"dim": "卖点顺序", "value": "核心利益优先"}, {"dim": "字幕贴片", "value": "数字利益点"}, {"dim": "叙事结构", "value": "给出解决方案"}]},
        {"id": 3, "want": "使用演示和效果证明", "duration": 5.0, "role": "proof", "breakdown": [{"dim": "商品展示", "value": "真实使用过程"}, {"dim": "证据形式", "value": "前后效果对比"}, {"dim": "景别", "value": "中近景切换"}, {"dim": "BGM节奏", "value": "卡点切镜"}, {"dim": "叙事结构", "value": "建立信任"}]},
        {"id": 4, "want": "总结利益点并行动引导", "duration": 3.0, "role": "cta", "breakdown": [{"dim": "CTA形式", "value": "口播与贴片同步"}, {"dim": "商品展示", "value": "定格主体"}, {"dim": "字幕贴片", "value": "行动指令"}, {"dim": "BGM节奏", "value": "收束落点"}, {"dim": "叙事结构", "value": "明确转化"}]},
    ]
    return {"industry_guess": "ecom", "industry_reason": "参考内容呈现钩子、主体展示、效果证明和转化闭环", "total_duration_sec": 15.0, "hook": {"type": "pain", "duration_sec": 3}, "narrative_structure": ["痛点", "主体", "证明", "CTA"], "rhythm": {"avg_shot_sec": 3.75, "cut_density": "fast"}, "selling_points_order": ["核心价值", "效果证明"], "cta": {"type": "direct", "text": "立即行动"}, "packaging": {"subtitles": True, "bgm": "upbeat"}, "shot_slots": shots, "schemes": [{"id": "faithful", "name": "忠于素材方案", "strategy": "faithful", "desc": "优先使用已有素材复刻结构", "dimensions": dims}, {"id": "balanced", "name": "平衡增强方案", "strategy": "balanced", "desc": "素材与生成能力平衡", "dimensions": dims}, {"id": "regenerate", "name": "创意重构方案", "strategy": "regenerate", "desc": "保留爆点逻辑并重做视觉", "dimensions": dims}]}


async def stream(system: str, user: str, *, vision: bool = False, media=None):
    """Yield ``{"reasoning": str}`` chunks then a final ``{"content": str}``.

    Text-only requests go through AgentScope streaming to preserve reasoning.
    Vision requests bypass AgentScope and hit the wenchain gateway directly
    with a native ``video_url`` payload — AgentScope 2.0.4's formatter drops
    non-TextBlock content, so a direct call is currently the only way to
    actually feed pixels to the VLM.
    """
    model_name = pick_model(vision)
    has_media = bool([m for m in (media or []) if isinstance(m, dict) and m.get("url")])
    if USE_WENCHAIN and WENCHAIN_API_KEY:
        yield {"reasoning": f"连接远端模型 {model_name}。"}
        if vision and has_media:
            try:
                yield {"reasoning": "视觉直连模式：正在编码视频（大文件会自动降分辨率）…"}
                t0 = time.time()
                blocks, info = await asyncio.to_thread(_encode_vision_blocks, user, media)
                _log.info("vision encode done %.1fs info=%s", time.time() - t0, info)
                if info:
                    yield {"reasoning": f"编码完成（{round(time.time() - t0, 1)}s）：{json.dumps(info, ensure_ascii=False)}"}
                yield {"reasoning": f"已发送 video_url，等待 {model_name} 返回（长视频可能较慢）…"}
                t1 = time.time()
                content = await asyncio.to_thread(_post_wenchain_vision, system, blocks, model_name)
                _log.info("vision model=%s returned %.1fs content_len=%d", model_name, time.time() - t1, len(content or ""))
                yield {"reasoning": f"模型返回（{round(time.time() - t1, 1)}s），整理 JSON。"}
                yield {"content": content}
                return
            except Exception as exc:
                if _is_media_rejected(exc):
                    _log.warning("vision media rejected, fallback to text: %s", exc)
                    yield {"reasoning": f"网关拒绝多模态输入（{exc}），回退纯文本。"}
                    async for item in _model_stream(model_name, system, user, None):
                        yield item
                    return
                _log.error("vision call failed: %s", exc)
                raise
        try:
            async for item in _model_stream(model_name, system, user, media):
                yield item
            return
        except Exception as exc:
            if media and _is_media_rejected(exc):
                yield {"reasoning": "网关拒绝多模态输入，回退纯文本再试一次。"}
                async for item in _model_stream(model_name, system, user, None):
                    yield item
                return
            raise
    if not ALLOW_MOCK:
        raise RuntimeError(
            "LLM 未配置：请设置 WENCHAIN_API_KEY 或 USE_WENCHAIN_OPENAI=1，"
            "或临时设置 ALLOW_MOCK_LLM=1 使用离线示例数据。"
        )
    for piece in ("正在识别任务约束。", f"离线模式，默认模型 {model_name}。", "正在提取结构与可复刻维度。"):
        yield {"reasoning": piece}
    yield {"content": json.dumps(_mock_payload(system, user), ensure_ascii=False)}


async def complete(system: str, user: str, *, vision: bool = False, media=None) -> str:
    content = ""
    async for item in stream(system, user, vision=vision, media=media):
        if "content" in item:
            content = item["content"]
    return content


async def complete_json(system: str, user: str, *, vision: bool = False, media=None) -> dict:
    return parse_json(await complete(system, user, vision=vision, media=media))
