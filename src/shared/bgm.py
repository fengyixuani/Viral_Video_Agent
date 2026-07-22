"""参考视频 BGM / 音频理解（真实实现）。

流程：ffmpeg 从参考视频单独抽出音频（mp3）→ base64 → 调 Gemini
（oneapi-comate 网关 ``gemini-3.1-pro-preview``，支持音频理解，见 API_USAGE.md 第 7 节）
→ 解析成结构化 BGM 信息（是否有 BGM/曲风/情绪/BPM/能量/乐器/高潮区间/卡点建议）。

结果按视频 (abspath+mtime+size) 缓存到 ``bgm`` 命名空间，避免重复调用。
Gemini 是强推理模型，``max_tokens`` 需给大（否则会被 length 截断返回空）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile

import requests

import cache
import obs

_log = obs.get_logger("bgm")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATEWAY = os.getenv("ONEAPI_BASE_URL", "https://oneapi-comate.baidu-int.com").rstrip("/")
TOKEN = os.getenv("ONEAPI_TOKEN", "sk-c0JdPzze1N7mvVxm85B6705a1b144a3394C895641f09F868")
MODEL = os.getenv("GEMINI_AUDIO_MODEL", "gemini-3.1-pro-preview")
MAX_SEC = int(os.getenv("BGM_ANALYZE_MAX_SEC", "120"))
# gemini-3.1-pro-preview 是强推理模型，会先生成一大段 reasoning 再输出 JSON，reasoning 也占
# 输出预算——预算给小了会在"思考"阶段就被 finish_reason=length 截断，拿不到 JSON。给足 12000。
MAX_TOKENS = int(os.getenv("BGM_ANALYZE_MAX_TOKENS", "12000"))
TIMEOUT = int(os.getenv("BGM_ANALYZE_TIMEOUT", "150"))

_PROMPT = (
    "你是短视频配乐分析师。下面给你的是一条爆款短视频**单独抽出来的音频**。"
    "请判断它的背景音乐（BGM）特征。注意区分「纯人声口播/没有配乐」与「有背景音乐」。\n"
    "严格只输出 JSON（不要 markdown 代码块、不要多余文字）：\n"
    "{\n"
    '  "has_bgm": true/false,            // 是否存在背景音乐\n'
    '  "is_speech_only": true/false,     // 是否几乎只有人声口播\n'
    '  "has_speech": true/false,         // 是否有人在说话/口播旁白（纯唱歌、纯器乐不算说话）\n'
    '  "vocal": "纯器乐/含人声演唱/口播为主/口播+背景乐",\n'
    '  "style": "曲风（如 电子/流行/抒情/国风/放克/Lo-fi 等；无则填 无）",\n'
    '  "mood": "情绪（如 紧张/欢快/治愈/high 等；无则填 无）",\n'
    '  "bpm": 整数或 null,               // 估计的节奏 BPM，无法判断填 null\n'
    '  "energy": "low/medium/high",\n'
    '  "instruments": ["主要乐器/音色，最多4个"],\n'
    '  "hot_section": {"start": 秒(数字), "end": 秒(数字)} 或 null,  // 音乐情绪最强/最适合卡点的区间\n'
    '  "sync_suggestion": "一句话卡点/剪辑建议（结合节奏与高潮区间）",\n'
    '  "summary": "一句话总结这段配乐"\n'
    "}"
)


def _resolve(uri: str) -> str:
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for candidate in (uri, os.path.join(PROJECT_ROOT, uri)):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _digest(path: str) -> dict:
    try:
        stat = os.stat(path)
        return {"path": path, "mtime": int(stat.st_mtime), "size": stat.st_size, "model": MODEL, "v": 1}
    except OSError:
        return {"path": path, "model": MODEL, "v": 1}


def _extract_audio(local: str) -> str:
    """抽单声道 16k mp3（≤MAX_SEC 秒）到临时文件，返回路径；失败返回空串。"""
    fd, dst = tempfile.mkstemp(prefix="bgm_", suffix=".mp3")
    os.close(fd)
    try:
        subprocess.run(
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-t", str(MAX_SEC),
             "-i", local, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90,
        )
        if os.path.getsize(dst) > 0:
            return dst
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("bgm audio extract failed: %s", exc)
    try:
        os.remove(dst)
    except OSError:
        pass
    return ""


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (ValueError, TypeError):
                return {}
        return {}


def _ensure_flags(result: dict) -> dict:
    """给结果补齐 has_speech / pure_music（兼容加这两个字段之前的旧缓存）。"""
    if not isinstance(result, dict) or not result.get("available"):
        return result
    has_bgm = bool(result.get("has_bgm"))
    if "has_speech" not in result:
        vocal = result.get("vocal", "") or ""
        result["has_speech"] = bool(result.get("is_speech_only")) or ("口播" in vocal)
    if "pure_music" not in result:
        result["pure_music"] = has_bgm and not result.get("has_speech")
    return result


def analyze_bgm(video_uri: str, *, use_cache: bool = True) -> dict:
    """分析参考视频的 BGM/音频，返回结构化结果 ``{available, has_bgm, style, ...}``。

    失败/无音频时返回 ``{"available": False, "reason": ...}``，调用方据此降级展示。
    """
    local = _resolve(video_uri)
    if not local:
        return {"available": False, "reason": "参考视频不是本地文件，无法抽取音频"}

    digest = _digest(local)
    if use_cache:
        cached = cache.get("bgm", digest)
        if cached and isinstance(cached.get("payload"), dict):
            return _ensure_flags(cached["payload"])

    mp3 = _extract_audio(local)
    if not mp3:
        return {"available": False, "reason": "音频抽取失败"}
    try:
        b64 = base64.b64encode(open(mp3, "rb").read()).decode()
    finally:
        try:
            os.remove(mp3)
        except OSError:
            pass

    try:
        resp = requests.post(
            f"{GATEWAY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
                ]}],
                "max_tokens": MAX_TOKENS,
            },
            timeout=TIMEOUT,
        )
    except (requests.RequestException, OSError) as exc:
        _log.warning("bgm gemini request failed: %s", exc)
        return {"available": False, "reason": f"Gemini 请求失败：{exc!s}"}

    if resp.status_code != 200:
        _log.warning("bgm gemini HTTP %s: %s", resp.status_code, resp.text[:200])
        return {"available": False, "reason": f"Gemini HTTP {resp.status_code}"}

    body = resp.json()
    choice = (body.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        return {"available": False, "reason": "Gemini 输出被截断（max_tokens 不足）"}
    content = (choice.get("message") or {}).get("content", "")
    parsed = _parse_json(content)
    if not parsed:
        return {"available": False, "reason": "Gemini 返回无法解析为 JSON"}

    has_bgm = bool(parsed.get("has_bgm", False))
    # has_speech：优先用模型显式判断；缺失则从 is_speech_only / vocal 推断
    vocal = parsed.get("vocal", "") or ""
    if "has_speech" in parsed:
        has_speech = bool(parsed.get("has_speech"))
    else:
        has_speech = bool(parsed.get("is_speech_only")) or ("口播" in vocal)
    result = {
        "available": True,
        "has_bgm": has_bgm,
        "is_speech_only": bool(parsed.get("is_speech_only", False)),
        "has_speech": has_speech,
        # 纯音乐：有配乐但没有口播旁白（用于复刻时默认关 TTS、按节奏/时长对齐选片）
        "pure_music": has_bgm and not has_speech,
        "vocal": vocal,
        "style": parsed.get("style", ""),
        "mood": parsed.get("mood", ""),
        "bpm": parsed.get("bpm"),
        "energy": parsed.get("energy", ""),
        "instruments": parsed.get("instruments", []) or [],
        "hot_section": parsed.get("hot_section") if isinstance(parsed.get("hot_section"), dict) else None,
        "sync_suggestion": parsed.get("sync_suggestion", ""),
        "summary": parsed.get("summary", ""),
        "model": MODEL,
    }
    try:
        cache.set("bgm", digest, result)
    except OSError:
        pass
    return result
