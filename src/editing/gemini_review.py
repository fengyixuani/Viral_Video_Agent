"""审片 Agent 的 Gemini 视觉+听觉后端（oneapi-comate ``gemini-3.1-pro-preview``）。

qwen 视觉后端（as_core vision）只看画面、听不到声音，无法判断"烧录字幕与口播是否一致"。
Gemini 3.1 Pro 能同时理解**画面帧 + 音轨**，因此用它审片时可以抓出「字幕和原声两张皮」
这类问题。

网关只认 ``input_audio``/``image_url`` 这类 content block；实测把视频以
``image_url`` + ``data:video/mp4;base64,...`` 的 mime 方式传入，网关会正确路由为
视频输入并同时解码画面与音频（见 API_USAGE.md 第 7 节的 mime 路由技巧）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile

import requests

import obs

_log = obs.get_logger("agent_edit_gemini_review")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

GATEWAY = os.getenv("ONEAPI_BASE_URL", "https://oneapi-comate.baidu-int.com").rstrip("/")
TOKEN = os.getenv("ONEAPI_TOKEN", "sk-c0JdPzze1N7mvVxm85B6705a1b144a3394C895641f09F868")
MODEL = os.getenv("GEMINI_REVIEW_MODEL", "gemini-3.1-pro-preview")
# 强推理模型，reasoning 也吃输出预算，给足以免 JSON 被 length 截断
MAX_TOKENS = int(os.getenv("GEMINI_REVIEW_MAX_TOKENS", "12000"))
TIMEOUT = int(os.getenv("GEMINI_REVIEW_TIMEOUT", "240"))
# 送审视频编码参数：压小体积同时保住字幕可读性
MAX_SEC = int(os.getenv("GEMINI_REVIEW_MAX_SEC", "120"))
SCALE_W = int(os.getenv("GEMINI_REVIEW_SCALE_W", "480"))
FPS = int(os.getenv("GEMINI_REVIEW_FPS", "8"))


def _encode_video(path: str) -> str:
    """把成片压成竖屏小 mp4（保留音轨）并 base64；失败返回空串。"""
    fd, dst = tempfile.mkstemp(prefix="review_", suffix=".mp4")
    os.close(fd)
    try:
        subprocess.run(
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-t", str(MAX_SEC), "-i", path,
             "-vf", f"scale={SCALE_W}:-2,fps={FPS}", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
             "-c:a", "aac", "-ar", "16000", "-ac", "1", "-b:a", "48k", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180,
        )
        if os.path.getsize(dst) > 0:
            return base64.b64encode(open(dst, "rb").read()).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("gemini review video encode failed: %s", exc)
    finally:
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


def review_video(system_prompt: str, review_user: str, video_path: str) -> dict:
    """用 Gemini 审一条成片：同时看画面 + 听声音，返回审片 JSON（失败返回带 _error 的 dict）。"""
    if not video_path or not os.path.isfile(video_path):
        return {"_error": f"成片文件不存在：{video_path}"}
    b64 = _encode_video(video_path)
    if not b64:
        return {"_error": "成片编码失败，无法送 Gemini 审片"}

    # Gemini 单轮 content：把 system 规则并进 user 文本（网关无独立 system 语义时也稳妥）
    text = system_prompt + "\n\n[本次审片输入]\n" + review_user + \
        "\n\n请同时结合画面与声音审阅，尤其检查烧录字幕是否与人物口播/画面一致。严格只输出上面约定的 JSON。"
    try:
        resp = requests.post(
            f"{GATEWAY}/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{b64}"}},
                ]}],
                "max_tokens": MAX_TOKENS,
            },
            timeout=TIMEOUT,
        )
    except (requests.RequestException, OSError) as exc:
        _log.warning("gemini review request failed: %s", exc)
        return {"_error": f"Gemini 审片请求失败：{exc!s}"}

    if resp.status_code != 200:
        _log.warning("gemini review HTTP %s: %s", resp.status_code, resp.text[:200])
        return {"_error": f"Gemini 审片 HTTP {resp.status_code}"}
    body = resp.json()
    choice = (body.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        return {"_error": "Gemini 审片输出被截断（max_tokens 不足）"}
    content = (choice.get("message") or {}).get("content", "")
    parsed = _parse_json(content)
    if not parsed:
        return {"_error": "Gemini 审片返回无法解析为 JSON"}
    return parsed
