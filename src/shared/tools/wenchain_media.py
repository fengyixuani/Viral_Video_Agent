"""wenchain 媒体网关客户端：文生图 / 图生图 / 文生视频 / 图生视频 + 下载 + ffmpeg 拼接。

纯业务层，不含任何 Agent / SSE 逻辑。所有生成能力都走同一个内网 wenchain 网关
(`/wenchain/strategy/incommonuserr`)，channel 即鉴权凭证（无独立 token）。

已实测：
- seedream 5.0 T2I / i2i：i2i 的 ``seedream_options.image`` 接受 base64 data URL，
  且支持多张参考图；输出为公网 ``bos_url``。
- seedance 2.0 T2V / i2v：i2v 首帧必须是公网 URL（base64 会超时），故首帧用
  seedream 输出的公网 bos_url。
"""
import base64
import mimetypes
import os
import subprocess
import tempfile
import time

import requests

# ---- 网关 / 鉴权 ----
MEDIA_URL = os.getenv(
    "WENCHAIN_MEDIA_URL",
    "http://wenku-openai.baidu-int.com/wenchain/strategy/incommonuserr",
)
CHANNEL = os.getenv("WENCHAIN_CHANNEL", os.getenv("WENCHAIN_API_KEY", "wangpantob_all_video_copy"))

# ---- 模型 ----
T2I_MODEL = os.getenv("T2I_MODEL", "doubao-seedream-5-0-260128")
VIDEO_MODEL = os.getenv("I2V_MODEL", "doubao-seedance-2-0")

# ---- 规格 ----
T2I_SIZE = os.getenv("T2I_SIZE", "2048x2048")
ASPECT_RATIO = os.getenv("ASPECT_RATIO", "9:16")
IMG_TIMEOUT = int(os.getenv("MEDIA_IMG_TIMEOUT", "180"))
VIDEO_TIMEOUT = int(os.getenv("MEDIA_VIDEO_TIMEOUT", "600"))

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")


class MediaError(RuntimeError):
    """网关返回非 0 或网络错误。"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _to_ref_url(ref: str) -> str:
    """参考图既可传公网 URL，也可传本地路径（本地→base64 data URL）。"""
    if not ref:
        return ""
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    if os.path.isfile(ref):
        return _data_url(ref)
    return ref


def _base_payload(model: str, prompt: str) -> dict:
    qid = _now_ms()
    msg = [{"content": prompt, "role": "user"}]
    return {
        "channel": CHANNEL,
        "chat_id": qid,
        "query_id": qid,
        "model": model,
        "stream": False,
        "messages": msg,
        "message_user": msg,
        "message_prompt": msg,
    }


def _post(payload: dict, timeout: int) -> dict:
    resp = requests.post(
        MEDIA_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise MediaError(f"wenchain HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    status = body.get("status", {}) or {}
    if status.get("code") != 0:
        raise MediaError(f"wenchain code={status.get('code')} msg={status.get('msg')}")
    return body


# ============================ 文生图 / 图生图 ============================

def gen_image(prompt: str, *, size: str = None, ref_images=None) -> str:
    """文生图 / 图生图，返回首图公网 URL。

    Args:
        prompt: 图像描述。
        size: 输出尺寸，如 ``2048x2048``（默认 :data:`T2I_SIZE`）。
        ref_images: 参考图列表（公网 URL 或本地路径），非空即走 i2i。
    """
    size = size or T2I_SIZE
    payload = _base_payload(T2I_MODEL, prompt)
    opts = {
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "url",
        "watermark": False,
    }
    refs = [_to_ref_url(r) for r in (ref_images or []) if r]
    refs = [r for r in refs if r]
    if refs:
        opts["image"] = refs
    payload["seedream_options"] = opts
    body = _post(payload, IMG_TIMEOUT)
    items = (body.get("data", {}) or {}).get("data", []) or []
    for it in items:
        url = it.get("bos_url") or it.get("url")
        if url:
            return url
    raise MediaError(f"gen_image: no image url in response: {str(body)[:300]}")


# ============================ 文生视频 / 图生视频 ============================

def _quantize_duration(sec: float) -> int:
    import math

    return max(4, min(15, int(math.ceil(sec))))


def gen_video_t2v(prompt: str, duration_sec: float, *, aspect_ratio: str = None) -> str:
    """纯文生视频（无首帧），返回视频公网 URL。"""
    aspect_ratio = aspect_ratio or ASPECT_RATIO
    dur = _quantize_duration(duration_sec)
    text = f"{prompt}  --ratio {aspect_ratio}  --dur {dur}"
    payload = _base_payload(VIDEO_MODEL, text)
    payload["seedancepro_options"] = {"content": [{"type": "text", "text": text}]}
    body = _post(payload, VIDEO_TIMEOUT)
    return _extract_video_url(body)


def gen_video_i2v(prompt: str, first_frame_url: str, duration_sec: float,
                  *, aspect_ratio: str = None) -> str:
    """图生视频（硬首帧），首帧必须是公网 URL。返回视频公网 URL。"""
    aspect_ratio = aspect_ratio or ASPECT_RATIO
    dur = _quantize_duration(duration_sec)
    text = f"{prompt}  --ratio {aspect_ratio}  --dur {dur}"
    payload = _base_payload(VIDEO_MODEL, text)
    payload["seedancepro_options"] = {
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"},
        ]
    }
    body = _post(payload, VIDEO_TIMEOUT)
    return _extract_video_url(body)


def _extract_video_url(body: dict) -> str:
    data = body.get("data", {}) or {}
    # seedance 返回结构：data.content.video_url
    content = data.get("content")
    if isinstance(content, dict) and content.get("video_url"):
        return content["video_url"]
    items = data.get("data", []) or []
    for it in items:
        url = it.get("video_url") or it.get("url") or it.get("bos_url")
        if url:
            return url
    if data.get("video_url"):
        return data["video_url"]
    # 失败时把网关内嵌的 error 透出来（如真人风控 40000002）
    err = data.get("error") or {}
    if err:
        raise MediaError(f"video failed: code={err.get('code')} msg={err.get('message') or err.get('msg')}")
    raise MediaError(f"video: no url in response: {str(body)[:300]}")


# ============================ 下载 / 拼接 ============================

def download(url: str, dst: str, *, timeout: int = 300) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)
    return dst


def _has_audio(path: str) -> bool:
    """探测视频是否含音频流（缺失时 concat 会音画错位，需补静音）。"""
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", path, "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).stdout
        return "Audio:" in out
    except Exception:
        return False


def _normalize_clip(src: str, dst: str, *, fps: int = 24,
                    width: int = 720, height: int = 1280) -> str:
    """统一分辨率/帧率/编码，保证 concat 无缝；无音频则补静音轨（保留人声若有）。"""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}")
    if _has_audio(src):
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", dst,
        ]
    else:
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", src, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-shortest", "-movflags", "+faststart", dst,
        ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return dst


def concat_videos(clips, dst: str, *, fps: int = 24,
                  width: int = 720, height: int = 1280) -> str:
    """把多个视频片段规范化后无缝拼接为一个成片。"""
    if not clips:
        raise MediaError("concat_videos: empty clip list")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="drama_concat_")
    normalized = []
    try:
        for i, clip in enumerate(clips):
            out = os.path.join(tmpdir, f"n{i:03d}.mp4")
            normalized.append(_normalize_clip(clip, out, fps=fps, width=width, height=height))
        listfile = os.path.join(tmpdir, "list.txt")
        with open(listfile, "w") as f:
            for n in normalized:
                f.write(f"file '{n}'\n")
        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listfile,
            "-c", "copy", "-movflags", "+faststart", dst,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        return dst
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
