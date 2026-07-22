"""AIGC 生成客户端：seedream 文/图生图 + seedance 图/文生视频（自包含，走 wenchain 网关）。

参考 /root/chengzhiyang/Viral_Video_Agent_Self_Evo/t2i.md 的调用方式，但**不依赖**短剧
项目的 agent.core 包（避免与本项目 src/agent 包名冲突），也不依赖 BNS / BOS 上传：

- 网关入口复用 as_core 的 ``WENCHAIN_BASE_URL``（OpenAI 兼容主机同源），path 换成
  ``/incommonuserr``；鉴权用 payload 里的 ``channel``（= as_core.WENCHAIN_API_KEY，
  默认 ``wangpantob_all_video_copy``），无独立 token。
- seedream 图生图的参考图可直接用 **base64 data URI**（实测 code=0），因此产品参考帧
  无需先传公网 BOS；seedance 首帧则用 seedream 返回的公网 bos_url（本身即公网直链）。

所有函数都是阻塞的（requests / ffmpeg），调用方用 ``asyncio.to_thread`` 包起来。
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time

import obs
import as_core

_log = obs.get_logger("aigc_gen")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

# 模型（见 t2i.md 速查表）
T2I_MODEL = os.getenv("AIGC_T2I_MODEL", "doubao-seedream-5-0-260128")
I2V_MODEL = os.getenv("AIGC_I2V_MODEL", "doubao-seedance-2-0")
# 竖屏 9:16 规格
T2I_SIZE = os.getenv("AIGC_T2I_SIZE", "1664x2368")
ASPECT_RATIO = os.getenv("AIGC_ASPECT", "9:16")
T2I_TIMEOUT = int(os.getenv("AIGC_T2I_TIMEOUT", "300"))
I2V_TIMEOUT = int(os.getenv("AIGC_I2V_TIMEOUT", "900"))


def available() -> bool:
    """网关是否可用（有 requests + 配置了 wenchain）。"""
    return requests is not None and bool(as_core.WENCHAIN_API_KEY) and as_core.USE_WENCHAIN


def _endpoint() -> str:
    return as_core.WENCHAIN_BASE_URL.rstrip("/") + "/incommonuserr"


def _channel() -> str:
    return as_core.WENCHAIN_API_KEY or "wangpantob_all_video_copy"


def _base_payload(model: str, tag: str) -> dict:
    qid = int(time.time() * 1000) + random.randint(0, 999)
    msgs = [{"role": "user", "content": tag}]
    return {"channel": _channel(), "chat_id": qid, "query_id": qid, "model": model,
            "stream": False, "message_user": msgs, "messages": msgs, "message_prompt": msgs}


def _post(payload: dict, timeout: int) -> dict:
    if requests is None:
        raise RuntimeError("aigc_gen 需要 requests 包")
    resp = requests.post(_endpoint(),
                         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                         headers={"Content-Type": "application/json"}, timeout=timeout)
    try:
        return resp.json()
    except ValueError:
        return {"_http_status": resp.status_code, "_raw": resp.text[:600]}


def _check_ok(body: dict):
    status = (body or {}).get("status") or {}
    if status.get("code") != 0:
        raise RuntimeError("wenchain 生成失败: %s" % json.dumps(body, ensure_ascii=False)[:500])


def _data_uri(image_path: str) -> str:
    import base64
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return "data:image/%s;base64,%s" % (ext, b64)


def gen_image(prompt: str, *, size: str = None, ref_image_paths=None, ref_image_urls=None) -> str:
    """seedream 文/图生图，返回首图公网 bos_url。

    ref_image_paths 为本地图（转 base64 data URI）；ref_image_urls 为公网 URL；两者任一非空
    则走图生图 i2i（保留参考主体：商品外观/配色/logo），否则纯文生图。
    """
    payload = _base_payload(T2I_MODEL, "t2i")
    opts = {"prompt": prompt, "size": size or T2I_SIZE, "n": 1,
            "response_format": "url", "watermark": False}
    refs = list(ref_image_urls or [])
    for p in (ref_image_paths or []):
        if p and os.path.isfile(p):
            refs.append(_data_uri(p))
    if refs:
        opts["image"] = refs
    payload["seedream_options"] = opts
    body = _post(payload, T2I_TIMEOUT)
    _check_ok(body)
    for item in ((body.get("data") or {}).get("data") or []):
        url = item.get("bos_url") or item.get("url")
        if url:
            return url
    raise RuntimeError("seedream 响应缺少图片 url: %s" % json.dumps(body, ensure_ascii=False)[:400])


def _quantize_duration(duration_sec) -> int:
    """seedance 支持 4-15s 任意整数：向上取整并夹到 [4,15]。"""
    try:
        d = int(duration_sec)
        if float(duration_sec) > d:
            d += 1
    except (TypeError, ValueError):
        d = 5
    return max(4, min(15, d))


def _extract_video_url(body: dict) -> str:
    data = (body or {}).get("data") or {}
    content = data.get("content") or {}
    if isinstance(content, dict) and content.get("video_url"):
        return content["video_url"]
    result_text = data.get("result")
    if result_text:
        try:
            c = (json.loads(result_text).get("content") or {})
            if isinstance(c, dict) and c.get("video_url"):
                return c["video_url"]
        except (ValueError, TypeError):
            pass
    return ""


def gen_video_i2v(prompt: str, first_frame_url: str, duration_sec, ratio: str = None) -> str:
    """seedance 图生视频：首帧须公网 URL（seedream 输出 bos_url 即可）。返回视频 URL。"""
    dur = _quantize_duration(duration_sec)
    directive = "%s  --ratio %s  --dur %d" % ((prompt or "").strip(), ratio or ASPECT_RATIO, dur)
    payload = _base_payload(I2V_MODEL, "i2v")
    payload["seedancepro_options"] = {"content": [
        {"type": "text", "text": directive},
        {"type": "image_url", "image_url": {"url": first_frame_url}},
    ]}
    body = _post(payload, I2V_TIMEOUT)
    _check_ok(body)
    url = _extract_video_url(body)
    if not url:
        raise RuntimeError("seedance i2v 缺少 video_url: %s" % json.dumps(body, ensure_ascii=False)[:400])
    return url


def gen_video_t2v(prompt: str, duration_sec, ratio: str = None) -> str:
    """seedance 文生视频（无首帧兜底）。返回视频 URL。"""
    dur = _quantize_duration(duration_sec)
    directive = "%s  --ratio %s  --dur %d" % ((prompt or "").strip(), ratio or ASPECT_RATIO, dur)
    payload = _base_payload(I2V_MODEL, "t2v")
    payload["seedancepro_options"] = {"content": [{"type": "text", "text": directive}]}
    body = _post(payload, I2V_TIMEOUT)
    _check_ok(body)
    url = _extract_video_url(body)
    if not url:
        raise RuntimeError("seedance t2v 缺少 video_url: %s" % json.dumps(body, ensure_ascii=False)[:400])
    return url


def extract_frame(video_path: str, timestamp: float, out_jpg: str) -> bool:
    """从本地视频在 timestamp 秒抽一帧存为 jpg（用作产品参考帧）。"""
    if not video_path or not os.path.isfile(video_path):
        return False
    os.makedirs(os.path.dirname(out_jpg) or ".", exist_ok=True)
    cmd = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, float(timestamp or 0)):.3f}", "-i", video_path,
           "-frames:v", "1", "-vf", "scale=1080:-2", "-q:v", "3", out_jpg]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return r.returncode == 0 and os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def download(url: str, out_path: str, timeout: int = 300) -> bool:
    """把生成结果（图/视频公网 URL）下载到本地。"""
    if requests is None or not url:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                _log.warning("download HTTP %s for %s", r.status_code, url[:80])
                return False
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except (OSError, requests.RequestException) as exc:
        _log.warning("download failed %s: %s", url[:80], exc)
        return False


def probe_duration(path: str) -> float:
    try:
        return as_core._probe_duration(path)
    except Exception:  # noqa: BLE001
        return 0.0
