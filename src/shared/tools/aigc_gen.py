"""AIGC 生成客户端：seedream 文/图生图 + seedance 图/文生视频（自包含，走 wenchain 网关）。

参考 /root/chengzhiyang/Viral_Video_Agent_Self_Evo/t2i.md 的调用方式，但**不依赖**短剧
项目的 agent.core 包（避免与本项目 src/agent 包名冲突），也不依赖 BNS / BOS 上传：

- 网关入口复用 as_core 的 ``WENCHAIN_BASE_URL``（OpenAI 兼容主机同源），path 换成
  ``/incommonuserr``；鉴权用 payload 里的 ``channel``（= as_core.WENCHAIN_API_KEY，
  默认 ``wangpantob_all_video_copy``），无独立 token。
- seedream 图生图的参考图可直接用 **base64 data URI**（实测 code=0），因此产品参考帧
  无需先传公网 BOS；seedance 图片同样接受 base64 data URL（实测 80–110KB 的 jpg OK，多图时
  每张须带 ``role``），所以商品参考帧可以直喂 seedance，不必再经 seedream 转成公网 bos_url。

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
# 竖屏 9:16 规格。1440x2560 = 0.5625，和用户素材帧（多为 1080x1920）、seedance 的 --ratio 9:16
# 以及最终成片完全一致。此前默认的 1664x2368 其实是 0.703（≈5:7），图生图时模型被迫重新构图，
# 商品外观在"参考帧→首帧→视频"链路上要被重构两次，是生成商品不像真实商品的一大来源。
T2I_SIZE = os.getenv("AIGC_T2I_SIZE", "1440x2560")
ASPECT_RATIO = os.getenv("AIGC_ASPECT", "9:16")
# seedance 2.0 默认会顺带生成背景音乐/音效，混进成片就是两条 BGM 打架。这句约束追加在每条
# 视频 prompt 末尾；模型不一定听话，所以落地后还会用 strip_audio 硬删音轨（双保险）。
NO_AUDIO_HINT = os.getenv("AIGC_NO_AUDIO_HINT",
                          "。画面只要视频，不要任何背景音乐、音效、人声或旁白，输出静音画面")
# 产品参考帧抽帧宽度：素材多是 4K，压太小会丢商品纹理细节（图生图吃的就是这些细节）
FRAME_WIDTH = int(os.getenv("AIGC_FRAME_WIDTH", "1440"))
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
    directive = "%s%s  --ratio %s  --dur %d" % ((prompt or "").strip(), NO_AUDIO_HINT,
                                                ratio or ASPECT_RATIO, dur)
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
    directive = "%s%s  --ratio %s  --dur %d" % ((prompt or "").strip(), NO_AUDIO_HINT,
                                                ratio or ASPECT_RATIO, dur)
    payload = _base_payload(I2V_MODEL, "t2v")
    payload["seedancepro_options"] = {"content": [{"type": "text", "text": directive}]}
    body = _post(payload, I2V_TIMEOUT)
    _check_ok(body)
    url = _extract_video_url(body)
    if not url:
        raise RuntimeError("seedance t2v 缺少 video_url: %s" % json.dumps(body, ensure_ascii=False)[:400])
    return url


def gen_video_multiref(prompt: str, image_paths, duration_sec, ratio: str = None) -> str:
    """seedance 多参考图直生视频：本地图片以 base64 data URL 直接喂给 seedance。返回视频 URL。

    实测（2026-08-06）：
    - seedance 2.0 的 ``image_url.url`` **接受 base64 data URL**（80–110KB 的 jpg，端到端约
      200s）。此前文档里"base64 首帧会超时、必须公网 URL"的结论只对 2048² 大图成立。
    - 多张图时**每张必须带 ``role``**，否则网关直接报
      ``40000002 role must be specified for image contents``；用 ``reference_image`` 实测 OK。
    这条路径省掉了"真实参考帧 → seedream 首帧"这一次重绘：商品外观不再被模型重构一遍，
    是保真度最大的一个来源。VLM 比对结论为"一致"。
    """
    paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not paths:
        raise ValueError("gen_video_multiref 需要至少一张本地参考图")
    dur = _quantize_duration(duration_sec)
    directive = "%s%s  --ratio %s  --dur %d" % ((prompt or "").strip(), NO_AUDIO_HINT,
                                                ratio or ASPECT_RATIO, dur)
    content = [{"type": "text", "text": directive}]
    for p in paths:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(p)},
                        "role": "reference_image"})
    payload = _base_payload(I2V_MODEL, "i2v")
    payload["seedancepro_options"] = {"content": content}
    body = _post(payload, I2V_TIMEOUT)
    _check_ok(body)
    url = _extract_video_url(body)
    if not url:
        raise RuntimeError("seedance 多参考图缺少 video_url: %s" % json.dumps(body, ensure_ascii=False)[:400])
    return url


def strip_audio(video_path: str) -> bool:
    """去掉视频自带音轨（seedance 2.0 会顺便生成背景音乐/音效）。

    成片的声音只应来自 BGM + 克隆配音 + 用户原声；生成片段自带的音乐混进去就是两条 BGM
    打架。prompt 里写"无背景音乐"不可靠（模型仍会配乐），所以落地后再硬删一次音轨。
    只复制视频流，不重编码。
    """
    if not video_path or not os.path.isfile(video_path):
        return False
    tmp = video_path + ".noaudio.mp4"
    cmd = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", video_path,
           "-c:v", "copy", "-an", tmp]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, video_path)
            return True
        _log.warning("strip_audio failed rc=%s %s", r.returncode, r.stderr[-200:])
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("strip_audio error %s", exc)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def extract_frame(video_path: str, timestamp: float, out_jpg: str) -> bool:
    """从本地视频在 timestamp 秒抽一帧存为 jpg（用作产品参考帧）。"""
    if not video_path or not os.path.isfile(video_path):
        return False
    os.makedirs(os.path.dirname(out_jpg) or ".", exist_ok=True)
    cmd = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, float(timestamp or 0)):.3f}", "-i", video_path,
           "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "2", out_jpg]
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
