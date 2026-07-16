"""VLM 视觉核验工具（业务层）。

把一个或多个候选素材片段，连同「调用方 Agent 自拟的 prompt」，交给视觉大模型核对，
返回基于画面的观察结论。定位：给可行性验证 / 审核 Agent 的一个**可选** tool——
是否调用、用什么 prompt，全部由 Agent 自己决定。

- prompt 完全由 Agent 决定，本工具不改写；
- 传入候选片段（含 source_path + source_time_range）时，按时间段裁一小段临时 mp4，
  只把该片段发给 VLM（更省 token、更聚焦），裁剪失败则退回整段素材；
- 底层复用 ``as_core`` 的 vision 通道（wenchain ``ali-qwen3.7-plus``，原生 video_url）。
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import as_core
import obs

_log = obs.get_logger("vlm")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # 与 asr.py 一致：优先用 imageio_ffmpeg 的静态 ffmpeg
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

_SYSTEM = (
    "你是短视频画面核验助手。只依据给到的画面/视频回答调用方的问题，如实描述可见的主体、"
    "动作、景别、场景与产品，不臆测、不编造画面里不存在的内容；画面看不清或无法判断时明确说明。"
)


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


class VLMTool:
    name = "视觉核验"

    def _resolve(self, path: str) -> str:
        if not path or path.startswith(("http://", "https://", "data:")):
            return path or ""
        for cand in (path, os.path.join(PROJECT_ROOT, path)):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        return ""

    def _kind(self, path: str) -> str:
        return "image" if os.path.splitext(path)[1].lower() in _IMAGE_EXTS else "video"

    def _trim(self, src: str, time_range: str):
        """按 time_range 裁一小段临时 mp4，返回 (path, is_temp)；失败退回整段。"""
        start, end = _parse_range(time_range)
        dur = end - start
        if dur <= 0.05:
            return src, False
        fd, dst = tempfile.mkstemp(prefix="vlm_", suffix=".mp4")
        os.close(fd)
        try:
            subprocess.run(
                [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{max(0.0, start):.2f}", "-i", src, "-t", f"{dur:.2f}",
                 "-c", "copy", "-movflags", "+faststart", dst],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
            )
            if os.path.getsize(dst) > 0:
                return dst, True
        except (subprocess.CalledProcessError, OSError):
            pass
        try:
            os.remove(dst)
        except OSError:
            pass
        return src, False

    async def inspect(self, prompt: str, targets: list = None, media: list = None,
                      max_targets: int = 4) -> dict:
        """用视觉大模型核验候选片段。

        Args:
            prompt: 调用方 Agent 自拟的问题（完全由 Agent 决定，不做改写）。
            targets: 候选片段列表，每项含 ``source_path`` / ``source_time_range`` /
                ``asset_id``（可选）；按 time_range 裁段后逐个发给 VLM。
            media: 额外直接指定的媒体 ``[{"type","url"}]``（可选）。
            max_targets: 单次最多核验的片段数，避免一次塞太多素材。

        Returns:
            ``{"observation": str, "inspected": [label...], "error": str}``
        """
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"observation": "", "inspected": [], "error": "empty prompt"}
        media_items, temps, labels = [], [], []
        for t in (targets or [])[:max_targets]:
            if not isinstance(t, dict):
                continue
            src = self._resolve(t.get("source_path", ""))
            if not src:
                continue
            kind = self._kind(src)
            if kind == "video":
                path, is_temp = self._trim(src, t.get("source_time_range", ""))
            else:
                path, is_temp = src, False
            if is_temp:
                temps.append(path)
            media_items.append({"type": kind, "url": path})
            labels.append(t.get("asset_id") or os.path.basename(src))
        for m in (media or []):
            if isinstance(m, dict) and m.get("url"):
                media_items.append({"type": m.get("type", "video"), "url": m["url"]})
                labels.append(m.get("url"))
        if not media_items:
            return {"observation": "", "inspected": [], "error": "no resolvable media"}
        try:
            observation = await as_core.complete(_SYSTEM, prompt, vision=True, media=media_items)
        except Exception as exc:  # noqa: BLE001
            _log.warning("vlm inspect failed: %s", exc)
            return {"observation": "", "inspected": labels, "error": str(exc)[:200]}
        finally:
            for path in temps:
                try:
                    os.remove(path)
                except OSError:
                    pass
        _log.info("vlm inspect targets=%d chars=%d", len(media_items), len(observation or ""))
        return {"observation": observation or "", "inspected": labels, "error": ""}
