"""ASR 辅助：转写视频人声，供理解①与验证⑤使用。

优先用本机离线 Qwen3-ASR（`tools.asr.ASRTool` + `asr_cache` 共享缓存）；
若 ASR 环境不可用（无 CUDA/模型/解释器），回退用 qwen3.7-plus 从视频直接听写，
保证「理解和验证都带上 ASR 文本」这一能力始终可用。
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import obs

_log = obs.get_logger("drama.asr")


def _fallback_qwen(video_path: str) -> dict:
    """ASR 不可用时，用视觉大模型从视频听写口播（无逐字时间戳）。"""
    import asyncio

    import as_core

    system = ("你是听写员。请仔细听这条视频里的全部人声/旁白/台词，"
              "原样转写成中文文本，只输出一个 JSON：{\"text\":\"完整口播文字\"}，不要额外说明。")
    try:
        obj = asyncio.run(as_core.complete_json(
            system, "请转写视频里的全部语音为 JSON。", vision=True,
            media=[{"type": "video", "url": video_path}]))
        return {"text": str(obj.get("text", "")).strip(), "segments": [],
                "engine": "qwen3.7-plus-fallback"}
    except Exception as exc:  # noqa: BLE001
        _log.warning("qwen asr fallback failed: %s", exc)
        return {"text": "", "segments": [], "engine": "unavailable", "error": str(exc)}


def transcribe(video_path: str) -> dict:
    """返回 {text, segments, engine}。带共享缓存；失败自动回退。"""
    if not video_path or not os.path.isfile(video_path):
        return {"text": "", "segments": [], "engine": "unavailable",
                "error": f"not found: {video_path}"}
    try:
        import asr_cache

        cached = asr_cache.get(video_path)
        if cached and cached.get("text") is not None:
            return {**cached, "engine": "qwen3-asr-0.6b(cache)"}
    except Exception:  # noqa: BLE001
        asr_cache = None

    try:
        from tools.asr import ASRTool

        result = ASRTool().transcribe(video_path, timestamps=True)
        if not result.get("error"):
            try:
                if asr_cache:
                    asr_cache.set(video_path, result)
            except Exception:  # noqa: BLE001
                pass
            return result
        _log.warning("local ASR unavailable (%s), fallback to qwen", result.get("error"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("local ASR crashed (%s), fallback to qwen", exc)

    return _fallback_qwen(video_path)


def format_for_prompt(asr: dict, max_chars: int = 1500) -> str:
    """把 ASR 结果压成可读文本片段，喂给理解/验证 prompt。"""
    if not asr:
        return "（无 ASR 结果）"
    segs = asr.get("segments") or []
    if segs:
        lines = [f"[{s.get('start',0):.1f}-{s.get('end',0):.1f}s] {s.get('text','')}" for s in segs]
        text = "\n".join(lines)
    else:
        text = asr.get("text", "")
    text = (text or "").strip()
    if not text:
        return "（ASR 未识别到人声）"
    return text[:max_chars]
