"""Agent 侧轻量 ffmpeg 剪辑器（agent_cut 分支「纯 Agent 剪辑」链路用）。

完全不依赖 Viral_Video_Split。按剪辑 Agent 给出的 edit_plan，逐片段执行：
trim（按 source_time_range + target_duration）→ scale/pad 到竖屏 → 可选变速 →
可选字幕烧录 → 统一编码；再 concat 拼接 → 可选 BGM 混音。
**每一步操作都记录进 ops**，供审片 Agent 复盘定位问题。

字幕烧录用 Pillow 把文字渲染成整帧透明 PNG，再用 ffmpeg overlay 叠加——因为随包的
imageio-ffmpeg 静态构建**没有 drawtext（缺 libfreetype）**，但都带 overlay。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import obs
from agent_edit.caption_tool import render_caption, CAPTION_AVAILABLE

_log = obs.get_logger("agent_editor")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CAPTION_OK = CAPTION_AVAILABLE


def _abspath(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _parse_range(text):
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _run(cmd, timeout=180):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if r.returncode == 0:
            return True, ""
        return False, r.stderr.decode("utf-8", "ignore")[-400:]
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout"
    except (OSError, FileNotFoundError) as exc:
        return False, str(exc)


def _render_caption_png(text: str, width: int, height: int, out_png: str) -> bool:
    """转调字幕烧录工具（agent_edit.caption_tool.render_caption）。"""
    return render_caption(text, width, height, out_png)


def build_video(clips: list, out_path: str, *, bgm_path: str = "",
                width: int = 720, height: int = 1080, fps: int = 30) -> dict:
    """按 clips 逐片段剪辑并拼接成片，返回 ``{output, ops, error, clip_count}``。

    clips: ``[{slot_id, source_path, source_time_range, target_duration, caption_text?, speed?}]``
    ops: 每个操作的记录（trim/字幕/拼接/BGM 及成功与否），供审片 Agent 复盘。
    """
    ops = []
    work = tempfile.mkdtemp(prefix="agentedit_")
    seg_paths = []
    try:
        for i, clip in enumerate(clips or []):
            src = _abspath(clip.get("source_path", ""))
            if not src or not os.path.isfile(src):
                ops.append({"idx": i, "slot_id": clip.get("slot_id"), "op": "trim",
                            "ok": False, "error": "source not found", "source": clip.get("source_path", "")})
                continue
            start, end = _parse_range(clip.get("source_time_range", ""))
            seg_len = (end - start) if end > start else 0.0
            # 时长优先看效果：普通镜用「所选片段的自然长度」，不去硬凑 target_duration（避免拉慢/垫白帧）；
            # 配音镜例外，时长以 TTS 音频为准（target_duration 已被设为配音时长）。
            if clip.get("tts_audio_path"):
                dur = max(0.3, float(clip.get("target_duration") or seg_len or 2.0))
            else:
                dur = max(0.3, seg_len or float(clip.get("target_duration") or 2.0))
            speed = float(clip.get("speed") or 1.0)
            caption = clip.get("caption_text", "") or ""
            if clip.get("burn_caption") is False:
                caption = ""
            seg = os.path.join(work, f"seg_{i:02d}.mp4")

            vchain = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                      f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}")
            if abs(speed - 1.0) > 1e-3:
                vchain += f",setpts=PTS/{speed:.3f}"
            af = f"atempo={speed:.3f}" if abs(speed - 1.0) > 1e-3 else "anull"

            cap_png = os.path.join(work, f"cap_{i:02d}.png")
            cap_burned = bool(caption) and _render_caption_png(caption, width, height, cap_png)

            tts_audio = _abspath(clip.get("tts_audio_path", "")) if clip.get("tts_audio_path") else ""
            if tts_audio and os.path.isfile(tts_audio):
                # 克隆配音镜：视频取源片段（不足配音时长则冻结末帧补足），音轨换成 TTS wav
                seg_len = (end - start) if end > start else dur
                pad = max(0.0, dur - seg_len)
                vtts = vchain + (f",tpad=stop_mode=clone:stop_duration={pad:.3f}" if pad > 0.05 else "")
                base = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{max(0.0, start):.3f}", "-i", src, "-i", tts_audio]
                if cap_burned:
                    base += ["-i", cap_png, "-filter_complex",
                             f"[0:v]{vtts}[bg];[bg][2:v]overlay=0:0:format=auto[v]",
                             "-map", "[v]", "-map", "1:a"]
                else:
                    base += ["-filter_complex", f"[0:v]{vtts}[v]", "-map", "[v]", "-map", "1:a"]
                base += ["-t", f"{dur:.3f}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", seg]
            else:
                # -ss/-t 作为**输入选项**放在 -i src 之前：精确读取 [start, start+dur]，
                # 与是否叠加字幕 PNG 无关（放在两个 -i 之间会被误当成后一个输入的选项）。
                base = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", src]
                if cap_burned:
                    base += ["-i", cap_png, "-filter_complex",
                             f"[0:v]{vchain}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
                             "-map", "[v]", "-map", "0:a?", "-af", af]
                else:
                    base += ["-vf", vchain, "-af", af]
                base += ["-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                         "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", seg]
            ok, err = _run(base, timeout=150)
            ops.append({"idx": i, "slot_id": clip.get("slot_id"), "op": "trim",
                        "source": os.path.basename(src), "source_time_range": clip.get("source_time_range", ""),
                        "target_duration": round(dur, 2), "speed": speed,
                        "caption": caption, "caption_burned": cap_burned,
                        "tts": bool(tts_audio), "ok": ok, "error": err[:200]})
            if ok and os.path.isfile(seg) and os.path.getsize(seg) > 0:
                seg_paths.append(seg)

        if not seg_paths:
            return {"output": "", "ops": ops, "error": "没有任何片段成功生成", "clip_count": 0}

        listf = os.path.join(work, "list.txt")
        with open(listf, "w", encoding="utf-8") as fh:
            fh.write("".join(f"file '{p}'\n" for p in seg_paths))
        concat = os.path.join(work, "concat.mp4")
        ok, err = _run([_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", concat], 150)
        if not ok:
            ok, err = _run([_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                            "-f", "concat", "-safe", "0", "-i", listf,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-ar", "44100", concat], 240)
        ops.append({"op": "concat", "clips": len(seg_paths), "ok": ok, "error": err[:200]})
        if not ok:
            return {"output": "", "ops": ops, "error": f"拼接失败：{err[:200]}", "clip_count": len(seg_paths)}

        final = concat
        resolved_bgm = _abspath(bgm_path) if bgm_path else ""
        if resolved_bgm and os.path.isfile(resolved_bgm):
            mixed = os.path.join(work, "mixed.mp4")
            ok, err = _run([_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                            "-i", concat, "-i", resolved_bgm,
                            "-filter_complex",
                            "[1:a]volume=0.32[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", mixed], 150)
            ops.append({"op": "bgm_mix", "bgm": os.path.basename(resolved_bgm), "ok": ok, "error": err[:200]})
            if ok:
                final = mixed

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(final, out_path)
        _log.info("agent edit built %s from %d clips", out_path, len(seg_paths))
        return {"output": out_path, "ops": ops, "error": "", "clip_count": len(seg_paths)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
