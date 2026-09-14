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
from editing.caption_tool import render_caption, CAPTION_AVAILABLE

_log = obs.get_logger("agent_editor")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CAPTION_OK = CAPTION_AVAILABLE

# 旧的定音量混音：BGM 固定 0.32 再 amix。amix 默认 normalize=1 还会把人声再压 6dB，
# 参考素材录得轻时人声直接被埋（实测配音 -40dB、成片 ASR 42s 只认出 20 字）。只在新链路
# 跑不起来时兜底用。
_MIX_FALLBACK = ("[1:a]volume=0.32[bg];"
                 "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]")


def _mix_filter():
    """人声优先的 BGM 混音：人声定响度 + BGM 压到人声之下 + 说话时自动闪避。

    三件事一起做，缺一件人声都可能听不清：
      1) 人声轨 loudnorm 到 AGENT_VOICE_LUFS（默认 -16）——各镜原声/克隆配音的电平差异很大
         （克隆配音的响度跟着参考素材走，实测能差 19dB），不归一就忽大忽小；
      2) BGM loudnorm 到 AGENT_BGM_LUFS（默认 -30），先天低人声一档；
      3) sidechaincompress 以人声为触发，说话时把 BGM 再压 2-3dB，说完自己恢复
         （阈值 0.1：只有真正说话才触发，环境声不至于让 BGM 全程被压）。
    amix 必须 normalize=0（默认会把每路都除以路数，等于人声再掉 6dB），末尾 alimiter 防削顶。

    注意闪避的 key 取的是**归一化之前**的人声（asplit 在 loudnorm 前面）：把 key 接在
    loudnorm 后面会因为 loudnorm 动态模式的内部缓冲与 BGM 支路不同步，实测输出被截短约 2s
    （12s 的测试片只出 10s）。配音在 tts.py 里已经归一过，所以 key 的电平足够触发。

    实测（8.3s 配音 + BGM，后接 4s 静音）：说话段 -16.0dB / 纯 BGM 段 -28.7dB，人声高出
    12.7dB；单看 BGM 支路说话中 -37.3dB、说完 -32.2dB，闪避深度 5.1dB。旧的定音量混音同一
    素材是人声 -32.0dB、BGM -31.8dB，人声只高 0.2dB——等于听不见。
    """
    voice = (os.getenv("AGENT_VOICE_LUFS", "-16") or "-16").strip()
    bgm = (os.getenv("AGENT_BGM_LUFS", "-30") or "-30").strip()
    return (
        "[0:a]aresample=44100,asplit=2[v0][vkey];"
        "[v0]loudnorm=I={v}:TP=-1.5:LRA=11[v1];"
        "[1:a]aresample=44100,loudnorm=I={b}:TP=-6:LRA=11[bgn];"
        "[bgn][vkey]sidechaincompress=threshold=0.1:ratio=2:attack=20:release=400[bgduck];"
        "[v1][bgduck]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
        "alimiter=limit=0.95[a]"
    ).format(v=voice, b=bgm)


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
                width: int = 720, height: int = 1080, fps: int = 30,
                mute_source: bool = False) -> dict:
    """按 clips 逐片段剪辑并拼接成片，返回 ``{output, ops, error, clip_count}``。

    clips: ``[{slot_id, source_path, source_time_range, target_duration, caption_text?, speed?}]``
    mute_source: 静音原声（纯音乐参考时用）——丢弃素材原声、加静音音轨，最终只保留 BGM。
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
            tgt = float(clip.get("target_duration") or 0.0)
            # 普通镜：剪到「复刻分镜的镜头时长」(target_duration)——从所选片段起点截取该时长；
            # 但不超过所选片段的可用长度、也不拉伸/垫白帧（源不够长就用可用长度）。
            # 配音镜以 TTS 时长为准；卡点镜用 force_duration（吸附到鼓点）。
            if clip.get("tts_audio_path"):
                dur = max(0.3, float(clip.get("target_duration") or seg_len or 2.0))
            elif clip.get("force_duration"):
                dur = max(0.3, float(clip["force_duration"]))
            elif seg_len > 0 and tgt > 0:
                dur = max(0.3, min(seg_len, tgt))
            else:
                dur = max(0.3, seg_len or tgt or 2.0)
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
                if mute_source:
                    # 静音：丢弃原声，补一条静音音轨（保证 concat 流一致 + 后续 BGM 能混音）
                    base += ["-f", "lavfi", "-t", f"{dur:.3f}",
                             "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
                    if cap_burned:
                        base += ["-i", cap_png, "-filter_complex",
                                 f"[0:v]{vchain}[bg];[bg][2:v]overlay=0:0:format=auto[v]",
                                 "-map", "[v]", "-map", "1:a"]
                    else:
                        base += ["-filter_complex", f"[0:v]{vchain}[v]", "-map", "[v]", "-map", "1:a"]
                elif cap_burned:
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
                        "tts": bool(tts_audio), "beat_synced": bool(clip.get("beat_synced")),
                        "ok": ok, "error": err[:200]})
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
            base = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", concat, "-i", resolved_bgm]
            tail = ["-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", mixed]
            ok, err = _run(base + ["-filter_complex", _mix_filter()] + tail, 240)
            if not ok:
                # 归一化/闪避链跑不起来（旧 ffmpeg 缺 sidechaincompress 等）就退回原来的定音量混音
                _log.warning("人声优先混音失败，退回固定音量混音：%s", err[:200])
                ok, err = _run(base + ["-filter_complex", _MIX_FALLBACK] + tail, 150)
            ops.append({"op": "bgm_mix", "bgm": os.path.basename(resolved_bgm), "ok": ok, "error": err[:200]})
            if ok:
                final = mixed

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(final, out_path)
        _log.info("agent edit built %s from %d clips", out_path, len(seg_paths))
        return {"output": out_path, "ops": ops, "error": "", "clip_count": len(seg_paths)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def mix_bgm(video_path: str, bgm_path: str) -> dict:
    """给已剪好的成片**原地**补混 BGM（临时文件写好后 replace 回原路径）。

    供「曲库选曲延后到成片出来之后」的链路用：先无 BGM 出片 → 按成片内容选曲 →
    再把选中的 BGM 混进去，视频 uri 不变。混音链与 build_video 一致（人声优先 +
    闪避，失败退回定音量）。返回 {ok, error}。
    """
    video_abs, bgm_abs = _abspath(video_path), _abspath(bgm_path)
    if not (video_abs and os.path.isfile(video_abs)):
        return {"ok": False, "error": "成片文件不存在：{}".format(video_path)}
    if not (bgm_abs and os.path.isfile(bgm_abs)):
        return {"ok": False, "error": "BGM 文件不存在：{}".format(bgm_path)}
    fd, mixed = tempfile.mkstemp(prefix="bgmmix_", suffix=".mp4",
                                 dir=os.path.dirname(video_abs))
    os.close(fd)
    try:
        base = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_abs, "-i", bgm_abs]
        tail = ["-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", mixed]
        ok, err = _run(base + ["-filter_complex", _mix_filter()] + tail, 240)
        if not ok:
            _log.warning("补混 BGM 人声优先链失败，退回固定音量混音：%s", err[:200])
            ok, err = _run(base + ["-filter_complex", _MIX_FALLBACK] + tail, 150)
        if not ok or not os.path.getsize(mixed):
            return {"ok": False, "error": "BGM 混音失败：{}".format(err[:200])}
        os.replace(mixed, video_abs)
        _log.info("成片补混 BGM 完成 %s + %s", os.path.basename(video_abs), os.path.basename(bgm_abs))
        return {"ok": True, "error": ""}
    finally:
        try:
            if os.path.isfile(mixed):
                os.remove(mixed)
        except OSError:
            pass
