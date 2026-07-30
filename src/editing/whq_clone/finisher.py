"""finisher — 成片收尾: 烧录配音字幕(文案) + 可选迁移参考视频真实 BGM。

字幕: 直接用配音脚本(voiceover 的 TTS plan items)按时间轴生成 SRT, ffmpeg subtitles
      滤镜(libass)烧白字黑边, 与主流水线 burn_synced_captions 风格一致。
BGM : 复用 whq_get_bgm/run.sh 把**参考视频真实 BGM** 迁移到成片(最贴近参考); 需 meishe
      环境, 不可达时优雅跳过, 保留「有配音+字幕」的成片。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import FFMPEG, REPO

_CAPTION_STYLE = ("FontName=Noto Sans CJK SC,FontSize=13,PrimaryColour=&H00FFFFFF,"
                  "OutlineColour=&H80000000,BorderStyle=1,Outline=1,Shadow=0,"
                  "Alignment=2,MarginV=70")


def _run(cmd, quiet=True):
    print("+", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    kw = dict(check=True)
    if quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run([str(c) for c in cmd], **kw)


def _ts(seconds):
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    s = total_ms // 1000
    return "{:02d}:{:02d}:{:02d},{:03d}".format(s // 3600, (s % 3600) // 60, s % 60, ms)


def _char_w(ch):
    return 2 if re.match(r"[\u4e00-\u9fff]", ch) else 1


def _wrap(text, max_width=28):
    text = re.sub(r"\s+", "", str(text or "").strip())
    lines, cur, w = [], "", 0
    for ch in text:
        cw = _char_w(ch)
        if cur and w + cw > max_width:
            lines.append(cur)
            cur, w = ch, cw
        else:
            cur += ch
            w += cw
    if cur:
        lines.append(cur)
    if len(lines) <= 2:
        return "\\N".join(lines)
    return "\\N".join([lines[0], "".join(lines[1:])])


def burn_captions(video, tts_items, out_video, work_dir):
    """按 TTS plan items(start/end/text)生成 SRT 并烧录到视频。"""
    os.makedirs(work_dir, exist_ok=True)
    srt = os.path.join(work_dir, "captions.srt")
    entries = []
    for it in tts_items:
        start = float(it.get("start") or 0.0)
        end = float(it.get("end") or 0.0)
        # caption_text: 原声段的字幕同音错字更正版(音频是实录不动), 有则优先烧录
        text = _wrap(it.get("caption_text") or it.get("text") or "")
        if end > start and text:
            entries.append((start, end, text))
    with open(srt, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(entries, 1):
            f.write("{}\n{} --> {}\n{}\n\n".format(i, _ts(s), _ts(e), t.replace("\\N", "\n")))
    sub_path = srt.replace("\\", "/").replace(":", "\\:").replace("'", r"'\\''")
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    try:
        _run([FFMPEG, "-y", "-i", video,
              "-map", "0:v:0", "-vf", "subtitles='{}':force_style='{}'".format(sub_path, _CAPTION_STYLE),
              "-map", "0:a?", "-c:a", "aac", "-ar", "44100", "-ac", "2",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart",
              out_video])
    except subprocess.CalledProcessError as exc:
        shutil.copy2(video, out_video)
        print("[finisher] 字幕烧录失败, 回退无字幕: {}".format(exc), flush=True)
    print("[finisher] captioned ->", out_video, flush=True)
    return out_video


def migrate_reference_bgm(ref_video, target_video, out_video, endpoint="online"):
    """复用 whq_get_bgm 迁移参考视频真实 BGM(需 meishe 环境)。失败则跳过。"""
    script = os.path.join(REPO, "whq_get_bgm", "run.sh")
    if not ref_video or not os.path.exists(script):
        print("[finisher] 跳过 BGM 迁移(无参考视频或脚本)", flush=True)
        return None
    try:
        subprocess.run(["bash", script, ref_video, target_video, out_video, endpoint],
                       cwd=REPO, check=True)
        if os.path.exists(out_video) and os.path.getsize(out_video) > 0:
            print("[finisher] ref-bgm ->", out_video, flush=True)
            return out_video
    except Exception as exc:
        print("[finisher] BGM 迁移失败, 保留无 BGM 成片: {}".format(str(exc)[:200]), flush=True)
    return None


def finish(voiced_video, tts_items, out_captioned, work_dir,
           ref_video=None, out_final=None, migrate_bgm=False, endpoint="online"):
    captioned = burn_captions(voiced_video, tts_items, out_captioned, work_dir)
    final = captioned
    if migrate_bgm and ref_video and out_final:
        bgm_out = migrate_reference_bgm(ref_video, captioned, out_final, endpoint=endpoint)
        if bgm_out:
            final = bgm_out
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description="收尾: 烧字幕 + 可选迁移参考 BGM")
    ap.add_argument("--video", required=True, help="有配音的视频")
    ap.add_argument("--tts-plan", required=True, help="voiceover 的 tts_plan_input.json")
    ap.add_argument("--out", required=True, help="烧字幕后的输出")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--ref", help="参考视频(迁移 BGM 用)")
    ap.add_argument("--out-final", help="迁移 BGM 后的最终输出")
    ap.add_argument("--migrate-bgm", action="store_true")
    ap.add_argument("--endpoint", default="online")
    args = ap.parse_args(argv)
    plan = json.load(open(args.tts_plan, encoding="utf-8"))
    items = plan.get("items", []) if isinstance(plan, dict) else plan
    final = finish(args.video, items, args.out, args.work_dir,
                   ref_video=args.ref, out_final=args.out_final,
                   migrate_bgm=args.migrate_bgm, endpoint=args.endpoint)
    print("final ->", final)


if __name__ == "__main__":
    main()
