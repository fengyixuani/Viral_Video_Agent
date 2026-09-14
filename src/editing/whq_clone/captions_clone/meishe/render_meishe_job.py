"""美摄渲染子进程脚本(在 meishe repo 环境 / python3.13 下运行)。

移植 copy_zimu/v2/meishe/render_via_meishe.py, 但改成【自包含 + 参数化】: 只吃一个
blocks.json(由 captions_clone 产出) + 源视频, 组装美摄统一 schema inputs, 提交 RenderVideo,
并把成片下载到 --out。成功时最后一行打印 ``FINAL_OUTPUT <path>``。

不可直接被 Agent 主进程 import: 依赖 meishe repo 的 meishe_asset_examples /
open_storyline.*(BOS/服务发现), 只在那个环境里可用。由 render.py 起子进程调用:

  MEISHE_REPO=/path/to/audio_video_editing_agent \
  MEISHE_PYTHON=/root/miniconda3/bin/python3.13 \
  PYTHONPATH=$MEISHE_REPO/src:$MEISHE_REPO $MEISHE_PYTHON render_meishe_job.py \
      --blocks blocks.json --video in.mp4 --out out.mp4
"""
import argparse
import json
import os
import sys
import tempfile
import time

# 醒目角色走 fancy_texts(带 styleId); 口播走 subtitles 轨。
_FANCY_ROLES = {"emphasis", "highlight", "label", "hook"}
# 每角色花字填充色(#AARRGGBB, alpha=ff)。
_ROLE_COLOR = {"emphasis": "#ffEA3717", "highlight": "#ffFFF9C4",
               "label": "#ffFFFFE0", "hook": "#ffFFFFFF"}
# 每角色花字字号占比(相对画布高)。
_ROLE_FONT_RATIO = {"emphasis": 0.13, "highlight": 0.12, "label": 0.09, "hook": 0.13}
# 渲染服务上确定可用的字幕字体。清单里的美摄字体服务端未必注册; 花字走 styleId 自带样式不受影响。
_SAFE_FONT = "Noto Sans CJK SC [NotoSansCJKsc-Regular]"
_CHANNEL = os.getenv("MEISHE_CHANNEL", "video_editing_agent_api_key")


def _pos_translation(position, canvas_w, canvas_h):
    """清单位置 -> 美摄 translationX/Y(画布中心为原点, 顶部取负 Y)。"""
    top_y = -int(canvas_h * 0.33)
    off_x = int(canvas_w * 0.22)
    table = {
        "顶部居中": (0, top_y),
        "顶部偏左": (-off_x, -int(canvas_h * 0.31)),
        "顶部偏右": (off_x, -int(canvas_h * 0.31)),
        "底部居中": (0, int(canvas_h * 0.34)),
    }
    return table.get(position, (0, top_y))


def probe_video(path):
    """ffprobe 取 (width, height, fps, duration_s)。"""
    import subprocess
    ffprobe = os.getenv("FFPROBE") or os.getenv("COPY_ZIMU_FFPROBE") or "ffprobe"
    out = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration", "-of", "json", path,
    ], text=True)
    d = json.loads(out)
    st = d["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, den = st.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(d.get("format", {}).get("duration") or 0.0)
    return w, h, fps, dur


def build_inputs(blocks, video_path, w, h, fps, dur_ms, global_font):
    """blocks -> 美摄统一 schema inputs(视频轨 + 口播 subtitles + 醒目块 fancy_text_rec)。"""
    subtitles, fancy = [], []
    for b in blocks:
        s_ms = int(round(float(b["start"]) * 1000))
        e_ms = int(round(float(b["end"]) * 1000))
        if b.get("role") in _FANCY_ROLES and b.get("style_id"):
            tx, ty = _pos_translation(b.get("position"), w, h)
            fancy.append({
                "type": "general",
                "styleId": b["style_id"],
                "text": b["phrase"],
                "inPoint": s_ms,
                "duration": max(1, e_ms - s_ms),
                "fontSizeRatio": _ROLE_FONT_RATIO.get(b["role"], 0.12),
                "color": _ROLE_COLOR.get(b["role"], "#ffffffff"),
                "weight": 700,
                "font": b.get("font") or global_font,
                "translationX": tx,
                "translationY": ty,
            })
        else:
            subtitles.append({"text": b["phrase"],
                              "timeline_window": {"start": s_ms, "end": e_ms}})

    video_clip = {
        "source_path": video_path, "media_id": "media_0001", "clip_id": "clip_0001",
        "source_window": {"start": 0, "end": dur_ms},
        "timeline_window": {"start": 0, "end": dur_ms},
        "size": [w, h], "fps": fps, "playback_rate": 1.0,
    }
    media = {"media_id": "media_0001", "media_type": "video", "path": video_path,
             "metadata": {"duration": dur_ms, "width": w, "height": h,
                          "video_info": {"width": w, "height": h,
                                         "frame_rate": int(round(fps)),
                                         "duration": dur_ms / 1000.0}}}
    inputs = {
        "output_max_dimension_px": max(w, h), "aspect_ratio": "9:16",
        "load_media": {"media": [media]},
        "plan_timeline": {"tracks": {"video": [video_clip], "subtitles": subtitles}},
        "text_rec": [{"font_name": global_font}],
        "fancy_text_rec": fancy,
    }
    return inputs, len(subtitles), len(fancy)


def _load_examples(meishe_repo):
    """加载 meishe repo 的示例模块(含 _load_build_render_payload)。"""
    import importlib.util
    path = os.path.join(meishe_repo, "meishe_asset_examples.py")
    spec = importlib.util.spec_from_file_location("meishe_asset_examples", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    p = argparse.ArgumentParser(description="提交美摄 RenderVideo 烧花字字幕并下载成片")
    p.add_argument("--blocks", required=True, help="captions_clone 产出的 blocks.json")
    p.add_argument("--video", required=True, help="无字幕源视频(本地 mp4)")
    p.add_argument("--out", required=True, help="下载成片到此路径")
    p.add_argument("--font", default=os.getenv("MEISHE_GLOBAL_FONT", _SAFE_FONT),
                   help="口播全局字幕字体(服务端须注册)")
    p.add_argument("--endpoint", default=os.getenv("MEISHE_ENDPOINT", "production"))
    args = p.parse_args(argv)

    meishe_repo = os.getenv("MEISHE_REPO", "")
    if not (meishe_repo and os.path.isdir(meishe_repo)):
        print("MEISHE_REPO 未配置或不存在", flush=True)
        return 2

    with open(args.blocks, encoding="utf-8") as f:
        blocks = json.load(f)
    w, h, fps, dur = probe_video(args.video)
    dur_ms = int(round(dur * 1000))
    inputs, n_sub, n_fancy = build_inputs(blocks, args.video, w, h, fps, dur_ms, args.font)
    print("blocks: {} | subtitles(口播): {} | fancy(花字): {} | canvas {}x{} fps {} dur {:.2f}s"
          .format(len(blocks), n_sub, n_fancy, w, h, fps, dur), flush=True)

    M = _load_examples(meishe_repo)
    build = M._load_build_render_payload()
    from open_storyline.bos_manager.video_bos_manager import VideoBosManager
    from open_storyline.config_meishe import get_meishe_render_video_urls
    import httpx

    tag = "whq_captions_" + os.path.splitext(os.path.basename(args.video))[0][:24]
    bos = VideoBosManager()
    payload = build(inputs, bos, session_id=tag)

    log_id = "{}_{}".format(tag, int(time.time() * 1000))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        json_path = f.name
    bos_url, _ = bos.upload_result_file_with_cdn(json_path, "{}/{}/payload".format(tag, log_id))
    print("payload BOS:", bos_url, flush=True)

    req = {"bos_url": bos_url, "user_id": tag, "channel": _CHANNEL, "log_id": log_id}
    last = None
    for url in get_meishe_render_video_urls(args.endpoint)[:5]:
        try:
            last = httpx.post(url, json=req, timeout=900,
                              headers={"Content-Type": "application/json"}).json()
            if last.get("error") == 0:
                video_url = last["data"]["video_url"]
                print("SUCCESS task_id={}".format(last["data"].get("task_id")), flush=True)
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with httpx.stream("GET", video_url, timeout=900, follow_redirects=True) as r:
                    r.raise_for_status()
                    with open(args.out, "wb") as fo:
                        for chunk in r.iter_bytes():
                            fo.write(chunk)
                print("FINAL_OUTPUT", args.out, flush=True)
                return 0
            print("{} -> error={} errmsg={}".format(url, last.get("error"), last.get("errmsg")),
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print("{} -> EXC {}".format(url, exc), flush=True)
    print("FAILED last=", last, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
