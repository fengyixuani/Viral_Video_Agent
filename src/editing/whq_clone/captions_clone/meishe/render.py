"""美摄花字后端编排: seq -> 挑花字/字体 -> 子进程提交美摄渲染 -> 成片。

美摄 SDK/服务只在 meishe repo 环境可用, 所以渲染一步起子进程(同 finisher.migrate_reference_bgm
的既有做法)。环境缺失/渲染失败一律返回 None, 由 caption_clone 回退本机 ASS 烧录 —— 不能因为
外部服务不可达就出不了片。

env:
  MEISHE_REPO        meishe 仓库根目录(含 meishe_asset_examples.py)
  MEISHE_PYTHON      跑该仓的解释器(默认 /root/miniconda3/bin/python3.13)
  MEISHE_GLOBAL_FONT 口播全局字体(服务端须注册, 默认 Noto Sans CJK SC)
  MEISHE_ENDPOINT    production(默认) / sandbox
"""
import json
import os
import subprocess

from . import fancy_match

_JOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_meishe_job.py")


def _env():
    repo = os.getenv("MEISHE_REPO", "")
    py = os.getenv("MEISHE_PYTHON", "/root/miniconda3/bin/python3.13")
    if not (repo and os.path.isdir(repo)):
        print("[captions_clone] 未配置 MEISHE_REPO, 跳过美摄花字后端", flush=True)
        return None, None
    if not os.path.exists(py):
        print("[captions_clone] MEISHE_PYTHON 不存在({}), 跳过美摄花字后端".format(py), flush=True)
        return None, None
    return repo, py


def burn(seq, video_in, video_out, work_dir, model=None):
    """给 seq 挑美摄花字/字体并提交云端渲染。成功返回成片路径, 否则 None。"""
    if not seq:
        return None
    repo, py = _env()
    if not repo:
        return None
    os.makedirs(work_dir, exist_ok=True)
    blocks = fancy_match.assign_assets(seq, model=model)
    blocks_path = os.path.join(work_dir, "meishe_blocks.json")
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["MEISHE_REPO"] = repo
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(repo, "src"), repo, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        subprocess.run([py, _JOB, "--blocks", blocks_path, "--video", video_in,
                        "--out", video_out], cwd=repo, env=env, check=True)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 美摄渲染失败(回退本机 ASS): {}".format(str(exc)[:200]), flush=True)
        return None
    if os.path.exists(video_out) and os.path.getsize(video_out) > 0:
        print("[captions_clone] 美摄花字成片 -> {}".format(video_out), flush=True)
        return video_out
    return None
