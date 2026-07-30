"""whq_understand — 在 whq_clone 内调用 Split 的**深度理解脚本**，产出 Split 级理解产物。

（模块名刻意用 whq_understand，避开与 Agent 的 src/understanding 包重名——后者在 sys.modules
里会抢占 `understanding` 这个名字，导致 import 到错的模块。）

目的：让网页/自动链路也能拿到与手工版同等质量的 DNA + 素材理解（全模型，无需人工）。
- understand_reference.py  -> dna_understanding/<prefix>_dna_template*.md（参考视频 VLM 深度分析）
- understand_user_segments.py -> user_understanding/<prefix>_user_assets_*/all_user_assets.json
  （逐素材 VLM 分段理解 + 词级 ASR）

两个脚本跑在 Split 的 viral-split 环境、cwd=Split 根；VLM/ASR 走既有网关与模型（config 默认）。
脚本会打印 ``OUTPUT: <path>``，这里解析拿到产物路径，再喂给 whq_clone（dna_md/assets_json）。
比 reference_dna 轻量重建 + Agent 实时理解更精细，但更慢（逐素材 VLM，数分钟级）。
"""
import hashlib
import hashlib
import os
import subprocess

from _common import REPO, COMMON, VENDOR

SPLIT_PY = os.getenv("SPLIT_UNDERSTAND_PYTHON", "/root/miniconda3/envs/viral-split/bin/python")
ASR_PY = os.getenv("ASR_PYTHON", "/root/miniconda3/envs/viral-split-asr/bin/python")


def _run(script, env_extra, timeout=None, retries=2):
    """跑 Split 理解脚本，返回其打印的 OUTPUT 路径。失败重试；耗尽才抛（含完整日志）。"""
    timeout = timeout or int(os.getenv("WHQ_UNDERSTAND_TIMEOUT", "3600"))
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items() if v is not None})
    env.setdefault("USE_WENCHAIN_OPENAI", "1")
    env.setdefault("VIDEO_VLM_USE_WENCHAIN", "1")
    # Split 脚本 import config/pipeline_utils（在 common/），需把 common 与 vendor 挂进 PYTHONPATH
    pp = [COMMON, VENDOR, os.environ.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pp if p])
    script_path = os.path.join(REPO, script)
    if not (os.path.exists(SPLIT_PY) and os.path.exists(script_path)):
        raise RuntimeError("缺 Split 理解脚本/解释器: {} / {}".format(SPLIT_PY, script_path))
    attempts = max(1, retries + 1)
    last_out = ""
    for attempt in range(1, attempts + 1):
        print("[whq_understand] RUN {} {} (attempt {}/{})".format(SPLIT_PY, script, attempt, attempts), flush=True)
        proc = subprocess.run([SPLIT_PY, script], cwd=REPO, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = proc.stdout or ""
        last_out = out
        if proc.returncode == 0:
            path = None
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("OUTPUT:"):
                    path = s[len("OUTPUT:"):].strip()
            if path and os.path.exists(path):
                return path
            print("[whq_understand] exit0 但无有效 OUTPUT，重试", flush=True)
        else:
            # 完整错误落盘，便于定位（不再截断）
            log_path = os.path.join(REPO, "outputs", "whq_understand_last_error.log")
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as fh:
                    fh.write(out)
            except Exception:
                pass
            print("[whq_understand] {} 失败(exit {}, attempt {}/{}); 完整日志: {}\n{}".format(
                script, proc.returncode, attempt, attempts, log_path, out[-800:]), flush=True)
    raise RuntimeError("{} 连续 {} 次失败; 末次日志尾: {}".format(script, attempts, last_out[-500:]))


def understand_reference(ref_video, prefix, gpu="1"):
    """参考视频深度理解 -> DNA md 路径。同一参考视频结果缓存复用（降随机、提速）。"""
    if not (ref_video and os.path.exists(ref_video)):
        print("[whq_understand] 无参考视频, 跳过 DNA 深度理解", flush=True)
        return None
    # 缓存键：参考视频路径+大小+mtime。命中则复用，保证多次跑 DNA 一致（可用 WHQ_UNDERSTAND_NOCACHE=1 关）
    cache_md = None
    try:
        st = os.stat(ref_video)
        key = hashlib.md5("{}|{}|{}".format(ref_video, st.st_size, int(st.st_mtime)).encode()).hexdigest()[:16]
        cache_dir = os.path.join(REPO, "outputs", "whq_dna_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_md = os.path.join(cache_dir, "dna_{}.md".format(key))
        if os.path.exists(cache_md) and os.getenv("WHQ_UNDERSTAND_NOCACHE") != "1":
            print("[whq_understand] DNA 命中缓存(复用, 保证一致): {}".format(cache_md), flush=True)
            return cache_md
    except Exception:  # noqa: BLE001
        cache_md = None
    env = {"REFERENCE_VIDEO": ref_video, "OUTPUT_NAME_PREFIX": prefix,
           "OUTPUT_SUFFIX": "dna_template"}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    path = _run("understanding/understand_reference.py", env)
    if cache_md and path and os.path.exists(path):
        try:
            import shutil
            shutil.copyfile(path, cache_md)
        except Exception:  # noqa: BLE001
            pass
    return path


def understand_user_segments(material_paths, prefix, gpu="1"):
    """逐素材深度理解 + ASR -> all_user_assets.json 路径。"""
    paths = [p for p in dict.fromkeys(material_paths or []) if p and os.path.exists(p)]
    if not paths:
        print("[whq_understand] 无可用素材, 跳过素材深度理解", flush=True)
        return None
    env = {
        "USER_VIDEO_GLOB": os.pathsep.join(paths),  # 每个绝对路径即一个 glob，os.pathsep 分隔
        "OUTPUT_NAME_PREFIX": prefix,
        # 关闭这里的 ASR：voiceover 需要的词级 ASR 由 whq_input 的 asr_tokens 统一产出，
        # 避免同一批素材被 ASR 两遍（深度理解一遍 + asr_tokens 一遍），省几分钟。
        "ENABLE_USER_ASR": "0",
        "REUSE_USER_UNDERSTANDING": "1",
        "USER_UNDERSTAND_CONCURRENCY": os.getenv("WHQ_UNDERSTAND_CONCURRENCY", "4"),
    }
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return _run("understanding/understand_user_segments.py", env)


def deep_understand(ref_video, material_paths, prefix, gpu="1"):
    """产出 (dna_md, assets_json)。任一失败返回该项 None，由上层回退到轻量重建。"""
    dna = assets = None
    try:
        dna = understand_reference(ref_video, prefix, gpu=gpu)
    except Exception as exc:  # noqa: BLE001
        print("[whq_understand] DNA 深度理解失败(回退轻量重建): {}".format(str(exc)[:200]), flush=True)
    try:
        assets = understand_user_segments(material_paths, prefix, gpu=gpu)
    except Exception as exc:  # noqa: BLE001
        print("[whq_understand] 素材深度理解失败(回退实时理解): {}".format(str(exc)[:200]), flush=True)
    return dna, assets
