"""asr_tokens — 词级 ASR 子步骤（Agent 移植新增）。

whq 的「原声保留对窗」和「语速贴参考」都依赖**逐字（token 级）时间戳**：
- 用户素材词级 ASR -> all_source_asr.json -> voiceover.load_user_speech / edit_planner.build_speech_map
- 参考视频词级 ASR -> ref_asr_items -> edit_planner._ref_cps_for_range / reference_dna 描述

Agent 现有的 ``src/shared/tools/asr.py`` 只吐句级 segments（丢弃了 token 时间戳），且属共享层。
按用户决策，whq_clone **自带独立的词级 ASR 子步骤**，复用 Split 兄弟仓的
``understanding/batch_qwen3_asr.py``（viral-split-asr 环境 + Qwen3-ASR 权重），产出标准
all_source_asr.json。与 run_clone.collect_reference_asr 同一套调法，抽成公共函数供两处复用。

无 ASR 环境/失败时优雅降级：返回空（-> 全克隆 + 全段默认语速，等价旧行为，不阻断主链路）。
"""
import glob
import json
import os
import subprocess

from _common import REPO, FFMPEG


def _asr_env(cache_dir):
    env = os.environ.copy()
    env["NARIS_ASR_DIR"] = cache_dir
    env.setdefault("FFMPEG", FFMPEG)
    env.setdefault("QWEN3_ASR_MODEL", os.path.join(REPO, "models", "Qwen3-ASR-0.6B"))
    env.setdefault("QWEN3_FORCED_ALIGNER", os.path.join(REPO, "models", "Qwen3-ForcedAligner-0.6B"))
    return env


def _input_fingerprint(source_glob):
    """glob 命中文件的 (路径,大小,mtime) 指纹。

    缓存原先只看 all_source_asr.json 是否存在, 同一 cache_dir 换了参考视频/素材会直接
    读到**上一次的 ASR 结果**(原声对窗与 ref_cps 全错且无任何报错)。指纹落在 sidecar,
    不符即重跑。
    """
    from pipeline_utils import fingerprint, file_fingerprint
    paths = sorted(glob.glob(source_glob)) or ([source_glob] if os.path.exists(source_glob) else [])
    return fingerprint(file_fingerprint(paths))


def _fp_path(out_json):
    return out_json + ".fp"


def _cache_valid(out_json, fp):
    if not os.path.exists(out_json):
        return False
    try:
        with open(_fp_path(out_json), encoding="utf-8") as f:
            return f.read().strip() == fp
    except OSError:
        return False


def _mark_cache(out_json, fp):
    try:
        with open(_fp_path(out_json), "w", encoding="utf-8") as f:
            f.write(fp)
    except OSError:
        pass


def run_asr(source_glob, cache_dir):
    """对 ``source_glob`` 匹配的视频跑 Qwen3-ASR，产出 ``cache_dir/all_source_asr.json``。

    输入指纹未变则复用缓存（ASR 重、耗时几十秒）；输入换了会自动重跑。
    返回 json 路径；环境缺失/失败返回 None。
    """
    os.makedirs(cache_dir, exist_ok=True)
    out_json = os.path.join(cache_dir, "all_source_asr.json")
    fp = _input_fingerprint(source_glob)
    if _cache_valid(out_json, fp):
        return out_json
    asr_python = os.getenv("ASR_PYTHON", "/root/miniconda3/envs/viral-split-asr/bin/python")
    script = os.path.join(REPO, "understanding", "batch_qwen3_asr.py")
    if not (os.path.exists(asr_python) and os.path.exists(script)):
        print("[asr_tokens] 无 ASR 环境/脚本, 跳过词级 ASR (回退全克隆+默认语速)", flush=True)
        return None
    env = _asr_env(cache_dir)
    env["NARIS_SOURCE_GLOB"] = source_glob
    try:
        print("[asr_tokens] 跑 Qwen3-ASR: {} -> {}".format(source_glob, out_json), flush=True)
        subprocess.run([asr_python, script], cwd=REPO, env=env, check=True)
    except Exception as exc:
        print("[asr_tokens] ASR 失败(降级): {}".format(str(exc)[:200]), flush=True)
        return None
    if not os.path.exists(out_json):
        return None
    _mark_cache(out_json, fp)
    return out_json


def reference_asr_items(ref_video, cache_dir):
    """参考视频逐字 asr_items 列表（供 ref_cps / key_beats 描述）。失败返回 []。"""
    if not ref_video or not os.path.exists(ref_video):
        print("[asr_tokens] 无参考视频, 跳过参考语速采集", flush=True)
        return []
    out_json = run_asr(ref_video, cache_dir)
    if not out_json:
        return []
    try:
        records = json.load(open(out_json, encoding="utf-8"))
        items = []
        for r in (records or []):
            items.extend(r.get("asr_items") or [])
        print("[asr_tokens] 参考语速: {} 个逐字 token".format(len(items)), flush=True)
        return items
    except Exception as exc:
        print("[asr_tokens] 参考 ASR 解析失败(降级): {}".format(str(exc)[:160]), flush=True)
        return []


def source_asr_json(source_paths, cache_dir):
    """对多条用户素材源视频跑词级 ASR，合并成一份 all_source_asr.json，返回路径。

    优化：把所有素材软链到一个临时目录，**一次** batch_qwen3_asr 调用（模型只加载一次，
    循环处理全部素材），比逐文件调用（每次都重载模型）快数倍；再把记录 source_path 映射回
    原始路径供下游按路径匹配。失败/无环境回退逐文件。
    """
    # 排序去重：指纹与素材出现顺序无关（候选顺序变了不该导致缓存失效、重跑几分钟 ASR）
    paths = sorted(p for p in dict.fromkeys(source_paths) if p and os.path.exists(p))
    if not paths:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    out_json = os.path.join(cache_dir, "all_source_asr.json")
    from pipeline_utils import fingerprint, file_fingerprint
    fp = fingerprint(file_fingerprint(paths))
    if _cache_valid(out_json, fp):
        return out_json

    # 软链到临时目录 -> 一个 glob 一次批量跑（一次模型加载）
    link_dir = os.path.join(cache_dir, "_asr_links")
    if os.path.isdir(link_dir):
        for f in os.listdir(link_dir):
            try:
                os.remove(os.path.join(link_dir, f))
            except OSError:
                pass
    os.makedirs(link_dir, exist_ok=True)
    link2real = {}
    for i, p in enumerate(paths):
        name = "{:03d}_{}".format(i, os.path.basename(p))
        link = os.path.join(link_dir, name)
        try:
            os.symlink(os.path.abspath(p), link)
            link2real[os.path.splitext(name)[0]] = p  # source_id -> 原始路径
        except OSError:
            pass
    merged = []
    j = run_asr(os.path.join(link_dir, "*"), os.path.join(cache_dir, "_batch"))
    if j:
        try:
            for r in json.load(open(j, encoding="utf-8")) or []:
                sid = r.get("source_video_id") or os.path.splitext(os.path.basename(r.get("source_path", "")))[0]
                real = link2real.get(sid)
                if real:  # 路径映射回原始素材
                    r["source_path"] = real
                    r["source_video_id"] = os.path.splitext(os.path.basename(real))[0]
                merged.append(r)
        except Exception:  # noqa: BLE001
            merged = []

    # 批量失败则回退逐文件（稳妥兜底）
    if not merged:
        for i, p in enumerate(paths):
            jf = run_asr(p, os.path.join(cache_dir, "src_{:03d}".format(i)))
            if jf:
                try:
                    merged.extend(json.load(open(jf, encoding="utf-8")) or [])
                except Exception:  # noqa: BLE001
                    continue
    if not merged:
        return None
    json.dump(merged, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    _mark_cache(out_json, fp)
    return out_json
