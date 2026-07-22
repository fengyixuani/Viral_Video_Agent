"""BGM 鼓点/节拍分析工具，供 Agent 做卡点剪辑。

用 librosa 的 beat_track 检测 BGM 的节拍点（鼓点）时间戳。librosa 不在 server 主 env 里，
所以走子进程调一个带 librosa 的 conda env（BEAT_PYTHON，默认从常见 env 里挑存在的）。
结果按音频 (abspath+mtime+size) 缓存。
"""
from __future__ import annotations

import json
import os
import subprocess

import cache
import obs

_log = obs.get_logger("agent_edit_beats")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BEAT_PY_CANDIDATES = [
    os.getenv("BEAT_PYTHON", ""),
    "/root/chengzhiyang/miniconda3/envs/viral-asr/bin/python",
    "/root/chengzhiyang/miniconda3/envs/qwen3-asr-cu128/bin/python",
    "/root/chengzhiyang/Viral_Video_Split/third_party/TTS_Clone/cosyvoice_env/bin/python",
]
TIMEOUT = int(os.getenv("BEAT_DETECT_TIMEOUT", "180"))

_SCRIPT = (
    "import sys,json,librosa\n"
    "y,sr=librosa.load(sys.argv[1],mono=True)\n"
    "tempo,frames=librosa.beat.beat_track(y=y,sr=sr)\n"
    "beats=[round(float(t),3) for t in librosa.frames_to_time(frames,sr=sr)]\n"
    "print(json.dumps({'tempo':float(tempo),'beats':beats}))\n"
)


def _beat_python() -> str:
    for p in _BEAT_PY_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return ""


def _abspath(uri: str) -> str:
    if not uri:
        return ""
    return uri if os.path.isabs(uri) else os.path.abspath(os.path.join(AGENT_ROOT, uri))


def _digest(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"path": path, "mtime": int(st.st_mtime), "size": st.st_size, "v": 1}
    except OSError:
        return {"path": path, "v": 1}


def detect_beats(audio_path: str, *, use_cache: bool = True) -> dict:
    """检测音频节拍点，返回 {ok, beats:[秒...], tempo, error}；结果缓存复用。"""
    local = _abspath(audio_path)
    if not local or not os.path.isfile(local):
        return {"ok": False, "error": f"音频不存在：{audio_path}"}
    digest = _digest(local)
    if use_cache:
        cached = cache.get("beats", digest)
        if cached and isinstance(cached.get("payload"), dict):
            return cached["payload"]
    py = _beat_python()
    if not py:
        return {"ok": False, "error": "没有可用的 librosa 环境做鼓点检测"}
    try:
        r = subprocess.run([py, "-c", _SCRIPT, local],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        _log.warning("beat detect failed: %s", exc)
        return {"ok": False, "error": f"鼓点检测失败：{exc!s}"}
    if r.returncode != 0:
        return {"ok": False, "error": f"鼓点检测失败：{r.stderr.decode('utf-8','ignore')[-200:]}"}
    try:
        data = json.loads(r.stdout.decode("utf-8", "ignore").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": "鼓点检测输出无法解析"}
    beats = [float(b) for b in data.get("beats", []) if isinstance(b, (int, float))]
    result = {"ok": True, "beats": beats, "tempo": data.get("tempo", 0.0)}
    try:
        cache.set("beats", digest, result)
    except OSError:
        pass
    return result
