"""复用参考视频 BGM：从参考视频分离/抽取背景音乐，供 Agent 成片使用。

- 纯音乐参考：整条音轨就是 BGM，直接 ffmpeg 抽出即可（快）。
- 含口播的参考：用 demucs（``--two-stems=vocals``）做人声/伴奏分离，取 ``no_vocals`` 伴奏作 BGM。
  demucs 在 Split 的 cosyvoice_env 里可用（GPU，较慢），结果按 (视频, pure_music) 缓存复用。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

import obs

_log = obs.get_logger("agent_edit_bgm_reuse")

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    _FFMPEG = os.getenv("FFMPEG", "ffmpeg")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMUCS_PYTHON = os.getenv(
    "DEMUCS_PYTHON",
    "/root/chengzhiyang/Viral_Video_Split/third_party/TTS_Clone/cosyvoice_env/bin/python")
DEMUCS_TIMEOUT = int(os.getenv("BGM_REUSE_DEMUCS_TIMEOUT", "600"))
OUT_DIR = os.path.join(AGENT_ROOT, "uploads", "bgm_reuse")


def _resolve(uri: str) -> str:
    if not uri or uri.startswith(("http://", "https://", "data:")):
        return ""
    for c in (uri, os.path.join(AGENT_ROOT, uri)):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def _key(path: str, pure_music: bool) -> str:
    try:
        st = os.stat(path)
        raw = f"{path}:{int(st.st_mtime)}:{st.st_size}:{pure_music}"
    except OSError:
        raw = f"{path}:{pure_music}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_audio(src: str, dst_wav: str) -> bool:
    try:
        r = subprocess.run(
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
             "-vn", "-ac", "2", "-ar", "44100", dst_wav],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return r.returncode == 0 and os.path.isfile(dst_wav) and os.path.getsize(dst_wav) > 0
    except (subprocess.SubprocessError, OSError) as exc:
        _log.warning("audio extract failed: %s", exc)
        return False


def extract_reference_bgm(video_uri: str, *, pure_music: bool = False) -> dict:
    """产出参考视频的 BGM 音轨（wav）。返回 {ok, output, mode, error}；结果缓存复用。"""
    src = _resolve(video_uri)
    if not src:
        return {"ok": False, "error": "参考视频不是本地文件，无法复用 BGM"}
    os.makedirs(OUT_DIR, exist_ok=True)
    # 缓存文件名带上产出方式：纯音乐整轨(track) / demucs 伴奏(demucs)。
    # 旧的 bgm_<key>.wav 是「没有 demucs 就抽整轨」时代的产物（含人声），换名等于自然作废，
    # 不会被当成伴奏复用。
    tag = "track" if pure_music else "demucs"
    out_wav = os.path.join(OUT_DIR, f"bgm_{_key(src, pure_music)}_{tag}.wav")
    if os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0:
        return {"ok": True, "output": out_wav, "mode": "cached"}

    # 纯音乐：整条音轨即 BGM，直接抽取
    if pure_music:
        if _extract_audio(src, out_wav):
            return {"ok": True, "output": out_wav, "mode": "extract"}
        return {"ok": False, "error": "参考音频抽取失败"}

    # 含口播：demucs 人声/伴奏分离，取 no_vocals 作 BGM
    if not (DEMUCS_PYTHON and os.path.exists(DEMUCS_PYTHON)):
        # 没有 demucs 环境：整条音轨里带着参考视频的人声，混进成片会听到别人在说话——
        # 那不是「参考的 BGM」而是「参考的整条声音」，判为不可用，让调用方去曲库选曲。
        # 真要旧行为（直接拿整轨）可设 BGM_REUSE_ALLOW_FULL_TRACK=1。
        if os.getenv("BGM_REUSE_ALLOW_FULL_TRACK", "").strip() not in ("1", "true", "yes"):
            _log.warning("demucs python 不存在(%s)，参考含口播，拒绝用整条音轨当 BGM", DEMUCS_PYTHON)
            return {"ok": False,
                    "error": "参考视频含口播且 demucs 环境不可用（{}），"
                             "整条音轨带人声不能当 BGM".format(os.path.basename(DEMUCS_PYTHON or "-"))}
        _log.warning("demucs python 不存在，按 BGM_REUSE_ALLOW_FULL_TRACK 退化为直接抽音轨")
        full = out_wav.replace("_demucs.wav", "_fulltrack.wav")
        if os.path.isfile(full) and os.path.getsize(full) > 0:
            return {"ok": True, "output": full, "mode": "cached"}
        if _extract_audio(src, full):
            return {"ok": True, "output": full, "mode": "extract_fallback"}
        return {"ok": False, "error": "参考音频抽取失败"}

    tmp = tempfile.mkdtemp(prefix="bgmsep_")
    try:
        wav_in = os.path.join(tmp, "in.wav")
        if not _extract_audio(src, wav_in):
            return {"ok": False, "error": "参考音频抽取失败"}
        sep_out = os.path.join(tmp, "sep")
        try:
            r = subprocess.run(
                [DEMUCS_PYTHON, "-m", "demucs", "--two-stems=vocals", "-o", sep_out, wav_in],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=DEMUCS_TIMEOUT)
        except (subprocess.SubprocessError, OSError) as exc:
            _log.warning("demucs failed: %s", exc)
            return {"ok": False, "error": f"demucs 分离失败：{exc!s}"}
        if r.returncode != 0:
            return {"ok": False, "error": f"demucs 分离失败：{r.stderr.decode('utf-8','ignore')[-200:]}"}
        # 找 no_vocals.wav（sep_out/<model>/in/no_vocals.wav）
        found = ""
        for root, _dirs, files in os.walk(sep_out):
            for f in files:
                if f == "no_vocals.wav":
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if not found:
            return {"ok": False, "error": "demucs 未产出 no_vocals 伴奏"}
        shutil.copy2(found, out_wav)
        return {"ok": True, "output": out_wav, "mode": "demucs"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
