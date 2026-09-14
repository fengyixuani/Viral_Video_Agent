"""美摄资产库 BGM：按参考视频的配乐风格挑一首，下载到本地供成片混音。

catalog 来自剪辑 Agent 仓（``baidu/wk-strategy/audio_video_editing_agent`` 的 **meishe 分支**）
``resource/bgms/meta.json``：83 首，每首带 scene/genre/mood 标签、中文 description、bpm、
逐拍 beats(ms)、能量(dB)，以及 bos_url / poms_url 两个下载源。优先读 ``MEISHE_RESOURCE_DIR``
指向的原仓库（会更新），读不到用 ``assets/meishe_bgms.json`` 随包快照 —— 原仓库只存在于 meishe
分支且不一定在本机，不能当硬依赖。

mp3 不入库、实体在云端：先试 bos_url（BOS 直链，带限时鉴权签名），失败再试 poms_url（备用源），
下载后按 id 缓存到 ``uploads/bgm_lib/``。

选曲走文本模型，配乐需求按优先级取两档：
1. **参考视频的音频**：``shared/bgm.analyze_bgm`` 分析出的 style/mood/bpm/energy/instruments
   （按视频缓存），最贴参考；
2. 参考没有配乐信息（无音频/纯口播/分析失败）时，**看剪辑后的成片**：把成片（画面+声音）喂给
   Gemini（``analyze_edited_video``），产出这条片子需要什么样的配乐，按它去匹配。
两档都拿不到才退化为按商品调性挑。模型不可用或挑了个不存在的 id 就退化成标签打分。

卡点：catalog 自带逐拍时间，换算成秒直接给 ``_snap_clips_to_beats`` 用，不用再跑 librosa 子进程。
"""
from __future__ import annotations

import asyncio
import json
import os

import requests

import as_core
import bgm as bgm_analyzer
import obs
from editing import gemini_review

_log = obs.get_logger("agent_edit_bgm_lib")

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "meishe_bgms.json")
OUT_DIR = os.path.join(AGENT_ROOT, "uploads", "bgm_lib")
TIMEOUT = int(os.getenv("BGM_LIB_DOWNLOAD_TIMEOUT", "90"))
# 喂给模型的候选上限（83 首全给也不长，留个闸以防 catalog 变大）
MAX_CANDIDATES = int(os.getenv("BGM_LIB_MAX_CANDIDATES", "90"))


def catalog_path() -> str:
    """catalog 文件路径：原仓库优先，回退随包快照。"""
    repo = os.getenv("MEISHE_RESOURCE_DIR", "")
    if repo:
        p = os.path.join(repo, "bgms", "meta.json")
        if os.path.isfile(p):
            return p
        _log.warning("MEISHE_RESOURCE_DIR 下没有 bgms/meta.json（%s），用随包快照", p)
    return SNAPSHOT


def catalog() -> list:
    """[{id, name, tags, description, bpm, duration, beats_ms, energy_db, bos_url, poms_url}]。"""
    path = catalog_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f) or []
    except (OSError, ValueError) as exc:
        _log.warning("BGM catalog 读取失败(%s): %s", path, exc)
        return []
    out = []
    for e in raw:
        bid = str(e.get("id") or "")
        if not bid or not (e.get("bos_url") or e.get("poms_url")):
            continue           # 没 id 或没下载源的条目用不了
        out.append({
            "id": bid,
            "name": os.path.basename(str(e.get("path") or bid)),
            "tags": list(e.get("scene") or []) + list(e.get("genre") or []) + list(e.get("mood") or []),
            "description": str(e.get("description") or ""),
            "bpm": float(e.get("bpm") or 0.0),
            "duration": float(e.get("duration") or 0.0) / 1000.0,
            "beats_ms": list(e.get("beats") or []),
            "energy_db": e.get("energy_mean_db"),
            "bos_url": str(e.get("bos_url") or ""),
            "poms_url": str(e.get("poms_url") or ""),
        })
    return out


def beats_seconds(entry: dict) -> list:
    """逐拍时间 ms -> 秒（升序、去掉非法值）。"""
    out = []
    for ms in entry.get("beats_ms") or []:
        try:
            t = float(ms) / 1000.0
        except (TypeError, ValueError):
            continue
        if t > 0:
            out.append(round(t, 3))
    return sorted(set(out))


def download(entry: dict) -> str:
    """下载 BGM 到 uploads/bgm_lib/<id><原扩展名>（已下过直接复用）。失败返回 ""。

    库里既有 mp3 也有 wav，扩展名跟 catalog 的 path 走，别一律写成 .mp3（ffmpeg 能按内容读，
    但文件名骗人不利于排查）。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ext = os.path.splitext(entry["name"])[1].lower() or ".mp3"
    dst = os.path.join(OUT_DIR, "{}{}".format(entry["id"], ext))
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst
    for tag, url in (("bos", entry.get("bos_url")), ("poms", entry.get("poms_url"))):
        if not url:
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code != 200:
                _log.warning("BGM %s 源 %s HTTP %s", entry["name"], tag, r.status_code)
                continue
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(tmp) > 0:
                os.replace(tmp, dst)
                _log.info("BGM %s 下载完成(%s, %.1fMB)", entry["name"], tag,
                          os.path.getsize(dst) / 1e6)
                return dst
        except (requests.RequestException, OSError) as exc:
            _log.warning("BGM %s 源 %s 下载失败: %s", entry["name"], tag, str(exc)[:160])
    return ""


def _ref_brief(ref: dict) -> str:
    """参考视频 BGM 分析结果 -> 一句话选曲需求。"""
    if not ref or not ref.get("available") or not ref.get("has_bgm"):
        return ""
    bits = [("曲风", ref.get("style")), ("情绪", ref.get("mood")), ("能量", ref.get("energy")),
            ("BPM", ref.get("bpm")), ("人声", ref.get("vocal"))]
    line = "、".join("{}={}".format(k, v) for k, v in bits if v)
    inst = "、".join(str(x) for x in (ref.get("instruments") or [])[:4])
    if inst:
        line += "，主要乐器：" + inst
    if ref.get("summary"):
        line += "。原配乐总结：" + str(ref["summary"])
    return line


_VIDEO_PROMPT = (
    "你是短视频配乐分析师。给你的是一条刚剪好的带货短视频成片（还没有配背景音乐）。"
    "请结合画面内容、剪辑节奏、口播语气与情绪，判断这条片子最适合什么样的 BGM。\n"
    "严格只输出 JSON（不要 markdown 代码块、不要多余文字）：\n"
    "{\n"
    '  "style": "适合的曲风（如 电子/流行/抒情/国风/放克/Lo-fi 等）",\n'
    '  "mood": "适合的情绪（如 紧张/欢快/治愈/high 等）",\n'
    '  "bpm": 整数或 null,               // 建议的节奏 BPM，无法判断填 null\n'
    '  "energy": "low/medium/high",\n'
    '  "instruments": ["建议的乐器/音色，最多4个"],\n'
    '  "summary": "一句话说明为什么这条片子适合这种配乐"\n'
    "}"
)


def analyze_edited_video(video_path: str) -> dict:
    """看剪辑后的成片（画面+声音），产出它需要的配乐特征。

    复用 gemini_review 的视频编码/网关通道（同一个 Gemini 模型能同时理解画面与音轨）。
    失败返回 {"available": False, "reason": ...}。
    """
    if not video_path or not os.path.isfile(video_path):
        return {"available": False, "reason": "成片文件不存在：{}".format(video_path)}
    b64 = gemini_review._encode_video(video_path)
    if not b64:
        return {"available": False, "reason": "成片编码失败"}
    try:
        resp = requests.post(
            "{}/v1/chat/completions".format(gemini_review.GATEWAY),
            headers={"Authorization": "Bearer {}".format(gemini_review.TOKEN),
                     "Content-Type": "application/json"},
            json={
                "model": gemini_review.MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": _VIDEO_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:video/mp4;base64," + b64}},
                ]}],
                "max_tokens": gemini_review.MAX_TOKENS,
            },
            timeout=gemini_review.TIMEOUT,
        )
    except (requests.RequestException, OSError) as exc:
        _log.warning("成片配乐分析请求失败: %s", str(exc)[:200])
        return {"available": False, "reason": "Gemini 请求失败：{!s}".format(exc)}
    if resp.status_code != 200:
        _log.warning("成片配乐分析 HTTP %s: %s", resp.status_code, resp.text[:200])
        return {"available": False, "reason": "Gemini HTTP {}".format(resp.status_code)}
    body = resp.json()
    choice = (body.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        return {"available": False, "reason": "Gemini 输出被截断（max_tokens 不足）"}
    parsed = gemini_review._parse_json((choice.get("message") or {}).get("content", ""))
    if not parsed:
        return {"available": False, "reason": "Gemini 返回无法解析为 JSON"}
    return {
        "available": True,
        "style": parsed.get("style", ""),
        "mood": parsed.get("mood", ""),
        "bpm": parsed.get("bpm"),
        "energy": parsed.get("energy", ""),
        "instruments": parsed.get("instruments", []) or [],
        "summary": parsed.get("summary", ""),
    }


def _video_brief(info: dict) -> str:
    """成片配乐分析结果 -> 一句话选曲需求。"""
    if not info or not info.get("available"):
        return ""
    bits = [("曲风", info.get("style")), ("情绪", info.get("mood")),
            ("能量", info.get("energy")), ("BPM", info.get("bpm"))]
    line = "、".join("{}={}".format(k, v) for k, v in bits if v)
    inst = "、".join(str(x) for x in (info.get("instruments") or [])[:4])
    if inst:
        line += "，建议乐器：" + inst
    if info.get("summary"):
        line += "。分析：" + str(info["summary"])
    return line


def _score(entry: dict, ref: dict) -> float:
    """无模型时的兜底打分：标签/描述命中需求的曲风情绪 + BPM 接近程度。"""
    hay = " ".join(entry["tags"]) + " " + entry["description"]
    score = 0.0
    for key in (ref.get("style"), ref.get("mood"), ref.get("energy"), ref.get("vocal")):
        for word in str(key or "").replace("/", " ").split():
            if word and word in hay:
                score += 1.0
    try:
        ref_bpm = float(ref.get("bpm") or 0)
    except (TypeError, ValueError):
        ref_bpm = 0.0
    if ref_bpm and entry["bpm"]:
        score += max(0.0, 2.0 - abs(entry["bpm"] - ref_bpm) / 20.0)
    return score


_SYSTEM = (
    "你是短视频配乐编辑。任务：从给定的曲库候选里，挑出与【配乐需求】最匹配的一首，用于成片 BGM。"
    "配乐需求可能来自参考爆款原配乐的分析，也可能来自剪辑后成片内容的分析，按给出的需求匹配即可。"
    "优先对齐曲风/情绪/能量，其次对齐 BPM（节奏快慢差太多会破坏剪辑节奏）。"
    "候选之外不要臆造。严格只返回 JSON：{\"id\": \"候选里的 id\", \"reason\": \"一句话理由\"}"
)


async def _llm_pick(items: list, need: str, product_name: str):
    """让模型挑一首。返回 (entry, reason)；失败返回 (None, "")。"""
    cands = [{"id": e["id"], "标签": "、".join(e["tags"][:8]), "bpm": round(e["bpm"], 1),
              "时长秒": round(e["duration"], 1), "描述": e["description"][:110]}
             for e in items[:MAX_CANDIDATES]]
    user = "".join([
        "配乐需求：{}\n".format(need or "（没分析出来，按商品调性挑）"),
        "成片商品：{}\n\n".format(product_name or "（未提供）"),
        "曲库候选（JSON）：\n", json.dumps(cands, ensure_ascii=False),
        "\n\n挑一首最匹配的，返回它的 id。",
    ])
    try:
        data = await as_core.complete_json(_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001
        _log.warning("BGM 选曲模型调用失败: %s", str(exc)[:200])
        return None, ""
    bid = str((data or {}).get("id") or "").strip()
    by_id = {e["id"]: e for e in items}
    if bid in by_id:
        return by_id[bid], str(data.get("reason") or "")[:200]
    _log.warning("BGM 选曲返回了不存在的 id=%s，改用标签打分", bid[:40])
    return None, ""


async def choose_bgm(reference_video: str = "", product_name: str = "",
                     edited_video: str = "") -> dict:
    """挑一首美摄库 BGM 并下载到本地。

    选曲依据按优先级：参考视频的音频分析 > 剪辑后成片的内容分析（传了 ``edited_video``
    且参考没配乐信息时）> 商品调性。

    返回 {ok, path, beats(秒), name, bpm, reason, matched_by, error}。beats 直接来自 catalog，
    调用方可以拿它卡点，不必再跑 librosa。matched_by 取值
    reference_audio / edited_video / product，表示这次实际按哪档需求匹配的。
    """
    items = catalog()
    if not items:
        return {"ok": False, "error": "美摄 BGM 资产库不可用（catalog 读不到）"}
    ref = {}
    if reference_video:
        try:
            ref = await asyncio.to_thread(bgm_analyzer.analyze_bgm, reference_video)
        except Exception as exc:  # noqa: BLE001
            _log.warning("参考 BGM 分析失败: %s", str(exc)[:160])
    brief = _ref_brief(ref)
    need, traits, matched_by = "", ref, "product"
    if brief:
        need = "对齐参考爆款的原配乐特征：" + brief
        matched_by = "reference_audio"
    elif edited_video:
        # 参考没有配乐信息（无音频/纯口播/分析失败）→ 看剪辑后的成片，按成片内容匹配
        info = await asyncio.to_thread(analyze_edited_video, edited_video)
        vb = _video_brief(info)
        if vb:
            need = "参考视频没有配乐信息。根据剪辑后成片（画面+声音）分析出的配乐需求：" + vb
            traits, matched_by = info, "edited_video"
        else:
            _log.warning("成片配乐分析不可用(%s)，退化按商品调性挑", info.get("reason", ""))
    entry, reason = await _llm_pick(items, need, product_name)
    if entry is None:
        entry = max(items, key=lambda e: _score(e, traits))
        reason = "标签/BPM 打分兜底（模型未给出可用结果）"
    path = await asyncio.to_thread(download, entry)
    if not path:
        return {"ok": False, "error": "BGM 下载失败：{}".format(entry["name"])}
    beats = beats_seconds(entry)
    _log.info("BGM 选定 %s (bpm=%.1f, %d 拍, matched_by=%s) 理由=%s", entry["name"], entry["bpm"],
              len(beats), matched_by, reason[:80])
    return {"ok": True, "path": path, "beats": beats, "name": entry["name"],
            "bpm": entry["bpm"], "reason": reason, "matched_by": matched_by,
            "matched_reference": matched_by == "reference_audio"}


def reference_music_available(reference_video: str) -> bool:
    """参考视频是否有可用的配乐信息（有音频且分析出 BGM）。

    analyze_bgm 结果按视频缓存，这里探测过之后 choose_bgm 再取是缓存命中，不重复调模型。
    调用方据此决定：能对齐参考就即时选曲；不能就把选曲延后到成片出来后按成片内容匹配。
    """
    if not reference_video:
        return False
    try:
        return bool(_ref_brief(bgm_analyzer.analyze_bgm(reference_video)))
    except Exception as exc:  # noqa: BLE001
        _log.warning("参考 BGM 探测失败: %s", str(exc)[:160])
        return False
