"""参考字幕特效复刻的对外入口: 一步把成片烧上「跟参考视频同风格」的字幕。

被 finisher.burn_captions 调用(WHQ_CAPTION_CLONE != 0 时)。四步:

  1. profile_cache.analyze_reference  参考视频 -> style_profile(VLM + 像素校准, 按指纹缓存)
  2. charstream                       对成片跑词级 ASR -> 逐字流 -> 逐句念白(文本以 tts plan 为准)
  2.5 target_match.correct_typos      同音错字等长纠错(原声段字幕只能来自 ASR, 会有同音错字)
  3. target_match.plan                LLM 按参考风格拆块 + 标关键词 -> 单条字幕流 seq
  4. 烧录                             ass(本机 libass, 整块弹出/逐字揭示) / meishe(云端花字)

任何一步失败都返回 None, 由 finisher 回退它原有的句级 SRT 白字黑边 —— 字幕特效是增强项,
不能让它把出片卡死。

env:
  WHQ_CAPTION_CLONE=0|ass|meishe   总开关与后端(默认 ass; 0/off 关闭走旧句级 SRT)
  WHQ_CAPTION_REF_FPS=1.0          参考视频抽帧率(VLM + 像素校准)
  WHQ_CAPTION_VLM_MODEL            参考字幕识别用视觉模型(默认 as_core.VISION_MODEL)
  WHQ_CAPTION_LLM_MODEL            拆块/挑花字用文本模型(默认网关默认模型)
  WHQ_CAPTION_PROFILE              直接指定参考 style_profile.json(跳过分析)
  WHQ_CAPTION_PROFILE_DIR          参考风格缓存目录(跨任务共享, 推荐)
  WHQ_CAPTION_FONT                 ASS 主字体(默认 Noto Sans CJK SC)
  WHQ_CAPTION_REVEAL=block|char    字幕揭示方式(默认 block 整块一次弹出; char 逐字揭示)
  WHQ_CAPTION_ANIM=off|on          入场动画(默认 off 直接展示无缩放淡入; on 弹入/放大镜/回弹)
  WHQ_CAPTION_HOT_SCALE=1.25       块内关键词相对正文的放大倍数(1.0=只换色不变大)
  WHQ_CAPTION_TYPO_FIX=1           同音错字纠错开关(0 关闭; 只接受等长改写, 不会动时间轴)
  WHQ_CAPTION_POS=bottom|mixed     位置(默认 bottom 全部底部居中; mixed 彩色大字上顶部)
"""
import os

from . import ass_burn, charstream, profile_cache, target_match


def backend():
    """当前字幕后端: "" (关闭) / "ass" / "meishe"。"""
    v = (os.getenv("WHQ_CAPTION_CLONE") or "ass").strip().lower()
    if v in ("0", "off", "false", "no", ""):
        return ""
    if v in ("1", "on", "true", "ass"):
        return "ass"
    if v == "meishe":
        return "meishe"
    print("[captions_clone] 未知 WHQ_CAPTION_CLONE={}, 按 ass 处理".format(v), flush=True)
    return "ass"


def burn_styled_captions(video, tts_items, out_video, work_dir, ref_video=None):
    """成片 + tts plan + 参考视频 -> 烧上参考风格字幕的成片。失败返回 None。

    ``tts_items`` 为空时(比如前端「字幕特效模仿」直接对任意成片跑, 手上没有配音 plan),
    退化为纯按成片 ASR 切句 —— 字幕文本来自 ASR 识别结果, 可能有同音错字。
    """
    mode = backend()
    if not mode:
        return None
    os.makedirs(work_dir, exist_ok=True)
    llm_model = os.getenv("WHQ_CAPTION_LLM_MODEL") or None
    try:
        profile = profile_cache.analyze_reference(ref_video, work_dir)
        stream = charstream.char_stream(video, os.path.join(work_dir, "final_asr"))
        if tts_items:
            lines = charstream.build_lines(tts_items, stream)
        else:
            print("[captions_clone] 无 tts plan, 字幕文本改用成片 ASR", flush=True)
            lines = charstream.lines_from_stream(stream)
        if not lines:
            print("[captions_clone] 无可用念白句, 跳过", flush=True)
            return None
        if os.getenv("WHQ_CAPTION_TYPO_FIX", "1").strip().lower() not in ("0", "off", "false", "no"):
            target_match.correct_typos(lines, model=llm_model)
        seq, by_role = target_match.plan(lines, profile, model=llm_model)
        if not seq:
            print("[captions_clone] 字幕块为空, 跳过", flush=True)
            return None
        target_match.dump_md(
            seq, by_role, os.path.join(work_dir, "目标字幕清单.md"),
            os.path.basename(out_video), os.path.basename(ref_video or "-"))

        if mode == "meishe":
            from . import meishe
            done = meishe.burn(seq, video, out_video, work_dir, model=llm_model)
            if done:
                return done
            print("[captions_clone] 美摄后端未出片, 回退本机 ASS 烧录", flush=True)
        return ass_burn.burn(seq, video, out_video, work_dir)
    except Exception as exc:  # noqa: BLE001
        print("[captions_clone] 字幕特效复刻失败(回退句级 SRT): {}".format(str(exc)[:300]),
              flush=True)
        return None
