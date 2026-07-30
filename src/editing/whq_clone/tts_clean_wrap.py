"""tts_clean_wrap — CosyVoice3 合成的**薄封装**: 先跑真实合成脚本, 再修掉每段 wav 的
起点伪声, 让**每段原始片段本身**就干净(不止最终成片干净)。

背景(实测逐帧频带诊断): CosyVoice3 每段真语音之前会先合成一段**低频哼鸣伪声**
(实测 ~150..470ms, 基频~88Hz 有调性, 能量集中在 50..300Hz, >=800Hz 频带几乎无能量,
听感"嗡/哼"; 真语音的辅音/元音共振峰必有 >=800Hz 能量, 在哼鸣之后才起来)。
早期版本只静音到能量"谷底"(~140ms), 谷底之后的哼鸣整段漏过, 用户仍能听到杂音。

做法(additive, 不改主流水线): 把 build_tts_overlay 的 TTS_SCRIPT 指到本封装, 本封装
  1) 用 WHQ_REAL_TTS_SCRIPT 跑真实 CosyVoice 合成到 --output
  2) 用高频带(>=800Hz)能量定位**真语音起点**, 确认起点前确有"低频响+高频静"的哼鸣帧,
     则把 [0, 起点-40ms] 整段静音 + hsin 升余弦淡入真语音; 检测不到伪声时退回 8ms 淡入。
     尾部再做 hsin 淡出消 offset; 采样率/时长不变。
之后 build_tts_overlay 的 atempo 适配/adelay 拼接消费到的就是干净片段。
"""
import argparse
import os
import subprocess
import sys

FFMPEG = os.getenv("FFMPEG", "ffmpeg")
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")
ONSET_FADE = float(os.getenv("WHQ_TTS_ONSET_FADE", "0.008"))
# CosyVoice3 输出偏闷(实测 95% 能量滚降仅 ~1.7..2.8kHz, 3..6kHz 辅音清晰度频带比
# 克隆参考低 ~7dB), 听感"人声不清晰"。置一个温和高频搜架提升 presence;
# 置空字符串可关闭。
CLARITY_EQ = os.getenv("WHQ_TTS_CLARITY_EQ", "highshelf=f=2500:g=3")


def _duration(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _sample_rate(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True).stdout.strip()
        return int(out)
    except Exception:
        return 24000


def _lead_mute_time(wav_path):
    """检测开头的伪声(宽带冲头 + 低频哼鸣), 返回应静音到的时刻(秒); 无伪声返回 0.0。

    实测(多次合成逐帧频带诊断): CosyVoice3 真语音之前会先吐一段伪声, 结构为
    [近满幅宽带冲头 ~0..120ms] + [低频哼鸣 ~120..470ms(基频~88Hz, 50..300Hz 响且
    >=800Hz 几乎无能量)] → 真语音。冲头含高频能量, 会被"高频带=语音"的判据误认,
    故以**哼鸣段本身**为锚: 找起始<=250ms、总长>=90ms 的连续哼鸣段(低频响 lo>-20dB
    且高频静 hi<-26dB, 容3帧间断), 静音到哼鸣段末尾+1帧(=真语音起点); 冲头在哼鸣之前
    自然被覆盖。另要求开头 120ms 内确有高幅头(max lo>-12dB), 避免把真语音的鼻音误判。
    """
    try:
        import math
        import numpy as np
        raw = subprocess.run(
            [FFMPEG, "-v", "error", "-i", wav_path, "-f", "f32le", "-ac", "1",
             "-ar", "24000", "-"],
            check=True, capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        sr = 24000
        hop = int(0.010 * sr)
        win = int(0.020 * sr)
        n = min((len(a) - win) // hop, 100)  # 只看前 1s
        if n < 12:
            return 0.0
        hi_db, lo_db = [], []
        for i in range(n):
            w = a[i * hop:i * hop + win].astype(np.float64)
            sp = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
            freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
            hi = math.sqrt(float(sp[freqs >= 800].sum()) / len(w)) + 1e-12
            lo = math.sqrt(float(sp[(freqs >= 50) & (freqs < 300)].sum()) / len(w)) + 1e-12
            hi_db.append(20 * math.log10(hi))
            lo_db.append(20 * math.log10(lo))
        # 1) 伪声总以近满幅冲头开始: 开头 120ms 内低频带必须响
        if max(lo_db[:12]) <= -12.0:
            return 0.0
        # 2) 找哼鸣段: lo>-20 且 hi<-26, 起始<=250ms, 容 3 帧间断, 总长>=9帧(90ms)
        hum = [i for i in range(n) if lo_db[i] > -20.0 and hi_db[i] < -26.0]
        if not hum or hum[0] * 0.010 > 0.25:
            return 0.0
        run_end, count, prev = hum[0], 1, hum[0]
        for i in hum[1:]:
            if i - prev <= 3:
                run_end, count, prev = i, count + 1, i
            else:
                break
        if count < 9:
            return 0.0
        return min((run_end + 1) * 0.010, 0.8)
    except Exception:
        pass
    return 0.0


def _repair_onset(wav_path, fade=ONSET_FADE):
    """就地: 静音开头低频哼鸣([0,语音起点-40ms]) + hsin 淡入真语音 + 尾部 hsin 淡出。"""
    dur = _duration(wav_path)
    if dur <= 2 * fade:
        return
    sr = _sample_rate(wav_path)
    mute_to = _lead_mute_time(wav_path)
    # 检测到伪声: 静音到语音起点前 + 40ms 淡入; 否则退回起点 8ms 淡入
    st_in = mute_to if mute_to > 0.0 else 0.0
    d_in = 0.030 if mute_to > 0.0 else fade
    if st_in + d_in >= dur - fade:  # 兜底: 别把整段吃掉
        st_in, d_in = 0.0, fade
    tmp = wav_path + ".clean.wav"
    af = ("afade=t=in:curve=hsin:st={si:.3f}:d={di},"
          "afade=t=out:curve=hsin:st={fo:.3f}:d={d}").format(
        si=st_in, di=d_in, d=fade, fo=max(0.0, dur - fade))
    if CLARITY_EQ:
        af += "," + CLARITY_EQ
    subprocess.run([FFMPEG, "-y", "-i", wav_path, "-af", af, "-ar", str(sr),
                    "-c:a", "pcm_s16le", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, wav_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args, _ = ap.parse_known_args()
    # whq 位于 generation/whq/ 下, 其父目录即 generation/, 真实合成脚本与 whq 同级。
    real = os.getenv("WHQ_REAL_TTS_SCRIPT",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "run_cosyvoice3_zero_shot.py"))
    # 1) 跑真实合成(透传全部原始参数)。真实脚本可能需要独立解释器(如 VoxCPM2 在 voxcpm env),
    #    用 WHQ_REAL_TTS_PYTHON 指定; 默认沿用当前解释器(CosyVoice3 在 viral-split-tts env)。
    real_py = os.getenv("WHQ_REAL_TTS_PYTHON", sys.executable)
    subprocess.run([real_py, real] + sys.argv[1:], check=True)
    # 2) 修掉起点低频哼鸣伪声 + 尾部 offset
    if os.path.exists(args.output):
        try:
            _repair_onset(args.output)
        except Exception as exc:  # 修不动也保底给出原片, 不阻断流程
            print("[tts_clean_wrap] onset repair skipped: {}".format(str(exc)[:160]), flush=True)


if __name__ == "__main__":
    main()
