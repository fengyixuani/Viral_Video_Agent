#!/usr/bin/env python3
"""run_voxcpm2_zero_shot — VoxCPM2 零样本声音克隆, CLI 与 run_cosyvoice3_zero_shot.py 对齐,
可作为 whq 的 WHQ_REAL_TTS_SCRIPT 直接替换 CosyVoice3。

与 CosyVoice3 路径一致的步骤:
  1) 读 prompt_asr.json 的转写作为参考音文本(ultimate cloning: 参考音 + 其转写, 相似度最高);
  2) 用 prompt.wav 做参考音克隆用户音色, 念 --text/--text-file 的目标文本;
  3) 输出到 --output。之后由 whq 的 tts_clean_wrap 统一做起点伪声修复 + clarity EQ(与 CosyVoice 同款净化)。

注意: 本机 torchcodec 与 cu128 不兼容, 故 load_denoiser=False(不走 zipenhancer/torchaudio.load),
仅用生成核心 + soundfile 落盘, 规避 libtorchcodec 加载失败。
"""
import argparse
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import soundfile as sf
from voxcpm import VoxCPM

MODEL_ID = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
CFG_VALUE = float(os.getenv("WHQ_VOXCPM_CFG", "2.0"))
STEPS = int(os.getenv("WHQ_VOXCPM_STEPS", "30"))
NORMALIZE = os.getenv("WHQ_VOXCPM_NORMALIZE", "1") not in ("0", "false", "False")


def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 zero-shot voice cloning (CosyVoice-CLI compatible).")
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-asr", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    # 兼容 CosyVoice CLI 的多余参数(忽略)
    args, _ = ap.parse_known_args()

    text = args.text
    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read().strip()
    if not text:
        raise SystemExit("empty synthesis text")

    prompt_text = ""
    try:
        asr = json.load(open(args.prompt_asr, encoding="utf-8"))
        prompt_text = str(asr["results"][0]["text"] or "").strip()
    except Exception as exc:  # noqa: BLE001 —— 无转写则退化为基础克隆
        print("[voxcpm] prompt_asr 读取失败, 退化基础克隆: {}".format(str(exc)[:120]), flush=True)

    # 高保真参考音覆盖: whq 默认 prompt 是 16k(为 CosyVoice/ASR 服务), 带宽封顶会让 VoxCPM(48k)
    # 输出发闷。用 WHQ_VOXCPM_REF_WAV 指向 44.1k 干净参考(如 demucs 纯人声/从源 .mov 重抽),
    # 并用 WHQ_VOXCPM_REF_TEXT 给其精确转写(ultimate cloning)。未设置则回退流水线 16k prompt。
    ref_wav = args.prompt_wav
    ref_text = prompt_text
    ov_wav = os.getenv("WHQ_VOXCPM_REF_WAV", "").strip()
    if ov_wav and os.path.exists(ov_wav):
        ref_wav = ov_wav
        ref_text = os.getenv("WHQ_VOXCPM_REF_TEXT", "").strip() or prompt_text
        print("[voxcpm] 使用高保真参考音: {}".format(ref_wav), flush=True)

    model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False)
    sr = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)

    kwargs = dict(text=text, reference_wav_path=ref_wav,
                  cfg_value=CFG_VALUE, inference_timesteps=STEPS, normalize=NORMALIZE)
    if ref_text:  # ultimate cloning: 同段做 reference + prompt + 转写
        kwargs["prompt_wav_path"] = ref_wav
        kwargs["prompt_text"] = ref_text

    wav = model.generate(**kwargs)
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 可选把 VoxCPM 原生 48k 先降到指定采样率(如 24000)再落盘, 之后由 build_tts_overlay 重采样进
    # 44.1k 混音。降到 24k 会低通到 12kHz, 可切掉最高频段的伪声/杂音(折衷: 保留比 16k 更多的高频)。
    out_sr = int(os.getenv("WHQ_VOXCPM_OUTPUT_SR", str(sr)))
    if out_sr and out_sr != sr:
        import subprocess
        tmp = out + ".48k.wav"
        sf.write(tmp, wav, sr)
        ffmpeg = os.getenv("FFMPEG", "ffmpeg")
        subprocess.run([ffmpeg, "-y", "-i", tmp, "-ar", str(out_sr), "-ac", "1", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp)
        print("[voxcpm] 输出重采样 {}Hz -> {}Hz".format(sr, out_sr), flush=True)
    else:
        sf.write(out, wav, sr)
    print(out)


if __name__ == "__main__":
    main()
