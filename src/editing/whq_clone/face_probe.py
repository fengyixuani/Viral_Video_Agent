#!/usr/bin/env python3
"""face_probe — 用 YuNet(onnx) 探测视频窗口内是否有人脸。

独立小脚本: whq 主流程跑在无 cv2 的环境里, 本脚本由 voiceover._probe_faces
以子进程方式用带 cv2 的解释器(env WHQ_CV_PYTHON, 默认 /root/miniconda3/bin/python3.13)
调用。只输出一行 JSON: {sampled, face_frames, face_ratio, max_scores}。
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--score", type=float, default=0.7)
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    full_dur = (total_frames / fps) if fps > 0 else 0.0
    start = max(0.0, args.start)
    dur = args.duration if args.duration > 0 else max(0.0, full_dur - start)

    det = None
    sampled = 0
    face_frames = 0
    max_scores = []
    for i in range(max(1, args.frames)):
        t = start + dur * (i + 0.5) / max(1, args.frames)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        sampled += 1
        h, w = frame.shape[:2]
        # YuNet 对小图更快且足够准; 长边压到 480
        if max(h, w) > 480:
            scale = 480.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]
        if det is None:
            det = cv2.FaceDetectorYN.create(args.model, "", (w, h), args.score)
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is not None and len(faces) > 0:
            face_frames += 1
            max_scores.append(round(float(max(f[-1] for f in faces)), 3))
    cap.release()
    print(json.dumps({
        "sampled": sampled,
        "face_frames": face_frames,
        "face_ratio": round(face_frames / sampled, 3) if sampled else 0.0,
        "max_scores": max_scores,
    }))


if __name__ == "__main__":
    main()
