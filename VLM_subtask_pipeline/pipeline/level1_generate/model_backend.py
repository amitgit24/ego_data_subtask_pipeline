"""Qwen3-VL model loading + video probing for the single-pass subtask pipeline.

Identical to VLM_seg_subtask_pipeline's model_backend.py — the same
AutoModelForImageTextToText / AutoProcessor pattern verified working against
this model path in this environment. The model is loaded once and reused for
every window of every episode.
"""

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def load_model(model_path, attn_implementation="sdpa", dtype="bfloat16"):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=_DTYPES[dtype], device_map="auto",
        attn_implementation=attn_implementation)
    model.eval()
    return model, processor


def probe_video(video_path, fps_fallback=30.0):
    """(duration_sec, fps) for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
    cap.release()
    if n <= 0 or fps <= 0:
        raise RuntimeError(f"invalid probe result for {video_path}: "
                           f"{n} frames @ {fps} fps")
    return n / fps, fps
