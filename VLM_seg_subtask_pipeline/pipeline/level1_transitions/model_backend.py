"""Qwen3-VL model loading + video probing for the Qwen-direct pipeline.

Self-contained replacement for the external robot_labeling/process_native.py
and process.py helpers referenced in ../../ref_code/segment_transitions.py
(that path is machine-specific and does not exist in this environment).
Uses the same AutoModelForImageTextToText / AutoProcessor loading pattern
already verified working in this environment against this exact model path
(see Kinematics_pipeline's TransformersBackend and the initial env check
in egodex-study-setup memory).

Both Level 1 (transition detection) and Level 2 (segment labeling) load the
model through this module so run_pipeline.py can load it ONCE and reuse it
across both levels in one process, instead of paying the ~7 minute weight
load twice.
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
    """(duration_sec, fps) for a video file — replacement for the external
    process.probe_video referenced in ref_code."""
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
