"""Level 2 — keyframe selection and seek-based extraction per segment.

Simpler than Kinematics_pipeline's version: there are no grasp/release
events to prioritize (Level 1 here has no event detector, only the
transition itself, which is already the segment's start/end), so keyframes
are just evenly spaced between start and end. `extract_keyframes` is a
small, self-contained duplicate of Kinematics_pipeline's version — not
worth cross-importing for something this size and dataset/pose-agnostic.
"""

from pathlib import Path

import cv2
import numpy as np


def select_keyframes(segment, max_frames):
    """Return [(frame, tag)], always including start and end."""
    s, e = segment["start_frame"], segment["end_frame"]
    if max_frames <= 2 or e <= s:
        return [(s, "segment start"), (e, "segment end")]
    mids = np.linspace(s, e, max_frames)[1:-1].round().astype(int)
    chosen = {s: "segment start", e: "segment end"}
    for i, f in enumerate(mids):
        chosen.setdefault(int(f), f"middle {i + 1}/{len(mids)}")
    return sorted(chosen.items())


def extract_keyframes(mp4_path, frame_tags, long_side, out_dir, seg_id):
    """Seek-decode the selected frames, downscale, save JPEGs.

    Returns [(frame, tag, path)]. Raises on any decode failure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    results = []
    for frame, tag in frame_tags:
        out = out_dir / f"seg{seg_id:02d}_f{frame:05d}.jpg"
        if not out.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"failed to decode frame {frame} of {mp4_path}")
            h, w = img.shape[:2]
            scale = long_side / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (round(w * scale), round(h * scale)))
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        results.append((frame, tag, str(out)))
    cap.release()
    return results
