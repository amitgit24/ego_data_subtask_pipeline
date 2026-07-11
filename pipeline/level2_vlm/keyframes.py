"""Level 2 — keyframe selection and seek-based extraction per segment.

Up to keyframes_per_segment frames: segment start, end, grasp/release event
frames (most informative instants), then the temporal midpoint if room.
Only the selected frames are decoded from the MP4.
"""

from pathlib import Path

import cv2


def select_keyframes(segment, max_frames):
    """Return [(frame, tag)] sorted by frame, capped at max_frames.

    Priority: start, end, event frames (largest aperture-velocity magnitude
    first), midpoint. Matches the spec's 'first, last, midpoint + events'
    with events outranking the midpoint when space runs out."""
    s, e = segment["start_frame"], segment["end_frame"]
    chosen = {s: "segment start", e: "segment end"}

    events = sorted(segment["events_inside"],
                    key=lambda ev: -ev.get("magnitude", 0.0))
    for ev in events:
        if len(chosen) >= max_frames:
            break
        chosen.setdefault(ev["frame"], f"{ev['type']} event ({ev['hand']} hand)")

    mid = (s + e) // 2
    if len(chosen) < max_frames:
        chosen.setdefault(mid, "middle")

    return sorted(chosen.items())


def extract_keyframes(mp4_path, frame_tags, long_side, out_dir, seg_id):
    """Seek-decode the selected frames, downscale, save JPEGs.

    Returns [(frame, tag, path)]. Raises on any decode failure (fail loudly).
    """
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
