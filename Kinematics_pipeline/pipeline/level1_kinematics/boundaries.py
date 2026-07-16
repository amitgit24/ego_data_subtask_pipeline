"""Level 1 — candidate subtask boundaries from kinematic signals.

Three detectors (pause > grasp/release > gaze in the merge priority given by
the spec: grasp/release > pause > gaze), fused, minimum-length enforced.
The VLM never sees this code; it only consumes the resulting segments.

CLI:
    python boundaries.py [--config ../config.yaml] --episodes task/idx [...]
    python boundaries.py --sample 5          # deterministic sample
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (apply_task_overrides, episode_id, episode_slug,  # noqa: E402
                    load_config, sample_episodes)
from signals import compute_signals  # noqa: E402

# release outranks grasp: when a release (end of action N) and a grasp
# (start of action N+1) collide inside merge_window/min_segment_frames, the
# cut belongs at the release — the hand's retraction toward the next object
# is part of the NEXT action cycle, not the one just finished (human review
# 2026-07-15: 12/17 boundary errors were retraction glued to the previous
# segment because the grasp edge won the collision instead)
PRIORITY = {"release": 4, "grasp": 3, "pause": 2, "gaze": 1}


def detect_events(sig, cfg):
    """Grasp/release events from aperture velocity, per hand (pre-fusion).

    Kept separately from boundaries: Level 2 uses them for keyframes and
    hand assignment even when the event did not survive boundary fusion."""
    k = cfg["kinematics"]
    # An aperture-opening peak is only a RELEASE if the wrist is slow at
    # that moment: letting go of a placed object happens with the hand
    # decelerated at the placement point. The hand also opens mid-swing to
    # pre-shape for the next grasp — at high wrist speed — and treating
    # that as a release puts the boundary at the END of the retraction,
    # gluing the trace-back to the previous segment (add_remove_lid round-4
    # review; measured separation: true releases <= 0.18 m/s, pre-grasp
    # openings >= 0.30 m/s). Disabled unless release_speed_max is set.
    release_speed_max = k.get("release_speed_max")
    events = []
    for side in ("left", "right"):
        av = sig[f"aperture_vel_{side}"]
        speed = sig[f"speed_{side}"]
        # gate on THIS hand's fingertip validity only — the union mask
        # silenced one hand's real events whenever the other was occluded
        masked = sig[f"masked_pinch_{side}"]
        for kind, series in (("grasp", -av), ("release", av)):
            peaks, props = find_peaks(series, height=k["aperture_vel_thresh"])
            for p, h in zip(peaks, props["peak_heights"]):
                if masked[p]:
                    continue
                if (kind == "release" and release_speed_max
                        and speed[p] > release_speed_max):
                    continue
                events.append({"frame": int(p), "hand": side,
                               "type": kind, "magnitude": float(h)})
    events.sort(key=lambda e: e["frame"])
    return events


def detect_candidates(sig, events, cfg):
    """Union of pause, grasp/release and gaze-shift boundary candidates."""
    k = cfg["kinematics"]
    cands = []

    # 1. pause: local minima of combined speed — a wrist-only signal, so
    # only suppress where BOTH wrists are untracked (fingertip occlusion
    # during manipulation must not silence it; that is where placements
    # and their pauses happen)
    sc = sig["speed_combined"]
    wrists_gone = sig["masked_wrist_left"] & sig["masked_wrist_right"]
    minima, props = find_peaks(-sc, prominence=k["pause_prominence"])
    for m in minima:
        if sc[m] < k["pause_speed_max"] and not wrists_gone[m]:
            cands.append({"frame": int(m), "source": "pause",
                          "hand": None, "event_type": None,
                          "magnitude": float(-sc[m])})

    # 2. grasp/release events
    for e in events:
        cands.append({"frame": e["frame"], "source": e["type"],
                      "hand": e["hand"], "event_type": e["type"],
                      "magnitude": e["magnitude"]})

    # 3. gaze shift (secondary) — head pose only; hand-tracking masks are
    # irrelevant to it, so no mask gate (the camera transform has no
    # confidence channel and audited 0 NaNs)
    hr = sig["head_rot_speed"]
    thresh = max(np.percentile(hr, k["head_rot_percentile"]),
                 k.get("head_rot_min_rad_s", 0.25))
    peaks, props = find_peaks(hr, height=thresh)
    for p, h in zip(peaks, props["peak_heights"]):
        cands.append({"frame": int(p), "source": "gaze",
                      "hand": None, "event_type": None,
                      "magnitude": float(h)})

    cands.sort(key=lambda c: c["frame"])
    return cands


def fuse(cands, n_frames, cfg, speed_combined):
    """Merge nearby candidates, enforce min segment length and an actual
    movement burst between boundaries, add episode edges."""
    k = cfg["kinematics"]

    # merge candidates within merge_window, keep highest priority then magnitude
    merged = []
    for c in cands:
        if merged and c["frame"] - merged[-1]["frame"] <= k["merge_window"]:
            best = max(merged[-1], c, key=lambda x: (PRIORITY[x["source"]],
                                                     x["magnitude"]))
            merged[-1] = best
        else:
            merged.append(dict(c))

    # drop candidates hugging the episode edges (would violate min length)
    min_len = k["min_segment_frames"]
    interior = [c for c in merged
                if min_len <= c["frame"] <= n_frames - 1 - min_len]

    # enforce min segment length between interior boundaries
    def weakest(pair):
        return min(pair, key=lambda x: (PRIORITY[x["source"]], x["magnitude"]))

    changed = True
    while changed and len(interior) > 1:
        changed = False
        for a, b in zip(interior, interior[1:]):
            if b["frame"] - a["frame"] < min_len:
                interior.remove(weakest((a, b)))
                changed = True
                break

    # a boundary inside a fully static stretch separates nothing: drop it
    # when the hands never burst above move_speed_min on EITHER side
    # (typing/writing ripples); one moving side keeps it (reach -> slow action)
    move_min = k["move_speed_min"]

    def moved(f0, f1):
        return f1 - f0 < 2 or speed_combined[f0:f1].max() >= move_min

    changed = True
    while changed and interior:
        changed = False
        frames = [0] + [c["frame"] for c in interior] + [n_frames - 1]
        for i, c in enumerate(interior):
            prev_f, next_f = frames[i], frames[i + 2]
            if not moved(prev_f, c["frame"]) and not moved(c["frame"], next_f):
                interior.remove(c)
                changed = True
                break

    start = {"frame": 0, "source": "episode_start", "hand": None,
             "event_type": None, "magnitude": 0.0}
    end = {"frame": n_frames - 1, "source": "episode_end", "hand": None,
           "event_type": None, "magnitude": 0.0}
    return [start] + interior + [end]


def build_segments(boundaries, events, fps):
    segments = []
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        inside = [e for e in events if a["frame"] <= e["frame"] < b["frame"]]
        segments.append({
            "id": i,
            "start_frame": a["frame"],
            "end_frame": b["frame"],
            "start_time": round(a["frame"] / fps, 4),
            "end_time": round(b["frame"] / fps, 4),
            "duration": round((b["frame"] - a["frame"]) / fps, 4),
            "start_source": a["source"],
            "end_source": b["source"],
            "events_inside": inside,
        })
    return segments


def propose_boundaries(h5_path, cfg):
    """Full Level 1 result for one episode."""
    cfg = apply_task_overrides(cfg, h5_path.parent.name)
    sig = compute_signals(h5_path, cfg)
    events = detect_events(sig, cfg)
    cands = detect_candidates(sig, events, cfg)
    boundaries = fuse(cands, sig["n_frames"], cfg, sig["speed_combined"])
    segments = build_segments(boundaries, events, sig["fps"])
    return {
        "episode_id": episode_id(h5_path),
        "n_frames": sig["n_frames"],
        "fps": sig["fps"],
        "masked_regions": [list(r) for r in sig["masked_regions"]],
        "boundaries": [
            {"frame": b["frame"], "time": round(b["frame"] / sig["fps"], 4),
             "source": b["source"], "hand": b["hand"],
             "event_type": b["event_type"]}
            for b in boundaries
        ],
        "events": events,
        "segments": segments,
    }, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="episode ids like task_name/12")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="accepted for driver symmetry; boundaries always "
                         "recompute")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        paths = [data_dir / f"{e}.hdf5" for e in args.episodes]
    elif args.sample:
        paths = sample_episodes(data_dir, args.sample,
                                min_frames=8 * cfg["video"]["fps"])
    else:
        ap.error("give --episodes or --sample")

    for p in paths:
        result, _ = propose_boundaries(p, cfg)
        out = out_dir / f"{episode_slug(p)}.json"
        out.write_text(json.dumps(result, indent=2))
        n_seg = len(result["segments"])
        print(f"{result['episode_id']}: {result['n_frames']} frames -> "
              f"{n_seg} segments -> {out}")


if __name__ == "__main__":
    main()
