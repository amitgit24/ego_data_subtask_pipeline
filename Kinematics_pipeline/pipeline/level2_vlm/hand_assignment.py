"""Level 2 — hand assignment computed from Level 1 kinematics, never by the
VLM. The VLM receives the result as context and may only flag disagreement."""

import numpy as np


def _activity(sig, side, s, e, ha_cfg):
    """(normalized score, raw-activity flag) for one hand inside [s, e).

    The normalized score (episode-median units) ranks hands against each
    other; the raw flag decides whether a hand moved at all — otherwise an
    idle hand's tracking noise, divided by its near-zero episode median,
    outscores the hand doing the work."""
    speed = sig[f"speed_{side}"]
    apv = np.abs(sig[f"aperture_vel_{side}"])
    raw_speed = float(speed[s:e].mean())
    raw_apv = float(apv[s:e].mean())
    score = (raw_speed / max(np.median(speed), 0.01)
             + raw_apv / max(np.median(apv), 0.005))
    active = (raw_speed > ha_cfg["active_speed_min"]
              or raw_apv > ha_cfg["active_apv_min"])
    return float(score), active


def _is_handover(events, sig, ha_cfg, fps):
    """Grasp on one hand within handover_window_s of a release on the other,
    AND the wrists near each other at that moment — passing an object between
    hands physically requires the hands to meet. Without the distance gate,
    fast alternating bimanual work (one hand sets an object down while the
    other grabs the next one) fires constantly: measured on add_remove_lid,
    21/21 time-only "handover" pairs had the wrists 0.24-0.55 m apart."""
    win = ha_cfg["handover_window_s"] * fps
    max_dist = ha_cfg["handover_max_wrist_dist"]
    d = np.linalg.norm(sig["wrist_pos_left"] - sig["wrist_pos_right"], axis=1)
    for g in (e for e in events if e["type"] == "grasp"):
        for r in (e for e in events if e["type"] == "release"):
            if g["hand"] != r["hand"] and abs(g["frame"] - r["frame"]) <= win:
                if d[(g["frame"] + r["frame"]) // 2] <= max_dist:
                    return True
    return False


def _is_joint_carry(sig, s, e, ha_cfg):
    """Inter-wrist distance stays within +-tol of its segment mean for more
    than wrist_dist_frac of the segment (jointly moving one object)."""
    d = np.linalg.norm(sig["wrist_pos_left"][s:e] - sig["wrist_pos_right"][s:e],
                       axis=1)
    mean = d.mean()
    if mean < 1e-6:
        return False
    frac = (np.abs(d - mean) < ha_cfg["wrist_dist_tol"] * mean).mean()
    return frac > ha_cfg["wrist_dist_frac"]


def assign_hand(sig, segment, cfg):
    """Return {'hand', 'score_left', 'score_right', 'coordination'} for one
    Level 1 segment."""
    ha = cfg["hand_assignment"]
    s, e = segment["start_frame"], segment["end_frame"] + 1
    events = segment["events_inside"]

    score_l, active_l = _activity(sig, "left", s, e, ha)
    score_r, active_r = _activity(sig, "right", s, e, ha)

    coordination = None
    if active_l and not active_r:
        hand = "left"
    elif active_r and not active_l:
        hand = "right"
    elif not active_l and not active_r:
        hand = "left" if score_l >= score_r else "right"
    elif score_l > ha["dominance_ratio"] * score_r:
        hand = "left"
    elif score_r > ha["dominance_ratio"] * score_l:
        hand = "right"
    else:
        hand = "both"
        if _is_handover(events, sig, ha, sig["fps"]):
            hand, coordination = "both_coordinating", "handover"
        elif _is_joint_carry(sig, s, e, ha):
            hand, coordination = "both_coordinating", "joint_carry"

    return {"hand": hand, "score_left": round(score_l, 3),
            "score_right": round(score_r, 3),
            "active_left": active_l, "active_right": active_r,
            "coordination": coordination}


def kinematic_summary(sig, segment, hand_info, fps):
    """Human-readable per-segment kinematics for the VLM prompt."""
    s, e = segment["start_frame"], segment["end_frame"] + 1
    parts = [f"Active hand (computed from motion): {hand_info['hand']}"
             + (f" ({hand_info['coordination']})"
                if hand_info["coordination"] else "") + "."]
    for side in ("left", "right"):
        path = np.linalg.norm(
            np.diff(sig[f"wrist_pos_{side}"][s:e], axis=0), axis=1).sum()
        evs = [ev for ev in segment["events_inside"] if ev["hand"] == side]
        bits = []
        if path < 0.05:
            bits.append("mostly static")
        else:
            bits.append(f"moved {path:.2f}m")
        for ev in evs:
            bits.append(f"{ev['type']} event at {ev['frame'] / fps:.1f}s")
        parts.append(f"{side.capitalize()} hand: {', '.join(bits)}.")
    return " ".join(parts)
