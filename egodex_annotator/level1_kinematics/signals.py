"""Level 1 — per-frame kinematic signals from world-frame hand/head poses.

All signals are computed in the world frame (confirmed by Level 0), with
short invalid-tracking gaps interpolated and longer gaps masked.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import joint_validity  # noqa: E402


def _interp_gaps(pos, valid, max_gap):
    """Linearly interpolate invalid runs up to max_gap frames.

    Returns (filled positions, mask of frames still invalid)."""
    pos = pos.copy()
    still_invalid = np.zeros(len(pos), dtype=bool)
    if valid.all():
        return pos, still_invalid
    idx = np.arange(len(pos))
    # find contiguous invalid runs
    runs = []
    start = None
    for i, v in enumerate(valid):
        if not v and start is None:
            start = i
        elif v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))

    for s, e in runs:
        gap = e - s
        interpolable = gap <= max_gap and s > 0 and e < len(pos)
        if interpolable:
            for c in range(pos.shape[1]):
                pos[s:e, c] = np.interp(idx[s:e], [s - 1, e], [pos[s - 1, c], pos[e, c]])
        else:
            still_invalid[s:e] = True
    return pos, still_invalid


def _smooth(x, window, poly):
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window <= poly:
        return x
    return savgol_filter(x, window, poly)


def _speed(pos, fps, window, poly):
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    v = np.concatenate([[v[0]], v])  # keep length N (prepend first value)
    return _smooth(v, window, poly)


def _angular_speed(R, fps, window, poly):
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    w = np.arccos(tr) * fps
    w = np.concatenate([[w[0]], w])
    return _smooth(w, window, poly)


def compute_signals(h5_path, cfg):
    """Return dict of per-frame signals plus validity masks for one episode."""
    k = cfg["kinematics"]
    h = cfg["hdf5"]
    fps = cfg["video"]["fps"]
    win, poly = k["savgol_window"], k["savgol_poly"]
    conf_min = h["confidence_valid_min"]

    with h5py.File(h5_path, "r") as f:
        n = f[h["camera_pose"]].shape[0]
        out = {"n_frames": n, "fps": fps, "t": np.arange(n) / fps}
        masked_any = np.zeros(n, dtype=bool)

        for side in ("left", "right"):
            wrist_key = h["wrist"][side]
            thumb_key, index_key = h["pinch_pair"][side]

            wrist = f[wrist_key][:, :3, 3].astype(np.float64)
            thumb = f[thumb_key][:, :3, 3].astype(np.float64)
            index = f[index_key][:, :3, 3].astype(np.float64)

            valid = (joint_validity(f, wrist_key, conf_min)
                     & joint_validity(f, thumb_key, conf_min)
                     & joint_validity(f, index_key, conf_min))

            wrist, bad_w = _interp_gaps(wrist, valid, k["max_gap_interp"])
            thumb, bad_t = _interp_gaps(thumb, valid, k["max_gap_interp"])
            index, bad_i = _interp_gaps(index, valid, k["max_gap_interp"])
            masked = bad_w | bad_t | bad_i
            masked_any |= masked

            aperture = _smooth(np.linalg.norm(thumb - index, axis=1), win, poly)
            out[f"speed_{side}"] = _speed(wrist, fps, win, poly)
            out[f"aperture_{side}"] = aperture
            out[f"aperture_vel_{side}"] = np.gradient(aperture) * fps
            out[f"masked_{side}"] = masked
            out[f"wrist_pos_{side}"] = wrist  # world frame, gaps interpolated

        cam = f[h["camera_pose"]][:].astype(np.float64)
        out["head_rot_speed"] = _angular_speed(cam[:, :3, :3], fps, win, poly)

    out["speed_combined"] = np.maximum(out["speed_left"], out["speed_right"])
    out["masked"] = masked_any
    out["masked_regions"] = _mask_regions(masked_any)
    return out


def _mask_regions(masked):
    """Contiguous masked frame ranges as [start, end) pairs."""
    regions = []
    start = None
    for i, m in enumerate(masked):
        if m and start is None:
            start = i
        elif not m and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(masked)))
    return regions
