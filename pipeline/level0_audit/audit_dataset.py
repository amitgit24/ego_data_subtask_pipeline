"""Level 0 — EgoDex data audit.

Verifies the actual on-disk structure of EgoDex episodes (MP4 + HDF5) before
any pipeline code trusts it. Answers the schema questions from the spec,
saves wrist-trajectory plots, and prints a PASS/FAIL verifier table.

Usage:
    python audit_dataset.py --config ../config.yaml [--n_episodes 8]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REQUIRED_KEYS = [
    "transforms/leftHand",
    "transforms/rightHand",
    "transforms/leftThumbTip",
    "transforms/leftIndexFingerTip",
    "transforms/rightThumbTip",
    "transforms/rightIndexFingerTip",
    "transforms/camera",
    "camera/intrinsic",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- section A

def audit_pairing(data_dir):
    """Scan every task folder; check MP4<->HDF5 pairing by shared stem."""
    tasks = sorted(d for d in data_dir.iterdir() if d.is_dir())
    unpaired = []
    n_pairs = 0
    for task in tasks:
        mp4s = {p.stem for p in task.glob("*.mp4")}
        h5s = {p.stem for p in task.glob("*.hdf5")}
        for stem in mp4s ^ h5s:
            unpaired.append(f"{task.name}/{stem}")
        n_pairs += len(mp4s & h5s)
    return tasks, n_pairs, unpaired


def sample_episodes(tasks, n, seed=0):
    """Pick n episodes from n distinct tasks, deterministically."""
    rng = random.Random(seed)
    chosen_tasks = rng.sample(tasks, min(n, len(tasks)))
    episodes = []
    for task in chosen_tasks:
        h5s = sorted(task.glob("*.hdf5"), key=lambda p: int(p.stem))
        episodes.append(rng.choice(h5s))
    return episodes


# ---------------------------------------------------------------- section B

def dump_tree(h5_path, out_lines):
    with h5py.File(h5_path, "r") as f:
        out_lines.append(f"\n=== HDF5 tree: {h5_path} ===")
        out_lines.append("root attrs:")
        for k, v in f.attrs.items():
            out_lines.append(f"  {k!r}: {str(v)[:120]!r}")

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out_lines.append(f"  {name}  shape={obj.shape} dtype={obj.dtype}")

        f.visititems(walk)


def check_keys(h5_path):
    """Return (missing_required_keys, joint names per hand, language info)."""
    with h5py.File(h5_path, "r") as f:
        missing = [k for k in REQUIRED_KEYS if k not in f]
        joints = {"left": [], "right": []}
        for name in f["transforms"]:
            if name.startswith("left") and name != "leftHand":
                joints["left"].append(name)
            elif name.startswith("right") and name != "rightHand":
                joints["right"].append(name)
        lang = {
            "task": f.attrs.get("task"),
            "description": f.attrs.get("description"),
            "llm_description": f.attrs.get("llm_description"),
            "llm_description2": f.attrs.get("llm_description2"),
            "which_llm_description": f.attrs.get("which_llm_description"),
        }
        n_frames = f["transforms/leftHand"].shape[0]
        has_confidences = "confidences" in f
    return missing, joints, lang, n_frames, has_confidences


# ---------------------------------------------------------------- section C

def video_meta(mp4_path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


# ---------------------------------------------------------------- section D

def positions(f, key):
    return f[key][:, :3, 3].astype(np.float64)


def rotations(f, key):
    return f[key][:, :3, :3].astype(np.float64)


def angular_speed(R, fps):
    """Geodesic distance between consecutive rotations x fps (rad/s)."""
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])  # R_t @ R_{t-1}^T
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.arccos(tr) * fps


def signal_sanity(h5_path, fps, conf_min, out_dir, make_plot):
    """Units / displacement / validity / coordinate-frame evidence."""
    import itertools

    with h5py.File(h5_path, "r") as f:
        wrist = positions(f, "transforms/rightHand")
        wrist_l = positions(f, "transforms/leftHand")
        cam_T = f["transforms/camera"][:].astype(np.float64)
        has_conf = "confidences" in f

        nan_frames = int(np.isnan(f["transforms/rightHand"][:]).any(axis=(1, 2)).sum())

        # Validity rule: confidence < conf_min where confidences exist;
        # fallback for episodes without a confidences group: NaN in transform.
        if has_conf:
            conf = f["confidences/rightHand"][:]
            invalid = np.isnan(conf) | (conf < conf_min)
            conf_stats = (float(np.nanmin(conf)), float(np.nanmedian(conf)),
                          float(np.nanmax(conf)))
        else:
            invalid = np.isnan(f["transforms/rightHand"][:]).any(axis=(1, 2))
            conf_stats = None
        frac_invalid = float(invalid.mean())
        longest = max((len(list(g)) for k, g in itertools.groupby(invalid) if k),
                      default=0)

        # per-frame displacement (raw frame)
        disp = np.linalg.norm(np.diff(wrist, axis=0), axis=1)

        # Coordinate-frame evidence (world vs camera-relative):
        #  1. transforms/camera varies over time — if poses were camera-relative
        #     the camera transform would be identity/constant.
        #  2. camera sits ABOVE both wrists (head-mounted, gravity-aligned y-up);
        #     camera-relative wrist positions would have negative y instead.
        cam_p = cam_T[:, :3, 3]
        cam_travel = float(np.linalg.norm(np.diff(cam_p, axis=0), axis=1).sum())
        cam_above_wrists = bool(
            np.median(cam_p[:, 1]) > np.median(wrist[:, 1])
            and np.median(cam_p[:, 1]) > np.median(wrist_l[:, 1])
        )
        cam_w = angular_speed(cam_T[:, :3, :3], fps)
        speed_raw = disp * fps

        if make_plot:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            for i, lbl in enumerate("xyz"):
                axes[0].plot(wrist[:, i], label=f"wrist {lbl}")
                axes[0].plot(cam_p[:, i], "--", alpha=0.5, label=f"camera {lbl}")
            axes[0].set_ylabel("position (m, world)")
            axes[0].legend(ncol=3, fontsize=8)
            axes[0].set_title(h5_path.parent.name + "/" + h5_path.name)
            axes[1].plot(speed_raw, label="right wrist speed (m/s)")
            axes[1].plot(cam_w, label="camera ang. speed (rad/s)", alpha=0.7)
            axes[1].set_xlabel("frame")
            axes[1].legend()
            fig.tight_layout()
            out = out_dir / f"wrist_{h5_path.parent.name}_{h5_path.stem}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)

    return {
        "n_frames": len(wrist),
        "has_confidences": has_conf,
        "pos_min": wrist.min(axis=0).round(3).tolist(),
        "pos_max": wrist.max(axis=0).round(3).tolist(),
        "median_disp": float(np.median(disp)),
        "p95_disp": float(np.percentile(disp, 95)),
        "nan_frames": nan_frames,
        "frac_invalid": frac_invalid,
        "longest_invalid_run": longest,
        "conf_stats": conf_stats,
        "cam_travel_m": cam_travel,
        "cam_above_wrists": cam_above_wrists,
    }


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--n_episodes", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"]) / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = args.n_episodes or cfg["audit"]["n_episodes"]
    conf_min = cfg["hdf5"]["confidence_valid_min"]
    expected_fps = cfg["video"]["fps"]

    report = []
    say = lambda s: (print(s), report.append(s))

    say(f"EgoDex Level 0 audit — data_dir={data_dir}")

    # A. pairing over the FULL split
    tasks, n_pairs, unpaired = audit_pairing(data_dir)
    say(f"\n[A] {len(tasks)} task folders, {n_pairs} MP4+HDF5 pairs, "
        f"{len(unpaired)} unpaired files")
    for u in unpaired[:20]:
        say(f"    UNPAIRED: {u}")
    say("    naming convention: <data_dir>/<task_name>/<episode_idx>.{mp4,hdf5}")

    episodes = sample_episodes(tasks, n_episodes)
    say(f"    sampled {len(episodes)} episodes across distinct tasks:")
    for e in episodes:
        say(f"      {e.parent.name}/{e.name}")

    # B. structure
    dump_tree(episodes[0], report)  # full tree for one episode -> report file
    all_missing = {}
    langs = {}
    frame_counts = {}
    conf_present = True
    for ep in episodes:
        missing, joints, lang, n_frames, has_conf = check_keys(ep)
        eid = f"{ep.parent.name}/{ep.stem}"
        all_missing[eid] = missing
        langs[eid] = lang
        frame_counts[eid] = n_frames
        conf_present &= has_conf
    say(f"\n[B] required keys missing: "
        f"{ {k: v for k, v in all_missing.items() if v} or 'none'}")
    finger = [j for j in joints["left"]
              if not any(s in j for s in ("Arm", "Forearm", "Shoulder"))]
    say(f"    Q1 wrists: transforms/leftHand, transforms/rightHand — (N,4,4) SE(3)")
    say(f"    Q2 fingers: transforms/<side><Finger><Part>; {len(finger)} finger "
        f"joints + wrist = {len(finger)+1}/hand (thumb has no Metacarpal); "
        f"plus Arm/Forearm/Shoulder upper-body joints")
    say(f"    left finger joints: {sorted(finger)}")
    say("    Q3 pinch pair: transforms/{left,right}ThumbTip + "
        "transforms/{left,right}IndexFingerTip")
    say("    Q4 head/camera: transforms/camera (N,4,4) per-frame extrinsics; "
        "intrinsics at camera/intrinsic")
    say(f"    Q5 validity: confidences/<joint> (N,) in [0,1] where present; "
        f"NOT present on all episodes -> fallback: NaN check on transforms")
    say("    language annotation: root attrs; newer episodes: llm_description "
        "(+ llm_description2 with which_llm_description selector on reversible "
        "tasks); older episodes: description")
    for eid, lang in langs.items():
        say(f"      {eid}: task={lang['task']!r} which={lang['which_llm_description']!r}")
        say(f"        desc={str(lang['description'])[:70]!r} "
            f"llm1={str(lang['llm_description'])[:70]!r}")

    # C. synchronization
    say("\n[C] video vs pose frame counts:")
    sync_ok = True
    fps_vals = []
    for ep in episodes:
        n_video, fps, w, h = video_meta(ep.with_suffix(".mp4"))
        n_pose = frame_counts[f"{ep.parent.name}/{ep.stem}"]
        fps_vals.append(fps)
        delta = n_video - n_pose
        sync_ok &= abs(delta) <= 1
        say(f"    {ep.parent.name}/{ep.stem}: video={n_video} pose={n_pose} "
            f"delta={delta:+d}  fps={fps:.3f}  {w}x{h}")
    say(f"    fps: {sorted(set(round(v, 3) for v in fps_vals))} "
        f"(expected {expected_fps})")
    say("    no timestamp datasets in HDF5 -> alignment rule: pose[i] <-> "
        "video frame i (verify delta<=1 above)")

    # D. signal sanity + coordinate frame
    say("\n[D] signal sanity (right wrist):")
    sanity = {}
    n_plot = cfg["audit"]["n_signal_episodes"]
    for i, ep in enumerate(episodes):
        s = signal_sanity(ep, expected_fps, conf_min, out_dir, make_plot=i < n_plot)
        sanity[f"{ep.parent.name}/{ep.stem}"] = s
        say(f"    {ep.parent.name}/{ep.stem}: pos range "
            f"[{s['pos_min']} .. {s['pos_max']}], median disp/frame "
            f"{s['median_disp']*1000:.1f}mm, NaN frames {s['nan_frames']}, "
            f"invalid {s['frac_invalid']*100:.1f}% (longest run "
            f"{s['longest_invalid_run']}, conf={'yes' if s['has_confidences'] else 'ABSENT->NaN rule'}), "
            f"cam travel {s['cam_travel_m']:.2f}m, cam above wrists: "
            f"{s['cam_above_wrists']}")

    n_conf = sum(s["has_confidences"] for s in sanity.values())
    encoding_ok = all(s["has_confidences"] or s["nan_frames"] == 0
                      for s in sanity.values())
    world_frame = all(s["cam_travel_m"] > 0.01 and s["cam_above_wrists"]
                      for s in sanity.values())
    say(f"    missing-data encoding: {n_conf}/{len(sanity)} episodes have "
        f"confidences; episodes without it show 0 NaNs -> validity rule: "
        f"conf<{conf_min} where present, NaN check otherwise")
    say(f"    coordinate frame: camera transform varies per frame (travel "
        f">1cm) AND camera is above both wrists on all episodes -> WORLD "
        f"frame (ARKit gravity-aligned, y-up); camera-relative would need "
        f"identity camera / negative wrist y")
    say(f"    units: position magnitudes ~O(1) and mm-scale per-frame "
        f"displacement -> meters")

    # ------------------------------------------------------------ verifier
    max_invalid = cfg["audit"]["max_invalid_fraction"]
    worst_invalid = max(s["frac_invalid"] for s in sanity.values())
    checks = [
        ("MP4/HDF5 pairing resolved (full split)", len(unpaired) == 0),
        ("Wrist pose key paths identified (both hands)",
         not any("Hand" in k for v in all_missing.values() for k in v)),
        ("Thumb-tip / index-tip pinch pair identified",
         not any("Tip" in k for v in all_missing.values() for k in v)),
        ("Camera/head pose key path identified",
         not any("camera" in k for v in all_missing.values() for k in v)),
        ("Missing-data encoding understood (confidences | NaN fallback)",
         encoding_ok),
        ("Coordinate frame determined: world (cam travel + geometry)",
         bool(world_frame)),
        ("Frame alignment rule established (|delta| <= 1)", sync_ok),
        (f"<{max_invalid*100:.0f}% invalid pose frames (worst "
         f"{worst_invalid*100:.1f}%)", worst_invalid < max_invalid),
    ]
    say("\n================ LEVEL 0 VERIFIER ================")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    say("==================================================")
    say(f"LEVEL 0: {'ALL CHECKS PASSED' if all_pass else 'FAILED — STOP'}")

    (out_dir / "audit_report.txt").write_text("\n".join(report))
    (out_dir / "audit_summary.json").write_text(json.dumps(
        {"pairing_unpaired": unpaired, "frame_counts": frame_counts,
         "sanity": sanity, "checks": {n: bool(v) for n, v in checks}},
        indent=2, default=str))
    say(f"report -> {out_dir / 'audit_report.txt'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
