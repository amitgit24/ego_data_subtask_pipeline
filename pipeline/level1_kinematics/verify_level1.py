"""Level 1 verifier — visual + statistical checks on sample episodes.

Per episode: signal plot with boundary lines, boundary filmstrip from the
video, statistics table, degenerate-case checks. Exit code 1 on any failure.

    python verify_level1.py [--config ../config.yaml] [--n 5]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import episode_slug, load_config, sample_episodes  # noqa: E402
from boundaries import propose_boundaries  # noqa: E402

SOURCE_COLORS = {
    "pause": "tab:blue", "grasp": "tab:green", "release": "tab:red",
    "gaze": "tab:orange", "episode_start": "gray", "episode_end": "gray",
}


def plot_signals(sig, result, out_png):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t = sig["t"]

    axes[0].plot(t, sig["speed_left"], label="left", alpha=0.8)
    axes[0].plot(t, sig["speed_right"], label="right", alpha=0.8)
    axes[0].plot(t, sig["speed_combined"], "k--", lw=1, label="combined (max)")
    axes[0].set_ylabel("wrist speed (m/s)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, sig["aperture_left"] * 100, label="left")
    axes[1].plot(t, sig["aperture_right"] * 100, label="right")
    axes[1].set_ylabel("pinch aperture (cm)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, sig["head_rot_speed"], color="tab:purple")
    axes[2].set_ylabel("head rot speed (rad/s)")
    axes[2].set_xlabel("time (s)")

    for b in result["boundaries"]:
        c = SOURCE_COLORS[b["source"]]
        for ax in axes:
            ax.axvline(b["time"], color=c, lw=1.2,
                       ls="-" if b["source"].startswith("episode") else "--",
                       alpha=0.9)
    handles = [plt.Line2D([0], [0], color=c, ls="--", label=s)
               for s, c in SOURCE_COLORS.items() if not s.startswith("episode")]
    axes[0].legend(handles=axes[0].get_legend().legend_handles + handles,
                   loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(result["episode_id"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def filmstrip(mp4_path, result, out_png, offset=15, thumb_w=320):
    """One row per boundary: frames at b-offset, b, b+offset."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    n = result["n_frames"]
    rows = []
    for b in result["boundaries"]:
        tiles = []
        for df in (-offset, 0, offset):
            idx = int(np.clip(b["frame"] + df, 0, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {idx} of {mp4_path}")
            h = int(frame.shape[0] * thumb_w / frame.shape[1])
            tile = cv2.resize(frame, (thumb_w, h))
            label = f"f{idx}" + ("  <- " + b["source"] if df == 0 else "")
            cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2)
            tiles.append(tile)
        rows.append(cv2.hconcat(tiles))
    cap.release()
    cv2.imwrite(str(out_png), cv2.vconcat(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"]) / "boundaries" / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    min_len = cfg["kinematics"]["min_segment_frames"]

    episodes = sample_episodes(data_dir, args.n,
                               min_frames=8 * cfg["video"]["fps"])

    stats_rows = []
    degenerate_fails = []
    for ep in episodes:
        result, sig = propose_boundaries(ep, cfg)
        slug = episode_slug(ep)
        plot_signals(sig, result, out_dir / f"signals_{slug}.png")
        filmstrip(ep.with_suffix(".mp4"), result, out_dir / f"strip_{slug}.png")

        segs = result["segments"]
        durs = [s["duration"] for s in segs]
        minutes = result["n_frames"] / result["fps"] / 60
        interior = [b for b in result["boundaries"]
                    if not b["source"].startswith("episode")]
        src = {s: sum(b["source"] == s for b in interior) for s in
               ("pause", "grasp", "release", "gaze")}
        covered = sum(s["end_frame"] - s["start_frame"] for s in segs)
        stats_rows.append((result["episode_id"], len(segs),
                           len(segs) / minutes, min(durs), float(np.median(durs)),
                           max(durs), src, covered / (result["n_frames"] - 1)))

        # degenerate checks
        ok_len = all(s["end_frame"] - s["start_frame"] >= min_len for s in segs) \
            or result["n_frames"] - 1 < min_len
        ok_mask = all(not any(a <= b["frame"] < e for a, e in result["masked_regions"])
                      for b in interior)
        ok_edges = (result["boundaries"][0]["frame"] == 0
                    and result["boundaries"][-1]["frame"] == result["n_frames"] - 1)
        for name, ok in (("min_length", ok_len), ("mask", ok_mask),
                         ("edges", ok_edges)):
            if not ok:
                degenerate_fails.append(f"{result['episode_id']}: {name}")

    print(f"\n{'episode':<45}{'segs':>5}{'seg/min':>9}{'dur min':>9}"
          f"{'med':>7}{'max':>7}  sources (pause/grasp/release/gaze)  coverage")
    sane = 0
    for eid, nseg, spm, dmin, dmed, dmax, src, cov in stats_rows:
        sane += 4 <= spm <= 20
        print(f"{eid:<45}{nseg:>5}{spm:>9.1f}{dmin:>9.2f}{dmed:>7.2f}"
              f"{dmax:>7.2f}   {src['pause']}/{src['grasp']}/{src['release']}"
              f"/{src['gaze']}{cov:>27.0%}")

    print("\n================ LEVEL 1 VERIFIER ================")
    checks = [
        (f"segments/minute in [4, 20] on >=4/{len(stats_rows)} episodes "
         f"({sane} sane)", sane >= min(4, len(stats_rows))),
        ("degenerate checks (min length, masks, edges)",
         not degenerate_fails),
        ("plots + filmstrips written for human review", True),
    ]
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for d in degenerate_fails:
        print(f"      degenerate: {d}")
    print("==================================================")
    print(f"LEVEL 1: {'PASSED (pending human review of plots)' if all_pass else 'FAILED'}")
    print(f"outputs -> {out_dir}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
