"""Level 1 verifier — filmstrip per detected transition (the primary human
check, since there is no kinematic signal to plot) + statistics + degenerate
checks. Mirrors Kinematics_pipeline's verify_level1.py filmstrip pattern.

    python verify_level1.py [--n 5]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import episode_slug, load_config, sample_episodes  # noqa: E402


def filmstrip(mp4_path, result, out_png, offset=15, thumb_w=320):
    """One row per detected transition (not episode_start/end): frames at
    b-offset, b, b+offset — a correct transition should show a visible
    change in hand/object configuration across the strip."""
    transitions = [b for b in result["boundaries"] if b["source"] == "vlm_transition"]
    if not transitions:
        return False
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    n = result["n_frames"]
    rows = []
    for b in transitions:
        tiles = []
        for df in (-offset, 0, offset):
            idx = int(np.clip(b["frame"] + df, 0, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {idx} of {mp4_path}")
            h = int(frame.shape[0] * thumb_w / frame.shape[1])
            tile = cv2.resize(frame, (thumb_w, h))
            label = f"f{idx}"
            if df == 0:
                label += f"  {b['before']} -> {b['after']}"
            cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 2)
            tiles.append(tile)
        rows.append(cv2.hconcat(tiles))
    cap.release()
    cv2.imwrite(str(out_png), cv2.vconcat(rows))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    boundaries_dir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    out_dir = boundaries_dir / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    min_len = cfg["level1_transitions"]["min_segment_frames"]

    episodes = sample_episodes(data_dir, args.n)
    stats_rows, degenerate_fails, missing = [], [], []
    for ep in episodes:
        bfile = boundaries_dir / f"{episode_slug(ep)}.json"
        if not bfile.exists():
            missing.append(ep.name)
            continue
        result = json.loads(bfile.read_text())
        slug = episode_slug(ep)
        has_transitions = filmstrip(ep.with_suffix(".mp4"), result,
                                    out_dir / f"strip_{slug}.png")

        segs = result["segments"]
        durs = [s["duration"] for s in segs]
        minutes = result["n_frames"] / result["fps"] / 60
        n_trans = sum(1 for b in result["boundaries"]
                     if b["source"] == "vlm_transition")
        covered = sum(s["end_frame"] - s["start_frame"] for s in segs)
        stats_rows.append((result["episode_id"], len(segs), len(segs) / minutes,
                          min(durs), float(np.median(durs)), max(durs), n_trans,
                          covered / max(result["n_frames"] - 1, 1)))

        ok_len = all(s["end_frame"] - s["start_frame"] >= min_len for s in segs) \
            or result["n_frames"] - 1 < min_len
        ok_edges = (result["boundaries"][0]["frame"] == 0
                   and result["boundaries"][-1]["frame"] == result["n_frames"] - 1)
        ok_monotonic = all(a["end_frame"] == b["start_frame"]
                          for a, b in zip(segs, segs[1:]))
        for name, ok in (("min_length", ok_len), ("edges", ok_edges),
                         ("monotonic_contiguous", ok_monotonic)):
            if not ok:
                degenerate_fails.append(f"{result['episode_id']}: {name}")
        if not has_transitions:
            print(f"note: {result['episode_id']} found ZERO transitions "
                 f"(one segment spans the whole episode)")

    print(f"\n{'episode':<45}{'segs':>5}{'seg/min':>9}{'dur min':>9}"
          f"{'med':>7}{'max':>7}{'trans':>7}  coverage")
    sane = 0
    for eid, nseg, spm, dmin, dmed, dmax, ntr, cov in stats_rows:
        sane += 4 <= spm <= 20
        print(f"{eid:<45}{nseg:>5}{spm:>9.1f}{dmin:>9.2f}{dmed:>7.2f}"
              f"{dmax:>7.2f}{ntr:>7}{cov:>10.0%}")

    print("\n================ LEVEL 1 VERIFIER ================")
    n = len(stats_rows)
    checks = [
        (f"segments/minute in [4, 20] on >=4/{n} episodes ({sane} sane)",
         sane >= min(4, n) if n else False),
        ("degenerate checks (min length, edges, contiguity)", not degenerate_fails),
        ("filmstrips written for human review", n > 0),
        (f"boundaries found for all sampled episodes ({len(missing)} missing)",
         not missing),
    ]
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for d in degenerate_fails:
        print(f"      degenerate: {d}")
    for m in missing:
        print(f"      missing boundaries (run detect_transitions.py first): {m}")
    print("==================================================")
    print(f"LEVEL 1: {'PASSED (pending human review of filmstrips)' if all_pass else 'FAILED'}")
    print(f"outputs -> {out_dir}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
