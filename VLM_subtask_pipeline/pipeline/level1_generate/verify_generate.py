"""Level 1 verifier — filmstrip per generated subtask boundary (the primary
human check, since there is no kinematic signal to plot) + statistics +
degenerate checks. Mirrors VLM_seg_subtask_pipeline's verify_level1.py.

    python verify_generate.py [--n 5]
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
    """One row per generated boundary (not episode_start/end): frames at
    b-offset, b, b+offset — a correct boundary shows a visible change."""
    bounds = [b for b in result["boundaries"] if b["source"] == "vlm_subtask"]
    if not bounds:
        return False
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    n = result["n_frames"]
    rows = []
    for b in bounds:
        tiles = []
        for df in (-offset, 0, offset):
            idx = int(np.clip(b["frame"] + df, 0, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {idx} of {mp4_path}")
            h = int(frame.shape[0] * thumb_w / frame.shape[1])
            tile = cv2.resize(frame, (thumb_w, h))
            cv2.putText(tile, f"f{idx}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
    min_len = cfg["generate"]["min_segment_frames"]

    episodes = sample_episodes(data_dir, args.n)
    stats_rows, degenerate_fails, missing = [], [], []
    for ep in episodes:
        bfile = boundaries_dir / f"{episode_slug(ep)}.json"
        if not bfile.exists():
            missing.append(ep.name)
            continue
        result = json.loads(bfile.read_text())
        slug = episode_slug(ep)
        has_bounds = filmstrip(ep.with_suffix(".mp4"), result,
                               out_dir / f"strip_{slug}.png")
        segs = result["segments"]
        durs = [s["duration"] for s in segs]
        seconds = result["n_frames"] / result["fps"]
        n_bnd = sum(1 for b in result["boundaries"] if b["source"] == "vlm_subtask")
        covered = sum(s["end_frame"] - s["start_frame"] for s in segs)
        stats_rows.append((result["episode_id"], len(segs),
                          len(segs) / (seconds / 60),
                          min(durs), float(np.median(durs)), max(durs), n_bnd,
                          covered / max(result["n_frames"] - 1, 1), seconds))
        # Pre-L3 invariants ONLY: this stage emits candidate intervals that
        # may have gaps (L3 inserts idle) and sub-floor segments (L3 enforces
        # min length) — contiguity/min-length here would be checking L3's
        # contract at the wrong stage. What must already hold: intervals in
        # range, sorted, and NON-OVERLAPPING (stitch()'s postcondition).
        n_f = result["n_frames"]
        ok_range = all(0 <= s["start_frame"] < s["end_frame"] <= n_f - 1
                       for s in segs)
        ok_nonoverlap = all(a["end_frame"] <= b["start_frame"]
                            for a, b in zip(segs, segs[1:]))
        for name, ok in (("in_range", ok_range),
                         ("monotonic_nonoverlapping", ok_nonoverlap)):
            if not ok:
                degenerate_fails.append(f"{result['episode_id']}: {name}")
        n_short = sum(1 for s in segs
                      if s["end_frame"] - s["start_frame"] < min_len)
        if n_short:
            print(f"note: {result['episode_id']} has {n_short} sub-floor "
                 f"candidate(s) (<{min_len} frames) — L3 will absorb them")
        if not has_bounds:
            print(f"note: {result['episode_id']} produced ZERO internal "
                 f"boundaries (one segment spans the whole episode)")

    print(f"\n{'episode':<45}{'segs':>5}{'seg/min':>9}{'dur min':>9}"
          f"{'med':>7}{'max':>7}{'bnds':>6}  coverage")
    # seg/min is only a meaningful rate on episodes long enough to average
    # over — a 2s episode with 4 segments is 120/min but perfectly fine, so
    # short episodes are shown but don't gate the check.
    RATE_MIN_SEC = 20.0
    sane = rated = 0
    for eid, nseg, spm, dmin, dmed, dmax, nbn, cov, seconds in stats_rows:
        print(f"{eid:<45}{nseg:>5}{spm:>9.1f}{dmin:>9.2f}{dmed:>7.2f}"
              f"{dmax:>7.2f}{nbn:>6}{cov:>10.0%}")
        if seconds >= RATE_MIN_SEC:
            rated += 1
            sane += 4 <= spm <= 20

    print("\n============ GENERATE (L1) VERIFIER ============")
    n = len(stats_rows)
    checks = [
        (f"segments/minute in [4, 20] on episodes >={RATE_MIN_SEC:.0f}s "
         f"({sane}/{rated} sane; {n - rated} too short to rate)",
         sane == rated if rated else True),
        ("degenerate checks (in range, monotonic non-overlapping)",
         not degenerate_fails),
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
        print(f"      missing (run generate_subtasks.py first): {m}")
    print("================================================")
    print(f"GENERATE: {'PASSED (pending human review)' if all_pass else 'FAILED'}")
    print(f"outputs -> {out_dir}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
