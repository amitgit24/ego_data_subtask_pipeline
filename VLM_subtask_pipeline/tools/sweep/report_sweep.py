"""Automatic metrics report for the window-geometry sweep.

    python report_sweep.py

Per config (over the common episode set): cost (windows, wall time), structure
(segments/episode, durations, idle fraction), label health (verb diversity,
low-confidence and flag rates), overlap effectiveness (candidate->stitched
merge ratio), and cross-config boundary consensus — pairwise boundary F1 at a
±0.5 s tolerance, which shows whether a config produces the same cut points as
the others or is an outlier. Automatic metrics rank stability/cost; the blind
human review (make_sweep_review.py) is the judge of which is actually RIGHT.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
from common import load_config  # noqa: E402


def boundary_frames(ann):
    """Internal cut points of an annotation (excludes 0 and last frame)."""
    return [s["start_frame"] for s in ann["subtasks"][1:]]


def f1(a, b, tol):
    """Symmetric boundary-match F1: a boundary matches if within tol frames."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp_a = sum(1 for x in a if min(abs(x - y) for y in b) <= tol)
    tp_b = sum(1 for y in b if min(abs(y - x) for x in a) <= tol)
    prec = tp_a / len(a)
    rec = tp_b / len(b)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def entropy(counter):
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values()) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tol-sec", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sweep_root = Path(cfg["paths"]["output_dir"]) / "sweep"
    manifest = json.loads((sweep_root / "sweep_manifest.json").read_text())
    configs = list(manifest["configs"])
    episodes = manifest["episodes"]

    # load everything: per config -> per episode -> (label record, annotation)
    data = {c: {} for c in configs}
    for c in configs:
        for e in episodes:
            slug = e.replace("/", "__")
            lf = sweep_root / c / "labels" / f"{slug}.json"
            af = sweep_root / c / "annotations" / f"{slug}.json"
            if lf.exists() and af.exists():
                data[c][e] = (json.loads(lf.read_text()), json.loads(af.read_text()))
    common = [e for e in episodes if all(e in data[c] for c in configs)]
    print(f"sweep report over {len(common)}/{len(episodes)} episodes "
          f"complete in ALL {len(configs)} configs\n")
    if not common:
        sys.exit("nothing to report")

    rows = {}
    for c in configs:
        wins = cands = stitched = wall = idle_frames = tot_frames = 0
        nsubs, durs, verbs = [], [], Counter()
        low_conf = flagged = tot = 0
        for e in common:
            rec, ann = data[c][e]
            wins += rec["num_windows"]
            cands += rec["n_candidates"]
            stitched += rec["n_stitched"]
            wall += rec.get("wall_sec", 0)
            fps = ann["fps"]
            nsubs.append(len(ann["subtasks"]))
            for s in ann["subtasks"]:
                tot += 1
                durs.append(s["duration"])
                verbs[s["action"]] += 1
                nfr = s["end_frame"] - s["start_frame"]
                tot_frames += nfr
                if s["action"] == "idle":
                    idle_frames += nfr
                if s["confidence"] == "low":
                    low_conf += 1
                if s["flags"]:
                    flagged += 1
        durs.sort()
        rows[c] = {
            "windows": wins, "wall_min": wall / 60,
            "sec_per_win": wall / max(wins, 1),
            "merge_ratio": 1 - stitched / max(cands, 1),
            "subs_per_ep": sum(nsubs) / len(nsubs),
            "med_dur": durs[len(durs) // 2] if durs else 0,
            "idle_frac": idle_frames / max(tot_frames, 1),
            "verbs": len(verbs), "verb_entropy": entropy(verbs),
            "low_conf": low_conf / max(tot, 1), "flagged": flagged / max(tot, 1),
        }

    hdr = (f"{'config':<8}{'win':>5}{'s/win':>7}{'wall_m':>8}{'merge%':>8}"
           f"{'sub/ep':>8}{'meddur':>8}{'idle%':>7}{'verbs':>6}{'H(v)':>6}"
           f"{'low%':>6}{'flag%':>7}")
    print(hdr)
    for c in configs:
        r = rows[c]
        print(f"{c:<8}{r['windows']:>5}{r['sec_per_win']:>7.1f}"
              f"{r['wall_min']:>8.1f}{r['merge_ratio']:>8.0%}"
              f"{r['subs_per_ep']:>8.1f}{r['med_dur']:>8.2f}"
              f"{r['idle_frac']:>7.1%}{r['verbs']:>6}{r['verb_entropy']:>6.2f}"
              f"{r['low_conf']:>6.0%}{r['flagged']:>7.0%}")

    # cross-config boundary consensus
    print(f"\nboundary consensus (pairwise F1, tol ±{args.tol_sec}s):")
    pair_f1 = {c: {} for c in configs}
    for i, a in enumerate(configs):
        for b in configs:
            if a == b:
                continue
            scores = []
            for e in common:
                fps = data[a][e][1]["fps"]
                tol = args.tol_sec * fps
                scores.append(f1(boundary_frames(data[a][e][1]),
                                 boundary_frames(data[b][e][1]), tol))
            pair_f1[a][b] = sum(scores) / len(scores)
    print(f"{'':<8}" + "".join(f"{b:>9}" for b in configs) + f"{'mean':>9}")
    for a in configs:
        vals = [pair_f1[a][b] for b in configs if b != a]
        cells = "".join(f"{pair_f1[a][b]:>9.2f}" if b != a else f"{'—':>9}"
                        for b in configs)
        print(f"{a:<8}{cells}{sum(vals)/len(vals):>9.2f}")

    print("\nverb histograms:")
    for c in configs:
        verbs = Counter()
        for e in common:
            for s in data[c][e][1]["subtasks"]:
                verbs[s["action"]] += 1
        top = ", ".join(f"{v}:{n}" for v, n in verbs.most_common(8))
        print(f"  {c:<8} {top}")

    print("\nNOTE: these metrics rank stability and cost; blind human review "
          "(make_sweep_review.py -> report_sweep_verdicts) decides correctness.")


if __name__ == "__main__":
    main()
