"""Un-blind and tally the sweep A/B human verdicts.

    python report_sweep_verdicts.py sweep_review_verdicts.json

Maps each episode's best/worst letter back to its config via
outputs/sweep/review/blinding_key.json and reports best/worst counts and a
net score (best - worst) per config, plus reviewer notes.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
from common import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sweep_root = Path(cfg["paths"]["output_dir"]) / "sweep"
    key = json.loads((sweep_root / "review" / "blinding_key.json").read_text())["map"]
    verdicts = json.loads(Path(args.verdicts).read_text())["verdicts"]

    best, worst = Counter(), Counter()
    judged = 0
    notes = []
    for eid, v in verdicts.items():
        if eid not in key:
            print(f"warning: {eid} not in blinding key, skipping")
            continue
        m = key[eid]
        if v.get("best") in m:
            best[m[v["best"]]] += 1
        if v.get("worst") in m:
            worst[m[v["worst"]]] += 1
        if v.get("best") in m and v.get("worst") in m:
            judged += 1
        if v.get("note"):
            notes.append(f"{eid}: {v['note']}")

    configs = sorted(set(best) | set(worst),
                     key=lambda c: -(best[c] - worst[c]))
    print(f"========= SWEEP A/B VERDICTS ({judged} episodes fully judged) =========")
    print(f"{'config':<10}{'best':>6}{'worst':>7}{'net':>6}")
    for c in configs:
        print(f"{c:<10}{best[c]:>6}{worst[c]:>7}{best[c]-worst[c]:>6}")
    if notes:
        print("\nreviewer notes:")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
