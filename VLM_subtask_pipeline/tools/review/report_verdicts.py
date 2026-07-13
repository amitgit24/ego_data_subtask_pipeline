"""Score exported human verdicts against the single-pass subtask annotations.

    python report_verdicts.py path/to/vlm_subtask_review_verdicts.json

Identical metrics to the sibling pipelines' scorers so the three pipelines'
human-review results are directly comparable — accuracy per axis (boundary /
label / sentence / episode coherence), label accuracy per action verb, and flag
calibration (do low_confidence / short_segment flags predict human-found
errors?).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from common import load_config  # noqa: E402


def rate(pairs):
    judged = [ok for ok in pairs if ok is not None]
    if not judged:
        return "n/a", 0
    return f"{sum(judged) / len(judged):.0%}", len(judged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts", help="verdicts JSON exported from the UI")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ann_dir = Path(cfg["paths"]["output_dir"]) / "annotations"
    verdicts = json.loads(Path(args.verdicts).read_text())["verdicts"]

    axes = {"boundary": [], "label": [], "sentence": []}
    by_verb = defaultdict(list)
    flag_cal = {True: [], False: []}
    episode_ok = []
    notes = []

    for eid, ev in verdicts.items():
        ann_file = ann_dir / (eid.replace("/", "__") + ".json")
        if not ann_file.exists():
            print(f"warning: no annotation for {eid}, skipping")
            continue
        ann = json.loads(ann_file.read_text())
        subs = {s["id"]: s for s in ann["subtasks"]}
        episode_ok.append(ev.get("episode_ok"))
        if ev.get("note"):
            notes.append(f"{eid}: {ev['note']}")
        for sid, g in ev.get("segments", {}).items():
            s = subs.get(int(sid))
            if s is None:
                continue
            for axis in axes:
                axes[axis].append(g.get(axis))
            by_verb[s["action"]].append(g.get("label"))
            flag_cal[bool(s["flags"])].append(g.get("label"))
            if g.get("note"):
                notes.append(f"{eid}#{sid} ({s['action']}): {g['note']}")

    print("===== SINGLE-PASS SUBTASK HUMAN VERDICT REPORT =====")
    for axis, vals in axes.items():
        r, n = rate(vals)
        print(f"  {axis:<10s} accuracy: {r:>5s}  (n={n})")
    r, n = rate(episode_ok)
    print(f"  episode coherence: {r:>5s}  (n={n})")

    print("\nlabel accuracy by verb (n>=3):")
    for verb, vals in sorted(by_verb.items(),
                             key=lambda kv: -len([x for x in kv[1] if x is not None])):
        judged = [x for x in vals if x is not None]
        if len(judged) >= 3:
            print(f"  {verb:<12s} {sum(judged)/len(judged):>4.0%}  (n={len(judged)})")

    print("\nflag calibration (label accuracy):")
    for flagged, vals in flag_cal.items():
        r, n = rate(vals)
        tag = "flagged " if flagged else "clean   "
        print(f"  {tag} segments: {r:>5s}  (n={n})")

    if notes:
        print("\nreviewer notes:")
        for x in notes:
            print(f"  - {x}")


if __name__ == "__main__":
    main()
