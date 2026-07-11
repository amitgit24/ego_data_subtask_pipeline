"""Level 3 verifier — schema validation of every annotation, invariant
re-check, annotated preview videos, summary report.

    python verify_level3.py [--previews 2]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402
from schema import TIME_TOL, EpisodeAnnotation  # noqa: E402


def render_preview(ann, mp4_path, out_path):
    """Burn current subtask + progress bar onto the video."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    subs = ann.subtasks
    si = 0
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while si + 1 < len(subs) and f >= subs[si].end_frame:
            si += 1
        s = subs[si]
        band_h = 110
        frame[h - band_h:] = frame[h - band_h:] // 3  # darken caption band
        cv2.putText(frame, f"#{s.id} [{s.action}] hand={s.hand}",
                    (20, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)
        cv2.putText(frame, s.subtask[:95], (20, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        prog = (f - s.start_frame) / max(s.end_frame - s.start_frame, 1)
        cv2.rectangle(frame, (20, h - 20), (w - 20, h - 12), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, h - 20),
                      (20 + int((w - 40) * min(prog, 1.0)), h - 12),
                      (0, 220, 0), -1)
        out.write(frame)
        f += 1
    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--previews", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ann_dir = Path(cfg["paths"]["output_dir"]) / "annotations"
    data_dir = Path(cfg["paths"]["data_dir"])
    preview_dir = ann_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in ann_dir.glob("*.json"))
    if not files:
        print("no annotation files — run assemble.py first")
        sys.exit(1)

    valid, invalid = [], []
    for f in files:
        try:
            # model_validate re-runs every invariant incl. exact duration
            # arithmetic (TIME_TOL) — schema validation IS the re-check
            ann = EpisodeAnnotation.model_validate(json.loads(f.read_text()))
            valid.append(ann)
        except Exception as e:
            invalid.append((f.name, str(e).splitlines()[0]))

    for name, err in invalid:
        print(f"INVALID {name}: {err}")

    n_subs = sum(len(a.subtasks) for a in valid)
    flagged = sum(1 for a in valid for s in a.subtasks if s.flags)
    verbs = Counter(s.action for a in valid for s in a.subtasks)
    n_review = sum(len(a.review_queue) for a in valid)

    for ann in valid[:args.previews]:
        mp4 = data_dir / f"{ann.episode_id}.mp4"
        out = preview_dir / (ann.episode_id.replace("/", "__") + "_preview.mp4")
        render_preview(ann, mp4, out)
        print(f"preview -> {out}")

    print(f"\nepisodes processed: {len(valid)} valid / {len(files)} total")
    print(f"total subtasks: {n_subs}  mean/episode: {n_subs / max(len(valid), 1):.1f}")
    print(f"flagged for review: {n_review} subtasks "
          f"({flagged / max(n_subs, 1):.1%} of all)")
    print(f"verb histogram: {dict(verbs.most_common())}")

    checks = [
        (f"schema-valid: {len(valid)}/{len(files)} (must be 100%)",
         not invalid),
        (f"invariants + duration arithmetic exact to {TIME_TOL}", not invalid),
        (f"{min(args.previews, len(valid))} preview videos rendered "
         "for human review", True),
    ]
    print("\n================ LEVEL 3 VERIFIER ================")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("==================================================")
    print(f"LEVEL 3: {'PASSED (pending human review of previews)' if all_pass else 'FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
