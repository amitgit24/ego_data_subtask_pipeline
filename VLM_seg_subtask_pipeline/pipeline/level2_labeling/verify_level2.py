"""Level 2 verifier — contact sheets (human check) + label statistics.

Adapted from Kinematics_pipeline's verify_level2.py. No kinematic-semantic
agreement check exists here (no independent kinematic signal to check
against); instead reports the post-hoc-validation retry rate and the
fallback-to-'other' rate, which are this pipeline's equivalent failure-mode
signals (see label_segments.py's label_one()).

    python verify_level2.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402


def contact_sheet(ep, out_dir):
    slug = ep["episode_id"].replace("/", "__")
    rows = []
    for s in ep["segments"]:
        imgs = "".join(f'<img src="../keyframes/{slug}/{Path(p).name}" width="230">'
                       for p in s["keyframes"])
        v = s["vlm"]
        rows.append(f"""
<div class="seg">
  <div class="imgs">{imgs}</div>
  <p><b>#{s['id']} [{s['start_time']:.1f}s–{s['end_time']:.1f}s]
  {v['action']}</b> — {v['object']} — hand: {v['hand']}
  — conf: {v['confidence']} — style {s['style_id']}
  {'<b style="color:red">FALLBACK (other/low)</b>' if v['action']=='other' else ''}
  <br><i>{v['subtask']}</i>
  <br><small>boundary: "{s['transition_before']}" -> "{s['transition_after']}"</small></p>
</div>""")
    html = f"""<!doctype html><meta charset="utf-8">
<title>{ep['episode_id']}</title>
<style>body{{font-family:sans-serif;max-width:1250px;margin:auto}}
.seg{{border-bottom:1px solid #ccc;padding:10px 0}}</style>
<h2>{ep['episode_id']}</h2><p>Task: {ep['task']}</p>
{''.join(rows)}"""
    out = out_dir / f"{slug}.html"
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    sheet_dir = labels_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in labels_dir.glob("*.json") if f.name != "skip_list.json")
    if not files:
        print("no label files found — run label_segments.py first")
        sys.exit(1)

    episodes = [json.loads(f.read_text()) for f in files]
    segs = [s for ep in episodes for s in ep["segments"]]
    for ep in episodes:
        contact_sheet(ep, sheet_dir)

    actions = Counter(s["vlm"]["action"] for s in segs)
    hands = Counter(s["vlm"]["hand"] for s in segs)
    confs = Counter(s["vlm"]["confidence"] for s in segs)
    styles = Counter(s["style_id"] for s in segs)
    n = len(segs)

    retried = sum(1 for s in segs if s["vlm_attempts"] > 0)
    max_retries = cfg["level2_labeling"]["max_retries"]
    fallback = sum(1 for s in segs if s["vlm_attempts"] >= max_retries
                  and s["vlm"]["action"] == "other")

    bigrams = Counter(" ".join(s["vlm"]["subtask"].lower().split()[:2]) for s in segs)
    top_bigram, top_count = bigrams.most_common(1)[0] if segs else ("", 0)

    inconsistent = []
    for ep in episodes:
        objs = {s["vlm"]["object"].lower().strip().removeprefix("the ")
               for s in ep["segments"]}
        if len(objs) > max(2, len(ep["segments"]) * 2 // 3):
            inconsistent.append((ep["episode_id"], sorted(objs)))

    print(f"{len(episodes)} episodes, {n} segments")
    print(f"actions: {dict(actions.most_common())}")
    print(f"hands (VLM-judged, no kinematic ground truth): {dict(hands.most_common())}")
    print(f"confidence: {dict(confs.most_common())}")
    print(f"styles used: {dict(sorted(styles.items()))}")
    print(f"top first-2-words: {top_bigram!r} {top_count}/{n} "
         f"({top_count / max(n,1):.1%})")
    print(f"retried segments: {retried}/{n}; hard fallback-to-'other': "
         f"{fallback}/{n} ({fallback / max(n,1):.1%})")
    if inconsistent:
        for eid, objs in inconsistent:
            print(f"object naming to review — {eid}: {objs}")

    checks = [
        ("fallback-to-'other' rate < 5%", fallback / max(n, 1) < 0.05),
        ("'other' action < 20%", actions.get("other", 0) / max(n, 1) < 0.20),
        ("no single action > 50%", actions.most_common(1)[0][1] / max(n, 1) <= 0.50
         if actions else False),
        ("low confidence <= 30%", confs.get("low", 0) / max(n, 1) <= 0.30),
        ("phrasing diversity: top bigram < 30%", top_count / max(n, 1) < 0.30),
        ("contact sheets written for human review", True),
    ]
    print("\n================ LEVEL 2 VERIFIER ================")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("==================================================")
    print(f"LEVEL 2: {'PASSED (pending human review of contact sheets)' if all_pass else 'FAILED'}")
    print(f"contact sheets -> {sheet_dir}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
