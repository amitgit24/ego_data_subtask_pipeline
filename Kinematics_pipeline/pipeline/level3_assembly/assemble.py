"""Level 3 — final JSON writer: Level 2 labels -> postprocess -> schema ->
outputs/annotations/{episode}.json.

CLI:
    python assemble.py --episodes task/idx [...] | --all [--force]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import apply_task_overrides, episode_slug, load_config  # noqa: E402
from postprocess import postprocess  # noqa: E402
from schema import EpisodeAnnotation  # noqa: E402


def assemble_episode(label_file, cfg, mirror_to_source=False):
    record = json.loads(label_file.read_text())
    task_name = record["episode_id"].split("/")[0]
    subtasks, review = postprocess(record, apply_task_overrides(cfg, task_name))
    ann = EpisodeAnnotation(
        episode_id=record["episode_id"],
        task=record["task"],
        fps=record["fps"],
        total_frames=record["n_frames"],
        pipeline_version=cfg["assembly"]["pipeline_version"],
        subtasks=subtasks,
        review_queue=review,
    )
    out_dir = Path(cfg["paths"]["output_dir"]) / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / label_file.name
    body = ann.model_dump_json(indent=2)
    out.write_text(body)
    if mirror_to_source:
        # deliverable copy, same folder/basename as the source .hdf5 -- the
        # outputs/annotations/ copy above stays the pipeline's own working
        # copy (resumability skip-check, verify_level3.py, make_review.py
        # all read from there; this is purely an additional write).
        mirror = Path(cfg["paths"]["data_dir"]) / f"{record['episode_id']}.json"
        mirror.write_text(body)
    return ann, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mirror-to-source", action="store_true",
                    help="also write a copy of each episode's annotation "
                         "next to its source .hdf5 (same folder, same "
                         "base filename)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    ann_dir = Path(cfg["paths"]["output_dir"]) / "annotations"

    if args.episodes:
        files = [labels_dir / (e.replace("/", "__") + ".json")
                 for e in args.episodes]
    elif args.all:
        files = sorted(f for f in labels_dir.glob("*.json")
                       if f.name != "skip_list.json")
    else:
        ap.error("give --episodes or --all")

    done, skipped, failed = 0, 0, []
    for f in files:
        if (ann_dir / f.name).exists() and not args.force:
            skipped += 1
            continue
        try:
            ann, out = assemble_episode(f, cfg, args.mirror_to_source)
            done += 1
            print(f"[{done}] {ann.episode_id}: {len(ann.subtasks)} subtasks "
                  f"({len(ann.review_queue)} queued for review) -> {out.name}")
        except Exception as e:
            failed.append((f.name, repr(e)))
            print(f"FAILED {f.name}: {e!r}", file=sys.stderr)

    print(f"\n{done} assembled, {skipped} skipped, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
