"""Window-geometry sweep for the single-pass subtask pipeline.

Runs the SAME episode set through several (frame_skip, overlap_frac) configs so
the two open questions — how dense the frame sampling must be, and how much
window overlap coherence needs — can be answered empirically (auto metrics via
report_sweep.py) and by blind human A/B (make_sweep_review.py).

Per config the full pipeline runs (generate + assemble) with output redirected
to outputs/sweep/<config>/, so every config yields complete, schema-valid
annotations for the same episodes. The model is loaded ONCE for all configs.
Resumable: episodes whose labels exist under a config are skipped.

    python run_sweep.py --episodes-file episodes.txt [--configs s1_o33,s2_o33]
                        [--force]
    python run_sweep.py --episodes task/idx [...]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "level1_generate"))
sys.path.insert(0, str(PIPELINE / "level3_assembly"))

from common import episode_id, episode_slug, load_config  # noqa: E402

# The sweep grid: skip swept at the default overlap, overlap swept at the
# default skip — 5 configs answer both axes without the full 3x3.
CONFIGS = {
    "s0_o33": {"frame_skip": 0, "overlap_frac": 0.33},
    "s1_o33": {"frame_skip": 1, "overlap_frac": 0.33},   # shipping default
    "s2_o33": {"frame_skip": 2, "overlap_frac": 0.33},
    "s1_o00": {"frame_skip": 1, "overlap_frac": 0.00},
    "s1_o50": {"frame_skip": 1, "overlap_frac": 0.50},
}


def sweep_cfg(base_cfg, name, params, sweep_root):
    cfg = copy.deepcopy(base_cfg)
    cfg["generate"].update(params)
    cfg["paths"]["output_dir"] = str(sweep_root / name)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--episodes-file", default=None,
                    help="text file, one task/idx per line")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    sweep_root = Path(base_cfg["paths"]["output_dir"]) / "sweep"

    if args.episodes_file:
        episodes = [ln.strip() for ln in Path(args.episodes_file).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]
    elif args.episodes:
        episodes = args.episodes
    else:
        ap.error("give --episodes or --episodes-file")
    config_names = [c.strip() for c in args.configs.split(",")]
    for c in config_names:
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; choose from {list(CONFIGS)}")

    data_dir = Path(base_cfg["paths"]["data_dir"])
    paths = [data_dir / f"{e}.hdf5" for e in episodes]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing episode: {p}")

    from model_backend import load_model
    from generate_subtasks import process_episode
    from assemble import assemble_episode

    print(f"sweep: {len(config_names)} configs x {len(episodes)} episodes")
    print(f">>> loading {base_cfg['paths']['model_path']} ...")
    model, processor = load_model(base_cfg["paths"]["model_path"],
                                  base_cfg["model"]["attn_implementation"],
                                  base_cfg["model"]["dtype"])

    t_start = time.time()
    failures = []
    for name in config_names:
        cfg = sweep_cfg(base_cfg, name, CONFIGS[name], sweep_root)
        labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
        print(f"\n===== config {name} ({CONFIGS[name]}) =====")
        for i, p in enumerate(paths):
            slug = episode_slug(p)
            label_file = labels_dir / f"{slug}.json"
            try:
                if label_file.exists() and not args.force:
                    rec = json.loads(label_file.read_text())
                    print(f"  [{i+1}/{len(paths)}] skip (exists): {episode_id(p)}")
                else:
                    rec = process_episode(model, processor, p, cfg)
                    print(f"  [{i+1}/{len(paths)}] {rec['episode_id']}: "
                         f"{rec['num_windows']} win, {rec['n_candidates']} cand "
                         f"-> {len(rec['segments'])} seg  ({rec['wall_sec']}s)")
                ann, _ = assemble_episode(label_file, cfg)
            except Exception as e:
                failures.append((name, episode_id(p), repr(e)))
                print(f"  FAILED {name}/{episode_id(p)}: {e!r}", file=sys.stderr)

    dt = time.time() - t_start
    print(f"\nsweep done in {dt/60:.1f} min, {len(failures)} failure(s)")
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    (sweep_root / "sweep_manifest.json").write_text(json.dumps({
        "episodes": episodes, "configs": {c: CONFIGS[c] for c in config_names},
        "failures": failures, "wall_min": round(dt / 60, 1)}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
