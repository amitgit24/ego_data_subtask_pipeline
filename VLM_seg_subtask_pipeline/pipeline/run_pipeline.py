"""End-to-end pipeline driver for the Qwen-direct pipeline.

    python run_pipeline.py --episodes <all | N_random | task/idx [...]>
                           [--levels 1,2,3] [--force] [--verify-only]

Unlike Kinematics_pipeline's driver, Levels 1 and 2 both need the (large,
slow-to-load) local Qwen3-VL model in-process — there is no vLLM server to
share. When both are requested together, the model is loaded ONCE and both
levels run in a single Python process per episode, instead of paying the
weight-load cost twice by subprocessing each level separately. Level 3 has
no model and runs as a lightweight subprocess, same as Kinematics_pipeline.

Levels are resumable: each level skips episodes whose output already exists
(unless --force).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import episode_id, episode_slug, load_config, sample_episodes

ROOT = Path(__file__).resolve().parent
PY = sys.executable

LEVEL_DIRS = {1: "level1_transitions", 2: "level2_labeling", 3: "level3_assembly"}
VERIFIERS = {
    1: ("level1_transitions", "verify_level1.py"),
    2: ("level2_labeling", "verify_level2.py"),
    3: ("level3_assembly", "verify_level3.py"),
}


def run_subprocess(module_dir, script, extra):
    cmd = [PY, str(ROOT / module_dir / script)] + extra
    print(f"\n>>> {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, cwd=ROOT / module_dir)
    if res.returncode != 0:
        print(f"{script} exited with {res.returncode} — stopping.", file=sys.stderr)
        sys.exit(res.returncode)


def run_levels_1_2(paths, cfg, force):
    """Load the model once, run Level 1 (if its output is missing) and
    Level 2 for each episode in this single process."""
    sys.path.insert(0, str(ROOT / "level1_transitions"))
    sys.path.insert(0, str(ROOT / "level2_labeling"))
    from model_backend import load_model
    from detect_transitions import process_episode as detect_boundaries
    from label_segments import DebugBudget
    from label_segments import process_episode as label_episode

    boundaries_dir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> loading {cfg['paths']['model_path']} ...")
    model, processor = load_model(cfg["paths"]["model_path"],
                                  cfg["model"]["attn_implementation"],
                                  cfg["model"]["dtype"])
    debug_budget = DebugBudget(cfg["level2_labeling"]["debug_first_n"])

    done, failed = 0, []
    for p in paths:
        label_file = labels_dir / f"{episode_slug(p)}.json"
        if label_file.exists() and not force:
            print(f"skip (labels exist): {episode_id(p)}")
            continue
        try:
            bfile = boundaries_dir / f"{episode_slug(p)}.json"
            if not bfile.exists() or force:
                import json
                result = detect_boundaries(model, processor, p, cfg)
                bfile.write_text(json.dumps(result, indent=2))
                print(f"  L1 {result['episode_id']}: {result['n_frames']} frames "
                     f"-> {len(result['segments'])} segments")
            out = label_episode(model, processor, p, cfg, debug_budget)
            done += 1
            print(f"[{done}] L2 {out['episode_id']}: {len(out['segments'])} "
                 f"segments labeled")
        except Exception as e:
            failed.append((episode_id(p), repr(e)))
            print(f"FAILED {episode_id(p)}: {e!r}", file=sys.stderr)

    print(f"\n{done} episodes labeled, {len(failed)} failed")
    if failed:
        for eid, err in failed:
            print(f"  {eid}: {err}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="+", default=["all"],
                    help="'all', 'N_random', or explicit task/idx ids")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = [int(x) for x in args.levels.split(",")]

    if args.verify_only:
        for lv in levels:
            if lv in VERIFIERS:
                run_subprocess(*VERIFIERS[lv], [])
        return

    spec = args.episodes
    if spec == ["all"]:
        data_dir = Path(cfg["paths"]["data_dir"])
        episodes = [f"{p.parent.name}/{p.stem}"
                   for p in sorted(data_dir.glob("*/*.hdf5"))]
    elif len(spec) == 1 and spec[0].endswith("_random"):
        n = int(spec[0].split("_")[0])
        paths = sample_episodes(cfg["paths"]["data_dir"], n, seed=args.seed)
        episodes = [episode_id(p) for p in paths]
    else:
        episodes = spec
    print(f"pipeline over {len(episodes)} episodes, levels {levels}")

    data_dir = Path(cfg["paths"]["data_dir"])
    paths = [data_dir / f"{e}.hdf5" for e in episodes]

    if 1 in levels or 2 in levels:
        run_levels_1_2(paths, cfg, args.force)
    if 3 in levels:
        force = ["--force"] if args.force else []
        run_subprocess("level3_assembly", "assemble.py",
                       ["--episodes"] + episodes + force)

    print("\npipeline complete.")


if __name__ == "__main__":
    main()
