"""End-to-end driver for the single-pass subtask pipeline.

    python run_pipeline.py --episodes <all | N_random | task/idx [...]>
                           [--levels 1,3] [--force] [--verify-only]

Level 1 (generate_subtasks) needs the large local Qwen3-VL model in-process and
loads it once for all episodes. Level 3 (assemble) has no model and runs as a
lightweight subprocess. There is no Level 2: segmentation and labeling are one
step here, which is the whole point of this pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import episode_id, episode_slug, load_config, sample_episodes

ROOT = Path(__file__).resolve().parent
PY = sys.executable

VERIFIERS = {
    1: ("level1_generate", "verify_generate.py"),
    3: ("level3_assembly", "verify_level3.py"),
}


def run_subprocess(module_dir, script, extra):
    cmd = [PY, str(ROOT / module_dir / script)] + extra
    print(f"\n>>> {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, cwd=ROOT / module_dir)
    if res.returncode != 0:
        print(f"{script} exited with {res.returncode} — stopping.", file=sys.stderr)
        sys.exit(res.returncode)


def run_generate(paths, cfg, force):
    sys.path.insert(0, str(ROOT / "level1_generate"))
    from model_backend import load_model
    from generate_subtasks import DebugBudget, process_episode

    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    print(f"\n>>> loading {cfg['paths']['model_path']} ...")
    model, processor = load_model(cfg["paths"]["model_path"],
                                  cfg["model"]["attn_implementation"],
                                  cfg["model"]["dtype"])
    debug_budget = DebugBudget(cfg["generate"]["debug_first_n"])

    done, failed = 0, []
    for p in paths:
        out_file = labels_dir / f"{episode_slug(p)}.json"
        if out_file.exists() and not force:
            print(f"skip (labels exist): {episode_id(p)}")
            continue
        try:
            rec = process_episode(model, processor, p, cfg, debug_budget)
            done += 1
            print(f"[{done}] {rec['episode_id']}: {rec['num_windows']} windows "
                 f"-> {len(rec['segments'])} segments")
        except Exception as e:
            failed.append((episode_id(p), repr(e)))
            print(f"FAILED {episode_id(p)}: {e!r}", file=sys.stderr)

    print(f"\n{done} episodes generated, {len(failed)} failed")
    if failed:
        for eid, err in failed:
            print(f"  {eid}: {err}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="+", default=["all"])
    ap.add_argument("--levels", default="1,3")
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

    if 1 in levels:
        run_generate(paths, cfg, args.force)
    if 3 in levels:
        force = ["--force"] if args.force else []
        run_subprocess("level3_assembly", "assemble.py",
                       ["--episodes"] + episodes + force)

    print("\npipeline complete.")


if __name__ == "__main__":
    main()
