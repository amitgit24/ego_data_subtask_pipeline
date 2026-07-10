"""End-to-end pipeline driver.

    python run_pipeline.py --episodes <all | N_random | task/idx [task/idx ...]>
                           [--levels 1,2,3] [--seed 0] [--force] [--verify-only]

Levels are resumable: each level skips episodes whose output already exists
(unless --force). `N_random` draws N episodes that have no labels yet, from
distinct tasks where possible, deterministically per --seed. Level 0 is the
dataset audit and runs on its own sample. --verify-only reruns verifiers.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

from common import episode_slug, load_config

ROOT = Path(__file__).resolve().parent
PY = sys.executable

LEVEL_SCRIPTS = {
    0: ("level0_audit", "audit_dataset.py"),
    1: ("level1_kinematics", "boundaries.py"),
    2: ("level2_vlm", "label_segments.py"),
    3: ("level3_assembly", "assemble.py"),
}
VERIFIERS = {
    1: ("level1_kinematics", "verify_level1.py"),
    2: ("level2_vlm", "verify_level2.py"),
    3: ("level3_assembly", "verify_level3.py"),
}


def run(module_dir, script, extra):
    cmd = [PY, str(ROOT / module_dir / script)] + extra
    print(f"\n>>> {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, cwd=ROOT / module_dir)
    if res.returncode != 0:
        print(f"{script} exited with {res.returncode} — stopping.",
              file=sys.stderr)
        sys.exit(res.returncode)


def pick_random(cfg, n, seed):
    """N unlabeled episodes, spread across distinct tasks, deterministic."""
    data_dir = Path(cfg["paths"]["data_dir"])
    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    candidates = [p for p in sorted(data_dir.glob("*/*.hdf5"))
                  if not (labels_dir / f"{episode_slug(p)}.json").exists()]
    if len(candidates) < n:
        raise SystemExit(f"only {len(candidates)} unlabeled episodes left")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked, seen_tasks = [], set()
    for p in candidates:  # prefer distinct tasks, then fill up
        if p.parent.name not in seen_tasks:
            picked.append(p)
            seen_tasks.add(p.parent.name)
        if len(picked) == n:
            break
    for p in candidates:
        if len(picked) == n:
            break
        if p not in picked:
            picked.append(p)
    return [f"{p.parent.name}/{p.stem}" for p in picked]


def check_vlm_endpoint(cfg):
    import urllib.request
    url = cfg["vlm"]["endpoint_url"].rstrip("/") + "/models"
    try:
        urllib.request.urlopen(url, timeout=5)
    except Exception as e:
        raise SystemExit(
            f"vLLM endpoint {url} unreachable ({e}). Start it with:\n"
            f"  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 "
            f"vllm serve {cfg['paths']['model_path']} "
            f"--served-model-name {cfg['vlm']['model_name']} "
            f"--max-model-len 16384 --limit-mm-per-prompt '{{\"image\": 6}}' "
            f"--gpu-memory-utilization 0.92 --port 8000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default=None, help="overrides config")
    ap.add_argument("--episodes", nargs="+", required=False,
                    default=["all"],
                    help="'all', 'N_random', or explicit task/idx ids")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["paths"]["data_dir"] = args.data_dir
    levels = [int(x) for x in args.levels.split(",")]

    if args.verify_only:
        for lv in levels:
            if lv in VERIFIERS:
                run(*VERIFIERS[lv], [])
        return

    # resolve episode list
    spec = args.episodes
    if spec == ["all"]:
        data_dir = Path(cfg["paths"]["data_dir"])
        episodes = [f"{p.parent.name}/{p.stem}"
                    for p in sorted(data_dir.glob("*/*.hdf5"))]
    elif len(spec) == 1 and spec[0].endswith("_random"):
        episodes = pick_random(cfg, int(spec[0].split("_")[0]), args.seed)
    else:
        episodes = spec
    print(f"pipeline over {len(episodes)} episodes, levels {levels}")

    force = ["--force"] if args.force else []
    for lv in levels:
        if lv == 0:
            run(*LEVEL_SCRIPTS[0], [])
            continue
        if lv == 2:
            check_vlm_endpoint(cfg)
        run(*LEVEL_SCRIPTS[lv], ["--episodes"] + episodes + force)

    print("\npipeline complete.")


if __name__ == "__main__":
    main()
