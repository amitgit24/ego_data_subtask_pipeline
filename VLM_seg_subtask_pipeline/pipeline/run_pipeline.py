"""End-to-end pipeline driver for the VLM-direct (transitions -> labeling)
pipeline.

    python run_pipeline.py --episodes <all | N_random | task/idx [...]>
                           [--levels 1,2,3] [--force] [--verify-only]

Levels 1 and 2 both need Qwen3-VL, via one of two backends
(cfg["vlm"]["backend"]):
  - vllm_endpoint (default, since 2026-07-15): talks to a vLLM server over
    HTTP, same pattern as the sibling pipelines — episodes run CONCURRENTLY
    (cfg["vlm"]["concurrency"] workers). Within one episode, Level 1's
    windows and Level 2's segments both stay sequential (Level 2 labels
    segments in order so each prompt carries the story so far). Only one
    Qwen3-VL-32B should be resident on the GPU at a time — stop any other
    pipeline's server/process holding the model first.
  - transformers: loads the model in-process once, generates sequentially,
    episode by episode, no batching (the original implementation; much
    slower, kept as a fallback / for A-B comparison against the endpoint
    backend). Still loads the model ONCE and runs both levels in one
    process per episode, exactly as before.
Level 3 has no model and runs as a lightweight subprocess, same as
Kinematics_pipeline.

Levels are resumable: each level skips episodes whose output already exists
(unless --force).
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
    """Run Level 1 (if its output is missing) and Level 2 for each episode."""
    sys.path.insert(0, str(ROOT / "level1_transitions"))
    sys.path.insert(0, str(ROOT / "level2_labeling"))
    from detect_transitions import process_episode as detect_boundaries
    from label_segments import DebugBudget
    from label_segments import process_episode as label_episode

    boundaries_dir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    todo = [p for p in paths
           if force or not (labels_dir / f"{episode_slug(p)}.json").exists()]
    for p in paths:
        if p not in todo:
            print(f"skip (labels exist): {episode_id(p)}")

    debug_budget = DebugBudget(cfg["level2_labeling"]["debug_first_n"])
    done, failed = 0, []

    def _run_one(p, l1_backend, l2_backend):
        import json
        bfile = boundaries_dir / f"{episode_slug(p)}.json"
        if not bfile.exists() or force:
            result = detect_boundaries(l1_backend, p, cfg)
            bfile.write_text(json.dumps(result, indent=2))
            print(f"  L1 {result['episode_id']}: {result['n_frames']} frames "
                 f"-> {len(result['segments'])} segments")
        return label_episode(l2_backend, p, cfg, debug_budget)

    if cfg["vlm"]["backend"] == "vllm_endpoint":
        from transitions_vlm_backend import EndpointBackend as L1Backend
        from transitions_vlm_backend import check_vlm_endpoint
        from labeling_vlm_backend import EndpointBackend as L2Backend
        print(f"\n>>> checking vLLM endpoint {cfg['vlm']['endpoint_url']} ...")
        check_vlm_endpoint(cfg)
        # Level 1 and Level 2 are guided-JSON-decoded against DIFFERENT
        # schemas (transitions array vs. single action/hand/object/
        # subtask/confidence) -- each needs its OWN EndpointBackend
        # instance (schema is bound at construction). Sharing one silently
        # forces every response into whichever schema won, with no error
        # (found the hard way: Level 1 calls came back with zero
        # transitions, every time, because the shared backend was
        # Level-2-shaped).
        l1_backend = ("vllm_endpoint", L1Backend(cfg))
        l2_backend = ("vllm_endpoint", L2Backend(cfg))
        with ThreadPoolExecutor(max_workers=cfg["vlm"]["concurrency"]) as pool:
            futures = {pool.submit(_run_one, p, l1_backend, l2_backend): p
                      for p in todo}
            for fut, p in futures.items():
                try:
                    out = fut.result()
                    done += 1
                    print(f"[{done}] L2 {out['episode_id']}: "
                         f"{len(out['segments'])} segments labeled")
                except Exception as e:
                    failed.append((episode_id(p), repr(e)))
                    print(f"FAILED {episode_id(p)}: {e!r}", file=sys.stderr)
    else:
        from model_backend import load_model
        print(f"\n>>> loading {cfg['paths']['model_path']} ...")
        model, processor = load_model(cfg["paths"]["model_path"],
                                      cfg["model"]["attn_implementation"],
                                      cfg["model"]["dtype"])
        backend = ("transformers", model, processor)
        for p in todo:
            try:
                out = _run_one(p, backend, backend)
                done += 1
                print(f"[{done}] L2 {out['episode_id']}: "
                     f"{len(out['segments'])} segments labeled")
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
