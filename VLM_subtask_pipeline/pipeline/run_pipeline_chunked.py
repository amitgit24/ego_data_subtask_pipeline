"""End-to-end driver for the CHUNKED (non-overlapping) single-pass subtask
pipeline -- parallel to run_pipeline.py (which drives the sliding-window
generator; that file is untouched). See generate_subtasks_chunked.py's
docstring for what's different.

Reuses level3_assembly/assemble.py, level1_generate/verify_generate.py, and
level3_assembly/verify_level3.py UNCHANGED via --config config_chunked.yaml
-- they only read cfg[...] keys, all of which config_chunked.yaml also
defines under the same names. Separate paths.output_dir keeps every artifact
of this pipeline out of the sliding-window sibling's tree, so both can be
run against the same source data / vLLM server without collision.

    python run_pipeline_chunked.py --episodes <all | N_random | task/idx [...]>
                                   [--levels 1,3] [--force] [--verify-only]
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import episode_id, episode_slug, load_config, sample_episodes

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_CONFIG = ROOT / "config_chunked.yaml"

VERIFIERS = {
    1: ("level1_generate", "verify_generate.py"),
    3: ("level3_assembly", "verify_level3.py"),
}


def run_subprocess(module_dir, script, extra, config_path):
    cmd = [PY, str(ROOT / module_dir / script), "--config", str(config_path)] + extra
    print(f"\n>>> {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, cwd=ROOT / module_dir)
    if res.returncode != 0:
        print(f"{script} exited with {res.returncode} — stopping.", file=sys.stderr)
        sys.exit(res.returncode)


def run_generate(paths, cfg, force):
    sys.path.insert(0, str(ROOT / "level1_generate"))
    from generate_subtasks_chunked import DebugBudget, process_episode

    labels_dir = Path(cfg["paths"]["output_dir"]) / "labels"
    todo = [p for p in paths
           if force or not (labels_dir / f"{episode_slug(p)}.json").exists()]
    for p in paths:
        if p not in todo:
            print(f"skip (labels exist): {episode_id(p)}")

    debug_budget = DebugBudget(cfg["generate"]["debug_first_n"])
    done, failed = 0, []

    def _run_one(p, backend):
        return process_episode(backend, p, cfg, debug_budget)

    if cfg["vlm"]["backend"] == "vllm_endpoint":
        from chunked_vlm_backend import EndpointBackend, check_vlm_endpoint
        print(f"\n>>> checking vLLM endpoint {cfg['vlm']['endpoint_url']} ...")
        check_vlm_endpoint(cfg)
        backend = ("vllm_endpoint", EndpointBackend(cfg))
        with ThreadPoolExecutor(max_workers=cfg["vlm"]["concurrency"]) as pool:
            futures = {pool.submit(_run_one, p, backend): p for p in todo}
            for fut, p in futures.items():
                try:
                    rec = fut.result()
                    done += 1
                    print(f"[{done}] {rec['episode_id']}: {rec['num_windows']} "
                         f"chunks -> {len(rec['segments'])} segments")
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
                rec = _run_one(p, backend)
                done += 1
                print(f"[{done}] {rec['episode_id']}: {rec['num_windows']} "
                     f"chunks -> {len(rec['segments'])} segments")
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

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    cfg = load_config(str(config_path))
    levels = [int(x) for x in args.levels.split(",")]

    if args.verify_only:
        for lv in levels:
            if lv in VERIFIERS:
                run_subprocess(*VERIFIERS[lv], [], config_path)
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
    print(f"chunked pipeline over {len(episodes)} episodes, levels {levels}")

    data_dir = Path(cfg["paths"]["data_dir"])
    paths = [data_dir / f"{e}.hdf5" for e in episodes]

    if 1 in levels:
        run_generate(paths, cfg, args.force)
    if 3 in levels:
        force = ["--force"] if args.force else []
        run_subprocess("level3_assembly", "assemble.py",
                       ["--episodes"] + episodes + force, config_path)

    print("\npipeline complete.")


if __name__ == "__main__":
    main()
