# Experiment log

Chronological record of the EgoDex subtask-annotation pipeline's
development: what was built, what was measured, what broke, and what
changed as a result. Each file is a snapshot at the time it was written —
check `pipeline/config.yaml` and `pipeline/task_categories.yaml` for current
values, since thresholds have moved on since some of these were logged.

| # | Experiment | Status |
|---|---|---|
| [01](01_level0_data_audit.md) | Level 0 — dataset audit | PASS, 8/8 checks |
| [02](02_level1_kinematic_boundaries.md) | Level 1 — kinematic boundary proposal | PASS, tuned twice |
| [03](03_level2_vlm_labeling.md) | Level 2 — Qwen3-VL-32B labeling | PASS, 3 quality fixes applied |
| [04](04_level3_assembly_and_e2e_test.md) | Level 3 + first end-to-end run (20 episodes) | PASS, 0 failures |
| [05](05_task_categories_and_prompt_optimization.md) | Kinematic task categories + prompt hints | 111/111 tasks mapped |
| [06](06_pick_place_200_episode_run.md) | Production run: 200 pick_place episodes | Done, open question on flag calibration |
| [07](07_human_review_tool.md) | Human verification tool | Built, awaiting first reviewed batch |

## Reproducing any run

```bash
conda activate egodex
cd pipeline
python run_pipeline.py --episodes <ids | N_random | --category NAME --n N> --levels 1,2,3
python run_pipeline.py --verify-only --levels 3
```

Serve the VLM first (see the repo README) — Level 2 checks the endpoint and
prints the launch command if it's down.
