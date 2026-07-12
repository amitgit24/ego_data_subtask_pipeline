# EgoDex Subtask Annotation — Kinematics Pipeline

A 3-level pipeline that converts [EgoDex](https://arxiv.org/abs/2505.11709) episodes
(MP4 video + HDF5 3D hand poses) into schema-validated per-episode subtask
annotations: frame-accurate boundaries, a closed action taxonomy, the acting
hand (computed from kinematics), the manipulated object, and a style-varied
one-sentence description. Built to generate VLM training data for VLA models.

This is the **kinematics-first** approach: subtask boundaries come purely
from 3D hand-pose signals, and the VLM only labels segments it's handed — it
never chooses where to cut. `Qwen_direct_pipeline/` (sibling folder) is a
second, simpler approach that asks Qwen3-VL to find segment boundaries
itself directly from dense video windows, for comparison — built and
verified end to end on one episode so far, not yet run at scale.

## Repo layout

```
test/                        EgoDex dataset (gitignored) — shared across pipelines
Kinematics_pipeline/           this pipeline
  pipeline/                    all pipeline code (Levels 0–3), config, task categories
  tools/review/                 human verification UI (segment-clip review + scoring)
  experiments/                 one markdown file per experiment: what was run, what
                               was measured, what changed and why — read this first
  outputs/                     all generated artifacts (gitignored) — audits,
                               boundaries, labels, annotations, review media
Qwen_direct_pipeline/          sibling project: VLM-only segmentation
  pipeline/                    Levels 1–3 (no Level 0 audit — reuses Kinematics_pipeline's)
  tools/review/                 its own copy of the reviewer (same verdict axes, for a
                               fair head-to-head; separate localStorage key)
  ref_code/                    the original reference script this pipeline was built from
  outputs/                     gitignored, same shape as Kinematics_pipeline's
```

Each pipeline's `tools/review/` is a self-contained copy — same review
protocol and UI on both, adapted only for where each pipeline's annotations
live and what its flag vocabulary is (see
`Qwen_direct_pipeline/pipeline/level3_assembly/schema.py` for how its flags
differ from this pipeline's kinematic cross-checks).

See `Kinematics_pipeline/experiments/README.md` for the chronological
build/validation log of this pipeline.

## Architecture

- **Level 0 — data audit** (`pipeline/level0_audit/`): verifies the real
  HDF5/MP4 structure (key paths, world coordinate frame, video↔pose sync,
  missing-data encoding) before any pipeline code trusts it. Confirmed facts
  live in `pipeline/config.yaml`.
- **Level 1 — kinematic boundary proposal** (`pipeline/level1_kinematics/`):
  candidate subtask boundaries purely from hand-pose signals — wrist-speed
  pauses, pinch-aperture grasp/release events, gaze shifts. No vision model.
- **Level 2 — VLM semantic labeling** (`pipeline/level2_vlm/`): Qwen3-VL-32B
  labels each segment from 3–5 keyframes + kinematic context, served locally
  via vLLM with JSON-schema-guided decoding. Segments are labeled
  sequentially so each prompt carries the episode story so far; the hand is
  computed from kinematics, never by the VLM. Prompts are also
  category-aware — see `pipeline/task_categories.yaml`. The VLM never
  chooses frame boundaries.
- **Level 3 — validation & assembly** (`pipeline/level3_assembly/`): merges
  duplicate adjacent segments, cross-checks semantics against kinematics
  (flags, never deletes), computes all timing in code, and emits
  pydantic-validated JSON with a review queue.

Each level has a verifier (`verify_levelN.py`) that must pass before the
next level runs. `pipeline/run_pipeline.py` drives everything end to end and
is resumable.

## Usage

```bash
conda activate egodex
pip install -r requirements.txt

# serve the VLM (single GPU; see notes below)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve <path-to>/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 6}' --gpu-memory-utilization 0.92 --port 8000

# run the pipeline
cd Kinematics_pipeline/pipeline
python run_pipeline.py --episodes 20_random --levels 1,2,3
python run_pipeline.py --category pick_place --n 200 --levels 1,2,3
python run_pipeline.py --verify-only --levels 3

# check task-category coverage (all 111 tasks must map to exactly one)
python check_categories.py
```

Paths, HDF5 key conventions, and every threshold live in
`Kinematics_pipeline/pipeline/config.yaml`. Kinematic task-family
definitions (motion signature, prompt hints, verb priors) live in
`Kinematics_pipeline/pipeline/task_categories.yaml`.

### Human review tool

```bash
cd Kinematics_pipeline/tools/review
python make_review.py --category pick_place --n 20
# open ../../outputs/review/index.html in a browser, judge boundary/label/
# sentence per segment + episode coherence, Export verdicts JSON
python report_verdicts.py ~/Downloads/review_verdicts.json
```

`Qwen_direct_pipeline/tools/review/` works the same way (no `--category`
flag, since that pipeline has no task-category concept) and writes its
export as `qwen_direct_review_verdicts.json` under a distinct localStorage
key, so reviewing both pipelines in the same browser doesn't collide.

## Notes

- Measured throughput: ~65–70 labeled segments/min on one RTX PRO 6000
  Blackwell (96 GB), episode-level concurrency 8.
- On Blackwell (sm_120) GPUs, `VLLM_USE_FLASHINFER_SAMPLER=0` is required with
  vLLM 0.24 (FlashInfer capability probe fails), and single-GPU serving is
  preferred over tensor parallel (NCCL P2P hangs on workstation boards).
- The dataset (`test/`) and all pipeline outputs (`Kinematics_pipeline/outputs/`)
  are gitignored; only code, configuration, and experiment docs are tracked.
