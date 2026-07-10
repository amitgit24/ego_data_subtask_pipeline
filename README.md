# EgoDex Subtask Annotation Pipeline

A 3-level pipeline that converts [EgoDex](https://arxiv.org/abs/2505.11709) episodes
(MP4 video + HDF5 3D hand poses) into schema-validated per-episode subtask
annotations: frame-accurate boundaries, a closed action taxonomy, the acting
hand (computed from kinematics), the manipulated object, and a style-varied
one-sentence description. Built to generate VLM training data for VLA models.

## Architecture

- **Level 0 — data audit** (`level0_audit/`): verifies the real HDF5/MP4
  structure (key paths, world coordinate frame, video↔pose sync, missing-data
  encoding) before any pipeline code trusts it. Confirmed facts live in
  `config.yaml`.
- **Level 1 — kinematic boundary proposal** (`level1_kinematics/`): candidate
  subtask boundaries purely from hand-pose signals — wrist-speed pauses,
  pinch-aperture grasp/release events, gaze shifts. No vision model.
- **Level 2 — VLM semantic labeling** (`level2_vlm/`): Qwen3-VL-32B labels
  each segment from 3–5 keyframes + kinematic context, served locally via
  vLLM with JSON-schema-guided decoding. Segments are labeled sequentially so
  each prompt carries the episode story so far; the hand is computed from
  kinematics, never by the VLM. The VLM never chooses frame boundaries.
- **Level 3 — validation & assembly** (`level3_assembly/`): merges duplicate
  adjacent segments, cross-checks semantics against kinematics (flags, never
  deletes), computes all timing in code, and emits pydantic-validated JSON
  with a review queue.

Each level has a verifier (`verify_levelN.py`) that must pass before the next
level runs. `run_pipeline.py` drives everything end to end and is resumable.

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
cd egodex_annotator
python run_pipeline.py --episodes 20_random --levels 1,2,3
python run_pipeline.py --verify-only --levels 3
```

Paths, HDF5 key conventions, and every threshold live in
`egodex_annotator/config.yaml`.

## Notes

- Measured throughput: ~65 labeled segments/min on one RTX PRO 6000 Blackwell
  (96 GB), episode-level concurrency 8.
- On Blackwell (sm_120) GPUs, `VLLM_USE_FLASHINFER_SAMPLER=0` is required with
  vLLM 0.24 (FlashInfer capability probe fails), and single-GPU serving is
  preferred over tensor parallel (NCCL P2P hangs on workstation boards).
- The dataset itself (`test/`) and all pipeline outputs are gitignored; only
  code and configuration are tracked.
