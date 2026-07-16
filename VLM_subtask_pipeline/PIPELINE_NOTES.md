# VLM_subtask_pipeline — status, findings, open questions

Working notes for the single-pass VLM subtask pipeline. Sibling of
`Kinematics_pipeline` (see its `PIPELINE_NOTES.md` for the full history of
findings this pipeline's `add_remove_lid` tuning is ported from). Update this
file as findings change.

## What's here (2026-07-15)

- **Per-task config mechanism**, ported from Kinematics_pipeline:
  `pipeline/task_configs/<task_name>.yaml`, one file per task, absent = untouched
  defaults. `common.load_task_config()` / `common.apply_task_overrides()` merge
  a task's `generate:` block over the global one. **Both** `generate_subtasks.py`
  (Level 1) and `assemble.py` (Level 3) call `apply_task_overrides` — same
  lesson learned the hard way in the Kinematics sibling: if only one side
  applies a task override, L3's min-length assert can reject segments L1
  legitimately produced under a different floor.
- **`task_configs/add_remove_lid.yaml`**: direction-aware prompt hints
  (`prompt_hint_by_attr: which_llm_description`, `"1"`=add / `"2"`=remove),
  carrying forward everything the Kinematics sibling's 5 human-review rounds
  established: per-cup decomposition, cut-at-release (not at the end of the
  hand's empty retrace), cup-position naming, lid-is-a-free-object framing.
  No `generate:` override yet (window geometry untouched) — unlike the
  Kinematics sibling, this hasn't been human-review-tuned for this task yet,
  only prompt-ported.
- **Embodiment rotation**: reuses `taxonomy.embodiment_for(episode_id)` from
  the Kinematics sibling directly (same crc32-threshold function, same
  ~40%-arm/60%-hand-per-episode split) — an episode gets the SAME term in
  BOTH pipelines, since it's a pure function of the episode id. Wired into
  `build_prompt`/`build_prompt_content` for both backends.
- **New vLLM-endpoint backend** (`level1_generate/vlm_backend.py`), now the
  *default* (`vlm.backend: vllm_endpoint` in config.yaml). The original
  implementation loaded Qwen3-VL-32B in-process via `transformers` and called
  `model.generate()` once per window, sequentially, no batching — much
  slower than the Kinematics sibling's vLLM-served approach (no continuous
  batching, no concurrent requests, no paged/reused KV cache). The new
  backend:
  - Talks to the SAME vLLM server the sibling pipelines use, over HTTP
    (OpenAI-compatible `/v1/chat/completions`), with JSON-schema-guided
    decoding for structured output — same mechanism proven in the
    Kinematics sibling's Level 2, extended from "one label object" to "an
    array of subtask objects" (a window can contain several).
  - Sends **keyframe images**, not raw video: extracts N evenly-spaced
    frames per window via `common.extract_keyframes` (imported straight
    from the Kinematics sibling's `keyframes.py` — the same seek-decode
    function, reused as-is) and sends them as a base64 image list. This
    is a deliberate choice over vLLM's native `video_url` support, which
    exists in this vLLM version but has never been exercised against this
    Qwen3-VL-32B checkpoint in this environment — going with the already-
    validated pattern instead of a cold, untested integration surface.
  - Decouples `generate.model_frames` (window SOURCE SPAN / candidate grid,
    unchanged meaning) from the new `vlm.keyframes_per_window` (how many of
    those candidates actually become images sent to the model, evenly
    subsampled, always keeping first/last). Needed because the server's
    `--limit-mm-per-prompt` must bound the image count — relaunched with
    `{"image": 12}` and `--max-model-len 24576` (up from the sibling's
    6-image / 16384 config, since this pipeline needs more images per
    request to cover a wide, structurally-unknown window instead of a few
    keyframes of an already-known single segment).
  - Runs episodes CONCURRENTLY (`vlm.concurrency`, default 4,
    `ThreadPoolExecutor` over episodes in `run_pipeline.py`) — windows
    *within* one episode stay sequential, since each window's prompt
    depends on the stitched story-so-far of the previous ones (same
    constraint as the Kinematics sibling's per-segment sequencing).
  - `DebugBudget` made thread-safe (was a bare counter; race-prone once
    episodes run concurrently).
  - The original `transformers` backend is KEPT (not deleted) as
    `vlm.backend: transformers` for A-B comparison / fallback — renamed
    `generate_window` -> `generate_window_transformers`, otherwise
    unchanged logic. Only lightly smoke-tested (one episode, pre-refactor);
    not re-validated after the backend-dispatch refactor beyond an
    import/signature check.
  - **Only one Qwen3-VL-32B should be resident on the GPU at a time.** The
    transformers-direct smoke test required stopping the Kinematics
    sibling's vLLM server first (94GB already in use, no room for a second
    copy of the same 32B weights). Whoever runs these pipelines needs to
    coordinate which one holds the model.
- **`tools/review/make_review.py`**: added `--episodes <id ...>` (same port
  as the Kinematics sibling) to target an exact episode list instead of only
  random `--n`/`--seed` sampling — needed to review the SAME 10 episodes
  across pipelines for a fair comparison. Distinct localStorage key
  (`egodex_vlm_subtask_review_v1`) already existed, so verdicts don't
  collide with the sibling reviewers in one browser.

## Validated so far (2026-07-15) — NOT yet human-reviewed

Ran the full pipeline (Level 1 vllm_endpoint -> Level 3) on the same 10-episode
batch (seed 777) already used across 5 human-review rounds in the Kinematics
sibling: `946, 1837, 1836, 1520, 2369, 1113, 1387, 2213, 165, 2562`.
- 10/10 generated, 0 failed; L3 verifier: 10/10 schema-valid, all invariants
  (coverage, contiguity, duration-arithmetic) exact.
- Spot-checked output quality (not a substitute for human review, but a
  sanity pass): direction correct, cup positions named
  ("leftmost"/"third cup from the left"), embodiment term consistent within
  each episode, and the retract/reach decomposition between consecutive
  cups' cycles emerges without any window-geometry tuning — encouraging,
  since that exact pattern took the Kinematics sibling multiple fix rounds
  (release-priority, per-signal masks) to reach.
- **Episode `2562`** (independently confirmed in the Kinematics sibling's
  review to be a data-integrity bug — hdf5 metadata claims cups/lids, actual
  video is someone unwrapping a box of snacks to reveal a chessboard) got
  correctly described here: "unfold the dark red box lid to reveal the
  chessboard inside." This pipeline has to interpret raw frames to find
  boundaries at all, so it surfaces this kind of data mismatch on its own —
  the Kinematics sibling only caught it because a human happened to pull raw
  frames manually. Worth treating as a real strength of this architecture:
  a fast, cheap "does this episode's content match its task label" check
  could run this pipeline once over new data before deeper processing.
- Review page built: `outputs/review/index.html`, same 10 episodes, 2
  auto-flagged (`2213`, `2562`). NOT YET graded by a human — that's the
  next step, exactly mirroring the Kinematics sibling's iterative
  review -> root-cause -> fix -> re-review loop. Do not assume this
  pipeline's output quality without that pass; the spot-check above is
  informal.

## `hand` field overusing "both" (2026-07-15, fixed)

User observation reviewing the page: too many subtasks labeled `hand=both`
even where visually only one hand seemed to be doing the work. Root cause,
confirmed by pulling actual frames for several `both`-labeled segments:
unlike the Kinematics sibling (where `hand` is COMPUTED from real 3D
wrist/aperture kinematics and the VLM only flags disagreement), this
pipeline has the VLM guess `hand` from pixels with **zero criteria** — the
prompt just listed `left, right, both, both_coordinating` with no
instruction on when to use which. Supporting evidence: `both_coordinating`
was used 0/100 times despite being in the vocabulary for exactly this case
(hand-to-hand pass, joint grip) — everything collapsed into the flatter
`both`.

Fix: added explicit selection criteria to the prompt (both backends) —
single hand if only one is actively manipulating (a resting/idle/nearby
hand doesn't count), `both` only if both are independently manipulating,
`both_coordinating` only if both work as one unit on the same grip/motion.
Re-ran the same 10-episode batch: `both` 72->65, single-hand (right+left)
28->34, `both_coordinating` 0->1. Modest, not dramatic — spot-checking
frames myself beforehand found several `both` labels genuinely correct
(these episodes' demonstrator does pinch small paper lids with two hands
together often), so a large swing wasn't expected or necessarily desirable.
Not yet human-reviewed post-fix — that's the pending step.

## Chunked (non-overlapping) generator — a PARALLEL pipeline (2026-07-15)

Added `generate_subtasks_chunked.py` + `chunked_vlm_backend.py` +
`config_chunked.yaml` + `run_pipeline_chunked.py` as a second, independent
generator alongside the sliding-window one above — explicitly requested as a
parallel process, so **none of the files above were modified**; everything
reusable (taxonomy prompt block, JSON parsing, candidate validation,
keyframe subsampling, `DebugBudget`, L3 assembly/postprocess/verify) is
imported read-only from the existing modules, addressed via a separate
`config_chunked.yaml` (own `paths.output_dir: outputs_chunked/`, so the two
generators' label files never collide) that `assemble.py` / `verify_*.py`
accept unmodified via `--config`.

**The one real design difference**: the sliding-window generator re-shows a
fraction of each window to the next one so a boundary-crossing subtask stays
visually coherent (the model re-derives continuity by literally re-seeing
the frames). This generator shows each source frame to the model EXACTLY
ONCE — fixed 48-frame chunks, back to back, no re-showing (`frame_skip: 0`
by default here, vs. the sibling's `1`, since there's no overlapping
neighbor to backfill a skipped frame from). Since the model never re-sees a
frame, cross-chunk coherence has to come entirely from TEXT: each prompt
carries the normal "story so far" PLUS an explicit continuation handoff —
if the previous chunk ended mid-subtask (`"ongoing": true`), the prompt
states its exact `action`/`object`/true `start_frame` and instructs the
model to either report that SAME subtask again (reusing the given
start_frame) or start the NEXT subtask at this chunk's first frame. That
continuation-vs-next call is the one thing this prompt has to do that the
sliding-window one doesn't.

Stitching is correspondingly different: no global IOU clustering across all
candidates (nothing to cluster — chunks never overlap by construction).
Instead `process_chunk_candidates()` runs a single incremental pass, folding
each chunk's result into a running `(segments, open_tail)` state as chunks
arrive, in order — O(n) instead of the sibling's O(n²) global stitch.

**Two real bugs found via smoke-testing episode 165 (2 cups, the same
episode used throughout this session), both fixed before the batch run**:
1. **Continuation matching on label TEXT was wrong.** First attempt matched
   a chunk's first candidate against the open tail by
   `action == action and normalized_object == normalized_object`. Real
   example: chunk 1 left `place` open on `"the round white lid near the
   left cup"`; chunk 2 correctly continued it (same start_frame, per the
   prompt) but reworded the object to `"the round white lid onto the
   leftmost cup"` — different string, so the text match rejected it as "not
   the same subtask," producing a duplicate: `26-47 place` (old, wrongly
   finalized) AND `26-74 place` (new, wrongly treated as fresh), a visible
   9-frame overlap in the output. Fixed by dropping text matching entirely
   and using frame arithmetic instead: a FRESH subtask can never start
   before this chunk's own `w_start` (chunks are disjoint by construction),
   so `first["start_frame"] < w_start` is a sufficient and much more robust
   continuation signal than string equality — whatever wording the model
   used this time simply wins (fresher read), only the true `start_frame`
   is taken from the story, not re-trusted from the model's echo.
2. **Intra-chunk overlap, no cross-check within one response.** Even after
   fix #1, episode 165 still showed a 9-frame overlap: chunk `[48,95]`'s
   own single JSON response listed BOTH a `retract` ending at 95 AND a
   `reach` spanning `86-95` — two candidates that overlap EACH OTHER within
   one model response. The code only ever checked a chunk's FIRST candidate
   against the previous chunk's open tail; nothing validated ordering
   between candidates 2..N of the same response. Fixed with
   `_resolve_intra_chunk_overlaps()`: a single sequential clamp pass over
   the chunk's own candidate list (clamp each candidate's start to the
   previous one's end; drop it if that makes it empty or it was fully
   contained) — same overlap resolution the sibling's `stitch()` does
   across windows, just one linear pass since a chunk's own list is already
   chronological, no cross-window merging needed.

**Validated**: 4-episode batch (`165, 1387, 946, 2562`) through
`run_pipeline_chunked.py --levels 1,3 --force`, 0 failed at generation, 0
failed/skipped at assembly. `verify_level3.py --config config_chunked.yaml`:
4/4 schema-valid, invariants + duration arithmetic exact, coverage
contiguous on every episode (spot-checked directly: no gaps, no overlaps
post-fix). Episode `165` (2 cups) → clean 7-segment per-cup decomposition.
Episode `1387` (7 cups, 449 frames, 10 chunks) → clean per-cup
reach/align/place/retract cycles spanning chunk boundaries correctly.
Episode `2562` (the known chess-box/bag data-integrity episode) is
independently caught here too: "grasps the clear plastic bag..." / "unfold
the...chessboard" — third-plus confirmation across pipelines that this is a
real corpus issue, not a pipeline artifact.

**Not done / known gaps for this generator specifically**:
- No human review yet (same stage as the sliding-window sibling was at
  before its own review rounds) — `make_review.py` was not touched or
  pointed at `outputs_chunked/` yet; would need `--config config_chunked.yaml`
  passed through (it already accepts `--config`, untested against this
  output tree) and its own care around the shared localStorage key if
  reviewed in the same browser as the sibling.
- `frame_skip: 0` / `model_frames: 48` / `min_segment_frames: 10` in
  `config_chunked.yaml` are defaults carried from the sliding-window
  sibling's un-tuned starting point, not yet swept or reviewed for this
  generator's own boundary granularity.
- Environment note: the `qwen3vl` conda env had `qwen_vl_utils` but not
  `openai` (needed by any `vllm_endpoint` backend, including the
  already-existing sliding-window one) — `pip install openai` was required
  before ANY vllm_endpoint pipeline could run in this env. Not a code
  change, just an environment gap worth knowing about if a fresh shell hits
  the same `ModuleNotFoundError`.
- `transformers` backend implemented (mirrors the sibling's dual-backend
  pattern) but not exercised at all — only the `vllm_endpoint` path was run.

## Known gaps / things NOT ported or NOT yet done

- **No human review of this pipeline's output yet.** Everything above is my
  own spot-check, not a substitute for the actual review process. This is
  the single biggest gap before trusting this pipeline's `add_remove_lid`
  output.
- **Window geometry is untuned for `add_remove_lid`.** The Kinematics
  sibling needed 5 review rounds to reach good boundary quality via several
  kinematic-threshold fixes; this pipeline has ONLY had the prompt hint
  ported over, not an equivalent tuning pass on `frame_skip`/`overlap_frac`/
  `model_frames`/`stitch.iou_merge`. If review finds boundary problems, this
  is where to look — and the fix mechanism (a `generate:` block in this
  task's yaml) already exists, unused so far.
  - `vlm.keyframes_per_window` / `vlm.keyframe_size` are NOT yet
    task-overridable — `apply_task_overrides` only merges the `generate:`
    block, not `vlm:`. If a task needs a different image budget or
    resolution, extending the merge to `vlm:` is a small, direct change
    (same pattern), just not built since no task has needed it yet.
- **`verify_generate.py` does not call `apply_task_overrides`** — reads
  `cfg["generate"]["min_segment_frames"]` from the global config only. Not
  currently triggering a bug (no task has a `generate:` override yet), but
  it's the exact shape of gap that broke L3 in the Kinematics sibling the
  first time a task override was added — fix this BEFORE adding a
  `generate:` override to any task config, not after hitting the failure.
- **`embodiment` is not carried into the final schema** (`EpisodeAnnotation`/
  `Subtask` in `schema.py`) — it lives only in the raw `labels/*.json`
  record, same as the Kinematics sibling (confirmed, not a regression
  introduced here). If training needs to filter/balance by embodiment term
  from the final `annotations/*.json`, it needs adding to both pipelines'
  schemas, not just this one.
- **`transformers` backend not re-validated after the refactor** — only an
  import/signature check was done post-refactor (the one working end-to-end
  run was before the backend-dispatch split). Low risk (mechanical rename,
  logic untouched) but not proven the way the vllm_endpoint path is.
- **No corpus-wide Level-1-only structural sweep** for this pipeline (the
  Kinematics sibling has one: all 2,565 episodes run through boundary
  detection alone, no VLM cost, checked segment-count-vs-cup-count sanity).
  No equivalent script exists here yet — would need one that calls
  `generate_window_vllm` (or a cheaper proxy) across the corpus; unlike the
  Kinematics sibling's kinematics-only sweep, this pipeline can't produce
  boundaries without VLM calls, so a full-corpus sweep here costs real VLM
  time, not just CPU.
- **vLLM server coordination is manual.** Both this pipeline and the
  Kinematics sibling assume the SAME server process/port serves both, and
  only one 32B model fits in GPU memory. No lock/check prevents two
  pipelines' `run_pipeline.py` invocations from colliding if run at the
  same time by different people/sessions.
