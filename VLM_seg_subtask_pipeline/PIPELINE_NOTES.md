# VLM_seg_subtask_pipeline — status, findings, open questions

Working notes for the VLM-direct, two-stage (transitions -> labeling)
pipeline. Sibling of `Kinematics_pipeline` and `VLM_subtask_pipeline` — see
their `PIPELINE_NOTES.md` for the full history this pipeline's `add_remove_lid`
tuning is ported from. Update this file as findings change.

## What makes this pipeline different (kept, not collapsed)

Unlike `VLM_subtask_pipeline` (single pass: one window emits boundaries AND
labels together), this pipeline stays genuinely TWO-STAGE:
- **Level 1** (`level1_transitions/detect_transitions.py`): walks the video in
  non-overlapping windows and finds only TRANSITION frames (grasp/release/
  state-change instants) — no labels, no subtasks, just frame + short
  before/after description.
- **Level 2** (`level2_labeling/label_segments.py`): labels each segment Level
  1 produced, one at a time, in order (so each prompt carries the story so
  far) — structurally close to Kinematics_pipeline's Level 2, except there is
  no kinematic pose signal, so `hand` is asked for directly instead of
  computed.
This segment-then-subtask shape was explicitly preserved per instruction —
the upgrade below is backend/prompt/task-config, not a redesign.

## What's here (2026-07-15)

- **Repointed stale config paths** — `config.yaml` still had the original
  `/mnt/3466ADC766AD8A66/...` paths (this pipeline had literally never been
  run in this environment); fixed to match the siblings' current layout.
- **Per-task config mechanism**, ported from the sibling pipelines:
  `pipeline/task_configs/<task_name>.yaml`, merged via
  `common.apply_task_overrides()` over BOTH `level1_transitions:` and
  `level2_labeling:` blocks (this pipeline has two config blocks that
  matter, unlike the single-pass sibling's one `generate:` block). Applied
  in Level 1, Level 2, AND Level 3's `assemble.py` — the by-now-familiar
  lesson: skip any one of the three and L3's min-length assert can reject
  segments L1/L2 legitimately produced under a task override.
- **`task_configs/add_remove_lid.yaml`**: a `level1_hint` (direction-agnostic
  — the resolved instruction already carries add/remove — calibrates
  transition-count expectations: one grasp+one release pair PER CUP, don't
  merge multiple cups into one pair) plus direction-aware Level 2 hints
  (`level2_prompt_hint_by_attr: which_llm_description`) carrying forward
  everything the siblings' review rounds established: per-cup decomposition,
  cup named by position, lid-is-a-free-object framing, retract/reach
  phrasing. Also a `level1_transitions.min_segment_frames: 8` override —
  see "Fast-transition floor" below.
- **New vLLM-endpoint backend for BOTH levels**
  (`level1_transitions/transitions_vlm_backend.py`,
  `level2_labeling/labeling_vlm_backend.py` — deliberately NOT both named
  `vlm_backend.py`; see "Module name collision" below). Originally both
  levels loaded Qwen3-VL-32B in-process via `transformers` and called
  `generate()` sequentially — Level 2 additionally had NO guided decoding
  (closed vocabulary only checked post-hoc, with retries on violation).
  Now, `vlm.backend: vllm_endpoint` (default) talks to the same vLLM server
  the siblings use, with real JSON-schema-guided decoding for both levels:
  Level 1's schema is an array of `{frame, before, after}`; Level 2's is a
  single `{action, hand, object, subtask, confidence}` (has `hand` in the
  schema, unlike Kinematics_pipeline's, for the reason above). Episodes run
  CONCURRENTLY (`vlm.concurrency`, `ThreadPoolExecutor`); windows/segments
  within one episode stay sequential (Level 1 windows are independent in
  principle but processed in order for simplicity; Level 2 segments must be
  sequential for the story-so-far context). Server relaunched with
  `--limit-mm-per-prompt '{"image": 12}'` and `--max-model-len 24576`
  (shared with `VLM_subtask_pipeline` — same numbers happen to work for
  both). `transformers` backend kept as a fallback (`vlm.backend:
  transformers`), same dual-backend pattern as the sibling.
- **Hand-selection criteria added to the Level 2 prompt** — ported
  proactively from the exact failure mode found and fixed in
  `VLM_subtask_pipeline` (everything defaulting to "both" because the
  prompt gave zero criteria for when to use single/both/both_coordinating).
  Fixed here from day one instead of rediscovering it.
- **Embodiment rotation** — reuses `taxonomy.embodiment_for(episode_id)`
  directly, same function as both siblings, so an episode gets the SAME
  arm/hand term across all three pipelines (pure function of episode id).
- **`tools/review/make_review.py`**: added `--episodes <id...>` (this
  pipeline didn't have it at all before — only random `--n`/`--seed`
  selection existed) and confirmed `review_folder_name()` (task-named output
  folder, added earlier this session) works correctly.

## Module name collision (real bug, caught before it shipped)

Both levels' vLLM backend modules were initially both named `vlm_backend.py`
(one per level directory). `run_pipeline.py` and `label_segments.py` both put
BOTH directories on `sys.path` in the same process, so Python's `sys.modules`
cache could silently serve whichever backend module got imported FIRST to
*every* subsequent `import vlm_backend` anywhere in that process, regardless
of which directory the importing code lived in. Renamed to
`transitions_vlm_backend.py` / `labeling_vlm_backend.py` — globally unique
names, collision impossible. This is the same class of hazard `common.py`
already documents defensively for `common`/`keyframes` module names across
pipelines (explicit-path `importlib` loading) — worth remembering as a
pattern: **any two Python modules with the same basename, in directories that
might BOTH end up on `sys.path` in one process, are a latent collision.**

## Shared-backend-instance bug (real bug, caught by output inspection, not by an error)

`run_pipeline.py`'s vLLM path originally constructed ONE `EndpointBackend`
(Level 2's) and passed it to both `detect_boundaries()` (Level 1) and
`label_episode()` (Level 2). This is NOT the module-collision bug above — it
compiled and ran with no exception. But vLLM's guided-JSON decoding binds the
response schema at backend CONSTRUCTION time (`response_format` is fixed in
`EndpointBackend.generate()`), so every Level 1 call was silently FORCED to
emit Level 2's schema (`action/hand/object/subtask/confidence`) instead of
`{transitions: [...]}`. Level 1 code just does
`parsed.get("transitions") or []` — no such key exists in a Level-2-shaped
response, so it silently returned an empty list, every window, every
episode. Full pipeline run completed with 0 failures and confidently wrong
output: every episode reported exactly 1 segment (i.e. zero transitions
found). Caught by noticing episode 165 (known 2-cup episode from the
Kinematics review) came back with only 1 segment, and by reproducing the
SAME call manually outside the driver, where it worked correctly using
Level 1's own backend — the discrepancy between the two code paths pointed
straight at it. Fixed by constructing separate `l1_backend`/`l2_backend`
instances in both `run_pipeline.py` and inside `label_segments.py`'s own
internal fallback call to `detect_boundaries` (used when `boundaries.json`
is missing and Level 2 is run standalone without Level 1 having run first).
**Lesson for any future guided-JSON backend split across call sites: a
backend object bound to one schema must never be reused for a different
schema, even when it "should" talk to the same server** — there's no runtime
error to catch this, only silently-wrong structured output.

## Fast-transition floor (same finding as both siblings, third confirmation)

Global `level1_transitions.min_segment_frames` (30 frames / 1s) is the
sibling-comparison baseline, not tuned for this task. Measured directly on
episode 165 (2 cups, 141 frames): raw Level 2 output was a clean 7-segment
`grasp -> align -> place -> retract -> grasp -> place -> retract`
decomposition with individual phases as short as 4-17 frames — at the
30-frame floor, L3's `enforce_min_length` collapsed all of that down to 2
generic "place" segments, discarding the retract/align/grasp structure
entirely. Same root finding as both sibling pipelines on this exact task,
now confirmed a third time under a third architecture. Fixed via
`level1_transitions.min_segment_frames: 8` in the task config; re-verified
on 165 post-fix: 5 subtasks (grasp/place/retract/grasp/place), structure
preserved.

## Validated so far (2026-07-15) — NOT yet human-reviewed

Ran the full pipeline (Level 1 -> Level 2 -> Level 3, `vllm_endpoint`
backend) on the same 10-episode batch (seed 777) used across every review
round in the Kinematics sibling: `946, 1837, 1836, 1520, 2369, 1113, 1387,
2213, 165, 2562`.
- 10/10 generated and assembled, 0 failed. L3 verifier: 10/10 schema-valid,
  all invariants exact; 102 total subtasks, 4.9% flagged for review, verb
  histogram dominated by place/retract/align/reach/grasp — the expected
  per-cup-cycle vocabulary.
- Spot-checked output (not a substitute for human review): 1836 shows clean
  per-cup decomposition (pick/place/retract per cup) with cup positions
  named and consistent embodiment term.
- **Episode `2562`** (independently confirmed in both sibling pipelines'
  review/spot-check to be a data-integrity bug — hdf5 metadata claims
  cups/lids, actual video is someone unwrapping a box of snacks to reveal a
  chessboard) is again correctly described here: "pick up the clear plastic
  bag with green contents", "open the foldable wooden chess board" — third
  independent architecture now catching the same mismatch on its own,
  reinforcing that this is a real corpus-quality issue worth a cheap
  full-corpus check, not an artifact of any one pipeline's design.
- Review page built: `outputs/review/add_remove_lid/vlm_seg_subtask_pipeline_review.html`,
  same 10 episodes, 4 auto-flagged (946, 1387, 2213, 2562). NOT YET graded
  by a human — that's the next step, same as both siblings. Don't assume
  output quality beyond the spot-check above without that pass.

## Known gaps / things NOT ported or NOT yet done

- **No human review of this pipeline's output yet** — the biggest gap, same
  as every pipeline at this stage of first setup.
- **`verify_level1.py` / `verify_level2.py` do not call
  `apply_task_overrides`** — same dormant-gap shape already flagged in
  `VLM_subtask_pipeline`'s notes; not yet triggering a bug (nothing reads
  min-length there), but fix before relying on those verifiers for a task
  with numeric overrides.
- **`embodiment` is not carried into the final schema** — lives only in the
  raw `labels/*.json` record, consistent with both siblings (not a
  regression here specifically).
- **`transformers` backend not exercised at all after the vLLM port** —
  unlike `VLM_subtask_pipeline` (which at least import/signature-checked
  it), this pipeline's transformers path has only been read, not run, since
  the edits. Lower risk (logic largely untouched, only renamed/reorganized)
  but unverified.
- **Level 1's window-independence is not exploited** — windows don't
  actually depend on each other's transitions (no story-so-far mechanism
  for Level 1, only Level 2), so they could in principle run CONCURRENTLY
  within one episode too, not just sequentially. Left sequential to match
  the original design and minimize scope; a possible speedup if this
  pipeline's Level 1 throughput ever matters at full-corpus scale.
- **No corpus-wide Level-1-only sweep** (the Kinematics sibling has one for
  free, no VLM cost; VLM_subtask_pipeline doesn't have one either, for the
  same reason — Level 1 here also requires real VLM calls, so a full-corpus
  structural sweep costs real time/tokens, not just CPU).
