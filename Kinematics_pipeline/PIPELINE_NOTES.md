# Kinematics pipeline — status, findings, open questions

Working notes on the kinematics-driven subtask segmentation pipeline for
EgoDex. Purpose: track what's been measured, what's still a hypothesis, and
what to check before making changes. Update this file as findings change —
don't let it go stale.

## Where things live

- **Human review verdicts**: `Kinematics_pipeline/review_verdicts.json` —
  exported from the review UI (`tools/review/make_review.py` builds the UI at
  `outputs/review/index.html`; the Export button in that page downloads this
  JSON). Scored with `tools/review/report_verdicts.py review_verdicts.json`.
- **Annotations** (Level 1+2+3 pipeline output, one file per episode):
  `outputs/annotations/{task}__{idx}.json`.
- **Task taxonomy**: `pipeline/task_categories.yaml` — 10 categories grouping
  all EgoDex tasks by *motion signature* (not by name), each with an
  `expected_verbs` prior and a `prompt_hint` fed to the Level 2 VLM step.
  This is the right unit to reason about "does the pipeline work for X" —
  individual task names within a category can still behave very differently
  (see below), but the category's kinematic description is what drives the
  Level 1/2 logic.
- **Kinematics config**: `pipeline/config.yaml`, `kinematics:` block.
- **Boundary detection logic**: `pipeline/level1_kinematics/boundaries.py`
  (`detect_events`, `detect_candidates`, `fuse`).
- **Raw dataset**: `/mnt/.../egodex/test/{task}/{idx}.hdf5` + `.mp4`, 111 task
  folders currently on disk, 3,243 episodes total (a larger ~196-task
  download was in progress as of the last check — re-run
  `pipeline/level0_audit` or `check_categories.py` after it lands, since new
  tasks must map into `task_categories.yaml` or the taxonomy audit fails
  loudly).

## What's been measured so far

First human review batch: 20 episodes, scored via `report_verdicts.py`.

| axis | accuracy | n |
|---|---|---|
| boundary | 64% | 74 |
| label | 86% | 73 |
| sentence | 66% | 74 |
| episode coherence | 16% | 19 |

Label/verb classification is solid across the board (fold/release/reach/align
100%, stack 91%, place 83%). The weak axes are boundary and sentence, and the
review-queue flag heuristic barely predicts errors (flagged 81% vs clean 88%
label accuracy — not much signal).

**Important caveat**: all 20 reviewed episodes fall into exactly one of the
10 taxonomy categories (`pick_place`). The other 9 categories
(precision_insertion, rotation_twist, assembly_multi_part, cyclic_surface,
fine_motor_tool, deformable_bimanual, container_kitchen,
open_close_articulated, dynamic_play) have zero human-reviewed episodes.
7 of those 9 already have annotated-but-unreviewed episodes on disk (~200
episodes total) and can be reviewed with no further pipeline/VLM cost via
`make_review.py --category <name> --n 10`. `dynamic_play` and
`open_close_articulated` currently have 0 annotated episodes — need a
pipeline run before they can be reviewed at all.

### Confirmed root cause: short-episode segment collapse

`min_segment_frames` (kinematics config, currently 30 frames = 1s at 30fps)
sets the minimum length of a subtask segment, enforced two ways in `fuse()`:
1. any boundary candidate within `min_segment_frames` of the episode start/end
   is dropped (so a boundary can only exist if the episode is at least
   `2 × min_segment_frames` = 2s long);
2. two adjacent boundaries closer together than `min_segment_frames` collapse
   to whichever has higher priority/magnitude.

Measured empirically across a 600-episode sample:

| duration | % that collapse to exactly 1 segment | mean segments |
|---|---|---|
| <2s | 100% | 1.00 |
| 2–5s | 71% | 1.33 |
| 5–10s | 39% | 2.16 |
| 10–20s | 37% | 3.09 |
| ≥20s | 6% | 8.97 |

This is not probabilistic below 2s — it's a hard constraint, mechanically
guaranteed regardless of content. 47% of the full 3,243-episode dataset is
under 5 seconds. Every episode the human reviewer flagged as "should have
been split into 2-3 segments" in the review batch turned out to be under
5.7s — a direct, confirmed match, not a guess.

Cross-check: in `arrange_topple_dominoes/6`, the one fully-reviewed episode
scored 100% accurate across 15 segments, the *shortest verified-correct real
segment is exactly 1.00s* (30 frames), median 1.87s — so 30 frames wasn't
picked arbitrarily; it roughly matches the empirical minimum duration of a
real, distinct pick/place action cycle for well-separated tasks. The problem
is applying one fixed floor uniformly to episodes of wildly different total
length.

### Attempted, did not validate

Two "free" (no-VLM, no-human-review) proxy metrics computed directly from
kinematic signals were tried to predict which of the other ~91 unreviewed
tasks would score well, without spending more review effort:
- raw candidate-collision rate near the merge threshold — dominated by
  benign pause/gaze dedup noise, uncorrelated with actual accuracy.
- grasp/release-vs-grasp/release collision rate specifically — went
  *backwards*: the confirmed-good category averaged a *higher* collision
  rate than the confirmed-bad one.

Conclusion: task-level accuracy isn't reliably predictable from Level-1
kinematic signals alone at this sample size. Don't reach for this shortcut
again without a much larger validated ground-truth set to check it against —
extending an unvalidated proxy to unreviewed tasks risks silently
mis-prioritizing work.

## Kinematics parameters (`pipeline/config.yaml`, `kinematics:` block)

| param | value | role |
|---|---|---|
| `max_gap_interp` | 10 frames | interpolate short invalid-tracking gaps; longer gaps get masked out instead |
| `savgol_window` / `savgol_poly` | 15 / 3 | Savitzky-Golay smoothing of speed/aperture/rotation signals. Segments shorter than this window get progressively less-reliable smoothing (the smoother clamps its window to segment length) |
| `pause_prominence` | 0.05 m/s | local-minima prominence threshold on combined hand speed → pause candidates |
| `pause_speed_max` | 0.08 m/s | a pause minimum must also be below this absolute speed to count |
| `aperture_vel_thresh` | 0.15 m/s | peak threshold on thumb-index aperture velocity → grasp/release candidates |
| `head_rot_percentile` / `head_rot_min_rad_s` | 90 / 0.25 rad/s | gaze-shift candidates: per-episode percentile threshold with an absolute floor so near-static heads don't emit spurious ones |
| `merge_window` | 15 frames (0.5s) | candidates within this window of each other are treated as duplicate detections of the same event and fused into one, keeping the higher-priority/magnitude one. This creates a de facto minimum event spacing of 0.5s *independent of* `min_segment_frames` — lowering `min_segment_frames` below this is mostly moot |
| `min_segment_frames` | 30 frames (1s) | minimum subtask segment length — see collapse analysis above. This is the main lever under discussion |
| `move_speed_min` | 0.15 m/s | a boundary sitting inside a stretch where neither side ever moves this fast gets dropped (stops near-static tasks like typing/writing from fragmenting on kinematic noise) |

Boundary source priority when merging: `grasp`/`release` (3) > `pause` (2) >
`gaze` (1) — ties broken by detection magnitude.

## Ideas to explore

- **Scale `min_segment_frames` with episode duration** instead of a fixed
  30-frame floor — e.g. something like `min(30, n_frames // 4)` — so short
  clips get a chance at 2-3 segments instead of being hard-capped at 1.
  Re-validate against the reviewed episodes afterward: confirm it recovers
  the missing splits on the short ones (stack_unstack_plates/17,
  pick_place_food/29, stack_unstack_cups/6, insert_dump_blocks/2,
  gather_roll_dice/5, stack_remove_jenga/15) without fragmenting the
  currently-100%-accurate long episode (arrange_topple_dominoes/6) into
  something worse. `merge_window` (15 frames) is a natural lower bound to
  not go below.
- **Review the other 9 taxonomy categories.** Only `pick_place` has any
  human-review signal. 7 categories already have annotated-but-unreviewed
  episodes ready to go via `make_review.py --category <name> --n 10` with
  zero further pipeline/VLM cost — this is probably the single
  highest-value next step, since right now 90% of the taxonomy is untested.
- **Recalibrate or replace the review-queue flag.** Flagged vs. clean
  segments show almost the same label accuracy (81% vs 88%) — the current
  flagging heuristic isn't a useful triage signal and shouldn't be trusted
  to prioritize review effort as-is.
- **Investigate within-category task variance.** Structurally similar tasks
  in the same category disagreed sharply in the one review batch so far
  (`stack_unstack_tupperware` scored 81% overall vs. `stack_unstack_cups` /
  `stack_unstack_plates` near 33%, despite all three being
  stack-then-unstack tasks in the `pick_place` category). With n=1 episode
  reviewed for most tasks, it's currently impossible to tell whether this is
  a genuine task-level effect or per-episode demonstrator-speed noise — a
  targeted second batch on exactly these disagreeing pairs (a few more
  episodes each) would resolve it before drawing conclusions from task name
  or category alone.
- **Semantic (non-boundary) failures need a different fix.** The
  `continuous_semantic`-type failures found so far (chess piece direction
  reversed, a card flip the VLM missed, tape-measure continuous motion with
  no discrete grasp/release structure) are Level 2 (VLM) or fundamentally
  kinematics-mismatched problems, not boundary-tuning problems — don't
  expect `min_segment_frames` or other Level 1 changes to fix these.

## Per-task kinematics overrides (`pipeline/task_configs/<task_name>.yaml`)

Added 2026-07-14, revised same day into one-file-per-task after initial
feedback that a single shared override file (keyed by task name inside one
big yaml) still meant every task's edit touched a file holding every other
task's tuning too. Current design: `pipeline/task_configs/` holds at most
one file per task, named `<task_name>.yaml`, created only for tasks that
need tuning — a task with no file just uses category/global defaults,
completely untouched. Each file is self-contained (kinematics overrides +
prompt hint + a short note on that task's motion profile) and should never
reference other tasks by name — comparative/cross-task context like "this
differs from tasks X/Y" belongs here in PIPELINE_NOTES.md instead, so a
task file stays readable and safely editable in isolation.

`common.load_task_config(task_name)` reads the file (or `{}` if absent);
`common.apply_task_overrides(cfg, task_name)` merges its `kinematics:`
block over the global one, returning `cfg` unchanged (same object) when
there's no file. `boundaries.py` (`propose_boundaries`) and
`level3_assembly/assemble.py` both call it, so Level 1 boundary detection
and the Level 3 `min_segment_frames` hard-constraint check stay consistent
— **if you add a kinematics override for a task, both call sites must see
it or Level 3 will assert-fail on segments Level 1 correctly produced**
(hit this immediately on first use, see below). `label_segments.py` reads
the file's `prompt_hint` directly and overrides just that task's category
prompt_hint for the Level 2 VLM prompt, so one mismatched task in a
category doesn't force changing the hint for every other task sharing it.

First task tuned: **`add_remove_lid`** (2,565 episodes, `open_close_articulated`
category — the category also holds `insert_remove_drawer`,
`open_close_insert_remove_box/case/tupperware`, which keep the original
category prompt_hint since they're genuinely hinge/slide, unlike this one).
Findings, full detail in `pipeline/task_configs/add_remove_lid.yaml`:
- Duration scales almost linearly with cup count, read from
  `attrs['object']` (`number:N`): mean ~2.6-3.7s per cup (1 cup: 3.79s avg /
  113 frames; 4 cups: 10.45s avg / 313 frames). This is a strong, free,
  per-episode complexity signal — worth checking for other multi-object
  tasks before assuming duration alone is the only lever.
- The category's shared kinematic prior ("hinge or rail... push/pull/slide")
  doesn't fit this task: a cup lid is a small free object lifted clear off
  and carried, not an articulated joint — much closer to a `pick_place`
  grasp/transport/release cycle, and picking/placing it is a fast motion
  (~0.3-0.5s grasp-to-release). Overrode just this task's `prompt_hint` to
  say so.
- `min_segment_frames` lowered 30 -> 20 frames for this task only: the
  1-cup episodes (401 total, mean 3.79s) sit closest to the generic
  short-episode collapse threshold (2×min_segment_frames=2s) for what's
  usually just a single fast 2-phase grasp+place cycle; 20 frames (0.67s,
  still above the 15-frame merge_window floor) gives that cycle room to
  split into reach/grasp vs. place/release instead of being forced to 1
  segment.
- Instruction direction (add vs. remove) is already handled correctly and
  generically by `resolve_description()` via `which_llm_description` —
  no task-specific change needed there. Confirmed ~50/50 add/remove split
  (1,295 / 1,270) and spot-checked in output (episode 356: task desc
  "Remove the lid..." -> labeled action `remove`, correct direction).
- Reversible-task direction is exploitable, not just a solved input problem:
  every episode is annotated `llm_type: reversible` with BOTH an
  add-phrased description (`llm_description`, always leads "Add"/"Place")
  and a remove-phrased one (`llm_description2`, always leads "Remove"),
  and `which_llm_description` ("1"/"2") says which actually happened.
  Checked all 2,565 episodes: 100% classify cleanly (1,295 add / 1,270
  remove), no ambiguous cases. `task_configs/add_remove_lid.yaml` uses this
  to split the single `prompt_hint` into two direction-specific ones
  (`prompt_hint_by_attr: which_llm_description` + `prompt_hints: {"1":...,
  "2":...}`, generic mechanism in `common.select_prompt_hint`) — add and
  remove are different motions (lid starts loose vs. starts seated) and
  benefit from different verb guidance in the same prompt slot.

- **10-episode human-review pilot, scored** (2026-07-14, seed 42, run
  through the full add-direction/remove-direction prompt fix above):
  boundaries.py -> label_segments.py -> assemble.py, review built via
  `make_review.py --category open_close_articulated --n 10`. Graded by
  visually inspecting every segment's extracted keyframes against its
  label/sentence (`outputs/review/review_verdicts_pilot10.json`,
  scored with `report_verdicts.py`):

  | axis | accuracy | n |
  |---|---|---|
  | boundary | 80% | 25 |
  | label | 100% | 25 |
  | sentence | 100% | 25 |
  | episode coherence | 100% | 10 |

  Sharp improvement over the original `pick_place` baseline batch (64% /
  86% / 66% / 16%) — label/sentence/coherence are clean here, and boundary
  is the one remaining soft spot, with a specific, consistent cause:
  **when the pipeline outputs fewer segments than there are cups (2
  segments for a 3-cup episode, etc.), the last segment silently absorbs
  2+ cups' worth of grasp/place or grasp/remove cycles into one long
  segment instead of splitting them** — seen in `102`, `419`, `456`,
  `571`, `1003` (5 of 10 episodes). The *labels* on these merged segments
  are still correct (e.g. "place lids onto the cups", plural, generically
  worded) so this doesn't corrupt content, only granularity — but it means
  segment count still undercounts true cup count more often than not for
  3+ cup episodes even with the lowered `min_segment_frames`. Episode
  `2233` (4 cups, 4 segments, one segment per cup) shows the pipeline can
  hit ideal per-cup decomposition, so the signal to split on is present —
  the boundary detector just isn't picking up a clean pause/pick
  discontinuity between every consecutive cup in ~half these episodes.
  Next lever to try if this needs to improve further: this is a
  pause-detection recall problem, not a `min_segment_frames` floor
  problem (lowering the floor further won't split a *missing* candidate)
  — look at `pause_prominence`/`pause_speed_max` sensitivity for this task
  specifically, or whether grasp/release event detection is missing the
  brief inter-cup transition.

  This is the human verification gate before running the remaining
  ~2,555 episodes. Given label/sentence/coherence are already clean and
  the boundary gap is a well-understood, scoped issue (not a correctness
  problem), the pipeline is in reasonable shape to scale up on this task;
  whether to fix inter-cup pause recall first or scale now and revisit is
  a call for whoever owns that tradeoff.

- **Duration extremes checked separately** (2026-07-14) — the 10-episode
  pilot above only spans 2.8-19.9s; tested the dataset's actual shortest
  and longest episodes directly to see if conclusions hold outside that
  range:
  - **<0.6s tail (~10 episodes, 0.4%, e.g. `1123` at 15 frames, `1281` at
    16 frames)**: these are truncated recordings, not fast demonstrations
    — inspected `1123`'s frames directly, all 4 cups are still fully
    capped in the final frame, i.e. zero action happens in the clip. The
    pipeline correctly outputs a single `idle` segment rather than
    inventing a remove action — right behavior, but no tuning fixes a
    clip that doesn't contain its annotated task. Worth excluding/flagging
    episodes under ~1s as unusable-by-construction rather than tuning
    for them.
  - **20s+ tail (~21 episodes, 0.8%, 5-6 cups)**: mixed and one real
    failure found. `2451` (24.9s, 6 cups) did well — 5 distinct `remove`
    segments, each naming a specific cup position ("fourth cup from the
    left" etc.), better resolution than most of the main pilot. `21`
    (28s, 6 cups) collapsed to 3 segments total (`reach`, one 13s
    `remove`, one 14s `place`) — and inspecting frames inside the 14s
    `place` segment shows it's **not just coarse, it's wrong**: frame 629
    (mid-segment) still shows a hand mid-lift removing a lid from a
    still-capped cup, i.e. real removal work happening inside a segment
    labeled "place". This is a step beyond the 3-4 cup merging issue
    above (which stayed label-accurate, just coarse) — at 5-6 cups the
    same under-segmentation can start misclassifying part of a merged
    segment's content. Expect the inter-cup pause-recall fix to matter
    more here, not less.

**Environment note**: the old `/mnt/3466ADC766AD8A66/...` mount referenced in
`config.yaml` no longer exists on this machine. Repointed `data_dir` to
`/data/egocentric_data/ego_dex_data/ego_dex_data_Extracted/part1/part1`
(only `part1` extracted so far — 26 of the ~196 task folders; `part2.zip`
still unextracted) and `model_path` to the Qwen3-VL-32B-Instruct snapshot
under `/data/huggingface_cache/hub/`. If task folder counts look wrong,
check whether `part2.zip` has been extracted since.

## Second human-review batch found two real bugs (2026-07-14)

A second 10-episode batch (seed 777, disjoint from the first batch + the
4 duration-extreme episodes) was reviewed by an actual human this time
(previous batch was self-graded by inspecting keyframes, which missed
these). Verdicts: boundary 17%, sentence 30%, episode coherence 0%(!) —
much worse than the self-graded batch. Root-caused to two bugs, both now
fixed and re-validated across all 24 episodes tested so far:

1. **`aperture_vel_thresh` (0.15 m/s global default) was starving grasp/
   release detection for this task.** This task's lid pinch/release is
   small and fast; several episodes never crossed 0.15 at all over the
   *entire* clip (episode 165: max 0.149 m/s, 2-cup episode -> zero
   grasp/release events detected, full stop). This single miscalibration
   explained both of the reviewer's main complaints at once: missing
   splits *between* cups (no detected event = no boundary) and missing
   reach-vs-place splits *within* one cup's cycle (the grasp event is
   that internal boundary, and it wasn't firing). Swept 0.05-0.15 against
   known cup counts across the 24-episode sample; added
   `aperture_vel_thresh: 0.08` to `task_configs/add_remove_lid.yaml`.
   Immediate effect on the flagged episodes: 165 1->2 segments, 1520
   1->4, 1836 3->4, 2213 2->3, without over-fragmenting episodes that
   were already correct (356, 1123, 1281 stayed at 1).

2. **`merge_segments()` in `level3_assembly/postprocess.py` was silently
   re-merging genuinely distinct per-cup segments** (shared code, affects
   every task, not just this one). It only compared action + normalized
   object + hand + time gap — it never checked *why* the boundary between
   two segments existed. Concrete case (episode 1126): segment 0 "Right
   hand grasps white lid... places it onto right cup", segment 1 "Both
   hands coordinate to place the white lid onto the left cup" — different
   cups, but both got action=`assemble`, object=`white lid` (normalized),
   hand=`both_coordinating`, so `mergeable()` collapsed them into one,
   silently erasing the split. The boundary between them had
   `start_source: "release"` — a real kinematic event — which is exactly
   the signal that should have blocked the merge. Fixed by refusing to
   merge across any boundary whose `start_source` is `grasp`/`release`.
   This alone recovered roughly half the affected episodes in the
   24-episode sample (1126, 1003, 2233, 1837, 1836, 1520, 1113, 1387, 165
   all kept segments that were previously merged away) — this bug was
   independently undoing a large fraction of fix #1's benefit, so both
   needed catching together.

   Since this is shared Level 3 code, it should also reduce false-merges
   on every other task, not just `add_remove_lid` — worth keeping an eye
   on whether it changes segment counts elsewhere when other tasks get
   their pilot review.

3. **Episode `2562` is a genuine data-integrity bug, not a pipeline bug**:
   its hdf5 metadata claims "add lids onto three cups... red background"
   but the actual video (verified by pulling raw frames directly via
   ffmpeg, bypassing the pipeline entirely) shows someone unwrapping a
   box of small bagged items — no cups, no lids, wrong content entirely.
   Spot-checked 15 more random episodes' first frame against their task
   metadata and all 15 were correct, so this looks like an isolated
   mislabel/misfiled clip rather than a systemic corpus problem, but it
   was found from a random draw of ~40 episodes total, so the true rate
   across the full 2,565 is unknown — worth a cheap, larger-scale sanity
   sweep (first-frame content vs. task folder, no VLM needed) before
   fully trusting the corpus at scale.

Batch was re-run through both fixes and the review page rebuilt
(`outputs/review/index.html`) for human re-review — not self-graded this
time, per explicit instruction, since the first self-graded pass is what
missed these two bugs.

## Review rounds 2-3 and the fixes they drove (2026-07-14/15)

Same 10-episode batch (seed 777) re-reviewed by the human after each fix
round. Score trajectory (boundary / label / sentence / coherence):
round 1 17/86/30/0 -> round 2 76/97/76/40 -> round 3 61/87/87/n-a.
The round-3 boundary "drop" is not a regression — Level 1 output was
byte-identical between rounds 2 and 3; the reviewer graded the same
boundaries more strictly and their notes converged on one pattern.

Fixes between rounds 2 and 3 (both verified in output):
- **False handovers**: `_is_handover` was time-only; fast alternating
  bimanual work fired it constantly (1387: 5/7 segments -> VLM wrote "cup
  transferred left hand to right hand" everywhere). Measured all 21
  qualifying pairs across 4 episodes: wrists 0.20-0.55 m apart — physically
  impossible for a real handover. Added a spatial gate
  (`hand_assignment.handover_max_wrist_dist: 0.15` in config.yaml,
  distance checked at the pair midpoint frame). All 21 false positives
  eliminated; sentence accuracy 76%->87%.
- **Cup identity in sentences**: both direction prompt_hints now require
  naming the cup by position ("second cup from the left"); 33/39 sentences
  complied immediately.

Fixes after round 3 (pending round-4 human validation):
- **Retraction bleed** — 12/17 round-3 boundary errors were the hand's
  trace-back after placing a lid staying glued to the previous assemble
  segment. Root cause: release(cup N) and grasp(lid N+1) collide inside
  merge_window/min_segment_frames and the collision resolution kept the
  grasp edge (magnitude tie-break), putting the cut at the *start* of the
  next grasp instead of at the release. Two changes:
  1. `PRIORITY` in boundaries.py now ranks `release` (4) above `grasp`
     (3) — when the two edges collide, the cut lands at the release, and
     the retraction/reach belongs to the next action cycle. Global change,
     but principled (an action ends when the hand lets go) and no other
     task has validated boundaries yet.
  2. Task overrides tightened: `min_segment_frames` 20 -> 15,
     `merge_window` 15 -> 10, so release+grasp pairs >= 0.33s apart
     survive as two separate boundaries instead of fusing.
  Verified post-fix: 946/1113/2369 segments now all end on `release`
  edges. Residual known issue: 2213 seg1 and 2369 seg1 still each span
  two cups — no aperture event fires between those specific cup pairs
  even at the lowered 0.08 threshold, so no boundary is available there;
  if round 4 still flags these, the next lever is pause detection
  (`pause_prominence`/`pause_speed_max`), not further event-threshold
  lowering.
- The 5 round-3 label/sentence failures were all on segments whose
  boundaries spanned the wrong cup(s) — cascade damage from the boundary
  problem, not VLM regressions.

## Round 4: two deeper root causes under the "retraction" complaint (2026-07-15)

Round-4 verdicts (partial grade, n=16 boundary / 14 label+sentence):
label and sentence 100%, boundary 75%, but the reviewer reported the
retraction bleed "not corrected" on 1113/1520/1836/946 — despite the
round-3 release-priority fix demonstrably moving cuts onto `release`
edges. Investigation found the *edges themselves* were wrong, for two
stacked reasons:

1. **Mid-swing hand openings masqueraded as releases.** The aperture
   detector treats any opening peak as a "release", but the hand also
   opens to pre-shape for the next grasp — mid-swing, at the END of the
   retraction. Measured on 1113/946: openings at the placement point
   happen at 0.01-0.18 m/s wrist speed, pre-grasp openings at
   0.30-0.67 m/s — cleanly separable. Added a wrist-speed gate on
   release classification (`release_speed_max`, off globally, 0.25 in
   the task config): an opening only counts as a release if the hand is
   slow. Boundary landing at a fake release = retraction glued to the
   previous segment, which is exactly what the reviewer kept seeing.

2. **Mask starvation: Level 1 was running on ~25% of frames.** These
   episodes are 69-82% masked because *fingertip* confidence collapses
   exactly while the hand manipulates the cup (occlusion) — while the
   wrist stays >92% valid. The old code built one union mask (wrist &
   thumb & index, both hands OR'd together) and suppressed every
   candidate inside it, which silenced:
   - true releases and placement pauses (they happen mid-occlusion), and
   - one hand's real events whenever the *other* hand was occluded.
   Concrete: 2213 had textbook placement pauses (speed 0.001 and ~0,
   prominence 0.55/0.69 — passing every configured gate) that never even
   became candidates; that's why "two cups in one segment" survived all
   previous fixes. Reworked masks to be per-signal in `signals.py` +
   `boundaries.py` (shared code, principled): aperture events gate on
   that hand's fingertip mask only; pauses gate on both-wrists-gone
   only; gaze (head-only signal) is not gated by hand masks at all.
   The global union mask remains for reporting.

Post-fix Level 1 on the batch: 2213 3->6 segments with pause cuts at
2.50s/4.83s (the placement moments) — two-cup merge gone; 1113 4->9,
946 5->9, each cup cycle now splitting into act + trace-back/reach the
way the reviewer wanted. Watch item for round 5: over-fragmentation —
segment counts roughly doubled; if the reviewer now flags splits as too
fine, first lever is nudging `pause_prominence` up in the task config,
not reverting the mask fix.

## Trace-back labeling + full-task generalization sweep (2026-07-15)

Two follow-ups after the round-4 fixes landed:

- **Trace-back segments now labeled as retract/reach, not repeated
  placements.** Generic change in `label_segments.py`: every prompt now
  states the *kinematic provenance* of the segment's boundaries (via
  `BOUNDARY_GLOSS` — "starts here because an object was just let go",
  "N grasp/release events inside"), and a segment that starts at a
  release with zero events inside gets an explicit instruction that the
  hands are travelling empty. Plus one task-hint sentence per direction.
  Result on the review batch: 946 reads assemble->retract->assemble->
  retract per cup with correct cup naming; 19 retract/reach segments
  across the 10 episodes where before there were ~0.

- **Level-1-only sweep over all 2,565 episodes** (no VLM cost, ~4 min)
  to answer "does the tuning generalize across the task's variation":
  0 failures, and median segment count scales linearly with cup count —
  1 cup -> 3, 2 -> 5, 3 -> 7, 4 -> 9, 5 -> 12, 6 -> 14.5, i.e. almost
  exactly `2*cups + 1` (act + retract per cup, plus the initial reach),
  which is the decomposition the human reviewer converged on as ideal.
  Outliers: 9 episodes (0.35%) suspect under-segmented (all very fast
  demos, 4 cups in ~4.5s) and 27 (1.05%) suspect over-segmented (long
  fiddly demos, e.g. 1 cup taking 6-10s); 98.6% inside the expected
  envelope. CSV of per-episode counts in the session scratchpad
  (`l1_sweep.csv`); the sweep script pattern is worth reusing for any
  task with a countable-object attr before spending VLM/review effort.
  Caveat: this validates *boundary structure* statistically, not label
  content — the 10-episode human review stays the semantic gate.

## Embodiment term per episode for VLA transfer (2026-07-15)

The subtask sentences are VLA training targets and the executing policy
may drive a robot arm, not a human hand. The end-effector noun is chosen
once per EPISODE — all of an episode's subtasks use the same term, so
the sequence reads as one consistent instruction stream — with ~40% of
episodes "arm" and ~60% "hand" (`taxonomy.ARM_EPISODE_FRACTION` +
`embodiment_for(episode_id)`, crc32-threshold based: deterministic,
rerun-reproducible, no RNG state; realized split across all 2,565
add_remove_lid ids: 40.4%). The term is recorded per segment as
`embodiment` in the labels; the prompt orders the VLM to use it even
where a style example says "hand". Verified on the 10-episode batch:
every episode internally consistent, 3 arm / 7 hand.
(First iteration rotated the term per SEGMENT — rejected by the user:
mixed hand/arm wording inside one episode; keep the choice episode-level.)
Extend beyond hand/arm (e.g. "gripper") only with a matching re-review —
"gripper" describing five visible human fingers may create a
vision-language mismatch worth checking before adopting.

## Things to consider when optimizing for a particular task or category

- Check which of the 10 `task_categories.yaml` families the task belongs to
  first, and read its `kinematics:` description — that tells you what signal
  shape to expect (discrete grasp/release cycles vs. oscillatory strokes vs.
  near-static fine-motor work vs. rotation-dominant, etc.) before assuming a
  generic pick_place-style fix applies.
- Check episode duration distribution for the task before trusting boundary
  output — short episodes (<5s, and especially <2s) cannot produce
  multi-segment output under the current fixed `min_segment_frames`
  regardless of tuning elsewhere.
- Don't extrapolate one episode's result to the whole task. The
  tupperware/cups/plates case shows same-category, similarly-named tasks can
  land on opposite ends of the accuracy range from a single sample.
- Separate boundary/segmentation problems from label/sentence (semantic)
  problems before proposing a fix — they have different root causes and live
  in different pipeline stages (Level 1 kinematics vs. Level 2 VLM).
- A proxy metric computed only from kinematic signals (no VLM, no human
  review) is tempting because it's free to run over the whole dataset, but
  the two attempts so far didn't validate against ground truth — treat any
  future proxy metric as a hypothesis to test against real review data, not
  a substitute for it.
