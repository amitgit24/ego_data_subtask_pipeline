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
