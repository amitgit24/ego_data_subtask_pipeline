# 05 — Kinematic task categories + category-aware prompting

**Motivating question:** does pause/grasp-based segmentation even make sense
for a task like `clean_surface`, which is arguably "just one wiping
action"? And can per-task-type prompt strategies (pick-place vs
wipe-clean vs write) do better than one generic prompt for all 111 tasks?

## Diagnosis: cyclic-task labeling failure

`clean_cups/7` (10 s of scrubbing): Level 1 correctly emitted **zero
interior boundaries** (one segment for the whole episode — the guards from
experiment 02 worked). But the VLM labeled it `align` — *"Align the blue
sponge with the red cup"* — clearly wrong; should be `wipe`. Root cause:
**periodic motion is invisible in 3 static keyframes.** The kinematic
summary said "moved 1.2m" but never said the motion was *oscillatory*, so
the model's best story from stills was "aligning."

## Design: `task_categories.yaml`

10 kinematic families covering all 111 test tasks, grouped by **motion
signature** (not name similarity — `insert_remove_drawer` is kinematically a
*slide*, not an insertion). Explicit task lists, no globbing.

| Category | Tasks | Signature |
|---|---|---|
| `pick_place` | 25 | reach–grasp–transport–release, strong aperture events |
| `precision_insertion` | 18 | fast approach → slow fine-alignment → seat/extract |
| `deformable_bimanual` | 14 | both hands reshape material in place |
| `cyclic_surface` | 12 | oscillatory strokes, no grasp events after tool pickup |
| `assembly_multi_part` | 11 | chained align-insert-secure, one hand stabilizing |
| `fine_motor_tool` | 9 | near-static wrist, activity in fingers |
| `container_kitchen` | 7 | wrist rotation while carrying = pour/tilt |
| `rotation_twist` | 5 | high angular velocity, low translation |
| `open_close_articulated` | 5 | short constrained hinge/rail strokes |
| `dynamic_play` | 5 | ballistic bursts, tracking may degrade at peak speed |

Each category carries a `prompt_hint` (one line, injected into the Level 2
prompt), `expected_verbs` (a **soft prior, never a taxonomy constraint** —
audited to be a strict subset of the closed action vocabulary),
`direction_aware` (5 categories inject the insert-vs-remove direction from
`which_llm_description`), and `keyframes_bonus` (cyclic and fine-motor
categories get +2 keyframes, since static frames under-represent oscillatory
or fine-finger motion).

`check_categories.py` audits: every on-disk task mapped exactly once, no
ghost entries, every `expected_verb` in the taxonomy. All PASS on first run
(111/111 tasks mapped).

## Design rule: priors must never become constraints

If a category hard-restricted the verb enum, the prior would become
self-fulfilling — every pick-place segment forced into a pick-place verb,
real `idle`/`adjust_grip` segments mislabeled, and evaluation metrics would
measure the prior instead of the model. The taxonomy stays fully open; the
category only adds one prompt sentence. Confirmed non-circular on the
pick_place production run (see 06): 27.7% of labels fell *outside* the
category's expected-verb list, which is the expected/healthy outcome.

## Validation

Relabeled `pick_place_food/21` and `vertical_pick_place/50` with the new
category block — the earlier `reposition` mislabel is gone; clean
`pick → pick → place` sequence. Level 1 sanity across 10 pick_place tasks:
median 17.8 segments/min (range 3.1–31.3), 6/10 in the [4,20] sane band.
