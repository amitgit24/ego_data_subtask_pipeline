# 04 — Level 3 assembly + first end-to-end pipeline test

**Goal:** merge duplicate adjacent segments, label uncovered gaps as idle,
enforce hard coverage/ordering constraints, cross-check semantics against
kinematics (flag, never delete), compute all timing in code, emit a
pydantic-validated schema with a review queue.

**Then:** prove the whole pipeline (Levels 1→2→3) runs unattended on fresh
data via a single driver command — no manual steps between levels.

## Level 3 components

- `schema.py` — pydantic model that **re-derives every timestamp from frame
  indices** at validation time (`duration == (end-start)/fps` to `1e-6`). A
  file that validates is arithmetically exact by construction; timing is
  never trusted from anywhere else.
- `postprocess.py` — merges adjacent segments with identical
  action+object(normalized)+hand and a gap under `merge_gap_s`; inserts
  explicit `idle` segments for uncovered gaps over `idle_gap_s`; flags
  (never deletes) segments whose verb implies a grasp/release event that
  Level 1 didn't detect nearby, plus low-confidence and hand-disagreement
  segments, into a `review_queue`.

## End-to-end test — 20 random fresh episodes, one command

```
python run_pipeline.py --episodes 20_random --levels 1,2,3 --seed 42
```

No manual intervention anywhere in the run. Result:

```
20 episodes labeled (47 segments) in 45s (62.4 seg/min), 0 skipped, 0 failed
20 assembled, 0 skipped, 0 failed

[PASS] schema-valid: 25/25 (must be 100%)          # 20 new + 5 dev episodes
[PASS] invariants + duration arithmetic exact to 1e-06
[PASS] preview videos rendered for human review
flagged for review: 2/55 subtasks (3.6%)
```

Merge rule confirmed working: 6 raw `type_keyboard` segments collapsed to 1
`press`; 2 raw `write` segments collapsed to 1 `write`.

`kinematic_mismatch` fired correctly on `boil_serve_egg/2` segment 0
("places the pot" with no release event detected nearby) — the flag
mechanism catches real weak spots without deleting the segment.

## Spot-check quality (by eye, against the actual frames)

- `make_sandwich/6`: coherent 6-step story (assemble breads → stack layers →
  final assembly).
- `tie_and_untie_shoelace/17`: one clean `untie` subtask spanning the whole
  episode.
- `boil_serve_egg/2`: plausible 7-step sequence; its weakest subtask was
  correctly self-flagged.
- Known soft spot at this point: vague object names on uncommon items
  ("pink circular object" for what was probably a tomato slice).

This run established `run_pipeline.py` as resumable (`--episodes all` skips
anything already labeled) and confirmed the full stack works without a human
in the loop — the precondition for the category-scale production run in
experiment 06.
