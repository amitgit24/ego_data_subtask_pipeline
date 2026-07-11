# 06 — Production run: 200 `pick_place` episodes

**Goal:** first category-scale run with the optimized (category-aware)
prompt, to see real distribution statistics before committing to the full
3,243-episode split.

**Command:**
```
python run_pipeline.py --category pick_place --n 200 --levels 1,2,3 --seed 7
```
Episode selection round-robins across all 25 `pick_place` tasks so no single
task dominates the sample.

## Result

```
200 episodes labeled, 0 skipped, 0 failed
200 assembled, 0 skipped, 0 failed
```

Combined with pre-existing dev episodes, the full annotation set stood at
**225 episodes / 677 subtasks**, all schema-valid, all duration arithmetic
exact. Pick_place-category subset specifically: **202 episodes, 624
subtasks (3.1/episode)**.

### Verb distribution (healthy — no collapse to one dominant verb)
`reposition` 117, `stack` 92, `pick` 88, `place` 47, `align` 39,
`unstack` 34, `insert` 34, `remove` 32, `idle` 26, `reach` 22, `open` 17,
`handover` 10.

### Hands
right 340 / left 122 / both 76 / both_coordinating 86 — plausible
right-hand dominance for a mostly-unimanual category.

### Prior calibration check
27.7% of labels (173/624) fell **outside** the category's `expected_verbs`
prior (align, insert, remove, open, handover) — confirms the prior is a
genuine hint, not a self-fulfilling constraint (see experiment 05's design
rule). No prior-echo detected.

### Phrasing diversity
Top opening bigram "right hand" at 19.4% of sentences — under the 30% gate.

## Open finding: review-queue rate jumped to ~17% (108/624)

Almost all flags are `kinematic_mismatch` on `pick` (62) and `place` (37).
**Hypothesis, not yet confirmed by human review:** this is a Level 1
detector calibration gap, not a Level 2 labeling error. Pick-place grasps on
small objects (dice, cards, beads, coins) are gentle pinches with aperture
velocity below `aperture_vel_thresh: 0.15` m/s — the Level 1 sanity check in
experiment 02 already showed this category's episodes average only ~1
detected grasp/release event, vs ~8 dataset-wide. If true, most of these 108
flags are the cross-check correctly noticing a missing *event*, while the
*label* itself (pick/place) is right.

**Deliberately not fixed yet** — tuning the detector threshold without
ground truth risks quietly making the alarm quieter rather than more
correct. This is the first concrete question for the human review tool
(experiment 07): compare label accuracy on flagged vs. clean segments. If
flagged `pick`/`place` segments turn out to be labeled correctly at the same
rate as clean ones, the fix is a lower `aperture_vel_thresh` for
small-object categories (or accepting sustained closed-aperture as grasp
evidence in the cross-check) — not a Level 2 prompt change.
