# 02 — Level 1: kinematic boundary proposal

**Goal:** candidate subtask boundaries from hand-pose signals alone
(pauses, grasp/release events, gaze shifts) — no vision model.

**Method:** `signals.py` computes per-frame speed/aperture/head-rotation
signals (Savitzky-Golay smoothed, short gaps interpolated); `boundaries.py`
fuses pause / grasp-release / gaze-shift candidates, enforces a minimum
segment length, and always keeps frame 0 and the last frame as boundaries.

## Iteration 1 — spec defaults, first verifier run

Ran on 5 sample episodes across distinct tasks.

```
episode                                        segs  seg/min
write/8                                           2     12.0
insert_remove_usb/47                              3     12.5
type_keyboard/7                                   6     36.0   <- too high
lock_unlock_key/5                                 4     24.6   <- too high
assemble_disassemble_furniture_bench_lamp/14      4     22.4
[FAIL] segments/minute in [4, 20] on >=4/5 episodes (2 sane)
```

**Root cause:** near-static tasks (typing, key-turning) never drop below
`pause_speed_max`, so every small ripple in wrist speed registered as a
pause. The fusion logic had no way to tell "hands genuinely stopped moving"
from "hands were basically still the whole time."

## Fix — `move_speed_min` rule

Added a fusion-time check: a boundary only survives if the hands showed an
actual movement burst (peak speed ≥ `move_speed_min: 0.15` m/s) on at least
one side of it. Also added `head_rot_min_rad_s: 0.25` as an absolute floor
for the gaze detector — a pure within-episode percentile fired spuriously on
near-still heads.

## Iteration 2 — result

```
episode                                        segs  seg/min
write/8                                           2     12.0
insert_remove_usb/47                              3     12.5
type_keyboard/7                                   6     36.0
lock_unlock_key/5                                 3     18.4
assemble_disassemble_furniture_bench_lamp/14      3     16.8
[PASS] segments/minute in [4, 20] on >=4/5 episodes (4 sane)
[PASS] degenerate checks (min length, masks, edges)
[PASS] plots + filmstrips written for human review
```

Visual check on `insert_remove_usb/47`: two pause boundaries at clean speed
minima splitting reach+grasp / insert / remove+return; filmstrip rows show a
genuine change in hand/object configuration at each cut.

## Known limitation (see experiment 06)

Grasp/release detection (`aperture_vel_thresh: 0.15` m/s) is tuned against
dataset-wide aperture-velocity peaks (median 0.076, p90 0.19 m/s). On
`pick_place`-family episodes with small objects the real grasp events peak
well below this threshold — Level 1 sanity check on 10 pick_place tasks
found a median of **1 grasp/release event per episode**, vs ~8 dataset-wide.
This under-detection later shows up as `kinematic_mismatch` flags on
correctly-labeled `pick`/`place` segments in Level 3 — an open calibration
question, not yet resolved.
