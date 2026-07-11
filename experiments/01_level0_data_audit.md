# 01 — Level 0: dataset audit

**Goal:** verify the real on-disk structure of EgoDex before writing any
pipeline code against assumptions from the paper.

**Method:** `pipeline/level0_audit/audit_dataset.py` samples 8 episodes
across distinct tasks and inspects file pairing, HDF5 key paths, video/pose
sync, and per-frame signal sanity.

## Findings

- **Layout:** `test/{task_name}/{episode_idx}.{mp4,hdf5}` — 111 task folders,
  3,243 episode pairs, 0 unpaired files.
- **Poses:** `transforms/{joint}` as `(N,4,4)` float32 SE(3) matrices.
  Wrists at `transforms/leftHand` / `rightHand`; pinch pair at
  `{side}ThumbTip` + `{side}IndexFingerTip`; camera at `transforms/camera`.
  24 finger joints + wrist per hand.
- **Coordinate frame is WORLD**, not camera-relative — confirmed two ways:
  the camera transform itself moves up to 0.3 m per episode (camera-relative
  data would show an identity camera), and the camera sits ~25 cm above both
  wrists on every sampled episode (geometrically impossible in camera
  coordinates). Velocities can be computed directly, no transform needed.
- **Sync is exact:** video frame count == pose frame count on all 8 sampled
  episodes (delta 0), confirmed 30.000 fps, 1920×1080. Alignment rule:
  `pose[i] ↔ video frame i`.
- **Two surprises, both handled in `common.py`:**
  1. `confidences/{joint}` (per-frame validity, 0–1) is **missing on some
     episodes** (older annotator version). Fallback: NaN-check on the
     transform itself (those episodes showed 0 NaNs).
  2. Language annotation location **varies by annotator version**: newest
     episodes use `llm_description` (+ `llm_description2` /
     `which_llm_description` for reversible tasks), older ones use
     `description`. Resolution order implemented in
     `common.resolve_description()`.
- **Signal quality:** 0% invalid frames on all 8 sampled episodes, positions
  in meters, mm-scale per-frame displacement.

## Verifier result

```
[PASS] MP4/HDF5 pairing resolved (full split)
[PASS] Wrist pose key paths identified (both hands)
[PASS] Thumb-tip / index-tip pinch pair identified
[PASS] Camera/head pose key path identified
[PASS] Missing-data encoding understood (confidences | NaN fallback)
[PASS] Coordinate frame determined: world (cam travel + geometry)
[PASS] Frame alignment rule established (|delta| <= 1)
[PASS] <5% invalid pose frames (worst 0.0%)
```

All confirmed facts are written into `pipeline/config.yaml` under `hdf5:` —
no other module hardcodes a key path.
