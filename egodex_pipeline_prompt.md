# Claude Code Prompt: EgoDex Subtask Annotation Pipeline

Copy everything below this line into Claude Code.

---

## Mission

Build a 3-level pipeline that takes EgoDex episodes (MP4 video + HDF5 pose file) and produces a JSON annotation per episode, where each episode is segmented into subtasks. Each subtask has: start_frame, end_frame, start_time, end_time, duration, action (closed taxonomy), hand (left/right/both/both_coordinating — computed from kinematics), object, and subtask (a one-sentence description with deliberately varied phrasing across the dataset).

Architecture (do not deviate without flagging it to me):
- **Level 0 — Data audit**: verify the HDF5/MP4 structure before writing any pipeline code.
- **Level 1 — Kinematic boundary proposal**: candidate subtask boundaries computed purely from hand pose signals (no VLM, no vision model).
- **Level 2 — VLM semantic labeling**: Qwen3-VL-32B labels each candidate segment using 3-5 keyframes + kinematic context. The VLM never chooses boundaries; it only classifies and describes.
- **Level 3 — Validation and assembly**: merge, constraint-check, cross-check kinematics vs semantics, compute all timing in code, emit schema-validated JSON.

Each level must end with a VERIFIER script that prints a pass/fail report before we move to the next level. Do not build Level N+1 until the Level N verifier passes on the sample data I provide. I will provide the EgoDex validation/test split for development.

Work incrementally. After each level, stop and show me the verifier output.

## Project layout

Create this structure:

```
egodex_annotator/
├── config.yaml               # all thresholds and paths in one place
├── requirements.txt
├── level0_audit/
│   └── audit_dataset.py      # data structure inspection + report
├── level1_kinematics/
│   ├── signals.py            # velocity, aperture, head-motion extraction
│   ├── boundaries.py         # candidate boundary detection
│   └── verify_level1.py      # visual + statistical verifier
├── level2_vlm/
│   ├── keyframes.py          # keyframe selection per segment
│   ├── taxonomy.py           # closed verb vocabulary
│   ├── label_segments.py     # Qwen3-VL-32B inference
│   └── verify_level2.py
├── level3_assembly/
│   ├── postprocess.py        # merge / smooth / constraint checks
│   ├── schema.py             # pydantic output schema
│   ├── assemble.py           # final JSON writer
│   └── verify_level3.py
├── run_pipeline.py           # end-to-end driver
└── outputs/
    ├── audits/
    ├── boundaries/
    ├── labels/
    └── annotations/
```

Use Python 3.10+, h5py, numpy, scipy, opencv-python or decord for video, pydantic for schemas, matplotlib for verifier plots. For Qwen3-VL-32B use the transformers/vLLM interface (I will tell you where the model is served; write the client so it can hit either a local vLLM OpenAI-compatible endpoint or load via transformers — make the backend configurable in config.yaml).

---

## LEVEL 0 — Data audit (do this FIRST, before any pipeline code)

The EgoDex paper says the dataset provides 30 Hz 1080p video with paired 3D pose annotations for head, upper body, wrists, and 25 joints per hand, stored as HDF5 alongside MP4, plus camera extrinsics and a natural-language task annotation. Do NOT trust this description blindly — verify against the actual files.

Write `audit_dataset.py` that takes a directory of episodes and, for a sample of at least 5 episodes across different tasks, reports:

### A. File pairing and naming
- Every MP4 has a matching HDF5 (and note the naming convention linking them).
- How task names / language annotations are stored (folder name? HDF5 attribute? separate metadata file?). Print exactly where the language annotation lives.

### B. HDF5 structure dump
- Recursively walk the HDF5 tree; print every group/dataset path, shape, and dtype.
- Explicitly answer these questions in the audit report:
  1. Where are the wrist poses for left and right hand? Are they 4x4 transform matrices (SE(3)), position+quaternion, or something else? What is the exact key path?
  2. Where are the individual finger joint poses? How many joints per hand are actually present? List the joint names.
  3. Is there a thumb tip and index tip joint we can use for a pinch-aperture signal? Print their exact key names.
  4. Where is the head/camera pose? Is it camera extrinsics per frame?
  5. Is there a confidence/validity field per joint per frame? (Vision Pro tracking can drop out — we need to know how missing data is encoded: NaN, zeros, confidence array, or absent frames.)
  6. What coordinate frame is everything in (world frame from SLAM, or camera-relative)? Check whether wrist positions stay smooth when the head moves — if positions jump with head motion, they are camera-frame and we must transform to world frame using the camera extrinsics before computing velocities.

### C. Synchronization
- Number of pose frames vs number of video frames per episode. Report any off-by-N mismatch and the exact frame-to-pose alignment rule.
- Confirm effective FPS (should be 30; verify from video metadata AND from any timestamps in HDF5).

### D. Signal sanity
- For 3 episodes, extract right-wrist position over time and plot xyz vs frame index. Save to outputs/audits/.
- Report: units (meters?), typical magnitude of per-frame displacement, fraction of frames with missing/invalid pose, longest gap of invalid tracking.

### LEVEL 0 VERIFIER — pass criteria
Print a table with PASS/FAIL for each:
- [ ] MP4/HDF5 pairing resolved for 100% of sampled episodes
- [ ] Wrist pose key paths identified for both hands
- [ ] Thumb-tip and index-tip (or equivalent pinch pair) key paths identified
- [ ] Camera/head pose key path identified
- [ ] Missing-data encoding understood and documented
- [ ] Coordinate frame determined (world vs camera) with evidence
- [ ] Frame alignment rule between video and pose established
- [ ] <5% invalid pose frames on sampled episodes (if higher, report and we discuss)

If ANY item fails, STOP and show me the audit report. The rest of the pipeline depends on these facts. Write the confirmed key paths and conventions into config.yaml so no other module hardcodes assumptions.

---

## LEVEL 1 — Kinematic boundary proposal

Only start after Level 0 passes. All parameters below go in config.yaml with the defaults given; we will tune them on real data.

### signals.py
Per episode, compute per-frame signals (all in world frame; interpolate over short invalid-tracking gaps up to `max_gap_interp: 10` frames, mask longer gaps):

1. `speed_left`, `speed_right`: wrist translational speed in m/s. Compute finite differences of wrist position, multiply by FPS, then smooth with a Savitzky-Golay filter (`savgol_window: 15` frames, `savgol_poly: 3`).
2. `speed_combined`: max(speed_left, speed_right) — a boundary requires BOTH hands to be slow only if both were recently active; simpler and effective default is the max.
3. `aperture_left`, `aperture_right`: Euclidean distance between thumb tip and index tip per hand, smoothed the same way.
4. `aperture_velocity`: first derivative of aperture — sharp negative = closing (grasp), sharp positive = opening (release).
5. `head_rot_speed`: angular speed of the head/camera orientation (geodesic distance between consecutive rotation matrices × FPS), smoothed.

### boundaries.py
Generate candidate boundaries as the union of three detectors:

1. **Pause detector**: local minima of `speed_combined` found with `scipy.signal.find_peaks` on the negated signal, with `prominence: 0.05` (m/s) and minimum value below `pause_speed_max: 0.08` m/s. Humans decelerate between subtasks; these minima are the primary boundary source.
2. **Grasp/release detector**: peaks of |aperture_velocity| above `aperture_vel_thresh: 0.15` m/s, classified as grasp (closing) or release (opening) by sign. Tag each event with hand and type.
3. **Gaze-shift detector** (secondary, lower weight): peaks of `head_rot_speed` above `head_rot_thresh` (default: 90th percentile within the episode).

Fusion:
- Merge candidates closer than `merge_window: 15` frames (0.5 s) into one, keeping the highest-priority source (grasp/release > pause > gaze).
- Enforce minimum segment length `min_segment_frames: 30` (1 s): drop the weaker of any pair of boundaries violating it.
- Always include frame 0 and the last valid frame as boundaries.
- Output per episode: `outputs/boundaries/{episode_id}.json` with a list of boundaries: `{frame, time, source, hand (nullable), event_type (nullable)}` and the derived segments.

### LEVEL 1 VERIFIER — verify_level1.py
For each of 5 sample episodes:
1. **Plot**: one figure with speed_combined, apertures, and head_rot_speed stacked, with vertical lines at detected boundaries colored by source. Save as PNG. This is the primary human check — I will look at these.
2. **Filmstrip**: for each detected boundary, extract the video frame at boundary-15 frames, at the boundary, and at boundary+15 frames; tile them into a strip image per episode. A correct boundary should show a visible change in hand/object configuration across the strip.
3. **Statistics table** printed to console:
   - segments per minute (sane range for tabletop manipulation: roughly 4–20; far outside = thresholds wrong)
   - segment duration distribution (min/median/max)
   - boundary source breakdown (% pause vs grasp/release vs gaze)
   - % of episode covered by valid segments
4. **Degenerate-case checks**: no segment shorter than min_segment_frames; no boundary in a masked invalid-tracking region; first boundary = 0 and last = final frame.

Pass criteria: plots visually sensible on all 5 episodes (I will confirm), segments/minute in sane range on at least 4 of 5, all degenerate checks pass. STOP and show me the plots and filmstrips before Level 2.

---

## LEVEL 2 — VLM semantic labeling (Qwen3-VL-32B)

The VLM only labels segments produced by Level 1. It never proposes or adjusts frame boundaries.

### taxonomy.py
Define a closed ACTION vocabulary as an enum, grouped for coverage across EgoDex's 194 tabletop tasks:

- **Approach/retract**: reach, move, retract, hover
- **Acquire/release**: grasp, pick, lift, place, put_down, release, drop
- **Transport/exchange**: transfer (object moved point A to B), handover (object passed between hands), carry, reposition
- **Object-state change**: open, close, fold, unfold, insert, remove, rotate, flip, pour, press, push, pull, slide, stack, unstack, twist, tie, untie, tear, cut, wipe, shake, squeeze
- **Bimanual-specific**: hold_steady (one hand stabilizes while other acts), align (bringing two objects together), assemble, disassemble
- **Neutral**: hold, idle, adjust_grip
- **Escape**: other (description must state what was seen)

Also define a HAND enum: `left`, `right`, `both`, `both_coordinating`.

### hand_assignment.py (new module — computed from kinematics, NOT by the VLM)
Per segment, determine the hand field from Level 1 signals:
- Compute per-hand activity score = mean wrist speed + aperture-change magnitude within the segment, each normalized by its episode-level median.
- One hand's score > `hand_dominance_ratio: 3.0` × the other → `left` or `right`.
- Both above activity threshold → `both`; upgrade to `both_coordinating` if EITHER: (a) inter-wrist distance stays within ±15% of its segment mean for >70% of the segment (jointly carrying/stabilizing one object), or (b) a grasp event on one hand occurs within 1 s of a release event on the other (handover pattern).
- Store the computed hand plus the activity scores in the segment record. Pass it to the VLM as context; the VLM may flag disagreement but never overrides it.

### keyframes.py
Per segment, select up to `keyframes_per_segment: 5` frames: first, last, temporal midpoint, and the frames of any grasp/release events inside the segment (these are the most informative instants). Decode only these frames from the MP4 (seek-based, do not decode the full video). Downscale to `keyframe_size: 768` px on the long side.

### label_segments.py
For each segment, build one Qwen3-VL-32B request containing:
- The keyframe images in temporal order, each preceded by a text tag: "Frame at t=X.Xs (segment start / middle / grasp event / end)".
- The episode's task-level language annotation from the dataset (from the location found in Level 0): "Overall task: {task_annotation}".
- A kinematic summary string built from Level 1 + hand_assignment outputs, e.g.: "Active hand: left. Left hand: grasp event at 2.1s, moved 0.42m, release at 4.8s. Right hand: mostly static."
- The instruction: classify the segment. Respond ONLY with JSON matching:
```json
{"action": "<one of the taxonomy actions>",
 "object": "<manipulated object, short noun phrase with attributes, e.g. 'white cup'>",
 "subtask": "<one sentence describing the subtask, following the style instruction>",
 "confidence": "high|medium|low",
 "hand_disagreement": false}
```

### Subtask description variation (important — the dataset must not read monotone)
Define at least 6 style templates in taxonomy.py. Select the template deterministically per segment: `style_id = (hash(episode_id) + segment_index) % n_styles`, so reruns are reproducible but phrasing varies across the dataset. Inject the chosen style into the prompt as a one-line instruction. Example style families:

1. **Imperative-plain**: "Pick up the white cup with the left hand."
2. **Agent-action**: "The person moves their left hand toward the white cup."
3. **Progressive**: "Reaching across the table, the left hand grasps the white cup."
4. **Manner-adverb**: "Carefully lifts the white cup using the left hand."
5. **Goal-oriented**: "To free the saucer, the white cup is picked up with the left hand."
6. **Compact-telegraphic**: "Left hand grasps white cup and lifts it off the table."

Rules for all styles: mention the hand when it is left/right/both_coordinating; include the object with a distinguishing attribute when visible (color, size, position); one sentence; present tense; factual content identical across styles — only phrasing varies. Add a config flag `style_variation: true|false` so we can disable it for debugging.

Inference rules:
- temperature 0, JSON-constrained decoding if the serving backend supports it; otherwise parse with a tolerant JSON extractor and retry once on parse failure.
- Batch requests where the backend allows; add retry with exponential backoff; checkpoint after every episode so reruns skip completed segments.
- Log the full prompt and raw response for the first 10 segments to outputs/labels/debug/ so I can inspect prompt quality.

Output: `outputs/labels/{episode_id}.json` — Level 1 segments enriched with verb/object/description/confidence.

### LEVEL 2 VERIFIER — verify_level2.py
1. **Contact sheet**: for 5 episodes, generate an HTML page per episode showing each segment's keyframes with the predicted verb/object/description underneath. This is the primary human check.
2. **Statistics**:
   - action distribution (a healthy result is diverse; >50% one action or >20% `other` = prompt or taxonomy problem)
   - hand distribution (left/right/both/both_coordinating) and hand_disagreement rate (>10% VLM disagreement with kinematic hand assignment = investigate whose fault)
   - confidence distribution (>30% low = flag)
   - JSON parse failure rate (must be <2% after retry)
   - object-name consistency: within one episode, cluster object strings; wildly inconsistent naming for the same object across segments = flag
   - **phrasing diversity check**: compute the fraction of subtask sentences sharing the same first two words across the set; must be <30%. Also report distribution of style_ids actually used. Monotone output = style injection not working
3. **Kinematic-semantic agreement spot check**: segments whose boundary source was a grasp event should mostly get verbs from {pick, reach, hold, ...} at the start; release-bounded segments should correlate with {place, insert, pour, ...}. Print the agreement rate. This is a soft signal, not a hard gate — report it.

Pass criteria: parse failure <2%, `other` verb <20%, contact sheets look right to me on majority of segments. STOP and show me the contact sheets.

---

## LEVEL 3 — Validation and assembly

### postprocess.py
1. **Merge**: adjacent segments with identical verb AND same object (string-normalized) AND gap < `merge_gap_s: 0.5` s → merge into one segment (recompute frames from the outer bounds; keep the higher confidence).
2. **Idle labeling**: any uncovered gap > 1 s between segments becomes an explicit segment with verb `idle`.
3. **Hard constraints** (assert, fail loudly): segments sorted, non-overlapping, within [0, last_frame], every segment ≥ min length after merging, full-episode coverage including idle segments.
4. **Cross-check flags** (do not delete data, only flag): VLM said `pick` but no grasp event inside or near segment → flag `kinematic_mismatch`. VLM confidence low → flag `low_confidence`. Flagged segments go into a `review_queue` list in the output.

### schema.py + assemble.py
Pydantic schema; all timing computed in code from frame indices and the FPS confirmed in Level 0 — never taken from model output:

```json
{
  "episode_id": "str",
  "task": "str (dataset-provided language annotation)",
  "fps": 30.0,
  "total_frames": 0,
  "pipeline_version": "str",
  "subtasks": [
    {
      "id": 0,
      "action": "str (taxonomy)",
      "hand": "left|right|both|both_coordinating",
      "object": "str",
      "subtask": "str (one-sentence description, style-varied)",
      "start_frame": 0,
      "end_frame": 0,
      "start_time": 0.0,
      "end_time": 0.0,
      "duration": 0.0,
      "boundary_source_start": "pause|grasp|release|gaze|episode_start",
      "boundary_source_end": "pause|grasp|release|gaze|episode_end",
      "style_id": 0,
      "confidence": "high|medium|low",
      "flags": ["kinematic_mismatch", "low_confidence", "hand_disagreement"]
    }
  ],
  "review_queue": [0]
}
```

Merge rule update: adjacent segments merge only if action AND object AND hand all match.

### LEVEL 3 VERIFIER — verify_level3.py
1. Schema validation on every output file (pydantic strict mode).
2. Invariant re-check across the whole processed set: no overlaps, full coverage, duration arithmetic exact (`duration == (end_frame - start_frame) / fps` to 1e-6).
3. **Annotated video render** for 2 episodes: burn the current subtask description and segment progress bar onto the video frames and write an MP4 to outputs/annotations/preview/. This is the final human check — watching 2 of these tells us more than any metric.
4. Summary report: episodes processed, total subtasks, mean subtasks/episode, % flagged for review, verb histogram across the set.

Pass criteria: 100% schema-valid, 100% invariants hold, preview videos look correct to me.

---

## run_pipeline.py
CLI: `python run_pipeline.py --data_dir <path> --episodes <all|list|N_random> --levels 0,1,2,3`. Levels are resumable; each level reads the previous level's outputs. `--verify-only` reruns just the verifiers.

## OPTIONAL LEVEL 4 — Fine-tuning data export (build only when I say so)

If zero-shot Qwen3-VL-32B labeling quality is insufficient (Level 2 verifier shows high `other` rate, high hand_disagreement, or poor contact-sheet accuracy), we will fine-tune. I can provide labeled data. Prepare for this:

### export_finetune.py (in level2_vlm/)
- Convert human-verified annotations (segments corrected via the review queue) into Qwen-VL chat fine-tuning format: one sample per segment = {images: keyframe paths, messages: [user: same prompt as inference (keyframes + task annotation + kinematic summary + style instruction), assistant: gold JSON]}.
- Emit in the ShareGPT/Qwen conversation JSONL format compatible with LLaMA-Factory or ms-swift for Qwen3-VL LoRA fine-tuning.
- Include a stratified split script: hold out 10% by episode (never split one episode across train/val) and stratify by task category and action label.
- Report label counts per action/hand so we can see class imbalance before training.
- Rule: fine-tune only on human-verified segments, never on the model's own unreviewed outputs (avoids self-distilling errors).

## Ground rules
1. Never let the VLM decide frame numbers, durations, or timing arithmetic.
2. Never hardcode HDF5 key paths outside config.yaml; Level 0 is the single source of truth for the data schema.
3. Every threshold in config.yaml with the defaults specified above.
4. Fail loudly: no silent exception swallowing; a bad episode is logged to a skip-list with the reason and the pipeline continues.
5. After finishing each level, STOP, run its verifier on the sample episodes, and show me the outputs before continuing.

Start with Level 0 now. Ask me for the path to the EgoDex validation data directory.
