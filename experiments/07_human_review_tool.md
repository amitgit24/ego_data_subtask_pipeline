# 07 — Human verification tool

**Goal:** a way to actually check, by eye, whether (a) segment boundaries
land on real transitions, (b) each segment's action/hand/object/sentence is
correct, and (c) the full episode's subtask sequence tells a coherent story
— and to turn those judgments into hard accuracy numbers, especially for the
open question from experiment 06 (does the `kinematic_mismatch` flag
actually predict errors?).

## Design

`tools/review/make_review.py` builds one self-contained HTML page per batch:

- Selects N annotated episodes, **half review-flagged / half clean**, spread
  across distinct tasks — directly targets the flag-calibration question.
- Transcodes each episode's video to a small H.264 MP4 (browser-playable;
  the source EgoDex MP4s are not reliably playable in-browser).
- One page: episode sidebar, video player, a colored segment timeline
  synced to playback (live caption of the current subtask sentence),
  per-segment cards with a **play-with-context** button (0.5 s pre/post-roll
  around the segment, so cut-point quality is actually visible), and verdict
  buttons per segment (boundary / label / sentence, each ✓/✗) plus one
  episode-level coherence verdict.
- Verdicts persist in the browser's localStorage as you go; an Export
  button downloads a `review_verdicts.json`.

`tools/review/report_verdicts.py` scores an exported verdicts file against
the pipeline's own annotations: accuracy per axis, per-verb label accuracy,
and — the key output — **label accuracy on flagged vs. clean segments**,
which directly answers whether `kinematic_mismatch` flags are trustworthy.

## Iteration notes

- v1: horizontal table, 640px video, light theme, default CRF 26.
- v2 (user request): 1080px video at CRF 18 (visibly sharper), task
  instruction shown bold at 19px, segment table replaced with vertical
  stacked cards (title → time → flags → hand → object → sentence → verdict
  rows) — everything readable top-to-bottom without a wide-table scroll.
- v3 (user request): full dark theme.
- **Bug found and fixed:** verdict buttons for boundary/label/sentence were
  unresponsive. Cause: `onclick='${cb}'` wrapped the handler in single
  quotes, but `cb` for segment verdicts was `segv(0,'boundary',true)` —
  itself single-quoted. The browser terminated the attribute at the first
  inner quote, silently truncating the handler. The episode-level
  Coherent/Issues buttons worked because `epv(true)` has no inner quotes.
  Fixed by switching the wrapper to double quotes.

## Status

First batch generated: 20 `pick_place` episodes (10 flagged / 10 clean),
`outputs/review/index.html` + `outputs/review/media/`. **Not yet reviewed**
— this is the next concrete step: get human verdicts, run
`report_verdicts.py`, and use the flag-calibration result to decide whether
experiment 06's `aperture_vel_thresh` hypothesis holds before tuning
anything.
