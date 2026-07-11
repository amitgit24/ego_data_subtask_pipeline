# 03 — Level 2: Qwen3-VL-32B semantic labeling

**Goal:** label each Level 1 segment (action, object, hand, one-sentence
description) from 3–5 keyframes + kinematic context. The VLM never chooses
boundaries — it only classifies what's already been cut.

**Serving:** Qwen3-VL-32B-Instruct via vLLM 0.24.0, OpenAI-compatible
endpoint, JSON-schema-guided decoding (`xgrammar`), temperature 0.

## Blackwell (sm_120) serving gotchas

Both cost real debugging time and are now permanent facts in memory /
`README.md`:

1. **Tensor-parallel=2 hangs** at NCCL P2P init on this workstation board.
   Fix: one single-GPU vLLM server per card instead — 32B bf16 fits
   comfortably in 96 GB with room for KV cache.
2. **FlashInfer's capability probe crashes** on sm_120
   (`RuntimeError: FlashInfer requires GPUs with sm75 or higher`, despite
   sm_120 > sm_75 — a version-string parsing bug against CUDA < 12.9).
   Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` in the server launch env.

## Quality fixes (found by reading actual frames against actual labels)

### 1. Task-verb leakage
First pass: every segment of `insert_remove_usb/47` was labeled `remove`,
including the retract phase, because the prompt only warned "don't blindly
copy the task verb" — insufficient. **Fix:** segments are now labeled
**sequentially within an episode**, and each prompt includes the previous
segments' labels ("Previously labeled segments of this episode: ..."), with
an explicit rule that a completed action doesn't repeat. Parallelism moved
from segment-level to episode-level (throughput unaffected). Result:
`remove → press → reposition` instead of `remove → remove → remove`.

### 2. Hand-assignment scoring bug
An idle hand's tracking noise, divided by its near-zero episode-median
activity, could outscore the hand actually doing the work (observed: left
7.9 vs right 3.3 on a visibly right-handed extraction). **Fix:** a hand must
show raw physical motion above an absolute floor (`active_speed_min` /
`active_apv_min`) to count as "active" at all; the median-normalized score
only ranks hands that already cleared that floor.

### 3. Taxonomy gaps
`write` and `wrap`/`unwrap` were missing from the closed action vocabulary —
the model was contorting `hold_steady` and `object_state` verbs to cope with
writing/drawing and food-wrapping tasks. Added both.

## Sentence style — retuned for VLA training (see also 05, 06)

Original 6 style templates included leading subordinate clauses ("To adjust
the orientation of...", "Reaching across the table..."). Since these
sentences are the training target for a VLA's language encoder (not just a
QA description), they were rewritten to be short (8–14 words), action-first,
present tense, with no leading subordinate clause — diversity now comes from
word order only. The prompt also now says: take the *purpose* of a motion
from the episode's overall task, never invent one (rotating a part in a
*removal* task means loosening it, not "adjusting its orientation").

## Throughput (measured)

~65–70 labeled segments/min on one RTX PRO 6000 Blackwell, episode-level
concurrency 8. Full 3,243-episode test split extrapolates to ~75–90 min on
one GPU.

## Verifier result (5-episode dev sample, after all fixes)

```
[PASS] parse failure rate < 2% after retry     (0 failures — guided decoding)
[PASS] 'other' action < 20%                    (0%)
[PASS] no single action > 50%
[PASS] hand_disagreement <= 10%                (0%)
[PASS] low confidence <= 30%                   (0% — all high)
[PASS] phrasing diversity: top bigram < 30%    (23.5%, all 6 styles used)
```
