"""Non-overlapping chunked subtask generation -- a PARALLEL generator to
generate_subtasks.py, not a replacement or a modification of it (that file
is untouched; everything reusable is imported from it read-only).

Where the sliding-window generator re-shows a fraction of each window to the
next one so a boundary-crossing subtask stays visually coherent, this
generator shows each source frame to the model EXACTLY ONCE: fixed-size,
back-to-back chunks (48 frames, then the next 48 frames, and so on -- no
re-showing). Since the model never sees a frame twice, cross-chunk coherence
has to come entirely from the TEXT story-so-far instead of from re-observed
frames -- so the prompt explicitly tells the model whether the previous
chunk ended mid-subtask (and what that subtask's true start_frame was) and
asks it to either CONTINUE that exact subtask or open the NEXT one at this
chunk's first frame. That "continuation vs. next subtask" call is the one
thing this generator's prompt optimizes that the sliding-window prompt does
not need to (it just re-shows the frames and lets the model re-derive it).

Shares the taxonomy prompt block, JSON parsing, candidate validation, keyframe
subsampling, and record shape with generate_subtasks.py via direct import.

CLI:
    python generate_subtasks_chunked.py --episodes task/idx [...]
    python generate_subtasks_chunked.py --sample 5
Config defaults to config_chunked.yaml (sibling of config.yaml, separate
output_dir) unless --config is given.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import h5py
import torch
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ACTIONS, HANDS, STYLE_RULES, apply_task_overrides,  # noqa: E402
                    embodiment_for, episode_id, episode_slug,
                    extract_keyframes, load_config, load_task_config,
                    resolve_description, sample_episodes, select_prompt_hint)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_backend import probe_video  # noqa: E402
from generate_subtasks import (DebugBudget, select_window_frames,  # noqa: E402
                               taxonomy_block, to_record_segments,
                               tolerant_json, valid_subtask)
from chunked_vlm_backend import EndpointBackend  # noqa: E402

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PIPELINE_ROOT / "config_chunked.yaml"


# ------------------------------------------------------------------ windows

def plan_chunks(total_frames, chunk_frames, frame_skip):
    """Fixed-size, back-to-back, NON-overlapping spans -- chunk i+1 starts
    exactly where chunk i ended, so every source frame in range is shown to
    the model exactly once (contrast generate_subtasks.plan_windows(), whose
    windows deliberately re-show a shared tail/head)."""
    stride = frame_skip + 1
    span = chunk_frames * stride
    windows = []
    start = 0
    while start <= total_frames - 1:
        end = min(start + span - 1, total_frames - 1)
        windows.append((start, end))
        start = end + 1
    return windows


# ------------------------------------------------------------------ prompt

def _continuation_block(open_tail, chunk_start):
    if open_tail is None:
        return (f"Nothing was left open -- the previous chunk ended cleanly. "
                f"The first subtask you report should START at frame "
                f"{chunk_start} (this chunk's first frame).")
    return (
        f"The previous chunk ended MID-SUBTASK (marked ongoing): it was "
        f"action \"{open_tail['action']}\" on \"{open_tail['object']}\", "
        f"true start_frame {open_tail['start_frame']} -- \"{open_tail['subtask']}\". "
        f"If that SAME subtask is still happening at the start of this "
        f"chunk, report it again FIRST with start_frame {open_tail['start_frame']} "
        f"(its original start, NOT this chunk's first frame) and extend "
        f"end_frame to wherever it actually finishes now. If it already "
        f"finished between the two chunks, start the NEXT subtask at frame "
        f"{chunk_start} instead -- do not report the finished one again.")


def _shared_instructions(chunk_start, chunk_end, task, task_hint, story_text,
                         open_tail, embodiment):
    hand_line = ", ".join(HANDS)
    story_block = story_text or "(this is the first chunk -- no subtasks yet)"
    hint_block = f"\nTask-specific guidance: {task_hint.strip()}\n" if task_hint else ""
    continuation_block = _continuation_block(open_tail, chunk_start)
    return f"""Overall task for the WHOLE video: "{task}"
{hint_block}
Finalized subtasks from earlier chunks (DO NOT re-report or renumber these):
{story_block}

{continuation_block}
""", story_block, hint_block, continuation_block, hand_line


def build_prompt_chunked(task, frame_entries, total_frames, story_text,
                         open_tail, chunk_start, chunk_end, task_hint=None,
                         embodiment="hand"):
    frame_lines = "\n".join(f"  Frame {fr}: t={t:.2f}s" for fr, t in frame_entries)
    header, story_block, hint_block, continuation_block, hand_line = \
        _shared_instructions(chunk_start, chunk_end, task, task_hint,
                             story_text, open_tail, embodiment)

    return f"""You are annotating an egocentric human manipulation video by
splitting it into consecutive SUBTASKS and labeling each one. The video is
being shown to you in BACK-TO-BACK, NON-OVERLAPPING chunks -- you have never
seen any frame before this chunk and will never see it again after this
call, so this report is the ONLY chance to describe what happens in frames
{chunk_start}-{chunk_end}.

{header}
You are shown {len(frame_entries)} frames from THIS CHUNK ONLY: source
frames {chunk_start} to {chunk_end} (of {total_frames} total), sampled in
chronological order:
{frame_lines}

Report every subtask that STARTS or is ONGOING within this chunk, in order.
For each:
- start_frame / end_frame: ABSOLUTE frame numbers. Follow the continuation
  rule above for the first subtask; every subsequent subtask in this chunk
  starts fresh. A subtask ENDS at the moment the object is released/placed/
  let go -- the hand then travelling back empty toward the next object is
  the START of the NEXT subtask (reach/retract), never the tail of the one
  just finished. Set "ongoing": true on the LAST subtask you report if it is
  still happening at frame {chunk_end} (unfinished when this chunk's frames
  run out) -- this is what lets the NEXT chunk continue it correctly, since
  it will never see these frames again.
- action: EXACTLY ONE of these words (never a group/category name such as
  "object_state"):
  {", ".join(ACTIONS)}
  (grouped below only to help you choose -- answer with a single word):
{taxonomy_block()}
- hand: one of: {hand_line}. Pick the SINGLE hand ("left"/"right") if only
  one hand is actively manipulating the object, even if the other hand is
  visible in frame resting, idle, or merely nearby without gripping
  anything. Use "both" only when both hands are each independently
  manipulating (e.g. one steadies the cup while the other places the lid).
  Use "both_coordinating" only when both hands are working AS ONE unit on
  the same grip/motion (e.g. passing an object hand-to-hand, or both hands
  gripping one object together). A hand merely being in frame is not
  evidence it is acting.
- object: short noun phrase with one visible attribute.
- subtask: {STYLE_RULES} Call the end-effector "{embodiment}" (e.g. "left
  {embodiment}", "both {embodiment}s") in every sentence, even if a style
  example shows a different word.
- confidence: high | medium | low

Do not invent subtasks; if the hands are merely idle, say so with action "idle".
Return ONLY a JSON object, no prose, no code fences:
{{"subtasks": [{{"start_frame": <int>, "end_frame": <int>, "action": "<word>", "hand": "<one of {hand_line}>", "object": "<str>", "subtask": "<one sentence>", "confidence": "high|medium|low", "ongoing": <bool>}}]}}"""


def build_prompt_content_chunked(task, frame_data, total_frames, story_text,
                                 open_tail, chunk_start, chunk_end, task_hint,
                                 embodiment):
    """OpenAI-format content list (text + interleaved images) -- vllm_endpoint
    path, same instructions as build_prompt_chunked() addressed to explicit
    keyframe images instead of a native video blob."""
    header, story_block, hint_block, continuation_block, hand_line = \
        _shared_instructions(chunk_start, chunk_end, task, task_hint,
                             story_text, open_tail, embodiment)

    content = [{"type": "text", "text": f"""You are annotating an egocentric human manipulation video by
splitting it into consecutive SUBTASKS and labeling each one. The video is
being shown to you in BACK-TO-BACK, NON-OVERLAPPING chunks -- you have never
seen any frame before this chunk and will never see it again after this
call, so this report is the ONLY chance to describe what happens in frames
{chunk_start}-{chunk_end}.

{header}
You are shown {len(frame_data)} keyframes sampled from THIS CHUNK ONLY:
source frames {chunk_start} to {chunk_end} (of {total_frames} total), in
chronological order, each labeled with its absolute source frame number and
timestamp."""}]
    for frame, tag, path in frame_data:
        content.append({"type": "text", "text": f"Frame {frame}: t={tag}"})
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content.append({"type": "text", "text": f"""
Report every subtask that STARTS or is ONGOING within this chunk, in order.
For each:
- start_frame / end_frame: ABSOLUTE frame numbers from the frames shown
  above. Follow the continuation rule above for the first subtask; every
  subsequent subtask in this chunk starts fresh. A subtask ENDS at the
  moment the object is released/placed/let go -- the hand then travelling
  back empty toward the next object is the START of the NEXT subtask
  (reach/retract), never the tail of the one just finished. Set "ongoing":
  true on the LAST subtask you report if it is still happening at frame
  {chunk_end} (unfinished when this chunk's frames run out) -- this is what
  lets the NEXT chunk continue it correctly, since it will never see these
  frames again.
- action: EXACTLY ONE of these words (never a group/category name such as
  "object_state"):
  {", ".join(ACTIONS)}
  (grouped below only to help you choose -- answer with a single word):
{taxonomy_block()}
- hand: one of: {hand_line}. Pick the SINGLE hand ("left"/"right") if only
  one hand is actively manipulating the object, even if the other hand is
  visible in frame resting, idle, or merely nearby without gripping
  anything. Use "both" only when both hands are each independently
  manipulating (e.g. one steadies the cup while the other places the lid).
  Use "both_coordinating" only when both hands are working AS ONE unit on
  the same grip/motion (e.g. passing an object hand-to-hand, or both hands
  gripping one object together). A hand merely being in frame is not
  evidence it is acting.
- object: short noun phrase with one visible attribute.
- subtask: {STYLE_RULES} Call the end-effector "{embodiment}" (e.g. "left
  {embodiment}", "both {embodiment}s") in every sentence, even if a style
  example shows a different word.
- confidence: high | medium | low

Do not invent subtasks; if the hands are merely idle, say so with action "idle".
Respond ONLY with JSON matching the schema."""})
    return content


@torch.inference_mode()
def generate_chunk_transformers(model, processor, video_path, w_start, w_end,
                                chunk_frames, fps, total_frames, task,
                                story_text, open_tail, v, task_hint=None,
                                embodiment="hand"):
    nframes = min(chunk_frames, w_end - w_start)
    nframes = max(2, nframes - (nframes % 2))  # qwen_vl_utils wants even

    video_content = {
        "type": "video",
        "video": f"file://{Path(video_path).resolve()}",
        "video_start": w_start / fps,
        "video_end": w_end / fps,
        "nframes": nframes,
    }
    _, videos_probe, _ = process_vision_info(
        [{"role": "user", "content": [video_content]}],
        return_video_kwargs=True, return_video_metadata=True)
    _, meta = videos_probe[0]
    frame_entries = [(int(i), float(i) / fps) for i in meta["frames_indices"]]

    prompt = build_prompt_chunked(task, frame_entries, total_frames,
                                  story_text, open_tail, w_start, w_end,
                                  task_hint, embodiment)
    messages = [{"role": "user",
                "content": [video_content, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True)
    video_kwargs = {k: (val[0] if isinstance(val, list) and len(val) == 1 else val)
                    for k, val in video_kwargs.items()}
    videos, video_metadata = zip(*videos)
    inputs = processor(text=[text], images=images, videos=list(videos),
                       video_metadata=list(video_metadata), return_tensors="pt",
                       **video_kwargs).to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=v["max_new_tokens"], do_sample=True,
        temperature=v["temperature"], repetition_penalty=v["repetition_penalty"])
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed = tolerant_json(out_text)
    return parsed.get("subtasks") or [], out_text


def generate_chunk_vllm(backend, video_path, w_start, w_end, chunk_frames,
                        frame_skip, fps, total_frames, task, story_text,
                        open_tail, task_hint, embodiment, keyframes_per_window,
                        keyframe_size, kf_out_dir, window_index):
    stride = frame_skip + 1
    frames = select_window_frames(w_start, w_end, stride, chunk_frames,
                                  keyframes_per_window)
    frame_tags = [(f, f"{f / fps:.2f}s") for f in frames]
    kf = extract_keyframes(video_path, frame_tags, keyframe_size,
                           kf_out_dir, window_index)
    content = build_prompt_content_chunked(task, kf, total_frames, story_text,
                                           open_tail, w_start, w_end,
                                           task_hint, embodiment)
    out_text = backend.generate(content)
    parsed = tolerant_json(out_text)
    return parsed.get("subtasks") or [], out_text


def valid_chunk_subtask(s, w_start, w_end):
    """generate_subtasks.valid_subtask(), plus keeping the "ongoing" flag it
    normally drops -- the chunked generator needs it to decide whether a
    subtask stays open across the next chunk boundary."""
    vs = valid_subtask(s, w_start, w_end)
    if vs is None:
        return None
    vs["ongoing"] = bool(s.get("ongoing", False))
    return vs


# ------------------------------------------------------------------ stitching

def story_text(segments, n_context):
    if not segments:
        return ""
    lines = [f"  frames {s['start_frame']}-{s['end_frame']}: "
            f"{s['action']} — {s['subtask']}" for s in segments[-n_context:]]
    return "\n".join(lines)


def _resolve_intra_chunk_overlaps(seq):
    """The model's own single JSON response can occasionally list two
    candidates that overlap EACH OTHER -- seen in practice on episode 165's
    chunk [48,95]: the model reported both a "retract" ending at 95 and a
    "reach" spanning 86-95 in the same response, 9 overlapping frames.
    Cross-chunk overlap can't happen by construction (chunks are disjoint
    ranges), so this is purely an intra-response ordering slip. Clamp each
    candidate's start to the previous one's end, in list order (already
    chronological), dropping anything that becomes empty or was fully
    contained -- same overlap resolution generate_subtasks.stitch() applies
    across windows, just a single sequential pass since there is no cross-
    window IOU clustering to do here."""
    out = []
    for c in seq:
        if out and c["start_frame"] < out[-1]["end_frame"]:
            if c["end_frame"] <= out[-1]["end_frame"]:
                continue  # fully contained in the previous one -- drop
            c = dict(c, start_frame=out[-1]["end_frame"])
        if c["end_frame"] > c["start_frame"]:
            out.append(c)
    return out


def process_chunk_candidates(kept, open_tail, segments, w_start):
    """Fold one chunk's validated, ordered candidates into the running
    (segments, open_tail) state, IN PLACE on `segments`. No IOU/overlap
    stitching needed here (chunks never overlap) -- the continuation signal
    is frame arithmetic, not label text: a FRESH subtask can never start
    before w_start (that's this chunk's own first frame), so if the model's
    first reported start_frame is < w_start, it can only be referencing the
    open subtask from the story (per the continuation instruction) --
    whatever action/object wording it used this time wins (fresher read),
    but the true start_frame is taken from open_tail, not re-trusted from
    the model's echo. Matching on label text instead was tried and broke on
    a real example: the model correctly reused start_frame 26 across a
    chunk boundary but reworded the object ("...near the left cup" ->
    "...onto the leftmost cup"), which a strict text match rejected as
    "not the same subtask", producing a duplicate overlapping segment.
    Returns the new open_tail (or None)."""
    cur_open = open_tail
    seq = list(kept)
    if seq and cur_open is not None:
        first = seq[0]
        if first["start_frame"] < w_start:
            merged = dict(first)
            merged["start_frame"] = cur_open["start_frame"]
            seq[0] = merged
        else:
            segments.append(cur_open)  # ended where last recorded; no new info
        cur_open = None
    seq = _resolve_intra_chunk_overlaps(seq)
    for idx, c in enumerate(seq):
        is_last = idx == len(seq) - 1
        if is_last and c.get("ongoing"):
            cur_open = c
        else:
            segments.append(c)
    return cur_open


# ------------------------------------------------------------------ episode

def process_episode(backend, h5_path, cfg, debug_budget=None):
    """backend: ("transformers", model, processor) or ("vllm_endpoint", EndpointBackend)."""
    task_name = h5_path.parent.name
    cfg = apply_task_overrides(cfg, task_name)
    v = cfg["generate"]
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        task = resolve_description(attrs, cfg)
    task_cfg = load_task_config(task_name)
    task_hint = select_prompt_hint(task_cfg, attrs)
    embodiment = embodiment_for(episode_id(h5_path))
    video_path = h5_path.with_suffix(".mp4")
    duration, fps = probe_video(video_path, cfg["video"]["fps_fallback"])
    total_frames = int(round(duration * fps))
    slug = episode_slug(h5_path)
    out_root = Path(cfg["paths"]["output_dir"])
    kf_dir = out_root / "labels" / "keyframes" / slug

    chunks = plan_chunks(total_frames, v["model_frames"], v["frame_skip"])
    t0 = time.time()
    segments, open_tail = [], None
    debug_windows = []
    for wi, (w_start, w_end) in enumerate(chunks):
        story = story_text(segments, v["story_context_segments"])
        raw = None
        for attempt in range(1, v["max_retries"] + 1):
            try:
                if backend[0] == "transformers":
                    _, model, processor = backend
                    subs, raw = generate_chunk_transformers(
                        model, processor, video_path, w_start, w_end,
                        v["model_frames"], fps, total_frames, task, story,
                        open_tail, v, task_hint, embodiment)
                else:
                    _, endpoint = backend
                    subs, raw = generate_chunk_vllm(
                        endpoint, video_path, w_start, w_end,
                        v["model_frames"], v["frame_skip"], fps, total_frames,
                        task, story, open_tail, task_hint, embodiment,
                        cfg["vlm"]["keyframes_per_window"],
                        cfg["vlm"]["keyframe_size"], kf_dir, wi)
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if attempt < v["max_retries"]:
                    time.sleep(v["retry_backoff_sec"] * attempt)
                else:
                    subs, raw = [], f"[chunk failed after retries: {e!r}]"
        kept = [vs for vs in (valid_chunk_subtask(s, w_start, w_end) for s in subs)
                if vs is not None]
        open_tail = process_chunk_candidates(kept, open_tail, segments, w_start)
        if debug_budget is not None and debug_budget.pop():
            debug_windows.append({"chunk": wi, "span": [w_start, w_end],
                                  "story": story, "raw_response": raw,
                                  "kept": kept, "open_tail_after": open_tail})

    if open_tail is not None:
        segments.append(open_tail)
    if not segments:
        segments = [{"start_frame": 0, "end_frame": total_frames - 1,
                    "action": "other", "hand": "both", "object": "unknown",
                    "subtask": "Unable to reliably segment this episode.",
                    "confidence": "low"}]
    record_segments = to_record_segments(segments)

    record = {"episode_id": episode_id(h5_path), "task": task, "fps": fps,
              "n_frames": total_frames, "embodiment": embodiment,
              "generate_params": {k: v[k] for k in ("model_frames", "frame_skip")},
              "num_windows": len(chunks),
              "n_candidates": len(segments),
              "n_stitched": len(segments),
              "wall_sec": round(time.time() - t0, 1),
              "segments": record_segments}

    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / f"{slug}.json").write_text(json.dumps(record, indent=2))
    boundaries = [{"frame": 0, "time": 0.0, "source": "episode_start"}] + [
        {"frame": s["start_frame"], "time": round(s["start_frame"] / fps, 4),
         "source": "vlm_subtask"} for s in record_segments[1:]] + [
        {"frame": total_frames - 1, "time": round((total_frames - 1) / fps, 4),
         "source": "episode_end"}]
    (out_root / "boundaries").mkdir(parents=True, exist_ok=True)
    (out_root / "boundaries" / f"{slug}.json").write_text(json.dumps(
        {"episode_id": record["episode_id"], "n_frames": total_frames, "fps": fps,
         "num_windows": len(chunks), "boundaries": boundaries,
         "segments": [{"id": i, "start_frame": s["start_frame"],
                       "end_frame": s["end_frame"],
                       "start_time": round(s["start_frame"] / fps, 4),
                       "end_time": round(s["end_frame"] / fps, 4),
                       "duration": round((s["end_frame"] - s["start_frame"]) / fps, 4)}
                      for i, s in enumerate(record_segments)]}, indent=2))
    if debug_windows:
        dbg = out_root / "labels" / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / f"{slug}.json").write_text(json.dumps(debug_windows, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config or str(DEFAULT_CONFIG))
    data_dir = Path(cfg["paths"]["data_dir"])
    out_root = Path(cfg["paths"]["output_dir"]) / "labels"

    if args.episodes:
        paths = [data_dir / f"{e}.hdf5" for e in args.episodes]
    elif args.sample:
        paths = sample_episodes(data_dir, args.sample)
    else:
        ap.error("give --episodes or --sample")

    if cfg["vlm"]["backend"] == "vllm_endpoint":
        from chunked_vlm_backend import check_vlm_endpoint
        check_vlm_endpoint(cfg)
        backend = ("vllm_endpoint", EndpointBackend(cfg))
    else:
        from model_backend import load_model
        model, processor = load_model(cfg["paths"]["model_path"],
                                      cfg["model"]["attn_implementation"],
                                      cfg["model"]["dtype"])
        backend = ("transformers", model, processor)

    debug_budget = DebugBudget(cfg["generate"]["debug_first_n"])
    for p in paths:
        out_file = out_root / f"{episode_slug(p)}.json"
        if out_file.exists() and not args.force:
            print(f"skip (exists): {episode_id(p)}")
            continue
        rec = process_episode(backend, p, cfg, debug_budget)
        print(f"{rec['episode_id']}: {rec['num_windows']} chunks -> "
              f"{len(rec['segments'])} segments")


if __name__ == "__main__":
    main()
