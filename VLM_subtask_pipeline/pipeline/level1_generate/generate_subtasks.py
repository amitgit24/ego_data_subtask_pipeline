"""Single-pass subtask generation over sliding, optionally-overlapping windows.

This is the whole modeling stage of the pipeline: for each window the model
emits BOTH the segment boundaries AND the labels in one JSON (frame + subtask
"in one go"), and the per-window outputs are stitched into one coherent
per-episode segment list that L3 assembly then finalizes with the SAME schema
as the sibling pipelines.

Window geometry (all config, meant to be swept):
  - model_frames frames are fed to the model per window.
  - frame_skip sets the sampling stride (skip+1), so ONE window reaches over
    model_frames*(skip+1) source frames (48 / 96 / 144 for skip 0 / 1 / 2).
  - overlap_frac of that span is shared with the previous window; the overlap
    is re-shown to the model so a subtask crossing a window boundary stays
    coherent. The exact absolute frame indices the model sees are read back
    from qwen_vl_utils and listed in the prompt (same verified mechanism as
    VLM_seg_subtask_pipeline), so reported frames are in source coordinates.

Coherence across windows: each window's prompt echoes the "story so far" (the
last finalized subtasks plus the currently-open one) so the model continues the
narrative and reuses consistent object/action wording instead of restarting.

Stitching: candidates from all windows are clustered (same-action overlaps
merge, keeping the higher-confidence label) and then resolved to a sorted,
non-overlapping segment list; gaps and min-length are handled in L3.

CLI:
    python generate_subtasks.py --episodes task/idx [...]
    python generate_subtasks.py --sample 5
"""

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path

import h5py
import torch
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ACTION_GROUPS, ACTIONS, CONFIDENCES, HANDS,  # noqa: E402
                    STYLE_RULES, apply_task_overrides, embodiment_for,
                    episode_id, episode_slug, extract_keyframes, load_config,
                    load_task_config, resolve_description, sample_episodes,
                    select_prompt_hint)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_backend import probe_video  # noqa: E402
from vlm_backend import EndpointBackend  # noqa: E402

CONF_RANK = {"high": 2, "medium": 1, "low": 0}

GLOSSES = {
    "transfer": "object moved from point A to B",
    "handover": "object passed between the two hands",
    "hold_steady": "one hand stabilizes while the other acts",
    "align": "bringing two objects/parts together",
    "adjust_grip": "regrasp or shift fingers without moving the object",
    "other": "none of the above fits; describe what you saw",
}


def tolerant_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def taxonomy_block():
    lines = []
    for group, actions in ACTION_GROUPS.items():
        items = [f"{a} ({GLOSSES[a]})" if a in GLOSSES else a for a in actions]
        lines.append(f"- {group}: {', '.join(items)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ windows

def plan_windows(total_frames, model_frames, frame_skip, overlap_frac):
    """List of (w_start, w_end) source-frame spans covering the whole video.

    One window reaches over model_frames*(skip+1) source frames; consecutive
    windows advance by span*(1-overlap_frac). The final window is anchored to
    end on the last frame so nothing at the tail is missed."""
    stride = frame_skip + 1
    span = model_frames * stride
    advance = max(1, round(span * (1.0 - overlap_frac)))
    windows = []
    start = 0
    while True:
        end = min(start + span - 1, total_frames - 1)
        windows.append((start, end))
        if end >= total_frames - 1:
            break
        start += advance
    return windows


def build_prompt(task, frame_entries, total_frames, story, task_hint=None,
                 embodiment="hand"):
    frame_lines = "\n".join(
        f"  Frame {fr}: t={t:.2f}s" for fr, t in frame_entries)
    first, last = frame_entries[0][0], frame_entries[-1][0]
    hand_line = ", ".join(HANDS)
    story_block = story or "(this is the first window — no subtasks yet)"
    hint_block = f"\nTask-specific guidance: {task_hint.strip()}\n" if task_hint else ""

    return f"""You are annotating an egocentric human manipulation video by
splitting it into consecutive SUBTASKS and labeling each one, all in a single
pass. A subtask is one atomic phase of manipulation (approach, grasp, remove,
place, retract, hold, idle, ...), bounded by the frames where the hand-object
interaction state changes.

Overall task for the WHOLE video: "{task}"
{hint_block}
Story so far (subtasks already decided in earlier windows — DO NOT re-number or
repeat these; continue after them, and keep object/action wording consistent):
{story_block}

You are shown ONE WINDOW: source frames {first} to {last} (of {total_frames}
total), sampled in chronological order:
{frame_lines}

Report every subtask that STARTS or is ONGOING within this window. For each:
- start_frame / end_frame: ABSOLUTE frame numbers from the list above. If a
  subtask continues from the story-so-far, start it at that earlier frame; if
  it is still ongoing at the end of this window, set end_frame to the last
  frame shown and "ongoing": true. A subtask ENDS at the moment the object is
  released/placed/let go — the hand then travelling back empty toward the
  next object is the START of the NEXT subtask (reach/retract), never the
  tail of the one just finished.
- action: EXACTLY ONE of these words (never a group/category name such as
  "object_state"):
  {", ".join(ACTIONS)}
  (grouped below only to help you choose — answer with a single word):
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


@torch.inference_mode()
def generate_window_transformers(model, processor, video_path, w_start, w_end,
                                 model_frames, fps, total_frames, task, story,
                                 v, task_hint=None, embodiment="hand"):
    # qwen_vl_utils derives the available frame count from the TIME range
    # (video_start/video_end * fps) with its own rounding, which can come out
    # one LESS than w_end - w_start + 1 on a tail window — decord then rejects
    # nframes > available ("nframes should in interval"), and this env has no
    # torchvision fallback. Requesting span-1 (even, >=2) stays safely inside
    # the range at the cost of at most one tail frame.
    nframes = min(model_frames, w_end - w_start)
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

    prompt = build_prompt(task, frame_entries, total_frames, story,
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


def select_window_frames(w_start, w_end, stride, model_frames, keyframes_per_window):
    """Candidate grid (same span/stride the transformers backend would see),
    evenly subsampled down to keyframes_per_window actual images -- always
    keeping the first and last candidate so the window's boundary frames are
    never dropped."""
    candidates = list(range(w_start, w_end + 1, stride))[:model_frames]
    if len(candidates) <= keyframes_per_window:
        return candidates
    step = (len(candidates) - 1) / (keyframes_per_window - 1)
    idxs = sorted({round(i * step) for i in range(keyframes_per_window)})
    return [candidates[i] for i in idxs]


def build_prompt_content(task, frame_data, total_frames, story, task_hint,
                         embodiment):
    """OpenAI-format content list (text + interleaved images) for the
    vllm_endpoint backend -- same instructions as build_prompt(), addressed
    to explicit keyframe images instead of a native video blob."""
    hand_line = ", ".join(HANDS)
    story_block = story or "(this is the first window — no subtasks yet)"
    hint_block = f"\nTask-specific guidance: {task_hint.strip()}\n" if task_hint else ""
    first, last = frame_data[0][0], frame_data[-1][0]

    content = [{"type": "text", "text": f"""You are annotating an egocentric human manipulation video by
splitting it into consecutive SUBTASKS and labeling each one, all in a single
pass. A subtask is one atomic phase of manipulation (approach, grasp, remove,
place, retract, hold, idle, ...), bounded by the frames where the hand-object
interaction state changes.

Overall task for the WHOLE video: "{task}"
{hint_block}
Story so far (subtasks already decided in earlier windows — DO NOT re-number or
repeat these; continue after them, and keep object/action wording consistent):
{story_block}

You are shown {len(frame_data)} keyframes sampled from ONE WINDOW: source
frames {first} to {last} (of {total_frames} total), in chronological order,
each labeled with its absolute source frame number and timestamp."""}]
    for frame, tag, path in frame_data:
        content.append({"type": "text", "text": f"Frame {frame}: t={tag}"})
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content.append({"type": "text", "text": f"""
Report every subtask that STARTS or is ONGOING within this window. For each:
- start_frame / end_frame: ABSOLUTE frame numbers from the frames shown above.
  If a subtask continues from the story-so-far, start it at that earlier
  frame; if it is still ongoing at the end of this window, set end_frame to
  the last frame shown and "ongoing": true. A subtask ENDS at the moment the
  object is released/placed/let go — the hand then travelling back empty
  toward the next object is the START of the NEXT subtask (reach/retract),
  never the tail of the one just finished.
- action: EXACTLY ONE of these words (never a group/category name such as
  "object_state"):
  {", ".join(ACTIONS)}
  (grouped below only to help you choose — answer with a single word):
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


def generate_window_vllm(backend, video_path, w_start, w_end, model_frames,
                         frame_skip, fps, total_frames, task, story,
                         task_hint, embodiment, keyframes_per_window,
                         keyframe_size, kf_out_dir, window_index):
    stride = frame_skip + 1
    frames = select_window_frames(w_start, w_end, stride, model_frames,
                                  keyframes_per_window)
    frame_tags = [(f, f"{f / fps:.2f}s") for f in frames]
    kf = extract_keyframes(video_path, frame_tags, keyframe_size,
                           kf_out_dir, window_index)
    content = build_prompt_content(task, kf, total_frames, story, task_hint,
                                   embodiment)
    out_text = backend.generate(content)
    parsed = tolerant_json(out_text)
    return parsed.get("subtasks") or [], out_text


def valid_subtask(s, w_start, w_end):
    try:
        sf, ef = int(s["start_frame"]), int(s["end_frame"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (s.get("action") in ACTIONS and s.get("hand") in HANDS
            and s.get("confidence") in CONFIDENCES):
        return None
    if not (isinstance(s.get("object"), str) and s["object"].strip()
            and isinstance(s.get("subtask"), str) and s["subtask"].strip()):
        return None
    # clamp to a sensible range; a subtask may legitimately start before this
    # window (continuation) but must not run past it
    ef = min(ef, w_end)
    if ef <= sf:
        return None
    return {"start_frame": sf, "end_frame": ef, "action": s["action"],
            "hand": s["hand"], "object": s["object"].strip(),
            "subtask": s["subtask"].strip(), "confidence": s["confidence"]}


# ------------------------------------------------------------------ stitching

def _norm_obj(s):
    s = s.lower().strip()
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            return s[len(art):]
    return s


def _same_label(a, b):
    return a["action"] == b["action"] and _norm_obj(a["object"]) == _norm_obj(b["object"])


def _iou(a, b):
    lo = max(a["start_frame"], b["start_frame"])
    hi = min(a["end_frame"], b["end_frame"])
    inter = max(0, hi - lo)
    union = (max(a["end_frame"], b["end_frame"])
             - min(a["start_frame"], b["start_frame"]))
    return inter / union if union > 0 else 0.0


def stitch(candidates, iou_merge):
    """Overlapping-window candidates -> sorted, non-overlapping labeled
    segments (gaps allowed; L3 fills them)."""
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (c["start_frame"], c["end_frame"]))

    # phase A: cluster same-label overlapping candidates (union span, keep the
    # higher-confidence label)
    clusters = []
    for c in cands:
        hit = None
        for m in clusters:
            if _same_label(m, c) and _iou(m, c) >= iou_merge:
                hit = m
                break
        if hit is None:
            clusters.append(dict(c))
        else:
            if CONF_RANK[c["confidence"]] > CONF_RANK[hit["confidence"]]:
                hit.update({k: c[k] for k in
                            ("action", "hand", "object", "subtask", "confidence")})
            hit["start_frame"] = min(hit["start_frame"], c["start_frame"])
            hit["end_frame"] = max(hit["end_frame"], c["end_frame"])

    # phase B: resolve residual overlaps between DIFFERENT labels into a sorted,
    # non-overlapping sequence
    clusters.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
    resolved = []
    for c in clusters:
        c = dict(c)
        if resolved and c["start_frame"] < resolved[-1]["end_frame"]:
            p = resolved[-1]
            if _same_label(p, c):
                p["end_frame"] = max(p["end_frame"], c["end_frame"])
                if CONF_RANK[c["confidence"]] > CONF_RANK[p["confidence"]]:
                    p.update({k: c[k] for k in
                              ("action", "hand", "object", "subtask", "confidence")})
                continue
            if c["end_frame"] <= p["end_frame"]:
                continue  # fully-contained different-label blip -> drop
            mid = (c["start_frame"] + p["end_frame"]) // 2
            mid = max(p["start_frame"] + 1, min(mid, c["end_frame"] - 1))
            p["end_frame"] = mid
            c["start_frame"] = mid
        if c["end_frame"] > c["start_frame"]:
            resolved.append(c)
    return resolved


def to_record_segments(segments):
    """Stitched labeled intervals -> L3-postprocess segment dicts."""
    out = []
    for s in segments:
        out.append({
            "start_frame": s["start_frame"], "end_frame": s["end_frame"],
            "start_source": "vlm_subtask", "end_source": "vlm_subtask",
            "transition_before": None, "transition_after": None, "style_id": -1,
            "vlm": {"action": s["action"], "hand": s["hand"],
                    "object": s["object"], "subtask": s["subtask"],
                    "confidence": s["confidence"]},
        })
    return out


def story_text(segments, n_context):
    if not segments:
        return ""
    lines = []
    for s in segments[-n_context:]:
        lines.append(f"  frames {s['start_frame']}-{s['end_frame']}: "
                     f"{s['action']} — {s['subtask']}")
    return "\n".join(lines)


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

    windows = plan_windows(total_frames, v["model_frames"], v["frame_skip"],
                           v["overlap_frac"])
    t0 = time.time()
    all_candidates = []
    debug_windows = []
    for wi, (w_start, w_end) in enumerate(windows):
        story = story_text(stitch(all_candidates, cfg["stitch"]["iou_merge"]),
                           v["story_context_segments"])
        raw = None
        for attempt in range(1, v["max_retries"] + 1):
            try:
                if backend[0] == "transformers":
                    _, model, processor = backend
                    subs, raw = generate_window_transformers(
                        model, processor, video_path, w_start, w_end,
                        v["model_frames"], fps, total_frames, task, story, v,
                        task_hint, embodiment)
                else:
                    _, endpoint = backend
                    subs, raw = generate_window_vllm(
                        endpoint, video_path, w_start, w_end,
                        v["model_frames"], v["frame_skip"], fps, total_frames,
                        task, story, task_hint, embodiment,
                        cfg["vlm"]["keyframes_per_window"],
                        cfg["vlm"]["keyframe_size"], kf_dir, wi)
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if attempt < v["max_retries"]:
                    time.sleep(v["retry_backoff_sec"] * attempt)
                else:
                    subs, raw = [], f"[window failed after retries: {e!r}]"
        kept = [vs for vs in (valid_subtask(s, w_start, w_end) for s in subs)
                if vs is not None]
        all_candidates.extend(kept)
        if debug_budget is not None and debug_budget.pop():
            debug_windows.append({"window": wi, "span": [w_start, w_end],
                                  "story": story, "raw_response": raw,
                                  "kept": kept})

    stitched = stitch(all_candidates, cfg["stitch"]["iou_merge"])
    if not stitched:  # model found nothing usable -> one flagged 'other' segment
        stitched = [{"start_frame": 0, "end_frame": total_frames - 1,
                     "action": "other", "hand": "both", "object": "unknown",
                     "subtask": "Unable to reliably segment this episode.",
                     "confidence": "low"}]
    segments = to_record_segments(stitched)

    record = {"episode_id": episode_id(h5_path), "task": task, "fps": fps,
              "n_frames": total_frames, "embodiment": embodiment,
              "generate_params": {k: v[k] for k in
                                  ("model_frames", "frame_skip", "overlap_frac")},
              "num_windows": len(windows),
              # sweep metrics: how much the overlap actually merged, and cost
              "n_candidates": len(all_candidates),
              "n_stitched": len(stitched),
              "wall_sec": round(time.time() - t0, 1),
              "segments": segments}

    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / f"{slug}.json").write_text(json.dumps(record, indent=2))
    # a boundaries.json mirror for the verifier / cross-pipeline comparison
    boundaries = [{"frame": 0, "time": 0.0, "source": "episode_start"}] + [
        {"frame": s["start_frame"], "time": round(s["start_frame"] / fps, 4),
         "source": "vlm_subtask"} for s in segments[1:]] + [
        {"frame": total_frames - 1, "time": round((total_frames - 1) / fps, 4),
         "source": "episode_end"}]
    (out_root / "boundaries").mkdir(parents=True, exist_ok=True)
    (out_root / "boundaries" / f"{slug}.json").write_text(json.dumps(
        {"episode_id": record["episode_id"], "n_frames": total_frames, "fps": fps,
         "num_windows": len(windows), "boundaries": boundaries,
         "segments": [{"id": i, "start_frame": s["start_frame"],
                       "end_frame": s["end_frame"],
                       "start_time": round(s["start_frame"] / fps, 4),
                       "end_time": round(s["end_frame"] / fps, 4),
                       "duration": round((s["end_frame"] - s["start_frame"]) / fps, 4)}
                      for i, s in enumerate(segments)]}, indent=2))
    if debug_windows:
        dbg = out_root / "labels" / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / f"{slug}.json").write_text(json.dumps(debug_windows, indent=2))
    return record


class DebugBudget:
    """Thread-safe: episodes now run concurrently under the vllm_endpoint
    backend, so decrementing must be atomic or the budget miscounts."""

    def __init__(self, n):
        self.n = n
        self.lock = threading.Lock()

    def pop(self):
        with self.lock:
            if self.n > 0:
                self.n -= 1
                return True
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_root = Path(cfg["paths"]["output_dir"]) / "labels"

    if args.episodes:
        paths = [data_dir / f"{e}.hdf5" for e in args.episodes]
    elif args.sample:
        paths = sample_episodes(data_dir, args.sample)
    else:
        ap.error("give --episodes or --sample")

    if cfg["vlm"]["backend"] == "vllm_endpoint":
        from vlm_backend import check_vlm_endpoint
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
        print(f"{rec['episode_id']}: {rec['num_windows']} windows -> "
              f"{len(rec['segments'])} segments")


if __name__ == "__main__":
    main()
