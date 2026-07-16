"""Level 1 — VLM transition-frame detection (ported from
../../ref_code/segment_transitions.py; see that file's docstring for the
full design rationale). No 3D pose is used anywhere in this pipeline.

Algorithm, unchanged from ref_code:
  - The episode's language instruction is read via common.resolve_description
    (same HDF5 attr resolution as Kinematics_pipeline).
  - The video is walked in NON-overlapping windows of `window_frames`
    consecutive source frames, stride == window size.
  - EVERY frame in the window is passed to the model as true video input
    (qwen_vl_utils' video_start/video_end/nframes mechanism), so nothing
    between boundaries is skipped — the frame indices the model sees are
    reported back in the prompt in absolute source-video coordinates.
  - Each window outputs only the transition frames it can see (or an empty
    list if the interaction state never changes inside that window).

Output shape is aligned to Kinematics_pipeline's boundaries.json (episode_id,
n_frames, fps, boundaries[], segments[]) so downstream Level 2/3 code and any
cross-pipeline comparison tooling can treat both pipelines' Level 1 output
uniformly. `source` on a boundary is "vlm_transition" (or episode_start /
episode_end) instead of Kinematics_pipeline's pause/grasp/release/gaze.

CLI:
    python detect_transitions.py --episodes task/idx [...]
    python detect_transitions.py --sample 5
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
from common import (apply_task_overrides, episode_id, episode_slug,  # noqa: E402
                    load_config, load_task_config, resolve_description,
                    sample_episodes)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_backend import probe_video  # noqa: E402
from transitions_vlm_backend import EndpointBackend  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "level2_labeling"))
from keyframes import extract_keyframes  # noqa: E402


def tolerant_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def _transitions_instructions(instruction, task_hint, window_desc, frames_desc):
    hint_block = f"\nTask-specific guidance: {task_hint.strip()}\n" if task_hint else ""
    return f"""You are segmenting an egocentric manipulation video into subtasks.

The overall task being performed in the FULL video is:
  "{instruction}"
{hint_block}
You are shown ONE WINDOW of that video: {window_desc}
{frames_desc}

Find the TRANSITION frames inside this window. A transition is the single frame
where the hand-object interaction state logically changes, i.e. a subtask
boundary. Examples of transitions:
  - an empty hand makes contact and grasps an object (free -> holding)
  - a held object is released / set down (holding -> free)
  - a lid/container changes state (closed -> open, open -> closed)
  - a held object is inserted into / removed from another object

NOT transitions: the hand merely moving, reaching, hovering, repositioning, or
an action continuing. Only the instant the interaction state flips counts.

For each transition, report:
- frame: the ABSOLUTE frame number from the list above where the state flips
- before: 2-6 words, the state just before (e.g. "hand moving to tupperware")
- after: 2-6 words, the state just after (e.g. "hand holding lid")

If the interaction state NEVER changes in this window (e.g. the hand is only
moving, or one action is still in progress the whole time), return an empty
list -- that is a perfectly good answer. Do not invent transitions."""


def build_prompt(instruction, frame_entries, total_frames, task_hint=None):
    frame_lines = "\n".join(
        f"  Frame {abs_frame}: t={t:.2f}s" for abs_frame, t in frame_entries)
    first_frame, last_frame = frame_entries[0][0], frame_entries[-1][0]
    window_desc = (f"source frames {first_frame} to {last_frame} "
                   f"(of {total_frames} total), every frame, in "
                   f"chronological order:")
    body = _transitions_instructions(instruction, task_hint, window_desc,
                                     frame_lines)
    return body + """

Return ONLY a JSON object, no prose, no code fences:
{"transitions": [{"frame": <int>, "before": "<str>", "after": "<str>"}]}"""


def select_window_frames(w_start, w_end, window_frames, keyframes_per_window):
    """Evenly-spaced source frames within [w_start, w_end], always keeping
    the first and last (vllm_endpoint backend: caps how many of the
    window's frames are actually rendered to images and sent)."""
    candidates = list(range(w_start, w_end + 1))[:window_frames]
    if len(candidates) <= keyframes_per_window:
        return candidates
    step = (len(candidates) - 1) / (keyframes_per_window - 1)
    idxs = sorted({round(i * step) for i in range(keyframes_per_window)})
    return [candidates[i] for i in idxs]


def build_prompt_content(instruction, kf_data, total_frames, task_hint):
    """OpenAI-format content list (text + interleaved images) for the
    vllm_endpoint backend."""
    first, last = kf_data[0][0], kf_data[-1][0]
    window_desc = (f"{len(kf_data)} keyframes sampled from source frames "
                   f"{first} to {last} (of {total_frames} total), in "
                   f"chronological order, each labeled with its absolute "
                   f"source frame number and timestamp.")
    content = [{"type": "text", "text":
        _transitions_instructions(instruction, task_hint, window_desc, "")}]
    for frame, tag, path in kf_data:
        content.append({"type": "text", "text": f"Frame {frame}: t={tag}"})
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text":
        "Respond ONLY with JSON matching the schema."})
    return content


@torch.inference_mode()
def detect_window_transformers(model, processor, video_path, instruction,
                               window_start, window_end, fps, total_frames,
                               cfg, task_hint=None):
    """Run one window. Returns (list_of_transitions, frame_entries)."""
    v = cfg["level1_transitions"]
    # span-1, not span+1: qwen_vl_utils derives the available frame count from
    # the TIME range with its own rounding, which can come out one less than
    # the inclusive span on a tail window — decord then rejects nframes >
    # available, and this env has no torchvision fallback (bug first hit by
    # VLM_subtask_pipeline's sweep on staple_paper/0).
    nframes = window_end - window_start
    nframes = max(2, nframes - (nframes % 2))  # qwen_vl_utils wants an even count

    video_content = {
        "type": "video",
        "video": f"file://{Path(video_path).resolve()}",
        "video_start": window_start / fps,
        "video_end": window_end / fps,
        "nframes": nframes,
    }

    # First pass: let qwen_vl_utils pick the exact absolute frame indices so
    # the prompt can list them with real timestamps.
    _, videos_probe, _ = process_vision_info(
        [{"role": "user", "content": [video_content]}],
        return_video_kwargs=True, return_video_metadata=True)
    _, metadata_probe = videos_probe[0]
    frame_entries = [(int(idx), float(idx) / fps)
                     for idx in metadata_probe["frames_indices"]]

    prompt = build_prompt(instruction, frame_entries, total_frames, task_hint)
    messages = [{"role": "user",
                "content": [video_content, {"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True)
    video_kwargs = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                    for k, v in video_kwargs.items()}
    videos, video_metadata = zip(*videos)
    inputs = processor(
        text=[text], images=images, videos=list(videos),
        video_metadata=list(video_metadata), return_tensors="pt",
        **video_kwargs).to(model.device)

    generated = model.generate(
        **inputs, max_new_tokens=v["max_new_tokens"], do_sample=True,
        temperature=v["temperature"], repetition_penalty=v["repetition_penalty"])
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed = tolerant_json(out_text)
    return parsed.get("transitions") or [], frame_entries


def detect_window_vllm(backend, video_path, instruction, window_start,
                       window_end, fps, total_frames, cfg, task_hint,
                       kf_out_dir, window_index):
    v = cfg["level1_transitions"]
    frames = select_window_frames(window_start, window_end, v["window_frames"],
                                  v["keyframes_per_window"])
    frame_tags = [(f, f"{f / fps:.2f}s") for f in frames]
    kf = extract_keyframes(video_path, frame_tags, v["keyframe_size"],
                           kf_out_dir, window_index)
    content = build_prompt_content(instruction, kf, total_frames, task_hint)
    out_text = backend.generate(content)
    parsed = tolerant_json(out_text)
    frame_entries = [(f, f / fps) for f, _, _ in kf]
    return parsed.get("transitions") or [], frame_entries


def process_episode(backend, h5_path, cfg):
    """Full Level 1 result for one episode. Returns the boundaries dict;
    does NOT write it to disk (caller decides, so run_pipeline.py can chain
    straight into Level 2 without a round-trip through the filesystem).

    backend: ("transformers", model, processor) or ("vllm_endpoint", EndpointBackend)."""
    task_name = h5_path.parent.name
    cfg = apply_task_overrides(cfg, task_name)
    v = cfg["level1_transitions"]
    video_path = h5_path.with_suffix(".mp4")
    with h5py.File(h5_path, "r") as f:
        instruction = resolve_description(dict(f.attrs), cfg)
    task_hint = load_task_config(task_name).get("level1_hint")

    duration, fps = probe_video(video_path, cfg["video"]["fps_fallback"])
    total_frames = int(round(duration * fps))
    window_frames = v["window_frames"]
    kf_out_dir = Path(cfg["paths"]["output_dir"]) / "boundaries" / "keyframes" / episode_slug(h5_path)

    all_transitions = []
    window_start, window_index = 0, 0
    while window_start < total_frames:
        window_end = min(window_start + window_frames - 1, total_frames - 1)

        transitions, last_err = None, None
        for attempt in range(1, v["max_retries"] + 1):
            try:
                if backend[0] == "transformers":
                    _, model, processor = backend
                    transitions, _ = detect_window_transformers(
                        model, processor, video_path, instruction,
                        window_start, window_end, fps, total_frames, cfg,
                        task_hint)
                else:
                    _, endpoint = backend
                    transitions, _ = detect_window_vllm(
                        endpoint, video_path, instruction, window_start,
                        window_end, fps, total_frames, cfg, task_hint,
                        kf_out_dir, window_index)
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_err = e
                if attempt < v["max_retries"]:
                    time.sleep(v["retry_backoff_sec"] * attempt)
        else:
            raise RuntimeError(
                f"window[{window_index}] failed after {v['max_retries']} "
                f"attempts: {last_err}")

        for tr in transitions:
            frame = int(tr["frame"])
            if window_start <= frame <= window_end:
                all_transitions.append({
                    "frame": frame, "time": round(frame / fps, 4),
                    "before": str(tr.get("before", "")).strip(),
                    "after": str(tr.get("after", "")).strip(),
                    "window": window_index})
            # else: out-of-range frame silently dropped, same as ref_code

        window_start += window_frames
        window_index += 1

    all_transitions.sort(key=lambda t: t["frame"])

    boundaries = (
        [{"frame": 0, "time": 0.0, "source": "episode_start",
          "before": None, "after": None}]
        + [{"frame": t["frame"], "time": t["time"], "source": "vlm_transition",
           "before": t["before"], "after": t["after"]} for t in all_transitions]
        + [{"frame": total_frames - 1, "time": round((total_frames - 1) / fps, 4),
           "source": "episode_end", "before": None, "after": None}]
    )
    segments = []
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        if b["frame"] <= a["frame"]:
            continue
        segments.append({
            "id": len(segments), "start_frame": a["frame"], "end_frame": b["frame"],
            "start_time": round(a["frame"] / fps, 4),
            "end_time": round(b["frame"] / fps, 4),
            "duration": round((b["frame"] - a["frame"]) / fps, 4),
            "start_source": a["source"], "end_source": b["source"],
            "transition_before": a["before"], "transition_after": a["after"],
        })

    return {
        "episode_id": episode_id(h5_path), "n_frames": total_frames, "fps": fps,
        "window_frames": window_frames, "num_windows": window_index,
        "boundaries": boundaries, "segments": segments,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        paths = [data_dir / f"{e}.hdf5" for e in args.episodes]
    elif args.sample:
        paths = sample_episodes(data_dir, args.sample)
    else:
        ap.error("give --episodes or --sample")

    if cfg["vlm"]["backend"] == "vllm_endpoint":
        from transitions_vlm_backend import check_vlm_endpoint
        check_vlm_endpoint(cfg)
        backend = ("vllm_endpoint", EndpointBackend(cfg))
    else:
        from model_backend import load_model
        model, processor = load_model(cfg["paths"]["model_path"],
                                      cfg["model"]["attn_implementation"],
                                      cfg["model"]["dtype"])
        backend = ("transformers", model, processor)

    for p in paths:
        out_file = out_dir / f"{episode_slug(p)}.json"
        if out_file.exists() and not args.force:
            print(f"skip (exists): {episode_id(p)}")
            continue
        result = process_episode(backend, p, cfg)
        out_file.write_text(json.dumps(result, indent=2))
        print(f"{result['episode_id']}: {result['n_frames']} frames, "
              f"{result['num_windows']} windows -> {len(result['segments'])} "
              f"segments -> {out_file}")


if __name__ == "__main__":
    main()
