"""Transition-frame detector for EgoDex videos using Qwen3-VL.

Goal: break a video into subtask-level segments by finding TRANSITION frames --
the moments where the hand's interaction state logically changes (empty hand
grasps an object, object is released, lid comes free, object crosses into a
container, ...). The transition frames are the segment boundaries.

How it works:
  - The episode's language instruction is read from the paired N.hdf5 file
    (root attr `llm_description`, or `llm_description2` when
    `which_llm_description == 2` on reversible tasks).
  - The video is walked in NON-overlapping windows of WINDOW_FRAMES (48)
    consecutive source frames, stride 48: frames 0-47, 48-95, 96-143, ...
    Every frame in the window is passed to the model (nframes == window size),
    so nothing between boundaries is skipped.
  - Frame indices come from qwen_vl_utils' video_start/video_end/nframes
    mechanism and are ABSOLUTE source-video frame numbers (same verified
    mechanism as process_sliding_window.py), so the model reports transition
    frames directly in source coordinates.
  - Each window must output ONLY the transition frames it can see, or an empty
    list if the interaction state never changes inside that window.

Model loading and video preprocessing are reused verbatim from
process_native.py (load_model, VIDEO_MAX_PIXELS, process_vision_info with
return_video_metadata) -- nothing else from that pipeline is used.
"""

import argparse
import json
import os
import sys
import time

import h5py
import torch
from qwen_vl_utils import process_vision_info

# The model-loading/video helpers live in the robot_labeling folder.
sys.path.insert(0, "/home/user/Desktop/qwen_instruct/robot_labeling")

from process import probe_video
from process_native import load_model, HF_MODEL_DEFAULT, VIDEO_MAX_PIXELS, _extract_json

WINDOW_FRAMES = 48        # frames per window; also the stride (no overlap)
MAX_NEW_TOKENS = 512      # a window only ever has a few transitions to report
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 3.0


def read_instruction(hdf5_path: str) -> str:
    """The episode's language instruction, honoring which_llm_description on
    reversible tasks (see EgoDex attrs: llm_description / llm_description2)."""
    with h5py.File(hdf5_path, "r") as f:
        a = f.attrs
        if a.get("which_llm_description", 1) == 2 and "llm_description2" in a:
            return str(a["llm_description2"])
        return str(a["llm_description"])


def build_prompt(instruction: str, frame_entries: list, total_frames: int) -> str:
    frame_lines = "\n".join(
        f"  Frame {abs_frame}: t={t:.2f}s" for abs_frame, t in frame_entries
    )
    first_frame = frame_entries[0][0]
    last_frame = frame_entries[-1][0]

    return f"""You are segmenting an egocentric manipulation video into subtasks.

The overall task being performed in the FULL video is:
  "{instruction}"

You are shown ONE WINDOW of that video: source frames {first_frame} to {last_frame}
(of {total_frames} total), every frame, in chronological order:
{frame_lines}

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
list -- that is a perfectly good answer. Do not invent transitions.

Return ONLY a JSON object, no prose, no code fences:
{{"transitions": [{{"frame": <int>, "before": "<str>", "after": "<str>"}}]}}"""


@torch.inference_mode()
def detect_window(model, processor, video_path: str, instruction: str,
                  window_start: int, window_end: int, fps: float,
                  total_frames: int) -> tuple:
    """Run one window. Returns (list_of_transitions, frame_entries)."""
    nframes = window_end - window_start + 1
    nframes = max(2, nframes - (nframes % 2))  # qwen_vl_utils wants an even count

    video_content = {
        "type": "video",
        "video": f"file://{os.path.abspath(video_path)}",
        "video_start": window_start / fps,
        "video_end": window_end / fps,
        "nframes": nframes,
        "max_pixels": VIDEO_MAX_PIXELS,
    }

    # First pass: let qwen_vl_utils pick the exact absolute frame indices so the
    # prompt can list them with real timestamps (same probe pattern as
    # process_sliding_window.py).
    _, videos_probe, _ = process_vision_info(
        [{"role": "user", "content": [video_content]}],
        return_video_kwargs=True, return_video_metadata=True,
    )
    _, metadata_probe = videos_probe[0]
    frame_entries = [
        (int(idx), float(idx) / fps) for idx in metadata_probe["frames_indices"]
    ]

    prompt = build_prompt(instruction, frame_entries, total_frames)
    messages = [{
        "role": "user",
        "content": [video_content, {"type": "text", "text": prompt}],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True,
    )
    video_kwargs = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in video_kwargs.items()
    }
    videos, video_metadata = zip(*videos)
    inputs = processor(
        text=[text], images=images, videos=list(videos), video_metadata=list(video_metadata),
        return_tensors="pt", **video_kwargs,
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.3,
    )
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed = _extract_json(out_text)
    return parsed.get("transitions") or [], frame_entries


def process_video(video_path: str, model, processor) -> dict:
    hdf5_path = os.path.splitext(video_path)[0] + ".hdf5"
    instruction = read_instruction(hdf5_path)

    duration, fps = probe_video(video_path)
    total_frames = int(round(duration * fps))
    print(f"  instruction: {instruction}")
    print(f"  {duration:.2f}s @ {fps:.1f}fps = {total_frames} frames "
          f"-> {-(-total_frames // WINDOW_FRAMES)} window(s) of {WINDOW_FRAMES}")

    all_transitions = []
    window_start = 0
    window_index = 0
    while window_start < total_frames:
        window_end = min(window_start + WINDOW_FRAMES - 1, total_frames - 1)

        transitions = None
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                transitions, frame_entries = detect_window(
                    model, processor, video_path, instruction,
                    window_start, window_end, fps, total_frames,
                )
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_err = e
                print(f"    window[{window_index}] attempt {attempt}/{MAX_RETRIES} failed ({e})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)
        else:
            raise last_err or RuntimeError(f"window[{window_index}] failed after retries")

        # Keep only transitions whose frame actually lies inside this window.
        kept = []
        for tr in transitions:
            frame = int(tr["frame"])
            if window_start <= frame <= window_end:
                kept.append({
                    "frame": frame,
                    "time_sec": round(frame / fps, 3),
                    "before": str(tr.get("before", "")).strip(),
                    "after": str(tr.get("after", "")).strip(),
                    "window": window_index,
                })
            else:
                print(f"    window[{window_index}] dropped out-of-range frame {frame}")

        label = ", ".join(str(t["frame"]) for t in kept) if kept else "none"
        print(f"    window[{window_index}] frames {window_start}-{window_end}: "
              f"transitions at [{label}]")
        all_transitions.extend(kept)

        window_start += WINDOW_FRAMES
        window_index += 1

    all_transitions.sort(key=lambda t: t["frame"])

    # Transitions define the segment boundaries.
    boundaries = [0] + [t["frame"] for t in all_transitions] + [total_frames - 1]
    segments = [
        {"start_frame": boundaries[i], "end_frame": boundaries[i + 1]}
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]

    return {
        "video": os.path.basename(video_path),
        "instruction": instruction,
        "fps": fps,
        "duration_sec": round(duration, 3),
        "total_frames": total_frames,
        "window_frames": WINDOW_FRAMES,
        "num_windows": window_index,
        "transitions": all_transitions,
        "segments": segments,
    }


def main():
    global WINDOW_FRAMES

    parser = argparse.ArgumentParser(
        description="Detect subtask transition frames in an EgoDex video using "
                    "Qwen3-VL over non-overlapping 48-frame windows."
    )
    parser.add_argument("video", help="Path to one N.mp4 (paired N.hdf5 must sit next to it)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: <video_stem>_transitions.json)")
    parser.add_argument("-m", "--model", default=HF_MODEL_DEFAULT)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    args = parser.parse_args()

    WINDOW_FRAMES = args.window_frames
    output_path = args.output or os.path.splitext(args.video)[0] + "_transitions.json"

    model, processor = load_model(args.model, args.attn)

    print(f"Processing {args.video}...")
    t0 = time.time()
    result = process_video(args.video, model, processor)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[SUCCESS] {len(result['transitions'])} transition(s), "
          f"{len(result['segments'])} segment(s), {time.time()-t0:.1f}s "
          f"-> {output_path}")


if __name__ == "__main__":
    main()
