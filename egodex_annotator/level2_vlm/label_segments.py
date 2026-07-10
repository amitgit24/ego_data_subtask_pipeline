"""Level 2 — Qwen3-VL-32B semantic labeling of Level 1 segments.

The VLM classifies and describes; it never chooses or adjusts boundaries.
Backends: an OpenAI-compatible vLLM endpoint (primary) or local transformers,
selected in config. Action vocabulary and output shape are enforced with
JSON-schema-guided decoding, temperature 0.

CLI:
    python label_segments.py --sample 5
    python label_segments.py --episodes task_name/12 ...
    python label_segments.py --all [--force]
"""

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "level1_kinematics"))

import h5py  # noqa: E402

from boundaries import propose_boundaries  # noqa: E402
from common import (episode_slug, load_config, resolve_description,  # noqa: E402
                    sample_episodes)
from hand_assignment import assign_hand, kinematic_summary  # noqa: E402
from keyframes import extract_keyframes, select_keyframes  # noqa: E402
from taxonomy import (ACTION_GROUPS, RESPONSE_SCHEMA, STYLE_RULES,  # noqa: E402
                      STYLES, style_for)

GLOSSES = {
    "transfer": "object moved from point A to B",
    "handover": "object passed between the two hands",
    "hold_steady": "one hand stabilizes while the other acts",
    "align": "bringing two objects/parts together",
    "adjust_grip": "regrasp or shift fingers without moving the object",
    "other": "none of the above fits; describe what you saw",
}


def taxonomy_block():
    lines = []
    for group, actions in ACTION_GROUPS.items():
        items = [f"{a} ({GLOSSES[a]})" if a in GLOSSES else a for a in actions]
        lines.append(f"- {group}: {', '.join(items)}")
    return "\n".join(lines)


def build_prompt(segment, task_desc, kin_summary, hand, style_id, style,
                 keyframe_data, n_segments, episode_dur, style_variation,
                 prev_context=""):
    """Returns OpenAI-format message content list (text + images interleaved)."""
    content = [{"type": "text", "text":
        "You label one segment of an egocentric human manipulation episode "
        "recorded from a head-mounted camera. Keyframes of THIS SEGMENT ONLY "
        "follow, in temporal order."}]
    for frame, tag, path, t in keyframe_data:
        content.append({"type": "text", "text": f"Frame at t={t:.1f}s ({tag}):"})
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    style_line = ""
    if style_variation:
        style_line = (f"Style for the 'subtask' sentence: {style[1]} "
                      f"Example shape: \"{style[2]}\"\n")

    content.append({"type": "text", "text": f"""
Overall task (context for the WHOLE episode): {task_desc}
This is segment {segment['id'] + 1} of {n_segments}, t={segment['start_time']:.1f}s to {segment['end_time']:.1f}s of a {episode_dur:.1f}s episode.
Kinematic context (computed from 3D hand tracking, trust it for motion facts): {kin_summary}
{prev_context}
IMPORTANT: the overall task describes the whole episode, but this segment may
be only one phase of it — approaching, aligning, transporting, placing,
retracting, holding, or idling. Do NOT copy the task verb unless the frames
of THIS segment actually show that action happening. BUT do take the purpose
of a motion from the overall task rather than inventing one: in a removal
task, rotating a part means loosening/detaching it (not "adjusting its
orientation"); in an insertion task, aligning a part is preparation for
inserting. If the purpose is not implied by the task, state only the motion.
Keep the episode story consistent with the previously labeled segments: an
action that was already completed does not happen again unless the frames
clearly show it repeating. Label only what changes between the first and
last frame shown.

Classify this segment. Choose 'action' from this closed vocabulary ONLY:
{taxonomy_block()}

The acting hand was computed from kinematics as: {hand}. Do not override it; if the frames clearly contradict it, set hand_disagreement=true.
{style_line}{STYLE_RULES}

Respond ONLY with JSON: {{"action": "...", "object": "<short noun phrase with one visible attribute>", "subtask": "<one sentence>", "confidence": "high|medium|low", "hand_disagreement": false}}"""})
    return content


# ---------------------------------------------------------------- backends

class EndpointBackend:
    def __init__(self, cfg):
        from openai import OpenAI
        v = cfg["vlm"]
        self.client = OpenAI(base_url=v["endpoint_url"], api_key=v["api_key"])
        self.model = v["model_name"]
        self.temperature = v["temperature"]
        self.max_tokens = v["max_tokens"]

    def generate(self, content):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "segment_label", "schema": RESPONSE_SCHEMA}},
        )
        return resp.choices[0].message.content


class TransformersBackend:
    """Local fallback; loads the model into this process (slow, debug only)."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        path = cfg["paths"]["model_path"]
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto")
        self.max_tokens = cfg["vlm"]["max_tokens"]

    def generate(self, content):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": c["image_url"]["url"]}
            if c["type"] == "image_url" else c for c in content]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                  do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def make_backend(cfg):
    kind = cfg["vlm"]["backend"]
    if kind == "vllm_endpoint":
        return EndpointBackend(cfg)
    if kind == "transformers":
        return TransformersBackend(cfg)
    raise ValueError(f"unknown vlm.backend: {kind}")


# ---------------------------------------------------------------- labeling

def tolerant_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def label_one(backend, content, max_retries):
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            raw = backend.generate(content)
            return tolerant_json(raw), raw, attempt
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e  # parse failure: re-ask once per spec, then give up
            content = content + [{"type": "text", "text":
                "Your previous answer was not valid JSON. Respond ONLY with "
                "the JSON object."}]
        except Exception as e:  # network/server: backoff and retry
            last_err = e
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"labeling failed after {max_retries} attempts: {last_err}")


def process_episode(h5_path, cfg, backend, debug_budget):
    v = cfg["vlm"]
    out_root = Path(cfg["paths"]["output_dir"]) / "labels"
    slug = episode_slug(h5_path)
    out_file = out_root / f"{slug}.json"

    bdir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    bfile = bdir / f"{slug}.json"
    if bfile.exists():
        result = json.loads(bfile.read_text())
        _, sig = propose_boundaries(h5_path, cfg)  # signals for hand scores
    else:
        result, sig = propose_boundaries(h5_path, cfg)
        bdir.mkdir(parents=True, exist_ok=True)
        bfile.write_text(json.dumps(result, indent=2))

    with h5py.File(h5_path, "r") as f:
        task_desc = resolve_description(dict(f.attrs), cfg)

    fps = result["fps"]
    episode_dur = (result["n_frames"] - 1) / fps
    segments = result["segments"]
    style_variation = v["style_variation"]

    # Segments are labeled IN ORDER so each prompt carries the story so far —
    # without this the model re-applies the task verb to every segment
    # (approach and retract phases all become "remove"). Parallelism happens
    # across episodes in main() instead.
    labeled = []
    for segment in segments:
        hand_info = assign_hand(sig, segment, cfg)
        kin = kinematic_summary(sig, segment, hand_info, fps)
        frames = select_keyframes(segment, v["keyframes_per_segment"])
        kf = extract_keyframes(h5_path.with_suffix(".mp4"), frames,
                               v["keyframe_size"],
                               out_root / "keyframes" / slug, segment["id"])
        kf_data = [(fr, tag, path, fr / fps) for fr, tag, path in kf]
        style_id, style = (style_for(result["episode_id"], segment["id"])
                           if style_variation else (0, STYLES[0]))
        prev_context = ""
        if labeled:
            lines = [f"  segment {s['id'] + 1} ({s['start_time']:.1f}-"
                     f"{s['end_time']:.1f}s): {s['vlm']['action']} — "
                     f"{s['vlm']['subtask']}" for s in labeled[-3:]]
            prev_context = ("Previously labeled segments of this episode:\n"
                            + "\n".join(lines) + "\n")
        content = build_prompt(segment, task_desc, kin, hand_info["hand"],
                               style_id, style, kf_data, len(segments),
                               episode_dur, style_variation, prev_context)
        label, raw, attempts = label_one(backend, content, v["max_retries"])
        enriched = dict(segment)
        enriched.update({
            "hand": hand_info["hand"], "hand_scores": hand_info,
            "style_id": style_id, "keyframes": [p for _, _, p in kf],
            "vlm": label, "vlm_attempts": attempts,
        })
        labeled.append(enriched)
        idx = debug_budget.take() if debug_budget else None
        if idx is not None:
            dbg = out_root / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            text_parts = [c["text"] for c in content if c["type"] == "text"]
            (dbg / f"seg{idx:03d}.json").write_text(json.dumps(
                {"episode": result["episode_id"], "segment": enriched["id"],
                 "prompt_text": text_parts,
                 "keyframes": enriched["keyframes"], "raw_response": raw},
                indent=2))
    out = {"episode_id": result["episode_id"], "task": task_desc,
           "fps": fps, "n_frames": result["n_frames"], "segments": labeled}
    out_root.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2))
    return out


class DebugBudget:
    """Thread-safe: episodes label in parallel, so the index must be claimed
    atomically or two threads write the same debug file."""

    def __init__(self, n):
        self.n = n
        self.next_idx = 0
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if self.next_idx >= self.n:
                return None
            self.next_idx += 1
            return self.next_idx - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_root = Path(cfg["paths"]["output_dir"]) / "labels"

    if args.episodes:
        paths = [data_dir / f"{e}.hdf5" for e in args.episodes]
    elif args.sample:
        paths = sample_episodes(data_dir, args.sample,
                                min_frames=8 * cfg["video"]["fps"])
    elif args.all:
        paths = sorted(data_dir.glob("*/*.hdf5"))
    else:
        ap.error("give --episodes, --sample or --all")

    backend = make_backend(cfg)
    debug_budget = DebugBudget(cfg["vlm"]["debug_first_n"])
    skipped, done, failed = 0, 0, []
    t0 = time.time()
    n_segs = 0

    todo = []
    for p in paths:
        out_file = out_root / f"{episode_slug(p)}.json"
        if out_file.exists() and not args.force:
            skipped += 1
        else:
            todo.append(p)

    # segments are sequential within an episode (story context), so the
    # parallelism unit is the episode
    with ThreadPoolExecutor(max_workers=cfg["vlm"]["concurrency"]) as pool:
        futures = {pool.submit(process_episode, p, cfg, backend, debug_budget): p
                   for p in todo}
        for fut, p in futures.items():
            try:
                out = fut.result()
                done += 1
                n_segs += len(out["segments"])
                print(f"[{done}] {out['episode_id']}: {len(out['segments'])} "
                      f"segments labeled")
            except Exception as e:
                failed.append((str(p), repr(e)))
                print(f"FAILED {p}: {e!r}", file=sys.stderr)

    dt = time.time() - t0
    print(f"\n{done} episodes labeled ({n_segs} segments) in {dt:.0f}s "
          f"({n_segs / max(dt, 1e-9) * 60:.1f} seg/min), {skipped} skipped, "
          f"{len(failed)} failed")
    if failed:
        skip_file = out_root / "skip_list.json"
        existing = json.loads(skip_file.read_text()) if skip_file.exists() else []
        skip_file.write_text(json.dumps(existing + failed, indent=2))
        print(f"failures appended to {skip_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
