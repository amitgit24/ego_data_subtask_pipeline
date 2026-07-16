"""Level 2 — per-segment semantic labeling (action, hand, object, sentence).

Structural differences from Kinematics_pipeline's label_segments.py:
  - No kinematic context, no pre-computed hand: the VLM is asked for hand
    directly, since there is no pose signal to compute it from.
  - No vLLM guided decoding available (local transformers generate() has no
    schema constraint), so the closed vocabulary is enforced POST-HOC:
    action/hand/confidence are validated against the imported taxonomy after
    parsing, and the request is retried on violation instead of being
    impossible to violate in the first place. This is a real trade-off
    against Kinematics_pipeline's guided decoding, not a bug.
  - Otherwise identical in spirit: segments are labeled IN ORDER within an
    episode so each prompt carries the story so far (the fix that resolved
    task-verb leakage in Kinematics_pipeline, ported here from day one
    instead of rediscovering the same bug empirically), and the same 6
    sentence style templates / action taxonomy are imported from
    Kinematics_pipeline for direct comparability.

CLI:
    python label_segments.py --episodes task/idx [...]
    python label_segments.py --sample 5
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ACTION_GROUPS, ACTIONS, CONFIDENCES, HANDS,  # noqa: E402
                    apply_task_overrides, embodiment_for, episode_id,
                    episode_slug, load_config, load_task_config,
                    resolve_description, sample_episodes, select_prompt_hint,
                    style_for)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "level1_transitions"))
from detect_transitions import process_episode as detect_boundaries  # noqa: E402
from detect_transitions import tolerant_json  # noqa: E402

from keyframes import extract_keyframes, select_keyframes  # noqa: E402
from labeling_vlm_backend import EndpointBackend  # noqa: E402

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


def build_prompt(segment, task_desc, n_segments, episode_dur, style_id, style,
                 style_variation, prev_context, task_hint=None,
                 embodiment="hand"):
    hand_line = ", ".join(HANDS)
    style_line = ""
    if style_variation:
        style_line = (f"Style for the 'subtask' sentence: {style[1]} "
                      f"Example shape: \"{style[2]}\"\n")
    hint_block = f"\nTask-specific guidance: {task_hint.strip()}\n" if task_hint else ""
    return f"""You label one segment of an egocentric human manipulation video.
Keyframes of THIS SEGMENT ONLY follow, in temporal order.

Overall task (context for the WHOLE episode): {task_desc}
This is segment {segment['id'] + 1} of {n_segments}, t={segment['start_time']:.1f}s to {segment['end_time']:.1f}s of a {episode_dur:.1f}s episode.
This segment's boundaries were found by watching the video directly: it starts right after "{segment['transition_before'] or 'the episode began'}" -> "{segment['transition_after'] or '(episode start)'}".
{hint_block}{prev_context}
IMPORTANT: the overall task describes the whole episode, but this segment may
be only one phase of it. Do NOT copy the task verb unless the frames of THIS
segment actually show that action happening. Take the purpose of a motion
from the overall task rather than inventing one. Keep the episode story
consistent with the previously labeled segments: an action already completed
does not happen again unless the frames clearly show it repeating. Label
only what changes between the first and last frame shown.

Classify this segment. The 'action' field MUST be exactly ONE of these
words — these are the ONLY valid values, and a group/category name such as
"object_state" or "acquire_release" is NEVER a valid answer:
{", ".join(ACTIONS)}

The same words are grouped by kind below only to help you pick the right one
(do NOT answer with a group name — answer with a single word from a group):
{taxonomy_block()}

Choose 'hand' from this closed vocabulary ONLY: {hand_line}
(there is no separate hand sensor here — judge it from the frames). Pick the
SINGLE hand ("left"/"right") if only one hand is actively manipulating the
object, even if the other hand is visible in frame resting, idle, or merely
nearby without gripping anything. Use "both" only when both hands are each
independently manipulating. Use "both_coordinating" only when both hands are
working AS ONE unit on the same grip/motion (e.g. passing an object
hand-to-hand, or both hands gripping one object together). A hand merely
being in frame is not evidence it is acting.
{style_line}{style[1] if not style_variation else ''}
Exactly one short sentence (8-14 words) for 'subtask', present tense, never
starting with a subordinate clause. Call the end-effector "{embodiment}"
(e.g. "left {embodiment}", "both {embodiment}s") in every sentence, even if
a style example shows a different word. Name the object with one visible
distinguishing attribute when possible.

Respond ONLY with JSON: {{"action": "<one of the taxonomy actions>",
 "hand": "<one of: {hand_line}>",
 "object": "<short noun phrase with one visible attribute>",
 "subtask": "<one sentence>", "confidence": "high|medium|low"}}"""


@torch.inference_mode()
def generate_label_transformers(model, processor, keyframe_paths, prompt_text, v):
    content = []
    for path, tag, t in keyframe_paths:
        content.append({"type": "text", "text": f"Frame at t={t:.1f}s ({tag}):"})
        content.append({"type": "image", "image": f"file://{Path(path).resolve()}"})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    images, _ = process_vision_info(messages)  # image-only: 2-tuple (no video kwargs)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(
        model.device)
    generated = model.generate(**inputs, max_new_tokens=v["max_new_tokens"],
                               do_sample=(v["temperature"] > 0),
                               temperature=max(v["temperature"], 1e-5))
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def generate_label_vllm(endpoint, keyframe_paths, prompt_text):
    import base64
    content = []
    for path, tag, t in keyframe_paths:
        content.append({"type": "text", "text": f"Frame at t={t:.1f}s ({tag}):"})
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt_text})
    return endpoint.generate(content)


def valid_label(label):
    return (label.get("action") in ACTIONS and label.get("hand") in HANDS
           and label.get("confidence") in CONFIDENCES
           and isinstance(label.get("object"), str) and label["object"]
           and isinstance(label.get("subtask"), str) and label["subtask"])


def label_one(backend, keyframe_paths, prompt_text, v):
    text = prompt_text
    last_err = None
    for attempt in range(v["max_retries"]):
        try:
            if backend[0] == "transformers":
                _, model, processor = backend
                raw = generate_label_transformers(model, processor, keyframe_paths, text, v)
            else:
                _, endpoint = backend
                raw = generate_label_vllm(endpoint, keyframe_paths, text)
            label = tolerant_json(raw)
            if not valid_label(label):
                raise ValueError(f"label failed vocabulary check: {label}")
            return label, raw, attempt
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            text = prompt_text + ("\n\nYour previous answer was invalid "
                                  f"({e}). Respond ONLY with the JSON object, "
                                  "using exactly the vocabulary given.")
    # Fail loudly but don't drop the segment: emit an explicit low-confidence
    # 'other' label so the episode stays fully covered; Level 3's
    # low_confidence flag will surface it for review.
    return ({"action": "other", "hand": "both", "object": "unknown",
            "subtask": "Unable to reliably label this segment.",
            "confidence": "low"}, str(last_err), v["max_retries"])


def process_episode(backend, h5_path, cfg, debug_budget=None):
    """backend: ("transformers", model, processor) or ("vllm_endpoint", EndpointBackend)."""
    task_name = h5_path.parent.name
    cfg = apply_task_overrides(cfg, task_name)
    v = cfg["level2_labeling"]
    out_root = Path(cfg["paths"]["output_dir"]) / "labels"
    slug = episode_slug(h5_path)

    bdir = Path(cfg["paths"]["output_dir"]) / "boundaries"
    bfile = bdir / f"{slug}.json"
    if bfile.exists():
        result = json.loads(bfile.read_text())
    else:
        # Level 1's guided-JSON schema (transitions array) differs from
        # Level 2's (single action/hand/object/subtask/confidence) — the
        # vllm_endpoint backend is schema-bound at construction, so THIS
        # level's backend cannot be reused for Level 1's call (found the
        # hard way: reusing it here silently forced every Level 1 response
        # into Level 2's schema, so "transitions" was always missing/empty,
        # not an error). transformers backend has no schema constraint, so
        # the (model, processor) pair is safely reusable as-is.
        if backend[0] == "vllm_endpoint":
            from transitions_vlm_backend import EndpointBackend as _L1Backend
            l1_backend = ("vllm_endpoint", _L1Backend(cfg))
        else:
            l1_backend = backend
        result = detect_boundaries(l1_backend, h5_path, cfg)
        bdir.mkdir(parents=True, exist_ok=True)
        bfile.write_text(json.dumps(result, indent=2))

    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        task_desc = resolve_description(attrs, cfg)
    task_cfg = load_task_config(task_name)
    level2_hint_cfg = {"prompt_hint_by_attr": task_cfg.get("level2_prompt_hint_by_attr"),
                       "prompt_hints": task_cfg.get("level2_prompt_hints")}
    task_hint = select_prompt_hint(level2_hint_cfg, attrs)
    embodiment = embodiment_for(episode_id(h5_path))

    fps = result["fps"]
    episode_dur = (result["n_frames"] - 1) / fps
    segments = result["segments"]

    labeled = []
    for segment in segments:
        frames = select_keyframes(segment, v["keyframes_per_segment"])
        kf = extract_keyframes(h5_path.with_suffix(".mp4"), frames,
                               v["keyframe_size"], out_root / "keyframes" / slug,
                               segment["id"])
        kf_data = [(path, tag, fr / fps) for fr, tag, path in kf]

        style_id, style = (style_for(result["episode_id"], segment["id"])
                           if v["style_variation"] else (0, (None, "plain, factual", "")))
        prev_context = ""
        if labeled:
            lines = [f"  segment {s['id'] + 1} ({s['start_time']:.1f}-"
                    f"{s['end_time']:.1f}s): {s['vlm']['action']} — "
                    f"{s['vlm']['subtask']}" for s in labeled[-3:]]
            prev_context = ("Previously labeled segments of this episode:\n"
                           + "\n".join(lines) + "\n")

        prompt = build_prompt(segment, task_desc, len(segments), episode_dur,
                              style_id, style, v["style_variation"], prev_context,
                              task_hint, embodiment)
        label, raw, attempts = label_one(backend, kf_data, prompt, v)

        enriched = dict(segment)
        enriched.update({"style_id": style_id, "keyframes": [p for _, _, p in kf],
                         "vlm": label, "vlm_attempts": attempts})
        labeled.append(enriched)

        if debug_budget and debug_budget.pop():
            dbg = out_root / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            # keyed by episode+segment so a --force rerun overwrites its own
            # captures instead of appending stale ones alongside them
            (dbg / f"{slug}_seg{enriched['id']:02d}.json").write_text(json.dumps(
                {"episode": result["episode_id"], "segment": enriched["id"],
                 "prompt": prompt, "keyframes": enriched["keyframes"],
                 "raw_response": raw}, indent=2))

    labeled.sort(key=lambda s: s["id"])
    out = {"episode_id": result["episode_id"], "task": task_desc, "fps": fps,
          "n_frames": result["n_frames"], "embodiment": embodiment,
          "segments": labeled}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{slug}.json").write_text(json.dumps(out, indent=2))
    return out


class DebugBudget:
    """Thread-safe: episodes run concurrently under the vllm_endpoint
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
        from labeling_vlm_backend import check_vlm_endpoint
        check_vlm_endpoint(cfg)
        backend = ("vllm_endpoint", EndpointBackend(cfg))
    else:
        from model_backend import load_model
        model, processor = load_model(cfg["paths"]["model_path"],
                                      cfg["model"]["attn_implementation"],
                                      cfg["model"]["dtype"])
        backend = ("transformers", model, processor)
    debug_budget = DebugBudget(cfg["level2_labeling"]["debug_first_n"])

    for p in paths:
        out_file = out_root / f"{episode_slug(p)}.json"
        if out_file.exists() and not args.force:
            print(f"skip (exists): {episode_id(p)}")
            continue
        out = process_episode(backend, p, cfg, debug_budget)
        print(f"{out['episode_id']}: {len(out['segments'])} segments labeled")


if __name__ == "__main__":
    main()
