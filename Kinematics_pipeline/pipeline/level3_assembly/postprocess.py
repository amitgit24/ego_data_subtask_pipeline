"""Level 3 — merge, idle labeling, hard constraints and cross-check flags.

Operates on Level 2 label records (dicts); emits plain dicts ready for the
pydantic schema. Never deletes data — suspicious segments are flagged and
queued for review, not dropped."""

import zlib

CONF_RANK = {"high": 2, "medium": 1, "low": 0}

NEEDS_GRASP = {"grasp", "pick", "lift"}
NEEDS_RELEASE = {"place", "put_down", "release", "drop"}
EVENT_NEAR_FRAMES = 15  # ±0.5 s tolerance when matching events to segments

IDLE_SENTENCES = [
    "Both hands rest on the table with no manipulation.",
    "The hands stay idle, not interacting with any object.",
    "No manipulation occurs; the hands remain still.",
]


def norm_obj(s):
    s = s.lower().strip()
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def mergeable(a, b, fps, merge_gap_s):
    # a real grasp/release event at this boundary means a new action cycle
    # genuinely started here (e.g. the next cup/object) -- never collapse
    # that even if the VLM assigned both sides the same coarse
    # action/object/hand triple (seen concretely: two segments "place lid
    # onto right cup" / "place lid onto left cup" share action=assemble,
    # object='white lid', hand='both_coordinating', so without this check
    # they silently merge into one, erasing a real per-cup split).
    if b["start_source"] in ("grasp", "release"):
        return False
    return (a["vlm"]["action"] == b["vlm"]["action"]
            and norm_obj(a["vlm"]["object"]) == norm_obj(b["vlm"]["object"])
            and a["hand"] == b["hand"]
            and (b["start_frame"] - a["end_frame"]) < merge_gap_s * fps)


def merge_segments(segments, fps, merge_gap_s):
    """Collapse adjacent same-action/object/hand segments; keep the
    higher-confidence segment's semantic fields and the outer bounds."""
    merged = []
    for seg in segments:
        if merged and mergeable(merged[-1], seg, fps, merge_gap_s):
            a = merged[-1]
            keep = a if (CONF_RANK[a["vlm"]["confidence"]]
                         >= CONF_RANK[seg["vlm"]["confidence"]]) else seg
            a["vlm"] = keep["vlm"]
            a["style_id"] = keep["style_id"]
            a["hand_scores"] = keep["hand_scores"]
            a["end_frame"] = seg["end_frame"]
            a["end_source"] = seg["end_source"]
            a["events_inside"] = a["events_inside"] + seg["events_inside"]
        else:
            merged.append(dict(seg))
    return merged


def insert_idle(segments, episode_id, n_frames, fps, idle_gap_s):
    """Explicit idle subtasks for uncovered gaps > idle_gap_s; shorter gaps
    are absorbed into the preceding segment so coverage stays total."""
    out = []
    cursor = 0
    for seg in segments:
        gap = seg["start_frame"] - cursor
        if gap > idle_gap_s * fps:
            out.append(_idle_segment(episode_id, cursor, seg["start_frame"],
                                     len(out)))
        elif gap > 0 and out:
            out[-1]["end_frame"] = seg["start_frame"]
        elif gap > 0:
            seg = dict(seg, start_frame=cursor)
        out.append(seg)
        cursor = seg["end_frame"]
    tail = (n_frames - 1) - cursor
    if tail > idle_gap_s * fps:
        out.append(_idle_segment(episode_id, cursor, n_frames - 1, len(out)))
    elif tail > 0:
        out[-1]["end_frame"] = n_frames - 1
    return out


def _idle_segment(episode_id, start, end, idx):
    sentence = IDLE_SENTENCES[(zlib.crc32(episode_id.encode()) + idx)
                              % len(IDLE_SENTENCES)]
    return {
        "start_frame": start, "end_frame": end,
        "start_source": "pause", "end_source": "pause",
        "events_inside": [], "hand": "both", "style_id": -1,
        "vlm": {"action": "idle", "object": "none", "subtask": sentence,
                "confidence": "high", "hand_disagreement": False},
        "hand_scores": None, "auto_idle": True,
    }


def cross_check_flags(seg, all_events):
    """Kinematics vs semantics; flags only, never deletion."""
    flags = []
    action = seg["vlm"]["action"]
    lo = seg["start_frame"] - EVENT_NEAR_FRAMES
    hi = seg["end_frame"] + EVENT_NEAR_FRAMES
    near = [e for e in all_events if lo <= e["frame"] <= hi]
    if action in NEEDS_GRASP and not any(e["type"] == "grasp" for e in near):
        flags.append("kinematic_mismatch")
    if action in NEEDS_RELEASE and not any(e["type"] == "release" for e in near):
        flags.append("kinematic_mismatch")
    if seg["vlm"]["confidence"] == "low":
        flags.append("low_confidence")
    if seg["vlm"]["hand_disagreement"]:
        flags.append("hand_disagreement")
    return flags


def postprocess(label_record, cfg):
    """Full Level 3 pass for one episode's Level 2 output.

    Returns (subtask dicts, review_queue). Hard constraints are asserted
    here and re-checked by the pydantic schema on assembly."""
    a = cfg["assembly"]
    fps = label_record["fps"]
    n_frames = label_record["n_frames"]
    episode_id = label_record["episode_id"]
    min_len = cfg["kinematics"]["min_segment_frames"]

    segments = sorted(label_record["segments"], key=lambda s: s["start_frame"])
    segments = merge_segments(segments, fps, a["merge_gap_s"])
    segments = insert_idle(segments, episode_id, n_frames, fps, a["idle_gap_s"])

    all_events = [e for s in label_record["segments"]
                  for e in s["events_inside"]]

    subtasks, review = [], []
    for i, seg in enumerate(segments):
        flags = cross_check_flags(seg, all_events)
        if flags:
            review.append(i)
        v = seg["vlm"]
        subtasks.append({
            "id": i,
            "action": v["action"],
            "hand": seg["hand"],
            "object": v["object"],
            "subtask": v["subtask"],
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "start_time": seg["start_frame"] / fps,
            "end_time": seg["end_frame"] / fps,
            "duration": (seg["end_frame"] - seg["start_frame"]) / fps,
            "boundary_source_start": seg["start_source"],
            "boundary_source_end": seg["end_source"],
            "style_id": seg["style_id"],
            "confidence": v["confidence"],
            "flags": flags,
        })

    # hard constraints — fail loudly, the schema re-checks them all
    assert subtasks[0]["start_frame"] == 0, "coverage must start at frame 0"
    assert subtasks[-1]["end_frame"] == n_frames - 1, "must cover final frame"
    for x, y in zip(subtasks, subtasks[1:]):
        assert y["start_frame"] == x["end_frame"], "segments must be contiguous"
    for s in subtasks:
        length = s["end_frame"] - s["start_frame"]
        assert length >= min_len or n_frames - 1 < min_len, (
            f"segment {s['id']} shorter than min_segment_frames after merge")
    return subtasks, review
