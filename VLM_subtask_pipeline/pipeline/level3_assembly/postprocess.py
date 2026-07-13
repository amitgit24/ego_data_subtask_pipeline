"""Level 3 — merge, min-length enforcement, idle labeling, and review flags.

Same logic as VLM_seg_subtask_pipeline's postprocess (merge same-label
neighbors; absorb sub-min-length segments; insert idle for uncovered gaps;
flag short_segment / low_confidence), reused unchanged except that
min_segment_frames is read from this pipeline's `generate` config block. The
single-pass generator can still emit two boundaries very close together, so the
min-length enforcement matters here exactly as it does in the sibling.
"""

import zlib

CONF_RANK = {"high": 2, "medium": 1, "low": 0}

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
    return (a["vlm"]["action"] == b["vlm"]["action"]
           and norm_obj(a["vlm"]["object"]) == norm_obj(b["vlm"]["object"])
           and a["vlm"]["hand"] == b["vlm"]["hand"]
           and (b["start_frame"] - a["end_frame"]) < merge_gap_s * fps)


def merge_segments(segments, fps, merge_gap_s):
    merged = []
    for seg in segments:
        if merged and mergeable(merged[-1], seg, fps, merge_gap_s):
            a = merged[-1]
            keep = a if (CONF_RANK[a["vlm"]["confidence"]]
                        >= CONF_RANK[seg["vlm"]["confidence"]]) else seg
            a["vlm"] = keep["vlm"]
            a["style_id"] = keep["style_id"]
            a["end_frame"] = seg["end_frame"]
            a["end_source"] = seg["end_source"]
            a["transition_after"] = seg["transition_after"]
        else:
            merged.append(dict(seg))
    return merged


def enforce_min_length(segments, min_len):
    """Absorb every sub-min_len segment into its higher-confidence neighbor
    (tie -> longer) until all survivors meet the floor or one remains. The
    single-pass generator has no minimum-gap rule, so close boundaries can
    produce a tiny segment; this keeps coverage total and contiguous."""
    segs = [dict(s) for s in segments]
    while len(segs) > 1:
        lengths = [s["end_frame"] - s["start_frame"] for s in segs]
        shortest = min(range(len(segs)), key=lambda i: lengths[i])
        if lengths[shortest] >= min_len:
            break
        cands = [j for j in (shortest - 1, shortest + 1) if 0 <= j < len(segs)]
        j = max(cands, key=lambda k: (CONF_RANK[segs[k]["vlm"]["confidence"]],
                                      segs[k]["end_frame"] - segs[k]["start_frame"]))
        left, right = sorted((shortest, j))
        a, b = segs[left], segs[right]
        winner = a if (CONF_RANK[a["vlm"]["confidence"]],
                       a["end_frame"] - a["start_frame"]) >= \
                      (CONF_RANK[b["vlm"]["confidence"]],
                       b["end_frame"] - b["start_frame"]) else b
        combined = dict(winner)
        combined.update({
            "start_frame": a["start_frame"], "end_frame": b["end_frame"],
            "start_source": a["start_source"], "end_source": b["end_source"],
            "transition_before": a["transition_before"],
            "transition_after": b["transition_after"],
        })
        segs[left:right + 1] = [combined]
    return segs


def insert_idle(segments, episode_id, n_frames, fps, idle_gap_s):
    out = []
    cursor = 0
    for seg in segments:
        gap = seg["start_frame"] - cursor
        if gap > idle_gap_s * fps:
            out.append(_idle_segment(episode_id, cursor, seg["start_frame"], len(out)))
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
        "start_source": "idle_gap", "end_source": "idle_gap",
        "transition_before": None, "transition_after": None,
        "style_id": -1,
        "vlm": {"action": "idle", "hand": "both", "object": "none",
               "subtask": sentence, "confidence": "high"},
    }


def cross_check_flags(seg, median_duration, short_frac):
    flags = []
    duration = seg["end_frame"] - seg["start_frame"]
    if median_duration > 0 and duration < short_frac * median_duration:
        flags.append("short_segment")
    if seg["vlm"]["confidence"] == "low":
        flags.append("low_confidence")
    return flags


def postprocess(label_record, cfg):
    a = cfg["assembly"]
    fps = label_record["fps"]
    n_frames = label_record["n_frames"]
    episode_id = label_record["episode_id"]
    min_len = cfg["generate"]["min_segment_frames"]

    segments = sorted(label_record["segments"], key=lambda s: s["start_frame"])
    segments = merge_segments(segments, fps, a["merge_gap_s"])
    segments = enforce_min_length(segments, min_len)
    segments = insert_idle(segments, episode_id, n_frames, fps, a["idle_gap_s"])

    durations = [s["end_frame"] - s["start_frame"] for s in segments]
    durations.sort()
    median_duration = durations[len(durations) // 2] if durations else 0

    subtasks, review = [], []
    for i, seg in enumerate(segments):
        flags = cross_check_flags(seg, median_duration, a["short_segment_frac"])
        if flags:
            review.append(i)
        v = seg["vlm"]
        subtasks.append({
            "id": i, "action": v["action"], "hand": v["hand"], "object": v["object"],
            "subtask": v["subtask"],
            "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
            "start_time": seg["start_frame"] / fps, "end_time": seg["end_frame"] / fps,
            "duration": (seg["end_frame"] - seg["start_frame"]) / fps,
            "boundary_source_start": seg["start_source"],
            "boundary_source_end": seg["end_source"],
            "style_id": seg["style_id"], "confidence": v["confidence"],
            "flags": flags,
        })

    assert subtasks[0]["start_frame"] == 0, "coverage must start at frame 0"
    assert subtasks[-1]["end_frame"] == n_frames - 1, "must cover final frame"
    for x, y in zip(subtasks, subtasks[1:]):
        assert y["start_frame"] == x["end_frame"], "segments must be contiguous"
    for s in subtasks:
        length = s["end_frame"] - s["start_frame"]
        assert length >= min_len or n_frames - 1 < min_len, (
            f"segment {s['id']} shorter than min_segment_frames after merge")
    return subtasks, review
