"""Level 2 — closed action taxonomy, hand vocabulary, and style templates.

The action list is enforced at decode time via vLLM guided JSON, so the model
physically cannot emit an out-of-vocabulary verb.
"""

import zlib

ACTION_GROUPS = {
    "approach_retract": ["reach", "move", "retract", "hover"],
    "acquire_release": ["grasp", "pick", "lift", "place", "put_down",
                        "release", "drop"],
    "transport_exchange": ["transfer", "handover", "carry", "reposition"],
    "object_state": ["open", "close", "fold", "unfold", "insert", "remove",
                     "rotate", "flip", "pour", "press", "push", "pull",
                     "slide", "stack", "unstack", "twist", "tie", "untie",
                     "tear", "cut", "wipe", "shake", "squeeze", "write",
                     "wrap", "unwrap"],
    "bimanual": ["hold_steady", "align", "assemble", "disassemble"],
    "neutral": ["hold", "idle", "adjust_grip"],
    "escape": ["other"],
}

ACTIONS = [a for group in ACTION_GROUPS.values() for a in group]

HANDS = ["left", "right", "both", "both_coordinating"]

CONFIDENCES = ["high", "medium", "low"]

# JSON schema enforced by vLLM guided decoding (xgrammar)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "object": {"type": "string", "maxLength": 60},
        "subtask": {"type": "string", "maxLength": 200},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "hand_disagreement": {"type": "boolean"},
    },
    "required": ["action", "object", "subtask", "confidence",
                 "hand_disagreement"],
    "additionalProperties": False,
}

# ---- subtask description style templates ----------------------------------
# Deterministic per segment: style_id = (crc32(episode_id) + seg_idx) % n.
# (zlib.crc32, NOT Python hash() — hash() is salted per process and would
# break rerun reproducibility.)

# All styles are deliberately concise and action-first (the sentences are
# training targets for VLA subtask conditioning): no leading subordinate
# clauses, no invented goals — diversity comes from word order only.
STYLES = [
    ("imperative_plain",
     "Write a plain imperative command: verb first.",
     "Pick up the white cup with the left hand."),
    ("hand_first",
     "Start with the hand, then verb, object, and short spatial outcome.",
     "Right hand repositions the white cross piece to the right of the base."),
    ("verb_object_hand",
     "Imperative: verb, object, then the hand at the end.",
     "Rotate the white cross piece on the base using the right hand."),
    ("agent_hand",
     "Third person, starting with 'The left/right hand' as the subject.",
     "The right hand rotates the white cross piece on the base."),
    ("object_first",
     "Passive, object as the subject, hand at the end.",
     "The white cross piece is lifted off the base with the right hand."),
    ("compact_telegraphic",
     "Compact telegraphic: hand, verb, object, short outcome; drop articles "
     "where natural.",
     "Left hand grasps white cup and lifts it off the table."),
]

STYLE_RULES = (
    "Exactly one short sentence (aim for 8-14 words), present tense. Never "
    "start with a subordinate clause ('To ...', 'Reaching ...', 'Holding "
    "...'). Mention the acting end-effector when it is left, right, or both "
    "acting together. Name the object with one visible distinguishing "
    "attribute when possible. Only phrasing follows the style; the facts "
    "stay exact."
)

# The sentences are VLA training targets and the policy may drive a robot
# arm rather than a human hand — vary the end-effector noun so the model
# doesn't overfit to "hand". Chosen once per EPISODE (all of an episode's
# subtasks use the same term, so the episode reads as one consistent
# instruction sequence), ~40% of episodes "arm" / ~60% "hand".
ARM_EPISODE_FRACTION = 0.40


def style_for(episode_id: str, segment_index: int):
    """Stable, rerun-reproducible style pick."""
    sid = (zlib.crc32(episode_id.encode()) + segment_index) % len(STYLES)
    return sid, STYLES[sid]


def embodiment_for(episode_id: str):
    """Stable per-episode end-effector term: 'arm' for ~40% of episodes,
    'hand' for the rest. crc32 is uniform over episode ids, so the
    threshold gives the target fraction without any RNG state."""
    frac = (zlib.crc32(f"{episode_id}|embodiment".encode()) % 10000) / 10000
    return "arm" if frac < ARM_EPISODE_FRACTION else "hand"
