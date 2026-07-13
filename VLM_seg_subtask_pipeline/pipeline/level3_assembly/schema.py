"""Level 3 — output schema. All timing is computed in code from frame
indices and the probed FPS; the model validator re-derives and enforces it,
so a file that validates is arithmetically exact by construction.

Differs from Kinematics_pipeline's schema.py in two Literal enums:
  - boundary_source: vlm_transition / episode_start / episode_end (no
    pause/grasp/release/gaze — there is no kinematic detector here).
  - flags: low_confidence (VLM self-report) and short_segment (computed in
    code from the episode's segment-duration distribution) replace
    kinematic_mismatch and hand_disagreement, which require an independent
    kinematic signal that does not exist in this pipeline.
"""

import sys
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ACTIONS, HANDS  # noqa: E402

TIME_TOL = 1e-6

BoundarySource = Literal["vlm_transition", "episode_start", "episode_end",
                         "idle_gap"]
Flag = Literal["low_confidence", "short_segment"]


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    action: str
    hand: str
    object: str
    subtask: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    boundary_source_start: BoundarySource
    boundary_source_end: BoundarySource
    style_id: int
    confidence: Literal["high", "medium", "low"]
    flags: List[Flag]

    @field_validator("action")
    @classmethod
    def action_in_taxonomy(cls, v):
        if v not in ACTIONS:
            raise ValueError(f"action {v!r} not in taxonomy")
        return v

    @field_validator("hand")
    @classmethod
    def hand_in_vocab(cls, v):
        if v not in HANDS:
            raise ValueError(f"hand {v!r} not in vocabulary")
        return v

    @field_validator("subtask")
    @classmethod
    def one_sentence(cls, v):
        if not v.strip():
            raise ValueError("empty subtask description")
        return v


class EpisodeAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task: str
    fps: float
    total_frames: int
    pipeline_version: str
    subtasks: List[Subtask]
    review_queue: List[int]

    @model_validator(mode="after")
    def invariants(self):
        subs = self.subtasks
        if not subs:
            raise ValueError("no subtasks")
        ids = [s.id for s in subs]
        if ids != list(range(len(subs))):
            raise ValueError(f"ids not consecutive: {ids}")
        if subs[0].start_frame != 0:
            raise ValueError("first subtask must start at frame 0")
        if subs[-1].end_frame != self.total_frames - 1:
            raise ValueError("last subtask must end at the final frame")
        for a, b in zip(subs, subs[1:]):
            if b.start_frame != a.end_frame:
                raise ValueError(
                    f"subtasks {a.id}->{b.id} not contiguous "
                    f"({a.end_frame} vs {b.start_frame})")
        for s in subs:
            if not (0 <= s.start_frame < s.end_frame <= self.total_frames - 1):
                raise ValueError(f"subtask {s.id} frames out of range")
            for name, got, want in (
                    ("start_time", s.start_time, s.start_frame / self.fps),
                    ("end_time", s.end_time, s.end_frame / self.fps),
                    ("duration", s.duration,
                     (s.end_frame - s.start_frame) / self.fps)):
                if abs(got - want) > TIME_TOL:
                    raise ValueError(
                        f"subtask {s.id} {name} {got} != {want} (from frames)")
        bad = set(self.review_queue) - set(ids)
        if bad:
            raise ValueError(f"review_queue references unknown ids: {bad}")
        return self
