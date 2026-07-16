"""Shared utilities for the VLM single-pass subtask pipeline.

Episode discovery, sampling, language-annotation resolution, the closed action
taxonomy, hand vocabulary, and sentence-style rules are reused from
Kinematics_pipeline (the one source of truth), loaded BY EXPLICIT PATH via
importlib under unique module names — never via sys.path, because that sibling
module is also named `common` and would shadow / circular-import this one. This
is the same wiring proven in VLM_seg_subtask_pipeline/pipeline/common.py.

This pipeline defines its OWN load_config bound to this folder's config.yaml.
"""

import importlib.util
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

_KINEMATICS_PIPELINE = HERE.parents[1] / "Kinematics_pipeline" / "pipeline"


def _load_sibling(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_kin_common = _load_sibling("kin_common", _KINEMATICS_PIPELINE / "common.py")
_kin_taxonomy = _load_sibling(
    "kin_taxonomy", _KINEMATICS_PIPELINE / "level2_vlm" / "taxonomy.py")
_kin_keyframes = _load_sibling(
    "kin_keyframes", _KINEMATICS_PIPELINE / "level2_vlm" / "keyframes.py")

episode_id = _kin_common.episode_id
episode_slug = _kin_common.episode_slug
list_tasks = _kin_common.list_tasks
resolve_description = _kin_common.resolve_description
sample_episodes = _kin_common.sample_episodes
select_prompt_hint = _kin_common.select_prompt_hint

ACTION_GROUPS = _kin_taxonomy.ACTION_GROUPS
ACTIONS = _kin_taxonomy.ACTIONS
CONFIDENCES = _kin_taxonomy.CONFIDENCES
HANDS = _kin_taxonomy.HANDS
STYLE_RULES = _kin_taxonomy.STYLE_RULES
embodiment_for = _kin_taxonomy.embodiment_for

# proven keyframe seek-extractor from the Kinematics pipeline's Level 2 VLM
# stage; reused as-is by the vLLM-endpoint backend (see vlm_backend.py) since
# it already does exactly what's needed: seek to specific absolute frame
# indices, downscale, save JPEGs.
extract_keyframes = _kin_keyframes.extract_keyframes

# THIS pipeline's own per-task tuning dir (one file per task, created only
# when a task needs tuning; absent file = untouched defaults). Same design
# proven in Kinematics_pipeline, but a separate directory: prompts here
# drive a single-pass segment+label generator, not a per-segment labeler,
# so the hints are written differently and must not be shared.
TASK_CONFIGS_DIR = HERE / "task_configs"


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_task_config(task_name):
    """pipeline/task_configs/<task_name>.yaml -> dict, or {} if absent."""
    return _kin_common.load_task_config(task_name, dir_path=TASK_CONFIGS_DIR)


def apply_task_overrides(cfg, task_name, task_cfg=None):
    """Merge a task's `generate:` block over the global one (this pipeline's
    equivalent of the sibling's kinematics overrides). Returns cfg unchanged
    (same object) when the task has no file — and, as learned in the
    sibling pipeline, EVERY stage that reads `generate:` values (level 1
    generation AND level 3 assembly's min-length assert) must go through
    this, or L3 will reject segments L1 legitimately produced."""
    task_cfg = load_task_config(task_name) if task_cfg is None else task_cfg
    if "generate" not in task_cfg:
        return cfg
    merged = dict(cfg)
    merged["generate"] = {**cfg["generate"], **task_cfg["generate"]}
    return merged
