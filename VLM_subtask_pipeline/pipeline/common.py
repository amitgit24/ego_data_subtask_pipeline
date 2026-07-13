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

episode_id = _kin_common.episode_id
episode_slug = _kin_common.episode_slug
list_tasks = _kin_common.list_tasks
resolve_description = _kin_common.resolve_description
sample_episodes = _kin_common.sample_episodes

ACTION_GROUPS = _kin_taxonomy.ACTION_GROUPS
ACTIONS = _kin_taxonomy.ACTIONS
CONFIDENCES = _kin_taxonomy.CONFIDENCES
HANDS = _kin_taxonomy.HANDS
STYLE_RULES = _kin_taxonomy.STYLE_RULES


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)
