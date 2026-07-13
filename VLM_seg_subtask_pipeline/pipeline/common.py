"""Shared utilities for the Qwen-direct pipeline.

Episode discovery, sampling, and language-annotation resolution are
IDENTICAL to Kinematics_pipeline (same dataset, same HDF5 facts) and are
reused from there rather than duplicated — a deliberate, read-only
dependency on a sibling pipeline so both draw from one source of truth. The
closed action taxonomy, hand vocabulary, and sentence style templates are
reused the same way, so both pipelines' outputs are directly comparable.

The two sibling files are loaded BY EXPLICIT PATH via importlib under unique
module names, NOT via sys.path. This is deliberate: Kinematics_pipeline's
module is ALSO named `common` and its level2_vlm package also has a
`keyframes` module, so putting those directories on sys.path would shadow
THIS pipeline's own `common`/`keyframes` (a self-import of the former is an
outright circular-import crash). Loading the files by path sidesteps the
name collision entirely.

NOT reused: Kinematics_pipeline's `load_config`, because it defaults to ITS
OWN config.yaml path — reusing that function would silently load the wrong
file if called with no argument. This module defines its own `load_config`
bound to this pipeline's config.yaml instead.
"""

import importlib.util
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

_KINEMATICS_PIPELINE = HERE.parents[1] / "Kinematics_pipeline" / "pipeline"


def _load_sibling(module_name, file_path):
    """Load a sibling-pipeline .py file under an explicit, non-colliding
    module name — never via sys.path (see module docstring)."""
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
STYLES = _kin_taxonomy.STYLES
STYLE_RULES = _kin_taxonomy.STYLE_RULES
style_for = _kin_taxonomy.style_for


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)
