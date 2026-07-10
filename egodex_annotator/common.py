"""Shared utilities: config loading, episode discovery/sampling, annotation
resolution. Every module reads HDF5 key paths from config — never hardcoded."""

import random
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def episode_id(h5_path):
    """'task_name/episode_idx' — the canonical episode identifier."""
    return f"{h5_path.parent.name}/{h5_path.stem}"


def episode_slug(h5_path):
    """Filesystem-safe id used for per-episode output files."""
    return f"{h5_path.parent.name}__{h5_path.stem}"


def list_tasks(data_dir):
    return sorted(d for d in Path(data_dir).iterdir() if d.is_dir())


def sample_episodes(data_dir, n, seed=0, min_frames=0):
    """Pick n episodes from n distinct tasks, deterministically.

    When min_frames > 0, prefer episodes at least that long within each
    sampled task (falls back to the longest episode of the task).
    """
    import h5py

    rng = random.Random(seed)
    tasks = list_tasks(data_dir)
    chosen_tasks = rng.sample(tasks, min(n, len(tasks)))
    episodes = []
    for task in chosen_tasks:
        h5s = sorted(task.glob("*.hdf5"), key=lambda p: int(p.stem))
        if min_frames:
            lengths = {}
            for p in h5s:
                with h5py.File(p, "r") as f:
                    lengths[p] = f["transforms/camera"].shape[0]
            long_enough = [p for p in h5s if lengths[p] >= min_frames]
            episodes.append(rng.choice(long_enough) if long_enough
                            else max(h5s, key=lambda p: lengths[p]))
        else:
            episodes.append(rng.choice(h5s))
    return episodes


def resolve_description(attrs, cfg):
    """Resolution order confirmed by Level 0 audit:
    llm_description2 (when which_llm_description=='2') > llm_description >
    description (older annotator versions)."""
    h = cfg["hdf5"]
    which = attrs.get(h["which_description_attr"])
    d2 = attrs.get(h["description2_attr"])
    if str(which) == "2" and d2 is not None and str(d2) != "None":
        return str(d2)
    d1 = attrs.get(h["description_attr"])
    if d1 is not None and str(d1) != "None":
        return str(d1)
    d0 = attrs.get(h["fallback_description_attr"])
    if d0 is not None and str(d0) != "None":
        return str(d0)
    raise KeyError(f"no language annotation found in attrs: {sorted(attrs)}")


def load_task_categories(path=None):
    """task_categories.yaml -> (categories dict, task -> category name map).

    Fails loudly on a task listed in two categories."""
    with open(path or CONFIG_PATH.parent / "task_categories.yaml") as f:
        categories = yaml.safe_load(f)
    task_map = {}
    for name, cat in categories.items():
        for task in cat["tasks"]:
            if task in task_map:
                raise ValueError(f"task {task!r} in both {task_map[task]!r} "
                                 f"and {name!r}")
            task_map[task] = name
    return categories, task_map


def category_for_task(task_name, categories, task_map):
    """Category dict for a task folder name, or None if unmapped (callers
    decide whether unmapped is an error)."""
    name = task_map.get(task_name)
    return dict(categories[name], name=name) if name else None


def joint_validity(f, joint_key, conf_min):
    """Per-frame validity for a transforms/<joint> dataset.

    Uses confidences/<joint> where the group exists (Level 0: absent on some
    episodes); falls back to a NaN check on the transform itself."""
    joint = joint_key.split("/")[-1]
    conf_key = f"confidences/{joint}"
    if conf_key in f:
        conf = f[conf_key][:]
        return ~(np.isnan(conf) | (conf < conf_min))
    return ~np.isnan(f[joint_key][:]).any(axis=(1, 2))
