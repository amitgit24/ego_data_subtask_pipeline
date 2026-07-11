"""Audit task_categories.yaml against the dataset and the taxonomy.

Pass criteria: every task folder in data_dir maps to exactly one category,
no category lists a task that does not exist on disk, and every
expected_verb is in the closed action taxonomy.

    python check_categories.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

from common import list_tasks, load_config, load_task_categories

sys.path.insert(0, str(Path(__file__).resolve().parent / "level2_vlm"))
from taxonomy import ACTIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    categories, task_map = load_task_categories()
    on_disk = {t.name for t in list_tasks(cfg["paths"]["data_dir"])}

    unmapped = sorted(on_disk - set(task_map))
    ghosts = sorted(set(task_map) - on_disk)
    bad_verbs = {name: sorted(set(cat["expected_verbs"]) - set(ACTIONS))
                 for name, cat in categories.items()
                 if set(cat["expected_verbs"]) - set(ACTIONS)}

    print(f"{len(on_disk)} tasks on disk, {len(task_map)} mapped, "
          f"{len(categories)} categories")
    for name, cat in categories.items():
        n = sum(1 for t in cat["tasks"] if t in on_disk)
        print(f"  {name:24s} {n:3d} tasks  "
              f"(direction_aware={cat['direction_aware']}, "
              f"kf_bonus={cat['keyframes_bonus']})")

    checks = [
        (f"every on-disk task mapped ({len(unmapped)} unmapped)", not unmapped),
        (f"no ghost tasks in yaml ({len(ghosts)} not on disk)", not ghosts),
        ("expected_verbs subset of taxonomy", not bad_verbs),
        ("no task in two categories (checked at load)", True),
    ]
    for t in unmapped:
        print(f"    UNMAPPED: {t}")
    for t in ghosts:
        print(f"    GHOST:    {t}")
    for name, verbs in bad_verbs.items():
        print(f"    BAD VERBS in {name}: {verbs}")

    print("\n================ CATEGORY AUDIT ================")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("================================================")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
