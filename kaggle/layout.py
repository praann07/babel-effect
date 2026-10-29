"""Which repo files make up the Kaggle inputs: code, tests, the runner and the seed candidate lists. No laptop results and no
laptop-derived fact files, so every number in the rerun is produced on Kaggle. Shared by build_bundle.py (optional dataset zip) and
driver.py (code fetched from GitHub when the notebook starts). Standard library only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

TREES = ["code", "tests", "kaggle"]
FILES = {"data/v2/facts_candidates.json": "data/v2/facts_candidates_seed.json",  # subjects/objects for template tuning only
         "data/v2/facts_candidates_extra.json": "data/v2/facts_candidates_extra.json",
         "data/v2/facts_candidates_a23.json": "data/v2/facts_candidates_a23.json"}
SKIP_NAMES = {"__pycache__", "babel_bundle.zip", ".ipynb_checkpoints", "notebooks"}
SKIP_SUFFIX = {".pyc", ".log"}


def members(root: Path) -> list[tuple[Path, str]]:
    out = []
    for d in TREES:
        for p in sorted((root / d).rglob("*")):
            if p.is_dir() or SKIP_NAMES & set(p.relative_to(root).parts) or p.suffix in SKIP_SUFFIX:
                continue
            out.append((p, p.relative_to(root).as_posix()))
    out += [(root / src, dst) for src, dst in FILES.items()]
    return out


def assemble(src: Path, dst: Path) -> int:
    n = 0
    for p, rel in members(src):
        t = dst / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
        n += 1
    (dst / "analysis" / "_logs").mkdir(parents=True, exist_ok=True)
    return n
