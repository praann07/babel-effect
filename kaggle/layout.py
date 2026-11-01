"""Which repo files make up the Kaggle inputs: code, tests, the runner, the fact/template data, and the two saved tracing runs the Spanish swap-test
analysis reads. No saved run is an input of the edit jobs (the notebook traces the facts itself). Shared by make_notebook.py (packs these files into
the notebook) and driver.py (assembles them into the working folder when the notebook starts). Standard library only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

TREES = ["code", "tests", "kaggle"]  # replaced by the fetched commit at every start, so a second run never keeps old code
DATA = ["data"]  # copied over, never deleted: jobs may have written new files here (resume)
RUNS = ["analysis/run_20261015_p3trace_v2_llama", "analysis/run_20261019_p3extra_10191057_llama", "analysis/run_20261024_p3seeds_10240109_llama"]  # read by the Spanish swap-test analysis (metric validity check): controls_overlap.SETS and the seeds run
SKIP_NAMES = {"__pycache__", "babel_bundle.zip", ".ipynb_checkpoints", "notebooks", "private", "wikitext", "audit", "repro"}
SKIP_SUFFIX = {".pyc", ".log", ".ipynb"}  # the notebook carries the pack, so it must not pack itself
MODEL_NAMES = {"llama": "llama-3.2-3b-instruct", "gemma2": "gemma-2-2b"}  # model key -> folder name of its config/tokenizer pack and fingerprint


def members(root: Path, models: list[str] | None = None, runs: bool = True) -> list[tuple[Path, str]]:
    """Every input file. models: only these models' config pack and fingerprint are packed (None = all); runs=False leaves out the saved laptop runs."""
    other = {n for k, n in MODEL_NAMES.items() if models is not None and k not in models}
    out = []
    for d in TREES + DATA + (RUNS if runs else []):
        for p in sorted((root / d).rglob("*")):
            if p.is_dir() or SKIP_NAMES & set(p.relative_to(root).parts) or p.suffix in SKIP_SUFFIX:
                continue
            if p.parent.name in ("model_files", "model_manifests") and any(p.name.startswith(n + ".") for n in other):
                continue
            out.append((p, p.relative_to(root).as_posix()))
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
