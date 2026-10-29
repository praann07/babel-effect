"""Build the Kaggle dataset (kaggle/babel_bundle.zip) and one notebook per step (kaggle/notebooks/).

Run from the repo root:  .venv/Scripts/python kaggle/build_bundle.py
The dataset holds only inputs: code, tests, the Kaggle runner, the seed candidate lists and wikitext. No laptop results and no
laptop-derived fact files, so every number in the rerun is produced on Kaggle (a missing earlier step fails loudly instead of
silently reusing a laptop file).
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
sys.path.insert(0, str(OUT))
from steps import STEPS  # noqa: E402

TREES = ["code", "tests", "kaggle", "data/wikitext"]
FILES = {"data/v2/facts_candidates.json": "data/v2/facts_candidates_seed.json",  # subjects/objects for template tuning only
         "data/v2/facts_candidates_extra.json": "data/v2/facts_candidates_extra.json",
         "data/v2/facts_candidates_a23.json": "data/v2/facts_candidates_a23.json"}
SKIP_NAMES = {"__pycache__", "babel_bundle.zip", ".ipynb_checkpoints", "notebooks"}
SKIP_SUFFIX = {".pyc", ".log"}


def build_zip() -> Path:
    zpath = OUT / "babel_bundle.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in TREES:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_dir() or SKIP_NAMES & set(p.relative_to(ROOT).parts) or p.suffix in SKIP_SUFFIX:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for src, dst in FILES.items():
            z.write(ROOT / src, dst)
            n += 1
        z.writestr("analysis/_logs/.keep", "")
    print(f"{zpath}  {zpath.stat().st_size / 1e6:.1f} MB  {n} files")
    return zpath


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


RUN = """
import glob, os, subprocess, sys
STEP = "{step}"
HOURS = 10.5  # jobs stop cleanly after this, then everything is packaged
drv = sorted(glob.glob("/kaggle/input/**/kaggle/driver.py", recursive=True))
if not drv:
    print("The babel-bundle dataset is not attached to this notebook.")
else:
    env = dict(os.environ, BABEL_STEP=STEP, HOURS=str(HOURS), PYTHONUNBUFFERED="1")
    try:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("HF_TOKEN secret not readable here (" + type(e).__name__ + "); the driver will try again itself.")
    p = subprocess.Popen([sys.executable, drv[0]], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    print("driver exit code:", p.returncode, "(0 = everything finished)")
"""

SUMMARY = """
import json, os
f = "/kaggle/working/results/RUN_SUMMARY.json"
print(json.dumps(json.load(open(f)), indent=1) if os.path.exists(f) else "no summary (see the messages above)")
"""


def build_notebooks() -> list[Path]:
    (OUT / "notebooks").mkdir(exist_ok=True)
    paths = []
    order = list(STEPS)
    for i, (sid, plan) in enumerate(STEPS.items()):
        prev = order[i - 1] if i else None
        inputs = "the babel-bundle dataset" + (f" and the output of the `{prev}` notebook" if prev else " only")
        cells = [md(f"# {plan['title']}\nInputs: {inputs}. Output: `babel_new_results.zip` (also `results/`), which the next step takes as input."),
                 code(RUN.replace("{step}", sid)), code(SUMMARY)]
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}}
        p = OUT / "notebooks" / f"{sid}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf8")
        paths.append(p)
        print(p)
    return paths


if __name__ == "__main__":
    build_zip()
    build_notebooks()
