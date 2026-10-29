"""Build kaggle/babel_bundle.zip (code + data + earlier run outputs, no model weights) and kaggle/babel_kaggle.ipynb.

Run from the repo root:  .venv/Scripts/python kaggle/build_bundle.py
The notebook is deliberately thin: all the work (and all error handling) is in kaggle/driver.py, which runs in its own process.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
INCLUDE_DIRS = ["code", "data", "analysis", "tests", "kaggle"]
SKIP_NAMES = {"__pycache__", "babel_bundle.zip", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".log"}


def build_zip() -> Path:
    zpath = OUT / "babel_bundle.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_dir() or SKIP_NAMES & set(p.parts) or p.suffix in SKIP_SUFFIX:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
    print(f"{zpath}  {zpath.stat().st_size / 1e6:.1f} MB  {n} files")
    return zpath


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("""
# The Babel Effect on Kaggle
One cell does everything (`kaggle/driver.py`, in its own process): install, checks, model download, number-format check,
then the paraphrase check, the new-facts run and the shared-edit test spread over every GPU, then merging, analyses and packaging.
Rerunning with this notebook's earlier output attached resumes where it stopped. Results: `babel_new_results.zip` in the output.
"""),
    code("""
import glob, os, subprocess, sys
HOURS = 10.5  # time budget; jobs stop cleanly after this so the session can save and package
drv = sorted(glob.glob("/kaggle/input/**/kaggle/driver.py", recursive=True))
if not drv:
    print("The babel-bundle dataset (with kaggle/driver.py) is not attached to this notebook.")
else:
    env = dict(os.environ, HOURS=str(HOURS), PYTHONUNBUFFERED="1")
    try:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("HF_TOKEN secret not readable here (" + type(e).__name__ + "); the driver will try again itself.")
    p = subprocess.Popen([sys.executable, drv[0]], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    print("driver exit code:", p.returncode)
"""),
    code("""
import json, os
f = "/kaggle/working/results/RUN_SUMMARY.json"
print(json.dumps(json.load(open(f)), indent=1) if os.path.exists(f) else "no summary yet (see the messages above)")
"""),
]


def build_notebook() -> Path:
    nb = {"cells": CELLS, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    p = OUT / "babel_kaggle.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf8")
    print(p)
    return p


if __name__ == "__main__":
    build_zip()
    build_notebook()
