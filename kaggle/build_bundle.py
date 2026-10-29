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

import layout  # noqa: E402


def build_zip() -> Path:
    """Optional: the notebooks fetch the code from GitHub themselves, so this dataset is only a fallback when GitHub is unreachable."""
    zpath = OUT / "babel_bundle.zip"
    members = layout.members(ROOT)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in members:
            z.write(p, rel)
        z.writestr("analysis/_logs/.keep", "")
    print(f"{zpath}  {zpath.stat().st_size / 1e6:.1f} MB  {len(members)} files")
    return zpath


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


RUN = """
import io, os, shutil, subprocess, sys, time, urllib.request, zipfile
STEP = "{step}"
HOURS = 10.5  # jobs stop cleanly after this, then everything is packaged
REPO, REF = "praann07/babel-effect", "main"  # REF may be a commit id to pin the code
SRC = os.environ.get("BABEL_SRC_DIR", "/tmp/babel_src")
INPUT = os.environ.get("BABEL_INPUT", "/kaggle/input")


def from_github():
    url = os.environ.get("BABEL_ZIP_URL") or "https://codeload.github.com/" + REPO + "/zip/" + REF
    err = ""
    for i in range(4):
        try:
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=90).read()))
            for d in (SRC, SRC + "_x"):
                shutil.rmtree(d, ignore_errors=True)
            z.extractall(SRC + "_x")
            shutil.move(os.path.join(SRC + "_x", os.listdir(SRC + "_x")[0]), SRC)
            return z.comment.decode() or REF  # GitHub stores the commit id in the zip comment
        except Exception as e:
            err = type(e).__name__ + ": " + str(e)[:150]
            time.sleep(5 * (i + 1))
    print("GitHub download failed (" + err + "); looking for the babel-bundle dataset instead.")
    return None


def from_dataset():
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if {"driver.py", "steps.py", "dtype_check.py"} <= set(files):
            return os.path.join(root, "driver.py")
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if "babel_bundle.zip" in files:
            zipfile.ZipFile(os.path.join(root, "babel_bundle.zip")).extractall("/tmp/babel_zip")
            return "/tmp/babel_zip/kaggle/driver.py"
    return None


env = dict(os.environ, BABEL_STEP=STEP, HOURS=str(HOURS), PYTHONUNBUFFERED="1")
sha = from_github()
if sha:
    drv = os.path.join(SRC, "kaggle", "driver.py")
    env.update(BABEL_SRC=SRC, BABEL_SRC_SHA=sha)
    print("code from GitHub " + REPO + " @ " + sha)
else:
    drv = from_dataset()
if not drv:
    print("No code found: GitHub was unreachable (is Internet on for this notebook?) and no babel-bundle dataset is attached.")
else:
    try:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("HF_TOKEN secret not readable here (" + type(e).__name__ + ": " + str(e)[:150] + "); the driver will try again itself.")
    p = subprocess.Popen([sys.executable, drv], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
        inputs = "nothing attached (the code is fetched from GitHub)" if not prev else f"the output of the `{prev}` notebook (attach it as input)"
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
