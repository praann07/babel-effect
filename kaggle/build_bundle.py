"""Build kaggle/babel_bundle.zip (code + data + earlier run outputs, no model weights) and kaggle/babel_kaggle.ipynb.

Run from the repo root:  .venv/Scripts/python kaggle/build_bundle.py
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle"
INCLUDE_DIRS = ["code", "data", "analysis", "tests", "kaggle"]
SKIP_NAMES = {"__pycache__", "babel_bundle.zip"}
SKIP_SUFFIX = {".pyc"}
# Torch and CUDA come from Kaggle's own image; Windows-only packages are useless there.
DROP_REQ = re.compile(r"^(torch|nvidia|triton|colorama|pywin32|pywinpty)\b", re.I)


def kaggle_requirements() -> str:
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf8").splitlines()]
    return "\n".join(ln for ln in lines if ln and not DROP_REQ.match(ln)) + "\n"


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
        z.writestr("requirements-kaggle.txt", kaggle_requirements())
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
Set `STAGE` in the first code cell, then use **Save Version > Save & Run All** (so it keeps running with your laptop shut).
- `STAGE = "first"`: checks the GPU, then runs the paraphrase check (GPU 0) and the new-facts run (GPU 1) at the same time.
- `STAGE = "second"`: the shared-edit test. Needs the output of the first session added as an input.
"""),
    code("""
STAGE = "first"   # "first" or "second"
HOURS = 10.5      # each long run stops cleanly after this many hours so the session can save
"""),
    md("""
## 0. Install packages (must run FIRST, before anything imports numpy or torch)
Only the few packages we need. Kaggle's own numpy, scipy, torch and pandas stay untouched.
If you ever see a numpy error after this cell: Run > Restart session, then run the cells again from the top.
"""),
    code("""
import subprocess, sys
pip = [sys.executable, "-m", "pip", "install", "-q"]
r = subprocess.run(pip + ["transformer-lens==3.9.0", "transformers==5.19.0", "accelerate==1.15.0",
                          "jaxtyping", "einops", "fancy-einsum", "better-abc", "beartype"], capture_output=True, text=True)
print("pip exit:", r.returncode); print(r.stderr[-1500:])
chk = subprocess.run([sys.executable, "-c", "import numpy, scipy, torch, transformers, transformer_lens as t; "
                      "print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| torch', torch.__version__, "
                      "'| transformers', transformers.__version__, '| transformer_lens', t.__version__)"], capture_output=True, text=True)
print(chk.stdout, chk.stderr[-1500:])
"""),
    md("## 1. GPU check"),
    code("""
import subprocess, time, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| gpus", torch.cuda.device_count(), "| bf16 supported:", torch.cuda.is_bf16_supported())
for dt in (torch.bfloat16, torch.float16):
    a = torch.randn(4096, 4096, device="cuda", dtype=dt); torch.cuda.synchronize(); t = time.time()
    for _ in range(20): a @ a
    torch.cuda.synchronize(); print(dt, "20 matmuls:", round(time.time() - t, 2), "s")
"""),
    md("## 2. Put the code and data in place"),
    code("""
import glob, os, shutil, zipfile
WORK = "/kaggle/working/babel-effect"
if not os.path.exists(WORK + "/code/common.py"):
    zips = glob.glob("/kaggle/input/**/babel_bundle.zip", recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
    else:  # Kaggle may have unzipped the upload already
        src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/code/common.py", recursive=True)[0]))
        shutil.copytree(src, WORK, dirs_exist_ok=True)
os.chdir(WORK)
if STAGE == "second":  # bring in run folders made by the first session
    for d in glob.glob("/kaggle/input/**/analysis/run_*", recursive=True):
        dst = os.path.join("analysis", os.path.basename(d))
        if not os.path.exists(dst):
            shutil.copytree(d, dst)
print(sorted(glob.glob("analysis/run_*_p4a21_llama")), os.getcwd())
START_DIRS = set(os.listdir("analysis"))  # to package only what is new
"""),
    md("## 3. Packages and the Llama weights (needs the Hugging Face token saved as a Kaggle secret named HF_TOKEN)"),
    code("""
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
tok = UserSecretsClient().get_secret("HF_TOKEN")
os.makedirs("/tmp/models", exist_ok=True)
snapshot_download("meta-llama/Llama-3.2-3B-Instruct", local_dir="/tmp/models/llama-3.2-3b-instruct",
                  token=tok, ignore_patterns=["original/*"])
if not os.path.exists("models"): os.symlink("/tmp/models", "models")
print(os.listdir("models/llama-3.2-3b-instruct"))
"""),
    md("## 4. Tests and a quick model check (stop here if anything fails)"),
    code("""
!PYTHONPATH=code python -m unittest discover -s tests -v 2>&1 | tail -25
"""),
    code("""
import sys, time; sys.path.insert(0, "code")
from common import load_model
t = time.time(); m = load_model("llama"); print("load s:", round(time.time() - t, 1))
toks = m.to_tokens("The capital of France is")
t = time.time()
for _ in range(10): lg = m(toks)
torch.cuda.synchronize(); print("10 forwards s:", round(time.time() - t, 2))
print(repr(m.to_string(lg[0, -1].argmax())), "| VRAM GB:", round(torch.cuda.max_memory_allocated() / 1e9, 1))
del m; torch.cuda.empty_cache()
"""),
    md("## 5. The runs"),
    code("""
import subprocess, os
env = dict(os.environ, HOURS=str(HOURS), PYTHONUNBUFFERED="1")
procs = []
if STAGE == "first":
    procs.append(subprocess.Popen(["bash", "kaggle/run1_paraphrase_check.sh"], env=dict(env, CUDA_VISIBLE_DEVICES="0")))
    procs.append(subprocess.Popen(["bash", "kaggle/run2_new_facts.sh"], env=dict(env, CUDA_VISIBLE_DEVICES="1" if torch.cuda.device_count() > 1 else "0")))
else:
    procs.append(subprocess.Popen(["bash", "kaggle/run3_shared_edit.sh"], env=dict(env, CUDA_VISIBLE_DEVICES="0")))
for p in procs: p.wait()
print("exit codes:", [p.returncode for p in procs])
"""),
    code("""
!tail -n 15 analysis/_logs/run*.log
"""),
    md("## 6. Package only the new results"),
    code("""
import shutil
new = [d for d in os.listdir("analysis") if d not in START_DIRS or d == "_logs"]
os.makedirs("/kaggle/working/results", exist_ok=True)
for d in new: shutil.copytree(f"analysis/{d}", f"/kaggle/working/results/{d}", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/babel_new_results", "zip", "/kaggle/working/results")
shutil.rmtree("/kaggle/working/babel-effect", ignore_errors=True)  # keep the saved output small
print(new)
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
