"""Builds one notebook per Kaggle step (kaggle/babel_llama.ipynb, kaggle/babel_gemma2.ipynb): two cells, the code and data of this repo packed inside the
notebook (no GitHub, no extra dataset), then the run cell (notebook_cell.py). Run from the repo root:  python kaggle/make_notebook.py [STEP ...]
(no argument = all). Standard library only. The pack is deterministic, so the same files always give the same code id (printed at the start of every
Kaggle run). A notebook carries only its own model's config pack and fingerprint, and the saved laptop runs only if its step reads them."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "babel_llama.ipynb"
DEFAULT_STEP = "babel_llama"  # the STEP written in notebook_cell.py
# step -> (notebook file, model keys packed, saved laptop runs packed). The Spanish/Hindi non-geography chain is only in the Llama step.
NOTEBOOKS = {"babel_llama": ("babel_llama.ipynb", ["llama"], True), "babel_gemma2": ("babel_gemma2.ipynb", ["gemma2"], False)}


def pack(root: Path, models: list[str] | None = None, runs: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, rel in layout.members(root, models, runs):
            zi = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))  # fixed date: same files give the same bytes
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes(), compresslevel=9)
    return buf.getvalue()


def cell(source: str, hidden: bool = False) -> dict:
    meta = {"trusted": True}
    if hidden:
        meta["jupyter"] = {"source_hidden": True}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": source.splitlines(keepends=True)}


def build(root: Path = ROOT, out: Path = OUT, step: str = DEFAULT_STEP, models: list[str] | None = None, runs: bool = True) -> tuple[Path, str, int]:
    blob = pack(root, models, runs)
    digest = hashlib.sha256(blob).hexdigest()
    b64 = base64.b64encode(blob).decode()
    lines = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
    payload = (f"# The code, fact files and saved laptop results of this project, packed into this notebook (no download needed). Nothing to run or edit here.\n"
               f"PAYLOAD_SHA256 = \"{digest}\"\nPAYLOAD = \"\"\"\n{lines}\n\"\"\"\n")
    run = (root / "kaggle" / "notebook_cell.py").read_text(encoding="utf8")
    if step != DEFAULT_STEP:
        marker = f'STEP = "{DEFAULT_STEP}"'
        if marker not in run:
            raise SystemExit(f"notebook_cell.py has no line {marker!r} to set the step in")
        run = run.replace(marker, f'STEP = "{step}"', 1)
    nb = {"cells": [cell(payload, hidden=True), cell(run)],
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out.write_text(json.dumps(nb, indent=1), encoding="utf8", newline="\n")
    return out, digest, len(blob)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(NOTEBOOKS)
    for st in wanted:
        if st not in NOTEBOOKS:
            raise SystemExit(f"unknown step {st!r}; known: {sorted(NOTEBOOKS)}")
        fname, ms, rn = NOTEBOOKS[st]
        path, digest, n = build(ROOT, ROOT / "kaggle" / fname, st, ms, rn)
        print(f"{st}: {path} ({path.stat().st_size / 1e6:.1f} MB), code id bundle-{digest[:12]}, {n / 1e6:.1f} MB packed")
