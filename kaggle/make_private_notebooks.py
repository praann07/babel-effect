"""Write ready-to-run copies of the Kaggle notebooks with the laptop's Hugging Face token already inside (kaggle/private/, git-ignored).

The token is read from ~/.cache/huggingface/token, checked against the gated model repos (config.json download only), and written only into
kaggle/private/. It is never printed and never committed. Keep these notebooks private on Kaggle.
Run from the repo root after build_bundle.py:  .venv/Scripts/python kaggle/make_private_notebooks.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "private"
GATED = {"babel_llama": ["meta-llama/Llama-3.2-3B-Instruct"], "babel_gemma2": ["google/gemma-2-2b"],
         "babel_llama_base": ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]}


def read_token() -> str:
    p = Path.home() / ".cache" / "huggingface" / "token"
    tok = p.read_text(encoding="utf8").strip()
    if not tok.startswith("hf_"):
        sys.exit("the token file does not look like a Hugging Face token")
    return tok


def check_access(tok: str, repos: list[str]) -> list[str]:
    from huggingface_hub import hf_hub_download
    bad = []
    for r in repos:
        try:
            with tempfile.TemporaryDirectory() as d:
                hf_hub_download(r, "config.json", token=tok, local_dir=d)
        except Exception as e:
            bad.append(f"{r}: {type(e).__name__}")
    return bad


def main() -> None:
    tok = read_token()
    OUT.mkdir(exist_ok=True)
    made = []
    for nb_path in sorted((ROOT / "kaggle" / "notebooks").glob("*.ipynb")):
        bad = check_access(tok, GATED.get(nb_path.stem, []))
        if bad:
            print(f"{nb_path.stem}: NO ACCESS to {bad}; not written")
            continue
        nb = json.loads(nb_path.read_text(encoding="utf8"))
        done = False
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "code" and 'HF_TOKEN_HERE = ""' in src:
                cell["source"] = src.replace('HF_TOKEN_HERE = ""', "HF_TOKEN_HERE = " + json.dumps(tok), 1).splitlines(True)
                done = True
        if not done:
            sys.exit(f"{nb_path.name}: no HF_TOKEN_HERE line found")
        dst = OUT / nb_path.name
        dst.write_text(json.dumps(nb, indent=1), encoding="utf8")
        made.append(dst)
        print(f"{nb_path.stem}: token valid for its gated models; written {dst}")


if __name__ == "__main__":
    main()
