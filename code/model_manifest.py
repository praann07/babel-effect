"""Per-tensor fingerprint of a model folder, so a copy obtained elsewhere (for example a Kaggle Model) can be proven identical to the laptop copy.

Standard library only (it runs before any package is installed). Tensors are hashed from the raw bytes inside the .safetensors files, so the
check does not care how the copy is split into shards.

  python code/model_manifest.py build  FOLDER MANIFEST.json   # write the fingerprint of FOLDER
  python code/model_manifest.py verify FOLDER MANIFEST.json   # exit code 1 and a list of problems if FOLDER differs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

CHUNK = 64 * 2**20
SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def sha256_range(fh, start: int, stop: int) -> str:
    h = hashlib.sha256()
    fh.seek(start)
    left = stop - start
    while left > 0:
        block = fh.read(min(CHUNK, left))
        if not block:
            raise EOFError("file is shorter than its header says")
        h.update(block)
        left -= len(block)
    return h.hexdigest()


def tensor_fingerprints(path: Path) -> dict[str, dict]:
    """name -> dtype, shape, byte count and sha256 of the tensor's bytes, read from one .safetensors file."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            out[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "bytes": b - a, "sha256": sha256_range(fh, base + a, base + b)}
    return out


def file_fingerprint(path: Path) -> dict:
    with open(path, "rb") as fh:
        return {"bytes": path.stat().st_size, "sha256": sha256_range(fh, 0, path.stat().st_size)}


def fingerprint(folder: Path) -> dict:
    tensors: dict[str, dict] = {}
    for p in sorted(folder.glob("*.safetensors")):
        for name, fp in tensor_fingerprints(p).items():
            if name in tensors:
                raise ValueError(f"tensor {name} appears in two shards")
            tensors[name] = fp
    files = {n: file_fingerprint(folder / n) for n in SMALL_FILES if (folder / n).exists()}
    return {"tensors": tensors, "files": files}


def compare(have: dict, want: dict) -> list[str]:
    """Differences between a folder's fingerprint and the reference one (empty list = identical)."""
    problems = []
    for name in sorted(want["tensors"].keys() - have["tensors"].keys()):
        problems.append(f"missing tensor {name}")
    for name in sorted(have["tensors"].keys() - want["tensors"].keys()):
        problems.append(f"extra tensor {name}")
    for name in sorted(want["tensors"].keys() & have["tensors"].keys()):
        a, b = have["tensors"][name], want["tensors"][name]
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            problems.append(f"tensor {name}: {a['dtype']} {a['shape']} instead of {b['dtype']} {b['shape']}")
        elif a["sha256"] != b["sha256"]:
            problems.append(f"tensor {name}: bytes differ")
    for name, fp in want["files"].items():
        got = have["files"].get(name)
        if got is None:
            problems.append(f"missing file {name}")
        elif got["sha256"] != fp["sha256"]:
            problems.append(f"file {name} differs")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("manifest", type=Path)
    a = ap.parse_args()
    fp = fingerprint(a.folder)
    if a.mode == "build":
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps(fp, indent=1, sort_keys=True), encoding="utf8")
        total = sum(t["bytes"] for t in fp["tensors"].values())
        print(f"wrote {a.manifest}: {len(fp['tensors'])} tensors ({total / 1e9:.3f} GB), {len(fp['files'])} small files")
        return 0
    problems = compare(fp, json.loads(a.manifest.read_text(encoding="utf8")))
    if problems:
        print(f"NOT IDENTICAL to {a.manifest.name}: {len(problems)} problems")
        for p in problems[:40]:
            print("  " + p)
        return 1
    print(f"IDENTICAL to {a.manifest.name}: {len(fp['tensors'])} tensors and {len(fp['files'])} small files match byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
