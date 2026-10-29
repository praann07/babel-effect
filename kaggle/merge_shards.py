"""Join the per-GPU shards of a Kaggle run into one ordinary run directory that the analysis scripts read.

Edit runs (knowledge_edit_v2.py --shard i/n): analysis/run_<date>_<tag>s<i>_<model>/records.jsonl  ->  analysis/run_<date>_<tag>_<model>/
Interchange runs (interchange.py --shard i/n): analysis/run_<date>_<tag>s<i>_<model>/{effects.npz,meta.json} -> analysis/run_<date>_<tag>_<model>/
Shard directories are kept untouched; only the merged directory is (re)written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def shard_dirs(analysis: Path, tag: str, model: str, i: int) -> list[Path]:
    return sorted(analysis.glob(f"run_*_{tag}s{i}_{model}"))


def merge_edit(analysis: Path, tag: str, model: str, n_shards: int) -> Path:
    rows, seen, cfgs, dirs = [], set(), [], []
    for i in range(n_shards):
        found = shard_dirs(analysis, tag, model, i)
        if not found:
            raise FileNotFoundError(f"no directory for shard {i} of {tag}")
        for d in found:  # a restart after midnight can leave two directories for one shard: take both, drop duplicate rows
            dirs.append(d.name)
            if (d / "config.json").exists():
                cfgs.append(json.loads((d / "config.json").read_text(encoding="utf8")))
            f = d / "records.jsonl"
            for line in (f.read_text(encoding="utf8").splitlines() if f.exists() else []):
                r = json.loads(line)
                key = (r["fact"], r["layer"], r.get("key_mode", "en"))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    first = shard_dirs(analysis, tag, model, 0)[0].name
    out = analysis / first.replace(f"{tag}s0_", f"{tag}_")
    out.mkdir(exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    cfg = dict(cfgs[0]) if cfgs else {}
    cfg["edit_ids"] = sorted({i for c in cfgs for i in c.get("edit_ids", [])})
    cfg["merged_from"] = dirs
    cfg["shard"] = f"merged {n_shards}/{n_shards}"
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf8")
    return out


def merge_interchange(analysis: Path, tag: str, model: str, n_shards: int) -> Path:
    deltas, flips, pairs, metas, dirs = [], [], [], [], []
    for i in range(n_shards):
        done = [d for d in shard_dirs(analysis, tag, model, i) if (d / "effects.npz").exists() and (d / "meta.json").exists()]
        if not done:
            raise FileNotFoundError(f"shard {i} of {tag} has no finished effects.npz")
        d = done[-1]
        with np.load(d / "effects.npz") as z:
            deltas.append(z["delta"]); flips.append(z["flip"])
        m = json.loads((d / "meta.json").read_text(encoding="utf8"))
        pairs += m["pairs"]; metas.append(m); dirs.append(d.name)
    first = [d for d in shard_dirs(analysis, tag, model, 0) if (d / "effects.npz").exists()][-1].name
    out = analysis / first.replace(f"{tag}s0_", f"{tag}_")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "effects.npz", delta=np.concatenate(deltas, axis=0), flip=np.concatenate(flips, axis=0))
    meta = dict(metas[0])
    meta.update({"pairs": pairs, "shard": f"merged {n_shards}/{n_shards}", "merged_from": dirs})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    return out


if __name__ == "__main__":
    kind, tag, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    print((merge_edit if kind == "edit" else merge_interchange)(Path("analysis"), tag, model, n))
