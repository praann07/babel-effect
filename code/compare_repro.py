"""Compare a reproduction run against the saved results. Pass criteria (fixed before the reproduction was run):
P1 n_kept identical; P3 n_valid within 2 and EN-ES MLP-cosine z within 0.5; P4 traced-layer EN success >= 80%.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def newest(pattern: str) -> Path:
    return Path(sorted(glob.glob(pattern))[-1])


def main() -> None:
    tag = sys.argv[1]
    p1 = json.loads((newest(f"analysis/run_*_p1filter_{tag}") / "accuracy.json").read_text())
    p3 = json.loads((newest(f"analysis/run_*_p3overlap_{tag}") / "results.json").read_text())
    p4 = json.loads((newest(f"analysis/run_*_p4edit_{tag}") / "tables.json").read_text())
    man = {c["id"]: c["value"] for c in json.loads(Path("results/manifest.json").read_text())["claims"]}
    rows = [("P1 n_kept", man["n_kept_p1"], p1["n_kept"], p1["n_kept"] == man["n_kept_p1"]),
            ("P3 n_valid", man["p3_n_valid"], p3["n_valid"], abs(p3["n_valid"] - man["p3_n_valid"]) <= 2)]
    z = p3["sets"]["primary_valid"]["pairs"]["en-es"]["mlp_cosine"]["z"]
    rows.append(("P3 EN-ES mlp cosine z", man["p3_en-es_mlp_cosine_z"], z, abs(z - man["p3_en-es_mlp_cosine_z"]) <= 0.5))
    rows.append(("P4 traced EN success", man["p4_traced_en_success_rate"], p4["traced"]["en_success_rate"],
                 p4["traced"]["en_success_rate"] >= 0.8))
    for r in rows:
        print(f"{r[0]:28s} saved={r[1]!s:>10} repro={r[2]!s:>10} {'PASS' if r[3] else 'FAIL'}")
    sys.exit(0 if all(r[3] for r in rows) else 1)


if __name__ == "__main__":
    main()
