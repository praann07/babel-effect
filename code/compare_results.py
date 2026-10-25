"""Compare two results.json files leaf by leaf: python code/compare_results.py SAVED NEW [--tol 1e-9]. Exit 1 if any numeric leaf differs by more than tol or any other leaf differs."""
from __future__ import annotations

import json
import sys


def walk(a, b, path, bad, stats):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                bad.append((path + [k], "missing key")); continue
            walk(a[k], b[k], path + [k], bad, stats)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i], bad, stats)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b); stats["n"] += 1; stats["max"] = max(stats["max"], d)
        if d > stats["tol"]:
            bad.append((path, f"{a} vs {b}"))
    else:
        stats["n_other"] += 1
        if a != b:
            bad.append((path, f"{str(a)[:40]} vs {str(b)[:40]}"))


def main() -> None:
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-9
    a, b = (json.load(open(p, encoding="utf8")) for p in sys.argv[1:3])
    bad, stats = [], {"n": 0, "n_other": 0, "max": 0.0, "tol": tol}
    walk(a, b, [], bad, stats)
    skip = [x for x in bad if not any(str(p) in ("source_run", "run_dir", "out", "created") for p in x[0])]
    print(f"{sys.argv[2]}: {stats['n']} numeric leaves, max abs diff {stats['max']:.3g}, {len(skip)} differing leaves")
    for p, m in skip[:5]:
        print("  ", "/".join(map(str, p)), m)
    sys.exit(1 if skip else 0)


if __name__ == "__main__":
    main()
