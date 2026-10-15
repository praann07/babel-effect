"""Build or verify results/manifest.json: every numeric claim in the paper -> artifact path + key path + value.

Usage: python code/make_manifest.py build | verify
`verify` re-reads each artifact and fails if a stored value differs by more than 1e-9 (these are post-hoc reads of
write-once run outputs, so exact equality is expected).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

P1 = "analysis/run_20261012_p1filter_v2/accuracy.json"
P2 = "analysis/figs_20261012_p2trace_v3/summary.json"
P3 = "analysis/run_20261012_p3overlap/results.json"
P4 = "analysis/run_20261013_p4edit/tables.json"
V2 = {
    "llama": {"p1": "analysis/run_20261014_p1v2b_llama/accuracy.json", "ov": "analysis/run_20261015_p3overlap_v2_llama/results.json",
              "ovs": "analysis/run_20261015_p3overlap_v2b_llama/results.json"},
    "gemma2": {"p1": "analysis/run_20261015_p1v2c_gemma2/accuracy.json", "ov": "analysis/run_20261015_p3overlap_v2c_gemma2/results.json",
               "ovs": None},
}
CMP = "analysis/run_20261015_compare_models/results.json"
MAN = Path("results/manifest.json")


def claims() -> list[dict]:
    c: list[tuple[str, str, list]] = []
    c += [("n_candidates", P1, ["n_candidates"]), ("n_kept_p1", P1, ["n_kept"])]
    for rel in ("capital_country", "capital_of", "landmark_city"):
        c.append((f"p1_kept_{rel}", P1, ["per_relation", rel, "kept"]))
        for lg in ("en", "es"):
            c.append((f"p1_acc_{rel}_{lg}", P1, ["per_relation", rel, lg]))
        c.append((f"p1_acc_{rel}_hi_v0_acc", P1, ["per_relation", rel, f"hi_v{0}"]))
    c += [("p2_n_pairs", P2, ["n_pairs"]), ("p2_n_valid", P2, ["n_valid"])]
    for lg in ("en", "es", "hi"):
        c += [(f"p2_crit_median_{lg}", P2, ["critical_mlp_layer", lg, "median"]),
              (f"p2_crit_frac_8_19_{lg}", P2, ["critical_mlp_layer", lg, "frac_in_8_19"])]
    c += [("p3_n_all", P3, ["n_all"]), ("p3_n_valid", P3, ["n_valid"])]
    for pair in ("en-es", "en-hi", "es-hi"):
        for stat in ("mlp_cosine", "mlp_spearman", "resid_cosine", "iou_top5", "iou_top10", "iou_top20"):
            base = ["sets", "primary_valid", "pairs", pair, stat]
            for k, key in (("obs", ["observed_median"]), ("ci_lo", ["ci95", 0]), ("ci_hi", ["ci95", 1]),
                           ("null_mean", ["null_mean"]), ("z", ["z"]), ("p", ["p_one_sided"])):
                c.append((f"p3_{pair}_{stat}_{k}", P3, base + key))
    for lg in ("en", "es", "hi"):
        c += [(f"p3_crit_median_{lg}", P3, ["critical_mlp_layer", lg, "median"]),
              (f"p3_crit_frac_8_19_{lg}", P3, ["critical_mlp_layer", lg, "frac_in_8_19"])]
    for pair in ("en-es", "en-hi", "es-hi"):
        c.append((f"p3_crit_absdiff_median_{pair}", P3, ["critical_layer_abs_diff", pair, "median"]))
        c.append((f"p3_crit_within2_{pair}", P3, ["critical_layer_abs_diff", pair, "frac_within_2"]))
    for sset in ("all_kept", "valid_no_landmark", "valid_capital_country", "valid_capital_of"):
        for pair in ("en-es", "en-hi", "es-hi"):
            c.append((f"p3_{sset}_{pair}_mlp_cosine_z", P3, ["sets", sset, "pairs", pair, "mlp_cosine", "z"]))
    for cond in ("traced", "mid", "random"):
        c += [(f"p4_{cond}_n", P4, [cond, "n_edits"]), (f"p4_{cond}_en_success_rate", P4, [cond, "en_success_rate"]),
              (f"p4_{cond}_specificity_worst", P4, [cond, "specificity_worst"]),
              (f"p4_{cond}_ppl_worst", P4, [cond, "ppl_ratio_worst"])]
        for lg in ("en", "es", "hi"):
            for k, i in (("med", 0), ("lo", 1), ("hi", 2)):
                c.append((f"p4_{cond}_dP_{lg}_{k}", P4, [cond, f"dP_{lg}", i]))
            c.append((f"p4_{cond}_top1_new_{lg}", P4, [cond, f"top1_new_rate_{lg}"]))
        for lg in ("es", "hi"):
            for k, i in (("med", 0), ("lo", 1), ("hi", 2)):
                c.append((f"p4_{cond}_transfer_{lg}_{k}", P4, [cond, f"transfer_{lg}_en_success", i]))
        for key in ("specificity_max_abs_dP", "ppl_ratio"):
            for k, i in (("med", 0), ("lo", 1), ("hi", 2)):
                c.append((f"p4_{cond}_{key}_{k}", P4, [cond, key, i]))
    for m, d in V2.items():
        c += [(f"v2_{m}_n_candidates", d["p1"], ["n_candidates"]), (f"v2_{m}_n_kept", d["p1"], ["n_kept"]),
              (f"v2_{m}_n_valid", d["ov"], ["n_valid"])]
        for rel in ("capital_of", "capital_country", "landmark_city", "invention_inventor", "event_year"):
            c.append((f"v2_{m}_kept_{rel}", d["p1"], ["per_relation", rel, "kept"]))
        for pair in ("en-es", "en-hi", "es-hi"):
            for stat in ("mlp_cosine", "mlp_spearman"):
                base = ["sets", "primary_valid", "pairs", pair, stat]
                for k, key in (("obs", ["observed_median"]), ("ci_lo", ["ci95", 0]), ("ci_hi", ["ci95", 1]), ("null_mean", ["null_mean"]),
                               ("z", ["z"]), ("p", ["p_one_sided"])):
                    c.append((f"v2_{m}_{pair}_{stat}_{k}", d["ov"], base + key))
            c.append((f"v2_{m}_{pair}_mlp_cosine_z_no_copyable", d["ov"], ["sets", "valid_no_copyable", "pairs", pair, "mlp_cosine", "z"]))
            if d["ovs"]:
                c.append((f"v2_{m}_{pair}_mlp_cosine_z_stratified", d["ovs"], ["sets", "primary_valid_stratified_by_relation", "pairs", pair, "mlp_cosine", "z"]))
        for lg in ("en", "es", "hi"):
            c += [(f"v2_{m}_crit_median_{lg}", d["ov"], ["critical_mlp_layer", lg, "median"]),
                  (f"v2_{m}_crit_frac_8_19_{lg}", d["ov"], ["critical_mlp_layer", lg, "frac_in_8_19"])]
    c.append(("cmp_n_both", CMP, ["n_both"]))
    for m in ("llama", "gemma2"):
        for pair in ("en-es", "en-hi", "es-hi"):
            c.append((f"cmp_{m}_{pair}_mlp_cosine_z", CMP, ["intersection", m, "pairs", pair, "mlp_cosine", "z"]))
    return [{"id": i, "artifact": a, "key_path": k} for i, a, k in c]


def read(art: str, kp: list):
    v = json.loads(Path(art).read_text(encoding="utf8"))
    for k in kp:
        v = v[k]
    return v


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "build":
        cl = claims()
        for c in cl:
            c["value"] = read(c["artifact"], c["key_path"])
        MAN.parent.mkdir(exist_ok=True)
        MAN.write_text(json.dumps({"n_claims": len(cl), "claims": cl}, indent=1))
        print("wrote", MAN, len(cl), "claims")
        return
    man = json.loads(MAN.read_text())
    bad = []
    for c in man["claims"]:
        v = read(c["artifact"], c["key_path"])
        ok = (abs(v - c["value"]) <= 1e-9) if isinstance(v, (int, float)) and isinstance(c["value"], (int, float)) else v == c["value"]
        if not ok:
            bad.append((c["id"], c["value"], v))
    print(f"verified {len(man['claims']) - len(bad)}/{len(man['claims'])} claims")
    if bad:
        print("MISMATCH:", bad[:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
