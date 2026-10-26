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
               "ovs": "analysis/run_20261015_p3overlap_v2c_gemma2/results.json"},
}
CMP = "analysis/run_20261015_compare_models/results.json"
HD = "analysis/run_20261015_hindi_diag/results.json"
HS = "analysis/run_20261015_hindi_diag/span_split.json"
LH = "analysis/run_20261019_layer_hist/layer_hist.json"
SWT = {"llama": "analysis/run_20261015_p4sweep_llama/sweep_tables.json", "gemma2": "analysis/run_20261019_p4sweep_gemma2/sweep_tables.json"}
OVA = {"llama": "analysis/run_20261019_p3overlap_v2all_10191057/results.json", "gemma2": "analysis/run_20261020_p3overlap_v2all_10200339/results.json"}
H8 = {"llama": "analysis/run_20261019_h8_10191057_llama/results.json", "gemma2": "analysis/run_20261019_h8_10191057_gemma2/results.json"}
INC = {"llama": "analysis/run_20261020_incontext_10200339_llama/summary.json", "gemma2": "analysis/run_20261020_incontext_10200339_gemma2/summary.json"}
CTL = "analysis/run_20261021_controls/results.json"
CF = "analysis/run_20261024_confound_A12a/results.json"
SD = "analysis/run_20261024_seeds_A13a_10240109/results.json"
CV = "analysis/run_20261024_seeds_covariates_A13b/results.json"
E2 = "analysis/run_20261025_e2_analysis_A13/results.json"
E2H = "analysis/run_20261026_e2h_analysis_A15/results.json"
MAN = Path("results/manifest.json")


PK = {"llama_v2": "analysis/run_20261025_p3overlap_v2_perm10k_llama/results.json", "gemma2_v2": "analysis/run_20261025_p3overlap_v2_perm10k_gemma2/results.json",
      "llama_all": "analysis/run_20261025_p3overlap_v2all_perm10k_llama/results.json", "gemma2_all": "analysis/run_20261025_p3overlap_v2all_perm10k_gemma2/results.json"}


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
    for m in ("llama", "gemma2"):  # F2 Hindi diagnostic (paper_v2 section 4.3)
        c += [(f"hd_{m}_spearman_enhi_vs_span", HD, [m, "spearman_enhi_cosine_vs_hi_subject_span"]), (f"hd_{m}_hi_span_median", HD, [m, "hi_span_median"])]
        for b in ("1-3", "4-6", "7+"):
            c.append((f"hd_{m}_acc_hi_{b}", HD, [m, "accuracy_by_object_tokens", f"hi/{b}", "acc"]))
        for part in ("short_hi_span", "long_hi_span"):
            c += [(f"hs_{m}_{part}_n", HS, [f"{m}/{part}", "n"]), (f"hs_{m}_{part}_cos_en-hi", HS, [f"{m}/{part}", "cos", "en-hi"]),
                  (f"hs_{m}_{part}_pooled_z_en-hi", HS, [f"{m}/{part}", "pooled_z", "en-hi"])]
    for m, name in (("llama", "Llama-3.2-3B"), ("gemma2", "Gemma-2-2B")):  # critical-layer distribution (section 4.2, Figure 4)
        for lg in ("en", "es", "hi"):
            c.append((f"lh_{m}_frac_8_19_{lg}", LH, [name, lg, "frac_layer_8_19"]))
        if m == "gemma2":
            for lg in ("en", "es", "hi"):
                c.append((f"lh_gemma2_hist_L14_{lg}", LH, [name, lg, "hist", 14]))
    for m, art in SWT.items():  # layer sweep (section 4.4)
        for cond in ("traced", "mid", "random"):
            c += [(f"sw_{m}_{cond}_n", art, ["conditions", cond, "n"]), (f"sw_{m}_{cond}_en_success", art, ["conditions", cond, "en_success_rate"]),
                  (f"sw_{m}_{cond}_transfer_es_med", art, ["conditions", cond, "transfer_es", 0]), (f"sw_{m}_{cond}_transfer_hi_med", art, ["conditions", cond, "transfer_hi", 0]),
                  (f"sw_{m}_{cond}_specificity_med", art, ["conditions", cond, "specificity", 0])]
        for key in ("transfer_es", "transfer_hi", "specificity"):
            for k, i in (("med", 0), ("lo", 1), ("hi", 2)):
                c.append((f"sw_{m}_H7_{key}_{k}", art, ["H7", key, "median_diff_near_mid_minus_early", i]))
            c.append((f"sw_{m}_H7_{key}_frac_pos", art, ["H7", key, "frac_facts_positive"]))
    for m, art in OVA.items():  # overlap incl. non-copyable relations (section 4.5)
        sets = ["primary_valid", "valid_no_copyable", "primary_valid_stratified_by_relation"] + (["valid_city_country"] if m == "llama" else [])
        for s in sets:
            c.append((f"ova_{m}_{s}_n", art, ["sets", s, "n_facts"]))
            for pr in ("en-es", "en-hi", "es-hi"):
                c += [(f"ova_{m}_{s}_{pr}_cos", art, ["sets", s, "pairs", pr, "mlp_cosine", "observed_median"]),
                      (f"ova_{m}_{s}_{pr}_z", art, ["sets", s, "pairs", pr, "mlp_cosine", "z"])]
    for m, art in H8.items():  # phrasing robustness (H8)
        c += [(f"h8_{m}_n", art, ["summary", "n_facts"]), (f"h8_{m}_within", art, ["summary", "median_within"]), (f"h8_{m}_cross", art, ["summary", "median_cross"]),
              (f"h8_{m}_d", art, ["summary", "median_d"]), (f"h8_{m}_d_lo", art, ["summary", "ci95_d", 0]), (f"h8_{m}_d_hi", art, ["summary", "ci95_d", 1]),
              (f"h8_{m}_frac_pos", art, ["summary", "frac_facts_d_positive"])]
    for m, art in INC.items():  # in-context baseline
        for k in ("en_context/en", "en_context/es", "en_context/hi"):
            c += [(f"inc_{m}_{k}_dP", art, ["summary", k, "median_dP"]), (f"inc_{m}_{k}_success", art, ["summary", k, "success_rate"])]
    for m7 in ("llama", "gemma2"):  # 30-fact follow-up edits (F7)
        f7 = f"analysis/run_20261020_p4f7_{m7}/sweep_tables.json"
        for cond in ("traced", "mid", "random"):
            c += [(f"f7_{m7}_{cond}_n", f7, ["conditions", cond, "n"]), (f"f7_{m7}_{cond}_en_success", f7, ["conditions", cond, "en_success_rate"]),
                  (f"f7_{m7}_{cond}_transfer_es_med", f7, ["conditions", cond, "transfer_es", 0]), (f"f7_{m7}_{cond}_transfer_hi_med", f7, ["conditions", cond, "transfer_hi", 0]),
                  (f"f7_{m7}_{cond}_specificity_med", f7, ["conditions", cond, "specificity", 0]), (f"f7_{m7}_{cond}_ppl_med", f7, ["conditions", cond, "ppl_ratio", 0])]
    for m in ("llama", "gemma2"):  # A11 controls (section 4.6)
        c += [(f"ctl_{m}_n", CTL, [m, "n"]), (f"ctl_{m}_clusters", CTL, [m, "C3", "n_clusters"]),
              (f"ctl_{m}_ci_lo", CTL, [m, "C3", "cluster_boot_ci95", 0]), (f"ctl_{m}_ci_hi", CTL, [m, "C3", "cluster_boot_ci95", 1]),
              (f"ctl_{m}_spearman_sim_cos", CTL, [m, "C2", "spearman_sim_vs_cos", 0]), (f"ctl_{m}_frac_identical", CTL, [m, "C2", "frac_identical_after_norm"])]
        c += [(f"ctl_{m}_C0_{k}", CTL, [m, "C0", k]) for k in ("same_fact_en_es", "diff_fact_same_rel_en_es", "diff_fact_same_rel_en_en", "diff_fact_same_rel_es_es")]
        for v in ("C1_uncentred_stratified", "C1_centred_stratified"):
            c += [(f"ctl_{m}_{v}_{k}", CTL, [m, v, k]) for k in ("median", "null_mean", "z")]
        for h in ("low", "high"):
            c.append((f"ctl_{m}_{h}_n", CTL, [m, "C2", h, "n"]))
            for v in ("uncentred", "centred"):
                c += [(f"ctl_{m}_{h}_{v}_median", CTL, [m, "C2", h, v, "median"]), (f"ctl_{m}_{h}_{v}_z", CTL, [m, "C2", h, v, "z"])]
    for m in ("llama", "gemma2"):  # A12/A12a confound-adjusted analysis (CPU)
        c += [(f"cf_{m}_n", CF, [m, "n"]), (f"cf_{m}_n_J0", CF, [m, "n_J0"]), (f"cf_{m}_n_NR0", CF, [m, "n_NR0"])]
        c += [(f"cf_{m}_class_{k}", CF, [m, "prefix_class_counts", k]) for k in ("IP", "EQ", "OT")]
        for nm in ("all", "informative", "EQ", "OT", "event_year_only", "non_event_year"):
            for v in ("NC", "raw_centred_A12"):
                c += [(f"cf_{m}_S1_{nm}_{v}_D", CF, [m, "S1", nm, v, "mean_D"]), (f"cf_{m}_S1_{nm}_{v}_p", CF, [m, "S1", nm, v, "p"])]
            c += [(f"cf_{m}_S1_{nm}_n", CF, [m, "S1", nm, "NC", "n"]), (f"cf_{m}_S1_{nm}_NC_ci_lo", CF, [m, "S1", nm, "NC", "ci97.5_cluster", 0]),
                  (f"cf_{m}_S1_{nm}_NC_ci_hi", CF, [m, "S1", nm, "NC", "ci97.5_cluster", 1])]
        for nm in ("all", "informative"):
            c += [(f"cf_{m}_retr_{nm}_top1", CF, [m, "S1", nm, "retrieval_NC", "top1"]), (f"cf_{m}_retr_{nm}_chance", CF, [m, "S1", nm, "retrieval_NC", "chance_top1"]),
                  (f"cf_{m}_retr_{nm}_p", CF, [m, "S1", nm, "retrieval_NC", "p_top1"])]
        c += [(f"cf_{m}_S4a_pred", CF, [m, "S4a_informative", "target_pred_D"]), (f"cf_{m}_S4a_ci_lo", CF, [m, "S4a_informative", "target_ci97.5", 0]),
              (f"cf_{m}_S4a_ci_hi", CF, [m, "S4a_informative", "target_ci97.5", 1]), (f"cf_{m}_S4a_n", CF, [m, "S4a_informative", "n"]),
              (f"cf_{m}_TE_rho", CF, [m, "S1b_TE_spearman_within_stratum", "rho"])]
        for b in ("early", "middle", "late"):
            c += [(f"cf_{m}_band_{b}_all_D", CF, [m, "S3_bands", b, "all", "mean_D"]), (f"cf_{m}_band_{b}_inf_D", CF, [m, "S3_bands", b, "informative", "mean_D"])]
    for m in ("llama", "gemma2"):  # A13a independent-noise re-trace
        c += [(f"sd_{m}_P4_pass", SD, [m, "P4_pass"]), (f"sd_{m}_G1_D", SD, [m, "G1", "D_ind_informative"]), (f"sd_{m}_G1_ci_lo", SD, [m, "G1", "ci97.5", 0]),
              (f"sd_{m}_G1_ci_hi", SD, [m, "G1", "ci97.5", 1]), (f"sd_{m}_G1_ceil_en", SD, [m, "G1", "ceiling_en"]), (f"sd_{m}_G1_ceil_es", SD, [m, "G1", "ceiling_es"])]
        for nm in ("all", "informative", "EQ", "OT", "event_year_only", "non_event_year"):
            c += [(f"sd_{m}_{nm}_n", SD, [m, "S", nm, "ind", "n"]), (f"sd_{m}_{nm}_ind_D", SD, [m, "S", nm, "ind", "mean_D"]), (f"sd_{m}_{nm}_ind_ci_lo", SD, [m, "S", nm, "ind", "ci97.5_cluster", 0]),
                  (f"sd_{m}_{nm}_ind_ci_hi", SD, [m, "S", nm, "ind", "ci97.5_cluster", 1]), (f"sd_{m}_{nm}_sh_D", SD, [m, "S", nm, "sh", "mean_D"]),
                  (f"sd_{m}_{nm}_orig_D", SD, [m, "S", nm, "orig", "mean_D"]), (f"sd_{m}_{nm}_ind5_D", SD, [m, "S", nm, "D_ind5_mean"]),
                  (f"sd_{m}_{nm}_delta", SD, [m, "S", nm, "Delta_SH_minus_IND5", "mean"]), (f"sd_{m}_{nm}_delta_lo", SD, [m, "S", nm, "Delta_SH_minus_IND5", "ci97.5_cluster", 0]),
                  (f"sd_{m}_{nm}_delta_hi", SD, [m, "S", nm, "Delta_SH_minus_IND5", "ci97.5_cluster", 1]),
                  (f"sd_{m}_{nm}_ceil_en", SD, [m, "S", nm, "enAB", "mean_D"]), (f"sd_{m}_{nm}_ceil_es", SD, [m, "S", nm, "esAB", "mean_D"])]
        for lg in ("en", "es"):
            c += [(f"sd_{m}_reg_{lg}_median_abs", SD, [m, "P4_regression", lg, "median_abs_diff"]), (f"sd_{m}_reg_{lg}_corr", SD, [m, "P4_regression", lg, "corr"])]
    for m in ("llama", "gemma2"):  # A13b covariate model on independent-noise increments
        for sub in ("informative", "all_non_IP"):
            c += [(f"cv_{m}_{sub}_n", CV, [m, sub, "n"]), (f"cv_{m}_{sub}_pred", CV, [m, sub, "target_pred_D"]), (f"cv_{m}_{sub}_ci_lo", CV, [m, sub, "target_ci97.5", 0]),
                  (f"cv_{m}_{sub}_ci_hi", CV, [m, sub, "target_ci97.5", 1]), (f"cv_{m}_{sub}_boot_lo", CV, [m, sub, "target_boot_ci97.5", 0]), (f"cv_{m}_{sub}_boot_hi", CV, [m, sub, "target_boot_ci97.5", 1]),
                  (f"cv_{m}_{sub}_J0", CV, [m, sub, "n_J0"]), (f"cv_{m}_{sub}_verdict", CV, [m, sub, "verdict"])]
    for m in ("llama", "gemma2"):  # A13 E2 interchange
        for d in ("en2es", "es2en"):
            c += [(f"e2_{m}_{d}_n_conf", E2, [m, d, "n_confirmatory"])]
            for st in ("A", "B"):
                b = [m, d, f"site_{st}"]
                c += [(f"e2_{m}_{d}_{st}_I", E2, b + ["confirmatory", "I"]), (f"e2_{m}_{d}_{st}_ci95_lo", E2, b + ["confirmatory", "ci95", 0]), (f"e2_{m}_{d}_{st}_ci95_hi", E2, b + ["confirmatory", "ci95", 1]),
                      (f"e2_{m}_{d}_{st}_verdict", E2, b + ["verdict"]), (f"e2_{m}_{d}_{st}_P1", E2, b + ["gate", "gate_P1"]), (f"e2_{m}_{d}_{st}_lstar", E2, b + ["gate", "l_star"]),
                      (f"e2_{m}_{d}_{st}_same_lstar", E2, b + ["gate", "mean_same_at_lstar"]), (f"e2_{m}_{d}_{st}_flip_same", E2, b + ["flip_at_lstar_confirmatory", "same"]),
                      (f"e2_{m}_{d}_{st}_flip_cross", E2, b + ["flip_at_lstar_confirmatory", "cross"]), (f"e2_{m}_{d}_{st}_I_all", E2, b + ["all_pairs", "I"])]
        c += [(f"e2_{m}_J0", E2, [m, "en2es", "covariate_model", "pred_ratio_at_J0"]), (f"e2_{m}_J1", E2, [m, "en2es", "covariate_model", "pred_ratio_at_J1"]),
              (f"e2_{m}_P5", E2, [m, "en2es", "covariate_model", "P5_lexical"]), (f"e2_{m}_P6_rho", E2, [m, "en2es", "P6_validity", "spearman"]),
              (f"e2_{m}_P6_n", E2, [m, "en2es", "P6_validity", "n_facts"]), (f"e2_{m}_P6_pass", E2, [m, "en2es", "P6_validity", "P6_pass"])]
    for m in ("llama", "gemma2"):  # A15 Hindi interchange
        c += [(f"e2h_{m}_n_facts", E2H, [m, "n_facts"]), (f"e2h_{m}_n_pair_dirs", E2H, [m, "n_pair_directions"])]
        for d in ("en2hi", "hi2en"):
            c += [(f"e2h_{m}_{d}_n_pairs", E2H, [m, d, "site_A", "confirmatory", "n_pairs"]), (f"e2h_{m}_{d}_n_src", E2H, [m, d, "site_A", "confirmatory", "n_source_entities"])]
            for st in ("A", "B"):
                b = [m, d, f"site_{st}"]
                c += [(f"e2h_{m}_{d}_{st}_I", E2H, b + ["confirmatory", "I"]), (f"e2h_{m}_{d}_{st}_ci95_lo", E2H, b + ["confirmatory", "ci95", 0]), (f"e2h_{m}_{d}_{st}_ci95_hi", E2H, b + ["confirmatory", "ci95", 1]),
                      (f"e2h_{m}_{d}_{st}_verdict", E2H, b + ["verdict"]), (f"e2h_{m}_{d}_{st}_P1", E2H, b + ["gate", "gate_P1"]), (f"e2h_{m}_{d}_{st}_flip_same", E2H, b + ["flip_at_lstar_confirmatory", "same"]),
                      (f"e2h_{m}_{d}_{st}_flip_cross", E2H, b + ["flip_at_lstar_confirmatory", "cross"]), (f"e2h_{m}_{d}_{st}_same_lstar", E2H, b + ["gate", "mean_same_at_lstar"])]
    for tag, art in PK.items():  # A14: overlap rerun with 10,000 permutations
        for pr in ("en-es", "en-hi", "es-hi"):
            for st in ("z", "p_one_sided", "observed_median"):
                c.append((f"pk_{tag}_{pr}_{st}", art, ["sets", "primary_valid", "pairs", pr, "mlp_cosine", st]))
            if tag.endswith("_all"):
                for st in ("z", "p_one_sided"):
                    c.append((f"pk_{tag}_strat_{pr}_{st}", art, ["sets", "primary_valid_stratified_by_relation", "pairs", pr, "mlp_cosine", st]))
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
