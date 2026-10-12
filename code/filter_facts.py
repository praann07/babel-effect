"""P1: keep candidate facts the model completes correctly in EN, ES and HI.

A fact is correct in a language if the greedy continuation of its prompt, after dropping leading spaces only (a leading newline fails), starts with the object string.
Hindi has two template variants per fact; the variant is chosen once per relation by whole-set Hindi accuracy
(never per fact), then the fact must pass under that variant. Raw generations go to a write-once run dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGS = ("en", "es", "hi")


@torch.no_grad()
def complete(model, tok, prompt: str, obj: str) -> tuple[str, float]:
    """Greedy continuation long enough to cover the object; returns (text, prob of first generated token)."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    n_new = len(tok(obj, add_special_tokens=False).input_ids) + 2
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    p0 = float(out.scores[0][0].float().softmax(-1).max())
    return tok.decode(out.sequences[0, ids.input_ids.shape[1]:], skip_special_tokens=True), p0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/llama-3.2-3b-instruct")
    ap.add_argument("--candidates", default="data/facts_candidates.json")
    ap.add_argument("--out", default="data/facts.json")
    ap.add_argument("--tag", default="p1filter")
    a = ap.parse_args()

    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    run.mkdir(parents=True, exist_ok=False)  # write-once
    (run / "config.json").write_text(json.dumps(vars(a) | {"model": "Llama-3.2-3B-Instruct (unsloth mirror)",
                                                          "dtype": "bfloat16", "decoding": "greedy"}, indent=1))

    cands = json.loads(Path(a.candidates).read_text(encoding="utf8"))["facts"]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16).to("cuda").eval()

    # results[id][(lang, variant)] = (ok, generation, p0)
    res: dict[str, dict[str, dict]] = {}
    for f in cands:
        r: dict[str, dict] = {}
        for lang in LANGS:
            L = f["lang"][lang]
            for v, prompt in enumerate(L["prompts"]):
                txt, p0 = complete(model, tok, prompt, L["object"])
                r[f"{lang}{v}"] = {"prompt": prompt, "gen": txt, "p_first": round(p0, 4),
                                   "ok": txt.lstrip(" ").startswith(L["object"])}
        res[f["id"]] = r
    (run / "generations.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")

    rels = sorted({f["relation"] for f in cands})
    by_rel = {rel: [f for f in cands if f["relation"] == rel] for rel in rels}
    hi_var: dict[str, int] = {}
    table: dict[str, dict] = {}
    for rel, fs in by_rel.items():
        n = len(fs)
        hi_acc = [sum(res[f["id"]][f"hi{v}"]["ok"] for f in fs) / n for v in range(2)]
        hi_var[rel] = int(hi_acc[1] > hi_acc[0])  # tie -> variant 0
        table[rel] = {"n": n, "en": sum(res[f["id"]]["en0"]["ok"] for f in fs) / n,
                      "es": sum(res[f["id"]]["es0"]["ok"] for f in fs) / n,
                      "hi_v0": hi_acc[0], "hi_v1": hi_acc[1], "hi_variant": hi_var[rel]}

    kept = []
    for f in cands:
        v = hi_var[f["relation"]]
        r = res[f["id"]]
        if r["en0"]["ok"] and r["es0"]["ok"] and r[f"hi{v}"]["ok"]:
            g = {"id": f["id"], "relation": f["relation"], "seed": f["seed"], "hi_variant": v, "lang": {}}
            for lang in LANGS:
                L = f["lang"][lang]
                g["lang"][lang] = {"subject": L["subject"], "object": L["object"],
                                   "prompt": L["prompts"][v if lang == "hi" else 0]}
            kept.append(g)
    for rel in rels:
        table[rel]["kept"] = sum(k["relation"] == rel for k in kept)
    summary = {"n_candidates": len(cands), "n_kept": len(kept), "per_relation": table,
               "all_langs_correct_rate": len(kept) / len(cands)}
    Path(a.out).write_text(json.dumps({"version": 1, "model": "Llama-3.2-3B-Instruct", "summary": summary,
                                       "facts": kept}, ensure_ascii=False, indent=1), encoding="utf8")
    (run / "accuracy.json").write_text(json.dumps(summary, indent=1))

    lines = ["| relation | n | EN acc | ES acc | HI acc v0 | HI acc v1 | HI variant | kept |", "|---|---|---|---|---|---|---|---|"]
    for rel, t in table.items():
        lines.append(f"| {rel} | {t['n']} | {t['en']:.2f} | {t['es']:.2f} | {t['hi_v0']:.2f} | {t['hi_v1']:.2f} | "
                     f"{t['hi_variant']} | {t['kept']} |")
    lines.append(f"| ALL | {len(cands)} | | | | | | {len(kept)} |")
    (run / "accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
