"""Study v2 filter (docs/HYPOTHESIS.md A5, A6): keep facts the model answers correctly in EN, ES and HI.

A fact-language pair is correct if the canonical phrasing (index 0) begins with the object or a listed alternative form
(strict, normalised) AND at least one other phrasing has it in its first line (lenient); see A6, A7.
Writes data/v2/facts_<model>.json and a write-once run dir with raw generations and the accuracy table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DTYPE, LANGS, MODELS
from matching import lenient_match, strict_match

N_NEW, BATCH = 24, 32


@torch.no_grad()
def generate(model, tok, prompts: list[str]) -> list[str]:
    outs: list[str] = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to("cuda")
        o = model.generate(**enc, max_new_tokens=N_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--candidates", default="data/v2/facts_candidates.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--facts-out", default=None, help="default: the facts_v2 path registered for --model")
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag or 'p1v2_' + a.model}"
    run.mkdir(parents=True, exist_ok=False)  # write-once
    (run / "config.json").write_text(json.dumps(vars(a) | {"model": spec["tl_name"], "dtype": str(DTYPE), "n_new": N_NEW,
                                                          "batch": BATCH, "padding": "left"}, indent=1))
    cands = json.loads(Path(a.candidates).read_text(encoding="utf8"))["facts"]
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=DTYPE, attn_implementation="eager").to("cuda").eval()

    jobs = [(f["id"], lg, k, p) for f in cands for lg in LANGS for k, p in enumerate(f["lang"][lg]["prompts"])]
    gens = generate(model, tok, [j[3] for j in jobs])
    raw: dict[str, dict] = {}
    for (fid, lg, k, p), g in zip(jobs, gens):
        raw.setdefault(fid, {}).setdefault(lg, {})[str(k)] = {"prompt": p, "gen": g}
    (run / "generations.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf8")

    kept, table = [], {}
    for f in cands:
        ok_all, rec = True, {}
        for lg in LANGS:
            L = f["lang"][lg]
            forms = [L["object"], *L["object_alts"]]
            gens_k = [raw[f["id"]][lg][str(k)]["gen"] for k in range(len(L["prompts"]))]
            m = [strict_match(gens_k[0], forms)] + [lenient_match(g, forms) or None for g in gens_k[1:]]
            correct = [x is not None for x in m]
            rec[lg] = {"subject": L["subject"], "object": L["object"], "object_alts": L["object_alts"], "prompts": L["prompts"],
                       "correct": correct, "target": m[0]}
            ok_all &= bool(correct[0] and any(correct[1:]))
            t = table.setdefault(f["relation"], {"n": 0, "kept": 0, **{f"{l}_p{k}": 0 for l in LANGS for k in range(3)}})
            for k, c in enumerate(correct):
                t[f"{lg}_p{k}"] += int(c)
        table.setdefault(f["relation"], {"n": 0, "kept": 0})["n"] += 1
        if ok_all:
            table[f["relation"]]["kept"] += 1
            kept.append({"id": f["id"], "relation": f["relation"], "seed": f["seed"], "lang": rec})
    summary = {"n_candidates": len(cands), "n_kept": len(kept), "per_relation": table}
    Path(a.facts_out or spec["facts_v2"]).write_text(json.dumps({"version": 2, "model": spec["tl_name"], "summary": summary, "facts": kept},
                                                 ensure_ascii=False, indent=1), encoding="utf8")
    (run / "accuracy.json").write_text(json.dumps(summary, indent=1))
    hdr = "| relation | n | kept | " + " | ".join(f"{l} p{k}" for l in LANGS for k in range(3)) + " |"
    lines = [hdr, "|" + "---|" * (3 + 9)]
    for r, t in table.items():
        lines.append(f"| {r} | {t['n']} | {t['kept']} | " + " | ".join(f"{t[f'{l}_p{k}'] / t['n']:.2f}" for l in LANGS for k in range(3)) + " |")
    lines.append(f"| ALL | {len(cands)} | {len(kept)} |" + " |" * 9)
    (run / "accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
