"""A13a: independent-noise re-trace of the EN/ES MLP profile at the last subject token (docs/HYPOTHESIS.md A13a).

For each valid fact and language, K corruption draws (IND: independent between languages; SH: shared between languages; ORIG: the original
per-fact seed). For each draw, restore the clean `hook_mlp_out` at the last subject token, one layer per batch row, and store the change in the
teacher-forced full-answer log-probability relative to the corrupted run. Same scorer, templates, sigma and facts as causal_trace_v2.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from causal_trace_v2 import SeqScorer, prompt_and_target
from circuit_overlap import is_valid, load
from common import MODELS, load_facts, load_model, subject_span
from controls_overlap import SETS

LANGS2 = ("en", "es")
K_IND, K_SH = 10, 5
ORIG_CFG = {"llama": "analysis/run_20261015_p3trace_v2_llama/config.json", "gemma2": "analysis/run_20261015_p3trace_v2c_gemma2/config.json"}


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest(), 16) % (2 ** 31)


@torch.no_grad()
def trace_draws(model, prompt: str, subject: str, target: str, sigma: float, seeds: list[int]) -> dict:
    tok = model.tokenizer
    n_layers, d = model.cfg.n_layers, model.cfg.d_model
    p_ids, t_ids = prompt_and_target(tok, prompt, target)
    _, span = subject_span(tok, prompt, subject)
    P, last = len(p_ids), span[-1]
    tokens = torch.tensor([p_ids + t_ids], device="cuda")
    scorer = SeqScorer(model, t_ids, P)
    names = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    clean = torch.stack([cache[n][0, last] for n in names])  # [L, d]
    del cache
    grab: dict[str, torch.Tensor] = {}

    def grab_hook(act, hook):
        grab["h"] = act
        return act

    def run(rows: int, noise, fwd):
        def corrupt(act, hook):
            act[:, span] += noise
            return act
        hooks = [("hook_embed", corrupt)] + fwd + [("ln_final.hook_normalized", grab_hook)]
        model.run_with_hooks(tokens.repeat(rows, 1), return_type=None, fwd_hooks=hooks)
        return scorer(grab["h"])[0]

    fwd = []
    for l in range(n_layers):
        def patch(act, hook, l=l):
            act[l, last] = clean[l]
            return act
        fwd.append((names[l], patch))
    lp_clean = float(scorer_clean(model, tokens, scorer, grab))
    prof, lpc = [], []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        noise = (torch.randn(len(span), d, generator=g) * sigma).to("cuda", model.cfg.dtype)
        lp_corr = run(1, noise, [])
        lp_rest = run(n_layers, noise, fwd)
        prof.append((lp_rest - lp_corr[0]).float().cpu().numpy())
        lpc.append(float(lp_corr[0]))
    return {"profiles": np.stack(prof), "lp_corr": np.array(lpc), "lp_clean": lp_clean, "span": span, "last": last, "tokens": p_ids}


def scorer_clean(model, tokens, scorer, grab):
    def grab_hook(act, hook):
        grab["h"] = act
        return act
    model.run_with_hooks(tokens, return_type=None, fwd_hooks=[("ln_final.hook_normalized", grab_hook)])
    return scorer(grab["h"])[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="p3seeds")
    a = ap.parse_args()
    r1, r2, f1, f2 = SETS[a.model]
    recs = {}
    for r in (r1, r2):
        recs.update(load(Path(r))[1])
    ids = sorted({f for f, _ in recs if all((f, lg) in recs and is_valid(recs[(f, lg)]) for lg in ("en", "es", "hi"))})
    if a.limit:
        ids = ids[:a.limit]
    facts = {f["id"]: f for p in (f1, f2) for f in load_facts(p)}
    std = json.loads(Path(ORIG_CFG[a.model]).read_text())["embed_std_pooled"]
    run_dir = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "grids").mkdir()
    model = load_model(a.model)
    (run_dir / "config.json").write_text(json.dumps({"model": a.model, "amendment": "A13a", "k_ind": K_IND, "k_sh": K_SH, "sigma_mult": 3.0, "embed_std_pooled": std,
                                                     "fact_ids": ids, "seed_rule": "sha256('id|lang|k') mod 2^31 (IND), sha256('id|shared|k') (SH)"}, indent=1))
    out = []
    for n, i in enumerate(ids):
        f = facts[i]
        for lg in LANGS2:
            L = f["lang"][lg]
            seeds = [seed_of(i, lg, k) for k in range(K_IND)] + [seed_of(i, "shared", k) for k in range(K_SH)] + [f["seed"]]
            r = trace_draws(model, L["prompts"][0], L["subject"], L["target"], 3.0 * std[lg], seeds)
            np.savez_compressed(run_dir / "grids" / f"{i}_{lg}.npz", ind=r["profiles"][:K_IND], sh=r["profiles"][K_IND:K_IND + K_SH], orig=r["profiles"][-1],
                                lp_corr=r["lp_corr"], lp_clean=r["lp_clean"], span=np.array(r["span"]), tokens=np.array(r["tokens"]), last=r["last"])
            out.append({"id": i, "lang": lg, "relation": f["relation"], "lp_clean": r["lp_clean"], "lp_corr_ind_mean": float(r["lp_corr"][:K_IND].mean()),
                        "lp_corr_orig": float(r["lp_corr"][-1])})
        print(n + 1, "/", len(ids), i, flush=True)
        torch.cuda.empty_cache()
    (run_dir / "records.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print("peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))


if __name__ == "__main__":
    main()
