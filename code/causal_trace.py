"""P2: causal tracing (Meng et al. 2022, arXiv:2202.05262) of factual recall in EN, ES, HI.

For each fact and language: corrupt the subject-token embeddings with Gaussian noise (sigma = 3 x pooled subject-embedding
std, per-fact seed), then restore the clean activation at one (layer, token) site of the residual stream (layer output)
or of the MLP output, and record the change in logit(object) - logit(competitor) relative to the corrupted run.
Saves full grids (layers x tokens) per fact/language to a write-once run dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from common import LANGS, load_facts, load_model, select_p2_facts, subject_span

KINDS = ("resid", "mlp")


def hook_name(kind: str, layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post" if kind == "resid" else f"blocks.{layer}.hook_mlp_out"


def pooled_embedding_std(model: HookedTransformer, facts: list[dict]) -> dict[str, float]:
    """Std of subject-token embeddings pooled over all facts, per language (the ROME noise scale basis)."""
    out = {}
    for lang in LANGS:
        vecs = []
        for f in facts:
            L = f["lang"][lang]
            ids, pos = subject_span(model.tokenizer, L["prompt"], L["subject"])
            vecs.append(model.W_E[torch.tensor([ids[p] for p in pos], device="cuda")].float().flatten())
        out[lang] = float(torch.cat(vecs).std())
    return out


class Scorer:
    """score = (final-norm hidden at last position) . (W_U[:, obj] - W_U[:, comp]), dot product in fp32."""

    def __init__(self, model: HookedTransformer, obj: int, comp: int) -> None:
        self.dir = (model.W_U[:, obj] - model.W_U[:, comp]).float()
        self.w = model.ln_final.w.float()

    def __call__(self, normalized_last: torch.Tensor) -> torch.Tensor:  # [B, d] before the RMSNorm weight
        return (normalized_last.float() * self.w) @ self.dir


@torch.no_grad()
def trace_one(model: HookedTransformer, prompt: str, subject: str, seed: int, sigma: float, n_layers: int) -> dict:
    ids, span = subject_span(model.tokenizer, prompt, subject)
    tokens = torch.tensor([ids], device="cuda")
    T = tokens.shape[1]
    g = torch.Generator().manual_seed(seed)
    noise = (torch.randn(len(span), model.cfg.d_model, generator=g) * sigma).to("cuda", model.cfg.dtype)

    def corrupt(act: torch.Tensor, hook) -> torch.Tensor:
        act[:, span] += noise
        return act

    names = [hook_name(k, l) for k in KINDS for l in range(n_layers)]
    clean_logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    cl = clean_logits[0, -1].float()
    obj = int(cl.argmax())
    corr_logits = model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", corrupt)])[0, -1].float()
    comp = int(corr_logits.masked_fill(torch.arange(corr_logits.numel(), device="cuda") == obj, float("-inf")).argmax())
    p_clean = float(cl.softmax(-1)[obj])
    p_corr = float(corr_logits.softmax(-1)[obj])
    scorer = Scorer(model, obj, comp)
    s_clean_logits = float(cl[obj] - cl[comp])
    s_corr_logits = float(corr_logits[obj] - corr_logits[comp])

    grab: dict[str, torch.Tensor] = {}

    def grab_hook(act: torch.Tensor, hook) -> torch.Tensor:
        grab["h"] = act[:, -1]
        return act

    def run(fwd: list) -> torch.Tensor:
        model.run_with_hooks(tokens.repeat(T, 1), return_type=None,
                             fwd_hooks=[("hook_embed", corrupt), *fwd, ("ln_final.hook_normalized", grab_hook)])
        return scorer(grab["h"])

    idx = torch.arange(T, device="cuda")
    base = run([])  # all T rows identical: corrupted baseline via the fast scoring path
    s_corr = float(base[0])
    grids = {k: np.zeros((n_layers, T), dtype=np.float32) for k in KINDS}
    for kind in KINDS:
        for layer in range(n_layers):
            clean = cache[hook_name(kind, layer)][0]

            def patch(act: torch.Tensor, hook, clean=clean) -> torch.Tensor:
                act[idx, idx] = clean
                return act

            s = run([(hook_name(kind, layer), patch)])
            grids[kind][layer] = (s - s_corr).float().cpu().numpy()
    # fast-path clean score for a consistency check
    model.run_with_hooks(tokens, return_type=None, fwd_hooks=[("ln_final.hook_normalized", grab_hook)])
    s_clean = float(scorer(grab["h"])[0])
    last = span[-1]
    return {"tokens": ids, "span": span, "last_subj": last, "obj": obj, "comp": comp,
            "obj_str": model.tokenizer.decode(obj), "comp_str": model.tokenizer.decode(comp),
            "obj_ok": bool(model.tokenizer.decode(obj).strip()), "p_clean": p_clean, "p_corr": p_corr, "collapsed": bool(p_corr < 0.5 * p_clean),
            "score_clean": s_clean, "score_corr": s_corr, "score_clean_logits": s_clean_logits,
            "score_corr_logits": s_corr_logits, "grids": grids,
            "crit_mlp_layer": int(grids["mlp"][:, last].argmax()),
            "crit_resid_layer": int(grids["resid"][:, last].argmax()),
            "sanity_last_site_gap": float(abs(grids["resid"][n_layers - 1, T - 1] - (s_clean - s_corr)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/facts.json")
    ap.add_argument("--subset", choices=["p2", "all"], default="p2")
    ap.add_argument("--tag", default="p2trace")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N facts")
    a = ap.parse_args()

    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    run.mkdir(parents=True, exist_ok=False)  # write-once
    (run / "grids").mkdir()
    all_facts = load_facts(a.facts)
    facts = select_p2_facts(all_facts) if a.subset == "p2" else all_facts
    if a.limit:
        facts = facts[: a.limit]
    model = load_model()
    n_layers = model.cfg.n_layers
    std = pooled_embedding_std(model, all_facts)
    cfg = vars(a) | {"model": "Llama-3.2-3B-Instruct", "dtype": "bfloat16", "noise_mult": 3.0,
                     "embed_std_pooled": std, "n_layers": n_layers, "fact_ids": [f["id"] for f in facts]}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))

    records = []
    for f in facts:
        for lang in LANGS:
            L = f["lang"][lang]
            r = trace_one(model, L["prompt"], L["subject"], f["seed"], 3.0 * std[lang], n_layers)
            np.savez_compressed(run / "grids" / f"{f['id']}_{lang}.npz", resid=r["grids"]["resid"], mlp=r["grids"]["mlp"],
                                tokens=np.array(r["tokens"]), span=np.array(r["span"]))
            rec = {k: v for k, v in r.items() if k not in ("grids", "tokens")} | {"id": f["id"], "lang": lang,
                                                                                  "relation": f["relation"], "seed": f["seed"],
                                                                                  "sigma": 3.0 * std[lang]}
            records.append(rec)
            print(f"{f['id']} {lang}: obj={r['obj_str']!r} p {r['p_clean']:.3f}->{r['p_corr']:.3f} "
                  f"crit_mlp={r['crit_mlp_layer']} crit_resid={r['crit_resid_layer']} gap={r['sanity_last_site_gap']:.3f} "
                  f"score clean/fast={r['score_clean']:.2f}/{r['score_clean_logits']:.2f} corr fast/logits={r['score_corr']:.2f}/{r['score_corr_logits']:.2f}",
                  flush=True)
            torch.cuda.empty_cache()
    (run / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf8")
    print("peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
