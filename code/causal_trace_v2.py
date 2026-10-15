"""Study v2 causal tracing (docs/HYPOTHESIS.md A5): full-answer log-probability attribution, 3 phrasings, any registered model.

Corrupt the subject-token embeddings (Gaussian noise, sigma = 3 x pooled subject-embedding std per language, per-fact seed),
then restore the clean activation at one (layer, prompt-token) site of the residual stream or MLP output and measure the
change in the teacher-forced log-probability of the whole answer (the model's own correct answer, stored as `target`),
relative to the corrupted run. The first-answer-token log-probability is stored as a secondary grid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from common import LANGS, MODELS, load_facts, load_model, subject_span

KINDS = ("resid", "mlp")
ROW_CHUNK = 4
VOCAB_CHUNK = 32768


def hook_name(kind: str, layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post" if kind == "resid" else f"blocks.{layer}.hook_mlp_out"


def prompt_and_target(tok, prompt: str, target: str) -> tuple[list[int], list[int]]:
    p_ids = tok(prompt)["input_ids"]
    full = tok(prompt + " " + target)["input_ids"]
    if full[:len(p_ids)] != p_ids or len(full) == len(p_ids):
        raise ValueError(f"target is not a clean continuation: {prompt!r} + {target!r}")
    return p_ids, full[len(p_ids):]


def pooled_embedding_std(model: HookedTransformer, facts: list[dict]) -> dict[str, float]:
    out = {}
    for lg in LANGS:
        vecs = []
        for f in facts:
            L = f["lang"][lg]
            ids, pos = subject_span(model.tokenizer, L["prompts"][0], L["subject"])
            vecs.append(model.W_E[torch.tensor([ids[p] for p in pos], device="cuda")].float().flatten())
        out[lg] = float(torch.cat(vecs).std())
    return out


class SeqScorer:
    """Teacher-forced log-probability of the target tokens from the final-norm hidden states (fp32 unembedding, soft cap)."""

    def __init__(self, model: HookedTransformer, target: list[int], p_len: int) -> None:
        self.model, self.p_len, self.n = model, p_len, len(target)
        self.tgt = torch.tensor(target, device="cuda")
        self.w = model.ln_final.w.float()
        cap = getattr(model.cfg, "output_logits_soft_cap", 0) or 0
        self.cap = float(cap) if cap and cap > 0 else 0.0

    def _logits(self, h: torch.Tensor) -> torch.Tensor:  # h: [b, n, d] float32 (weighted)
        W = self.model.W_U
        V = W.shape[1]
        out = torch.empty(h.shape[0], h.shape[1], V, device=h.device, dtype=torch.float32)
        for c in range(0, V, VOCAB_CHUNK):
            out[..., c:c + VOCAB_CHUNK] = h @ W[:, c:c + VOCAB_CHUNK].float()
        return self.cap * torch.tanh(out / self.cap) if self.cap else out

    def __call__(self, normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """normed: [B, T_full, d] (ln_final.hook_normalized). Returns (sum logP, first-token logP, teacher-forced top-1 match)."""
        hs = normed[:, self.p_len - 1:self.p_len - 1 + self.n].float() * self.w
        tot, first, match = [], [], []
        for i in range(0, hs.shape[0], ROW_CHUNK):
            lg = self._logits(hs[i:i + ROW_CHUNK])
            lp = lg.log_softmax(-1).gather(-1, self.tgt[None, :, None].expand(lg.shape[0], -1, 1)).squeeze(-1)
            tot.append(lp.sum(-1)); first.append(lp[:, 0])
            match.append((lg.argmax(-1) == self.tgt[None]).all(-1))
            del lg
        return torch.cat(tot), torch.cat(first), torch.cat(match)


@torch.no_grad()
def trace_one(model: HookedTransformer, prompt: str, subject: str, target: str, seed: int, sigma: float, n_layers: int) -> dict:
    tok = model.tokenizer
    p_ids, t_ids = prompt_and_target(tok, prompt, target)
    _, span = subject_span(tok, prompt, subject)
    P = len(p_ids)
    tokens = torch.tensor([p_ids + t_ids], device="cuda")
    g = torch.Generator().manual_seed(seed)
    noise = (torch.randn(len(span), model.cfg.d_model, generator=g) * sigma).to("cuda", model.cfg.dtype)

    def corrupt(act: torch.Tensor, hook) -> torch.Tensor:
        act[:, span] += noise
        return act

    scorer = SeqScorer(model, t_ids, P)
    names = [hook_name(k, l) for k in KINDS for l in range(n_layers)]
    grab: dict[str, torch.Tensor] = {}

    def grab_hook(act: torch.Tensor, hook) -> torch.Tensor:
        grab["h"] = act
        return act

    def run(rows: int, fwd: list, with_noise: bool = True):
        hooks = ([("hook_embed", corrupt)] if with_noise else []) + fwd + [("ln_final.hook_normalized", grab_hook)]
        model.run_with_hooks(tokens.repeat(rows, 1), return_type=None, fwd_hooks=hooks)
        return scorer(grab["h"])

    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    lp_clean, lp1_clean, match_clean = run(1, [], with_noise=False)
    lp_corr, lp1_corr, _ = run(1, [], with_noise=True)
    # cross-check the fast path against the model's own full-logit forward pass
    full = model(tokens)[0, P - 1:P - 1 + len(t_ids)].float().log_softmax(-1)
    ref = float(full.gather(-1, torch.tensor(t_ids, device="cuda")[:, None]).sum())
    idx = torch.arange(P, device="cuda")
    grids = {k: np.zeros((n_layers, P), dtype=np.float32) for k in KINDS}
    grids1 = {k: np.zeros((n_layers, P), dtype=np.float32) for k in KINDS}
    for kind in KINDS:
        for layer in range(n_layers):
            clean = cache[hook_name(kind, layer)][0, :P]

            def patch(act: torch.Tensor, hook, clean=clean) -> torch.Tensor:
                act[idx, idx] = clean
                return act

            lp, lp1, _ = run(P, [(hook_name(kind, layer), patch)])
            grids[kind][layer] = (lp - lp_corr[0]).float().cpu().numpy()
            grids1[kind][layer] = (lp1 - lp1_corr[0]).float().cpu().numpy()
    last = span[-1]
    sc, sk = float(lp_clean[0]), float(lp_corr[0])
    return {"tokens": p_ids, "span": span, "last_subj": last, "n_target": len(t_ids), "target_ids": t_ids,
            "target": target, "tf_match": bool(match_clean[0]),
            "p_clean": float(np.exp(sc)), "p_corr": float(np.exp(sk)), "collapsed": bool(sk < sc - np.log(2.0)),
            "obj_ok": True, "score_clean": sc, "score_corr": sk, "first_clean": float(lp1_clean[0]), "first_corr": float(lp1_corr[0]),
            "fastpath_vs_model_gap": abs(sc - ref), "grids": grids, "grids_first": grids1,
            "crit_mlp_layer": int(grids["mlp"][:, last].argmax()), "crit_resid_layer": int(grids["resid"][:, last].argmax()),
            "sanity_last_site_gap": float(abs(grids["resid"][n_layers - 1, P - 1] - (sc - sk)))}


def select_edit_facts(facts: list[dict], n: int = 10) -> list[dict]:
    """Stratified deterministic subset: round-robin over relations with >= 10 kept facts, by sorted id."""
    by_rel: dict[str, list[dict]] = {}
    for f in sorted(facts, key=lambda f: f["id"]):
        by_rel.setdefault(f["relation"], []).append(f)
    rels = [r for r, v in by_rel.items() if len(v) >= 10] or list(by_rel)
    out, i = [], 0
    while len(out) < n and any(i < len(by_rel[r]) for r in rels):
        out += [by_rel[r][i] for r in rels if i < len(by_rel[r])]
        i += 1
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--facts", default=None)
    ap.add_argument("--subset", choices=["all", "pilot"], default="all", help="pilot = 10 stratified facts")
    ap.add_argument("--phrasings", default="0", help="comma list of phrasing indices to trace, e.g. 0 or 0,1,2")
    ap.add_argument("--limit", type=int, default=None, help="only the first N facts by id")
    ap.add_argument("--tag", default="p3trace_v2")
    a = ap.parse_args()
    a.facts = a.facts or MODELS[a.model]["facts_v2"]
    run_dir = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run_dir.mkdir(parents=True, exist_ok=False)  # write-once
    (run_dir / "grids").mkdir()
    all_facts = load_facts(a.facts)
    facts = select_edit_facts(all_facts) if a.subset == "pilot" else sorted(all_facts, key=lambda f: f["id"])
    if a.limit:
        facts = facts[:a.limit]
    phr = [int(x) for x in a.phrasings.split(",")]
    model = load_model(a.model)
    n_layers = model.cfg.n_layers
    std = pooled_embedding_std(model, all_facts)
    (run_dir / "config.json").write_text(json.dumps(vars(a) | {"model_name": MODELS[a.model]["tl_name"], "dtype": "bfloat16",
                                                               "noise_mult": 3.0, "embed_std_pooled": std, "n_layers": n_layers,
                                                               "fact_ids": [f["id"] for f in facts]}, indent=1))
    records = []
    for f in facts:
        for lg in LANGS:
            L = f["lang"][lg]
            for k in phr:
                try:
                    r = trace_one(model, L["prompts"][k], L["subject"], L["target"], f["seed"], 3.0 * std[lg], n_layers)
                except ValueError as e:
                    records.append({"id": f["id"], "lang": lg, "phrasing": k, "relation": f["relation"], "error": str(e)})
                    print(f"{f['id']} {lg} p{k}: SKIP {e}", flush=True)
                    continue
                np.savez_compressed(run_dir / "grids" / f"{f['id']}_{lg}" if k == 0 else run_dir / "grids" / f"{f['id']}_{lg}_p{k}",
                                    resid=r["grids"]["resid"], mlp=r["grids"]["mlp"], resid_first=r["grids_first"]["resid"],
                                    mlp_first=r["grids_first"]["mlp"], tokens=np.array(r["tokens"]), span=np.array(r["span"]))
                rec = {kk: v for kk, v in r.items() if kk not in ("grids", "grids_first", "tokens")}
                rec |= {"id": f["id"], "lang": lg, "phrasing": k, "relation": f["relation"], "seed": f["seed"], "sigma": 3.0 * std[lg]}
                records.append(rec)
                print(f"{f['id']} {lg} p{k}: target={r['target']!r} n={r['n_target']} P {r['p_clean']:.3f}->{r['p_corr']:.4f} "
                      f"crit_mlp={r['crit_mlp_layer']} gap={r['sanity_last_site_gap']:.3f} fast-ref={r['fastpath_vs_model_gap']:.3f} "
                      f"tf={r['tf_match']}", flush=True)
                torch.cuda.empty_cache()
    (run_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf8")
    print("peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
