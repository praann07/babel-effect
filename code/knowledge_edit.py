"""P4/P5: ROME-style rank-1 MLP edit in English, then measure transfer to ES/HI, specificity and damage.

Design fixed in docs/HYPOTHESIS.md A4. One edit = one (fact, layer-condition); weights are restored afterwards.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from common import LANGS, MODELS, load_facts, load_model, select_p2_facts, share_transposed_mlp_weights, subject_span

PREFIXES = ["", "Fact: ", "Note: ", "Q: ", "Did you know? "]
N_COV_TOKENS, WINDOW = 30_000, 128
STEPS, LR, KL_FACTOR, WD, CLAMP = 25, 0.5, 0.0625, 0.5, 4.0
MODEL_KEY = "llama"  # set in main(); keys the covariance cache
Path(".cache").mkdir(exist_ok=True)


# ---------- wikitext ----------
def wiki_tokens(model: HookedTransformer) -> list[list[int]]:
    df = pd.read_parquet("data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet")
    text = " ".join(t.strip() for t in df["text"] if t.strip() and not t.strip().startswith("="))
    ids = model.tokenizer(text[:400_000], add_special_tokens=False)["input_ids"][:N_COV_TOKENS]
    bos = model.tokenizer.bos_token_id
    return [[bos] + ids[i:i + WINDOW - 1] for i in range(0, len(ids), WINDOW - 1)]


def wiki_sentences(n: int = 100, seed: int = 0) -> list[str]:
    df = pd.read_parquet("data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    sents = []
    for t in df["text"]:
        t = t.strip()
        if not t or t.startswith("="):
            continue
        for s in t.split(" . "):
            if 8 <= len(s.split()) <= 40:
                sents.append(s.strip() + " .")
    random.Random(seed).shuffle(sents)
    return sents[:n]


@torch.no_grad()
def covariance(model: HookedTransformer, layer: int, windows: list[list[int]]) -> torch.Tensor:
    path = Path(".cache") / (f"cov_L{layer}.pt" if MODEL_KEY == "llama" else f"cov_{MODEL_KEY}_L{layer}.pt")
    if path.exists():
        return torch.load(path, weights_only=True)
    d = model.cfg.d_mlp
    C = torch.zeros(d, d, device="cuda", dtype=torch.float32)
    n = 0
    name = f"blocks.{layer}.mlp.hook_post"
    for i in range(0, len(windows), 8):
        batch = windows[i:i + 8]
        L = min(len(w) for w in batch)
        toks = torch.tensor([w[:L] for w in batch], device="cuda")
        _, cache = model.run_with_cache(toks, names_filter=name, stop_at_layer=layer + 1)
        a = cache[name].reshape(-1, d).float()
        C += a.T @ a
        n += a.shape[0]
        del cache, a
    C = (C / n).cpu()
    torch.cuda.empty_cache()
    torch.save(C, path)
    return C


@torch.no_grad()
def perplexity(model: HookedTransformer, sents: list[str]) -> float:
    nll, cnt = 0.0, 0
    for s in sents:
        toks = model.to_tokens(s)
        logits = model(toks)[0, :-1].float()
        nll += float(F.cross_entropy(logits, toks[0, 1:], reduction="sum"))
        cnt += toks.shape[1] - 1
    return math.exp(nll / cnt)


# ---------- tokens and probabilities ----------
def first_obj_token(tok, prompt: str, obj: str) -> int:
    p = tok(prompt)["input_ids"]
    full = tok(prompt + " " + obj)["input_ids"]
    if full[:len(p)] != p:
        raise ValueError(f"prompt tokens are not a prefix for {prompt!r} + {obj!r}")
    return full[len(p)]


@torch.no_grad()
def last_probs(model: HookedTransformer, prompt: str) -> torch.Tensor:
    toks = torch.tensor([model.tokenizer(prompt)["input_ids"]], device="cuda")
    return model(toks)[0, -1].float().softmax(-1)


# ---------- the edit ----------
def context_inputs(model: HookedTransformer, prompt: str, subject: str) -> list[tuple[torch.Tensor, int]]:
    out = []
    for pre in PREFIXES:
        ids, span = subject_span(model.tokenizer, pre + prompt, subject)
        out.append((torch.tensor([ids], device="cuda"), span[-1]))
    return out


def has_post_norm(model: HookedTransformer) -> bool:
    """Gemma-2 normalises the MLP output (ln2_post) before hook_mlp_out, so the edit point is the norm's input."""
    return bool(model.cfg.use_normalization_before_and_after)


@torch.no_grad()
def get_key_and_value(model: HookedTransformer, ctxs: list, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean key (MLP hidden) and mean current MLP output (before any post-norm) at the last subject token."""
    ks, vs = [], []
    for toks, pos in ctxs:
        seen: dict[str, torch.Tensor] = {}
        h = None
        if has_post_norm(model):
            h = model.blocks[layer].ln2_post.register_forward_pre_hook(lambda m, a: seen.__setitem__("v", a[0][0, pos].float()))
        try:
            _, cache = model.run_with_cache(toks, names_filter=[f"blocks.{layer}.mlp.hook_post", f"blocks.{layer}.hook_mlp_out"],
                                            stop_at_layer=layer + 1)
        finally:
            if h is not None:
                h.remove()
        ks.append(cache[f"blocks.{layer}.mlp.hook_post"][0, pos].float())
        vs.append(seen["v"] if has_post_norm(model) else cache[f"blocks.{layer}.hook_mlp_out"][0, pos].float())
    return torch.stack(ks).mean(0), torch.stack(vs).mean(0)


def last_logits_grad(model: HookedTransformer, toks: torch.Tensor, hooks: list) -> torch.Tensor:
    """Last-position logits without the full unembedding copy that autograd would otherwise keep alive."""
    grab: dict[str, torch.Tensor] = {}

    def g(act: torch.Tensor, hook) -> torch.Tensor:
        grab["h"] = act[0, -1]
        return act

    model.run_with_hooks(toks, return_type=None, fwd_hooks=[*hooks, ("ln_final.hook_normalized", g)])
    lg = ((grab["h"] * model.ln_final.w).to(model.W_U.dtype) @ model.W_U).float()
    cap = getattr(model.cfg, "output_logits_soft_cap", 0) or 0
    return cap * torch.tanh(lg / cap) if cap and cap > 0 else lg


def optimise_delta(model: HookedTransformer, ctxs: list, essence: tuple[torch.Tensor, int], layer: int, target: int,
                   v0: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    name = f"blocks.{layer}.hook_mlp_out"
    delta = torch.zeros(model.cfg.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    with torch.no_grad():
        base_lp = last_logits_grad(model, essence[0], []).log_softmax(-1)

    def hook_at(pos: int):
        def f(act: torch.Tensor, hook) -> torch.Tensor:
            act = act.clone()
            act[:, pos] = act[:, pos] + delta.to(act.dtype)
            return act
        return f

    def run_delta(toks: torch.Tensor, pos: int) -> torch.Tensor:
        """Last-position logits with delta added to the MLP output (to the input of ln2_post when the model has one)."""
        if not has_post_norm(model):
            return last_logits_grad(model, toks, [(name, hook_at(pos))])

        def pre(mod, args):
            x = args[0].clone()
            x[:, pos] = x[:, pos] + delta.to(x.dtype)
            return (x, *args[1:])

        h = model.blocks[layer].ln2_post.register_forward_pre_hook(pre)
        try:
            return last_logits_grad(model, toks, [])
        finally:
            h.remove()

    losses = []
    tgt = torch.tensor([target], device="cuda")
    # One backward per forward; weights were re-laid out (common.share_transposed_mlp_weights) so peak VRAM stays < 8 GB.
    with torch.enable_grad():
        for _ in range(STEPS):
            opt.zero_grad()
            ce_total = 0.0
            for toks, pos in ctxs:
                lg = run_delta(toks, pos)
                ce = F.cross_entropy(lg[None], tgt) / len(ctxs)
                ce.backward()
                ce_total += float(ce.detach())
            lg_e = run_delta(essence[0], essence[1]).log_softmax(-1)
            (KL_FACTOR * F.kl_div(lg_e, base_lp, log_target=True, reduction="sum")).backward()
            (WD * delta.pow(2).sum() / v0.pow(2).sum()).backward()
            opt.step()
            with torch.no_grad():
                nrm, cap = delta.norm(), CLAMP * v0.norm()
                if nrm > cap:
                    delta.mul_(cap / nrm)
            losses.append(ce_total)
    return delta.detach(), losses


def apply_edit(model: HookedTransformer, layer: int, k: torch.Tensor, delta: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """W_out <- W_out + (C^-1 k) outer delta / ((C^-1 k) . k); returns the original weight for restoring."""
    W = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    orig = W.data.clone()
    Cr = C + 1e-3 * C.diagonal().mean() * torch.eye(C.shape[0])
    u = torch.linalg.solve(Cr, k.cpu()).cuda()
    update = torch.outer(u, delta) / (u @ k)
    W.data = (W.data.float() + update).to(W.dtype)
    return orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-run", default="analysis/run_20261012_p3trace")
    ap.add_argument("--tag", default="p4edit")
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--facts", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    global MODEL_KEY
    MODEL_KEY = a.model
    a.facts = a.facts or MODELS[a.model]["facts"]
    all_facts = load_facts(a.facts)
    recs = {(r["id"], r["lang"]): r for r in json.loads((Path(a.trace_run) / "records.json").read_text(encoding="utf8"))}
    valid = [f for f in all_facts if all(recs[(f["id"], lg)]["collapsed"] and recs[(f["id"], lg)]["obj_ok"] and
                                         recs[(f["id"], lg)]["score_clean"] > recs[(f["id"], lg)]["score_corr"] for lg in LANGS)]
    edit_facts = select_p2_facts(valid)
    if a.limit:
        edit_facts = edit_facts[:a.limit]

    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)  # write-once
    model = load_model(a.model)
    share_transposed_mlp_weights(model)
    mid_layer = model.cfg.n_layers // 2
    tok = model.tokenizer
    windows = wiki_tokens(model)
    sents = wiki_sentences()
    ppl0 = perplexity(model, sents)
    # baseline probabilities of every valid fact's true first token (EN) for specificity
    base_en = {}
    for f in valid:
        p = f["lang"]["en"]["prompt"]
        t = first_obj_token(tok, p, f["lang"]["en"]["object"])
        base_en[f["id"]] = (t, float(last_probs(model, p)[t]))
    (out / "config.json").write_text(json.dumps({"trace_run": a.trace_run, "edit_ids": [f["id"] for f in edit_facts], "ppl_before": ppl0,
                                                 "steps": STEPS, "lr": LR, "kl_factor": KL_FACTOR, "wd": WD, "clamp": CLAMP,
                                                 "prefixes": PREFIXES, "cov_tokens": N_COV_TOKENS, "mid_layer": mid_layer,
                                                 "n_valid": len(valid)}, indent=1))
    rows = []
    for f in edit_facts:
        rng = random.Random(f["seed"])
        same_rel = [g for g in valid if g["relation"] == f["relation"] and g["lang"]["en"]["object"] != f["lang"]["en"]["object"]]
        new = rng.choice(same_rel)
        peak = recs[(f["id"], "en")]["crit_mlp_layer"]
        rand_layer = rng.choice([l for l in range(model.cfg.n_layers) if l not in (peak, mid_layer)])
        controls = rng.sample([g for g in valid if g["id"] != f["id"]], 20)
        L = {lg: {"prompt": f["lang"][lg]["prompt"], "old_t": first_obj_token(tok, f["lang"][lg]["prompt"], f["lang"][lg]["object"]),
                  "new_t": first_obj_token(tok, f["lang"][lg]["prompt"], new["lang"][lg]["object"])} for lg in LANGS}
        before = {lg: last_probs(model, L[lg]["prompt"]) for lg in LANGS}
        for cond, layer in (("traced", peak), ("mid", mid_layer), ("random", rand_layer)):
            en = f["lang"]["en"]
            ctxs = context_inputs(model, en["prompt"], en["subject"])
            ess_ids, ess_span = subject_span(tok, f"{en['subject']} is", en["subject"])
            essence = (torch.tensor([ess_ids], device="cuda"), ess_span[-1])
            k, v0 = get_key_and_value(model, ctxs, layer)
            delta, losses = optimise_delta(model, ctxs, essence, layer, L["en"]["new_t"], v0)
            C = covariance(model, layer, windows)
            orig = apply_edit(model, layer, k, delta, C)
            try:
                row = {"fact": f["id"], "relation": f["relation"], "new_fact": new["id"], "condition": cond, "layer": layer,
                       "delta_norm_ratio": float(delta.norm() / v0.norm()), "ce_first": losses[0], "ce_last": losses[-1]}
                for lg in LANGS:
                    after = last_probs(model, L[lg]["prompt"])
                    nt, ot = L[lg]["new_t"], L[lg]["old_t"]
                    row[lg] = {"p_new_before": float(before[lg][nt]), "p_new_after": float(after[nt]),
                               "dP": float(after[nt] - before[lg][nt]), "p_old_before": float(before[lg][ot]),
                               "p_old_after": float(after[ot]), "top1_is_new": bool(int(after.argmax()) == nt),
                               "top1": tok.decode(int(after.argmax()))}
                spec = []
                for g in controls:
                    t0, p0 = base_en[g["id"]]
                    spec.append(abs(float(last_probs(model, g["lang"]["en"]["prompt"])[t0]) - p0))
                row["specificity_max_abs_dP"] = max(spec)
                row["ppl_ratio"] = perplexity(model, sents) / ppl0
            finally:
                model.blocks[layer].mlp.W_out.data = orig
            rows.append(row)
            print(f"{f['id']} {cond} L{layer}: EN dP={row['en']['dP']:.3f} top1_new={row['en']['top1_is_new']} "
                  f"ES dP={row['es']['dP']:.3f} HI dP={row['hi']['dP']:.3f} spec={row['specificity_max_abs_dP']:.3f} "
                  f"ppl x{row['ppl_ratio']:.3f} |d|/|v0|={row['delta_norm_ratio']:.2f}", flush=True)
            torch.cuda.empty_cache()
    # restore check
    chk = float(last_probs(model, edit_facts[0]["lang"]["en"]["prompt"])[base_en[edit_facts[0]["id"]][0]])
    (out / "records.json").write_text(json.dumps({"rows": rows, "restore_check": {"p_after_all": chk, "p_baseline": base_en[edit_facts[0]["id"]][1]}},
                                                 ensure_ascii=False, indent=1), encoding="utf8")
    print("restore check", chk, base_en[edit_facts[0]["id"]][1], "peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
