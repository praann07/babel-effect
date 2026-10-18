"""Study v2 editing with a full layer sweep (docs/HYPOTHESIS.md A5).

For each edit fact and EVERY layer: ROME-style rank-1 update of the MLP down-projection (English prompt only), fitted on the
full target answer (mean token cross-entropy), then evaluated in EN/ES/HI with full-answer probabilities and greedy decoding,
plus specificity (20 control facts) and damage (perplexity ratio). The traced / mid / random conditions are read from the sweep.
Results are appended per edit to records.jsonl so an interrupted run can be resumed with --resume.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

import knowledge_edit as ke
from causal_trace_v2 import prompt_and_target, select_edit_facts
from common import LANGS, MODELS, load_facts, load_model, share_transposed_mlp_weights, subject_span
from matching import strict_match

PREFIXES = ke.PREFIXES
STEPS, LR, KL_FACTOR, WD, CLAMP = ke.STEPS, ke.LR, ke.KL_FACTOR, ke.WD, ke.CLAMP


def seq_logits(model: HookedTransformer, toks: torch.Tensor, hooks: list, start: int, n: int) -> torch.Tensor:
    """Logits [n, V] (fp32, soft-capped if the model has a cap) at positions start .. start+n-1, without the full-vocab copy."""
    grab: dict[str, torch.Tensor] = {}

    def g(act: torch.Tensor, hook) -> torch.Tensor:
        grab["h"] = act[0, start:start + n]
        return act

    model.run_with_hooks(toks, return_type=None, fwd_hooks=[*hooks, ("ln_final.hook_normalized", g)])
    lg = ((grab["h"] * model.ln_final.w).to(model.W_U.dtype) @ model.W_U).float()
    cap = getattr(model.cfg, "output_logits_soft_cap", 0) or 0
    return cap * torch.tanh(lg / cap) if cap and cap > 0 else lg


@torch.no_grad()
def seq_logprob(model: HookedTransformer, prompt: str, target: str) -> float:
    p_ids, t_ids = prompt_and_target(model.tokenizer, prompt, target)
    toks = torch.tensor([p_ids + t_ids], device="cuda")
    lg = seq_logits(model, toks, [], len(p_ids) - 1, len(t_ids))
    return float(lg.log_softmax(-1).gather(-1, torch.tensor(t_ids, device="cuda")[:, None]).sum())


@torch.no_grad()
def greedy(model: HookedTransformer, prompt: str, n: int) -> str:
    tok = model.tokenizer
    ids = tok(prompt)["input_ids"]
    p = len(ids)
    for _ in range(n):
        ids.append(int(model(torch.tensor([ids], device="cuda"))[0, -1].argmax()))
    return tok.decode(ids[p:])


def contexts(model: HookedTransformer, prompt: str, subject: str, target: str) -> list[tuple[torch.Tensor, int, int, torch.Tensor]]:
    """(tokens incl. target, subject-last position, answer start position, target ids) for each prefix context."""
    out = []
    for pre in PREFIXES:
        p_ids, t_ids = prompt_and_target(model.tokenizer, pre + prompt, target)
        _, span = subject_span(model.tokenizer, pre + prompt, subject)
        out.append((torch.tensor([p_ids + t_ids], device="cuda"), span[-1], len(p_ids) - 1, torch.tensor(t_ids, device="cuda")))
    return out


@torch.no_grad()
def key_and_value(model: HookedTransformer, ctxs: list, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    return ke.get_key_and_value(model, [(c[0], c[1]) for c in ctxs], layer)


def optimise_delta_seq(model: HookedTransformer, ctxs: list, essence: tuple[torch.Tensor, int], layer: int, v0: torch.Tensor):
    delta = torch.zeros(model.cfg.d_model, device="cuda", dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    post = ke.has_post_norm(model)
    name = f"blocks.{layer}.hook_mlp_out"

    def hooks_for(pos: int):
        if post:
            def pre(mod, args):
                x = args[0].clone()
                x[:, pos] = x[:, pos] + delta.to(x.dtype)
                return (x, *args[1:])
            return [], model.blocks[layer].ln2_post.register_forward_pre_hook(pre)

        def f(act: torch.Tensor, hook) -> torch.Tensor:
            act = act.clone()
            act[:, pos] = act[:, pos] + delta.to(act.dtype)
            return act
        return [(name, f)], None

    def run(toks: torch.Tensor, pos: int, start: int, n: int) -> torch.Tensor:
        hooks, handle = hooks_for(pos)
        try:
            return seq_logits(model, toks, hooks, start, n)
        finally:
            if handle is not None:
                handle.remove()

    with torch.no_grad():
        base_lp = seq_logits(model, essence[0], [], essence[0].shape[1] - 1, 1)[0].log_softmax(-1)
    first = last = 0.0
    with torch.enable_grad():
        for step in range(STEPS):
            opt.zero_grad()
            tot = 0.0
            for toks, pos, start, tgt in ctxs:
                lg = run(toks, pos, start, tgt.numel())
                ce = F.cross_entropy(lg, tgt) / len(ctxs)
                ce.backward()
                tot += float(ce.detach())
            lg_e = run(essence[0], essence[1], essence[0].shape[1] - 1, 1)[0].log_softmax(-1)
            (KL_FACTOR * F.kl_div(lg_e, base_lp, log_target=True, reduction="sum")).backward()
            (WD * delta.pow(2).sum() / v0.pow(2).sum()).backward()
            opt.step()
            with torch.no_grad():
                nrm, cap = delta.norm(), CLAMP * v0.norm()
                if nrm > cap:
                    delta.mul_(cap / nrm)
            first = tot if step == 0 else first
            last = tot
    return delta.detach(), first, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--trace-run", required=True, help="v2 trace run dir (for each fact's EN critical layer)")
    ap.add_argument("--n-edit", type=int, default=10)
    ap.add_argument("--layers", default=None, help="comma list; default all layers")
    ap.add_argument("--tag", default="p4sweep")
    ap.add_argument("--skip-first", type=int, default=0, help="skip the first N edit facts (already done in an earlier run)")
    ap.add_argument("--add-traced", action="store_true", help="also edit each fact at its own EN traced layer and at L//2 (A9 F7 rule)")
    ap.add_argument("--resume", default=None, help="existing run dir to continue (crash recovery only)")
    a = ap.parse_args()
    ke.MODEL_KEY = a.model
    spec = MODELS[a.model]
    facts = load_facts(spec["facts_v2"])
    recs = {(r["id"], r["lang"]): r for r in json.loads((Path(a.trace_run) / "records.json").read_text(encoding="utf8"))
            if r.get("phrasing") == 0 and "error" not in r}
    valid = [f for f in facts if all((f["id"], lg) in recs and recs[(f["id"], lg)]["collapsed"] for lg in LANGS)]
    edit_facts = select_edit_facts(valid, a.n_edit)[a.skip_first:]

    out = Path(a.resume) if a.resume else Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    out.mkdir(parents=True, exist_ok=bool(a.resume))
    done: set[tuple[str, int]] = set()
    if a.resume and (out / "records.jsonl").exists():
        done = {(r["fact"], r["layer"]) for r in map(json.loads, (out / "records.jsonl").read_text(encoding="utf8").splitlines())}

    model = load_model(a.model)
    share_transposed_mlp_weights(model)
    n_layers = model.cfg.n_layers
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(n_layers))
    tok = model.tokenizer
    windows = ke.wiki_tokens(model)
    sents = ke.wiki_sentences()
    ppl0 = ke.perplexity(model, sents)
    base = {f["id"]: seq_logprob(model, f["lang"]["en"]["prompts"][0], f["lang"]["en"]["target"]) for f in valid}
    if not a.resume:
        (out / "config.json").write_text(json.dumps(vars(a) | {"model_name": spec["tl_name"], "n_layers": n_layers, "layers": layers,
                                                                 "edit_ids": [f["id"] for f in edit_facts], "ppl_before": ppl0,
                                                                 "n_valid": len(valid), "steps": STEPS, "lr": LR, "kl": KL_FACTOR,
                                                                 "wd": WD, "clamp": CLAMP, "prefixes": PREFIXES}, indent=1))
    rng_all = {f["id"]: random.Random(f["seed"]) for f in edit_facts}
    for f in edit_facts:
        rng = rng_all[f["id"]]
        pool = [g for g in valid if g["relation"] == f["relation"] and g["lang"]["en"]["target"].lower() != f["lang"]["en"]["target"].lower()]
        new = rng.choice(pool)
        controls = rng.sample([g for g in valid if g["id"] != f["id"]], 20)
        en = f["lang"]["en"]
        ctxs = contexts(model, en["prompts"][0], en["subject"], new["lang"]["en"]["target"])
        e_ids, e_span = subject_span(tok, f"{en['subject']} is", en["subject"])
        essence = (torch.tensor([e_ids], device="cuda"), e_span[-1])
        before = {lg: seq_logprob(model, f["lang"][lg]["prompts"][0], new["lang"][lg]["target"]) for lg in LANGS}
        f_layers = sorted(set(layers) | ({recs[(f["id"], "en")]["crit_mlp_layer"], n_layers // 2} if a.add_traced else set()))
        for layer in f_layers:
            if (f["id"], layer) in done:
                continue
            k, v0 = key_and_value(model, ctxs, layer)
            delta, ce_first, ce_last = optimise_delta_seq(model, ctxs, essence, layer, v0)
            C = ke.covariance(model, layer, windows)
            orig = ke.apply_edit(model, layer, k, delta, C)
            try:
                row = {"fact": f["id"], "relation": f["relation"], "new_fact": new["id"], "layer": layer,
                       "is_traced": layer == recs[(f["id"], "en")]["crit_mlp_layer"], "delta_norm_ratio": float(delta.norm() / v0.norm()),
                       "ce_first": ce_first, "ce_last": ce_last}
                for lg in LANGS:
                    L = f["lang"][lg]
                    tgt = new["lang"][lg]["target"]
                    lp_after = seq_logprob(model, L["prompts"][0], tgt)
                    n_gen = len(prompt_and_target(tok, L["prompts"][0], tgt)[1]) + 2
                    text = greedy(model, L["prompts"][0], n_gen)
                    row[lg] = {"logp_new_before": before[lg], "logp_new_after": lp_after,
                               "dP": math.exp(lp_after) - math.exp(before[lg]),
                               "success": strict_match(text, [tgt]) is not None, "gen": text}
                spec_d = [abs(math.exp(seq_logprob(model, g["lang"]["en"]["prompts"][0], g["lang"]["en"]["target"])) - math.exp(base[g["id"]]))
                          for g in controls]
                row["specificity_max_abs_dP"] = max(spec_d)
                row["ppl_ratio"] = ke.perplexity(model, sents) / ppl0
            finally:
                model.blocks[layer].mlp.W_out.data = orig
            with open(out / "records.jsonl", "a", encoding="utf8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{f['id']} L{layer}: EN dP={row['en']['dP']:.3f} ok={row['en']['success']} ES dP={row['es']['dP']:.3f} "
                  f"HI dP={row['hi']['dP']:.3f} spec={row['specificity_max_abs_dP']:.3f} ppl x{row['ppl_ratio']:.3f}", flush=True)
            torch.cuda.empty_cache()
    print("peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
