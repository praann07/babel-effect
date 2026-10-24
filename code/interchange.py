"""A13 (E2): cross-language subject interchange (docs/HYPOTHESIS.md A13).

For each ordered pair (target t, source s) of EN/ES facts of the same relation and each direction, the residual stream (`hook_resid_post`) of the
target prompt at the last subject token (site A) or the final prompt token (site B) is replaced, one layer per batch row, by the source's value.
Conditions: same (source s in the target language), cross (s in the other language), self (t in the other language), ctrl (s in the other
language at the token before the subject). Readout: teacher-forced log-probability of the two answers in the TARGET language.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from causal_trace_v2 import SeqScorer, prompt_and_target
from common import load_model, subject_span

CONDS = ("same", "cross", "self", "ctrl")
SITES = ("A", "B")


class Prompt:
    """One fact in one language: ids, subject span, answer ids and cached clean residual stream at the three positions."""

    def __init__(self, model, fact: dict, lg: str) -> None:
        tok = model.tokenizer
        L = fact["lang"][lg]
        self.fact, self.lg, self.text, self.answer = fact, lg, L["prompt"], L["target"]
        self.p_ids, self.a_ids = prompt_and_target(tok, L["prompt"], L["target"])
        _, self.span = subject_span(tok, L["prompt"], L["subject"])
        self.P = len(self.p_ids)
        self.pos = {"A": self.span[-1], "B": self.P - 1, "pre": max(self.span[0] - 1, 0)}
        toks = torch.tensor([self.p_ids + self.a_ids], device="cuda")
        names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        self.res = {k: torch.stack([cache[n][0, p] for n in names]) for k, p in self.pos.items()}  # [L, d]
        del cache


def jaccard0(tok, pe: Prompt, ps: Prompt) -> float:
    def pieces(ids):
        out = set()
        for t in ids:
            s = tok.decode([int(t)]).strip().lower()
            if s:
                out.add(s)
        return out
    a, b = pieces([pe.p_ids[k] for k in pe.span]), pieces([ps.p_ids[k] for k in ps.span])
    return len(a & b) / max(len(a | b), 1)


@torch.no_grad()
def score(model, tgt: Prompt, ans_ids: list[int], patch: torch.Tensor | None, pos: int | None):
    """Per-layer (sum logP, top1-match) of ans_ids after the target prompt, with row l patched at `pos` by patch[l] (patch None: clean, 1 row)."""
    n_layers = model.cfg.n_layers
    toks = torch.tensor([tgt.p_ids + ans_ids], device="cuda")
    scorer = SeqScorer(model, ans_ids, tgt.P)
    grab: dict = {}

    def grab_hook(act, hook):
        grab["h"] = act
        return act
    if patch is None:
        model.run_with_hooks(toks, return_type=None, fwd_hooks=[("ln_final.hook_normalized", grab_hook)])
        lp, _, m = scorer(grab["h"])
        return lp.float().cpu().numpy(), m.cpu().numpy()
    fwd = []
    for l in range(n_layers):
        def h(act, hook, l=l):
            act[l, pos] = patch[l]
            return act
        fwd.append((f"blocks.{l}.hook_resid_post", h))
    model.run_with_hooks(toks.repeat(n_layers, 1), return_type=None, fwd_hooks=fwd + [("ln_final.hook_normalized", grab_hook)])
    lp, _, m = scorer(grab["h"])
    return lp.float().cpu().numpy(), m.cpu().numpy()


def pair_effect(model, T: Prompt, S_tl: Prompt, S_ol: Prompt, T_ol: Prompt, a_s: list[int], a_t: list[int]):
    """Delta[cond][site][L] and flip[cond][site][L] for target T. S_tl: source in target language; S_ol: source in other language; T_ol: target fact in other language."""
    base = {k: score(model, T, ids, None, None)[0][0] for k, ids in (("s", a_s), ("t", a_t))}
    d0 = base["s"] - base["t"]
    D = np.zeros((4, 2, model.cfg.n_layers), np.float32)
    F = np.zeros((4, 2, model.cfg.n_layers), bool)
    for si, site in enumerate(SITES):
        for ci, cond in enumerate(CONDS):
            if cond == "same":
                src, pos = S_tl.res[site], T.pos[site]
            elif cond == "cross":
                src, pos = S_ol.res[site], T.pos[site]
            elif cond == "self":
                src, pos = T_ol.res[site], T.pos[site]
            else:  # ctrl: token before the subject (site A) / the first prompt token (site B), from the other-language source
                src, pos = S_ol.res["pre"], T.pos[site]
            ls, ms = score(model, T, a_s, src, pos)
            lt, _ = score(model, T, a_t, src, pos)
            D[ci, si] = (ls - lt) - d0
            F[ci, si] = ms
    return D, F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma2"])
    ap.add_argument("--pilot", type=int, default=0, help="run only this many pairs (debug pilot, excluded from analysis)")
    ap.add_argument("--unit-tests", action="store_true")
    ap.add_argument("--tag", default="e2")
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    facts = json.loads(Path(f"data/v2/e2_facts_{a.model}.json").read_text(encoding="utf8"))["facts"]
    model = load_model(a.model)
    tok = model.tokenizer
    rng = np.random.default_rng(a.seed)
    P = {}
    for f in facts:
        try:
            P[(f["id"], "en")], P[(f["id"], "es")] = Prompt(model, f, "en"), Prompt(model, f, "es")
        except ValueError:
            continue
    ok = sorted({k[0] for k in P if (k[0], "en") in P and (k[0], "es") in P})
    facts = [f for f in facts if f["id"] in ok]
    byid = {f["id"]: f for f in facts}
    rel = {f["id"]: f["relation"] for f in facts}
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run.mkdir(parents=True, exist_ok=False)
    if a.unit_tests:
        tests = {}
        t = P[(ok[0], "es")]
        # identity: patching the target with its own clean activation must give Delta 0 (tolerance for bf16 batching)
        other = P[(ok[1], "es")]
        D, _ = pair_effect(model, t, t, t, t, other.a_ids, t.a_ids)
        tests["identity_max_abs_delta"] = float(np.abs(D[0]).max())
        # same-language swap at layers 0-2 on the first single-token-subject pairs of one relation
        singles = [i for i in ok if len(P[(i, "es")].span) == 1 and rel[i] == rel[ok[0]]][:6]
        big = []
        for i, j in zip(singles[:-1], singles[1:]):
            tt, ss = P[(i, "es")], P[(j, "es")]
            if tt.a_ids[0] == ss.a_ids[0]:
                continue
            D, _ = pair_effect(model, tt, ss, ss, tt, ss.a_ids, tt.a_ids)
            big.append(float(D[0, 0, :3].max()))
        tests["single_token_same_swap_max_delta_layers_0_2"] = big
        (run / "unit_tests.json").write_text(json.dumps(tests, indent=1))
        print(tests)
        return
    pairs = []
    for t in facts:
        pool = [s for s in facts if s["id"] != t["id"] and rel[s["id"]] == rel[t["id"]]]
        rng.shuffle(pool)
        n = 0
        for s in pool:
            for direction in ("en2es", "es2en"):
                lt = direction[-2:]
                if P[(s["id"], lt)].a_ids[0] == P[(t["id"], lt)].a_ids[0]:
                    break
            else:
                pairs.append((t["id"], s["id"]))
                n += 1
                if n == 2:
                    break
    if a.pilot:
        pairs = pairs[:a.pilot]
    meta, DD, FF = [], [], []
    for n_, (ti, si) in enumerate(pairs):
        for direction in ("en2es", "es2en"):
            sl, tl = direction[:2], direction[-2:]
            T, S_tl, S_ol, T_ol = P[(ti, tl)], P[(si, tl)], P[(si, sl)], P[(ti, sl)]
            if S_tl.a_ids[0] == T.a_ids[0]:
                continue
            D, F = pair_effect(model, T, S_tl, S_ol, T_ol, S_tl.a_ids, T.a_ids)
            DD.append(D); FF.append(F)
            meta.append({"target": ti, "source": si, "direction": direction, "relation": rel[ti], "jaccard_source": jaccard0(tok, P[(si, "en")], P[(si, "es")])})
        if (n_ + 1) % 10 == 0:
            print(n_ + 1, "/", len(pairs), flush=True)
    np.savez_compressed(run / "effects.npz", delta=np.stack(DD), flip=np.stack(FF))
    (run / "meta.json").write_text(json.dumps({"pairs": meta, "conds": CONDS, "sites": SITES, "n_layers": model.cfg.n_layers, "seed": a.seed,
                                               "n_facts": len(facts), "pilot": a.pilot}, indent=1))
    print("done", len(meta), "peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))


if __name__ == "__main__":
    main()
