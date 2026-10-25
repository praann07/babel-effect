"""Build paper/latex/main.tex (+ PDF) from paper_v2_draft.md. Prose comes from the markdown; tables 2-3 are generated from run artifacts.

Usage (from repo root):  python paper/build_tex.py        (needs pandoc and pdflatex on PATH)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
OUT = PAPER / "latex"
MD = (PAPER / "paper_v2_draft.md").read_text(encoding="utf8")
OVA = {"Llama-3.2-3B": "analysis/run_20261019_p3overlap_v2all_10191057/results.json", "Gemma-2-2B": "analysis/run_20261020_p3overlap_v2all_10200339/results.json"}
SW = {"Llama-3.2-3B": "analysis/run_20261015_p4sweep_llama/sweep_tables.json", "Gemma-2-2B": "analysis/run_20261019_p4sweep_gemma2/sweep_tables.json"}
F7 = "analysis/run_20261020_p4f7_llama/sweep_tables.json"
F7G = "analysis/run_20261020_p4f7_gemma2/sweep_tables.json"
E2 = "analysis/run_20261025_e2_analysis_A13/results.json"
FIGS = {  # section heading prefix -> [(file, width, caption)]
    "4.1": [("fig1_profiles.pdf", 0.95, "Median normalised MLP indirect effect per layer at each language's last subject token (band: interquartile range), for Llama-3.2-3B (left) and Gemma-2-2B (right)."),
            ("fig2_overlap.pdf", 0.95, "MLP-profile cosine between languages: observed median with 95\\% bootstrap CI (dots) and mean of the shuffled-fact permutation null (bars).")],
    "4.2": [("fig4_layer_hist.pdf", 0.95, "Distribution of the critical MLP layer (argmax of the profile) per language. Exploratory: Gemma shows modes at layers 4, 6 and 14.")],
    "4.3": [("fig3_hindi.pdf", 0.95, "Canonical-phrasing accuracy by answer length in tokens, per language and model.")],
    "4.4": [("fig5_sweep_llama.png", 1.0, "Llama-3.2-3B edit sweep (10 facts, every layer): English success, transfer ratio, specificity and perplexity ratio by layer."),
            ("fig6_sweep_gemma2.png", 1.0, "Gemma-2-2B edit sweep, same layout as Figure~\\ref{fig:sw1}.")],
}
SIM = [(">=", "≥"), ("<=", "≤"), ("+/-", "±"), ("->", "→"), (" x pooled", " × pooled"),
       ("English-Spanish overlap is the only pair that survives the relation-stratified null and correction in both models.", "Table~\\ref{tab:main} gives the pre-registered overlap results and Figures~\\ref{fig:fig1} and~\\ref{fig:fig2} show the profiles and the overlap. English-Spanish overlap is the only pair that survives the relation-stratified null and correction in both models."),
       ("Figure 4, exploratory", "Figure~\\ref{fig:fig4}, exploratory"), ("(Figures 5 and 6)", "(Figures~\\ref{fig:sw1} and~\\ref{fig:sw2})"),
       ("for answers of 7 or more tokens (Llama", "for answers of 7 or more tokens (Figure~\\ref{fig:fig3}; Llama"),
       ("*Conditions (A5).* At the fact's", "*Conditions (A5).* Table~\\ref{tab:edit} lists the three conditions. At the fact's"),
       ("On the 104 non-copyable facts the", "On the 104 non-copyable facts (Table~\\ref{tab:ext}) the")]


def pandoc(text: str) -> str:
    for a, b in SIM:
        text = text.replace(a, b)
    text = re.sub(r"(?<=[\s(\[/])-(?=\d)", "−", text)  # minus sign before numbers
    r = subprocess.run(["pandoc", "-f", "markdown-smart+pipe_tables", "-t", "latex", "--wrap=none"], input=text, capture_output=True, text=True, encoding="utf8", check=True)
    return r.stdout.replace("\\pandocbounded", "")


def esc(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def rd(art: str):
    return json.loads((ROOT / art).read_text(encoding="utf8"))


def table_overlap() -> str:
    rows = []
    for m, art in OVA.items():
        j = rd(art)
        for s, lab in (("primary_valid", "all valid facts"), ("valid_no_copyable", "non-copyable"), ("primary_valid_stratified_by_relation", "all valid, stratified null")):
            S = j["sets"][s]
            cells = [f"{S['pairs'][p]['mlp_cosine']['observed_median']:.3f} ({S['pairs'][p]['mlp_cosine']['z']:.2f})".replace("(-", "($-$") for p in ("en-es", "en-hi", "es-hi")]
            rows.append(f"{m} & {lab} & {S['n_facts']} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    return ("\\begin{table}[t]\\centering\\small\\begin{tabular}{llrccc}\\toprule\nmodel & set & $n$ & EN-ES & EN-HI & ES-HI \\\\\\midrule\n" + "\n".join(rows[:-1]) +
            "\n\\bottomrule\\end{tabular}\\caption{Extended overlap (with the added relations). Median MLP-profile cosine (z against the shuffled-fact null). Stratified: permutations within relation. "
            "Gemma gained one fact, so its non-copyable set ($n=29$) is below the pre-registered 50.}\\label{tab:ext}\\end{table}\n")


def table_edit() -> str:
    def g(t, c, k, i=None):
        v = t["conditions"][c][k]
        return v[i] if i is not None else v
    rows = []
    for lab, art in list(SW.items()) + [("Llama-3.2-3B, follow-up", F7), ("Gemma-2-2B, follow-up", F7G)]:
        t = rd(art)
        for c in ("traced", "mid", "random"):
            rows.append(f"{lab} & {dict(random='A9 layer').get(c, c) if 'follow-up' in lab else c} & {g(t, c, 'n')} & {g(t, c, 'en_success_rate'):.2f} & {g(t, c, 'transfer_es', 0):.3g} & {g(t, c, 'transfer_hi', 0):.3g} & {g(t, c, 'specificity', 0):.3g} \\\\")
        rows.append("\\addlinespace")
    return ("\\begin{table}[t]\\centering\\small\\begin{tabular}{llrcccc}\\toprule\nmodel & layer & $n$ & EN success & ES transfer & HI transfer & specificity \\\\\\midrule\n" + "\n".join(rows[:-1]) +
            "\n\\bottomrule\\end{tabular}\\caption{Edit conditions: the fact's English traced layer, $L/2$ and a third layer (medians; in the 10-fact sweeps this is a randomly drawn layer, in the follow-ups it is the layer chosen by the A9 rule: 21 for Llama, 3 for Gemma; transfer = $\\Delta P_{L2}/\\Delta P_{EN}$; specificity = maximum absolute probability change over 20 unrelated facts). "
            "The first two blocks are the 10-fact sweeps; the last two are the 19-20-fact follow-ups.}\\label{tab:edit}\\end{table}\n")


def table_e2() -> str:
    j = rd(E2)
    rows = []
    for m, lab in (("llama", "Llama-3.2-3B"), ("gemma2", "Gemma-2-2B")):
        for d, dl in (("en2es", r"EN$\to$ES"), ("es2en", r"ES$\to$EN")):
            A = j[m][d]["site_A"]; c = A["confirmatory"]; f = A["flip_at_lstar_confirmatory"]
            rows.append(f"{lab} & {dl} & {c['n_pairs']} & {A['gate']['l_star']} & {f['same']:.2f} & {f['cross']:.2f} & {c['I']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}] & {A['verdict'].split()[1]} \\\\")
        rows.append(r"\addlinespace")
    head = r"\begin{table}[t]\centering\small\begin{tabular}{llrcccll}\toprule" + "\n" + r"model & direction & pairs & $l^*$ & flip same & flip cross & $I$ [95\% CI] & verdict \\\midrule" + "\n"
    cap = ("\n" + r"\bottomrule\end{tabular}\caption{Interchange test (site A, confirmatory pairs whose source subject shares no token piece across languages). "
           r"$I$ = cross-language effect divided by same-language effect, summed over the layers where the same-language effect is at least half its maximum; "
           r"CI by bootstrap over source entities. Verdict by the pre-registered rule (lower bound $\geq 0.7$: shared).}\label{tab:e2}\end{table}" + "\n")
    return head + "\n".join(rows[:-1]) + cap


def figs(prefix: str) -> str:
    out = []
    for k, (f, w, cap) in enumerate(FIGS.get(prefix, [])):
        lab = {"fig5_sweep_llama.png": "fig:sw1", "fig6_sweep_gemma2.png": "fig:sw2"}.get(f, "fig:" + f.split("_")[0])
        out.append(f"\\begin{{figure}}[t]\\centering\\includegraphics[width={w}\\linewidth]{{figures/{f}}}\\caption{{{cap}}}\\label{{{lab}}}\\end{{figure}}\n")
    return "".join(out)


def table_main() -> str:
    return r"""\begin{table}[t]\centering\small\begin{tabular}{llcccc}\toprule
model & pair & MLP cosine [95\% CI] & null mean & $z$ pooled & $z$ within relation \\\midrule
Llama ($n=92$) & EN-ES & 0.627 [0.557, 0.702] & 0.519 & 4.22 & 3.29 \\
 & EN-HI & 0.509 [0.335, 0.594] & 0.429 & 2.45 & 1.79 \\
 & ES-HI & 0.423 [0.313, 0.529] & 0.328 & 2.47 & 1.98 \\\addlinespace
Gemma ($n=58$) & EN-ES & 0.464 [0.329, 0.542] & 0.292 & 4.81 & 3.85 \\
 & EN-HI & 0.226 [0.168, 0.283] & 0.136 & 2.19 & 1.68 \\
 & ES-HI & 0.137 [0.014, 0.267] & 0.134 & 0.08 & $-0.45$ \\\bottomrule
\end{tabular}\caption{Pre-registered overlap analysis: median MLP-profile cosine across facts, shuffled-fact null mean, and $z$ against the pooled and the within-relation null.}\label{tab:main}\end{table}
"""


def bib(lines: list[str]) -> str:
    items = []
    for i, l in enumerate(lines):
        t = esc(l.lstrip("- ").strip())
        t = re.sub(r"arXiv:(\d{4}\.\d{4,5})", r"\\href{https://arxiv.org/abs/\1}{arXiv:\1}", t)
        items.append(f"\\bibitem{{r{i}}} {t}")
    return "\\begin{thebibliography}{99}\\small\n" + "\n".join(items) + "\n\\end{thebibliography}\n"


def main() -> None:
    abstract = re.search(r"## Abstract\n\n(.+?)\n\n## 1\.", MD, re.S).group(1)
    refs = [l for l in MD.split("## References")[1].splitlines() if l.startswith("- ")]
    body = MD.split("## 1. Introduction")[1].split("## References")[0]
    chunks = re.split(r"^(#{2,3}) (.+)$", "## 1. Introduction" + body, flags=re.M)  # ['', '##', title, text, ...]
    tex = []
    for i in range(1, len(chunks), 3):
        lvl, title, text = chunks[i], chunks[i + 1], chunks[i + 2]
        num = re.match(r"(\d+(?:\.\d+)?)\.? ", title).group(1)
        title = re.sub(r"^\d+(?:\.\d+)?\.? ", "", title)
        title = re.sub(r"\s*\*\*\[PENDING.*?\]\*\*", "", title)
        text = "\n".join(l for l in text.splitlines() if not l.startswith("|"))
        tex.append(("\\section{" if lvl == "##" else "\\subsection{") + pandoc(title).strip().rstrip("\n") + "}\n" + pandoc(text))
        if num == "4.1":
            tex.append(table_main() + figs("4.1") + table_overlap())
        elif num == "4.4":
            tex.append(figs("4.4") + table_edit())
        elif num == "4.7":
            tex.append(table_e2())
        elif num in FIGS:
            tex.append(figs(num))
    doc = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{mathptmx}\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,microtype,xcolor,caption}
\usepackage[hidelinks]{hyperref}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\emergencystretch=2em\captionsetup{font=small}
\title{The Babel Effect: Cross-Lingual Causal Profiles of Factual Recall in Two Small Language Models}
\author{Anonymous authors}
\date{Draft v2}
\begin{document}\maketitle
\begin{abstract}
""" + pandoc(abstract) + "\\end{abstract}\n" + "\n".join(tex) + bib(refs) + "\\end{document}\n"
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for f in (PAPER / "figures").glob("fig*.*"):
        if f.suffix in (".pdf", ".png"):
            shutil.copy(f, OUT / "figures" / f.name)
    (OUT / "main.tex").write_text(doc, encoding="utf8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT, capture_output=True, text=True, encoding="utf8", errors="replace")
    print("pdflatex exit", r.returncode)
    if r.returncode:
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("!") or "Error" in l)[:1500])


if __name__ == "__main__":
    main()
