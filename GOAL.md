# GOAL: The Babel Effect

**Goal (set by the user, 2026-10-29):** win. Get the strongest possible results, better than the current study, aimed at a Q1/Q2/Q3 journal (venue decided later). The work comes first; the paper is written as well as possible after.

**Question:** are facts stored in language-specific circuits or in a shared space? Models: Llama-3.2-3B-Instruct, Gemma-2-2B, Llama-3.2-3B base. Languages: English, Spanish, Hindi.

**Where we are (2026-10-29):** careful measurement-and-audit paper. Swap ratio (1.0 = fully shared): EN<->ES 0.90-1.21, EN<->HI 0.72-0.93 (whole-subject patch), base-model replicate within 0.15 in all four directions. Edits: layer 14 carries to Spanish (median 0.75) but not to Hindi (0.10). 1,941 numbers verified in `results/manifest.json`. Honest level today: Q3-ready, Q2 borderline, Q1 not yet. Detail: `docs/ROADMAP_Q1.md`, `docs/HYPOTHESIS.md`, `docs/EXPERIMENT_LOG.md`.

## Plan to raise the level

| # | What | Why it matters | Cost | Status |
|---|---|---|---|---|
| 1 | **A21 paraphrase control**: does an English edit carry to other English phrasings, and is Spanish as good as that? | Removes "the edit is brittle everywhere". | about 3 GPU h | pre-registered, code ready (`scripts_a21.sh`); needs the user's laptop window and "go" |
| 2 | **More fact types**: non-geography relations (language spoken, profession, currency, birth year) and 100+ facts per language. | "Geography only" is the first reviewer complaint. | facts written by the agent, then about 4 GPU h per model | CPU part in progress (A23) |
| 3 | **Native-speaker check** of the Hindi and Spanish facts. | Hand-written facts are a validity hole. | a friend or paid reviewer, a few hours | audit sheet being prepared; needs the user to find a checker |
| 4 | **Bigger models**: 7-9B (Llama-3.1-8B, Gemma-2-9B) in 4-bit on the 8 GB card. | Comparable papers use 7B or larger. | about 10-20 GPU h; first check that 4-bit patching works | not started; free cloud GPU (Kaggle/Colab) needs the user's yes |
| 5 | **New edit method (the Q1 hook)**: build the edit key from the shared representation (averaged over EN and ES subjects) and test whether Spanish and Hindi transfer jumps. | A positive method result, not only a measurement. | about 10 GPU h, pre-registered first | after 2 and 4 |
| 6 | **More languages**: add 2-3, including one non-Latin, non-Indic script. | Makes "shared across languages" general. | about 4 GPU h per language | not started |
| 7 | **Standard benchmark** slice (mLAMA or Bi-ZsRE). | Direct comparison with other papers. | about 3 GPU h | not started |
| 8 | **Paper repairs** (reconcile earlier results, n and CIs, missing citations, rebuild PDF). | Removes cheap rejection reasons. | CPU only | done 2026-10-29 except where noted in `docs/LOG.md` |

**Order:** 1 first, in parallel with 3 and 8. Then 2, 4, 6, 7. Do 5 last, once the data in 2 and 4 is solid.

**Needed from the user:** a laptop window for each GPU step (and a "go" each time); someone to check the Hindi and Spanish facts; a yes or no on a free cloud GPU for the 7-9B runs.

**Honest limit:** items 1-4 and 8 give a strong Q2 case. Q1 only works if item 5 gives a clear win; it is pre-registered so a null result still counts.

## Standing rules
$0 spend; one 8 GB GPU; pre-register in `docs/HYPOTHESIS.md` before any run; write-once run dirs; every number through `results/manifest.json`; never touch `..\RRAP`; ask before using the GPU/laptop or any outside GPU; quote paths (they contain spaces).
