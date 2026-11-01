# Part 7. Open items and next steps

## 7.1 When the Gemma Output zip arrives

1. Save it in `C:\Users\Som Praneeth\Desktop\reasearch\BABEL_EFFECT_ORGANIZED\08_waiting_for_gemma_output\` (and unpack a copy into `analysis\kaggle_output_<date>\` in the repo, write-once).
2. Read in this order: job list and failures in `driver.log`; fp16 decision; tracing; English-key edits; wording control; key similarity; key-mode comparison; layer map; graphs.
3. Check whether the Spanish key-similarity result replicates (bar 0.30, interval above 0), then Hindi, then partial rho with the edit layer controlled.
4. If a CPU analysis job failed (key similarity, key modes, graphs were never run on Gemma data), rerun it locally from the Output.
5. Register the verdicts in `docs/HYPOTHESIS.md` as an appendix (the registered text is not edited), add a Gemma section to the paper, rebuild the PDF, regenerate the manifest, rebuild this record.

## 7.2 Next models

Each new model needs: its own entry in `kaggle/steps.py`, a `driver.MODELS` entry, a config pack, a fingerprint, a registration in `docs/HYPOTHESIS.md`, a question-style smoke prompt checked on the CPU, and a dry run of every job type on the laptop GPU before the notebook goes to Kaggle.

| Order | Model | Why |
|---|---|---|
| 1 | Llama-3.2-3B base | Same family as the primary; tests whether instruction tuning matters for key similarity. |
| 2 | BLOOM-560M | Used by XME (2401.10521); small and comparable to published work. |
| 3 | mGPT-1.3B | Used by MEMLA (2406.11566). |
| 4 | BLOOM-1b7 | First pilot kept only 34 facts; needs a larger fact pool. |
| 5 | Qwen2.5-3B | Qwen appears in six published papers. |
| 6 | One 7 to 8B model | Only after a T4 test (both T4s or 4-bit; 4-bit changes the edited weights). |

## 7.3 Science that would raise the paper

- **Baselines on the same facts:** ROME, MEMIT, in-context; a Bi-ZsRE or MzsRE slice (no Hindi in those; check languages).
- **A causal test beyond the null shared-key edit:** for example editing a fact with a key made closer to the English key, pre-registered before it runs.
- **More relations** than geography (non-geography was untestable), registered first.
- **More languages** (two or three, one non-Latin non-Indic script).
- **Full reads** of the five closest papers: 2310.10378, 2601.17397, 2406.16416, 2504.09378, 2510.10280; then final related-work wording.
- **Tracing split across both GPUs** (GPU 1 idles about 44 to 73 minutes during tracing).

## 7.4 Housekeeping

| Item | Who |
|---|---|
| Hindi and Spanish native-speaker check of `data/audit/native_speaker_audit.csv` (362 facts) | You: a Hindi checker and a Spanish checker |
| OSF or Zenodo timestamp for the pre-registration | You: an account |
| Revoke the Hugging Face token | You |
| Confirm the Kaggle weekly GPU quota (about 30 hours is a guess) | You |
| Optional: rerun the English-key base edits on the laptop GPU (about 25 min, close other programs) to test the three analysis jobs on Gemma data | Needs your "restart" |
| Keep the one-folder copy in step with the repo | Assistant, on request |

## 7.5 How to revise this record

1. Edit the hand-written chapter in `paper/master/` (or the source document it copies).
2. Run `.venv/Scripts/python.exe paper/build_master.py`.
3. Check the PDF opens and the contents page matches.
4. Commit when you say so.
