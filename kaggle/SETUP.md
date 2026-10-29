# Babel Effect on Kaggle

Inputs: the `babel_bundle.zip` dataset (private) and the `HF_TOKEN` secret (Hugging Face read token, Llama-3.2 licence accepted). Accelerator: GPU T4 x2, Internet on.
Notebook: `babel_kaggle.ipynb`. Run it (Save & Run All); it needs no edits.

What `kaggle/driver.py` does, in its own process:
1. installs the few packages needed (Kaggle's numpy/scipy/torch untouched), checks the GPUs, data files and unit tests;
2. downloads Llama-3.2-3B-Instruct (retries);
3. runs a 2-fact edit in bf16 and fp16 on both GPUs to pick the number format (rule in docs/HYPOTHESIS.md A25);
4. runs the paraphrase check, the new-facts run and the shared-edit test as 10 jobs over all GPUs (each GPU always busy; failed jobs are retried and resumed);
5. merges the shards, runs the analyses, writes `results/` and `babel_new_results.zip` (always, even after a failure or the time limit).

To resume after the time limit: attach this notebook's output as an input and run it again. Finished jobs are skipped.
