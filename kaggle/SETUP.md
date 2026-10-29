# Babel Effect on Kaggle: the from-scratch rerun

- One notebook per model in `kaggle/notebooks/` (`babel_llama`, `babel_gemma2`; the base model comes next). They need nothing from each other and can run at the same time.
- Each notebook fetches the code from GitHub (praann07/babel-effect, `main`) when it starts and logs the commit. The `babel_bundle.zip` dataset (`python kaggle/build_bundle.py`) is only a fallback if GitHub is unreachable.
- Needs: GPU T4 x2, Internet on, secret `HF_TOKEN` (the account must have accepted the Llama-3.2 and Gemma-2 licences).
- `kaggle/driver.py` does everything: install, checks, model download, per-model bf16/fp16 check (docs/HYPOTHESIS.md A25), all jobs over both GPUs with retries and resume, packaging to `babel_new_results.zip`.
- Later stages (tracing, swap tests, edits, ...) are added to the same notebooks, so a notebook never changes: only the code on GitHub does.
