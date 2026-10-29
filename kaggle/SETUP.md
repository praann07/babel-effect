# Babel Effect on Kaggle: the from-scratch rerun

- Dataset: `kaggle/babel_bundle.zip` (code, tests, runner, seed candidate lists, wikitext; no laptop results). Upload once.
- One notebook per step in `kaggle/notebooks/`, run in order. Each step after the first takes the previous step's output as an input.
- Needs: GPU T4 x2, Internet on, secret `HF_TOKEN` (account must have accepted the Llama-3.2 and Gemma-2 licences).
- Each notebook runs `kaggle/driver.py`: install, checks, downloads, per-model bf16/fp16 check (docs/HYPOTHESIS.md A25), all jobs over every GPU with retries and resume, packaging to `babel_new_results.zip`.
- Rebuild after code changes: `.venv/Scripts/python kaggle/build_bundle.py`.
