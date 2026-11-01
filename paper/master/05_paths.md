# Part 9. Where everything is

All paths are on the laptop unless marked GitHub. Quote them in PowerShell and bash: the folder names contain spaces.

## 9.1 Main places

| What | Path |
|---|---|
| **Repository (source of truth)** | `C:\Users\Som Praneeth\Desktop\reasearch\babel-effect` |
| GitHub | https://github.com/praann07/babel-effect (branch `main`) |
| **One-folder copy of everything so far** (outside git, not on GitHub) | `C:\Users\Som Praneeth\Desktop\reasearch\BABEL_EFFECT_ORGANIZED\` (README.md, 01_STORY.md, notebooks, consoles, Llama Kaggle results, dry runs, paper, docs) |
| **This master record, PDF** | `paper\The_Babel_Effect_MASTER_RECORD.pdf` (in the repo) |
| Master record source | `paper\master\*.md` and `paper\build_master.py` |
| The short paper | `paper\paper_v2_draft.md`, PDF `paper\The_Babel_Effect_draft_v2.pdf`, LaTeX `paper\latex\main.tex` |
| Master goal file | `GOAL.md` |
| Pre-registrations | `docs\HYPOTHESIS.md` |
| Decision log / experiment log | `docs\LOG.md` / `docs\EXPERIMENT_LOG.md` |
| Next steps / Q1 roadmap | `docs\NEXT.md` / `docs\ROADMAP_Q1.md` |
| Results manifest (every number to a file) | `results\manifest.json` |
| Facts | `data\facts.json`, `data\v2\*.json`, audit sheet `data\audit\native_speaker_audit.csv` |
| Model manifests (tensor-by-tensor check against the laptop copy) | `data\model_manifests\`, small config packs `data\model_files\` |
| Local model weights (not in git) | `models\` |

## 9.2 Kaggle notebooks (self-contained; code packed inside)

| Notebook | Code id | What it needs on Kaggle |
|---|---|---|
| `kaggle\babel_gemma2.ipynb` | `bundle-e186cf5d5e1b` | GPU T4 x2, Internet on, Kaggle Model **Gemma 2 / Transformers / gemma-2-2b** (Google's own). Do not attach an old Output. |
| `kaggle\babel_llama.ipynb` | `bundle-4c26396318b4` | GPU T4 x2, Internet on, Kaggle Model **Meta Llama 3.2 / Transformers / 3b-instruct**. Do not attach an old Output. |

Both are in the repo, in the one-folder copy under `02_notebooks_to_run\`, and carry only their own model's files (Gemma 10.9 MB, Llama 8.3 MB).

## 9.3 Kaggle results and consoles

| What | Path |
|---|---|
| Llama final run (348 min, 86 facts): raw Output | `analysis\kaggle_output_20261031\` (copy: `BABEL_EFFECT_ORGANIZED\04_kaggle_results_llama\kaggle_output_20261031_A28_run`) |
| Llama analyses (key similarity, layer map, wording control, key modes) | `analysis\run_20261031_key_similarity_A28`, `..._layer_map_A28`, `..._para_analysis_A28`, `..._keymode_analysis_A28` |
| Llama raw edit runs | `analysis\run_20261030_p4a28*_llama` |
| Llama first pass (28 facts) | `analysis\run_20261030_key_similarity_A26_firstpass`, `..._para_analysis_A21_firstpass`, `analysis\run_20261030_keymode_analysis_A24` |
| Gemma failed run 2 (shared-key, `bundle-b7d6962e96d7`), console notebook you sent | `C:\Users\Som Praneeth\Downloads\notebook8a97a9a4df.ipynb` (copy and console text in `BABEL_EFFECT_ORGANIZED\03_kaggle_run_consoles\gemma2_run2_FAILED_shared_key_bundle-b7d6962e96d7\`) |
| Gemma run 4 (running, `bundle-e186cf5d5e1b`), console notebook you sent | `C:\Users\Som Praneeth\Downloads\notebook0476dd6f1a.ipynb` (copy and snapshot in `...\gemma2_run4_RUNNING_bundle-e186cf5d5e1b\`) |
| **Where the Gemma Output zip goes** | `BABEL_EFFECT_ORGANIZED\08_waiting_for_gemma_output\` (and, once read, `analysis\kaggle_output_<date>\` in the repo) |
| Laptop dry runs for Gemma | `analysis\run_20261031_p4a30dry*_gemma2`, copies and logs in `BABEL_EFFECT_ORGANIZED\05_laptop_dry_runs_gemma\` |

## 9.4 Commands (from the repo root, `.venv\Scripts\python.exe` is the Python with the libraries)

| Task | Command |
|---|---|
| Run the tests (69 pass, 1 skipped) | `.venv/Scripts/python.exe -m unittest discover -s tests -q` |
| Rebuild both notebooks after any code change | `.venv/Scripts/python.exe kaggle/make_notebook.py` |
| Check inside an unpacked notebook | unpack with the scratchpad script `unpack_nb.py`, then `PYTHONPATH=code python -m unittest discover -s tests` in the unpacked folder |
| Rebuild / verify the results manifest | `.venv/Scripts/python.exe code/make_manifest.py build` / `verify` |
| Rebuild the short paper PDF | `.venv/Scripts/python.exe paper/build_tex.py` (needs pandoc and pdflatex) |
| **Rebuild this master record** | `.venv/Scripts/python.exe paper/build_master.py` (needs pandoc and Microsoft Edge) |
| Key-similarity test on a Kaggle Output | `code/key_similarity.py` (CPU, minutes) |
| Layer map | `code/layer_map.py` (CPU) |

## 9.5 Agent side (the assistant's own notes)

| What | Path |
|---|---|
| Assistant memory (rules, state) | `C:\Users\Som Praneeth\.claude\projects\C--Users-Som-Praneeth-Desktop-reasearch\memory\` |
| Hive handoff | `C:\Users\Som Praneeth\HarnessAgents\hive\agents\rrap-research-assistant-mv52oi0t\handoff.md` |
| Full chat transcript | `C:\Users\Som Praneeth\.claude\projects\C--Users-Som-Praneeth-Desktop-reasearch\a63987bb-a70f-4518-9722-baa1fe5bdd7c.jsonl` |

## 9.6 Reading order for a new person

1. This Part 1 and Part 6 (goal and numbers).
2. Part 5, the paper.
3. Part 3, the problems, to see what to distrust.
4. Part 8, the pre-registrations, to see what was promised before each run.
5. Part 10, the logs, for day-by-day detail.

The rest of this part is generated from the repository each time the record is rebuilt: the top-level layout, the list of run folders, and the git history.
