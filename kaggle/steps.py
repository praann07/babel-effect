"""The Kaggle rerun, one step per notebook. Each step is a plan: models to download, models whose number format must be checked,
files that must already exist, and jobs (command, model, dependencies, whether it needs a GPU).

Command tokens expanded by kaggle/driver.py at launch time:
  {py}            the Python interpreter
  {tag:NAME}      NAME plus a unique suffix (write-once scripts refuse to reuse a folder)
  {run:PATTERN}   newest analysis/run_*_PATTERN folder made on Kaggle (the dataset carries no laptop runs, so nothing can mix)
  {resume:PATTERN} "--resume <folder>" if an unfinished folder for PATTERN exists, else nothing
All Kaggle tags start with "kg" so they never look like laptop runs.
"""
from __future__ import annotations


def _facts_chain(m: str, tpl: str, cands_out: str, events2: bool) -> list[dict]:
    p = m[0].upper()  # L / G
    return [
        {"name": f"{p}_templates", "model": m, "cmd": ["{py}", "code/select_templates.py", "--model", m, "--cands", "data/v2/facts_candidates_seed.json", "--tag", "{tag:kgtemplates}"]},
        {"name": f"{p}_templates_extra", "model": m, "deps": [f"{p}_templates"],
         "cmd": ["{py}", "code/select_templates.py", "--model", m, "--relations", "landmark_country,city_country", "--cands", "data/v2/facts_candidates_extra.json", "--tag", "{tag:kgtemplates}"]},
        {"name": f"{p}_candidates", "model": m, "gpu": False, "deps": [f"{p}_templates_extra"],
         "cmd": ["{py}", "code/build_facts_v2.py", "--templates", tpl, "--out", cands_out] + (["--with-events2"] if events2 else [])},
        {"name": f"{p}_candidates_extra", "model": m, "gpu": False, "deps": [f"{p}_templates_extra"],
         "cmd": ["{py}", "code/build_facts_extra.py", "--templates", tpl, "--out", f"data/v2/facts_candidates_extra_{m}.json"]},
        {"name": f"{p}_filter", "model": m, "deps": [f"{p}_candidates"],
         "cmd": ["{py}", "code/filter_facts_v2.py", "--model", m, "--candidates", cands_out, "--tag", "{tag:kgfilter_" + m + "}"]},
        {"name": f"{p}_filter_extra", "model": m, "deps": [f"{p}_candidates_extra"],
         "cmd": ["{py}", "code/filter_facts_v2.py", "--model", m, "--candidates", f"data/v2/facts_candidates_extra_{m}.json",
                 "--facts-out", f"data/v2/facts_extra_{m}.json", "--tag", "{tag:kgfilterextra_" + m + "}"]},
    ]


STEPS = {
    "s01_facts": {
        "title": "Step 1: fact preparation (template choice, candidate facts, keep facts answered correctly in EN, ES and HI) for Llama-3.2-3B-Instruct and Gemma-2-2B",
        "models": ["llama", "gemma2"],
        "calibrate": ["llama", "gemma2"],
        "needs": ["data/v2/facts_candidates_seed.json", "data/v2/facts_candidates_extra.json", "code/select_templates.py"],
        "jobs": _facts_chain("llama", "data/v2/templates.json", "data/v2/facts_candidates.json", False)
        + _facts_chain("gemma2", "data/v2/templates_gemma2.json", "data/v2/facts_candidates_gemma2.json", True),
        "produces": ["data/v2/templates.json", "data/v2/templates_gemma2.json", "data/v2/facts_llama.json", "data/v2/facts_gemma2.json",
                     "data/v2/facts_extra_llama.json", "data/v2/facts_extra_gemma2.json"],
    },
}
