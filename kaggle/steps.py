"""The Kaggle rerun, one notebook per model. Each step is a plan: models to download, models whose number format must be checked,
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


def _notebook(title: str, m: str, tpl: str, cands: str, events2: bool) -> dict:
    """One notebook = everything for one model, so notebooks need nothing from each other and can run side by side.
    Later stages (tracing, swap tests, edits, ...) are appended to the same job list."""
    return {
        "title": title,
        "models": [m],
        "calibrate": [m],
        "needs": ["data/v2/facts_candidates_seed.json", "data/v2/facts_candidates_extra.json", "code/select_templates.py"],
        "jobs": _facts_chain(m, tpl, cands, events2),
        "produces": [tpl, f"data/v2/facts_{m}.json", f"data/v2/facts_extra_{m}.json"],
    }


STEPS = {
    "babel_llama": _notebook("Llama-3.2-3B-Instruct: fact preparation (template choice, candidate facts, keep facts answered correctly in EN, ES and HI)",
                             "llama", "data/v2/templates.json", "data/v2/facts_candidates.json", False),
    "babel_gemma2": _notebook("Gemma-2-2B: fact preparation (template choice, candidate facts, keep facts answered correctly in EN, ES and HI)",
                              "gemma2", "data/v2/templates_gemma2.json", "data/v2/facts_candidates_gemma2.json", True),
}
