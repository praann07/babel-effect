"""The Kaggle step: models to prepare, models whose number format must be checked, files that must already exist, and jobs (command, model,
dependencies, whether it needs a GPU). Jobs are handed to free GPUs in list order, so the order is the priority.

Command tokens expanded by kaggle/driver.py at launch time:
  {py}            the Python interpreter
  {tag:NAME}      NAME plus a unique suffix (write-once scripts refuse to reuse a folder)
  {run:PATTERN}   newest analysis/run_*_PATTERN folder
  {resume:PATTERN} "--resume <folder>" if an unfinished folder for PATTERN exists, else nothing
"""
from __future__ import annotations

# per model key: folder name of its weights, fact file, fixed mid layer (L // 2), the cap of the edit jobs. The cap of Gemma is the number of facts that qualify
# (relations with >= 10 kept facts; a test checks it), so its progress bars and time estimates count to the real total.
MODEL = {"llama": {"dir": "llama-3.2-3b-instruct", "facts": "data/v2/facts_llama.json", "mid": 14, "n_edit": 200},
         # Gemma-2-2B is a base model: "The capital of France is" gives " a" (0.20), so its smoke prompt is a question with the answer slot (" Paris" 0.83, checked on the CPU)
         "gemma2": {"dir": "gemma-2-2b", "facts": "data/v2/facts_gemma2.json", "mid": 13, "n_edit": 52,
                    "smoke": ["--prompt", "Question: What is the capital of France? Answer:"]}}


def job(name: str, cmd: list[str], deps: list[str] | None = None, gpu: bool = True, model: str = "llama") -> dict:
    return {"name": name, "model": model, "cmd": cmd, "deps": deps or [], "gpu": gpu}


def needs(key: str) -> list[str]:
    return [f"data/model_files/{MODEL[key]['dir']}.tar.gz", f"data/model_manifests/{MODEL[key]['dir']}.json", MODEL[key]["facts"]]


def main_jobs(key: str) -> list[dict]:
    """The jobs every model's step shares: model check, tracing of every kept fact, the three edit runs, the CPU analyses and the graphs."""
    m = MODEL[key]
    j = lambda name, cmd, deps=None, gpu=True: job(name, cmd, deps, gpu, key)  # noqa: E731
    trace = "{run:p3trace_v2r*_" + key + "}"  # the tracing run of the job "trace": facts and traced layers the edit jobs start from (made on Kaggle)
    # all valid facts of the relations with >= 10 kept facts, English-key edit at the traced layer and at the mid layer, other phrasings scored
    edit = ["{py}", "code/knowledge_edit_v2.py", "--model", key, "--trace-run", trace, "--n-edit", str(m["n_edit"]), "--skip-first", "0", "--layers", str(m["mid"]),
            "--add-traced", "--para"]
    # layer map: the first 40 facts at fixed layers, English key and mean_all key (the key cosines are written for both); one shard per GPU
    layers = ["{py}", "code/knowledge_edit_v2.py", "--model", key, "--trace-run", trace, "--n-edit", "40", "--layers", "2,6,10,18,22", "--para",
              "--key-modes", "en,mean_all"]
    mdir = f"models/{m['dir']}"
    return [
        j("smoke", ["{py}", "kaggle/smoke.py", "--model", key] + m.get("smoke", [])),
        j("tokenizers", ["{py}", "kaggle/tokenizer_compare.py", "--mirror", mdir, "--official", "{official}"], gpu=False),
        # tracing of every kept fact: the traced layers of the edit jobs
        j("trace", ["{py}", "code/causal_trace_v2.py", "--model", key, "--subset", "all", "--tag", "{tag:p3trace_v2}"], ["smoke"]),
        # shared-key edit (longest first so the two GPUs finish together), then the English-key base run, then the layer map in two shards
        j("edit_shared", edit + ["--key-modes", "mean_all,joint_all", "{resume:p4a28shared*_" + key + "}", "--tag", "{tag:p4a28shared}"], ["trace"]),
        j("edit_base", edit + ["{resume:p4a28base*_" + key + "}", "--tag", "{tag:p4a28base}"], ["trace"]),
        j("edit_layers0", layers + ["--shard", "0/2", "{resume:p4a28layers0*_" + key + "}", "--tag", "{tag:p4a28layers0}"], ["trace"]),
        j("edit_layers1", layers + ["--shard", "1/2", "{resume:p4a28layers1*_" + key + "}", "--tag", "{tag:p4a28layers1}"], ["trace"]),
        j("para_analysis", ["{py}", "code/para_analysis.py", "{run:p4a28base*_" + key + "}"], ["edit_base"], gpu=False),
        j("keymode_analysis", ["{py}", "code/keymode_analysis.py", "{run:p4a28shared*_" + key + "}", "--base", "{run:p4a28base*_" + key + "}"],
          ["edit_shared", "edit_base"], gpu=False),
        j("key_similarity", ["{py}", "code/key_similarity.py", "{run:p4a28base*_" + key + "}", "--keys", "{run:p4a28shared*_" + key + "}",
                             "--facts", m["facts"], "--model-dir", mdir], ["edit_shared", "edit_base"], gpu=False),
        j("layer_map", ["{py}", "code/layer_map.py", "{run:p4a28layers0*_" + key + "}", "{run:p4a28layers1*_" + key + "}"], ["edit_layers0", "edit_layers1"], gpu=False),
        j("graphs", ["{py}", "kaggle/make_plots.py"], ["para_analysis", "keymode_analysis", "key_similarity", "layer_map"], gpu=False),
    ]


def a23_jobs() -> list[dict]:
    """Facts outside geography. Llama only: with 93 Llama facts it was already untestable, and Gemma has 58."""
    return [
        job("a23_filter_es", ["{py}", "code/e2_filter.py", "--model", "llama", "--set", "a23", "--tag", "{tag:e2a23filter}"], ["smoke"]),
        job("a23_filter_hi", ["{py}", "code/e2_filter_hi.py", "--model", "llama", "--set", "a23", "--tag", "{tag:e2ha23filter}"], ["smoke"]),
        job("a23_es_run", ["{py}", "code/interchange.py", "--model", "llama", "--langs", "en,es", "--facts", "data/v2/e2a23_facts_llama.json", "--k-src", "2",
                           "--shuffle-pairs", "--max-pairs", "200", "{resume:e2a23_A*_llama}", "--tag", "{tag:e2a23_A}"], ["a23_filter_es"]),
        job("a23_hi_run", ["{py}", "code/interchange.py", "--model", "llama", "--langs", "en,hi", "--facts", "data/v2/e2ha23_facts_llama.json", "--span", "end",
                           "--directions", "en2hi,hi2en", "--k-src", "4", "--shuffle-pairs", "--max-pairs", "200", "{resume:e2ha23_A*_llama}", "--tag", "{tag:e2ha23_A}"],
            ["a23_filter_hi"]),
        job("a23_an_es", ["{py}", "code/e2_analysis.py", "--models", "llama", "--langs", "en,es", "--llama-run", "{run:e2a23_A*_llama}", "--tag", "{tag:e2a23_analysis}"],
            ["a23_es_run"], gpu=False),
        job("a23_an_es_lf", ["{py}", "code/e2_analysis.py", "--models", "llama", "--langs", "en,es", "--exclude-leak", "--llama-run", "{run:e2a23_A*_llama}",
                             "--tag", "{tag:e2a23_analysis_leakfree}"], ["a23_es_run"], gpu=False),
        job("a23_an_hi", ["{py}", "code/e2_analysis.py", "--models", "llama", "--langs", "en,hi", "--sites", "A", "--llama-run", "{run:e2ha23_A*_llama}",
                          "--tag", "{tag:e2ha23_analysis}"], ["a23_hi_run"], gpu=False),
        job("a23_an_hi_lf", ["{py}", "code/e2_analysis.py", "--models", "llama", "--langs", "en,hi", "--sites", "A", "--exclude-leak", "--llama-run",
                             "{run:e2ha23_A*_llama}", "--tag", "{tag:e2ha23_analysis_leakfree}"], ["a23_hi_run"], gpu=False),
    ]


NEEDS = needs("llama")  # kept for older callers

STEPS = {
    # about 10 minutes: proves the attached model is the laptop model and the GPUs work, before the long step
    "babel_llama_check": {
        "title": "Llama-3.2-3B-Instruct on Kaggle: check only (byte-identical weights, number format, a known answer, tokenizer comparison)",
        "models": ["llama"],
        "calibrate": ["llama"],
        "needs": needs("llama"),
        "jobs": main_jobs("llama")[:2],
        "produces": [],
    },
    "babel_llama": {
        "title": "Llama-3.2-3B-Instruct on Kaggle: tracing, edit transfer with wording control, shared-key edit, key similarity and layer map, non-geography facts",
        "models": ["llama"],
        "calibrate": ["llama"],
        "needs": needs("llama"),
        "jobs": main_jobs("llama") + a23_jobs(),
        "produces": ["data/v2/e2a23_facts_llama.json", "data/v2/e2ha23_facts_llama.json"],
    },
    # the same experiments on Gemma-2-2B (base), without the non-geography chain
    "babel_gemma2": {
        "title": "Gemma-2-2B on Kaggle: tracing, edit transfer with wording control, shared-key edit, key similarity and layer map",
        "models": ["gemma2"],
        "calibrate": ["gemma2"],
        "needs": needs("gemma2"),
        "jobs": main_jobs("gemma2"),
        "produces": [],
    },
}
