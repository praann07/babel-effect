#!/usr/bin/env python
"""One script that does the whole Kaggle job: set up, check, run all GPUs flat out, merge, analyse, package.

Only the standard library is imported until the packages are installed, so a numpy/torch mismatch inside the notebook kernel cannot happen.
Every long step is a separate process with its own log; failed steps are retried and resumed; results are always packaged at the end.
Environment: HF_TOKEN (Hugging Face read token), HOURS (time budget, default 10.5), KAGGLE_WORK / KAGGLE_INPUT (for local tests).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()
HOURS = float(os.environ.get("HOURS", "10.5"))
DEADLINE = T0 + HOURS * 3600
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working/babel-effect"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = WORK.parent  # /kaggle/working: whatever is here when the notebook ends is saved as output
MODEL = "llama"
HF_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = Path(os.environ.get("KAGGLE_MODELS", "/tmp/models")) / "llama-3.2-3b-instruct"
PY = sys.executable
PACKAGES = ["transformer-lens==3.9.0", "transformers==5.19.0", "accelerate==1.15.0", "jaxtyping", "einops", "fancy-einsum", "better-abc", "beartype"]
TRACE_RUN = "analysis/run_20261015_p3trace_v2_llama"
NEEDED = ["code/common.py", "data/v2/facts_llama.json", f"{TRACE_RUN}/records.json", "data/v2/facts_candidates_a23.json",
          "data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet", "data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet",
          "analysis/run_20261024_p3seeds_10240109_llama", "analysis/run_20261020_p4f7_llama"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}"
    print(line, flush=True)
    try:
        (WORK / "analysis" / "_logs").mkdir(parents=True, exist_ok=True)
        with open(WORK / "analysis" / "_logs" / "driver.log", "a", encoding="utf8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def sh(cmd: list[str], env: dict | None = None, cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- set-up
def place_code() -> None:
    if (WORK / "code" / "common.py").exists():
        return
    zips = glob.glob(str(INPUT / "**" / "babel_bundle.zip"), recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
        return
    hits = glob.glob(str(INPUT / "**" / "code" / "common.py"), recursive=True)
    if not hits:
        raise SystemExit("FATAL: the babel-bundle dataset is not attached (no code/common.py under /kaggle/input)")
    shutil.copytree(Path(hits[0]).parent.parent, WORK, dirs_exist_ok=True)


def import_previous() -> set[str]:
    """Bring in run folders, new fact files and done-markers saved by an earlier session (attached as an input)."""
    done: set[str] = set()
    cands = [Path(p) for p in glob.glob(str(INPUT / "**" / "results"), recursive=True) if Path(p).is_dir()]
    for z in glob.glob(str(INPUT / "**" / "babel_new_results.zip"), recursive=True):
        tmp = Path("/tmp/prev_results") / hashlib.md5(z.encode()).hexdigest()[:8]
        if not tmp.exists():
            zipfile.ZipFile(z).extractall(tmp)
        cands.append(tmp)
    for r in cands:
        for d in sorted(r.glob("run_*")) + sorted(r.glob("figs_*")):
            dst = WORK / "analysis" / d.name
            if d.is_dir() and not dst.exists():
                shutil.copytree(d, dst)
        for f in (r / "data_v2").glob("*.json") if (r / "data_v2").exists() else []:
            shutil.copy2(f, WORK / "data" / "v2" / f.name)
        if (r / "kaggle_done.json").exists():
            done |= set(json.loads((r / "kaggle_done.json").read_text(encoding="utf8")))
        if (r / "dtype_decision.json").exists():
            (WORK / "analysis").mkdir(exist_ok=True)
            shutil.copy2(r / "dtype_decision.json", WORK / "analysis" / "kaggle_dtype_decision.json")
    if done:
        log(f"imported earlier session: done jobs = {sorted(done)}")
    return done


def install_packages() -> None:
    probe = "import numpy, scipy, pandas, torch, transformers, transformer_lens as t; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| transformer_lens', t.__version__)"
    p = sh([PY, "-c", probe])
    if p.returncode == 0 and "transformer_lens 3.9.0" in p.stdout:
        log("packages already fine: " + p.stdout.strip())
        return
    for attempt in (1, 2):
        r = sh([PY, "-m", "pip", "install", "-q"] + PACKAGES)
        log(f"pip install attempt {attempt}: exit {r.returncode} {r.stderr[-300:].strip()}")
        p = sh([PY, "-c", probe])
        if p.returncode == 0:
            log("packages: " + p.stdout.strip())
            return
        log("import check failed: " + p.stderr[-800:])
    raise SystemExit("FATAL: packages do not import after install (see the import check above)")


def gpu_list() -> list[dict]:
    r = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    if r.returncode:
        return []
    out = []
    for ln in r.stdout.strip().splitlines():
        i, name, tot, used, util = [x.strip() for x in ln.split(",")]
        out.append({"index": int(i), "name": name, "total_mib": int(tot), "used_mib": int(used), "util": int(util)})
    return out


def download_model(token: str | None) -> None:
    ok = (MODEL_DIR / "config.json").exists() and sum(f.stat().st_size for f in MODEL_DIR.glob("*.safetensors")) > 5e9
    if not ok:
        from huggingface_hub import snapshot_download
        for attempt in range(1, 5):
            try:
                snapshot_download(HF_ID, local_dir=str(MODEL_DIR), token=token, ignore_patterns=["original/*"])
                break
            except Exception as e:  # network blips, rate limits
                log(f"download attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
                if attempt == 4:
                    raise SystemExit("FATAL: could not download the model. Is HF_TOKEN set, and the Llama-3.2 licence accepted on huggingface.co?")
                time.sleep(20 * attempt)
    size = sum(f.stat().st_size for f in MODEL_DIR.glob("*.safetensors"))
    if size < 5e9:
        raise SystemExit(f"FATAL: model files look incomplete ({size / 1e9:.1f} GB)")
    link = WORK / "models"
    if not link.exists():
        os.symlink(MODEL_DIR.parent, link)
    log(f"model ready: {size / 1e9:.1f} GB in {MODEL_DIR}")


def preflight_files() -> None:
    missing = [p for p in NEEDED if not (WORK / p).exists()]
    if missing:
        raise SystemExit(f"FATAL: bundle is missing {missing}")


def run_tests() -> None:
    env = dict(os.environ, PYTHONPATH="code")
    r = sh([PY, "-m", "unittest", "discover", "-s", "tests"], env=env, cwd=WORK)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    log("unit tests: " + " | ".join(tail))
    if r.returncode:
        raise SystemExit("FATAL: unit tests fail in this environment; results would not be trustworthy")


# ---------------------------------------------------------------- number format
def calibrate_dtype(gpus: list[dict]) -> str:
    """Run the same 2-fact edit in bf16 and fp16 (one GPU each when there are two) and keep fp16 only if it is >= 1.5x faster
    and gives no gross failure (NaN, lost edit success, optimiser stuck). The decision is saved and recorded in every run config."""
    f = WORK / "analysis" / "kaggle_dtype_decision.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf8"))
        log(f"number format from earlier session: {d['dtype']} ({d['reason']})")
        return d["dtype"]
    import threading
    base = [PY, "code/knowledge_edit_v2.py", "--model", MODEL, "--trace-run", TRACE_RUN, "--n-edit", "2", "--layers", "14"]
    tags = {"bfloat16": "calbf16", "float16": "calfp16"}
    res: dict = {}

    def one(k: int, dt_name: str) -> None:
        gpu = str(gpus[min(k, len(gpus) - 1)]["index"])
        env = dict(os.environ, PYTHONPATH="code", PYTHONIOENCODING="utf8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu, BABEL_DTYPE=dt_name)
        t = time.time()
        with open(WORK / "analysis" / "_logs" / f"calibrate_{dt_name}.log", "w", encoding="utf8") as logf:
            rc = subprocess.run(base + ["--tag", f"{tags[dt_name]}{time.strftime('%H%M%S')}"], cwd=WORK, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
        res[dt_name] = {"rc": rc, "seconds": round(time.time() - t, 1), "gpu": gpu}

    log("number-format check: the same 2-fact edit in bf16 and fp16" + (" on both GPUs at once" if len(gpus) > 1 else ""))
    if len(gpus) > 1:
        th = [threading.Thread(target=one, args=(k, d)) for k, d in enumerate(("bfloat16", "float16"))]
        [t.start() for t in th]; [t.join() for t in th]
    else:
        one(0, "bfloat16"); one(0, "float16")
    for dt_name in ("bfloat16", "float16"):
        res.setdefault(dt_name, {"rc": -1, "seconds": 0.0})
        rows = []
        for fn in sorted(glob.glob(str(WORK / "analysis" / f"run_*_{tags[dt_name]}*_{MODEL}" / "records.jsonl"))):
            rows += [json.loads(x) for x in Path(fn).read_text(encoding="utf8").splitlines()]
        res[dt_name]["rows"] = [{"fact": r["fact"], "en_dP": r["en"]["dP"], "en_ok": r["en"]["success"], "ce_last": r["ce_last"], "delta": r["delta_norm_ratio"]} for r in rows]
    why = ""
    b, h = res["bfloat16"], res["float16"]
    if b["rc"] or not b["rows"]:
        dtype, why = "bfloat16", "bf16 calibration run failed, keeping the original format"
    elif h["rc"] or len(h["rows"]) < len(b["rows"]):
        dtype, why = "bfloat16", "fp16 calibration run failed or incomplete"
    else:
        import math
        bad = any(not all(math.isfinite(x) for x in (r["en_dP"], r["ce_last"], r["delta"])) for r in h["rows"])
        lost = sum(r["en_ok"] for r in h["rows"]) < sum(r["en_ok"] for r in b["rows"]) - 1
        stuck = any(hr["ce_last"] > 2 * br["ce_last"] + 0.5 for hr, br in zip(h["rows"], b["rows"]))
        speed = b["seconds"] / max(h["seconds"], 1e-9)
        res["speedup_fp16_over_bf16"] = round(speed, 2)
        if bad or lost or stuck:
            dtype, why = "bfloat16", f"fp16 gross failure (nonfinite={bad}, lost_success={lost}, optimiser_stuck={stuck})"
        elif speed < 1.5:
            dtype, why = "bfloat16", f"fp16 only {speed:.2f}x faster, not worth leaving the original format"
        else:
            dtype, why = "float16", f"fp16 {speed:.2f}x faster and no gross failure on the 2-fact check"
    f.write_text(json.dumps({"dtype": dtype, "reason": why, "detail": res}, indent=1), encoding="utf8")
    log(f"number format: {dtype} ({why})")
    return dtype


# ---------------------------------------------------------------- jobs
class Job:
    def __init__(self, name: str, build, deps: tuple = (), max_attempts: int = 3) -> None:
        self.name, self.build, self.deps, self.max_attempts = name, build, deps, max_attempts
        self.state, self.attempts = "pending", 0


def latest(pattern: str) -> Path | None:
    hits = sorted((WORK / "analysis").glob(pattern))
    return hits[-1] if hits else None


def prune_empty(d: Path | None) -> None:
    """A crashed first attempt can leave an empty run folder that would block the next one; remove it only if it holds no files."""
    if d and d.is_dir() and not any(p.is_file() for p in d.rglob("*")):
        shutil.rmtree(d)


def edit_job(name: str, tag: str, i: int, n: int, extra: list[str]) -> Job:
    def build(attempt: int) -> list[str]:
        d = latest(f"run_*_{tag}s{i}_{MODEL}")
        if d and not (d / "config.json").exists():
            prune_empty(d); d = None
        cmd = [PY, "code/knowledge_edit_v2.py", "--model", MODEL, "--trace-run", TRACE_RUN, "--n-edit", "40", "--skip-first", "10", "--layers", "14",
               "--add-traced", "--para", "--shard", f"{i}/{n}", "--tag", f"{tag}s{i}"] + extra
        return cmd + (["--resume", str(d)] if d else [])
    return Job(name, build)


def interchange_job(name: str, tag: str, i: int, n: int, deps: tuple, args: list[str]) -> Job:
    def build(attempt: int) -> list[str]:
        d = latest(f"run_*_{tag}s{i}_{MODEL}")
        if d and (d / "effects.npz").exists():  # finished in an earlier session; rerunning would collide with the folder
            return [PY, "-c", f"print('{d.name} already complete')"]
        if d and not list(d.glob("*.npz")):
            prune_empty(d)
        return [PY, "code/interchange.py", "--model", MODEL] + args + ["--max-pairs", "200", "--shuffle-pairs", "--tag", f"{tag}s{i}", "--shard", f"{i}/{n}"]
    return Job(name, build, deps)


def filter_job(name: str, script: str, tag: str, deps: tuple = ()) -> Job:
    # unique tag per attempt: the filter scripts refuse to reuse a folder (write-once), and a resumed session restarts attempt numbers
    return Job(name, lambda attempt: [PY, f"code/{script}", "--model", MODEL, "--set", "a23", "--tag", f"{tag}_{time.strftime('%H%M%S')}a{attempt}"], deps)


def make_jobs(n: int) -> list[Job]:
    jobs = [filter_job("filter_es", "e2_filter.py", "e2a23filter"), filter_job("filter_hi", "e2_filter_hi.py", "e2ha23filter", ("filter_es",))]
    for i in range(n):
        jobs.append(interchange_job(f"es_run_s{i}", "e2a23_A", i, n, ("filter_es",),
                                    ["--langs", "en,es", "--facts", f"data/v2/e2a23_facts_{MODEL}.json", "--k-src", "2"]))
    for i in range(n):
        jobs.append(interchange_job(f"hi_run_s{i}", "e2ha23_A", i, n, ("filter_hi",),
                                    ["--langs", "en,hi", "--facts", f"data/v2/e2ha23_facts_{MODEL}.json", "--span", "end", "--directions", "en2hi,hi2en", "--k-src", "4"]))
    for i in range(n):
        jobs.append(edit_job(f"para_s{i}", "p4a21", i, n, []))
    for i in range(n):
        jobs.append(edit_job(f"shared_s{i}", "p4a24", i, n, ["--key-modes", "mean_all,joint_all"]))
    return jobs


def run_jobs(jobs: list[Job], gpu_ids: list[int], env_base: dict, done: set[str], deadline: float, poll: float = 5.0, status_every: float = 120.0) -> None:
    for j in jobs:
        if j.name in done:
            j.state = "done"
    running: dict[int, tuple] = {}
    last_status = 0.0
    while True:
        for slot, (job, proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close(); del running[slot]
            if rc == 0:
                job.state = "done"; done.add(job.name)
                log(f"DONE {job.name} on GPU {slot} in {(time.time() - t0) / 60:.1f} min")
            else:
                job.attempts += 1
                job.state = "failed" if job.attempts >= job.max_attempts else "pending"
                log(f"FAIL {job.name} (exit {rc}, attempt {job.attempts}/{job.max_attempts}); see analysis/_logs/{job.name}.log" + ("" if job.state == "failed" else "; will resume"))
        for j in jobs:
            if j.state == "pending" and any(d_.state in ("failed", "skipped") for d_ in jobs if d_.name in j.deps):
                j.state = "skipped"; log(f"SKIP {j.name} (a step it needs failed)")
        if time.time() > deadline:
            for slot, (job, proc, fh, t0) in running.items():
                proc.terminate()
            for slot, (job, proc, fh, t0) in running.items():
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                fh.close(); job.state = "pending"; log(f"time budget reached: stopped {job.name}; rerun the notebook to resume it")
            return
        free = [g for g in gpu_ids if g not in running]
        for g in free:
            for j in jobs:
                if j.state == "pending" and all(d_.state == "done" for d_ in jobs if d_.name in j.deps):
                    cmd = j.build(j.attempts + 1)
                    fh = open(WORK / "analysis" / "_logs" / f"{j.name}.log", "a", encoding="utf8")
                    fh.write(f"\n=== attempt {j.attempts + 1}: {' '.join(cmd)}\n"); fh.flush()
                    env = dict(env_base, CUDA_VISIBLE_DEVICES=str(g))
                    running[g] = (j, subprocess.Popen(cmd, cwd=WORK, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
                    j.state = "running"
                    log(f"START {j.name} on GPU {g}")
                    break
        if not running and not any(j.state == "pending" and all(d_.state == "done" for d_ in jobs if d_.name in j.deps) for j in jobs):
            return
        if time.time() - last_status > status_every:
            last_status = time.time()
            g = " ".join(f"gpu{x['index']}:{x['util']}%/{x['used_mib'] / 1024:.1f}GiB" for x in gpu_list())
            log("status: " + ", ".join(f"{j.name}={j.state}" for j in jobs if j.state != "done") + (f" | {g}" if g else ""))
        time.sleep(poll)


# ---------------------------------------------------------------- merge, analyse, package
def post_process(jobs: list[Job], n: int, env: dict) -> list[str]:
    sys.path.insert(0, str(WORK / "kaggle"))
    import merge_shards as ms
    st = {j.name: j.state for j in jobs}
    ana = WORK / "analysis"
    notes: list[str] = []

    def step(name: str, cmd: list[str]) -> bool:
        r = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True)
        with open(ana / "_logs" / f"{name}.log", "a", encoding="utf8") as fh:
            fh.write(r.stdout + r.stderr)
        log(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {name}")
        if r.returncode:
            notes.append(f"{name} failed: see analysis/_logs/{name}.log")
        return r.returncode == 0

    def complete(prefix: str) -> bool:
        return all(st.get(f"{prefix}_s{i}") == "done" for i in range(n))

    def para() -> None:
        d = ms.merge_edit(ana, "p4a21", MODEL, n); log(f"merged {d.name}")
        ref = latest(f"run_*_p4a20_{MODEL}")
        step("analysis_paraphrase", [PY, "code/para_analysis.py", str(d), "--ref", "analysis/run_20261020_p4f7_llama"] + (["--ref", str(ref.relative_to(WORK))] if ref else []))

    def shared() -> None:
        d = ms.merge_edit(ana, "p4a24", MODEL, n); log(f"merged {d.name}")
        base = latest(f"run_*_p4a21_{MODEL}")
        if base:
            step("analysis_shared_edit", [PY, "code/keymode_analysis.py", str(d), "--base", str(base.relative_to(WORK))])
        else:
            notes.append("shared-edit analysis waits for the paraphrase-check run (rerun to finish it)")

    def newfacts(pair: str, tag: str, extra: list[str]) -> None:
        d = ms.merge_interchange(ana, tag, MODEL, n); log(f"merged {d.name}")
        rel = str(d.relative_to(WORK))
        base = [PY, "code/e2_analysis.py", "--models", MODEL, "--langs", pair] + extra + ["--llama-run", rel]
        step(f"analysis_{tag}", base + ["--tag", tag.replace("_A", "_analysis")])
        step(f"analysis_{tag}_leakfree", base + ["--exclude-leak", "--tag", tag.replace("_A", "_analysis_leakfree")])

    blocks = [("para", para), ("shared", shared), ("es_run", lambda: newfacts("en,es", "e2a23_A", [])),
              ("hi_run", lambda: newfacts("en,hi", "e2ha23_A", ["--sites", "A"]))]
    for prefix, fn in blocks:
        if complete(prefix):
            try:
                fn()
            except Exception as e:  # one broken merge/analysis must not stop the others or the packaging
                log(f"FAIL merge/analysis {prefix}: {type(e).__name__}: {e}")
                notes.append(f"{prefix} merge/analysis failed: {type(e).__name__}: {e}")
    return notes


def snapshot() -> tuple[set[str], dict[str, str]]:
    runs = {p.name for p in (WORK / "analysis").iterdir()} if (WORK / "analysis").exists() else set()
    data = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (WORK / "data" / "v2").glob("*.json")}
    return runs, data


def package(start_runs: set[str], start_data: dict[str, str], done: set[str], jobs: list[Job], notes: list[str]) -> None:
    res = OUT / "results"
    res.mkdir(exist_ok=True)
    ana = WORK / "analysis"
    for d in sorted(ana.iterdir()):
        if d.name not in start_runs or d.name == "_logs":
            shutil.copytree(d, res / d.name, dirs_exist_ok=True) if d.is_dir() else shutil.copy2(d, res / d.name)
    (res / "data_v2").mkdir(exist_ok=True)
    for p in (WORK / "data" / "v2").glob("*.json"):
        if start_data.get(p.name) != hashlib.md5(p.read_bytes()).hexdigest():
            shutil.copy2(p, res / "data_v2" / p.name)
    (res / "kaggle_done.json").write_text(json.dumps(sorted(done)), encoding="utf8")
    if (ana / "kaggle_dtype_decision.json").exists():
        shutil.copy2(ana / "kaggle_dtype_decision.json", res / "dtype_decision.json")
    summary = {"minutes": round((time.time() - T0) / 60, 1), "jobs": {j.name: j.state for j in jobs}, "notes": notes,
               "dtype": json.loads((ana / "kaggle_dtype_decision.json").read_text())["dtype"] if (ana / "kaggle_dtype_decision.json").exists() else None}
    (res / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    shutil.make_archive(str(OUT / "babel_new_results"), "zip", res)
    log("packaged: " + str(OUT / "babel_new_results.zip"))


def main() -> None:
    log(f"start; time budget {HOURS} h")
    WORK.mkdir(parents=True, exist_ok=True)
    place_code()
    os.chdir(WORK)
    (WORK / "analysis" / "_logs").mkdir(parents=True, exist_ok=True)
    start_runs, start_data = snapshot()
    done = import_previous()
    install_packages()
    gpus = gpu_list()
    if not gpus:
        raise SystemExit("FATAL: no GPU visible. Set the notebook accelerator to GPU T4 x2.")
    log("GPUs: " + ", ".join(f"{g['index']}={g['name']} {g['total_mib'] / 1024:.0f}GiB" for g in gpus))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
    preflight_files()
    download_model(token)
    cache = WORK / ".cache"  # covariance matrices (~0.3 GB per layer): keep them out of the saved output
    if not cache.exists():
        Path("/tmp/babel_cache").mkdir(exist_ok=True)
        os.symlink("/tmp/babel_cache", cache)
    run_tests()
    env = dict(os.environ, PYTHONPATH="code", PYTHONIOENCODING="utf8", PYTHONUNBUFFERED="1")
    jobs: list[Job] = []
    notes: list[str] = []
    packaged = False
    try:
        env["BABEL_DTYPE"] = calibrate_dtype(gpus)
        n = len(gpus)
        jobs = make_jobs(n)
        run_jobs(jobs, [g["index"] for g in gpus], env, done, DEADLINE)
        notes = post_process(jobs, n, env)
    finally:
        try:
            package(start_runs, start_data, done, jobs, notes)
            packaged = True
        except Exception as e:  # never lose the run because of a packaging slip: the working folder is then kept as the output
            log(f"packaging problem: {type(e).__name__}: {e}")
    bad = [j.name for j in jobs if j.state in ("failed", "skipped")]
    left = [j.name for j in jobs if j.state in ("pending", "running")]
    log("FINISHED. failed/skipped: " + (", ".join(bad) or "none") + " | not finished (rerun with this output attached to resume): " + (", ".join(left) or "none"))
    if packaged:  # everything needed to resume is inside results/ and the zip; the bundle copy is not needed in the output
        os.chdir(OUT)
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
