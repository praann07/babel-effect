#!/usr/bin/env python
"""Runs one step of the Kaggle rerun (kaggle/steps.py) end to end: set up, checks, model downloads, number-format check, all jobs spread
over every GPU (retries, resume), packaging. Only the standard library is imported until the packages are installed.
Environment: BABEL_STEP (step id), HF_TOKEN, HOURS (time budget, default 10.5), KAGGLE_WORK / KAGGLE_INPUT (local tests only).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import layout

T0 = time.time()
HOURS = float(os.environ.get("HOURS", "10.5"))
DEADLINE = T0 + HOURS * 3600
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working/babel-effect"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = WORK.parent  # /kaggle/working: what is here at the end is saved as the notebook output
MODEL_ROOT = Path(os.environ.get("KAGGLE_MODELS", "/tmp/models"))
PY = sys.executable
LAUNCH_GAP = 45.0  # seconds between GPU job starts: two models loading at once can exceed the 29 GB of CPU memory
PACKAGES = ["transformer-lens==3.9.0", "transformers==5.19.0", "accelerate==1.15.0", "jaxtyping", "einops", "fancy-einsum", "better-abc", "beartype"]
HF = {"llama": ("meta-llama/Llama-3.2-3B-Instruct", "llama-3.2-3b-instruct"), "llama_base": ("meta-llama/Llama-3.2-3B", "llama-3.2-3b-base"),
      "gemma2": ("google/gemma-2-2b", "gemma-2-2b")}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}"
    print(line, flush=True)
    try:
        (WORK / "analysis" / "_logs").mkdir(parents=True, exist_ok=True)
        with open(WORK / "analysis" / "_logs" / "driver.log", "a", encoding="utf8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def sh(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)


class Fatal(Exception):
    pass


# ---------------------------------------------------------------- set-up
def place_code() -> None:
    src = os.environ.get("BABEL_SRC")  # the notebook fetched the repo from GitHub
    if src:
        # always replace the code with the fetched commit: a working folder left by an earlier run in this session must not keep old code or tests
        for d in layout.TREES:
            shutil.rmtree(WORK / d, ignore_errors=True)
        log(f"code: {layout.assemble(Path(src), WORK)} files from GitHub commit {os.environ.get('BABEL_SRC_SHA', '?')}")
        return
    if (WORK / "code" / "common.py").exists():
        return
    zips = glob.glob(str(INPUT / "**" / "babel_bundle.zip"), recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
        return
    hits = glob.glob(str(INPUT / "**" / "code" / "common.py"), recursive=True)
    if not hits:
        raise Fatal("the babel-bundle dataset is not attached (no code/common.py under /kaggle/input)")
    shutil.copytree(Path(hits[0]).parent.parent, WORK, dirs_exist_ok=True)


def import_previous() -> set[str]:
    """Bring in run folders, new data files, done-markers and format decisions saved by earlier steps (their output attached as input)."""
    done: set[str] = set()
    cands = [Path(p) for p in glob.glob(str(INPUT / "**" / "results"), recursive=True) if Path(p).is_dir() and (Path(p) / "kaggle_done.json").exists()]
    for z in glob.glob(str(INPUT / "**" / "babel_new_results.zip"), recursive=True):
        tmp = Path("/tmp/prev_results") / hashlib.md5(z.encode()).hexdigest()[:8]
        if not tmp.exists():
            zipfile.ZipFile(z).extractall(tmp)
        cands.append(tmp)
    for r in cands:
        for d in sorted(r.glob("run_*")):
            dst = WORK / "analysis" / d.name
            if d.is_dir() and not dst.exists():
                shutil.copytree(d, dst)
        if (r / "data_v2").exists():
            for f in (r / "data_v2").glob("*.json"):
                shutil.copy2(f, WORK / "data" / "v2" / f.name)
        if (r / "kaggle_done.json").exists():
            done |= set(json.loads((r / "kaggle_done.json").read_text(encoding="utf8")))
        if (r / "dtype_decision.json").exists():
            old = load_decisions()
            new = json.loads((r / "dtype_decision.json").read_text(encoding="utf8"))
            save_decisions(new | old)
    if done:
        log(f"earlier steps imported: {len(done)} finished jobs")
    return done


PROBE = ("import importlib.metadata as md, numpy, scipy, pandas, torch, transformers, transformer_lens; print('numpy', numpy.__version__, "
         "'| scipy', scipy.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, "
         "'| transformer_lens', md.version('transformer-lens'))")


def install_packages() -> None:
    p = sh([PY, "-c", PROBE])
    if p.returncode == 0 and "transformer_lens 3.9.0" in p.stdout:
        log("packages: " + p.stdout.strip())
        return
    for attempt in (1, 2):
        r = sh([PY, "-m", "pip", "install", "-q"] + PACKAGES)
        log(f"pip install attempt {attempt}: exit {r.returncode} {r.stderr[-300:].strip()}")
        p = sh([PY, "-c", PROBE])
        if p.returncode == 0:
            log("packages: " + p.stdout.strip())
            return
        log("import check failed: " + p.stderr[-800:])
    raise Fatal("packages do not import after install (see the import check above)")


SECRET_LABELS = ("HF_TOKEN", "hf_token", "HF_Token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF-TOKEN", "HF_API_TOKEN")


def get_token() -> str:
    """Both models are gated, and TransformerLens reads the hub config of the gated repo at load time, so every job needs the token."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    err = ""
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            UserSecretsClient = None
            err = "not running on Kaggle"
        for label in SECRET_LABELS if UserSecretsClient else ():
            for attempt in range(3):
                try:
                    tok = UserSecretsClient().get_secret(label)
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:160]}"
                    if "No user secrets exist" in str(e):  # this label is not attached to the notebook: try the next spelling
                        break
                    time.sleep(5)  # the secrets service can fail for a moment
            if tok:
                log(f"token read from the Kaggle secret labelled {label}")
                break
    if not tok:
        raise Fatal(f"no Hugging Face token readable here ({err or 'not set'}). Tried secret labels {', '.join(SECRET_LABELS)}. Both models are gated, so nothing can run without it: attach the secret to this notebook, then run again.")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    return tok


WIKITEXT = {"train-00000-of-00001.parquet": 6357543, "test-00000-of-00001.parquet": 732610}  # byte sizes of the Hugging Face copy


def fetch_wikitext(token: str) -> None:
    """WikiText-2 (damage / perplexity text) is not in the repo; the Hugging Face copy is byte-identical to the laptop's (same sha256)."""
    from huggingface_hub import hf_hub_download
    root = WORK / "data" / "wikitext"
    for f, size in WIKITEXT.items():
        for attempt in range(1, 4):
            try:
                p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{f}", repo_type="dataset", local_dir=str(root), token=token)
                if Path(p).stat().st_size != size:
                    raise OSError(f"{f} has the wrong size")
                break
            except Exception as e:
                log(f"wikitext {f} attempt {attempt} failed: {type(e).__name__}: {str(e)[:150]}")
                if attempt == 3:
                    raise Fatal(f"could not download WikiText-2 ({f})")
                time.sleep(10 * attempt)


def gpu_list() -> list[dict]:
    try:
        r = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    except FileNotFoundError:
        return []
    if r.returncode:
        return []
    out = []
    for ln in r.stdout.strip().splitlines():
        i, name, tot, used, util = [x.strip() for x in ln.split(",")]
        out.append({"index": int(i), "name": name, "total_mib": int(tot), "used_mib": int(used), "util": int(util)})
    return out


def download_model(key: str, token: str | None) -> None:
    repo, dirname = HF[key]
    target = MODEL_ROOT / dirname
    size = lambda: sum(f.stat().st_size for f in target.glob("*.safetensors"))  # noqa: E731
    if not ((target / "config.json").exists() and size() > 4e9):
        from huggingface_hub import snapshot_download
        for attempt in range(1, 5):
            try:
                snapshot_download(repo, local_dir=str(target), token=token, ignore_patterns=["original/*"])
                break
            except Exception as e:  # network blips, rate limits, missing licence
                log(f"download {repo} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
                if attempt == 4:
                    raise Fatal(f"could not download {repo}. The HF_TOKEN account must have accepted that model's licence on huggingface.co.")
                time.sleep(20 * attempt)
    if size() < 4e9:
        raise Fatal(f"{repo} files look incomplete ({size() / 1e9:.1f} GB)")
    link = WORK / "models"
    if not link.exists():
        os.symlink(MODEL_ROOT, link)
    log(f"model ready: {repo} ({size() / 1e9:.1f} GB)")


def run_tests() -> None:
    r = sh([PY, "-m", "unittest", "discover", "-s", "tests"], env=dict(os.environ, PYTHONPATH="code"), cwd=WORK)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    log("unit tests: " + " | ".join(tail))
    if r.returncode:
        raise Fatal("unit tests fail in this environment; results would not be trustworthy. Full output: " + (r.stdout + r.stderr)[-1500:])


# ---------------------------------------------------------------- number format (rule pre-registered in docs/HYPOTHESIS.md A25)
def decisions_path() -> Path:
    return WORK / "analysis" / "kaggle_dtype_decision.json"


def load_decisions() -> dict:
    p = decisions_path()
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else {}


def save_decisions(d: dict) -> None:
    decisions_path().parent.mkdir(parents=True, exist_ok=True)
    decisions_path().write_text(json.dumps(d, indent=1), encoding="utf8")


def compare_formats(b: dict | None, h: dict | None) -> tuple[str, str, dict]:
    """b, h: dtype_check outputs for bfloat16 and float16 (None if that run failed)."""
    import math
    ok = lambda r: r is not None and all(x["finite"] for x in r["rows"])  # noqa: E731
    if not ok(b) and not ok(h):
        raise Fatal("the model fails the format check in both bf16 and fp16")
    if not ok(h):
        return "bfloat16", "fp16 run failed or gave non-finite values", {}
    if not ok(b):
        return "float16", "bf16 run failed here; fp16 is finite", {}
    agree = sum(x["top1"] == y["top1"] for x, y in zip(b["rows"], h["rows"])) / len(b["rows"])
    cos = []
    for gb, gh in zip(b["grads"], h["grads"]):
        nb, nh = math.sqrt(sum(v * v for v in gb)), math.sqrt(sum(v * v for v in gh))
        cos.append(sum(x * y for x, y in zip(gb, gh)) / (nb * nh) if nb > 0 and nh > 0 else 0.0)
    cos.sort()
    gcos = cos[len(cos) // 2]
    speed = b["seconds_3_passes"] / max(h["seconds_3_passes"], 1e-9)
    stats = {"top1_agreement": round(agree, 3), "median_grad_cosine": round(gcos, 4), "speedup_fp16": round(speed, 2)}
    if agree >= 0.95 and gcos >= 0.9 and speed >= 1.5:
        return "float16", f"fp16 {speed:.2f}x faster, top-1 agreement {agree:.2f}, gradient cosine {gcos:.3f}", stats
    return "bfloat16", f"fp16 not accepted (speed {speed:.2f}x, agreement {agree:.2f}, gradient cosine {gcos:.3f})", stats


def decide_formats(models: list[str], gpus: list[dict]) -> dict:
    dec = load_decisions()
    for m in models:
        if m in dec:
            log(f"number format for {m}: {dec[m]['dtype']} (decided earlier: {dec[m]['reason']})")
            continue
        res: dict = {}

        def one(k: int, dt: str) -> None:
            if k:
                time.sleep(LAUNCH_GAP)  # not both model loads at the same moment (CPU memory)
            out = WORK / "analysis" / "_logs" / f"dtype_{m}_{dt}.json"
            env = dict(os.environ, PYTHONPATH="code", PYTHONIOENCODING="utf8", CUDA_VISIBLE_DEVICES=str(gpus[min(k, len(gpus) - 1)]["index"]), BABEL_DTYPE=dt)
            r = sh([PY, "kaggle/dtype_check.py", "--model", m, "--out", str(out)], env=env, cwd=WORK)
            (WORK / "analysis" / "_logs" / f"dtype_{m}_{dt}.log").write_text(r.stdout + r.stderr, encoding="utf8")
            res[dt] = json.loads(out.read_text(encoding="utf8")) if r.returncode == 0 and out.exists() else None

        log(f"number-format check for {m} (bf16 and fp16" + (" at once on two GPUs)" if len(gpus) > 1 else ")"))
        if len(gpus) > 1:
            th = [threading.Thread(target=one, args=(k, d)) for k, d in enumerate(("bfloat16", "float16"))]
            [t.start() for t in th]
            [t.join() for t in th]
        else:
            one(0, "bfloat16"); one(0, "float16")
        dtype, why, stats = compare_formats(res.get("bfloat16"), res.get("float16"))
        dec[m] = {"dtype": dtype, "reason": why, "stats": stats}
        save_decisions(dec)
        log(f"number format for {m}: {dtype} ({why})")
    return {m: dec[m]["dtype"] for m in dec}


# ---------------------------------------------------------------- jobs
class Missing(Exception):
    pass


class Job:
    def __init__(self, name: str, build, deps: tuple = (), max_attempts: int = 3, gpu: bool = True, env: dict | None = None) -> None:
        self.name, self.build, self.deps, self.max_attempts, self.gpu, self.env = name, build, deps, max_attempts, gpu, env or {}
        self.state, self.attempts = "pending", 0


def runs(pattern: str) -> list[Path]:
    return sorted(d for d in (WORK / "analysis").glob(f"run_*_{pattern}") if d.is_dir())


def expand(cmd: list[str], attempt: int) -> list[str]:
    out: list[str] = []
    for tok in cmd:
        if tok == "{py}":
            out.append(PY)
        elif tok.startswith("{tag:"):
            out.append(f"{tok[5:-1]}r{time.strftime('%H%M%S')}a{attempt}")
        elif tok.startswith("{run:"):
            hits = runs(tok[5:-1])
            if not hits:
                raise Missing(f"no Kaggle run folder matches {tok[5:-1]} (did the earlier step finish?)")
            out.append(str(hits[-1].relative_to(WORK)).replace("\\", "/"))
        elif tok.startswith("{resume:"):
            hits = runs(tok[8:-1])
            d = hits[-1] if hits else None
            if d and not any(p.is_file() for p in d.rglob("*")):
                shutil.rmtree(d); d = None  # empty folder left by a crash would block the rerun
            if d and (d / "config.json").exists():
                out += ["--resume", str(d.relative_to(WORK)).replace("\\", "/")]
        else:
            out.append(tok)
    return out


def plan_jobs(step: str, plan: dict, dtypes: dict) -> list[Job]:
    jobs = []
    for spec in plan["jobs"]:
        env = {"BABEL_DTYPE": dtypes.get(spec.get("model"), "bfloat16")}
        jobs.append(Job(f"{step}:{spec['name']}", lambda attempt, c=spec["cmd"]: expand(c, attempt),
                        tuple(f"{step}:{d}" for d in spec.get("deps", [])), spec.get("max_attempts", 3), spec.get("gpu", True), env))
    return jobs


def run_jobs(jobs: list[Job], gpu_ids: list[int], env_base: dict, done: set[str], deadline: float, poll: float = 5.0, status_every: float = 120.0) -> None:
    for j in jobs:
        if j.name in done:
            j.state = "done"
    running: dict = {}  # key: GPU index, or ("cpu", job name) for CPU jobs
    last_status = 0.0
    last_start = [0.0]
    ready = lambda j: j.state == "pending" and all(d.state == "done" for d in jobs if d.name in j.deps)  # noqa: E731
    while True:
        for key, (job, proc, fh, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close(); del running[key]
            where = f"GPU {key}" if job.gpu else "CPU"
            if rc == 0:
                job.state = "done"; done.add(job.name)
                log(f"DONE {job.name} on {where} in {(time.time() - t0) / 60:.1f} min")
            else:
                job.attempts += 1
                job.state = "failed" if job.attempts >= job.max_attempts else "pending"
                log(f"FAIL {job.name} (exit {rc}, attempt {job.attempts}/{job.max_attempts}); log: analysis/_logs/{job.name.replace(':', '_')}.log"
                    + ("" if job.state == "failed" else "; retrying"))
        for j in jobs:
            if j.state == "pending" and any(d.state in ("failed", "skipped") for d in jobs if d.name in j.deps):
                j.state = "skipped"; log(f"SKIP {j.name} (a step it needs failed)")
        if time.time() > deadline:
            for job, proc, fh, t0 in running.values():
                proc.terminate()
            for job, proc, fh, t0 in running.values():
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                fh.close(); job.state = "pending"
                log(f"time budget reached: stopped {job.name}; run this step again with this output attached to resume")
            return

        def launch(j: Job, key) -> None:
            try:
                cmd = j.build(j.attempts + 1)
            except Missing as e:
                j.state = "failed"; log(f"FAIL {j.name}: {e}")
                return
            fh = open(WORK / "analysis" / "_logs" / f"{j.name.replace(':', '_')}.log", "a", encoding="utf8")
            fh.write(f"\n=== attempt {j.attempts + 1}: {' '.join(cmd)}\n"); fh.flush()
            env = dict(env_base, **j.env)
            if j.gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(key)
            running[key] = (j, subprocess.Popen(cmd, cwd=WORK, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
            j.state = "running"
            if j.gpu:
                last_start[0] = time.time()
            log(f"START {j.name} on " + (f"GPU {key}" if j.gpu else "CPU"))

        for j in jobs:
            if not j.gpu and ready(j):
                launch(j, ("cpu", j.name))
        for g in [g for g in gpu_ids if g not in running]:
            nxt = next((j for j in jobs if j.gpu and ready(j)), None)
            if nxt is None or time.time() - last_start[0] < LAUNCH_GAP:
                break
            launch(nxt, g)
        if not running and not any(ready(j) for j in jobs):
            return
        if time.time() - last_status > status_every:
            last_status = time.time()
            g = " ".join(f"gpu{x['index']}:{x['util']}%/{x['used_mib'] / 1024:.1f}GiB" for x in gpu_list())
            log("status: " + ", ".join(f"{j.name.split(':')[-1]}={j.state}" for j in jobs if j.state != "done") + (f" | {g}" if g else ""))
        time.sleep(poll)


# ---------------------------------------------------------------- package
def snapshot() -> tuple[set[str], dict[str, str]]:
    names = {p.name for p in (WORK / "analysis").iterdir()} if (WORK / "analysis").exists() else set()
    data = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (WORK / "data" / "v2").glob("*.json")}
    return names, data


def package(step: str, start_runs: set[str], start_data: dict[str, str], done: set[str], jobs: list[Job], notes: list[str]) -> None:
    res = OUT / "results"
    res.mkdir(exist_ok=True)
    ana = WORK / "analysis"
    for d in sorted(ana.iterdir()):
        if d.name not in start_runs or d.name == "_logs":
            if d.is_dir():
                shutil.copytree(d, res / d.name, dirs_exist_ok=True)
            else:
                shutil.copy2(d, res / d.name)
    (res / "data_v2").mkdir(exist_ok=True)
    for p in (WORK / "data" / "v2").glob("*.json"):
        if start_data.get(p.name) != hashlib.md5(p.read_bytes()).hexdigest():
            shutil.copy2(p, res / "data_v2" / p.name)
    (res / "kaggle_done.json").write_text(json.dumps(sorted(done)), encoding="utf8")
    if decisions_path().exists():
        shutil.copy2(decisions_path(), res / "dtype_decision.json")
    summary = {"step": step, "minutes": round((time.time() - T0) / 60, 1), "jobs": {j.name: j.state for j in jobs},
               "number_format": load_decisions(), "notes": notes,
               "code_version": os.environ.get("BABEL_SRC_SHA", "dataset")}
    (res / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    shutil.make_archive(str(OUT / "babel_new_results"), "zip", res)
    log("packaged: " + str(OUT / "babel_new_results.zip"))


def main() -> int:
    step = os.environ.get("BABEL_STEP", "")
    log(f"step {step or '?'}; time budget {HOURS} h")
    WORK.mkdir(parents=True, exist_ok=True)
    place_code()
    os.chdir(WORK)
    (WORK / "analysis" / "_logs").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORK / "kaggle"))
    from steps import STEPS
    if step not in STEPS:
        raise Fatal(f"unknown step {step!r}; known: {sorted(STEPS)}")
    plan = STEPS[step]
    log(plan["title"])
    start_runs, start_data = snapshot()
    done = import_previous()
    missing = [p for p in plan.get("needs", []) if not (WORK / p).exists()]
    if missing:
        raise Fatal(f"missing inputs {missing}: attach the output of the previous step")
    token = get_token()
    gpus = gpu_list()
    if not gpus:
        raise Fatal("no GPU visible: set the accelerator to GPU T4 x2")
    install_packages()
    log("GPUs: " + ", ".join(f"{g['index']}={g['name']} {g['total_mib'] / 1024:.0f}GiB" for g in gpus))
    fetch_wikitext(token)
    for m in plan["models"]:
        download_model(m, token)
    cache = WORK / ".cache"  # covariance matrices (~0.3 GB per layer) stay out of the saved output
    if not cache.exists():
        Path("/tmp/babel_cache").mkdir(exist_ok=True)
        os.symlink("/tmp/babel_cache", cache)
    run_tests()
    env = dict(os.environ, PYTHONPATH="code", PYTHONIOENCODING="utf8", PYTHONUNBUFFERED="1")
    jobs: list[Job] = []
    notes: list[str] = []
    packaged = False
    try:
        dtypes = decide_formats(plan.get("calibrate", []), gpus)
        jobs = plan_jobs(step, plan, dtypes)
        run_jobs(jobs, [g["index"] for g in gpus], env, done, DEADLINE)
        notes += [f"expected output missing: {p}" for p in plan.get("produces", []) if not (WORK / p).exists()]
    finally:
        try:
            package(step, start_runs, start_data, done, jobs, notes)
            packaged = True
        except Exception as e:  # never lose the run over a packaging slip: the working folder is then kept as the output
            log(f"packaging problem: {type(e).__name__}: {e}")
    bad = [j.name for j in jobs if j.state in ("failed", "skipped")]
    left = [j.name for j in jobs if j.state in ("pending", "running")]
    log("FINISHED. failed/skipped: " + (", ".join(bad) or "none") + " | not finished (run again with this output attached): " + (", ".join(left) or "none")
        + (" | " + "; ".join(notes) if notes else ""))
    if packaged:
        os.chdir(OUT)
        shutil.rmtree(WORK, ignore_errors=True)
    return 0 if not bad and not left and not notes else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as e:
        log(f"STOPPED: {e}")
        sys.exit(2)
