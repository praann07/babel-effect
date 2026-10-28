"""Live tracker for the night runs: hardware (GPU, CPU, RAM) and progress bars. Read-only; Ctrl+C to quit."""
import subprocess, time, glob, json, os
import psutil
from tqdm import tqdm

STEPS = ["filter_es", "filter_hi", "es_run", "hi_run", "an_es", "an_es_lf", "an_hi", "an_hi_lf", "cluster"]
NAMES = {"filter_es": "fact check EN/ES", "filter_hi": "fact check Hindi", "es_run": "English<->Spanish swap test",
         "hi_run": "English<->Hindi swap test", "an_es": "numbers EN/ES", "an_es_lf": "numbers EN/ES (no leaks)",
         "an_hi": "numbers EN/HI", "an_hi_lf": "numbers EN/HI (no leaks)", "cluster": "error-bar check"}


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        u, mu, mt, t, p = [x.strip() for x in o]
        return f"GPU {u}% | VRAM {float(mu)/1024:.1f}/{float(mt)/1024:.1f} GB | {t}C | {float(p):.0f}W"
    except Exception:
        return "GPU n/a"


def done(path):
    try:
        return [l.strip() for l in open(path, encoding="utf8") if l.strip()]
    except OSError:
        return []


def pairs_line():
    for pat in ("analysis/run_*_e2hb_A_llama_base", "analysis/run_*_e2b_A_llama_base"):
        for d in sorted(glob.glob(pat))[-1:]:
            for f in ("partial_meta.json", "meta.json"):
                try:
                    m = json.load(open(os.path.join(d, f), encoding="utf8"))
                    return f"{os.path.basename(d).split('_',2)[2]}: {m.get('n_done', len(m.get('pairs', [])))} pairs done"
                except Exception:
                    pass
    return ""


def last_line(path):
    try:
        t = open(path, encoding="utf8", errors="ignore").read().replace("\r", "\n").strip().splitlines()
        return t[-1][:90] if t else ""
    except OSError:
        return ""


psutil.cpu_percent()
bar = tqdm(total=len(STEPS) + 1, position=0, bar_format="{desc}  {bar} {n_fmt}/{total_fmt} steps", dynamic_ncols=True)
hw = tqdm(position=1, bar_format="{desc}", dynamic_ncols=True)
cur = tqdm(position=2, bar_format="{desc}", dynamic_ncols=True)
log = tqdm(position=3, bar_format="{desc}", dynamic_ncols=True)
try:
    while True:
        d19, d20 = done("analysis/_logs/a19.done"), done("analysis/_logs/a20.done")
        a20_fin = any("a20 exit" in l for l in done("analysis/_logs/a20.log"))
        n = sum(s in d19 for s in STEPS) + (1 if a20_fin else 0)
        bar.n = n; bar.set_description_str("PROGRESS"); bar.refresh()
        nxt = next((s for s in STEPS if s not in d19), None)
        what = NAMES[nxt] if nxt else ("edit test (20 more facts)" if not a20_fin else "ALL DONE")
        hw.set_description_str(f"{gpu()} | CPU {psutil.cpu_percent()}% | RAM {psutil.virtual_memory().percent}% | {time.strftime('%H:%M:%S')}")
        cur.set_description_str(f"NOW: {what}   {pairs_line() if nxt in ('es_run','hi_run') else ''}")
        log.set_description_str("log: " + last_line("analysis/_logs/a20.log" if not nxt else "analysis/_logs/a19.log"))
        time.sleep(5)
except KeyboardInterrupt:
    pass
