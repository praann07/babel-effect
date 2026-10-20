"""Live, plain-English dashboard for the GPU queue (read-only; never touches the queue).

Run:  .venv\\Scripts\\python watch_queue.py        (Ctrl+C quits; --once prints a single frame)
Best in a wide, maximised terminal (about 170 x 50). Graphs fill in as you watch.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
LOGD = ROOT / "analysis" / "_logs"
EV = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (START|OK|FAIL) (\S+)")
ITEM = re.compile(r"^(\S+) (en|es|hi) (p\d+): target='([^']*)' n=\d+ P ([\d.]+)->([\d.e-]+) crit_mlp=(\d+)")
LANG = {"en": "English", "es": "Spanish", "hi": "Hindi"}

# (queue step name, short title, one-line purpose, rough minutes, progress-log glob, expected-count key, plain-English explanation)
TRACE = ("Blur the subject of a question (e.g. 'Paris' in 'The capital of France is...'), so the model gets confused. Then put the model's clean internal state "
         "back one layer at a time and see which layer makes the right answer return. That layer is where the fact is 'looked up'. Done in English, Spanish and Hindi.")
OVERLAP = "Compares the 'which layer looks it up' patterns across languages and tests them against shuffled facts. If they match far more than chance, the languages share the same machinery."
H8T = "Same blur-and-restore test, but the question is worded differently. Proves the finding is not an accident of one particular sentence."
PREP = "Preparation: choose question templates and keep only facts the model already answers correctly in all 3 languages (nothing to trace otherwise)."
H8A = "Compares the original wording with the reworded questions: do the same layers matter?"
INCTX = "Baseline: instead of editing the model, just write the new fact in the prompt. Tells us how good the model edit really is compared with this simple trick."
F7 = ("The core experiment: overwrite a fact in the model's memory using English only (e.g. 'capital of France = Rome'), then check whether Spanish and Hindi also changed, "
      "and whether unrelated facts or normal text got damaged. 30 more facts makes the statistics stronger.")
PLAN = [
    ("llama_trace_extra", "Llama: trace new fact types", "Find where city->country, landmark->country facts live", 30, "trace_extra_llama_*.log", "extra_llama", TRACE),
    ("llama_overlap_all", "Llama: compare languages", "Same layers across EN/ES/HI?", 2, None, None, OVERLAP),
    ("llama_h8trace", "Llama: rewording test", "Re-trace with differently worded questions", 45, "h8trace_llama_*.log", "h8", H8T),
    ("gemma2_select_extra", "Gemma: choose question wording", "Pick question templates for Gemma", 15, None, None, PREP),
    ("gemma2_build_extra", "Gemma: build fact list", "Fill templates with the new facts", 1, None, None, PREP),
    ("gemma2_filter_extra", "Gemma: keep known facts", "Keep facts Gemma knows in all 3 languages", 15, None, None, PREP),
    ("gemma2_trace_extra", "Gemma: trace new fact types", "Find where Gemma stores those facts", 30, "trace_extra_gemma2_*.log", "extra_gemma2", TRACE),
    ("gemma2_overlap_all", "Gemma: compare languages", "Same layers across EN/ES/HI?", 2, None, None, OVERLAP),
    ("gemma2_h8trace", "Gemma: rewording test", "Re-trace with differently worded questions", 45, "h8trace_gemma2_*.log", "h8", H8T),
    ("llama_h8", "Llama: rewording result", "Analyse the rewording test", 1, None, None, H8A),
    ("gemma2_h8", "Gemma: rewording result", "Analyse the rewording test", 1, None, None, H8A),
    ("llama_incontext", "Llama: prompt-only baseline", "Edit vs just telling it in the prompt", 30, None, None, INCTX),
    ("gemma2_incontext", "Gemma: prompt-only baseline", "Same baseline for Gemma", 30, None, None, INCTX),
    ("llama_f7", "Llama: 30 more edits", "Edit in English, test Spanish/Hindi", 120, None, None, F7),
    ("gemma2_f7", "Gemma: 30 more edits", "Same for Gemma", 120, None, None, F7),
    ("bloom1b7", "BLOOM-1b7 pilot", "Third model: is weak Hindi a tokenizer effect?", 120, None, None,
     "A third model with a different tokenizer (how text is chopped into pieces). Tests whether Hindi's weaker result is just because Hindi words get split into many pieces. Full pipeline, small scale."),
    ("llama1b", "Llama-3.2-1B pilot", "Smaller sibling model", 90, None, None, "A smaller model of the same family: does the same pattern appear? Full pipeline only if enough facts survive the filter."),
]
GROUP_DONE = {"bloom1b7": ("bloom1b7_edit", "bloom1b7_below50"), "llama1b": ("llama1b_overlap", "llama1b_below50")}
HIST = {k: deque(maxlen=180) for k in ("gpu", "vram", "temp", "cpu", "ram")}
RATE: deque = deque(maxlen=90)
RATE_KEY = [None]
BLK = " ▁▂▃▄▅▆▇█"


def read_lines(p: Path) -> list[str]:
    try:
        return p.read_text(encoding="utf8", errors="replace").splitlines()
    except OSError:
        return []


def newest(glob: str) -> Path | None:
    fs = sorted(LOGD.glob(glob), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None


def fmt(sec: float) -> str:
    sec = max(0, int(sec))
    h, m = sec // 3600, sec % 3600 // 60
    return f"{h}h {m:02d}m" if h else f"{m}m {sec % 60:02d}s"


def bar(frac: float, color: str, width: int = 24) -> Text:
    frac = min(1.0, max(0.0, frac))
    n = int(round(frac * width))
    return Text("█" * n, style=color) + Text("░" * (width - n), style="grey37") + Text(f" {frac * 100:3.0f}%")


def chart(vals, vmax: float, height: int, width: int, color: str, label: str, unit: str = "%") -> Text:
    vals = list(vals)[-width:]
    levels = [min(height * 8, max(0, round(v / vmax * height * 8))) for v in vals]
    t = Text()
    for r in range(height, 0, -1):
        edge = f"{vmax:>4.0f}{unit}" if r == height else (f"{0:>4.0f}{unit}" if r == 1 else "      ")
        t.append(edge + " ", style="grey50")
        t.append("".join(BLK[max(0, min(8, lv - (r - 1) * 8))] for lv in levels).rjust(width), style=color)
        t.append("\n")
    now = f"{vals[-1]:.0f}{unit}" if vals else "-"
    t.append(f"      {label}: now {now}, last {len(vals) * 2 // 60 or 1} min", style="dim")
    return t


def spark(vals, vmax: float, width: int) -> str:
    vals = list(vals)[-width:]
    return "".join(BLK[max(0, min(8, round(v / vmax * 8)))] for v in vals).rjust(width)


def expected(key: str | None) -> int | None:
    if key is None:
        return None
    if key == "h8":
        return 60
    model = key.split("_", 1)[1]
    try:
        return json.loads((ROOT / "data" / "v2" / f"facts_extra_{model}.json").read_text(encoding="utf8"))["summary"]["n_kept"]
    except (OSError, KeyError, ValueError):
        return None


def progress(glob: str | None, key: str | None) -> tuple[int, int] | None:
    if not glob:
        return None
    f, total = newest(glob), expected(key)
    if f is None or not total:
        return None
    seen = {m.groups() for m in (re.match(r"^(\S+) en (p\d+):", l) for l in read_lines(f)) if m}
    return min(len(seen), total), total


def queue_alive() -> bool:
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any("scripts_run_queue2" in c for c in (p.info["cmdline"] or [])):
                return True
        except psutil.Error:
            pass
    return False


def gpu() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        u, mu, mt, t, w, c = (float(x) for x in out)
        return {"util": u, "mem": mu / 1024, "memtot": mt / 1024, "temp": t, "w": w, "clk": c}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def plain_item(line: str) -> Text | None:
    m = ITEM.match(line)
    if not m:
        return None
    fid, lg, _, ans, p0, p1, crit = m.groups()
    name = re.sub(r"_\d+$", "", fid).replace("_", " ")
    t = Text("• ")
    t.append(f"{name} #{fid.rsplit('_', 1)[-1]}, {LANG[lg]}: ", style="white")
    t.append(f"'{ans}' ", style="bold")
    t.append(f"model {float(p0) * 100:.0f}% sure -> {float(p1) * 100:.1f}% when blurred; lookup layer = {crit}", style="dim")
    return t


def build(width: int) -> Group:
    now = dt.datetime.now()
    wide = width >= 150
    colw = width // 2 - 6 if wide else width - 6
    events = [(dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), m[2], m[3]) for m in map(EV.match, read_lines(LOGD / "queue2.log")) if m]
    done = set(read_lines(LOGD / "queue2.done"))
    last_ev = events[-1] if events else None
    alive = queue_alive()

    # ---- live samples ------------------------------------------------------------------------------------------------------
    g, vm = gpu(), psutil.virtual_memory()
    if g:
        HIST["gpu"].append(g["util"]); HIST["vram"].append(g["mem"] / g["memtot"] * 100); HIST["temp"].append(g["temp"])
    HIST["cpu"].append(psutil.cpu_percent()); HIST["ram"].append(vm.percent)
    free = vm.available / 2**30

    # ---- step states ------------------------------------------------------------------------------------------------------
    def step_events(name: str) -> list:
        return [e for e in events if e[2] == name or (name in GROUP_DONE and e[2].startswith(name))]

    state: dict[str, tuple[str, float | None, float | None]] = {}
    for key, _, _, est, glob, ekey, _ in PLAN:
        evs = step_events(key)
        fin = any(d in done for d in GROUP_DONE[key]) if key in GROUP_DONE else key in done
        is_last = bool(last_ev and last_ev[1] == "START" and (last_ev[2] == key or (key in GROUP_DONE and last_ev[2].startswith(key))))
        failed = any(e[1] == "FAIL" for e in evs) and not fin
        if is_last and not fin:
            start = max(e[0] for e in evs if e[1] == "START")
            pr = progress(glob, ekey)
            state[key] = ("running" if alive else "interrupted", (now - start).total_seconds(), (pr[0] / pr[1]) if pr else None)
        elif fin:
            st = [e[0] for e in evs if e[1] == "START"]
            ok = [e[0] for e in evs if e[1] == "OK"]
            state[key] = ("done", (ok[-1] - st[-1]).total_seconds() if st and ok and ok[-1] >= st[-1] else None, 1.0)
        elif failed:
            state[key] = ("failed", None, None)
        else:
            state[key] = ("waiting", None, None)

    est_of = {k: e for k, _, _, e, _, _, _ in PLAN}
    factor = None
    for k, (s, el, fr) in state.items():
        if s == "running" and fr and fr > 0.05 and el:
            factor = (el / 60 / fr) / est_of[k]
    if factor is None:
        r = [(el / 60) / est_of[k] for k, (s, el, fr) in state.items() if s == "done" and el and est_of[k] >= 15]
        factor = statistics.median(r) if r else 1.5
    factor = min(3.0, max(0.5, factor))

    remaining, total_w, done_w = 0.0, sum(est_of.values()), 0.0
    cur_key = None
    for key, title, _, est, _, _, _ in PLAN:
        s, el, fr = state[key]
        if s == "done":
            done_w += est
        elif s == "running":
            cur_key = key
            f = fr if fr else min(0.9, (el or 0) / 60 / (est * factor))
            done_w += est * f
            remaining += max(60.0, ((el or 0) / f - (el or 0)) if f > 0.05 else est * factor * 60 - (el or 0))
        elif s in ("waiting", "interrupted", "failed") and s != "failed":
            remaining += est * factor * 60
    overall = done_w / total_w
    todo = [k for k, (s, _, _) in state.items() if s in ("waiting", "interrupted", "running")]

    # ---- 1. status ---------------------------------------------------------------------------------------------------------
    head = Text()
    head.append("BABEL EFFECT  -  live experiment monitor   ", style="bold cyan")
    head.append(now.strftime("%a %d %b  %H:%M:%S"), style="dim")
    head.append("\nStatus: ")
    if alive:
        head.append("RUNNING", style="bold green")
        head.append("  (the program is working through the list of steps below)")
    elif todo:
        head.append("PAUSED / STOPPED", style="bold red")
        head.append("  - nothing is computing. Double-click 'Start Experiments' on the Desktop to continue; finished steps are never redone.")
    else:
        head.append("ALL DONE", style="bold green")
    ov = Text("\nOverall progress  ") + bar(overall, "green", 40)
    ov.append(f"\nStill needs about {fmt(remaining)} of computer time" + (f"  ->  finishes around {(now + dt.timedelta(seconds=remaining)).strftime('%a %H:%M')} if left running" if alive else " once restarted"))
    ov.append(f"\n(a guess; so far steps run at {factor:.1f}x my first estimate)", style="dim")
    p_status = Panel(Group(head, ov), title="Status", title_align="left", border_style="cyan")

    # ---- 2. what's happening right now ------------------------------------------------------------------------------------
    now_t = Text()
    if cur_key:
        title = next(p[1] for p in PLAN if p[0] == cur_key)
        s, el, fr = state[cur_key]
        long = next(p[6] for p in PLAN if p[0] == cur_key)
        now_t.append(f"{title}   (running for {fmt(el or 0)})\n", style="bold yellow")
        now_t.append(long + "\n\n", style="white")
        if fr is not None:
            pr = progress(next(p[4] for p in PLAN if p[0] == cur_key), next(p[5] for p in PLAN if p[0] == cur_key))
            if RATE_KEY[0] != cur_key:
                RATE.clear(); RATE_KEY[0] = cur_key
            if pr:
                RATE.append((time.time(), pr[0]))
            now_t.append("Facts traced  ")
            now_t.append_text(bar(fr, "yellow", 30))
            now_t.append(f"   {pr[0]} of {pr[1]}\n" if pr else "\n")
            if len(RATE) >= 5 and RATE[-1][0] - RATE[0][0] >= 60 and RATE[-1][1] > RATE[0][1] and pr:
                rate = (RATE[-1][1] - RATE[0][1]) / (RATE[-1][0] - RATE[0][0]) * 60
                now_t.append(f"Speed: {rate:.1f} facts per minute  ->  about {fmt((pr[1] - pr[0]) / rate * 60)} left for this step\n", style="green")
        files = [p for p in LOGD.glob("*.log") if not re.match(r"(queue2|watchdog2)", p.name)]
        lf = max(files, key=lambda p: p.stat().st_mtime) if files else None
        if lf:
            age = time.time() - lf.stat().st_mtime
            items = [t for t in (plain_item(l) for l in read_lines(lf)[-60:]) if t][-5:]
            if items:
                now_t.append("\nMost recent results (each line = one question the model answered):\n", style="bold")
                for t in items:
                    now_t.append_text(t); now_t.append("\n")
            now_t.append(f"\nLog last updated {fmt(age)} ago" + ("   <- quiet for a while; may be a slow step or stuck" if age > 600 else ""), style="red" if age > 600 else "dim")
    elif todo:
        now_t.append("Nothing is running. Next step when restarted: ", style="bold")
        nxt = next(p for p in PLAN if p[0] in todo)
        now_t.append(f"{nxt[1]}\n\n", style="yellow")
        now_t.append(nxt[6])
    else:
        now_t.append("All steps finished.", style="bold green")
    p_now = Panel(now_t, title="What is happening right now", title_align="left", border_style="yellow")

    # ---- 3. graphs ---------------------------------------------------------------------------------------------------------
    cw = max(20, min(120, colw - 8))
    gt = Text()
    gt.append_text(chart(HIST["gpu"], 100, 5, cw, "magenta", "GPU busy (100% = working flat out)"))
    gt.append("\n\n")
    gt.append_text(chart(HIST["vram"], 100, 3, cw, "cyan", "GPU memory used (full = at the limit, normal for this work)"))
    gt.append("\n\n")
    for name, key, vmax, col, unit in (("GPU temp  ", "temp", 100, "red", "°C"), ("CPU busy  ", "cpu", 100, "blue", "%"), ("RAM used  ", "ram", 100, "green", "%")):
        v = HIST[key][-1] if HIST[key] else 0
        gt.append(f"{name}", style="bold")
        gt.append(spark(HIST[key], vmax, cw) + "\n", style=col)
        gt.append(f"           now {v:.0f}{unit}\n", style="dim")
    if g:
        gt.append(f"\nGPU: {g['mem']:.1f} of {g['memtot']:.1f} GB memory, {g['w']:.0f} W, clock {g['clk']:.0f} MHz.   RAM free: {free:.1f} GB.   Disk free: {shutil.disk_usage(ROOT).free / 2**30:.0f} GB", style="dim")
    p_graph = Panel(gt, title="Live graphs (update every 2 seconds, start empty when you open this)", title_align="left", border_style="magenta")

    # ---- 4. steps ----------------------------------------------------------------------------------------------------------
    t = Table(box=None, padding=(0, 1), show_header=True, header_style="bold")
    t.add_column("", width=2)
    t.add_column("Step")
    t.add_column("In plain words", style="dim", overflow="fold")
    t.add_column("Time taken", min_width=14)
    t.add_column("Status", justify="right")
    icon = {"done": ("✓", "green"), "running": ("▶", "bold yellow"), "waiting": ("·", "grey50"), "failed": ("✗", "bold red"), "interrupted": ("↺", "yellow")}
    mx = max([v[1] for v in state.values() if v[0] == "done" and v[1]] or [1])
    for key, title, why, est, _, _, _ in PLAN:
        s, el, fr = state[key]
        ic, col = icon[s]
        tb = Text("")
        if s == "done" and el:
            tb = Text("▇" * max(1, int(el / mx * 10)), style="green") + Text(f" {fmt(el)}", style="dim")
        stat = {"done": "done", "running": f"running {fmt(el or 0)}" + (f", {fr * 100:.0f}%" if fr is not None else ""),
                "failed": "FAILED - will retry", "interrupted": "stopped mid-way, restarts", "waiting": f"waiting ~{fmt(est * factor * 60)}"}[s]
        t.add_row(Text(ic, style=col), Text(title, style=col if s != "waiting" else "grey50"), why, tb, Text(stat, style=col))
    p_steps = Panel(t, title="All steps, in order", title_align="left", border_style="green")

    # ---- 5. results so far -------------------------------------------------------------------------------------------------
    r = Text()
    r.append("Question: does a model store a fact once, shared across languages, or separately per language?\n\n", style="bold")
    r.append("So far (numbers from the Oct 19 status report; new runs not yet added):\n")
    r.append("• English and Spanish use the SAME layers, clearly more than chance, in both models (Llama z=4.2, Gemma z=4.8).\n", style="green")
    r.append("• Hindi matches less; it depends on how many pieces the Hindi name is split into.\n", style="yellow")
    r.append("• Editing a fact in English only: Spanish also changes 41-48% of the time, Hindi only 18-20%.\n", style="yellow")
    r.append("• Does mid-layer editing transfer better? Yes for Gemma, no for Llama (model-dependent).\n")
    r.append("\nThe steps above test whether these hold for more fact types, wordings, a prompt-only baseline, and a third model.", style="dim")
    p_res = Panel(r, title="What we have found so far", title_align="left", border_style="blue")

    # ---- 6. watch list -----------------------------------------------------------------------------------------------------
    w = Text()
    warns = []
    if free < 2:
        warns.append(f"RAM is low ({free:.1f} GB free): close browsers/other apps, low RAM killed jobs before.")
    if g and g["temp"] > 85:
        warns.append(f"GPU is hot ({g['temp']:.0f} °C).")
    if not alive and todo:
        warns.append("Experiments are not running.")
    fails = [p[1] for p in PLAN if state[p[0]][0] == "failed"]
    if fails:
        warns.append("Failed step(s): " + ", ".join(fails) + " (retried automatically on restart).")
    if not warns:
        w.append("Nothing to worry about.", style="green")
    for x in warns:
        w.append("! " + x + "\n", style="bold red")
    p_warn = Panel(w, title="Things to watch", title_align="left", border_style="red" if warns else "green")

    # ---- 7. glossary -------------------------------------------------------------------------------------------------------
    gl = Text()
    for term, mean in (("Layer", "one processing stage inside the model (Llama has 28, Gemma 26); facts are looked up in particular ones"),
                       ("Tracing", "blur the subject, restore the model's state one layer at a time, see which layer brings the answer back"),
                       ("Overlap / z-score", "how similar two languages' patterns are compared with random pairing; z above 2 = unlikely to be chance"),
                       ("Edit", "overwrite a fact inside the model (rank-one change to one layer)"),
                       ("Transfer", "does an English edit also change the Spanish/Hindi answer?"),
                       ("Baseline", "a simple comparison method, here: putting the new fact in the prompt")):
        gl.append(term + ": ", style="bold"); gl.append(mean + "\n", style="dim")
    p_gloss = Panel(gl, title="Words used here", title_align="left", border_style="grey50")

    foot = Text("Ctrl+C closes this window's view only; it does not stop the experiments.", style="dim")
    if wide:
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1); grid.add_column(ratio=1)
        grid.add_row(Group(p_status, p_now, p_graph), Group(p_steps, p_res, p_warn, p_gloss))
        return Group(grid, foot)
    return Group(p_status, p_now, p_graph, p_steps, p_res, p_warn, p_gloss, foot)


def main() -> None:
    if "--once" in sys.argv:
        con = Console(width=190)
        con.print(build(190))
        return
    con = Console()
    with Live(build(con.size.width), console=con, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(2)
                live.update(build(con.size.width))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
