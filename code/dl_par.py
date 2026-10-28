"""Parallel ranged download of one HF file into OUT (resumable per 32 MiB block). Usage: dl_par.py REPO FILE OUTDIR TOKENFILE [workers]
Blocks are stored as OUTDIR/FILE.parts/<i>; when all exist they are concatenated into OUTDIR/FILE."""
import sys, time, threading, queue
from pathlib import Path
import requests
repo, name, out, tokf = sys.argv[1:5]; W = int(sys.argv[5]) if len(sys.argv) > 5 else 12
BLK = 32 * 2**20; tok = Path(tokf).read_text().strip(); hdr = {"Authorization": f"Bearer {tok}"}
url = f"https://huggingface.co/{repo}/resolve/main/{name}"
total = int(requests.head(url, headers=hdr, allow_redirects=True, timeout=60).headers["Content-Length"])
parts = Path(out) / (name + ".parts"); parts.mkdir(parents=True, exist_ok=True)
n = (total + BLK - 1) // BLK; q = queue.Queue()
for i in range(n): q.put(i)
def worker():
    while True:
        try: i = q.get_nowait()
        except queue.Empty: return
        p = parts / str(i); s = i * BLK; e = min(s + BLK, total) - 1
        if p.exists() and p.stat().st_size == e - s + 1: continue
        while True:
            try:
                r = requests.get(url, headers={**hdr, "Range": f"bytes={s}-{e}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == e - s + 1:
                    (parts / f"{i}.tmp").write_bytes(r.content); (parts / f"{i}.tmp").replace(p); break
            except Exception: pass
            time.sleep(2)
        print(f"{name} block {i+1}/{n}", flush=True)
ts = [threading.Thread(target=worker) for _ in range(W)]
[t.start() for t in ts]; [t.join() for t in ts]
with open(Path(out) / (name + ".full"), "wb") as fo:
    for i in range(n): fo.write((parts / str(i)).read_bytes())
(Path(out) / (name + ".full")).replace(Path(out) / name)
print(name, "ALL_DONE", flush=True)
