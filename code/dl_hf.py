"""Resumable chunked downloader for Hugging Face files (the stock downloader stalls at 0 bytes on this network).

Usage: python code/dl_hf.py REPO OUTDIR FILE [FILE ...] [--token-file PATH]
Each file is fetched in 16 MiB Range requests appended to OUTDIR/FILE; re-running resumes from the file size.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

CHUNK = 16 * 2**20


def fetch(repo: str, name: str, out: Path, token: str | None) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    dest = out / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, headers=hdr, allow_redirects=True, timeout=60)
    head.raise_for_status()
    if "Content-Length" not in head.headers:  # small files served without a length: fetch whole
        r = requests.get(url, headers=hdr, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"{name}: whole file ({len(r.content)} B)", flush=True)
        return
    total = int(head.headers["Content-Length"])
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            print(f"{name}: complete ({total / 2**20:.0f} MiB)", flush=True)
            return
        end = min(have + CHUNK, total) - 1
        try:
            r = requests.get(url, headers={**hdr, "Range": f"bytes={have}-{end}"}, timeout=60)
            if r.status_code not in (200, 206) or (r.status_code == 200 and have):
                raise RuntimeError(f"status {r.status_code}")
            with open(dest, "ab") as fh:
                fh.write(r.content[: end - have + 1])
            print(f"{name}: {(have + len(r.content)) / 2**20:.0f}/{total / 2**20:.0f} MiB", flush=True)
        except Exception as e:  # network hiccup: retry the same chunk
            print(f"{name}: retry after {type(e).__name__}", flush=True)
            time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out", type=Path)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--token-file", type=Path, default=None)
    a = ap.parse_args()
    token = a.token_file.read_text().strip() if a.token_file else None
    for f in a.files:
        fetch(a.repo, f, a.out, token)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
