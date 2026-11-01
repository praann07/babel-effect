"""Does Meta's own tokenizer file (the one on Kaggle) give the same token ids as the file used for every laptop run (the Unsloth mirror's)?
Runs over every string in the fact, template and candidate files (English, Spanish, Hindi). CPU only. Writes a JSON summary; exit code 0
whatever the answer is, because the answer is a result to report, not an error."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def strings(node, out: set[str]) -> None:
    if isinstance(node, str):
        if 0 < len(node) < 400:
            out.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            strings(v, out)
    elif isinstance(node, list):
        for v in node:
            strings(v, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=Path, required=True, help="folder with the laptop tokenizer files")
    ap.add_argument("--official", type=Path, required=True, help="folder with Meta's tokenizer files (the Kaggle Model)")
    ap.add_argument("--data", default="data/v2/*.json")
    ap.add_argument("--out", type=Path, default=Path("analysis/tokenizer_compare.json"))
    a = ap.parse_args()
    if not (a.official / "tokenizer.json").exists() and not (a.official / "tokenizer.model").exists():
        print(f"no tokenizer files in {a.official}: nothing to compare (every number of the run uses the laptop tokenizer in {a.mirror})")
        return 0
    from transformers import AutoTokenizer
    texts: set[str] = set()
    files = sorted(glob.glob(a.data))
    for f in files:
        strings(json.loads(Path(f).read_text(encoding="utf8")), texts)
    texts_sorted = sorted(texts)
    mirror, official = AutoTokenizer.from_pretrained(a.mirror), AutoTokenizer.from_pretrained(a.official)
    res = {"files": len(files), "strings": len(texts_sorted), "differ_with_special_tokens": 0, "differ_without_special_tokens": 0, "examples": []}
    for s in texts_sorted:
        with_bos = mirror(s)["input_ids"] != official(s)["input_ids"]
        without = mirror(s, add_special_tokens=False)["input_ids"] != official(s, add_special_tokens=False)["input_ids"]
        res["differ_with_special_tokens"] += with_bos
        res["differ_without_special_tokens"] += without
        if (with_bos or without) and len(res["examples"]) < 15:
            res["examples"].append({"text": s, "mirror": mirror(s)["input_ids"], "official": official(s)["input_ids"]})
    res["vocab_sizes"] = [len(mirror), len(official)]
    res["bos"] = [mirror.bos_token_id, official.bos_token_id]
    res["pad"] = [mirror.pad_token_id, official.pad_token_id]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"{res['strings']} strings from {res['files']} files: ids differ for {res['differ_with_special_tokens']} (with BOS) and "
          f"{res['differ_without_special_tokens']} (without special tokens); bos {res['bos']}, pad {res['pad']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
