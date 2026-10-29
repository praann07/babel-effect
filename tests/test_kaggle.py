"""Tests for the Kaggle runner: shard merging and the job scheduler (CPU only, fake jobs)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kaggle"))

import driver  # noqa: E402
import merge_shards as ms  # noqa: E402

PY = sys.executable


class MergeTests(unittest.TestCase):
    def test_merge_edit_joins_shards_and_drops_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as t:
            ana = Path(t)
            for i, facts in enumerate((["a", "c"], ["b"])):
                d = ana / f"run_20261029_p4a21s{i}_llama"
                d.mkdir()
                (d / "config.json").write_text(json.dumps({"edit_ids": facts, "lr": 0.5}))
                rows = [{"fact": f, "layer": 14} for f in facts] + ([{"fact": "a", "layer": 14}] if i == 0 else [])
                (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = ms.merge_edit(ana, "p4a21", "llama", 2)
            self.assertEqual(out.name, "run_20261029_p4a21_llama")
            rows = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(r["fact"] for r in rows), ["a", "b", "c"])
            cfg = json.loads((out / "config.json").read_text())
            self.assertEqual(cfg["edit_ids"], ["a", "b", "c"])
            self.assertEqual(cfg["lr"], 0.5)

    def test_merge_interchange_concatenates_effects_and_pairs(self):
        with tempfile.TemporaryDirectory() as t:
            ana = Path(t)
            for i, n in enumerate((3, 2)):
                d = ana / f"run_20261029_e2a23_As{i}_llama"
                d.mkdir()
                np.savez_compressed(d / "effects.npz", delta=np.full((n, 4, 2, 28), i, float), flip=np.zeros((n, 4, 2, 28)))
                (d / "meta.json").write_text(json.dumps({"pairs": [{"target": f"t{i}{k}"} for k in range(n)], "n_layers": 28, "shard": f"{i}/2"}))
            out = ms.merge_interchange(ana, "e2a23_A", "llama", 2)
            self.assertEqual(out.name, "run_20261029_e2a23_A_llama")
            with np.load(out / "effects.npz") as z:
                delta = z["delta"]
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(delta.shape, (5, 4, 2, 28))
            self.assertEqual(len(meta["pairs"]), 5)
            self.assertEqual(meta["pairs"][3]["target"], "t10")
            self.assertTrue((delta[3:] == 1).all())

    def test_merge_interchange_refuses_unfinished_shard(self):
        with tempfile.TemporaryDirectory() as t:
            ana = Path(t)
            (ana / "run_20261029_e2a23_As0_llama").mkdir()
            with self.assertRaises(FileNotFoundError):
                ms.merge_interchange(ana, "e2a23_A", "llama", 1)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        driver.WORK = Path(self.tmp.name)
        (driver.WORK / "analysis" / "_logs").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, name, code, deps=(), max_attempts=3):
        return driver.Job(name, lambda attempt: [PY, "-c", code.replace("ATTEMPT", str(attempt))], deps, max_attempts)

    def test_dependencies_retries_and_skips(self):
        flag = Path(self.tmp.name) / "flag"
        jobs = [self.job("a", "pass"),
                self.job("b", f"import pathlib,sys; p=pathlib.Path(r'{flag}'); sys.exit(0 if p.exists() else (p.touch() or 1))"),  # fails once, then works
                self.job("c", "import sys; sys.exit(3)", max_attempts=2),
                self.job("d", "pass", deps=("c",)),
                self.job("e", "pass", deps=("a", "b"))]
        done: set[str] = set()
        driver.run_jobs(jobs, [0, 1], dict(os.environ), done, time.time() + 120, poll=0.05)
        st = {j.name: j.state for j in jobs}
        self.assertEqual(st, {"a": "done", "b": "done", "c": "failed", "d": "skipped", "e": "done"})
        self.assertEqual(done, {"a", "b", "e"})

    def test_both_gpus_busy_at_once_and_done_jobs_not_rerun(self):
        marks = Path(self.tmp.name)
        code = (f"import os,time,pathlib; t=time.time(); time.sleep(1.0); "
                f"pathlib.Path(r'{marks}', 'gpu'+os.environ['CUDA_VISIBLE_DEVICES']).write_text(f'{{t}} {{time.time()}}')")
        jobs = [self.job("x", code), self.job("y", code), self.job("z", "import sys; sys.exit(1)")]
        driver.run_jobs(jobs, [0, 1], dict(os.environ), {"z"}, time.time() + 120, poll=0.05)
        self.assertEqual({j.name: j.state for j in jobs}, {"x": "done", "y": "done", "z": "done"})
        spans = {p.name: [float(x) for x in p.read_text().split()] for p in marks.glob("gpu*")}
        self.assertEqual(sorted(spans), ["gpu0", "gpu1"])
        self.assertLess(max(s for s, e in spans.values()), min(e for s, e in spans.values()))  # the two jobs overlapped in time

    def test_deadline_stops_jobs_and_leaves_them_pending(self):
        jobs = [self.job("slow", "import time; time.sleep(30)")]
        driver.run_jobs(jobs, [0], dict(os.environ), set(), time.time() + 0.5, poll=0.05)
        self.assertEqual(jobs[0].state, "pending")


if __name__ == "__main__":
    unittest.main()
