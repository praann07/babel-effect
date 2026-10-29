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
        driver.LAUNCH_GAP = 0.0
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

    def test_gpu_jobs_start_with_a_gap(self):
        marks = Path(self.tmp.name)
        code = f"import time,pathlib; pathlib.Path(r'{marks}', 'gNAME').write_text(str(time.time())); time.sleep(0.5)"
        driver.LAUNCH_GAP = 1.5
        jobs = [self.job("x", code.replace("NAME", "x")), self.job("y", code.replace("NAME", "y"))]
        driver.run_jobs(jobs, [0, 1], dict(os.environ), set(), time.time() + 120, poll=0.05)
        t = sorted(float(p.read_text()) for p in marks.glob("g[xy]"))
        self.assertEqual(len(t), 2)
        self.assertGreaterEqual(t[1] - t[0], 1.4)

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

    def test_cpu_jobs_run_beside_gpu_jobs_without_taking_a_gpu(self):
        marks = Path(self.tmp.name)
        code = (f"import time,pathlib; t=time.time(); time.sleep(1.0); "
                f"pathlib.Path(r'{marks}', 'NAME').write_text(f'{{t}} {{time.time()}}')")
        jobs = [self.job("g", code.replace("NAME", "g")), driver.Job("c", lambda a: [PY, "-c", code.replace("NAME", "c")], gpu=False)]
        driver.run_jobs(jobs, [0], dict(os.environ), set(), time.time() + 120, poll=0.05)
        spans = {p.name: [float(x) for x in p.read_text().split()] for p in marks.glob("[gc]")}
        self.assertEqual(sorted(spans), ["c", "g"])
        self.assertLess(max(s for s, e in spans.values()), min(e for s, e in spans.values()))

    def test_expand_tokens_and_missing_input_fails_job_without_retries(self):
        (driver.WORK / "analysis" / "run_20261030_kgtrace_llama").mkdir()
        (driver.WORK / "analysis" / "run_20261031_kgtrace_llama").mkdir()
        cmd = driver.expand(["{py}", "x.py", "--trace-run", "{run:kgtrace_llama}", "--tag", "{tag:kgedit}", "{resume:kgedit_llama}"], 2)
        self.assertEqual(cmd[0], PY)
        self.assertEqual(cmd[3], "analysis/run_20261031_kgtrace_llama")
        self.assertTrue(cmd[5].startswith("kgeditr") and cmd[5].endswith("a2"))
        self.assertEqual(len(cmd), 6)  # no unfinished edit folder: no --resume
        d = driver.WORK / "analysis" / "run_20261031_kgedit_llama"
        d.mkdir(); (d / "config.json").write_text("{}")
        self.assertEqual(driver.expand(["{resume:kgedit_llama}"], 1), ["--resume", "analysis/run_20261031_kgedit_llama"])
        jobs = [driver.Job("m", lambda a: driver.expand(["{py}", "-c", "pass", "{run:nothere}"], a))]
        driver.run_jobs(jobs, [0], dict(os.environ), set(), time.time() + 60, poll=0.05)
        self.assertEqual((jobs[0].state, jobs[0].attempts), ("failed", 0))

    def test_deadline_stops_jobs_and_leaves_them_pending(self):
        jobs = [self.job("slow", "import time; time.sleep(30)")]
        driver.run_jobs(jobs, [0], dict(os.environ), set(), time.time() + 0.5, poll=0.05)
        self.assertEqual(jobs[0].state, "pending")


class FormatTests(unittest.TestCase):
    @staticmethod
    def res(tops, grads, secs, finite=True):
        return {"rows": [{"top1": t, "finite": finite} for t in tops], "grads": grads, "seconds_3_passes": secs}

    def test_fp16_accepted_only_when_faster_and_consistent(self):
        g = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(driver.compare_formats(self.res([1, 2], g, 10), self.res([1, 2], g, 2))[0], "float16")
        self.assertEqual(driver.compare_formats(self.res([1, 2], g, 10), self.res([1, 2], g, 8))[0], "bfloat16")  # not 1.5x faster
        self.assertEqual(driver.compare_formats(self.res([1, 2], g, 10), self.res([1, 3], g, 2))[0], "bfloat16")  # top-1 disagrees
        self.assertEqual(driver.compare_formats(self.res([1, 2], g, 10), self.res([1, 2], [[0.0, 1.0], [1.0, 0.0]], 2))[0], "bfloat16")  # gradients differ
        self.assertEqual(driver.compare_formats(self.res([1, 2], g, 10), self.res([1, 2], g, 2, finite=False))[0], "bfloat16")
        self.assertEqual(driver.compare_formats(None, self.res([1, 2], g, 2))[0], "float16")
        with self.assertRaises(driver.Fatal):
            driver.compare_formats(None, None)

    def test_dtype_check_on_a_tiny_model(self):
        import torch
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        import dtype_check
        cfg = HookedTransformerConfig(n_layers=2, d_model=16, d_head=4, n_heads=4, d_mlp=32, d_vocab=50, n_ctx=16, act_fn="gelu", device="cpu")
        model = HookedTransformer(cfg)
        out = dtype_check.check(model, [torch.randint(0, 50, (1, 5)) for _ in range(3)], "cpu", passes=1)
        self.assertEqual(len(out["rows"]), 3)
        self.assertTrue(all(r["finite"] for r in out["rows"]))
        self.assertEqual(len(out["grads"][0]), 16)
        self.assertGreater(sum(abs(v) for v in out["grads"][0]), 0)


class SetupTests(unittest.TestCase):
    def test_package_probe_runs_in_this_environment(self):
        try:
            import scipy  # noqa: F401  (Kaggle has it; the laptop venv may not)
        except ImportError:
            self.skipTest("scipy not installed here")
        r = driver.sh([PY, "-c", driver.PROBE])
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("transformer_lens 3.9.0", r.stdout)

    def test_token_is_exported_for_jobs_and_missing_token_stops_early(self):
        old = {k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        try:
            with self.assertRaises(driver.Fatal) as cm:
                driver.get_token()
            self.assertIn("HF_TOKEN", str(cm.exception))
            os.environ["HF_TOKEN"] = "dummy"
            self.assertEqual(driver.get_token(), "dummy")
            self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "dummy")
        finally:
            for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                os.environ.pop(k, None)
                if old[k] is not None:
                    os.environ[k] = old[k]


class LayoutTests(unittest.TestCase):
    def test_inputs_only_from_a_full_checkout(self):
        """A made-up checkout holding laptop results next to the code: only code, tests, runner and the seed lists may come through."""
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            repo, out = Path(t1), Path(t2)
            for rel in ("code/common.py", "tests/test_x.py", "kaggle/driver.py", "kaggle/steps.py", "kaggle/notebooks/n.ipynb", "kaggle/private/p.ipynb",
                        "kaggle/__pycache__/driver.pyc", "kaggle/x.log", "data/v2/facts_candidates.json", "data/v2/facts_candidates_extra.json",
                        "data/v2/facts_candidates_a23.json", "data/v2/facts_llama.json", "data/v2/templates.json", "results/manifest.json",
                        "analysis/run_20261001_x/records.jsonl", "docs/LOG.md", "models/m/model.safetensors", "paper/paper.md"):
                f = repo / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("x")
            n = driver.layout.assemble(repo, out)
            names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
            self.assertEqual(n, len(names))
            self.assertEqual(names, ["code/common.py", "data/v2/facts_candidates_a23.json", "data/v2/facts_candidates_extra.json",
                                     "data/v2/facts_candidates_seed.json", "kaggle/driver.py", "kaggle/steps.py", "tests/test_x.py"])
            self.assertTrue((out / "analysis" / "_logs").is_dir())

    def test_old_code_in_the_working_folder_is_replaced_but_results_are_kept(self):
        """A second run in the same Kaggle session finds the first run's folder: new code must win, results must stay."""
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            repo, work = Path(t1), Path(t2)
            for rel, txt in (("code/common.py", "new"), ("tests/test_x.py", "new"), ("kaggle/driver.py", "new"), ("data/v2/facts_candidates.json", "seed"),
                             ("data/v2/facts_candidates_extra.json", "e"), ("data/v2/facts_candidates_a23.json", "a")):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text(txt)
            for rel, txt in (("code/common.py", "old"), ("code/gone.py", "old"), ("tests/test_x.py", "old"), ("analysis/run_1/records.jsonl", "keep"),
                             ("data/v2/facts_llama.json", "keep")):
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                (work / rel).write_text(txt)
            old = (driver.WORK, os.environ.get("BABEL_SRC"))
            driver.WORK, os.environ["BABEL_SRC"] = work, str(repo)
            try:
                driver.place_code()
            finally:
                driver.WORK = old[0]
                os.environ.pop("BABEL_SRC", None)
                if old[1] is not None:
                    os.environ["BABEL_SRC"] = old[1]
            self.assertEqual((work / "code" / "common.py").read_text(), "new")
            self.assertEqual((work / "tests" / "test_x.py").read_text(), "new")
            self.assertFalse((work / "code" / "gone.py").exists())
            self.assertEqual((work / "analysis" / "run_1" / "records.jsonl").read_text(), "keep")
            self.assertEqual((work / "data" / "v2" / "facts_llama.json").read_text(), "keep")


class StepsTests(unittest.TestCase):
    def test_every_step_is_well_formed(self):
        from steps import STEPS
        for sid, plan in STEPS.items():
            names = [j["name"] for j in plan["jobs"]]
            self.assertEqual(len(names), len(set(names)), sid)
            for j in plan["jobs"]:
                self.assertTrue(set(j.get("deps", [])) <= set(names), (sid, j["name"]))
                self.assertIn(j["model"], driver.HF, (sid, j["name"]))
                self.assertEqual(j["cmd"][0], "{py}")
                self.assertTrue((Path(__file__).resolve().parent.parent / j["cmd"][1]).exists(), j["cmd"][1])
            self.assertTrue(set(plan["calibrate"]) <= set(plan["models"]), sid)


if __name__ == "__main__":
    unittest.main()
