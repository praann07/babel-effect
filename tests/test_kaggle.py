"""Tests for the Kaggle runner: job scheduler, model preparation, layout, steps (CPU only, fake jobs and a made-up Kaggle Model)."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import driver  # noqa: E402

PY = sys.executable


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
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

    def test_a_failed_job_prints_its_error_and_a_done_job_prints_its_warning(self):
        import contextlib
        import io
        jobs = [self.job("bad", "print('Traceback (most recent call last):'); print('RuntimeError: boom'); import sys; sys.exit(1)", max_attempts=1),
                self.job("warn", "print('progress line'); print('WARNING: key modes that failed: joint_all x2')")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            driver.run_jobs(jobs, [0, 1], dict(os.environ), set(), time.time() + 120, poll=0.05)
        text = buf.getvalue()
        self.assertIn("    | RuntimeError: boom", text)
        self.assertIn("    WARNING: key modes that failed: joint_all x2", text)
        self.assertNotIn("progress line", text)

    def test_gpu_jobs_start_with_a_gap(self):
        marks = Path(self.tmp.name)
        code = f"import time,pathlib; pathlib.Path(r'{marks}', 'gNAME').write_text(str(time.time())); time.sleep(0.5)"
        driver.LAUNCH_GAP = 3.0  # the marks are written after each process has started, so their spacing varies by a few tenths of a second
        jobs = [self.job("x", code.replace("NAME", "x")), self.job("y", code.replace("NAME", "y"))]
        driver.run_jobs(jobs, [0, 1], dict(os.environ), set(), time.time() + 120, poll=0.05)
        t = sorted(float(p.read_text()) for p in marks.glob("g[xy]"))
        self.assertEqual(len(t), 2)
        self.assertGreaterEqual(t[1] - t[0], 2.0)  # without the gap both start within about half a second

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

    def test_progress_is_saved_after_each_finished_job_and_while_running(self):
        """A cut connection or a lost session must not lose finished jobs: the checkpoint runs after every job and every few seconds while jobs run."""
        saved: list[tuple[str, int]] = []
        done: set[str] = set()
        jobs = [self.job("a", "pass"), self.job("b", "import time; time.sleep(1.5)", deps=("a",))]
        driver.run_jobs(jobs, [0], dict(os.environ), done, time.time() + 60, poll=0.05, checkpoint=lambda why="": saved.append((why, len(done))),
                        checkpoint_every=0.4)
        self.assertIn(("1 jobs finished", 1), saved)
        self.assertIn(("2 jobs finished", 2), saved)
        self.assertTrue(any(why == "" for why, _ in saved))  # a save while job b was still running

    def test_runs_left_by_a_stopped_session_are_imported_without_the_shipped_laptop_runs(self):
        old = driver.INPUT
        try:
            driver.INPUT = Path(self.tmp.name) / "input"
            for rel in ("nb/babel-effect/analysis/run_20261030_kgedit_llama/config.json", "nb/babel-effect/analysis/run_20261030_kgedit_llama/records.jsonl",
                        "nb/babel-effect/analysis/run_20261019_p3extra_10191057_llama/config.json", "nb/babel-effect/analysis/run_x_nocfg/records.jsonl"):
                f = driver.INPUT / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("{}")
            driver.import_previous()
            got = sorted(p.name for p in (driver.WORK / "analysis").glob("run_*"))
            self.assertEqual(got, ["run_20261030_kgedit_llama"])
            self.assertTrue((driver.WORK / "analysis" / "run_20261030_kgedit_llama" / "records.jsonl").exists())
        finally:
            driver.INPUT = old

    def test_progress_text_for_edit_and_swap_jobs(self):
        run = driver.WORK / "analysis" / "run_20261030_kgedit_llama"
        run.mkdir()
        (run / "config.json").write_text("{}")
        (run / "records.jsonl").write_text("\n".join(json.dumps({"fact": f"f{i // 2}", "layer": 14 + i % 2}) for i in range(8)) + "\n")
        j = driver.Job("s:edit", lambda a: [])
        j.cmd, j.t0 = [PY, "code/knowledge_edit_v2.py", "--n-edit", "40", "--skip-first", "10", "--tag", "kgedit"], time.time() - 3600
        txt = driver.progress(j)
        self.assertIn("4/30", txt)
        self.assertIn("8 edits written", txt)
        self.assertIn("left", txt)
        j.cmd = [PY, "code/knowledge_edit_v2.py", "--n-edit", "40", "--skip-first", "10", "--resume", "analysis/run_20261030_kgedit_llama"]
        self.assertNotIn("left", driver.progress(j))  # a resumed run has no fair rate
        k = driver.Job("s:swap", lambda a: [])
        k.cmd, k.t0 = [PY, "code/interchange.py"], time.time() - 600
        (driver.WORK / "analysis" / "_logs" / "s_swap.log").write_text("some line\n10 / 200\n40 / 200\nwarning x\n")
        self.assertIn("40/200", driver.progress(k))
        self.assertEqual(driver.bar(5, 10, 10), "[#####-----] 5/10")

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
        self.assertTrue(all(r["finite_corrupted"] for r in out["rows"]))

    def test_corrupted_finite_catches_an_overflow_that_only_noise_causes(self):
        """Gemma-2 in fp16 may overflow only when the embeddings carry the tracing noise; the format check must see that and fall back to bf16."""
        import torch
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        import dtype_check
        cfg = HookedTransformerConfig(n_layers=2, d_model=16, d_head=4, n_heads=4, d_mlp=32, d_vocab=50, n_ctx=16, act_fn="gelu", device="cpu")
        model = HookedTransformer(cfg)
        toks = torch.randint(0, 50, (1, 5))
        self.assertTrue(dtype_check.corrupted_finite(model, toks, "cpu"))
        seen = []

        def overflowing(t, fwd_hooks):  # stands in for a model whose logits overflow as soon as the embeddings are corrupted
            seen.append([h[0] for h in fwd_hooks])
            return torch.full((1, t.shape[1], 50), float("nan"))

        model.run_with_hooks = overflowing
        self.assertFalse(dtype_check.corrupted_finite(model, toks, "cpu"))
        self.assertEqual(seen, [["hook_embed"]])


class SetupTests(unittest.TestCase):
    def test_package_probe_runs_in_this_environment(self):
        try:
            import scipy  # noqa: F401  (Kaggle has it; the laptop venv may not)
        except ImportError:
            self.skipTest("scipy not installed here")
        r = driver.sh([PY, "-c", driver.PROBE])
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("transformer_lens 3.9.0", r.stdout)


class ModelTests(unittest.TestCase):
    """prepare_model on a made-up Kaggle Model: weights are linked from it, config/tokenizer come from the repo archive, and the result must match the laptop fingerprint."""

    def build(self, root: Path, tamper: bool = False):
        import json
        import tarfile
        from test_model_manifest import write_safetensors
        import model_manifest
        laptop = root / "laptop"
        laptop.mkdir()
        t = {"a": bytes(range(8)), "b": bytes(range(8, 16))}
        write_safetensors(laptop / "m.safetensors", t)
        for n in model_manifest.SMALL_FILES + ("model.safetensors.index.json",):
            (laptop / n).write_text('{"laptop": "%s"}' % n)
        work = root / "work"
        (work / "data" / "model_files").mkdir(parents=True)
        (work / "data" / "model_manifests").mkdir(parents=True)
        with tarfile.open(work / "data" / "model_files" / "llama-3.2-3b-instruct.tar.gz", "w:gz") as tf:
            for n in model_manifest.SMALL_FILES + ("model.safetensors.index.json",):
                tf.add(laptop / n, arcname=n)
        (work / "data" / "model_manifests" / "llama-3.2-3b-instruct.json").write_text(json.dumps(model_manifest.fingerprint(laptop)))
        official = root / "input" / "models" / "metaresearch" / "llama-3.2" / "transformers" / "3b-instruct" / "1"
        official.mkdir(parents=True)
        # Meta's copy: other split of the shards, other config/tokenizer bytes (as on the real Kaggle Model)
        write_safetensors(official / "model-00001-of-00002.safetensors", {"a": t["a"]})
        write_safetensors(official / "model-00002-of-00002.safetensors", {"b": (bytes([99]) + t["b"][1:]) if tamper else t["b"]})
        (official / "config.json").write_text("{}")
        return work, official

    def run_prepare(self, work: Path, root: Path):
        old = (driver.WORK, driver.INPUT, driver.MODEL_ROOT)
        driver.WORK, driver.INPUT, driver.MODEL_ROOT = work, root / "input", root / "models_out"
        try:
            driver.prepare_model("llama")
            return driver.OFFICIAL, driver.MODEL_ROOT / "llama-3.2-3b-instruct"
        finally:
            driver.WORK, driver.INPUT, driver.MODEL_ROOT = old

    def test_identical_weights_give_a_model_folder_with_the_laptop_config(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            work, official = self.build(root)
            got_official, target = self.run_prepare(work, root)
            self.assertEqual(got_official, official)
            self.assertEqual(sorted(p.name for p in target.glob("*.safetensors")), ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
            self.assertEqual((target / "config.json").read_text(), '{"laptop": "config.json"}')  # the repo's file, not Meta's

    def test_one_changed_weight_byte_stops_the_run(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            work, _ = self.build(root, tamper=True)
            with self.assertRaises(driver.Fatal) as cm:
                self.run_prepare(work, root)
            self.assertIn("tensor b: bytes differ", str(cm.exception))

    def test_missing_model_is_a_clear_error(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            old = driver.INPUT
            driver.INPUT = Path(t)
            try:
                with self.assertRaises(driver.Fatal) as cm:
                    driver.find_official("llama")
                self.assertIn("Add Input", str(cm.exception))
            finally:
                driver.INPUT = old

    def attach(self, root: Path, *rels: str) -> None:
        for rel in rels:
            d = root / "input" / rel
            d.mkdir(parents=True)
            (d / "config.json").write_text("{}")
            (d / "model-00001-of-00001.safetensors").write_bytes(b"x")

    def find(self, root: Path, key: str) -> Path:
        old = driver.INPUT
        driver.INPUT = root / "input"
        try:
            return driver.find_official(key)
        finally:
            driver.INPUT = old

    def test_gemma_base_is_found_by_its_whole_name_never_the_instruct_or_keras_variation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            self.attach(root, "gemma-2/transformers/gemma-2-2b-it/1", "gemma-2/transformers/gemma-2-9b/1")
            with self.assertRaises(driver.Fatal) as cm:
                self.find(root, "gemma2")
            self.assertIn("gemma-2/transformers/gemma-2-2b-it/1", str(cm.exception))  # says what it did find
            self.assertIn("gemma-2-2b (the base model", str(cm.exception))
            self.attach(root, "gemma-2/transformers/gemma-2-2b/1")
            self.assertEqual(self.find(root, "gemma2"), root / "input" / "gemma-2" / "transformers" / "gemma-2-2b" / "1")

    def test_a_dataset_named_gemma_2_2b_that_holds_the_18_layer_first_gemma_is_refused_with_the_reason(self):
        """leimeng46/gemma-2-2b on Kaggle has the file sizes of google/gemma-2b (first generation, 18 layers), not of Gemma-2-2B (26 layers)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            self.attach(root, "gemma-2-2b")
            (root / "input" / "gemma-2-2b" / "config.json").write_text(json.dumps({"num_hidden_layers": 18}))
            with self.assertRaises(driver.Fatal) as cm:
                self.find(root, "gemma2")
            self.assertIn("gemma-2-2b is a different model (18 layers, this one has 26)", str(cm.exception))
            self.attach(root, "gemma-2/transformers/gemma-2-2b/1")
            (root / "input" / "gemma-2" / "transformers" / "gemma-2-2b" / "1" / "config.json").write_text(json.dumps({"num_hidden_layers": 26}))
            self.assertEqual(self.find(root, "gemma2"), root / "input" / "gemma-2" / "transformers" / "gemma-2-2b" / "1")

    def test_the_makers_layout_wins_over_a_dataset_of_the_same_name_and_a_keras_copy_has_no_safetensors(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            self.attach(root, "gemma-2-2b", "gemma-2/transformers/gemma-2-2b/1")
            (root / "input" / "gemma-2" / "keras" / "gemma2_2b_en" / "1").mkdir(parents=True)
            (root / "input" / "gemma-2" / "keras" / "gemma2_2b_en" / "1" / "config.json").write_text("{}")
            self.assertEqual(self.find(root, "gemma2"), root / "input" / "gemma-2" / "transformers" / "gemma-2-2b" / "1")
            self.assertEqual(self.find(root, "gemma2").name, "1")

    def test_a_dataset_copy_with_other_bytes_stops_the_run_before_any_job(self):
        """A re-saved half-precision copy (as a user-made dataset may be) is not the laptop weights: the byte check stops the run."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            work, official = self.build(root)
            (official / "model-00002-of-00002.safetensors").unlink()
            from test_model_manifest import write_safetensors
            write_safetensors(official / "model-00002-of-00002.safetensors", {"b": bytes(8)})
            with self.assertRaises(driver.Fatal) as cm:
                self.run_prepare(work, root)
            self.assertIn("not identical", str(cm.exception))

    def test_the_index_of_the_attached_shards_is_used_when_it_exists(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            root = Path(t)
            work, official = self.build(root)
            (official / "model.safetensors.index.json").write_text('{"official": true}')
            _, target = self.run_prepare(work, root)
            self.assertEqual((target / "model.safetensors.index.json").read_text(), '{"official": true}')


class PlotTests(unittest.TestCase):
    def test_the_mid_layer_label_comes_from_the_edit_run_of_the_model(self):
        import make_plots
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(make_plots.mid_layer(), "middle layer")
                d = Path("analysis/run_20261031_p4a28baser1a1_gemma2")
                d.mkdir(parents=True)
                (d / "config.json").write_text(json.dumps({"n_layers": 26}))
                self.assertEqual(make_plots.mid_layer(), "layer 13")
            finally:
                os.chdir(old)

    def test_graphs_are_written_from_made_up_results_and_missing_inputs_are_not_an_error(self):
        import make_plots
        ci = lambda m: {"median": m, "ci95": [m - 0.1, m + 0.1], "n": 10}  # noqa: E731
        cond = {"n": 10, **{k: ci(0.5) for k in ("en_para", "es_main", "es_para", "hi_main", "hi_para")}}
        key = {"es_held": ci(0.6), "hi_held": ci(0.3), "es_held_base": ci(0.4), "hi_held_base": ci(0.1)}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(make_plots.main(), 0)  # nothing to plot yet
                for name, res in (("para_analysis_A21", {"traced": cond, "mid": cond}), ("keymode_analysis_A24", {"mean_all": {"traced": key, "mid": key}, "joint_all": {"traced": key}})):
                    d = Path("analysis") / f"run_20261030_{name}"
                    d.mkdir(parents=True)
                    (d / "results.json").write_text(json.dumps(res))
                self.assertEqual(make_plots.main(), 0)
                self.assertTrue((Path("analysis/figures_kaggle/paraphrase_control.png")).stat().st_size > 5000)
                self.assertTrue((Path("analysis/figures_kaggle/shared_key_edit.png")).stat().st_size > 5000)
            finally:
                os.chdir(old)


class LayoutTests(unittest.TestCase):
    def test_inputs_only_from_a_full_checkout(self):
        """A made-up checkout: code, tests, runner, data and the saved tracing runs come through; models, other runs, docs and the paper do not."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t1, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t2:
            repo, out = Path(t1), Path(t2)
            for rel in ("code/common.py", "tests/test_x.py", "kaggle/driver.py", "kaggle/steps.py", "kaggle/notebooks/n.ipynb", "kaggle/private/p.ipynb",
                        "kaggle/__pycache__/driver.pyc", "kaggle/x.log", "data/v2/facts_llama.json", "data/v2/templates.json", "data/wikitext/t.parquet",
                        "data/audit/a.csv", "data/model_files/m.tar.gz", "analysis/run_20261019_p3extra_10191057_llama/config.json",
                        "analysis/run_20261019_p3extra_10191057_llama/grids/a.npz", "analysis/run_20261001_other/records.jsonl", "docs/LOG.md",
                        "models/m/model.safetensors", "paper/paper.md"):
                f = repo / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("x")
            n = driver.layout.assemble(repo, out)
            names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
            self.assertEqual(n, len(names))
            self.assertEqual(names, ["analysis/run_20261019_p3extra_10191057_llama/config.json", "analysis/run_20261019_p3extra_10191057_llama/grids/a.npz",
                                     "code/common.py", "data/model_files/m.tar.gz", "data/v2/facts_llama.json", "data/v2/templates.json",
                                     "kaggle/driver.py", "kaggle/steps.py", "tests/test_x.py"])
            self.assertTrue((out / "analysis" / "_logs").is_dir())

    def test_every_saved_run_the_spanish_analysis_reads_is_packed(self):
        """The Spanish swap-test analysis reads the laptop tracing runs (metric validity check); a run missing from the pack made it fail on Kaggle."""
        import controls_overlap
        wanted = {p for p in controls_overlap.SETS["llama"][:2]} | {"analysis/run_20261024_p3seeds_10240109_llama"}
        root = Path(__file__).resolve().parent.parent
        for p in wanted:
            self.assertIn(p, driver.layout.RUNS)
            if (root / "docs").is_dir() or (root / p).is_dir():  # a notebook of a step that does not read them (Gemma) does not carry the runs
                self.assertTrue((root / p).is_dir(), p)

    def test_a_models_notebook_carries_only_its_own_model_files_and_no_saved_runs_unless_asked(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            repo = Path(t)
            for rel in ("code/common.py", "data/model_files/llama-3.2-3b-instruct.tar.gz", "data/model_manifests/llama-3.2-3b-instruct.json",
                        "data/model_files/gemma-2-2b.tar.gz", "data/model_manifests/gemma-2-2b.json", "data/v2/facts_gemma2.json",
                        "analysis/run_20261019_p3extra_10191057_llama/config.json"):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text("x")
            for d in ("kaggle", "tests"):
                (repo / d).mkdir(exist_ok=True)
            names = lambda **kw: sorted(r for _, r in driver.layout.members(repo, **kw))  # noqa: E731
            self.assertEqual(len(names()), 7)  # no filter: everything
            self.assertEqual(names(models=["gemma2"], runs=False), ["code/common.py", "data/model_files/gemma-2-2b.tar.gz", "data/model_manifests/gemma-2-2b.json",
                                                                    "data/v2/facts_gemma2.json"])
            self.assertEqual(names(models=["llama"]), ["analysis/run_20261019_p3extra_10191057_llama/config.json", "code/common.py",
                                                       "data/model_files/llama-3.2-3b-instruct.tar.gz", "data/model_manifests/llama-3.2-3b-instruct.json",
                                                       "data/v2/facts_gemma2.json"])

    def test_old_code_in_the_working_folder_is_replaced_but_results_are_kept(self):
        """A second run in the same Kaggle session finds the first run's folder: new code must win, results must stay."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t1, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t2:
            repo, work = Path(t1), Path(t2)
            for rel, txt in (("code/common.py", "new"), ("tests/test_x.py", "new"), ("kaggle/driver.py", "new"), ("data/v2/templates.json", "fresh")):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text(txt)
            for rel, txt in (("code/common.py", "old"), ("code/gone.py", "old"), ("tests/test_x.py", "old"), ("analysis/run_1/records.jsonl", "keep"),
                             ("data/v2/templates.json", "stale"), ("data/v2/e2a23_facts_llama.json", "keep")):
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
            self.assertEqual((work / "data" / "v2" / "e2a23_facts_llama.json").read_text(), "keep")  # written by a job in the first run
            self.assertEqual((work / "data" / "v2" / "templates.json").read_text(), "fresh")  # committed data comes from the fetched commit


class NotebookTests(unittest.TestCase):
    def test_notebook_carries_the_code_and_unpacks_it(self):
        """The notebook file holds two cells; the first one decodes to exactly the repo files (not the notebook itself); rebuilding gives the same bytes."""
        import base64
        import hashlib
        import io
        import zipfile

        import make_notebook
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            repo = Path(t)
            for rel, txt in (("code/common.py", "c"), ("kaggle/driver.py", "d"), ("kaggle/notebook_cell.py", "print(PAYLOAD_SHA256)\n"), ("kaggle/old.ipynb", "x"),
                             ("data/v2/templates.json", "{}"), ("models/m/w.safetensors", "w")):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text(txt)
            out, digest, n = make_notebook.build(repo, repo / "nb.ipynb")
            first = out.read_bytes()
            nb = json.loads(first)
            self.assertEqual(len(nb["cells"]), 2)
            self.assertEqual("".join(nb["cells"][1]["source"]), "print(PAYLOAD_SHA256)\n")
            ns: dict = {}
            exec("".join(nb["cells"][0]["source"]), ns)
            blob = base64.b64decode("".join(ns["PAYLOAD"].split()))
            self.assertEqual(hashlib.sha256(blob).hexdigest(), ns["PAYLOAD_SHA256"])
            self.assertEqual(ns["PAYLOAD_SHA256"], digest)
            self.assertEqual(sorted(zipfile.ZipFile(io.BytesIO(blob)).namelist()), ["code/common.py", "data/v2/templates.json", "kaggle/driver.py", "kaggle/notebook_cell.py"])
            self.assertEqual(make_notebook.build(repo, repo / "nb2.ipynb")[0].read_bytes(), first)


    def test_the_gemma_notebook_runs_the_gemma_step_and_packs_only_gemma_files(self):
        import base64
        import io
        import zipfile

        import make_notebook
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            repo = Path(t)
            for rel, txt in (("code/common.py", "c"), ("kaggle/notebook_cell.py", 'STEP = "babel_llama"  # x\nprint(STEP)\n'), ("data/model_files/gemma-2-2b.tar.gz", "g"),
                             ("data/model_files/llama-3.2-3b-instruct.tar.gz", "l"), ("analysis/run_20261015_p3trace_v2_llama/config.json", "{}")):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text(txt)
            (repo / "tests").mkdir()
            out, digest, _ = make_notebook.build(repo, repo / "g.ipynb", "babel_gemma2", ["gemma2"], False)
            nb = json.loads(out.read_bytes())
            self.assertEqual("".join(nb["cells"][1]["source"]), 'STEP = "babel_gemma2"  # x\nprint(STEP)\n')
            ns: dict = {}
            exec("".join(nb["cells"][0]["source"]), ns)
            names = sorted(zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(ns["PAYLOAD"].split())))).namelist())
            self.assertEqual(names, ["code/common.py", "data/model_files/gemma-2-2b.tar.gz", "kaggle/notebook_cell.py"])
            self.assertEqual(json.loads(make_notebook.build(repo, repo / "l.ipynb")[0].read_bytes())["cells"][1]["source"][0], 'STEP = "babel_llama"  # x\n')
        self.assertEqual({s: v[1] for s, v in make_notebook.NOTEBOOKS.items()}, {"babel_llama": ["llama"], "babel_gemma2": ["gemma2"]})
        from steps import STEPS
        self.assertEqual(set(make_notebook.NOTEBOOKS), {s for s in STEPS if s != "babel_llama_check"})


class StepsTests(unittest.TestCase):
    def test_every_step_is_well_formed(self):
        from steps import STEPS
        for sid, plan in STEPS.items():
            names = [j["name"] for j in plan["jobs"]]
            self.assertEqual(len(names), len(set(names)), sid)
            for j in plan["jobs"]:
                self.assertTrue(set(j.get("deps", [])) <= set(names), (sid, j["name"]))
                self.assertIn(j["model"], driver.MODELS, (sid, j["name"]))
                self.assertEqual(j["cmd"][0], "{py}")
                self.assertTrue((Path(__file__).resolve().parent.parent / j["cmd"][1]).exists(), j["cmd"][1])
            self.assertTrue(set(plan["calibrate"]) <= set(plan["models"]), sid)

    def test_smoke_prompt_suits_the_model_type(self):
        """Gemma-2-2B is a base model and answers 'The capital of France is' with ' a' (checked on the CPU), which stopped its first Kaggle run at the smoke job.
        Its smoke job carries a question prompt; Llama-Instruct keeps the default one."""
        from steps import STEPS
        cmd = {sid: next(j["cmd"] for j in STEPS[sid]["jobs"] if j["name"] == "smoke") for sid in ("babel_llama", "babel_gemma2")}
        self.assertNotIn("--prompt", cmd["babel_llama"])
        self.assertEqual(cmd["babel_gemma2"][-2], "--prompt")
        self.assertIn("Answer:", cmd["babel_gemma2"][-1])
        self.assertNotIn("{", cmd["babel_gemma2"][-1])

    def test_each_models_step_uses_only_its_own_model_and_files(self):
        """A leftover 'llama' in the Gemma step would run the wrong model for hours; every file the step needs must exist in the checkout."""
        from steps import MODEL, STEPS
        root = Path(__file__).resolve().parent.parent
        for sid, key in (("babel_llama", "llama"), ("babel_gemma2", "gemma2")):
            plan = STEPS[sid]
            self.assertEqual(plan["models"], [key])
            self.assertEqual(driver.MODELS[key][0], MODEL[key]["dir"])
            self.assertEqual(driver.layout.MODEL_NAMES[key], MODEL[key]["dir"])
            full = (root / "docs").is_dir()  # a checkout; a Kaggle notebook carries only its own model's files
            for p in plan["needs"]:
                if full or (root / plan["needs"][0]).exists():
                    self.assertTrue((root / p).exists(), p)
            for j in plan["jobs"]:
                self.assertEqual(j["model"], key, (sid, j["name"]))
                if key != "llama":
                    self.assertFalse(any("llama" in tok for tok in j["cmd"]), (sid, j["name"], j["cmd"]))

    def test_gemma_edit_cap_is_its_real_number_of_facts_and_the_mid_layer_is_half_the_layers(self):
        """The cap of the Gemma edit jobs is also the denominator of their progress bar, so it must equal the facts that qualify (relations with >= 10 kept facts)."""
        import json as _json
        from causal_trace_v2 import select_edit_facts
        from steps import MODEL, STEPS
        root = Path(__file__).resolve().parent.parent
        facts = _json.loads((root / MODEL["gemma2"]["facts"]).read_text(encoding="utf8"))["facts"]
        self.assertEqual(len(select_edit_facts(facts, 10 ** 6)), MODEL["gemma2"]["n_edit"])
        cfg = _json.loads((root / "models" / "gemma-2-2b" / "config.json").read_text(encoding="utf8")) if (root / "models" / "gemma-2-2b" / "config.json").exists() else {"num_hidden_layers": 26}
        self.assertEqual(MODEL["gemma2"]["mid"], cfg["num_hidden_layers"] // 2)
        self.assertEqual(driver.MODELS["gemma2"][3], cfg["num_hidden_layers"])
        self.assertEqual(driver.MODELS["llama"][3], 28)
        self.assertEqual(MODEL["llama"]["mid"], 28 // 2)
        edit = next(j for j in STEPS["babel_gemma2"]["jobs"] if j["name"] == "edit_base")["cmd"]
        self.assertEqual(edit[edit.index("--layers") + 1], "13")
        self.assertEqual(edit[edit.index("--n-edit") + 1], "52")

    def test_the_gemma_weight_fingerprint_covers_every_tensor_of_the_laptop_index(self):
        import json as _json
        root = Path(__file__).resolve().parent.parent
        idx = root / "models" / "gemma-2-2b" / "model.safetensors.index.json"
        if not idx.exists():
            self.skipTest("laptop model folder not present")
        names = set(_json.loads(idx.read_text(encoding="utf8"))["weight_map"])
        have = set(_json.loads((root / "data" / "model_manifests" / "gemma-2-2b.json").read_text(encoding="utf8"))["tensors"])
        self.assertEqual(names, have)

    def test_every_run_a_job_reads_is_made_by_one_of_its_dependencies(self):
        """A mistyped folder pattern would only show up on Kaggle, hours in: each {run:..} must match a {tag:..} of a job it depends on, each {resume:..} its own."""
        import fnmatch
        from steps import STEPS
        for sid, plan in STEPS.items():
            jobs = {j["name"]: j for j in plan["jobs"]}

            def ancestors(n: str) -> set[str]:
                out: set[str] = set()
                for d in jobs[n].get("deps", []):
                    out |= {d} | ancestors(d)
                return out

            tags = {n: [t[5:-1] for t in j["cmd"] if t.startswith("{tag:")] for n, j in jobs.items()}
            all_tags = [t for v in tags.values() for t in v]
            self.assertEqual(len(all_tags), len(set(all_tags)), sid)
            for n, j in jobs.items():
                for tok in j["cmd"]:
                    if tok.startswith(("{run:", "{resume:")):
                        pat = tok.split(":", 1)[1][:-1]
                        pool = [t for a in (ancestors(n) if tok.startswith("{run:") else {n}) for t in tags[a]]
                        self.assertTrue(any(fnmatch.fnmatch(f"{t}r123456a1{s}", pat) for t in pool for s in ("", "_llama", "_gemma2")), (sid, n, tok))


if __name__ == "__main__":
    unittest.main()
