"""Unit tests for the pure helpers (matching rules, overlap statistics). CPU only; run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from circuit_overlap import analyse, cos_matrix, iou_matrix, is_valid, spearman_matrix, stat_block  # noqa: E402
from matching import lenient_match, norm, strict_match  # noqa: E402


class MatchingTests(unittest.TestCase):
    def test_norm_folds_case_hyphen_and_whitespace(self):
        self.assertEqual(norm("  Buenos-Aires  "), "buenos aires")

    def test_norm_drops_nukta_and_zero_width(self):
        self.assertEqual(norm("फ़्रांस"), norm("फ्रांस"))
        self.assertEqual(norm("a​b"), "ab")

    def test_strict_accepts_prefix_after_spaces(self):
        self.assertEqual(strict_match("  Paris, France", ["Paris"]), "Paris")

    def test_strict_rejects_leading_newline(self):
        self.assertIsNone(strict_match("\nParis", ["Paris"]))
        self.assertIsNone(strict_match(" \nParis", ["Paris"]))

    def test_strict_requires_the_answer_first(self):
        self.assertIsNone(strict_match("The city of Paris", ["Paris"]))

    def test_strict_takes_the_longest_alternative(self):
        self.assertEqual(strict_match("New York City is", ["New York", "New York City"]), "New York City")

    def test_lenient_finds_answer_in_first_line_only(self):
        self.assertTrue(lenient_match("It is Paris.\nOther", ["Paris"]))
        self.assertFalse(lenient_match("Unknown\nParis", ["Paris"]))
        self.assertFalse(lenient_match("x" * 70 + "Paris", ["Paris"]))  # beyond 60 characters


class StatTests(unittest.TestCase):
    def test_cosine_of_identical_rows_is_one(self):
        A = np.random.default_rng(0).normal(size=(5, 7))
        self.assertTrue(np.allclose(np.diag(cos_matrix(A, A)), 1.0, atol=1e-6))

    def test_spearman_equals_pearson_of_ranks(self):
        rng = np.random.default_rng(1)
        A, B = rng.normal(size=(4, 12)), rng.normal(size=(4, 12))
        M = spearman_matrix(A, B)
        for i in range(4):
            self.assertAlmostEqual(M[i, i], float(np.corrcoef(A[i].argsort().argsort(), B[i].argsort().argsort())[0, 1]), places=6)

    def test_iou_identical_is_one_and_disjoint_is_zero(self):
        A = np.arange(20, dtype=float)[None, :]
        self.assertAlmostEqual(float(iou_matrix(A, A, 5)[0, 0]), 1.0)
        self.assertAlmostEqual(float(iou_matrix(A, -A, 5)[0, 0]), 0.0)

    def test_is_valid_needs_collapse_correct_object_and_positive_effect(self):
        ok = {"collapsed": True, "obj_ok": True, "score_clean": -1.0, "score_corr": -5.0}
        self.assertTrue(is_valid(ok))
        self.assertFalse(is_valid({**ok, "collapsed": False}))
        self.assertFalse(is_valid({**ok, "obj_ok": False}))
        self.assertFalse(is_valid({**ok, "score_corr": -0.5}))

    def test_stat_block_detects_signal_and_not_noise(self):
        rng = np.random.default_rng(2)
        n = 40
        base = rng.normal(size=(n, 30))
        signal = cos_matrix(base, base + 0.3 * rng.normal(size=(n, 30)))
        noise = cos_matrix(base, rng.normal(size=(n, 30)))
        perms = np.stack([rng.permutation(n) for _ in range(300)])
        import circuit_overlap as co
        co.N_PERM, co.N_BOOT = 300, 200
        s, z = stat_block(signal, perms, rng), stat_block(noise, perms, rng)
        self.assertGreater(s["z"], 5)
        self.assertLess(abs(z["z"]), 3)
        self.assertLess(s["p_one_sided"], 0.01)

    def test_stratified_permutations_stay_within_strata(self):
        rng = np.random.default_rng(3)
        fids = [f"f{i}" for i in range(6)]
        arr = {}
        for i, f in enumerate(fids):
            for lg in ("en", "es", "hi"):
                v = rng.normal(size=8)
                arr[(f, lg)] = {"mlp": v, "resid": v, "cells": rng.normal(size=16)}
        import circuit_overlap as co
        co.N_PERM, co.N_BOOT = 50, 20
        out = analyse(arr, fids, seed=0, strata=["a", "a", "a", "b", "b", "b"])
        self.assertEqual(out["n_facts"], 6)
        self.assertIn("en-es", out["pairs"])



class JointEditTests(unittest.TestCase):
    """A24: the multi-key update satisfies dW^T k_j = delta for every key and reduces to the ROME update for one key."""

    def setUp(self):
        import torch
        from knowledge_edit_v2 import joint_update
        self.torch, self.joint_update = torch, joint_update
        g = torch.Generator().manual_seed(0)
        A = torch.randn(40, 40, generator=g, dtype=torch.float64)
        self.C = A @ A.T / 40 + torch.eye(40, dtype=torch.float64)
        self.K = torch.randn(40, 3, generator=g, dtype=torch.float64)
        self.delta = torch.randn(16, generator=g, dtype=torch.float64)

    def test_every_key_gets_delta(self):
        dW = self.joint_update(self.K, self.delta, self.C)
        for j in range(3):
            self.assertTrue(self.torch.allclose(dW.T @ self.K[:, j], self.delta, atol=1e-8))

    def test_a_copy_of_a_key_is_the_same_constraint_and_does_not_make_the_solve_fail(self):
        """A30: Gemma facts whose subject text and left context match in two languages give identical keys."""
        K = self.torch.cat([self.K[:, :1], self.K[:, :1].clone(), self.K[:, 1:]], 1)  # k0, k0, k1, k2
        dW = self.joint_update(K, self.delta, self.C)
        self.assertTrue(self.torch.allclose(dW, self.joint_update(self.K, self.delta, self.C), atol=1e-8))
        for j in range(3):
            self.assertTrue(self.torch.allclose(dW.T @ self.K[:, j], self.delta, atol=1e-8))

    def test_a_nearly_identical_key_gives_a_finite_update_that_still_satisfies_both_constraints(self):
        g = self.torch.Generator().manual_seed(1)
        K = self.torch.cat([self.K, self.K[:, :1] * (1 + 1e-3) + 1e-3 * self.torch.randn(40, 1, generator=g, dtype=self.torch.float64)], 1)
        dW = self.joint_update(K, self.delta, self.C)
        self.assertTrue(self.torch.isfinite(dW).all())
        for j in range(4):
            self.assertTrue(self.torch.allclose(dW.T @ K[:, j], self.delta, atol=1e-5))

    def test_float32_keys_with_an_exact_copy_give_a_float32_update(self):
        K = self.torch.cat([self.K[:, :1], self.K[:, :1].clone(), self.K[:, 1:]], 1).float()
        dW = self.joint_update(K, self.delta.float(), self.C.float())
        self.assertEqual(dW.dtype, self.torch.float32)
        self.assertTrue(self.torch.isfinite(dW).all())

    def test_one_key_equals_rome(self):
        k = self.K[:, :1]
        Cr = self.C + 1e-3 * self.C.diagonal().mean() * self.torch.eye(40, dtype=self.torch.float64)
        u = self.torch.linalg.solve(Cr, k[:, 0])
        rome = self.torch.outer(u, self.delta) / (u @ k[:, 0])
        self.assertTrue(self.torch.allclose(self.joint_update(k, self.delta, self.C), rome, atol=1e-10))


class KeyModeFailureTests(unittest.TestCase):
    """A30: a key mode after the first may fail without stopping the run; the first (required) mode raises."""

    def setUp(self):
        import torch
        import knowledge_edit_v2 as kv
        self.torch, self.kv = torch, kv
        self.kk = {lg: torch.randn(8) for lg in ("en", "es", "hi")}
        self.delta, self.C = torch.randn(4), torch.eye(8)
        self._orig = kv.apply_edit_joint
        kv.apply_edit_joint = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("singular matrix"))

    def tearDown(self):
        self.kv.apply_edit_joint = self._orig

    def test_optional_mode_failure_is_recorded_and_returns_none(self):
        failures: list = []
        out = self.kv.apply_key_mode(None, 13, "joint_all", self.kk, self.delta, self.C, "capital_of_001", failures, required=False)
        self.assertIsNone(out)
        self.assertEqual([(x["fact"], x["layer"], x["mode"]) for x in failures], [("capital_of_001", 13, "joint_all")])
        self.assertIn("RuntimeError: singular matrix", failures[0]["error"])

    def test_required_mode_failure_raises(self):
        with self.assertRaises(RuntimeError):
            self.kv.apply_key_mode(None, 13, "joint_all", self.kk, self.delta, self.C, "capital_of_001", [], required=True)


class KeySimilarityTests(unittest.TestCase):
    """A26 Test 1 statistics: rank correlation, the partial version, and the Holm adjustment."""

    def setUp(self):
        from key_similarity import holm, partial_spearman, perm_p_greater, spearman
        self.holm, self.partial, self.perm, self.spearman = holm, partial_spearman, perm_p_greater, spearman
        self.rng = np.random.default_rng(0)

    def test_monotone_relation_is_rho_one_and_noise_is_not(self):
        x = self.rng.normal(size=80)
        self.assertAlmostEqual(self.spearman(x, np.exp(x)), 1.0)
        self.assertLess(abs(self.spearman(x, self.rng.normal(size=80))), 0.4)

    def test_partial_removes_a_shared_cause(self):
        z = self.rng.normal(size=300)
        x, y = z + 0.3 * self.rng.normal(size=300), z + 0.3 * self.rng.normal(size=300)
        self.assertGreater(self.spearman(x, y), 0.8)
        self.assertLess(abs(self.partial(x, y, [z])), 0.2)

    def test_partial_keeps_a_direct_effect(self):
        z = self.rng.normal(size=300)
        x = self.rng.normal(size=300)
        y = x + z + 0.3 * self.rng.normal(size=300)
        self.assertGreater(self.partial(x, y, [z]), 0.7)

    def test_permutation_p_is_one_sided(self):
        x = self.rng.normal(size=60)
        self.assertLess(self.perm(x, x + 0.5 * self.rng.normal(size=60), self.rng), 0.01)
        self.assertGreater(self.perm(x, -x, self.rng), 0.99)

    def test_holm_orders_and_caps(self):
        np.testing.assert_allclose(self.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        np.testing.assert_allclose(self.holm([0.6, 0.9]), [1.0, 1.0])


class LayerMapTests(unittest.TestCase):
    """A28 layer map: records of the English key and the mean_all key are paired by (fact, layer); pooled rho follows the planted relation."""

    def rows(self, planted: bool) -> list[dict]:
        rng, out = np.random.default_rng(1), []
        for layer in (2, 6):
            for i in range(30):
                x = float(rng.uniform(0.1, 0.9))
                y = x if planted else float(rng.uniform(0, 1))
                for mode, add in (("en", 0.0), ("mean_all", 0.1)):
                    para = {f"{lg}{j}": {"dP": 0.9 * (y + add)} for lg in ("es", "hi") for j in (1, 2)}
                    out.append({"fact": f"f{i}", "layer": layer, "key_mode": mode, "key_cos": {"es": x, "hi": x}, "en": {"dP": 0.9}, "para": para})
        return out

    def test_usable_pairs_modes_and_reports_gain(self):
        from layer_map import usable
        u = usable(self.rows(True))
        self.assertEqual(sorted(u), [2, 6])
        self.assertEqual(len(u[2]), 30)
        self.assertAlmostEqual(u[2]["f0"]["gain"]["es"], 0.1)

    def test_pooled_rho_is_high_for_a_planted_relation_and_not_for_noise(self):
        from layer_map import pooled, usable
        for planted, ok in ((True, lambda v: v > 0.95), (False, lambda v: abs(v) < 0.3)):
            u = usable(self.rows(planted))
            self.assertTrue(ok(pooled(u, "es", [f"f{i}" for i in range(30)])))

    def test_low_energy_facts_are_dropped(self):
        from layer_map import usable
        rows = self.rows(True)
        for r in rows:
            if r["fact"] == "f0":
                r["en"]["dP"] = 0.05
        self.assertNotIn("f0", usable(rows)[2])


if __name__ == "__main__":
    unittest.main()
