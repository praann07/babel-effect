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

    def test_one_key_equals_rome(self):
        k = self.K[:, :1]
        Cr = self.C + 1e-3 * self.C.diagonal().mean() * self.torch.eye(40, dtype=self.torch.float64)
        u = self.torch.linalg.solve(Cr, k[:, 0])
        rome = self.torch.outer(u, self.delta) / (u @ k[:, 0])
        self.assertTrue(self.torch.allclose(self.joint_update(k, self.delta, self.C), rome, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
