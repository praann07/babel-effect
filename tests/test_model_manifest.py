"""Tests for the model-folder fingerprint (code/model_manifest.py). CPU only, no torch: tiny .safetensors files are written by hand."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from model_manifest import compare, fingerprint  # noqa: E402


def write_safetensors(path: Path, tensors: dict[str, bytes], shape=(2, 2), dtype="BF16") -> None:
    header, off = {}, 0
    for name, raw in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(tensors.values()))


T = {"a": bytes(range(8)), "b": bytes(range(8, 16)), "c": bytes(range(16, 24))}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def folder(self, name: str, shards: list[dict[str, bytes]], config: str = "{}") -> Path:
        d = self.root / name
        d.mkdir()
        for i, s in enumerate(shards):
            write_safetensors(d / f"model-{i}.safetensors", s)
        (d / "config.json").write_text(config)
        return d

    def test_same_tensors_in_a_different_shard_split_are_identical(self):
        ref = fingerprint(self.folder("ref", [T]))
        other = fingerprint(self.folder("other", [{"a": T["a"]}, {"b": T["b"], "c": T["c"]}]))
        self.assertEqual(compare(other, ref), [])

    def test_one_changed_byte_is_reported_with_the_tensor_name(self):
        ref = fingerprint(self.folder("ref", [T]))
        bad = dict(T, b=bytes([9]) + T["b"][1:])
        self.assertEqual(compare(fingerprint(self.folder("bad", [bad])), ref), ["tensor b: bytes differ"])

    def test_missing_and_extra_tensors_are_reported(self):
        ref = fingerprint(self.folder("ref", [T]))
        other = fingerprint(self.folder("other", [{"a": T["a"], "b": T["b"], "z": bytes(8)}]))
        self.assertEqual(compare(other, ref), ["missing tensor c", "extra tensor z"])

    def test_changed_config_is_reported(self):
        ref = fingerprint(self.folder("ref", [T], config="{}"))
        other = fingerprint(self.folder("other", [T], config='{"x": 1}'))
        self.assertEqual(compare(other, ref), ["file config.json differs"])

    def test_dtype_change_is_reported_even_if_bytes_match(self):
        ref = fingerprint(self.folder("ref", [T]))
        d = self.root / "f16"
        d.mkdir()
        write_safetensors(d / "m.safetensors", T, dtype="F16")
        (d / "config.json").write_text("{}")
        self.assertEqual(len(compare(fingerprint(d), ref)), 3)  # one line per tensor


if __name__ == "__main__":
    unittest.main()
