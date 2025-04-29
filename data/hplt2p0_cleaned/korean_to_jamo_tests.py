#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for korean_phonetic.py

Run all tests:
    python -m unittest korean_to_jamo_tests
"""
import unicodedata
import unittest

from korean_phonetic import decompose_chunk, recompose_chunk, phonetic_to_korean


class TestKoreanPhonetic(unittest.TestCase):
    def _debug(self, label, value):
        print(f"    {label}: {repr(value)}")

    # helper that prints I/E/O for debugging even on success
    def _roundtrip(self, s):
        out = recompose_chunk(decompose_chunk(s))
        self._debug("input", s)
        self._debug("output", out)
        self.assertEqual(out, s)

    # ───────── basic round‑trip tests ──
    def test_empty_string(self):
        self._roundtrip("")

    def test_ascii_only(self):
        self._roundtrip("Hello, World!")

    def test_hangul_only(self):
        self._roundtrip("안녕하세요")

    def test_mixed_hangul_latin(self):
        self._roundtrip("안녕 Hello 123!")

    # ───────── specific behaviours ──
    def test_jamo_only(self):
        inp = "ㅇㅏㄴㄴㅕㅇ"
        exp = inp  # expect raw jamo preserved
        out = phonetic_to_korean([inp])
        self._debug("input", inp)
        self._debug("expected", exp)
        self._debug("actual", out)
        self.assertEqual(out, exp)

    def test_extended_sequence(self):
        segs = ["ㅇㅏㄴㅇㅕㅇ"]
        exp = "안영"
        out = phonetic_to_korean(segs)
        self._debug("input", segs)
        self._debug("expected", exp)
        self._debug("actual", out)
        self.assertEqual(out, exp)

    def test_decompose_recompose(self):
        sample = "안녕 Hello 안영"
        out = recompose_chunk(decompose_chunk(sample))
        self._debug("input", sample)
        self._debug("expected", sample)
        self._debug("actual", out)
        self.assertEqual(out, sample)


if __name__ == "__main__":
    unittest.main(verbosity=2)

