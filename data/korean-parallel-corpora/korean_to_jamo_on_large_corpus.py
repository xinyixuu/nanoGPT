#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: korean_phonetic.py
Description: Stream-based Hangul↔Jamo converter for large files,
             safe wrapping markers, file input/output support.

Provides two CLI modes:
  - decompose: wrap each Hangul syllable's Jamo in private-use markers for safe round-trip.
  - recompose: rebuild original Hangul text by converting marked Jamo back to Hangul.

Markers used (Private Use Area, unlikely in normal text):
  START_DECOMP = '\uE000'
  END_DECOMP   = '\uE001'

Usage examples:
  # Decompose from input file to output file
  python korean_phonetic.py -i large.txt -o decomposed.txt decompose

  # Recompose from decomposed file to stdout
  python korean_phonetic.py -i decomposed.txt recompose

Designed for streaming so it can handle very large files with minimal memory.
"""
import unicodedata
import unittest
import argparse
import sys
from jamo import h2j, j2hcj, j2h, is_jamo

# Recovery markers in Private Use Area
START_DECOMP = '\uE000'
END_DECOMP   = '\uE001'


def is_hangul_syllable(char):
    """Check if a character is a precomposed Hangul syllable."""
    code = ord(char)
    return 0xAC00 <= code <= 0xD7A3


def decompose_chunk(chunk):
    """Process a chunk of text, wrapping Hangul syllables."""
    out = []
    for char in chunk:
        if is_hangul_syllable(char):
            jamo = j2hcj(h2j(char))
            out.append(f"{START_DECOMP}{jamo}{END_DECOMP}")
        else:
            out.append(char)
    return ''.join(out)


def recompose_chunk(chunk):
    """Process a chunk of text, recovering Hangul from markers."""
    out = []
    i = 0
    length = len(chunk)
    while i < length:
        if chunk[i] == START_DECOMP:
            i += 1
            jseq = []
            while i < length and chunk[i] != END_DECOMP:
                jseq.append(chunk[i]); i += 1
            # skip the end marker if present
            if i < length and chunk[i] == END_DECOMP:
                i += 1
            out.append(j2h(*h2j(''.join(jseq))) if len(jseq)==3 else phonetic_to_korean([''.join(jseq)]))
        else:
            out.append(chunk[i]); i += 1
    return ''.join(out)


def phonetic_to_korean(segments):
    """Reconstruct Hangul text from Jamo segments."""
    result = []
    for seg in segments:
        if seg and all(is_jamo(ch) for ch in seg):
            n = len(seg)
            try:
                if n == 3:
                    result.append(j2h(seg[0], seg[1], seg[2]))
                elif n == 2:
                    result.append(j2h(seg[0], seg[1]))
                elif n > 3:
                    out, i = '', 0
                    while i < n:
                        rem = n - i
                        if rem >= 3:
                            out += j2h(seg[i], seg[i+1], seg[i+2]); i += 3
                        elif rem == 2:
                            out += j2h(seg[i], seg[i+1]); i += 2
                        else:
                            out += seg[i]; i += 1
                    result.append(out)
                else:
                    result.append(seg)
            except Exception:
                result.append(seg)
        else:
            result.append(seg)
    return ''.join(result)


class TestKoreanPhonetic(unittest.TestCase):
    def assert_roundtrip(self, text):
        self.assertEqual(phonetic_to_korean(list(decompose_chunk(text))), text)

    def test_empty_string(self): self.assert_roundtrip("")
    def test_ascii_only(self):    self.assert_roundtrip("Hello, World!")
    def test_hangul_only(self):   self.assert_roundtrip("안녕하세요")
    def test_mixed_hangul_latin(self): self.assert_roundtrip("안녕 Hello 123!")
    def test_jamo_only(self):
        inp = "ㅇㅏㄴㄴㅕㅇ"
        # decomposing leaves raw Jamo, then reconstruct
        self.assertEqual(decompose_chunk(inp), inp)
        self.assertEqual(phonetic_to_korean(list(inp)), inp)
    def test_halfwidth_compatibility_jamo(self):
        inp = "\uffa0\uffa1\uffa4"
        norm = ''.join(unicodedata.normalize('NFKC', c) for c in inp)
        self.assertEqual(phonetic_to_korean(list(decompose_chunk(inp))), norm)
    def test_extended_sequence(self):
        segs = ["ㅇㅏㄴㅇㅕㅇ"]
        self.assertEqual(phonetic_to_korean(segs), "안영")
    def test_decompose_recompose(self):
        sample = "안녕 Hello 안영"
        decomposed = decompose_chunk(sample)
        recomposed = recompose_chunk(decomposed)
        self.assertEqual(recomposed, sample)


def run_cli():
    parser = argparse.ArgumentParser(description="Stream-based Hangul↔Jamo converter.")
    parser.add_argument('-i', '--input-file', help='Path to input file (defaults to stdin)')
    parser.add_argument('-o', '--output-file', help='Path to output file (defaults to stdout)')
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('decompose', help='Wrap Hangul syllables in markers')
    sub.add_parser('recompose', help='Recover Hangul from markers')
    args = parser.parse_args()

    inp = open(args.input_file, encoding='utf-8') if args.input_file else sys.stdin
    out = open(args.output_file, 'w', encoding='utf-8') if args.output_file else sys.stdout

    func = decompose_chunk if args.mode=='decompose' else recompose_chunk
    for line in inp:
        out.write(func(line))

    if inp is not sys.stdin: inp.close()
    if out is not sys.stdout: out.close()

if __name__ == '__main__':
    if any(cmd in sys.argv for cmd in ('decompose','recompose')):
        run_cli()
    else:
        unittest.main()

