#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: korean_phonetic.py
Description: Stream-based Hangul↔Jamo converter for large files,
             safe wrapping markers, file input/output support.

Provides two CLI modes:
  - decompose: wrap each Hangul syllable's Jamo in private‑use markers for safe round‑trip.
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
from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

from jamo import h2j, j2hcj, j2h, is_jamo

START_DECOMP = '\uE000'  # Private Use Area start marker
END_DECOMP = '\uE001'    # Private Use Area end marker


# ---------- low‑level helpers ---------- #

def is_hangul_syllable(char: str) -> bool:
    """Return True if *char* is a pre‑composed Hangul syllable (U+AC00–U+D7A3)."""
    code = ord(char)
    return 0xAC00 <= code <= 0xD7A3


# ---------- transformation primitives ---------- #

def decompose_chunk(chunk: str) -> str:
    """Wrap Hangul syllables in *chunk* with START/END markers and expose underlying jamo."""
    out: List[str] = []
    for ch in chunk:
        if is_hangul_syllable(ch):
            out.append(f"{START_DECOMP}{j2hcj(h2j(ch))}{END_DECOMP}")
        else:
            out.append(ch)
    return ''.join(out)


def phonetic_to_korean(segments: Iterable[str]) -> str:
    """
    Convert an iterable of jamo strings *segments* into Hangul text.
    Each element in *segments* is treated as an independent jamo sequence.
    """
    result: List[str] = []
    for seg in segments:
        if seg and all(is_jamo(c) for c in seg):
            n = len(seg)
            try:
                if n == 3:
                    result.append(j2h(seg[0], seg[1], seg[2]))
                elif n == 2:
                    result.append(j2h(seg[0], seg[1]))
                elif n > 3:
                    # greedy left‑to‑right packing
                    i = 0
                    buf = []
                    while i < n:
                        rem = n - i
                        if rem >= 3:
                            buf.append(j2h(seg[i], seg[i + 1], seg[i + 2]))
                            i += 3
                        elif rem == 2:
                            buf.append(j2h(seg[i], seg[i + 1]))
                            i += 2
                        else:
                            buf.append(seg[i])
                            i += 1
                    result.append(''.join(buf))
                else:
                    result.append(seg)
            except Exception:
                # fall back to raw jamo on any conversion failure
                result.append(seg)
        else:
            result.append(seg)
    return ''.join(result)


def recompose_chunk(chunk: str) -> str:
    """Undo *decompose_chunk* on a single chunk of text."""
    out: List[str] = []
    i = 0
    n = len(chunk)

    while i < n:
        ch = chunk[i]
        if ch == START_DECOMP:
            i += 1
            jamo_buf: List[str] = []
            while i < n and chunk[i] != END_DECOMP:
                jamo_buf.append(chunk[i])
                i += 1
            # consume END_DECOMP if present
            if i < n and chunk[i] == END_DECOMP:
                i += 1
            seq = ''.join(jamo_buf)
            out.append(j2h(*h2j(seq)) if len(jamo_buf) == 3 else phonetic_to_korean([seq]))
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


# ---------- CLI ---------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream‑based Hangul↔Jamo converter.")
    parser.add_argument(
        "-i", "--input-file",
        help="Path to input file (defaults to stdin).",
    )
    parser.add_argument(
        "-o", "--output-file",
        help="Path to output file (defaults to stdout).",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("decompose", help="Convert Hangul syllables to jamo wrapped by markers.")
    sub.add_parser("recompose", help="Recover Hangul syllables from marked jamo.")
    return parser.parse_args()


def _open_stream(path: str | None, mode: str):
    if path:
        return open(path, mode, encoding="utf-8")
    return sys.stdin if "r" in mode else sys.stdout


def main() -> None:
    args = _parse_args()
    transform = decompose_chunk if args.mode == "decompose" else recompose_chunk

    inp = _open_stream(args.input_file, "r")
    out = _open_stream(args.output_file, "w")

    try:
        for line in inp:
            out.write(transform(line))
    finally:
        if inp is not sys.stdin:
            inp.close()
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()

