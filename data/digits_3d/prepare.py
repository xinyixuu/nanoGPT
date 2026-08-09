#!/usr/bin/env python3
"""Build a fixed-vocabulary character dataset for the 3D token demo."""

import argparse
import pickle
from pathlib import Path

import numpy as np


VOCAB = "0123456789abcd"
SEQUENCE = "0123456789"


def build_dataset(out_dir: Path, train_repeats: int, val_repeats: int) -> None:
    if train_repeats < 2 or val_repeats < 2:
        raise ValueError("repeat counts must be at least 2")

    out_dir.mkdir(parents=True, exist_ok=True)
    stoi = {char: index for index, char in enumerate(VOCAB)}
    itos = {index: char for char, index in stoi.items()}

    def write_split(name: str, repeats: int) -> None:
        # Deliberately encode only digits. The letters remain valid vocabulary
        # entries so their untrained vectors can be compared with trained ones.
        ids = np.asarray([stoi[c] for c in SEQUENCE * repeats], dtype=np.uint16)
        ids.tofile(out_dir / f"{name}.bin")

    write_split("train", train_repeats)
    write_split("val", val_repeats)
    metadata = {
        "vocab_size": len(VOCAB),
        "stoi": stoi,
        "itos": itos,
        "train_tokens": len(SEQUENCE) * train_repeats,
        "val_tokens": len(SEQUENCE) * val_repeats,
        "description": "Repeated 0-9 sequence; a-d are vocabulary-only controls.",
    }
    with (out_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--train-repeats", type=int, default=2000)
    parser.add_argument("--val-repeats", type=int, default=200)
    args = parser.parse_args()
    build_dataset(args.out_dir, args.train_repeats, args.val_repeats)
    print(f"Wrote fixed vocabulary {VOCAB!r} to {args.out_dir}")


if __name__ == "__main__":
    main()
