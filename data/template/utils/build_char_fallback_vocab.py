"""Build a deterministic training-only character whitelist for byte fallback."""

import argparse
from collections import Counter
from pathlib import Path


def select_characters(train_input, limit):
    if limit < 1:
        raise ValueError("limit must be positive")
    counts = Counter()
    with Path(train_input).open(encoding="utf-8") as handle:
        for line in handle:
            counts.update(char for char in line if not char.isspace())
    # Tie-breaking by Unicode makes IDs reproducible across machines.
    return sorted(counts, key=lambda char: (-counts[char], char))[:limit]


def read_characters(path):
    characters = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            char = line.rstrip("\r\n")
            if not char or char.isspace():
                continue
            if len(char) != 1:
                raise ValueError(f"{path}:{line_number}: expected one Unicode character")
            if char in characters:
                raise ValueError(f"{path}:{line_number}: duplicate character {char!r}")
            characters.append(char)
    return characters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--characters_file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    characters = (read_characters(args.characters_file) if args.characters_file
                  else select_characters(args.train_input, args.limit))
    if not characters:
        parser.error("no non-whitespace characters available for the whitelist")
    Path(args.output).write_text("".join(char + "\n" for char in characters), encoding="utf-8")
    print(f"Byte-fallback vocabulary: 256 bytes + {len(characters)} characters")


if __name__ == "__main__":
    main()
