#!/usr/bin/env python3
"""Export the native 3D token embeddings from a sequence of checkpoints."""

import argparse
import json
import pickle
import re
from pathlib import Path

import torch


def iteration(path: Path) -> int:
    if path.name == "ckpt.pt":
        return 10**18
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def export(checkpoint_dir: Path, meta_path: Path, output: Path) -> None:
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    tokens = [meta["itos"][i] for i in range(meta["vocab_size"])]

    candidates = sorted(checkpoint_dir.glob("*.pt"), key=iteration)
    if not candidates:
        raise FileNotFoundError(f"no .pt checkpoints found in {checkpoint_dir}")

    frames = []
    seen_iterations = set()
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        step = int(checkpoint.get("iter_num", iteration(path)))
        if step in seen_iterations:
            continue
        state = checkpoint["model"]
        key = next((k for k in state if k.removeprefix("_orig_mod.") == "transformer.wte.weight"), None)
        if key is None:
            raise KeyError(f"transformer.wte.weight not found in {path}")
        vectors = state[key].detach().float()
        if vectors.shape != (len(tokens), 3):
            raise ValueError(f"{path}: expected {(len(tokens), 3)}, got {tuple(vectors.shape)}")
        frames.append({"iteration": step, "positions": vectors.tolist()})
        seen_iterations.add(step)

    payload = {
        "tokens": tokens,
        "trained_tokens": list("0123456789"),
        "unseen_tokens": list("abcd"),
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(frames)} frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/digits_3d/meta.pkl"))
    parser.add_argument("--output", type=Path, default=Path("report/threejs/digits-3d/token_trajectories.json"))
    args = parser.parse_args()
    export(args.checkpoint_dir, args.meta, args.output)


if __name__ == "__main__":
    main()
