# colorize_dataset.py  (softmax mode added)
"""Colourise a dataset split using a trained GPT model.

Modes
-----
* **minmax**  – map chosen‑token *logits* to a red→green gradient after
  min‑max normalisation.
* **softmax** – map chosen‑token *probabilities* (softmax) the same way.

Example
-------
```bash
python colorize_dataset.py \
  --out_dir out/my_run \
  --dataset tiny_shakespeare \
  --split val \
  --num_tokens 2048 \
  --mode softmax
```
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.text import Text

import tiktoken  # type: ignore

from model import GPT, GPTConfig

################################################################################
# ---------- helpers --------------------------------------------------------- #
################################################################################

def convert_rich_text_to_ansi(txt: Text) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor").print(txt)
    return buf.getvalue()


def _colorize(token_ids: List[int], scalars: List[float], decode: Callable[[Sequence[int]], str]) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    rich_txt = Text()
    for tid, v in zip(token_ids, norm):
        r = int((1 - v.item()) * 255)
        g = int(v.item() * 255)
        rich_txt.append(decode([tid]), style=f"bold #{r:02x}{g:02x}00")
    return rich_txt

# ----- custom-char-with-byte-fallback helpers -----

def _ccwb_encode(text: str, stoi: dict) -> List[int]:
    out: List[int] = []
    for ch in text:
        if ch in stoi:
            out.append(stoi[ch])
        else:
            out.extend(stoi[bytes([b])] for b in ch.encode("utf-8"))
    return out


def _ccwb_decode(ids: List[int], itos: dict) -> str:
    chars: List[str] = []
    byte_buf: List[bytes] = []
    def flush():
        if byte_buf:
            chars.append(b"".join(byte_buf).decode("utf-8", errors="replace"))
            byte_buf.clear()
    for tok in ids:
        if tok < 256:
            byte_buf.append(itos[tok])  # type: ignore[arg-type]
        else:
            flush()
            chars.append(itos[tok])  # type: ignore[index]
    flush()
    return "".join(chars)

################################################################################
# --------------------------- CLI / loading --------------------------------- #
################################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Colourise a dataset split with a trained GPT model")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--ckpt_name", default="ckpt.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--num_tokens", type=int, default=1024)
    p.add_argument("--block_size", type=int)
    p.add_argument("--mode", choices=["minmax", "softmax"], default="minmax")
    p.add_argument("--output_file")
    return p.parse_args()


def load_tokeniser(meta_path: Path):
    meta = pickle.load(meta_path.open("rb"))
    tk = meta.get("tokenizer")
    if tk == "tiktoken":
        enc = tiktoken.get_encoding(meta["tiktoken_encoding"])
        return (
            lambda s: enc.encode(s, allowed_special={""}),
            lambda l: enc.decode(l),
        )
    stoi, itos = meta["stoi"], meta["itos"]
    if tk == "sentencepiece":
        return (
            lambda s: [stoi[c] for c in s],
            lambda l: "".join(itos[i] for i in l),
        )
    if tk == "custom_char_with_byte_fallback":
        return (
            lambda s: _ccwb_encode(s, stoi),
            lambda l: _ccwb_decode(l, itos),
        )
    return (
        lambda s: [stoi[c] for c in s],
        lambda l: "".join(itos[i] for i in l),
    )

################################################################################
# ---------------------------------- main ------------------------------------ #
################################################################################

def main() -> None:
    args = parse_args()
    console = Console()

    ckpt = torch.load(Path(args.out_dir) / args.ckpt_name, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()
    torch.set_grad_enabled(False)

    if args.block_size:
        model.update_block_size(args.block_size)

    encode, decode = load_tokeniser(Path("data") / args.dataset / "meta.pkl")

    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if "cuda" in args.device else torch.no_grad()

    dtype = np.uint32 if model.config.vocab_size == 100277 else np.uint16
    data_mm = np.memmap(Path("data") / args.dataset / f"{args.split}.bin", dtype=dtype, mode="r")
    n_tokens = min(args.num_tokens, len(data_mm) - 1)

    block = args.block_size or model.config.block_size
    pos = 0
    ids: List[int] = []
    scalars: List[float] = []

    while pos < n_tokens:
        seq = data_mm[pos : pos + block + 1]
        ctx = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)
        with autocast:
            logits, _ = model(ctx)
        logits = logits.squeeze(0)  # (ctx_len, vocab)
        ctx_len = logits.size(0)
        offset = len(seq) - 1 - ctx_len
        steps = min(ctx_len, n_tokens - pos)

        if args.mode == "softmax":
            probs = F.softmax(logits, dim=-1)

        for j in range(steps):
            tgt = int(seq[offset + j + 1])
            ids.append(tgt)
            if args.mode == "softmax":
                scalars.append(probs[j, tgt].item())
            else:  # minmax
                scalars.append(logits[j, tgt].item())
        pos += steps

    coloured = _colorize(ids, scalars, decode)
    console.print(coloured)

    if args.output_file:
        Path(args.output_file).write_text(convert_rich_text_to_ansi(coloured), "utf-8", errors="replace")
        console.print(f"[cyan]Saved → {args.output_file}[/cyan]")


if __name__ == "__main__":
    main()

