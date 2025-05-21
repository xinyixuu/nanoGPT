# colorize_dataset.py
"""Efficient, no‑training dataset colourisation (min‑max mode)

Usage example
-------------
python colorize_dataset.py \
  --out_dir out/my_run            # directory containing ckpt.pt
  --dataset tiny_shakespeare      # dataset name (matches data/<dataset>)
  --split val                     # train | val
  --num_tokens 2048               # how many tokens to colourise
  --device cuda:0                 # or 'cpu'
  --block_size 512                # override context length (optional)
  --output_file coloured.txt      # save ANSI‑coloured text (optional)

The script:
* Loads the model checkpoint (no gradient tracking, no optimiser).
* Streams the chosen split from memmap, efficiently evaluating in blocks.
* Normalises chosen‑token logits via min‑max across the whole sample.
* Emits Rich‑coloured text to the console **and/or** an output file.

It re‑uses tokenisation rules from *meta.pkl* (tiktoken / sentencepiece /
CustomCharWithByteFallback) and is compatible with checkpoints produced by
*train.py*.
"""
from __future__ import annotations

import argparse
import os
import pickle
import io
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.text import Text

import tiktoken  # type: ignore

from model import GPT, GPTConfig  # local project imports

######################################################################################
# --------------------------- utility helpers (from sample.py) ---------------------- #
######################################################################################

def convert_rich_text_to_ansi(rich_text: Text) -> str:
    buf = io.StringIO()
    tmp_console = Console(file=buf, force_terminal=True, color_system="truecolor")
    tmp_console.print(rich_text)
    return buf.getvalue()


def colorize_text_minmax(
    token_ids: List[int],
    logits: List[float],  # chosen‑token logits (pre‑softmax)
    decode: Callable[[Sequence[int]], str],
) -> Text:
    """Return a Rich *Text* whose colour encodes the normalised logit value."""
    values = torch.tensor(logits, dtype=torch.float32)
    norm = (values - values.min()) / (values.max() - values.min() + 1e-6)

    text = Text()
    for tid, v in zip(token_ids, norm):
        r = int((1 - v.item()) * 255)
        g = int(v.item() * 255)
        token_str = decode([tid])
        text.append(token_str, style=f"bold #{r:02x}{g:02x}00")
    return text


# ---------------- tokeniser fallbacks ----------------- #

def _ccwb_encode(text: str, stoi: dict[int, int]) -> List[int]:
    out: List[int] = []
    for ch in text:
        if ch in stoi:
            out.append(stoi[ch])
        else:
            for b in ch.encode("utf-8"):
                out.append(stoi[bytes([b])])
    return out


def _ccwb_decode(ids: List[int], itos: dict[int, str | bytes]) -> str:
    parts: List[str] = []
    byte_buf: List[bytes] = []

    def flush():
        if byte_buf:
            parts.append(b"".join(byte_buf).decode("utf-8", errors="replace"))
            byte_buf.clear()

    for tok in ids:
        if tok < 256:
            byte_buf.append(itos[tok])  # type: ignore[arg-type]
        else:
            flush()
            parts.append(itos[tok])  # type: ignore[index]
    flush()
    return "".join(parts)

######################################################################################
# ------------------------------------- main --------------------------------------- #
######################################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colorise a dataset split using a trained GPT model (min‑max mode)")
    p.add_argument("--out_dir", required=True, help="Run folder containing ckpt.pt")
    p.add_argument("--dataset", required=True, help="Dataset name (data/<dataset>)")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--ckpt_name", default="ckpt.pt", help="Checkpoint filename inside out_dir")
    p.add_argument("--device", default="cuda", help="cpu | cuda | cuda:0 …")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num_tokens", type=int, default=1024, help="Number of tokens to colourise")
    p.add_argument("--block_size", type=int, default=None, help="Override model context window")
    p.add_argument("--output_file", type=str, default=None, help="Save ANSI‑coloured text here (optional)")
    return p.parse_args()


def load_tokeniser(meta_path: str) -> Tuple[Callable[[str], List[int]], Callable[[Sequence[int]], str]]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if meta.get("tokenizer") == "tiktoken":
        enc = tiktoken.get_encoding(meta["tiktoken_encoding"])
        encode = lambda s: enc.encode(s, allowed_special={""})  # type: ignore[return-value]
        decode = lambda l: enc.decode(l)                         # type: ignore[return-value]
    elif meta.get("tokenizer") == "sentencepiece":
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]                  # type: ignore[return-value]
        decode = lambda l: "".join(itos[i] for i in l)          # type: ignore[return-value]
    elif meta.get("tokenizer") == "custom_char_with_byte_fallback":
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: _ccwb_encode(s, stoi)                 # type: ignore[return-value]
        decode = lambda l: _ccwb_decode(l, itos)                 # type: ignore[return-value]
    else:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]                  # type: ignore[return-value]
        decode = lambda l: "".join(itos[i] for i in l)          # type: ignore[return-value]
    return encode, decode


def load_dataset_tokens(dataset: str, split: str, vocab_size: int) -> np.memmap:
    dtype = np.uint32 if vocab_size == 100277 else np.uint16
    path = os.path.join("data", dataset, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r")


def main() -> None:
    args = parse_args()
    console = Console()

    # --------------- checkpoint & config -----------------
    ckpt_path = Path(args.out_dir) / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()
    torch.set_grad_enabled(False)

    if args.block_size:
        model.update_block_size(args.block_size)

    # --------------- tokeniser ---------------------------
    meta_path = Path("data") / args.dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    encode, decode = load_tokeniser(str(meta_path))

    # --------------- data -------------------------------
    tokens_mm = load_dataset_tokens(args.dataset, args.split, gptconf.vocab_size)
    total_available = len(tokens_mm) - 1  # need +1 for next‑token prediction
    n_tokens = min(args.num_tokens, total_available)

    # --------------- forward pass in blocks -------------
    chosen_ids: List[int] = []
    chosen_logits: List[float] = []

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    ptdtype = dtype_map[args.dtype]
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if "cuda" in args.device else torch.no_grad()
    )

    bs = args.block_size or model.config.block_size
    pos = 0
    while pos < n_tokens:
        # we fetch *bs+1* tokens so we have the next‑token ground truth
        seq = tokens_mm[pos : pos + bs + 1]
        ctx_tokens = torch.from_numpy(seq[:-1].astype(np.int64))[None, ...].to(args.device)

        with ctx:
            logits, _ = model(ctx_tokens)
        logits = logits[0]  # (seq_len, vocab)

        # For each position j (predicting token j+1)
        steps = min(len(seq) - 1, n_tokens - pos)
        for j in range(steps):
            tgt = int(seq[j + 1])
            logit_val = logits[j, tgt].item()
            chosen_ids.append(tgt)
            chosen_logits.append(logit_val)

        pos += steps

    # --------------- colourise & emit -------------------
    coloured = colorize_text_minmax(chosen_ids, chosen_logits, decode)
    console.print(coloured)

    if args.output_file:
        ansi = convert_rich_text_to_ansi(coloured)
        with open(args.output_file, "w", encoding="utf-8", errors="replace") as f:
            f.write(ansi)
        console.print(f"[bold cyan]Saved ANSI output → {args.output_file}[/bold cyan]")


if __name__ == "__main__":
    main()

