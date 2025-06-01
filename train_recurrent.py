# ======================================================================
# train_recurrent.py  –  latent-chaining fine-tuning
# ======================================================================
#  * resumes from an existing checkpoint (no scratch / no GPT-2 import)
#  * feeds the HIDDEN state (after ln_f / scale_down) back as the next
#    “token”, skipping de-embedding, for the first `--latent_steps`
#  * keeps cross-entropy vs. ground-truth, with optional per-position
#    linear weighting and an initial “skip” window
# ----------------------------------------------------------------------

import argparse, os, time, math, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model       import GPT, GPTConfig
from train_args  import parse_args          # <- from your repo

# ----------------------------------------------------------------------
# 1)  ARGUMENTS
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Latent-chaining trainer")
    parse_args(p)               # all the usual nano-GPT flags

    # NEW knobs
    p.add_argument("--latent_steps", type=int, default=0,
                   help="Chain this many hidden states before falling "
                        "back to normal token IDs (0 = none, full "
                        "latent chaining if == block_size).")
    p.add_argument("--weight_start", type=float, default=1.0,
                   help="CE-loss weight at position 0")
    p.add_argument("--weight_end",   type=float, default=1.0,
                   help="CE-loss weight at position block_size-1")
    p.add_argument("--skip_steps",   type=int, default=0,
                   help="Loss for the first K positions is zeroed out")
    p.add_argument("--resume_ckpt",  type=str, required=True,
                   help="Path to checkpoint .pt produced by train.py")

    # handy defaults if the user forgets
    p.set_defaults(max_epochs=10, grad_clip=1.0)
    return p

args = build_arg_parser().parse_args()

# ----------------------------------------------------------------------
# 2)  LOAD CHECKPOINT + MODEL
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(args.resume_ckpt, map_location=device)

gpt_conf = GPTConfig(**ckpt["model_args"])
model    = GPT(gpt_conf).to(device)
model.load_state_dict(ckpt["model"])

# helpers exposed in patched model.py
embed_tokens     = model.embed_tokens
forward_embedded = lambda x: model.forward_embedded(x, return_hidden=True)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=args.learning_rate, betas=(0.9, 0.95))
if ckpt.get("optimizer"):
    optimizer.load_state_dict(ckpt["optimizer"])

best_val_loss = ckpt["best_val_loss"]
iter_num      = ckpt["iter_num"]          # not used, but preserved

block_size = gpt_conf.block_size

# ----------------------------------------------------------------------
# 3)  DATA (mmap – same layout as train.py)
# ----------------------------------------------------------------------
def load_bin(split):
    path = os.path.join("data", args.dataset, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")

train_bin, val_bin = load_bin("train"), load_bin("val")

# ----------------------------------------------------------------------
# 4)  LOSS-WEIGHT HELPER
# ----------------------------------------------------------------------
def make_loss_weights(bsz: int, T: int, device):
    w = torch.linspace(args.weight_start, args.weight_end, steps=T,
                       device=device).repeat(bsz, 1)
    if args.skip_steps:
        w[:, :args.skip_steps] = 0.0
    return w

# ----------------------------------------------------------------------
# 5)  ONE BLOCK (B,T)  →  scalar loss
# ----------------------------------------------------------------------
def train_block(x_tokens, y_tokens):
    B, T = x_tokens.shape
    weights     = make_loss_weights(B, T, x_tokens.device)
    nonzero_sum = weights.sum() + 1e-8

    # --- t = 0 ---------------------------------------------------------
    emb                 = embed_tokens(x_tokens[:, :1])          # (B,1,E)
    logits, hidden_prev = forward_embedded(emb)                  # (B,1,V)
    loss = (F.cross_entropy(logits.view(-1, logits.size(-1)),
                            y_tokens[:, 0], reduction="none")
            * weights[:, 0]).sum()

    # --- t = 1 … T-1 ---------------------------------------------------
    for t in range(1, T):
        if t < args.latent_steps:          # latent chaining
            inp = hidden_prev              # (B,1,E)
        else:                              # teacher forcing
            inp = embed_tokens(x_tokens[:, t:t+1])

        logits, hidden_prev = forward_embedded(inp)

        loss += (F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 y_tokens[:, t], reduction="none")
                 * weights[:, t]).sum()

    return loss / nonzero_sum              # mean CE

# ----------------------------------------------------------------------
# 6)  EPOCH LOOPS
# ----------------------------------------------------------------------
def run_epoch(split):
    data   = train_bin if split == "train" else val_bin
    losses = []
    ptr    = 0
    while ptr + block_size + 1 < len(data):
        seq = torch.from_numpy(np.array(
                  data[ptr:ptr + block_size + 1], dtype=np.int64)
              ).to(device)
        x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)  # (1,T)

        if split == "train":
            model.train()
            optimizer.zero_grad()
            loss = train_block(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                loss = train_block(x, y)

        losses.append(loss.item())
        ptr += block_size
    return sum(losses) / len(losses)

# ----------------------------------------------------------------------
# 7)  TRAINING DRIVER
# ----------------------------------------------------------------------
tb = SummaryWriter() if getattr(args, "tensorboard_log", False) else None
best_ckpt_path = os.path.join(os.path.dirname(args.resume_ckpt), "ckpt_lat.pt")

for epoch in range(args.max_epochs):
    t0 = time.time()
    train_loss = run_epoch("train")
    val_loss   = run_epoch("val")

    if tb:
        tb.add_scalar("loss/train", train_loss, epoch)
        tb.add_scalar("loss/val",   val_loss,   epoch)

    print(f"epoch {epoch:03d} | "
          f"train {train_loss:.4f} | val {val_loss:.4f} | "
          f"{(time.time()-t0):.1f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model":       model.state_dict(),
            "model_args":  ckpt["model_args"],
            "iter_num":    iter_num,
            "best_val_loss": best_val_loss,
        }, best_ckpt_path)
        print(f"  ➜ new best; checkpoint saved to {best_ckpt_path}")

if tb:
    tb.flush()
    tb.close()

print("done.")
# ======================================================================

