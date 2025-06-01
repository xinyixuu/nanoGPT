# ======================================================================
# train_recurrent.py  –  latent-chaining fine-tuning
# ======================================================================
#  * resumes from an existing checkpoint (no scratch / no GPT-2 import)
#  * feeds the HIDDEN state (after ln_f / scale_down) back as the next
#    “token”, skipping de-embedding, for the first `--latent_steps`
#  * keeps cross-entropy vs. ground-truth, with optional per-position
#    linear weighting and an initial “skip” window
# ----------------------------------------------------------------------

import argparse
import os
import time
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig           # your patched model.py


global_step = 0          # counts *training* iterations


# ----------------------------------------------------------------------
# 1)  ARGUMENT PARSER  (only what we actually need)
# ----------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Latent-chaining trainer")

    # -------- baseline knobs we use -----------------------------------
    p.add_argument("--dataset",      type=str, default="shakespeare_char")
    p.add_argument("--block_size",   type=int, default=128)
    p.add_argument("--learning_rate",type=float, default=3e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--max_epochs",   type=int,   default=10)
    p.add_argument("--tensorboard_log", action="store_true")
    p.add_argument("--log_every", type=int, default=10,
                   help="Print train-loop stats every N iterations")
    p.add_argument("--val_every", type=int, default=250,
                   help="Run validation & checkpoint every N iterations")


    # -------- new latent-chaining knobs -------------------------------
    p.add_argument("--resume_ckpt",   required=True,
                   help="Path to checkpoint .pt produced by train.py")
    p.add_argument("--latent_steps",  type=int, default=0,
                   help="Chain this many hidden states before falling "
                        "back to token IDs (0 = none).")
    p.add_argument("--skip_steps",    type=int, default=0,
                   help="Zero the loss for the first K positions in each block")
    p.add_argument("--weight_start",  type=float, default=1.0)
    p.add_argument("--weight_end",    type=float, default=1.0)

    return p

args = build_arg_parser().parse_args()


# ----------------------------------------------------------------------
# 2)  LOAD CHECKPOINT + MODEL
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(args.resume_ckpt, map_location=device)

gpt_conf = GPTConfig(**ckpt["model_args"])
model    = GPT(gpt_conf).to(device)

def unwrap_state_dict(wrapped_sd):
    """
    Remove '_orig_mod.' (torch.compile) and 'module.' (DDP) prefixes so the
    keys match a plain, single-GPU GPT instance.
    """
    clean = {}
    for k, v in wrapped_sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean

state_dict = unwrap_state_dict(ckpt["model"])
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"warning: {len(missing)} missing params (OK if all zero-grad)")
if unexpected:
    print(f"warning: {len(unexpected)} extra params ignored")

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
    """
    One recurrent block that **preserves full self-attention context**.
    We build a `hidden_buf` (B, ≤T, E); at each step we append either
    the latent vector from the previous step or the ground-truth embedding,
    then run the whole sequence through the model once.
    """
    B, T   = x_tokens.shape
    device = x_tokens.device

    weights = make_loss_weights(B, T, device)
    nz_sum  = weights.sum() + 1e-8

    hidden_buf  = None        # grows (B,t,E)
    hidden_prev = None        # last latent state (B,1,E)
    total_loss  = 0.0

    for t in range(T):
        # ---- decide what to append ---------------------------------
        if t == 0 or t >= args.latent_steps:
            new_piece = embed_tokens(x_tokens[:, t:t+1])     # GT token
        else:
            new_piece = hidden_prev                         # latent vector

        # ---- grow the buffer ---------------------------------------
        hidden_buf = new_piece if hidden_buf is None else \
                     torch.cat([hidden_buf, new_piece], dim=1)

        # ---- full forward pass on the whole buffer -----------------
        logits_all, h_all = forward_embedded(hidden_buf)

        logits_step = logits_all[:, -1, :]   # newest position only
        hidden_prev = h_all[:,  -1:, :]      # keep for next iteration

        ce = F.cross_entropy(logits_step, y_tokens[:, t], reduction="none")
        total_loss += (ce * weights[:, t]).sum()

    return total_loss / nz_sum

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
            global global_step
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"iter {global_step:>7} | "
                      f"loss {loss.item():.4f}")

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

