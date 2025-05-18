# train_variations/optimizer_variants.py
from __future__ import annotations

import torch
from torch.optim import (
    Adam, AdamW, Adamax, Adagrad, ASGD, LBFGS, NAdam,
    RAdam, RMSprop, SGD, SparseAdam
)
from torch.optim.optimizer import Optimizer


class OrthoAdam(Optimizer):
    """
    Memory-efficient implementation of the 'OrthoAdam' optimiser
    proposed in Kaul et al. 2024 (arXiv:2410.17174).
    For each parameter tensor we store **only**
      • a random permutation (int32) and
      • a ±1 sign vector (float32/float16),
    together representing an orthogonal matrix Q = diag(sign) · P.
    This keeps O(N) memory instead of O(N²).

    Args
    ----
    params : iterable of Tensors
    lr     : learning rate (η)
    betas  : (β1, β2)
    eps    : numerical eps
    weight_decay : AdamW-style decoupled weight decay
    permute_threshold : if tensor.numel() > threshold we skip rotation
                        (i.e. use identity Q) to avoid storing big buffers
    """

    # def __init__(self, params, *, lr=1e-3, betas=(0.9, 0.999),
    #              eps=1e-8, weight_decay=0.0, permute_threshold=1_000_000):
    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        permute_threshold=1_000_000,
        tiny_threshold=128,
    ):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            permute_threshold=permute_threshold,
        )
        super().__init__(params, defaults)

        # Build permutation & sign for each parameter once
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                n = p.numel()
                if n < tiny_threshold:  # tiny (bias/LN γ,β) → identity
                    state["perm"] = None
                    state["sign"] = None
                elif n > permute_threshold:  # huge → identity (memory)
                    state["perm"] = None
                    state["sign"] = None
                else:
                    device = p.device
                    state["perm"] = torch.randperm(n, device=device, dtype=torch.int64)
                    state["sign"] = (
                        torch.randint(0, 2, (n,), device=device, dtype=p.dtype) * 2.0
                        - 1.0
                    )

                # # Adam buffers in optimiser basis
                # state["step"] = 0
                # state["exp_avg"] = torch.zeros(n, device=p.device, dtype=p.dtype)
                # state["exp_avg_sq"] = torch.zeros(n, device=p.device, dtype=p.dtype)
                # Adam buffers (keep **always** in fp32 to avoid fp16 under-flow)
                state["step"] = 0
                state["exp_avg"] = torch.zeros(n, device=p.device, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros(
                    n, device=p.device, dtype=torch.float32
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Sparse parameters not supported.")

                # Flatten to 1-D for permutation maths
                g = grad.view(-1)
                state = self.state[p]
                perm, sign = state["perm"], state["sign"]

                # # Weight decay in *parameter* basis
                # if wd != 0.0:
                #     g = g.add(p.data.view(-1), alpha=wd)

                # ---- rotate gradient to optimiser basis
                if perm is not None:  # Q = diag(sign)·P
                    # make sure buffers are on same device as grad
                    if perm.device != g.device:
                        perm = state["perm"] = perm.to(g.device, non_blocking=True)
                        sign = state["sign"] = sign.to(g.device, non_blocking=True)
                    g_bar = g[perm] * sign
                else:  # identity
                    g_bar = g

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # -------- keep moments on the same device as the grad
                if exp_avg.device != g.device:
                    exp_avg = state["exp_avg"] = exp_avg.to(g.device, non_blocking=True)
                    exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(
                        g.device, non_blocking=True
                    )

                state["step"] += 1
                step = state["step"]

                # Adam moment updates in optimiser basis
                exp_avg.mul_(beta1).add_(g_bar, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_bar, g_bar, value=1 - beta2)

                # Bias correction
                bias_c1 = 1 - beta1**step
                bias_c2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
                step_bar = (exp_avg / bias_c1) / denom

                # ---- rotate update back to parameter basis
                if perm is not None:
                    inv_perm = torch.empty_like(perm)
                    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
                    step_vec = (step_bar * sign)[inv_perm]
                else:
                    step_vec = step_bar

                # Parameter update (in-place)
                # AdamW-style **decoupled** weight-decay _after_ inverse rotation
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(step_vec.view_as(p.data), alpha=-lr)

        return loss

# Momentum / SGD family
def _sgd(param_groups, args):
    return SGD(param_groups,
               lr=args.learning_rate,
               momentum=args.sgd_momentum,
               nesterov=args.sgd_nesterov,
               weight_decay=args.weight_decay)

# Adam family
def _adam(param_groups, args):
    return Adam(param_groups,
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=args.adamw_eps,
                weight_decay=args.weight_decay)

def _adamw(param_groups, args):
    return AdamW(param_groups,
                 lr=args.learning_rate,
                 betas=(args.beta1, args.beta2),
                 eps=args.adamw_eps,
                 weight_decay=args.adamw_weight_decay)

def _radam(param_groups, args):
    return RAdam(param_groups,
                 lr=args.learning_rate,
                 betas=(args.beta1, args.beta2),
                 eps=args.adamw_eps,
                 weight_decay=args.weight_decay)

def _adamax(param_groups, args):
    return Adamax(param_groups,
                  lr=args.learning_rate,
                  betas=(args.beta1, args.beta2),
                  eps=args.adamw_eps,
                  weight_decay=args.weight_decay)

def _nadam(param_groups, args):
    return NAdam(param_groups,
                 lr=args.learning_rate,
                 betas=(args.beta1, args.beta2),
                 eps=args.nadam_eps,
                 weight_decay=args.weight_decay)

# RMS / Ada family
def _rmsprop(param_groups, args):
    return RMSprop(param_groups,
                   lr=args.learning_rate,
                   alpha=args.rmsprop_alpha,
                   eps=args.adamw_eps,
                   weight_decay=args.weight_decay)

def _rprop(param_groups, args):
    """
    Plain PyTorch Rprop.  We expose the multiplicative factors (etas) and
    re‑use them for step‑size bounds for simplicity.
    """
    return torch.optim.Rprop(
        param_groups,
        lr=args.learning_rate,
        etas=(args.rprop_eta_min, args.rprop_eta_max),
        step_sizes=(args.rprop_eta_min, args.rprop_eta_max),
    )

def _adagrad(param_groups, args):
    return Adagrad(param_groups,
                   lr=args.learning_rate,
                   lr_decay=args.adagrad_lr_decay,
                   eps=args.adamw_eps,
                   weight_decay=args.weight_decay)

def _sparseadam(param_groups, args):
    return SparseAdam(param_groups,
                      lr=args.learning_rate,
                      betas=(args.beta1, args.beta2),
                      eps=args.adamw_eps)

# Second-order / others
def _asgd(param_groups, args):
    return ASGD(param_groups,
                lr=args.learning_rate,
                lambd=args.asgd_lambda,
                alpha=args.asgd_alpha,
                t0=args.asgd_t0,
                weight_decay=args.weight_decay)

def _lbfgs(param_groups, args):
    return LBFGS(param_groups,
                 lr=args.learning_rate,
                 max_iter=args.lbfgs_max_iter,
                 max_eval=args.lbfgs_max_eval,
                 tolerance_grad=args.lbfgs_tol_grad,
                 tolerance_change=args.lbfgs_tol_change,
                 history_size=args.lbfgs_history,
                 line_search_fn=args.lbfgs_line_search)

# ---- Ortho optimisers ------------------------------------------------
def _orthoadam(param_groups, args):
    """Memory-cheap OrthoAdam (permute + sign)."""
    flat_params = [p for g in param_groups for p in g["params"]]
    return OrthoAdam(
        flat_params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        permute_threshold=getattr(args, "ortho_perm_threshold", 1_000_000),
    )


optimizer_dictionary: dict[str, callable] = {
    # From pytorch
    "sgd": _sgd,
    "adam": _adam,
    "adamw": _adamw,
    "adamax": _adamax,
    "radam": _radam,
    "nadam": _nadam,
    "adagrad": _adagrad,
    "rmsprop": _rmsprop,
    "rprop": _rprop,
    "sparseadam": _sparseadam,
    "asgd": _asgd,
    "lbfgs": _lbfgs,
    # paper-driven implementations
    "orthoadam": _orthoadam,
}

