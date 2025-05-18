# train_variations/optimizer_variants.py
from __future__ import annotations

import torch
from torch.optim import (ASGD, LBFGS, SGD, Adagrad, Adam, Adamax, AdamW, NAdam,
                         RAdam, RMSprop, SparseAdam)
from torch.optim.optimizer import Optimizer

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None

try:
    from apollo_torch import APOLLOAdamW          # pip install apollo-torch
except ImportError as e:                           # graceful fallback
    APOLLOAdamW = None

# Other PyPI optimisers ---------------------------------------------------
try:  # AdaBelief
    from adabelief_pytorch import AdaBelief
except ImportError:
    AdaBelief = None

try:  # Adan
    from adan_pytorch import Adan
except ImportError:
    Adan = None

# --------------------------------------------------------------------
#  Optional:  third-party optimisers (torch-optimizer)
#             imported lazily so the code runs even if the package
#             is absent and the user doesn’t choose those options.
# --------------------------------------------------------------------
try:
    import torch_optimizer as topt
except ModuleNotFoundError:  # keep rest of repo usable
    topt = None


def _needs_topt():
    if topt is None:
        raise ImportError(
            "This optimiser relies on the `torch-optimizer` package.\n"
            "➡  pip install torch-optimizer"
        )


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
    return SGD(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        nesterov=args.sgd_nesterov,
        weight_decay=args.weight_decay,
    )


# Adam family
def _adam(param_groups, args):
    return Adam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _adamw(param_groups, args):
    return AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
    )


def _radam(param_groups, args):
    return RAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _adamax(param_groups, args):
    return Adamax(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _nadam(param_groups, args):
    return NAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.nadam_eps,
        weight_decay=args.weight_decay,
    )


# RMS / Ada family
def _rmsprop(param_groups, args):
    return RMSprop(
        param_groups,
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


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
    return Adagrad(
        param_groups,
        lr=args.learning_rate,
        lr_decay=args.adagrad_lr_decay,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _sparseadam(param_groups, args):
    return SparseAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
    )


# Second-order / others
def _asgd(param_groups, args):
    return ASGD(
        param_groups,
        lr=args.learning_rate,
        lambd=args.asgd_lambda,
        alpha=args.asgd_alpha,
        t0=args.asgd_t0,
        weight_decay=args.weight_decay,
    )


def _lbfgs(param_groups, args):
    return LBFGS(
        param_groups,
        lr=args.learning_rate,
        max_iter=args.lbfgs_max_iter,
        max_eval=args.lbfgs_max_eval,
        tolerance_grad=args.lbfgs_tol_grad,
        tolerance_change=args.lbfgs_tol_change,
        history_size=args.lbfgs_history,
        line_search_fn=args.lbfgs_line_search,
    )


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

# Community contributed Optimizers

def _lion(param_groups, args):
    if Lion is None:
        raise ImportError("pip install lion-pytorch to use --optimizer lion")
    return Lion(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


#    from adabelief_pytorch import AdaBelief
# AdaBelief ---------------------------------------------------------------
def _adabelief(param_groups, args):
    if AdaBelief is None:
        raise ImportError("pip install adabelief-pytorch  to use --optimizer adabelief")
    return AdaBelief(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adabelief_eps,
        weight_decay=args.weight_decay,
        rectify=False,  # keep as in the paper
    )


# Adan --------------------------------------------------------------------
def _adan(param_groups, args):
    if Adan is None:
        raise ImportError("pip install adan-pytorch  to use --optimizer adan")
    return Adan(
        param_groups,
        lr=args.learning_rate,
        betas=(
            args.beta1,
            args.beta2,
            args.adan_beta3 if hasattr(args, "adan_beta3") else 0.999,
        ),
        eps=args.adan_eps,
        weight_decay=args.adan_wd,
    )





# Yogi --------------------------------------------------------------------
def _yogi(param_groups, args):
    if Yogi is None:
        raise ImportError("pip install pytorch_optimizer  to use --optimizer yogi")
    return Yogi(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )




# Torch-Optimizer


# ---------- torch-optimizer wrappers --------------------
def _accsgd(param_groups, args):
    _needs_topt()
    return topt.AccSGD(
        param_groups,
        lr=args.learning_rate,
        kappa=1000,
        xi=10,
        small_const=0.7,
        weight_decay=args.opt_weight_decay,
    )


def _adabound(param_groups, args):
    _needs_topt()
    return topt.AdaBound(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        final_lr=args.adabound_final_lr,
        gamma=args.adabound_gamma,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _adamod(param_groups, args):
    _needs_topt()
    return topt.AdaMod(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        beta3=args.adamod_beta3,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _adamp(param_groups, args):
    _needs_topt()
    return topt.AdamP(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay
        )

def _adafactor(param_groups, args):
    _needs_topt()
    # When lr is None Adafactor uses the “relative step” schedule from
    # the paper (√t scaling + warm-up).  If the user provided
    # --learning_rate we forward it, otherwise stay in the default mode.
    lr_kw = {} if args.learning_rate is None else {"lr": args.learning_rate}
    return topt.Adafactor(
        param_groups,
        **lr_kw,
        eps2=(args.adafactor_eps_row, args.adafactor_eps_col),
        clip_threshold=args.adafactor_clip,
        decay_rate=args.adafactor_decay,
        beta1=(None if args.adafactor_beta1 < 0 else args.adafactor_beta1),
        weight_decay=args.opt_weight_decay,
        scale_parameter=args.adafactor_scale_param,
        relative_step=args.adafactor_relative_step,
        warmup_init=args.adafactor_warmup_init,
    )



def _aggmo(param_groups, args):
    _needs_topt()
    return topt.AggMo(
        param_groups, lr=args.learning_rate, betas=(0.0, 0.9, 0.99), weight_decay=args.opt_weight_decay
    )


def _diffgrad(param_groups, args):
    _needs_topt()
    return topt.DiffGrad(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _lamb(param_groups, args):
    _needs_topt()
    return topt.Lamb(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
        clamp_value=args.lamb_clamp,
        adam=args.lamb_adam,
        debias=args.lamb_debias,
    )


def _novograd(param_groups, args):
    _needs_topt()
    return topt.NovoGrad(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _pid(param_groups, args):
    _needs_topt()
    return topt.PID(
        param_groups,
        lr=args.learning_rate,
        momentum=args.pid_momentum,
        integral=args.pid_integral,
        derivative=args.pid_derivative,
        weight_decay=args.opt_weight_decay,
    )

def _qhadam(param_groups, args):
    _needs_topt()
    return topt.QHAdam(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            nus=(1.0, 1.0),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay
        )

def _qhm(param_groups, args):
    _needs_topt()
    return topt.QHM(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,  # reuse flag
        nu=0.7,
        weight_decay=args.opt_weight_decay,
    )


def _sgdp(param_groups, args):
    _needs_topt()
    return topt.SGDP(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _sgdw(param_groups, args):
    _needs_topt()
    return topt.SGDW(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        weight_decay=args.opt_weight_decay,
        nesterov=args.sgd_nesterov,
    )


def _shampoo(param_groups, args):
    _needs_topt()
    return topt.Shampoo(
        param_groups,
        lr=args.learning_rate,
        momentum=args.shampoo_momentum,
        weight_decay=args.opt_weight_decay,
        epsilon=args.shampoo_eps,
        update_freq=args.shampoo_update_freq,
    )


def _swats(param_groups, args):
    _needs_topt()
    return topt.SWATS(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )

def _yogi(param_groups, args):
    _needs_topt()
    return topt.Yogi(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay,
        )


## Apollo


def _apollo_adamw(param_groups, args):
    """
    Returns an APOLLOAdamW optimiser that shares the *same* generic
    hyper-parameters (lr / betas / eps / weight_decay) as the rest of your
    Adam-family variants, while adding Apollo’s low-rank options.

    Required extra CLI flags
    ------------------------
    --apollo_rank                int      rank of the low-rank projector
    --apollo_proj                str      ['random' | 'hadamard' | 'learned']
    --apollo_scale               int      scaling constant for projector
    --apollo_update_proj_gap     int      #steps between projector refresh
    --apollo_proj_type           str      ['std' | 'gaussian' | 'rademacher']
    """
    if APOLLOAdamW is None:
        raise ImportError(
            "Package `apollo-torch` is not installed. "
            "Run `pip install apollo-torch` or switch optimiser."
        )

    ############################################################
    # 1) Split the incoming *PyTorch* param groups into
    #    “low-rank” and “regular” groups so that users can still
    #    choose which tensors use the special update.
    ############################################################
    # A parameter is tagged “low-rank” if it has attribute `lowrank=True`
    # **or**  if user requested global low-rank via --apollo_apply_to_all.
    low_groups, reg_groups = [], []
    apply_to_all = getattr(args, "apollo_apply_to_all", False)

    for pg in param_groups:
        lr_params, reg_params = [], []
        for p in pg["params"]:
            if getattr(p, "lowrank", False) or apply_to_all:
                lr_params.append(p)
            else:
                reg_params.append(p)

        # preserve original group hyper-params
        shared = {k: v for k, v in pg.items() if k != "params"}

        if reg_params:
            reg_groups.append({"params": reg_params, **shared})
        if lr_params:
            low_groups.append(
                {
                    "params": lr_params,
                    "rank": args.apollo_rank,
                    "proj": args.apollo_proj,
                    "scale_type": "tensor",
                    "scale": args.apollo_scale,
                    "update_proj_gap": args.apollo_update_proj_gap,
                    "proj_type": args.apollo_proj_type,
                    **shared,
                }
            )

    # Stitch the list back together –– Apollo can accept mixed groups.
    pg = reg_groups + low_groups

    ############################################################
    # 2) Instantiate the optimiser
    ############################################################
    opt = APOLLOAdamW(
        pg,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )
    return opt



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
    # community contributed
    "lion": _lion,
    # from adabelief_pytorch
    "adabelief": _adabelief,
    # from adan pytorch
    "adan": _adan,
    # apollo_pytotrch
    "apollo_adamw": _apollo_adamw,
    # from torch-optimizer suite
    "adamp": _adamp,
    "adafactor": _adafactor,
    "accsgd": _accsgd,
    "adabound": _adabound,
    "adamod": _adamod,
    "aggmo": _aggmo,
    "diffgrad": _diffgrad,
    "lamb": _lamb,
    "novograd": _novograd,
    "pid": _pid,
    "qhm": _qhm,
    "sgdp": _sgdp,
    "sgdw": _sgdw,
    "shampoo": _shampoo,
    "swats": _swats,
    "qhadam": _qhadam,
    "yogi": _yogi,
}
