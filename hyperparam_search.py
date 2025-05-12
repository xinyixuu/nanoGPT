#!/usr/bin/env python3
"""
Greedy iterative hyper-parameter search for **train.py**

Changes vs. previous version
----------------------------
* `--random_iterations N` — run each candidate N times with **different seeds**.
  - The seed passed to `train.py` is       `baseline_seed + run_id`
    where `baseline_seed` is whatever is in the current baseline config
    (or the default 1337 if absent).
  - The candidate block in *sweep_log.yaml* now includes a `seeds:` list
    with per-seed results.
  - Efficiency is computed on the **average score** across those N runs.
* Everything else (YAML log structure, early-stop rule, resume, etc.) is
  unchanged.
"""

import argparse
import gc
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


# ───────────────────────── helpers ──────────────────────────
def dict_to_cli(d: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        elif isinstance(v, list):
            cli.append(f"--{k}")
            cli.extend(map(str, v))
        else:
            cli.extend([f"--{k}", str(v)])
    return cli


@contextmanager
def patched_argv(argv: List[str]):
    old = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def run_trial_inproc(cfg: Dict[str, Any]) -> Tuple[float, float, int]:
    """Return (best_val_loss, num_params, best_iter)."""
    from train import Trainer
    from train_args import parse_args as parse_train_args

    cli = ["train.py"] + dict_to_cli(cfg)
    with patched_argv(cli):
        args, mg, tg, lg = parse_train_args()
    tr = Trainer(args, mg, tg, lg)
    tr.train()
    loss = float(tr.best_val_loss)
    nparam = float(tr.raw_model.num_param)
    best_iter = int(getattr(tr, "iter_num_best_val_loss", 0))
    del tr
    torch.cuda.empty_cache()
    gc.collect()
    return loss, nparam, best_iter


def run_trial_subproc(cfg: Dict[str, Any]) -> Tuple[float, float, int]:
    script_dir = Path(__file__).parent
    cmd = [sys.executable, str(script_dir / "train.py")] + dict_to_cli(cfg)
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "WORLD_SIZE"}}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        print(p.stderr)
        raise RuntimeError("train.py failed")

    out_dir = Path(cfg.get("out_dir", "out"))
    line = (out_dir / "best_val_loss_and_iter.txt").read_text().strip().split(",")
    loss, best_iter, nparam = float(line[0]), int(line[1]), float(line[2])
    torch.cuda.empty_cache()
    gc.collect()
    return loss, nparam, best_iter


def load_log(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def save_log(path: Path, log: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(log, sort_keys=False))
    tmp.replace(path)


# ───────────────────────── search controller ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Greedy hyper-param search wrapper")
    ap.add_argument("--orig_settings", required=True)
    ap.add_argument("--param_names", nargs="+", required=True)
    ap.add_argument("--increments", nargs="+", type=float, required=True)
    ap.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="multiples of increment to try per parameter",
    )
    ap.add_argument(
        "--num_iterations", type=int, default=1, help="max outer search iterations"
    )
    ap.add_argument(
        "--random_iterations",
        type=int,
        default=1,
        help="how many different random seeds per candidate",
    )
    ap.add_argument("--results_file", default="sweep_log.yaml")
    ap.add_argument("--spawn_subprocess", action="store_true")
    args = ap.parse_args()

    # match increments to param list
    if len(args.increments) == 1:
        args.increments *= len(args.param_names)
    if len(args.increments) != len(args.param_names):
        sys.exit("--increments length mismatch")

    inc_map = dict(zip(args.param_names, args.increments))
    run_fn = run_trial_subproc if args.spawn_subprocess else run_trial_inproc

    baseline_cfg_master = yaml.safe_load(Path(args.orig_settings).read_text())
    log_path = Path(args.results_file)
    log = load_log(log_path)

    # Initialise log structure if new
    log.setdefault("baseline_config", deepcopy(baseline_cfg_master))
    log.setdefault("iterations", [])

    # ── restore / initialise baseline ─────────────────────────
    if log["iterations"]:
        last = log["iterations"][-1]
        baseline_cfg = deepcopy(last["baseline_config_after"])
        base_loss = last["baseline_metrics"]["loss"]
        base_score = last["baseline_metrics"]["score"]
        base_params = last["baseline_metrics"]["params"]
        cur_iter = last["iter"] + 1
    else:
        baseline_cfg = deepcopy(log["baseline_config"])
        print("[BASELINE] measuring initial config …")
        base_loss, base_params, base_best_iter = run_fn(deepcopy(baseline_cfg))
        base_score = 1 / math.exp(base_loss)
        log["iterations"].append(
            {
                "iter": -1,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "best_iter": base_best_iter,
                },
                "baseline_config_after": deepcopy(baseline_cfg),
            }
        )
        save_log(log_path, log)
        cur_iter = 0

    # ── outer greedy loop ─────────────────────────────────────
    while cur_iter < args.num_iterations:
        print(f"========== Iteration {cur_iter} ==========")
        candidates: List[Dict[str, Any]] = []
        best_choice: Tuple[float, Dict[str, Any]] | None = None

        for pname in args.param_names:
            if not isinstance(baseline_cfg.get(pname), (int, float)):
                continue
            base_val = baseline_cfg[pname]
            step = inc_map[pname]

            for m in range(1, args.iterations + 1):
                new_val = (
                    int(round(base_val + m * step))
                    if isinstance(base_val, int)
                    else float(base_val + m * step)
                )
                cfg_template = deepcopy(baseline_cfg)
                cfg_template[pname] = new_val

                # --- multiple random seeds --------------------------------
                seed0 = cfg_template.get("seed", 1337)
                seed_runs = []
                scores = []
                for r in range(args.random_iterations):
                    cfg = deepcopy(cfg_template)
                    cfg["seed"] = seed0 + r
                    print(f"[TEST] {pname}={new_val}  seed={cfg['seed']}")
                    try:
                        loss, nparam, best_it = run_fn(cfg)
                    except Exception as exc:
                        print("  ⚠", exc)
                        continue
                    score = 1 / math.exp(loss)
                    seed_runs.append(
                        {
                            "seed": cfg["seed"],
                            "loss": loss,
                            "score": score,
                            "best_iter": best_it,
                        }
                    )
                    scores.append(score)

                if not scores:  # all seeds failed
                    continue

                avg_score = sum(scores) / len(scores)
                avg_loss = -math.log(avg_score)
                d_score = avg_score - base_score
                d_param = nparam - base_params
                # eff = d_score / d_param if d_param else 0.0
                # Handle zero-cost changes:
                if d_param != 0:
                     eff = d_score / d_param
                elif d_score > 0:
                    eff = math.inf # Positive improvement at zero cost is infinitely efficient
                else:
                     eff = 0.0      # No improvement (or a loss) at zero cost


                cand = {
                    "param": pname,
                    "value": new_val,
                    "avg_loss": avg_loss,
                    "avg_score": avg_score,
                    "best_val_loss": avg_loss,  # keep same key for viewer
                    "best_iter": max(s["best_iter"] for s in seed_runs),
                    "num_params": nparam,
                    "delta_score": d_score,
                    "delta_params": d_param,
                    "efficiency": eff,
                    "seeds": seed_runs,
                }
                candidates.append(cand)

                if eff > 0:
                    if best_choice is None:
                        best_choice = (eff, cand)
                    else:
                        # Replace if eff is strictly better, OR if eff is Inf and equal, use delta_score as a tie-breaker.
                        old_eff, old_cand = best_choice
                        if (eff > old_eff) or (math.isinf(eff) and eff == old_eff and cand['delta_score'] > old_cand['delta_score']):
                             best_choice = (eff, cand)


        # -- pick or stop ---------------------------------------
        if best_choice is None:
            print("No positive-efficiency candidate — stopping.")
            log["stop_reason"] = "no_positive_efficiency"
            save_log(log_path, log)
            break

        _, chosen = best_choice
        print(
            f"[CHOSEN] {chosen['param']} → {chosen['value']}  eff={chosen['efficiency']:.3e}"
        )

        # update baseline
        baseline_cfg[chosen["param"]] = chosen["value"]
        base_loss = chosen["avg_loss"]
        base_score = chosen["avg_score"]
        base_params = chosen["num_params"]

        # log block
        log["iterations"].append(
            {
                "iter": cur_iter,
                "baseline_metrics": {
                    "loss": base_loss,
                    "score": base_score,
                    "params": base_params,
                    "best_iter": chosen["best_iter"],
                },
                "candidates": candidates,
                "chosen": chosen,
                "baseline_config_after": deepcopy(baseline_cfg),
            }
        )
        log["baseline_config"] = deepcopy(baseline_cfg)
        save_log(log_path, log)
        cur_iter += 1


if __name__ == "__main__":
    main()
