#!/usr/bin/env python3
"""Greedy iterative hyper‑parameter search for **train.py**
--------------------------------------------------------
This version replaces the old CSV ledger with a **YAML log** that keeps
per‑iteration structure and records which candidate was chosen.  Key changes:

1. **Structured log (`sweep_log.yaml`)**
   ```yaml
   baseline_config: { … }          # always the most‑recent baseline
   iterations:
     - iter: 0
       baseline_metrics: {loss: 1.23, score: 0.29, params: 4.1e6, best_iter: 4700}
       candidates:
         - {param: n_layer, value: 2,  efficiency: 3.2e‑8, …}
         - {param: n_head,  value: 2,  efficiency: 5.1e‑8, …}
       chosen: {param: n_head, value: 2, efficiency: 5.1e‑8, …}
       baseline_config_after: { … }     # new baseline for next round
   stop_reason: no_positive_efficiency  # only set if early‑stopped
   ```

2. **Greedy selection rule** – the chosen variant MUST have a positive
   Δscore/Δparams. If every candidate’s efficiency ≤ 0 the search stops early
   and `stop_reason` is set.

3. **Best‑val‑loss iteration** – we capture and log the training step (iteration
   number) at which the best validation loss was observed.

4. **Resume‑safe** – on restart we load `sweep_log.yaml`, recover the latest
   baseline config & metrics, and continue from the next unfinished iteration.
"""

import argparse
import gc
import math
import os
import subprocess
import sys
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

# ───────────────────────── helpers ────────────────────────────

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

# ───────────────────────── single training run ─────────────────────────

def run_trial_inproc(cfg: Dict[str, Any]) -> Tuple[float, float, int]:
    """Run `train.py` in‑process and return (best_val_loss, num_params, best_iter)."""
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
    torch.cuda.empty_cache(); gc.collect()
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
    torch.cuda.empty_cache(); gc.collect()
    return loss, nparam, best_iter

# ───────────────────────── YAML log helpers ────────────────────────────

def load_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def save_log(path: Path, log: Dict[str, Any]):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.dump(log, sort_keys=False))
    tmp.replace(path)

# ───────────────────────── search controller ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Greedy hyper‑param search wrapper")
    ap.add_argument("--orig_settings", required=True)
    ap.add_argument("--param_names", nargs="+", required=True)
    ap.add_argument("--increments", nargs="+", type=float, required=True)
    ap.add_argument("--iterations", type=int, required=True,
                    help="multiples of increment to try per parameter")
    ap.add_argument("--num_iterations", type=int, default=1)
    ap.add_argument("--results_file", default="sweep_log.yaml")
    ap.add_argument("--spawn_subprocess", action="store_true")
    args = ap.parse_args()

    if len(args.increments) == 1:
        args.increments *= len(args.param_names)
    if len(args.increments) != len(args.param_names):
        sys.exit("--increments must have 1 or len(param_names) values")
    inc_map = dict(zip(args.param_names, args.increments))

    baseline_cfg_master = yaml.safe_load(Path(args.orig_settings).read_text())
    log_path = Path(args.results_file)
    log = load_log(log_path)

    # Initialise log structure if new
    log.setdefault("baseline_config", deepcopy(baseline_cfg_master))
    log.setdefault("iterations", [])
    run_fn = run_trial_subproc if args.spawn_subprocess else run_trial_inproc

    # Reconstruct baseline & metrics from last completed iteration
    if log["iterations"]:
        last = log["iterations"][-1]
        baseline_cfg = deepcopy(last["baseline_config_after"])
        base_loss = last["chosen"]["best_val_loss"]
        base_score = last["chosen"]["score"]
        base_params = last["chosen"]["num_params"]
        cur_iter = last["iter"] + 1
    else:
        baseline_cfg = deepcopy(log["baseline_config"])
        print("[BASELINE] first measurement …")
        base_loss, base_params, base_best_iter = run_fn(deepcopy(baseline_cfg))
        base_score = 1 / math.exp(base_loss)
        # Store initial baseline entry so future resumes know metrics
        init_entry = {
            "iter": -1,
            "baseline_metrics": {
                "loss": base_loss, "score": base_score,
                "params": base_params, "best_iter": base_best_iter,
            },
            "baseline_config_after": deepcopy(baseline_cfg),
        }
        log["iterations"].append(init_entry)
        save_log(log_path, log)
        cur_iter = 0

    # ── main outer loop ────────────────────────────────────────────
    while cur_iter < args.num_iterations:
        print(f"========== Iteration {cur_iter} ==========")
        candidates: List[Dict[str, Any]] = []
        best_choice = None  # (eff, cand_dict)

        for pname in args.param_names:
            if not isinstance(baseline_cfg.get(pname), (int, float)):
                continue
            base_val = baseline_cfg[pname]
            step = inc_map[pname]
            for m in range(1, args.iterations + 1):
                raw_val = base_val + m * step
                cand_val = int(round(raw_val)) if isinstance(base_val, int) else float(raw_val)

                cfg = deepcopy(baseline_cfg)
                cfg[pname] = cand_val
                print(f"[TEST] {pname}={cand_val}")
                try:
                    loss, nparam, best_iter_val = run_fn(cfg)
                except Exception as e:
                    print(" ⚠", e)
                    continue

                score = 1 / math.exp(loss)
                d_score = score - base_score
                d_param = nparam - base_params
                efficiency = d_score / d_param if d_param != 0 else float("inf")

                cand = {
                    "param": pname,
                    "value": cand_val,
                    "best_val_loss": loss,
                    "best_iter": best_iter_val,
                    "score": score,
                    "num_params": nparam,
                    "delta_score": d_score,
                    "delta_params": d_param,
                    "efficiency": efficiency,
                }
                candidates.append(cand)

                if efficiency > 0 and (best_choice is None or efficiency > best_choice[0]):
                    best_choice = (efficiency, cand)

        # Choose or stop
        if best_choice is None:
            print("No candidate improved efficiency – stopping early.")
            log["stop_reason"] = "no_positive_efficiency"
            save_log(log_path, log)
            break

        _, chosen = best_choice
        print(f"[CHOSEN] {chosen['param']} → {chosen['value']}  eff={chosen['efficiency']:.3e}")

        # Update baseline
        baseline_cfg[chosen["param"]] = chosen["value"]
        base_loss, base_score, base_params = chosen["best_val_loss"], chosen["score"], chosen["num_params"]

        # Record iteration block in log
        iter_entry = {
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
        log["iterations"].append(iter_entry)
        log["baseline_config"] = deepcopy(baseline_cfg)
        save_log(log_path, log)

        cur_iter += 1

if __name__ == "__main__":
    main()

