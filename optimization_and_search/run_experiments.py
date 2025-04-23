import json
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import os

import yaml
from rich import print
from rich.console import Console
from rich.table import Table

# Constants
LOG_DIR = Path("exploration_logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_FILENAME = "best_val_loss_and_iter.txt"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments based on a JSON configuration file."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="out",
        help="Directory to place experiment outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for run names and output directories.",
    )
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="Prepend timestamp to run names and out_dir.",
    )
    return parser.parse_args()


def load_configurations(path: str) -> list[dict]:
    """
    Load a list of experiment configurations from a JSON file.

    Returns:
        List of configuration dicts.
    """
    with open(path) as f:
        return json.load(f)


def expand_range(val):
    """
    Expand dicts with 'range' into a list of values.
    """
    if isinstance(val, dict) and "range" in val:
        r = val["range"]
        start, end = r["start"], r["end"]
        step = r.get("step", 1 if isinstance(start, int) else 0.1)
        if isinstance(start, int):
            return list(range(start, end + 1, step))
        count = int((end - start) / step) + 1
        return [start + i * step for i in range(count)]
    return val


def generate_combinations(config: dict) -> dict:
    """
    Yield all valid parameter combinations for a single config dict.

    Returns:
        Iterator of parameter-combination dicts.
    """
    groups = config.pop("parameter_groups", [{}])
    base = {
        k: (expand_range(v) if isinstance(v, dict) and "range" in v else v)
        for k, v in config.items()
        if not (isinstance(v, dict) and "conditions" in v)
    }
    base = {k: (v if isinstance(v, list) else [v]) for k, v in base.items()}
    conditionals = {
        k: v for k, v in config.items() if isinstance(v, dict) and "conditions" in v
    }

    for grp in groups:
        merged = {**base, **grp}
        keys = list(merged)
        for combo in product(*(merged[k] for k in keys)):
            combo_dict = dict(zip(keys, combo))
            valid = [combo_dict]
            for param, spec in conditionals.items():
                next_valid = []
                for c in valid:
                    if all(c.get(key) == val for key, val in spec["conditions"]):
                        opts = spec["options"]
                        for opt in opts if isinstance(opts, list) else [opts]:
                            new = dict(c)
                            new[param] = opt
                            next_valid.append(new)
                    else:
                        next_valid.append(c)
                valid = next_valid
            for v in valid:
                yield v


def format_run_name(combo: dict, base: str, prefix: str) -> str:
    """
    Create a unique run name from parameters.
    """
    parts = [str(v) for v in combo.values()]
    return f"{prefix}{base}-{'-'.join(parts)}"


def read_metrics(out_dir: str) -> dict:
    """
    Read best_val_loss_and_iter.txt and parse five metrics.

    Returns:
        Dict with keys: best_val_loss, best_val_iter, num_params,
        better_than_chance, btc_per_param.
    """
    path = Path(out_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    line = path.read_text().strip()
    loss, iteration, params, btc, btc_pp = [p.strip() for p in line.split(",")]
    return {
        "best_val_loss": float(loss),
        "best_val_iter": int(iteration),
        "num_params": int(params),
        "better_than_chance": float(btc),
        "btc_per_param": float(btc_pp),
    }


def completed_runs(log_file: Path) -> set[str]:
    """
    Return set of run names already logged in the YAML file.
    """
    if not log_file.exists():
        return set()
    runs = set()
    for doc in yaml.safe_load_all(log_file.open()):
        if doc and "formatted_name" in doc:
            runs.add(doc["formatted_name"])
    return runs


def append_log(log_file: Path, name: str, combo: dict, metrics: dict) -> None:
    """
    Append a YAML entry with run details and metrics.
    """
    entry = {"formatted_name": name, "config": combo, **metrics}
    with log_file.open("a") as f:
        yaml.safe_dump(entry, f, explicit_start=True)


def build_command(combo: dict) -> list[str]:
    """
    Construct the command-line invocation for train.py.
    """
    cmd = ["python3", "train.py"]
    for k, v in combo.items():
        if isinstance(v, bool):
            cmd.append(f"--{'' if v else 'no-'}{k}")
        elif isinstance(v, list):
            for x in v:
                cmd += [f"--{k}", str(x)]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_experiment(combo: dict, base: str, args: argparse.Namespace) -> None:
    """
    Execute one experiment combo: skip if done, run train.py, record metrics.
    """
    run_name = format_run_name(combo, base, args.prefix)
    log_file = LOG_DIR / f"{base}.yaml"
    if run_name in completed_runs(log_file):
        print(f"[yellow]Skipping already-run:[/] {run_name}")
        return

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.use_timestamp else None
    out_dir_name = f"{timestamp}_{run_name}" if timestamp else run_name
    combo["out_dir"] = os.path.join(args.output_dir, out_dir_name)

    # Show parameters
    console = Console()
    table = Table("Parameters", show_header=False)
    for k, v in combo.items():
        table.add_row(k, str(v))
    console.print(table)

    # Build and run
    cmd = build_command(combo)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read metrics and log
    metrics = read_metrics(combo["out_dir"])
    append_log(log_file, run_name, combo, metrics)


def main():
    args = parse_args()
    base = Path(args.config).stem
    configs = load_configurations(args.config)

    for cfg in configs:
        for combo in generate_combinations(cfg):
            run_experiment(combo, base, args)


if __name__ == "__main__":
    main()
