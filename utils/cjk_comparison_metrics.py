"""Read validation-position-weighted metrics from a per-token snapshot."""

import argparse
import csv
import math


FIELDS = ("avg_target_probability", "avg_target_rank", "avg_left_probability")


def snapshot_metrics(path, iteration):
    totals = {field: 0.0 for field in FIELDS}
    counts = {field: 0 for field in FIELDS}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing probability/rank metrics")
        for row in reader:
            if int(row["iteration"]) != iteration:
                continue
            count = int(row["val_eval_count"])
            for field in FIELDS:
                value = float(row[field])
                if count > 0 and math.isfinite(value):
                    totals[field] += value * count
                    counts[field] += count
    if not all(counts.values()):
        raise ValueError(f"{path} has no evaluated probability/rank metrics at {iteration}")
    return [totals[field] / counts[field] for field in FIELDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("iteration", type=int)
    args = parser.parse_args()
    print("\t".join(f"{value:.8g}" for value in snapshot_metrics(args.csv_path, args.iteration)))


if __name__ == "__main__":
    main()
