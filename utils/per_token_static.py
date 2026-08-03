"""Static, vertically aligned per-token metric dashboards."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("training_seen_count", "Training occurrences"),
    ("val_loss", "Validation loss"),
    ("train_loss", "Sampled training loss"),
    ("vector_magnitude", "Token vector L2 magnitude"),
    ("min_pairwise_angle_deg", "Minimum pairwise angle (degrees)"),
)


def write_static_dashboards(output_dir, rows):
    """Write one five-panel PNG per dataset and requested high-to-low ordering."""
    paths = []
    datasets = sorted({row["dataset"] for row in rows})
    orderings = (
        ("frequency", "training_seen_count"),
        ("validation_loss", "val_loss"),
        ("training_loss", "train_loss"),
        ("vector_magnitude", "vector_magnitude"),
        ("minimum_pairwise_angle", "min_pairwise_angle_deg"),
    )
    for dataset in datasets:
        data = [row for row in rows if row["dataset"] == dataset]
        for label, sort_metric in orderings:
            ordered = sorted(
                data,
                key=lambda row: (
                    np.isfinite(row[sort_metric]),
                    row[sort_metric] if np.isfinite(row[sort_metric]) else -np.inf,
                ),
                reverse=True,
            )
            x = np.arange(len(ordered))
            fig, axes = plt.subplots(5, 1, figsize=(18, 18), sharex=True)
            for axis, (metric, metric_label) in zip(axes, METRICS):
                axis.scatter(x, [row[metric] for row in ordered], s=6)
                axis.set_ylabel(metric_label)
                axis.grid(alpha=0.25)
            axes[-1].set_xlabel(f"Token rank, sorted high-to-low by {label.replace('_', ' ')}")
            fig.suptitle(f"{dataset}: per-token metrics sorted by {label.replace('_', ' ')}")
            fig.tight_layout()
            safe_dataset = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
            path = os.path.join(output_dir, f"per_token_static_{safe_dataset}_by_{label}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths
