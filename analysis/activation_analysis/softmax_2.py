import argparse
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.style import Style

console = Console()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(), e_x, e_x / (1 + e_x.sum())

def relu(x):
    relu_values = np.maximum(0, x)
    total = relu_values.sum()
    normalized_relu = relu_values / total if total else np.zeros_like(relu_values)
    return relu_values, normalized_relu

def format_percentage(value):
    if value == "NaN": return value
    value = float(value)
    return f"[green]+{value:.2f}%[/green]" if value > 0 else f"[red]{value:.2f}%[/red]" if value < 0 else f"{value:.2f}%"

def format_ratio(value):
    if value == "NaN": return value
    value = float(value)
    return f"[green]{value:.2f}[/green]" if value > 1.0001 else f"[red]{value:.2f}[/red]" if value < 0.9999 else f"{value:.2f}"

def vector_similarity(x, y):
    x_norm = x / (np.linalg.norm(x) + 1e-16)
    y_norm = y / (np.linalg.norm(y) + 1e-16)
    return np.dot(x_norm, y_norm)

def magnitude_change_ratio(x, y):
    x_mag, y_mag = np.linalg.norm(x), np.linalg.norm(y)
    return y_mag / (x_mag + 1e-16) if x_mag else "NaN"

def sum_ratio(x, y):
    return np.sum(x) / np.sum(y) if np.sum(y) else "NaN"

def create_table(title, color, columns, rows):
    table = Table(title=title, style=Style(color=color))
    for col, justify in columns:
        table.add_column(col, justify=justify)
    for row in rows:
        table.add_row(*row)
    console.print(table)

def compute_changes(base, scaled):
    changes = []
    for b, s in zip(base, scaled):
        if b == 0:
            changes.append("NaN")
        else:
            changes.append(f"{(s - b) / b * 100:.2f}")
    return [format_percentage(c) for c in changes]

def main():
    parser = argparse.ArgumentParser(description="Compare softmax and ReLU results.")
    parser.add_argument("numbers", nargs="+", type=float)
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()

    numbers = np.array(args.numbers)
    scaled_numbers = numbers * args.scale

    methods = {
        "Softmax": softmax,
        "ReLU": relu
    }

    results = {}
    for name, func in methods.items():
        results[name] = {
            'base': func(numbers),
            'scaled': func(scaled_numbers)
        }

    # Unscaled and Scaled Results Tables
    for scale_label, data, color in [
        ("Unscaled Results", numbers, "cyan"),
        ("Scaled Results", scaled_numbers, "magenta")]:
        result_type = 'base' if scale_label == 'Unscaled Results' else 'scaled'
        rows = []
        for i, num in enumerate(data):
            rows.append([
                f"{num:.2f}",
                f"{results['Softmax'][result_type][0][i]:.2f}",
                f"{results['Softmax'][result_type][1][i]:.2f}",
                f"{results['Softmax'][result_type][2][i]:.2f}",
                f"{results['ReLU'][result_type][0][i]:.2f}",
                f"{results['ReLU'][result_type][1][i]:.2f}",
            ])


        create_table(
            scale_label, color,
            [("Input", "left"), ("Normalized", "right"), ("Non-normalized", "right"), ("OBO", "right"), ("ReLU", "right"), ("Normalized ReLU", "right")],
            rows
        )

    # Percentage Changes Table
    pct_rows = []
    for i in range(len(numbers)):
        pct_rows.append([
            f"{numbers[i]:.2f}",
            *compute_changes(results['Softmax']['base'][0][i:i+1], results['Softmax']['scaled'][0][i:i+1]),
            *compute_changes(results['Softmax']['base'][1][i:i+1], results['Softmax']['scaled'][1][i:i+1]),
            *compute_changes(results['Softmax']['base'][2][i:i+1], results['Softmax']['scaled'][2][i:i+1]),
            *compute_changes(results['ReLU']['base'][0][i:i+1], results['ReLU']['scaled'][0][i:i+1]),
            *compute_changes(results['ReLU']['base'][1][i:i+1], results['ReLU']['scaled'][1][i:i+1]),
        ])

    create_table("Percentage Changes", "green",
                 [("Input", "left"), ("Normalized %", "right"), ("Non-normalized %", "right"), ("OBO %", "right"), ("ReLU %", "right"), ("Normalized ReLU %", "right")],
                 pct_rows)

    # Combined Similarity and Magnitude Table
    combined_rows = []
    for name in ["Softmax", "ReLU"]:
        for idx, sub_name in enumerate(["Normalized", "Non-normalized", "OBO"] if name == "Softmax" else ["ReLU", "Normalized ReLU"]):
            combined_rows.append([
                sub_name,
                f"{vector_similarity(results[name]['base'][idx], results[name]['scaled'][idx]):.4f}",
                format_ratio(magnitude_change_ratio(results[name]['base'][idx], results[name]['scaled'][idx])),
                format_ratio(sum_ratio(results[name]['scaled'][idx], results[name]['base'][idx]))
            ])

    create_table("Vector Similarity and Magnitude Change", "blue",
                 [("Metric", "left"), ("Dot Product", "right"), ("Magnitude Ratio", "right"), ("Sum Ratio", "right")],
                 combined_rows)

if __name__ == "__main__":
    main()

