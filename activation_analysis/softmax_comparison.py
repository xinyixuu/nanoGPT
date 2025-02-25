import argparse
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.style import Style

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(), e_x, e_x / (1 + e_x.sum())

def relu(x):
    """Compute ReLU values."""
    relu_values = np.maximum(0, x)
    relu_sum = relu_values.sum()
    normalized_relu = relu_values / relu_sum if relu_sum != 0 else np.zeros_like(relu_values)
    return relu_values, normalized_relu

def format_percentage(value):
    """Format percentage with color."""
    if value == "NaN":
        return value
    value = float(value)
    if value > 0:
        return f"[green]+{value:.2f}%[/green]"
    elif value < 0:
        return f"[red]{value:.2f}%[/red]"
    else:
        return f"{value:.2f}%"

def format_ratio(value):
    """Format ratio with color."""
    if value == "NaN":
        return value
    value = float(value)
    eps = 0.001
    if value > 1 + eps:
        return f"[green]{value:.2f}[/green]"  # Green if scaled magnitude is greater
    elif value < 1 - eps:
        return f"[red]{value:.2f}[/red]"  # Red if scaled magnitude is smaller
    else:
        return f"{value:.2f}"

def vector_similarity(x, y):
    """Calculate the dot product of two vectors after normalizing by their magnitudes."""
    x_norm = x / (np.linalg.norm(x) + 1e-16)  # Adding a small constant to avoid division by zero
    y_norm = y / (np.linalg.norm(y) + 1e-16)
    return np.dot(x_norm, y_norm)

def magnitude_change_ratio(x, y):
    """Calculate the ratio of the magnitudes of two vectors."""
    x_mag = np.linalg.norm(x)
    y_mag = np.linalg.norm(y)
    return y_mag / (x_mag + 1e-16) if x_mag !=0 else "NaN"


def main():
    parser = argparse.ArgumentParser(description="Compare softmax and ReLU results.")
    parser.add_argument("numbers", nargs="+", type=float, help="List of numbers to process.")
    parser.add_argument("--scale", type=float, default=10.0, help="Constant to multiply input numbers by.")

    args = parser.parse_args()
    numbers = np.array(args.numbers)
    scaled_numbers = numbers * args.scale

    console = Console()

    # Calculations
    normalized, non_normalized, obo = softmax(numbers)
    scaled_normalized, scaled_non_normalized, scaled_obo = softmax(scaled_numbers)
    relu_result, normalized_relu = relu(numbers)
    scaled_relu_result, scaled_normalized_relu = relu(scaled_numbers)

    # --- Unscaled Results Table ---
    unscaled_table = Table(title="Unscaled Results", style=Style(color="cyan"))
    unscaled_table.add_column("Input")
    unscaled_table.add_column("Normalized", justify="right")
    unscaled_table.add_column("Non-normalized", justify="right")
    unscaled_table.add_column("OBO", justify="right")
    unscaled_table.add_column("ReLU", justify="right")
    unscaled_table.add_column("Normalized ReLU", justify="right")

    for i in range(len(numbers)):
        unscaled_table.add_row(
            f"{numbers[i]:.2f}",
            f"{normalized[i]:.2f}",
            f"{non_normalized[i]:.2f}",
            f"{obo[i]:.2f}",
            f"{relu_result[i]:.2f}",
            f"{normalized_relu[i]:.2f}",
        )
    console.print(unscaled_table)

    # --- Scaled Results Table ---
    scaled_table = Table(title="Scaled Results", style=Style(color="magenta"))
    scaled_table.add_column("Input")
    scaled_table.add_column("Scaled Normalized", justify="right")
    scaled_table.add_column("Scaled Non-normalized", justify="right")
    scaled_table.add_column("Scaled OBO", justify="right")
    scaled_table.add_column("Scaled ReLU", justify="right")
    scaled_table.add_column("Scaled Normalized ReLU", justify="right")

    for i in range(len(numbers)):
        scaled_table.add_row(
            f"{scaled_numbers[i]:.2f}",
            f"{scaled_normalized[i]:.2f}",
            f"{scaled_non_normalized[i]:.2f}",
            f"{scaled_obo[i]:.2f}",
            f"{scaled_relu_result[i]:.2f}",
            f"{scaled_normalized_relu[i]:.2f}",
        )
    console.print(scaled_table)

    # --- Percentage Change Table ---
    change_table = Table(title="Percentage Changes", style=Style(color="green"))
    change_table.add_column("Input")
    change_table.add_column("Normalized % Change", justify="right")
    change_table.add_column("Non-normalized % Change", justify="right")
    change_table.add_column("OBO % Change", justify="right")
    change_table.add_column("ReLU % Change", justify="right")
    change_table.add_column("Normalized ReLU % Change", justify="right")

    for i in range(len(numbers)):
        norm_change = "NaN" if normalized[i] == 0 else f"{((scaled_normalized[i] - normalized[i]) / normalized[i]) * 100:.2f}"
        non_norm_change = "NaN" if non_normalized[i] == 0 else f"{((scaled_non_normalized[i] - non_normalized[i]) / non_normalized[i]) * 100:.2f}"
        obo_change = "NaN" if obo[i] == 0 else f"{((scaled_obo[i] - obo[i]) / obo[i]) * 100:.2f}"
        relu_change = "NaN" if relu_result[i] == 0 else f"{((scaled_relu_result[i] - relu_result[i]) / relu_result[i]) * 100:.2f}"
        norm_relu_change = "NaN" if normalized_relu[i] == 0 else f"{((scaled_normalized_relu[i] - normalized_relu[i]) / normalized_relu[i]) * 100:.2f}"

        change_table.add_row(
            f"{numbers[i]:.2f}",
            format_percentage(norm_change),
            format_percentage(non_norm_change),
            format_percentage(obo_change),
            format_percentage(relu_change),
            format_percentage(norm_relu_change),
        )
    console.print(change_table)

     # --- Combined Similarity and Magnitude Table ---
    combined_table = Table(title="Vector Similarity and Magnitude Change", style=Style(color="blue"))
    combined_table.add_column("Metric")
    combined_table.add_column("Dot Product", justify="right")
    combined_table.add_column("Ratio", justify="right")

    combined_table.add_row("Softmax", f"{vector_similarity(normalized, scaled_normalized):.4f}", format_ratio(magnitude_change_ratio(normalized, scaled_normalized)))
    combined_table.add_row("Non-normalized Softmax", f"{vector_similarity(non_normalized, scaled_non_normalized):.4f}", format_ratio(magnitude_change_ratio(non_normalized, scaled_non_normalized)))
    combined_table.add_row("OBO Softmax", f"{vector_similarity(obo, scaled_obo):.4f}", format_ratio(magnitude_change_ratio(obo, scaled_obo)))
    combined_table.add_row("ReLU", f"{vector_similarity(relu_result, scaled_relu_result):.4f}", format_ratio(magnitude_change_ratio(relu_result, scaled_relu_result)))
    combined_table.add_row("Normalized ReLU", f"{vector_similarity(normalized_relu, scaled_normalized_relu):.4f}", format_ratio(magnitude_change_ratio(normalized_relu, scaled_normalized_relu)))

    console.print(combined_table)

if __name__ == "__main__":
    main()
