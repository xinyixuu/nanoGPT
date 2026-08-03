"""Static, vertically aligned per-token metric dashboards."""

import os
import math
import struct
import zlib


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
                    math.isfinite(row[sort_metric]),
                    row[sort_metric] if math.isfinite(row[sort_metric]) else -math.inf,
                ),
                reverse=True,
            )
            safe_dataset = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
            path = os.path.join(output_dir, f"per_token_static_{safe_dataset}_by_{label}.png")
            _write_dashboard_png(path, ordered)
            paths.append(path)
    return paths


def _write_dashboard_png(path, rows, width=1400, height=1800):
    """Render aligned scatter panels using only the Python standard library."""
    pixels = bytearray([255]) * (width * height * 3)
    left, right, top, gap = 70, 25, 30, 22
    panel_height = (height - top * 2 - gap * (len(METRICS) - 1)) // len(METRICS)
    colors = ((31, 119, 180), (214, 39, 40), (255, 127, 14), (44, 160, 44), (148, 103, 189))

    def point(x, y, color):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                px, py = x + dx, y + dy
                if 0 <= px < width and 0 <= py < height:
                    offset = (py * width + px) * 3
                    pixels[offset:offset + 3] = bytes(color)

    for panel, ((metric, _), color) in enumerate(zip(METRICS, colors)):
        y0 = top + panel * (panel_height + gap)
        y1 = y0 + panel_height - 1
        for x in range(left, width - right):
            for y in (y0, y1):
                offset = (y * width + x) * 3
                pixels[offset:offset + 3] = b"\x80\x80\x80"
        values = [row[metric] for row in rows if math.isfinite(row[metric])]
        if not values:
            continue
        low, high = min(values), max(values)
        span = high - low or 1.0
        for rank, row in enumerate(rows):
            value = row[metric]
            if not math.isfinite(value):
                continue
            x = left + round(rank * (width - left - right - 1) / max(1, len(rows) - 1))
            y = y1 - 8 - round((value - low) * (panel_height - 17) / span)
            point(x, y, color)

    raw = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as handle:
        handle.write(png)
