#!/usr/bin/env python3
"""
plot_view.py
============

Utility for visualising the “current view” from the Textual hyper-parameter
monitor.

Features
--------
• Scatter plot with matplotlib.
• Legend entry for every point (no text clutter on the canvas).
• Title formatted as “<y-col> vs <x-col>”.
• Optional lines connecting points that share the same *group key*.

Typical use inside the TUI
--------------------------
    import plot_view
    plot_view.plot_rows(
        current_entries,
        x="best_val_iter",
        y="best_val_loss",
        label="formatted_name",
        connect_by="seed",   # optional
    )

CLI usage
---------
    $ python plot_view.py view.json \
          --x best_val_iter \
          --y best_val_loss \
          --label formatted_name \
          --connect-by seed
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, DefaultDict

import itertools
import matplotlib.pyplot as plt


# ───────────────────────────── helpers ──────────────────────────────


def _extract(entry: Dict[str, Any], key: str) -> Any:
    """Return entry[key] if present, else entry["config"][key], else None."""
    if key in entry:
        return entry[key]
    return entry.get("config", {}).get(key)


# ───────────────────────────── public API ───────────────────────────


def plot_rows(
    rows: List[Dict[str, Any]],
    *,
    x: str,
    y: str,
    label: str = "formatted_name",
    connect_by: str | None = None,
) -> None:
    """
    Scatter-plot *y* vs *x*.

    Parameters
    ----------
    rows : list of run dictionaries.
    x, y : str
        Column names for the X and Y axes.
    label : str, default "formatted_name"
        Column used for legend labels (duplicates collapsed).
    connect_by : str | None
        If provided, points that share the same value for this key are joined
        by a line (sorted by X).
    """
    xs, ys, lbls = [], [], []
    groups: DefaultDict[Any, List[tuple[float, float, str]]] = defaultdict(list)

    for run in rows:
        xv, yv = _extract(run, x), _extract(run, y)
        if xv is None or yv is None:
            continue
        lbl = str(_extract(run, label) or "")
        if connect_by:
            gid = _extract(run, connect_by)
            groups[gid].append((xv, yv, lbl))
        else:
            xs.append(xv)
            ys.append(yv)
            lbls.append(lbl)

    if connect_by:
        # Flatten groups for scatter plotting
        for pts in groups.values():
            for xv, yv, lbl in pts:
                xs.append(xv)
                ys.append(yv)
                lbls.append(lbl)

    if not xs:
        raise ValueError("No plottable data in supplied rows")

    plt.figure()

    seen: Set[str] = set()
    if not connect_by:
        for xv, yv, lbl in zip(xs, ys, lbls):
            kw = {"label": lbl} if lbl not in seen else {"label": "_nolegend_"}
            seen.add(lbl)
            plt.scatter(xv, yv, s=40, **kw)

    if connect_by:

        # Consistent colours for points & their connecting line
        colour_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        for gid, pts in groups.items():
            col = next(colour_cycle)

            # Plot individual points in this group
            for xv, yv, _ in pts:
                plt.scatter(xv, yv, s=40, color=col, label="_nolegend_")

            # Plot connecting line (if ≥2 points)
            if len(pts) >= 2:
                pts_sorted = sorted(pts, key=lambda t: t[0])  # sort by X
                plt.plot(
                    [p[0] for p in pts_sorted],
                    [p[1] for p in pts_sorted],
                    color=col,
                    linewidth=1,
                    alpha=0.7,
                    label=f"{connect_by}={gid}",
                )

    plt.xlabel(x)
    plt.ylabel(y)
    title = f"{y} vs {x}"
    if connect_by:
        title += f"  (lines grouped by '{connect_by}')"
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


# ───────────────────────────── CLI wrapper ──────────────────────────


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Scatter-plot an exported monitor view (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("view_json", type=Path, help="Path to JSON from the TUI")
    ap.add_argument("--x", default="best_val_iter", help="Key for the X axis")
    ap.add_argument("--y", default="best_val_loss", help="Key for the Y axis")
    ap.add_argument("--label", default="formatted_name", help="Key for legend labels")
    ap.add_argument(
        "--connect-by",
        dest="connect_by",
        default=None,
        help="Join points that share this key with a line",
    )
    args = ap.parse_args()

    data = json.loads(args.view_json.read_text())["rows"]
    plot_rows(
        data,
        x=args.x,
        y=args.y,
        label=args.label,
        connect_by=args.connect_by,
    )


if __name__ == "__main__":
    _cli()
