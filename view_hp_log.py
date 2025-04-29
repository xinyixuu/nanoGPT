#!/usr/bin/env python3
"""view_hp_log.py — Textual TUI that tail-views *sweep_log.yaml*

Features
========
* **Live refresh** – polls the YAML every 5 s.
* **Iterations list** – pick a number for metrics + candidate table.
* **Summary** – shows a compact table of the best config after every iteration, starting with the *iter −1* baseline row.

Keys
----
↑ / ↓   select in sidebar  q   quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, DataTable, Static

# ── constants ─────────────────────────────────────────────
DEFAULT_LOG = "sweep_log.yaml"
POLL_INTERVAL = 5.0  # seconds
SUMMARY_LABEL = "Summary"
HILITE_STYLE = "bold orange3"

# ── helper functions ─────────────────────────────────────


def load_yaml(path: Path) -> Dict[str, Any]:
    """Return parsed YAML or empty dict if file missing/empty."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def fnum(val: Any, spec: str) -> str:
    return spec.format(val) if isinstance(val, (int, float)) else str(val)


def metrics_panel(m: Dict[str, Any]) -> Panel:
    g = Table.grid(padding=1)
    g.add_column(justify="right")
    g.add_column()
    g.add_row("Loss", fnum(m.get("best_val_loss", m.get("loss", "-")), "{:.4f}"))
    g.add_row("Score", fnum(m.get("score", "-"), "{:.4e}"))
    g.add_row("Params", fnum(m.get("num_params", m.get("params", "-")), "{:.3e}"))
    g.add_row("Best iter", str(m.get("best_iter", "-")))
    return Panel(g, title="Iteration stats", border_style="green")


# ── TUI application ──────────────────────────────────────


class SweepViewer(App):
    CSS = """#navbox{width:16;} #main{width:1fr;}"""
    BINDINGS = [Binding("q", "quit", show=False)]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._mtime: float = 0.0
        self.iters: List[Dict[str, Any]] = []
        self.idx: int = 0
        self.show_summary: bool = False
        self.base_cfg: Dict[str, Any] = {}
        self.base_metrics: Dict[str, Any] = {}
        self.base_iter: Dict[str, Any] | None = None

    # ── compose UI ──────────────────────────
    def compose(self):  # type: ignore[override]
        yield Header()
        with Horizontal():
            # sidebar
            with Vertical(id="navbox"):
                yield Static("Iterations", classes="title")
                yield DataTable(id="nav", show_header=False, zebra_stripes=True)
            # main pane
            with Vertical(id="main"):
                yield Static(id="panel")
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    # ── mount & poll ────────────────────────
    def on_mount(self):  # type: ignore[override]
        self._load_yaml(initial=True)
        self._build_nav()
        self._refresh_view()
        self.set_interval(POLL_INTERVAL, self._poll_yaml)

    def _poll_yaml(self):
        mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0
        if mtime == self._mtime:
            return
        remember = None
        if self.iters and not self.show_summary:
            remember = self.iters[self.idx]["iter"]
        self._load_yaml()
        self._build_nav()
        if remember is not None:
            for i, it in enumerate(self.iters):
                if it["iter"] == remember:
                    self.idx = i
                    break
        self._refresh_view()

    # ── YAML load ───────────────────────────
    def _load_yaml(self, *, initial=False):
        data = load_yaml(self.log_path)
        self.base_cfg = data.get("baseline_config", {})
        self.base_metrics = data.get("baseline_metrics", {})
        self.base_iter = None
        for it in data.get("iterations", []):
            if it.get("iter") == -1:
                self.base_iter = it
                self.base_metrics = it.get("baseline_metrics", self.base_metrics)
                break
        self.iters = [
            it
            for it in data.get("iterations", [])
            if it.get("iter", 0) >= 0 and "candidates" in it
        ]
        if self.iters:
            self.idx = (
                len(self.iters) - 1 if initial else min(self.idx, len(self.iters) - 1)
            )
        else:
            self.idx = 0
        self._mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0

    # ── navigation list ─────────────────────
    def _build_nav(self):
        nav = self.query_one("#nav", DataTable)
        nav.clear(columns=True)
        nav.add_column("item")
        for it in self.iters:
            nav.add_row(str(it["iter"]))
        nav.add_row(SUMMARY_LABEL)
        nav.cursor_type = "row"
        row_coord = (
            self.idx if (self.iters and not self.show_summary) else len(self.iters),
            0,
        )
        nav.cursor_coordinate = row_coord
        nav.focus()

    def on_data_table_row_highlighted(self, e: DataTable.RowHighlighted):  # type: ignore[override]
        if e.data_table.id != "nav":
            return
        row = e.cursor_row  # type: ignore[attr-defined]
        self.show_summary = row == len(self.iters)
        if not self.show_summary and self.iters:
            self.idx = row
        self._refresh_view()

    # ── summary helpers ─────────────────────
    def _summary_data(self):
        if not self.iters:
            return ["iter"], [["-1"]]
        changed = sorted({it["chosen"]["param"] for it in self.iters})
        hdrs = ["iter", *changed, "best_loss", "best_iter", "params", "Δparams", "eff."]
        rows: List[List[Any]] = []

        # baseline (iter -1) row
        base_src = (self.base_iter or {}).get("baseline_config_after", self.base_cfg)
        base_vals = [str(base_src.get(p, "-")) for p in changed]
        rows.append(
            [
                "-1",
                *base_vals,
                fnum(self.base_metrics.get("loss", "-"), "{:.4f}"),
                str(self.base_metrics.get("best_iter", "-")),
                fnum(self.base_metrics.get("params", "-"), "{:,}"),
                "-",
                "-",
            ]
        )
        for i, it in enumerate(self.iters):
            ch, after = it["chosen"], it["baseline_config_after"]
            vals = [
                Text(str(after.get(p, "-")), style=HILITE_STYLE)
                if p == ch["param"]
                else str(after.get(p, "-"))
                for p in changed
            ]
            vals += [
                f"{ch['best_val_loss']:.4f}",
                str(ch.get("best_iter", "-")),
                f"{int(ch['num_params']):,}",
                f"{int(ch['delta_params']):,}",
                f"{ch['efficiency']:.2e}",
            ]
            rows.append([str(i), *vals])
        return hdrs, rows

    # ── UI refresh ─────────────────────────
    def _refresh_view(self):
        panel = self.query_one("#panel", Static)
        table = self.query_one("#table", DataTable)

        # waiting for data …
        if not self.iters:
            table.visible = False
            panel.update(Panel("Waiting for data… (polling)", border_style="red"))
            self.sub_title = "No data yet"
            return

        table.visible = True
        if self.show_summary:
            panel.update(
                Panel("Summary (best config per iteration)", border_style="cyan")
            )
            hdrs, rows = self._summary_data()
            table.clear(columns=True)
            table.add_columns(*hdrs)
            for r in rows:
                table.add_row(*[c if isinstance(c, Text) else str(c) for c in r])
            self.sub_title = "Summary view"
            return

        blk = self.iters[self.idx]
        panel.update(metrics_panel(blk["baseline_metrics"]))
        self.sub_title = f"Iteration {blk['iter']}  (↑/↓ nav, q quit)"
        table.clear(columns=True)
        table.add_columns(
            "param", "value", "best_loss", "best_iter", "Δscore", "Δparams", "eff."
        )
        chosen = blk["chosen"]
        for c in blk["candidates"]:
            hl = c["param"] == chosen["param"] and c["value"] == chosen["value"]
            st = "bold yellow" if hl else ""
            table.add_row(
                Text(str(c["param"]), style=st),
                Text(str(c["value"]), style=st),
                f"{c['best_val_loss']:.4f}",
                str(c.get("best_iter", "-")),
                f"{c['delta_score']:.2e}",
                f"{c['delta_params']:.2e}",
                f"{c['efficiency']:.2e}",
            )


# ── entry point ─────────────────────────────


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG)
    SweepViewer(path).run()


if __name__ == "__main__":
    main()
