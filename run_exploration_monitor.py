#!/usr/bin/env python3
"""
Textual app to monitor hyperparameter search results from a YAML log file.
Refreshes every N seconds, showing all runs in a DataTable view.
Interactive keybindings:
  Enter - toggle sort by column
  h/l   - move column left/right
  d     - hide column
  o     - unhide all columns
  x     - hide all rows matching current cell in column
  i     - invert filter: keep only rows matching current cell in column
  O     - unhide all rows (clear row filters)
  e     - export current view to CSV
  s     - save current layout

Use `--hotkeys` to print this help and exit.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header

LAYOUT_FILE = "monitor_layout.json"


def load_runs(log_file: Path) -> List[Dict]:
    docs: List[Dict] = []
    if not log_file.exists():
        return docs
    with log_file.open() as f:
        for doc in yaml.safe_load_all(f):
            if doc:
                docs.append(doc)
    return docs


HOTKEYS_TEXT = (
    "Enter: toggle sort by column\n"
    "h/l: move column left/right\n"
    "d: hide column\n"
    "o: unhide all columns\n"
    "x: hide rows matching value\n"
    "i: keep only rows matching value\n"
    "O: clear all row filters\n"
    "e: export CSV\n"
    "s: save layout (columns, hidden-cols, filters)\n"
)


class MonitorApp(App):
    CSS = """
    Screen { align: center middle; }
    Container { height: 1fr; }
    DataTable#table {
        height: 1fr;
        width: 1fr;
        overflow-x: auto;
        overflow-y: auto;
    }
    """

    def __init__(self, log_file: Path, interval: float) -> None:
        super().__init__()
        self.log_file = log_file
        self.interval = interval
        self.param_keys: List[str] = []
        self.columns: List[str] = []
        self.all_columns: List[str] = []
        self.hidden_cols: set[str] = set()
        self.sort_column: Optional[int] = None
        self.sort_reverse: bool = False
        self.table: Optional[DataTable] = None
        self.original_entries: List[Dict] = []  # unfiltered data
        self.current_entries: List[Dict] = []  # view data with row filters

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            DataTable(id="table", zebra_stripes=True),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.table = self.query_one(DataTable)
        # Load all entries
        self.original_entries = load_runs(self.log_file)
        self.current_entries = list(self.original_entries)
        # Determine columns from config keys
        keys = set()
        for e in self.original_entries:
            keys.update(e.get("config", {}).keys())
        self.param_keys = sorted(keys)
        base_cols = ["best_val_loss", "best_val_iter", "num_params"] + self.param_keys
        self.all_columns = base_cols.copy()
        self.columns = base_cols.copy()
        # Load persisted layout
        if LAYOUT_FILE.exists():
            cfg = json.load(open(LAYOUT_FILE))
            self.all_columns = cfg.get("all_columns", self.all_columns)
            self.hidden_cols = set(cfg.get("hidden_cols", []))
            self.columns = [c for c in self.all_columns if c not in self.hidden_cols]
            self.sort_column = cfg.get("sort_column")
            self.sort_reverse = cfg.get("sort_reverse", False)
            # Row filters
            self.current_entries = [e for e in self.original_entries]
            row_filters = cfg.get("row_filters", [])
            for filt in row_filters:
                col, op, val = filt
                if op == "hide":
                    self.current_entries = [
                        e
                        for e in self.current_entries
                        if str(self.get_cell(e, col)) != val
                    ]
                elif op == "keep":
                    self.current_entries = [
                        e
                        for e in self.current_entries
                        if str(self.get_cell(e, col)) == val
                    ]
        # Build and schedule
        self.build_table()
        self.set_interval(self.interval, self.refresh_table)
        self.refresh_table()

    def build_table(self) -> None:
        if not self.table:
            return
        self.table.clear(columns=True)
        for col in self.columns:
            self.table.add_column(col, width=max(12, len(col) + 2))

    def get_cell(self, entry: Dict, col_name: str):
        if col_name in ("best_val_loss", "best_val_iter", "num_params"):
            return entry.get(col_name)
        return entry.get("config", {}).get(col_name)

    def apply_bubble_sort(self) -> None:
        col_name = self.columns[self.sort_column]
        n = len(self.current_entries)
        for i in range(n):
            for j in range(n - i - 1):
                a = self.get_cell(self.current_entries[j], col_name)
                b = self.get_cell(self.current_entries[j + 1], col_name)
                if (not self.sort_reverse and a > b) or (self.sort_reverse and a < b):
                    self.current_entries[j], self.current_entries[j + 1] = (
                        self.current_entries[j + 1],
                        self.current_entries[j],
                    )

    def refresh_table(self, new_cursor: Optional[int] = None) -> None:
        if not self.table:
            return
        # Reload data if no sort and no filters
        if self.sort_column is None and self.current_entries == self.original_entries:
            self.current_entries = list(self.original_entries)
        # Apply sort
        if self.sort_column is not None:
            self.apply_bubble_sort()
        # Save cursor
        old = self.table.cursor_coordinate
        ri = old.row if old else 0
        ci = new_cursor if new_cursor is not None else (old.column if old else 0)
        # Rebuild columns and rows
        self.build_table()
        for e in self.current_entries:
            row = []
            for col in self.columns:
                val = self.get_cell(e, col)
                row.append(f"{val:.6f}" if col == "best_val_loss" else str(val))
            self.table.add_row(*row)
        # Restore cursor
        maxr, maxc = len(self.current_entries) - 1, len(self.columns) - 1
        self.table.cursor_coordinate = (min(max(ri, 0), maxr), min(max(ci, 0), maxc))

    async def on_key(self, event: events.Key) -> None:
        if not self.table:
            return
        coord = self.table.cursor_coordinate
        if not coord:
            return
        r, c = coord.row, coord.column
        key = event.key
        # Export
        if key == "e":
            fname = f"monitor_export_{int(time.time())}.csv"
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.columns)
                for e in self.current_entries:
                    w.writerow(
                        [
                            f"{self.get_cell(e, col):.6f}" if col == "best_val_loss" else str(self.get_cell(e, col))
                            for col in self.columns
                        ]
                    )
            self.bell()
        # Save layout + filters
        elif key == "s":
            cfg = {
                "all_columns": self.all_columns,
                "hidden_cols": list(self.hidden_cols),
                "sort_column": self.sort_column,
                "sort_reverse": self.sort_reverse,
                "row_filters": getattr(self, "row_filters", []),
            }
            json.dump(cfg, open(LAYOUT_FILE, "w"))
            self.bell()
        # Toggle sort
        elif key == "enter":
            if self.sort_column == c:
                self.sort_column = None
                self.sort_reverse = False
            else:
                self.sort_column = c
                self.sort_reverse = False
            self.refresh_table(new_cursor=c)
        # Move columns
        elif key in ("h", "l"):
            t = c - 1 if key == "h" else c + 1
            if 0 <= t < len(self.columns):
                n1, n2 = self.columns[c], self.columns[t]
                i1, i2 = self.all_columns.index(n1), self.all_columns.index(n2)
                self.all_columns[i1], self.all_columns[i2] = (
                    self.all_columns[i2],
                    self.all_columns[i1],
                )
                self.columns = [col for col in self.all_columns if col not in self.hidden_cols]
                self.refresh_table(new_cursor=t)
        # Hide col
        elif key == "d":
            cn = self.columns[c]
            self.hidden_cols.add(cn)
            self.columns = [col for col in self.all_columns if col not in self.hidden_cols]
            self.refresh_table(new_cursor=c)
        # Unhide cols
        elif key == "o":
            self.hidden_cols.clear()
            self.columns = self.all_columns.copy()
            self.refresh_table(new_cursor=c)
        # Hide matching rows
        elif key == "x":
            col_name = self.columns[c]
            val = str(self.get_cell(self.current_entries[r], col_name))
            self.current_entries = [
                e
                for e in self.current_entries
                if str(self.get_cell(e, col_name)) != val
            ]
            self.row_filters = getattr(self, "row_filters", []) + [(col_name, "hide", val)]
            self.refresh_table(new_cursor=r)
        # Inverse filter
        elif key == "i":
            col_name = self.columns[c]
            val = str(self.get_cell(self.current_entries[r], col_name))
            self.current_entries = [
                e
                for e in self.current_entries
                if str(self.get_cell(e, col_name)) == val
            ]
            self.row_filters = getattr(self, "row_filters", []) + [(col_name, "keep", val)]
            self.refresh_table(new_cursor=0)
        # Unhide rows
        elif key == "O":
            self.current_entries = list(self.original_entries)
            self.row_filters = []
            self.refresh_table(new_cursor=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor hyperparameter search results (Textual TUI)."
    )
    parser.add_argument("log_file", type=Path, help="Path to YAML log file")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Refresh interval seconds"
    )
    parser.add_argument(
        "--hotkeys", action="store_true", help="Print available hotkeys and exit"
    )
    args = parser.parse_args()

    if args.hotkeys:
        print(HOTKEYS_TEXT)
        sys.exit(0)

    app = MonitorApp(args.log_file, args.interval)
    app.run()


if __name__ == "__main__":
    main()

