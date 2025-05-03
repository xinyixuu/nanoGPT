#!/usr/bin/env python3
"""
Textual app to monitor hyperparameter search results from a YAML log file.
Refreshes every N seconds, showing all runs in a DataTable view.
Press Enter on the selected cell to toggle sorting by that column; repeat to clear the sort.
Use 'h'/'l' to move the selected column left/right, with cursor following.
Press 'd' to hide the selected column, and 'o' to unhide all hidden columns.
Preserves cursor position and header order across refreshes, fully expanding to fill vertical space.
Only shows horizontal scrollbar if content overflows horizontally; vertical scrollbar appears only if content exceeds available height.
Sorting is implemented via bubble sort to preserve stability and applies new sorts on currently displayed order.
"""
import argparse
import yaml
from pathlib import Path
from typing import Optional, List, Dict
from textual.app import App, ComposeResult
from textual import events
from textual.containers import Container
from textual.widgets import DataTable, Header, Footer


def load_runs(log_file: Path) -> List[Dict]:
    docs: List[Dict] = []
    if not log_file.exists():
        return docs
    with log_file.open() as f:
        for doc in yaml.safe_load_all(f):
            if doc:
                docs.append(doc)
    return docs


class MonitorApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    Container {
        height: 1fr;
    }
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
        self.hidden: set[str] = set()
        self.sort_column: Optional[int] = None
        self.sort_reverse: bool = False
        self.table: Optional[DataTable] = None
        # Current displayed entries (maintains order across sorts)
        self.current_entries: List[Dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            DataTable(id="table", zebra_stripes=True),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.table = self.query_one(DataTable)
        # Load initial entries
        self.current_entries = load_runs(self.log_file)
        # Determine parameter columns
        keys = set()
        for e in self.current_entries:
            keys.update(e.get("config", {}).keys())
        self.param_keys = sorted(keys)

        # Base column ordering
        base = ["best_val_loss", "best_val_iter", "num_params"] + self.param_keys
        self.columns = base.copy()
        self.all_columns = base.copy()

        # Initial table setup
        self.build_table()

        # Schedule refresh and load
        self.set_interval(self.interval, self.refresh_table)
        self.refresh_table()

    def build_table(self) -> None:
        # Rebuild columns according to self.columns
        if not self.table:
            return
        self.table.clear(columns=True)
        for col in self.columns:
            self.table.add_column(col, width=max(12, len(col) + 2))

    def refresh_table(self, new_cursor: Optional[int] = None) -> None:
        if not self.table:
            return
        # Reload entries if no active sort to reflect disk changes
        if self.sort_column is None:
            self.current_entries = load_runs(self.log_file)

        entries = list(self.current_entries)
        # Apply bubble sort if requested
        if self.sort_column is not None:
            col_name = self.columns[self.sort_column]
            def get_val(e: Dict):
                if col_name in ("best_val_loss", "best_val_iter", "num_params"):
                    return e.get(col_name)
                return e.get("config", {}).get(col_name)
            n = len(entries)
            for i in range(n):
                for j in range(0, n - i - 1):
                    a = get_val(entries[j])
                    b = get_val(entries[j+1])
                    if (not self.sort_reverse and a > b) or (self.sort_reverse and a < b):
                        entries[j], entries[j+1] = entries[j+1], entries[j]
            # Update current_entries to new sorted order
            self.current_entries = entries

        # Save cursor
        old = self.table.cursor_coordinate
        row_idx = old.row if old else 0
        col_idx = new_cursor if new_cursor is not None else (old.column if old else 0)

        # Rebuild headers
        self.build_table()

        # Populate rows
        for e in entries:
            row = []
            for col in self.columns:
                if col == "best_val_loss":
                    val = f"{e.get(col, 0):.6f}"
                else:
                    raw = (
                        e.get(col)
                        if col in ("best_val_iter", "num_params")
                        else e.get("config", {}).get(col)
                    )
                    val = str(raw)
                row.append(val)
            self.table.add_row(*row)

        # Restore cursor position
        max_row = len(entries) - 1
        max_col = len(self.columns) - 1
        r = min(max(row_idx, 0), max_row)
        c = min(max(col_idx, 0), max_col)
        self.table.cursor_coordinate = (r, c)

    async def on_key(self, event: events.Key) -> None:
        if not self.table:
            return
        coord = self.table.cursor_coordinate
        if not coord:
            return
        row, col = coord.row, coord.column

        key = event.key
        if key == "enter":
            if self.sort_column == col:
                # Toggle off sort
                self.sort_column = None
                self.sort_reverse = False
            else:
                self.sort_column = col
                self.sort_reverse = False
            self.refresh_table(new_cursor=col)

        elif key in ("h", "l"):  # move columns
            target = col - 1 if key == "h" else col + 1
            if 0 <= target < len(self.columns):
                # Swap in all_columns
                name = self.columns[col]
                other = self.columns[target]
                idx1 = self.all_columns.index(name)
                idx2 = self.all_columns.index(other)
                self.all_columns[idx1], self.all_columns[idx2] = self.all_columns[idx2], self.all_columns[idx1]
                # Rebuild visible columns
                self.columns = [c for c in self.all_columns if c not in self.hidden]
                self.refresh_table(new_cursor=target)

        elif key == "d":  # hide column
            col_name = self.columns[col]
            self.hidden.add(col_name)
            self.columns = [c for c in self.all_columns if c not in self.hidden]
            self.refresh_table(new_cursor=col)

        elif key == "o":  # unhide all
            self.hidden.clear()
            self.columns = self.all_columns.copy()
            self.refresh_table(new_cursor=col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor hyperparameter search results (Textual TUI)."
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the YAML log file generated by run_experiments.py",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds",
    )
    args = parser.parse_args()

    app = MonitorApp(args.log_file, args.interval)
    app.run()


if __name__ == "__main__":
    main()

