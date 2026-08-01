import unittest

from run_exploration_monitor import MonitorApp


class ColumnSettingRemapTests(unittest.TestCase):
    def test_colour_and_sort_settings_follow_reordered_columns(self):
        app = object.__new__(MonitorApp)
        app.columns = ["gamma", "alpha", "beta"]
        app.colour_columns = {0: "high_low", 2: "low_high"}
        app.sort_stack = [(1, True), (0, False)]

        app._remap_indexed_column_settings(["alpha", "beta", "gamma"])

        self.assertEqual(app.colour_columns, {1: "high_low", 0: "low_high"})
        self.assertEqual(app.sort_stack, [(2, True), (1, False)])


if __name__ == "__main__":
    unittest.main()
