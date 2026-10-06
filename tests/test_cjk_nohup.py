"""Check detached success/failure reporting using tiny stand-in pipelines."""

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CjkNohupTests(unittest.TestCase):
    def check_status(self, exit_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demos = root / "demos"
            demos.mkdir()
            launcher = demos / "run_cjk_nohup.sh"
            launcher.write_text((ROOT / "demos" / launcher.name).read_text(), encoding="utf-8")
            pipeline = demos / "cjk_ipa_charbpe_compare.sh"
            pipeline.write_text(f"#!/bin/bash\nprintf 'stage=%s\\n' \"$1\"\nexit {exit_code}\n", encoding="utf-8")
            log_root = root / "logs"
            env = dict(os.environ, CJK_LOG_ROOT=str(log_root), CJK_OUT_ROOT=str(root / "out"))
            result = subprocess.run(["bash", str(launcher), "charbpe", "train"],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = (log_root / "status").read_text().strip()
                if status != "RUNNING":
                    break
                time.sleep(0.05)
            self.assertEqual(status, "0" if exit_code == 0 else "1")
            self.assertEqual((log_root / "exit_code").read_text().strip(), str(exit_code))
            self.assertIn("stage=train", (log_root / "nohup.log").read_text())
            self.assertTrue((log_root / "pid").read_text().strip().isdigit())
            self.assertTrue((log_root / "finished_at").is_file())
            original_status = (log_root / "status").read_text()
            duplicate = subprocess.run(["bash", str(launcher), "charbpe", "train"],
                                       env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual((log_root / "status").read_text(), original_status)

    def test_success_writes_zero(self):
        self.check_status(0)

    def test_failure_is_normalized_and_original_code_retained(self):
        self.check_status(7)


if __name__ == "__main__":
    unittest.main()
