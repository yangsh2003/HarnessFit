"""Regression checks for the distributed cohort, resume markers and holdout freeze."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_group as groups
import study


class GroupTest(unittest.TestCase):
    def setUp(self):
        self.plan = study.read(groups.STUDY / "plan.json")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_interactive_b_and_c_cover_disjoint_registered_slots(self):
        owners = {}
        for team in ("A", "B", "C"):
            registered = {k for split in ("dev", "holdout")
                          for k in groups.slots(self.plan, groups.GROUPS[team], split)}
            self.assertEqual(len(registered), 600 if team == "A" else 500)
            for key in registered:
                self.assertNotIn(key, owners)
                owners[key] = team
        self.assertEqual(len(owners), 1600)
        for team in ("B", "C"):
            with patch("builtins.input", side_effect=["A", team.lower()]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(groups.prompt_group(), team)

    def test_duplicate_progress_and_foreign_selection_rejected(self):
        key = next(groups.slots(self.plan, groups.GROUPS["B"], "dev"))
        path = self.dir / "progress.json"
        study.write(path, dict(version=1, plan_sha256=groups.PLAN_SHA256,
                               updated_at="fixture", completed=[key, key]))
        with self.assertRaisesRegex(ValueError, "重复"):
            groups.read_progress(path, self.plan)
        with self.assertRaisesRegex(ValueError, "selection.json"):
            groups.verify_selection(self.dir / "selection.json", self.plan)
        record = dict(plan_sha256=groups.PLAN_SHA256, protocol_sha256=self.plan["protocol_sha256"],
                      A="00000", B="00000", dev_solves={"qwen9b": dict.fromkeys(self.plan["configs"], 0)})
        study.write(path, record)
        groups.verify_selection(path, self.plan)
        study.write(path, {**record, "B": "11110"})
        with self.assertRaisesRegex(ValueError, "选择"):
            groups.verify_selection(path, self.plan)

    def test_dry_run_without_data_and_infrastructure_error_stops_new_work(self):
        with patch("sys.argv", ["run_group.py", "--dry-run", "--tasks", str(self.dir / "missing")]), \
                patch("builtins.input", return_value="c"), patch.object(groups, "verify_endpoint") as endpoint, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            groups.main()
            self.assertIn("C 组 · dev", output.getvalue())
            endpoint.assert_not_called()
        args = argparse.Namespace(group="B", split="dev", tasks=self.dir / "tasks", runs=self.dir,
                                  base_url="http://localhost/v1", jobs=1)
        with patch.dict("os.environ"), patch.object(groups.subprocess, "run", return_value=Mock(returncode=1)) as call, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "基础设施"):
            groups.generate(groups.slots(self.plan, groups.GROUPS["B"], "dev"), args, self.plan)
        self.assertEqual(call.call_count, 1)
        args = call.call_args.args[0]
        self.assertEqual(float(args[args.index("--temperature") + 1]), self.plan["settings"]["temperature"])


if __name__ == "__main__":
    unittest.main()
