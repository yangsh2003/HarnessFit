"""Synthetic checks for split isolation, paired reporting and artifact linkage."""

import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import study


class StudyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.object(study, "HERE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        runtime = patch.object(study.runner, "HERE", self.root)
        runtime.start()
        self.addCleanup(runtime.stop)
        for name in (
            "run.py",
            "agent.py",
            "sandbox.py",
            "image_configs.json",
            "images.lock.json",
        ):
            (self.root / name).write_text("# frozen test fixture\n")
        self.tasks = {
            f"repo__pkg-{i}": dict(
                instance_id=f"repo__pkg-{i}",
                repo="repo/pkg",
                base_commit="1" * 40,
                problem_statement="Minimal fixture",
            )
            for i in range(4)
        }
        tasks_file = self.root / "tasks.jsonl"
        tasks_file.write_text(
            "".join(json.dumps(x) + "\n" for x in self.tasks.values())
        )
        seen = self.root / "seen.txt"
        seen.write_text(next(iter(self.tasks)) + "\n")
        self.args = SimpleNamespace(
            study=self.root / "study",
            runs=self.root / "runs",
            tasks=tasks_file,
            seen_ids=seen,
            dev_size=2,
            split_seed=7607,
            baseline="small",
            target="large",
            api_model=[],
            seeds=[1, 2],
            grader_timeout=1800,
            revision=["small=v1", "large=v2"],
            full=True,
            **study.BUDGET,
        )
        study.init(self.args)
        self.plan = study.plan_for(self.args)

    def fill(self, split):
        for model, issue, config, seed in itertools.product(
            ["small", "large"], self.plan[split], study.GRID, [1, 2]
        ):
            slot = self.args.runs / model / config / str(seed) / issue
            if split == "dev":
                resolved = config == study.GRID[int(model == "large")]
            else:
                resolved = config == study.GRID[1] and (
                    issue == self.plan[split][0] or seed == 1
                )
            result = dict(
                instance_id=issue,
                model=model,
                config=config,
                seed=seed,
                base_commit="1" * 40,
                protocol_sha256=self.plan["protocol_sha256"],
                model_revision=self.plan["revisions"][model],
                task_sha256=study.runner.sha(study.runner.stable(self.tasks[issue])),
                image_id="image",
                status="submitted",
                settings=study.BUDGET,
                runner_sha256=self.plan["runner_sha256"],
                prompt_tokens=10,
                completion_tokens=5,
                elapsed_seconds=1,
            )
            study.write(slot / "result.json", result)
            study.write(
                slot / "prediction.json",
                {"instance_id": issue, "model_patch": "fixture"},
            )
            (slot / "trace.jsonl").write_text("{}\n")
            result.update(
                prediction_sha256=study.sha(slot / "prediction.json"),
                trace_sha256=study.sha(slot / "trace.jsonl"),
            )
            study.write(slot / "result.json", result)
            category = "resolved" if resolved else "unresolved"
            study.write(
                slot / "official.json",
                {"submitted_ids": [issue], category + "_ids": [issue]},
            )
            study.write(
                slot / "verdict.json",
                dict(
                    instance_id=issue,
                    resolved=resolved,
                    status="resolved" if resolved else "unresolved",
                    prediction_sha256=study.sha(slot / "prediction.json"),
                    origin="fixture",
                    report="official.json",
                    grader_version=study.JUDGE,
                    grader_timeout=1800,
                    eval_sha256=study.ROW_HASH["evaluation.jsonl"],
                ),
            )

    def test_paired_pipeline(self):
        self.assertIn(next(iter(self.tasks)), self.plan["dev"])
        self.assertFalse(set(self.plan["holdout"]) & set(self.plan["dev"]))
        self.fill("dev")
        # Selection must work with no holdout outcomes or traces on disk.
        study.selection(self.args, self.plan, self.tasks)
        sel = study.locked(self.args)
        self.assertEqual((sel["A"], sel["B"]), tuple(study.GRID[:2]))
        self.fill("holdout")
        study.report(self.args, self.plan, self.tasks)
        pair = study.read(self.args.study / "paired.json")
        self.assertEqual(
            (pair["net_issues"], pair["N"], pair["delta_pp"]), (1.5, 2, 75)
        )
        self.assertEqual([p["delta"] for p in pair["per_seed"]], [2, 1])
        self.assertEqual(
            study.read(self.args.study / "routing.json")["oracle"]["tokens_per_solve"],
            4 * 16 * 15 / 3,
        )
        path = Path(pair["case"]["A"]["prediction"])
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "missing/invalid"):
            study.report(self.args, self.plan, self.tasks)

    def test_pilot_and_scheduler(self):
        self.args.model = "small"
        self.args.split = "dev"
        self.args.pilot = True
        self.args.paired_only = False
        self.args.base_url = None
        self.args.limit = 3
        self.args.jobs = 2
        with patch.dict(study.os.environ, {"HFIT_MODEL_REVISION": "v1"}):
            with patch.object(study.subprocess, "run") as execute:
                study.run(self.args, self.plan)
                self.assertEqual(execute.call_count, 3)
                for call in execute.call_args_list:
                    cmd = call.args[0]
                    self.assertIn(
                        cmd[cmd.index("--config") + 1], self.plan["pilot_configs"]
                    )
                    self.assertEqual(call.kwargs["env"]["HFIT_MODEL_REVISION"], "v1")
        self.fill("dev")
        study.pilot_report(self.args, self.plan, self.tasks)
        out = study.read(self.args.study / "pilot.json")
        self.assertEqual(out["pairs"], len(self.plan["pilot"]) * 2)
        self.assertEqual(out["model"], "small")

    def test_equal_ranking_and_fraction(self):
        self.assertEqual(len(study.GRID), 16)
        for x, y in itertools.combinations(range(5), 2):
            self.assertEqual(len({(c[x], c[y]) for c in study.GRID}), 4)
            self.assertEqual(sum(c[x] == c[y] for c in study.GRID), 8)
        scores = dict.fromkeys(study.GRID, 0)
        self.assertIsNone(study.tau_b(scores, scores))
        self.assertEqual(study.choose(scores), "00000")

    def test_protocol_drift(self):
        (self.root / "agent.py").write_text("# altered")
        with self.assertRaisesRegex(ValueError, "Runner modules"):
            study.plan_for(self.args)


if __name__ == "__main__":
    unittest.main()
