#!/usr/bin/env python3
"""Export only the centrally verified slot IDs, after the legacy scheduler has drained."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os

import study
from run_group import DATA, PLAN_SHA256, ROOT, STUDY, frozen_plan, slots, verified_slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="完整中央结果目录 (内含 qwen9b/)")
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--tasks", type=Path, default=Path(os.getenv("HFIT_DATA", DATA)) / "tasks.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.runs / "qwen9b").is_dir():
        parser.error("中央目录下未找到 qwen9b/，请检查 --runs")
    args.dry_run = False
    plan = frozen_plan(args)
    tasks, images = study.task_rows(args.tasks), study.read(ROOT / "image_configs.json")
    completed = [key for split in ("dev", "holdout") for key in slots(plan, plan["configs"], split)
                 if verified_slot(args.runs, key, plan, tasks, images)]
    study.write(args.output, dict(version=1, plan_sha256=PLAN_SHA256,
                                  updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  completed=completed))
    print(f"已核对 {len(completed)} / 1600 份完整预测。分发给各组的进度文件为 {args.output}")


if __name__ == "__main__":
    main()
