#!/usr/bin/env python3
"""Run the frozen HarnessFit study for one teammate's configurations."""
import argparse
import itertools
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.request import Request, urlopen
import json
import sys
import subprocess

import dataset_tools
import run as runner
import study

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "study_9b_baai"
DATA = ROOT / "data/swe-bench-verified-mini"
PLAN_SHA256 = "c0c0141a628bda07357ec31e1cb446d8fe972a4736d72ef29af92300a021238a"
WEIGHTS_MANIFEST_SHA256 = "37048cc496c8992ea778fc1395f10b3c1d2dcb434f5de066f9f5c4bbf832903a"
MODEL = "qwen9b"
GROUPS = {
    "A": ("00000", "00011", "00101", "01010", "11101", "11110"),
    "B": ("00110", "01001", "10001", "10100", "11011"),
    "C": ("01100", "01111", "10010", "10111", "11000"),
}


def frozen_plan(args):
    registration = args.study / "plan.json"
    if study.sha(registration) != PLAN_SHA256:
        raise ValueError("plan.json 与 BAAI 冻结计划不匹配")
    plan = study.read(registration)
    if (study.sha(ROOT / "study.py") != plan["study_sha256"]
            or runner.protocol_hash(plan["settings"]) != plan["protocol_sha256"]):
        raise ValueError("harness 或参数与冻结计划不符，请恢复仓库中的原始版本")
    configs = [config for group in GROUPS.values() for config in group]
    if sorted(configs) != sorted(plan["configs"]):
        raise ValueError("分组未完整覆盖冻结配置")
    if args.tasks.exists() or not args.dry_run:
        plan = study.plan_for(args)
        if plan["revisions"] != study.read(registration)["revisions"]:
            raise ValueError("模型 revision 与冻结计划不符")
    return plan


def slots(plan, configs, split):
    # Issue-first order minimizes the number of distinct Docker layers in concurrent use.
    for issue, seed, config in itertools.product(plan[split], plan["seeds"], configs):
        yield config, seed, issue


def read_progress(path, plan):
    progress = study.read(path)
    if progress["version"] != 1 or progress["plan_sha256"] != PLAN_SHA256:
        raise ValueError(f"交接进度与冻结计划不匹配：{path}")
    registered = {key for split in ("dev", "holdout") for key in slots(plan, plan["configs"], split)}
    completed = set()
    for row in progress["completed"]:
        key = tuple(row)
        if len(key) != 3 or type(key[1]) is not int or key not in registered or key in completed:
            raise ValueError(f"进度文件有未知或重复槽：{row!r}")
        completed.add(key)
    return completed, progress["updated_at"]


def prompt_group():
    while True:
        answer = input("这台机器跑 B 组还是 C 组？[B/C] ").strip().upper()
        if answer in ("B", "C"):
            return answer
        print("请输入 B 或 C。", flush=True)


def verified_slot(runs, key, plan, tasks, images):
    """Validate an atomic completion before reusing it locally or advertising it to peers."""
    config, seed, issue = key
    slot = runs / MODEL / config / str(seed) / issue
    if not (slot / "result.json").is_file():
        return False
    result = study.read(slot / "result.json")
    task = {field: tasks[issue][field] for field in runner.PUBLIC}
    identity = dict(model=MODEL, config=config, seed=seed, instance_id=issue,
                    base_commit=task["base_commit"], task_sha256=runner.sha(runner.stable(task)),
                    model_revision=plan["revisions"][MODEL], runner_sha256=plan["runner_sha256"],
                    protocol_sha256=plan["protocol_sha256"], settings=plan["settings"],
                    image=runner.image_name(task))
    if any(result.get(field) != value for field, value in identity.items()):
        raise ValueError(f"已有结果来自不同实验：{slot}")
    if result["status"] == "infra_error":
        return False
    pred, trace = slot / "prediction.json", slot / "trace.jsonl"
    if not (pred.is_file() and trace.is_file()):
        return False
    if (dataset_tools.hash_file(pred) != result.get("prediction_sha256")
            or dataset_tools.hash_file(trace) != result.get("trace_sha256")):
        return False
    prediction = study.read(pred)
    if (result.get("image_id") != images[issue] or result["status"] not in
            {"submitted", "turn_limit", "context_limit", "budget_limit"}
            or prediction["instance_id"] != issue or prediction["model_name_or_path"] != MODEL
            or len(prediction["model_patch"].encode()) != result["patch_bytes"]):
        raise ValueError(f"槽内预测或镜像身份不一致：{slot}")
    return True


def verify_selection(path, plan):
    if not path.is_file():
        raise ValueError(f"留出集需要协调人用全部开发结果冻结的 selection.json：{path}")
    selection = study.read(path)
    scores = selection["dev_solves"][MODEL]
    if (selection["plan_sha256"] != PLAN_SHA256 or selection["protocol_sha256"] != plan["protocol_sha256"]
            or set(scores) != set(plan["configs"]) or selection["A"] != plan["reference_config"]
            or selection["B"] != study.choose(scores)):
        raise ValueError("selection.json 与本次冻结的开发集选择不一致")


def service_json(url):
    headers = {}
    if os.getenv("HFIT_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["HFIT_API_KEY"]
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def verify_endpoint(args, plan):
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("--base-url 必须以 /v1 结尾")
    if service_json(base[:-3] + "/version")["version"].split("+")[0] != "0.19.1":
        raise ValueError("请使用冻结实验的 vLLM 0.19.1")
    cards = [c for c in service_json(base + "/models")["data"] if c["id"] == MODEL]
    if len(cards) != 1 or cards[0]["max_model_len"] != plan["settings"]["context"]:
        raise ValueError("vLLM 需要提供 qwen9b 别名和 32768 的上下文窗口")
    served = Path(cards[0]["root"])
    if not served.is_absolute() or not served.is_dir():
        raise ValueError("请在推理服务所在机器运行入口，并以本地权重的绝对路径启动 vLLM")
    if args.model_dir and args.model_dir.resolve() != served.resolve():
        raise ValueError("vLLM 使用的权重目录与 --model-dir / HFIT_MODEL_DIR 不同")
    manifest = ROOT / "HFIT_SHA256SUMS"
    if study.sha(manifest) != WEIGHTS_MANIFEST_SHA256:
        raise ValueError("模型文件摘要清单与 BAAI 版本不符")
    for line in manifest.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if filename in {".gitattributes", "README.md", "LICENSE"}:
            continue
        if dataset_tools.hash_file(served / filename) != digest:
            raise ValueError(f"模型或分词器文件校验失败：{filename}")


def generate(todo, args, plan):
    rev = plan["revisions"][MODEL]
    if os.getenv("HFIT_MODEL_REVISION", rev) != rev:
        raise ValueError("HFIT_MODEL_REVISION 与冻结计划不匹配")
    env = dict(os.environ, HFIT_MODEL_REVISION=rev)
    command = [sys.executable, "-B", str(ROOT / "run.py"), "--tasks", str(args.tasks),
               "--out", str(args.runs), "--base-url", args.base_url, "--model", MODEL]
    for key, value in plan["settings"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])

    def run_one(key):
        config, seed, issue = key
        print(f"{args.group}/{args.split} {config} seed={seed} {issue}", flush=True)
        slot = command + ["--config", config, "--seed", str(seed), "--instance-id", issue]
        if subprocess.run(slot, env=env).returncode:
            raise RuntimeError(f"基础设施错误，检查 {args.runs / MODEL / config / str(seed) / issue} 后用原命令续跑")

    remaining = iter(todo)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        active = {pool.submit(run_one, key) for key in itertools.islice(remaining, args.jobs)}
        while active:
            finished, active = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
            active.update(pool.submit(run_one, key) for key in itertools.islice(remaining, len(finished)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, type=str.upper, help="默认交互选择 B 或 C；协调人可指定 A")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--tasks", type=Path, default=Path(os.getenv("HFIT_DATA", DATA)) / "tasks.jsonl")
    parser.add_argument("--runs", type=Path, help="结果目录，默认 runs/组名")
    parser.add_argument("--progress", type=Path, help="协调人导出的进度，默认 <study>/progress.json")
    parser.add_argument("--selection", type=Path, help="冻结的留出选择，默认 <study>/selection.json")
    parser.add_argument("--model-dir", type=Path, default=os.getenv("HFIT_MODEL_DIR"))
    parser.add_argument("--base-url", default=os.getenv("HFIT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--jobs", type=int, default=1, help="同时运行的槽数，默认 1")
    parser.add_argument("--dry-run", action="store_true", help="只核算分工，不启动 Docker 或模型")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须是正整数")
    args.group = args.group or prompt_group()
    args.runs = args.runs or ROOT / "runs" / args.group
    plan = frozen_plan(args)
    completed, stamp = read_progress(args.progress or args.study / "progress.json", plan)
    all_slots = list(slots(plan, GROUPS[args.group], args.split))
    outstanding = [key for key in all_slots if key not in completed]
    todo = outstanding
    if args.tasks.is_file() and args.runs.exists():
        tasks, images = study.task_rows(args.tasks), study.read(ROOT / "image_configs.json")
        todo = [key for key in outstanding if not verified_slot(args.runs, key, plan, tasks, images)]
    print(f"{args.group} 组 · {args.split} · {' '.join(GROUPS[args.group])}\n"
          f"总计 {len(all_slots)}，中央已完成 {len(all_slots) - len(outstanding)}（{stamp}），"
          f"另在本机核实 {len(outstanding) - len(todo)}，待运行 {len(todo)}。\n结果目录 {args.runs}", flush=True)
    if args.dry_run or not todo:
        return
    if args.split == "holdout":
        verify_selection(args.selection or args.study / "selection.json", plan)
    verify_endpoint(args, plan)
    generate(todo, args, plan)
    print(f"{args.group} 组 {args.split} 生成完毕。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, KeyError, ValueError, RuntimeError) as exc:
        sys.exit(f"实验已停止：{exc}")
