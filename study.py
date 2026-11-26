#!/usr/bin/env python3
"""Pre-register and analyze the HarnessFit experiments. Standard library only."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import run as runner
from dataset_tools import JUDGE, ROW_HASH

FACTORS = ("context", "planning", "acceptance", "tools", "retry")
GRID = [
    "".join(map(str, v)) + str(sum(v) % 2) for v in itertools.product((0, 1), repeat=4)
]
# A full-rank eight-row subset maximizing det(X'X) for intercept + five main effects.
API_GRID = ["00000", "00011", "00101", "00110", "01001", "01010", "10001", "11110"]
HERE = Path(__file__).resolve().parent
# CLI names and defaults of the runner, fixed before selection.
BUDGET = runner.DEFAULTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_id(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected a model or issue ID, got {value!r}")
    return runner.safe_component(value)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as f:
        json.dump(
            value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        f.write("\n")
    try:
        if frozen:
            try:
                os.link(f.name, path)
            except FileExistsError:
                if read(path) != value:
                    raise ValueError(
                        f"Frozen file conflicts with supplied inputs: {path}"
                    )
        else:
            os.replace(f.name, path)
    finally:
        Path(f.name).unlink(missing_ok=True)


def task_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        i = safe_id(task["instance_id"])
        if i in rows:
            raise ValueError(f"Duplicate ID {i}")
        if (
            not task.get("problem_statement")
            or not task.get("repo")
            or not task.get("base_commit")
        ):
            raise ValueError(f"Incomplete public fields for {i}")
        rows[i] = task
    return rows


def init(args):
    tasks = task_rows(args.tasks)
    seen = set(args.seen_ids.read_text().split()) if args.seen_ids else set()
    if not seen <= tasks.keys() or len(seen) > args.dev_size:
        raise ValueError(
            "Seen IDs must belong to this cohort and fit entirely in dev. Supply a fresh holdout if needed."
        )
    if not 0 < args.dev_size < len(tasks):
        raise ValueError(
            "dev-size must leave at least one dev issue and one holdout issue."
        )
    if len(set(args.seeds)) != 2 or min(args.seeds) < 0:
        raise ValueError("Two distinct non-negative seeds are required.")
    single = getattr(args, "single_model", False)
    open_models = [args.baseline] if single else [args.baseline, args.target]
    models = [safe_id(m) for m in open_models + args.api_model]
    if len(set(models)) != len(models):
        raise ValueError("Model IDs must be unique.")
    revisions = {}
    for entry in args.revision:
        model, sep, revision = entry.partition("=")
        if (
            not sep
            or model not in models
            or not revision.strip()
            or revision == "unspecified"
            or model in revisions
        ):
            raise ValueError(
                "--revision requires a unique registered MODEL=CHECKPOINT_REVISION"
            )
        revisions[model] = revision
    settings = {k: getattr(args, k) for k in BUDGET}
    if (
        args.grader_timeout < 1
        or not 0 <= settings["temperature"] <= 2
        or any(v <= 0 for k, v in settings.items() if k != "temperature")
        or settings["max_output"] >= settings["context"]
    ):
        raise ValueError(
            "Invalid pre-registered limits; check context, generation limits, temperature and timeout."
        )
    ranked = sorted(
        set(tasks) - seen,
        key=lambda i: hashlib.sha256(f"{args.split_seed}/{i}".encode()).digest(),
    )
    dev = sorted(seen.union(ranked[: args.dev_size - len(seen)]))
    plan = dict(
        version=1,
        factors=FACTORS,
        configs=GRID,
        api_configs=API_GRID,
        seeds=args.seeds,
        split_seed=args.split_seed,
        baseline=models[0],
        target=models[0] if single else models[1],
        comparison="basic_vs_selected" if single else "transfer_vs_reselection",
        reference_config="00000" if single else None,
        api_models=models[2:],
        dev=dev,
        holdout=sorted(set(tasks) - set(dev)),
        previously_seen=sorted(seen),
        pilot=(sorted(seen) + ranked[: args.dev_size - len(seen)])[:10],
        pilot_configs=["00000", "11110"],
        tasks_sha256=sha(args.tasks),
        runner_sha256=sha(HERE / "run.py"),
        study_sha256=sha(__file__),
        modules={f: sha(HERE / f) for f in ("agent.py", "sandbox.py")},
        settings=settings,
        protocol_sha256=runner.protocol_hash(settings),
        grader_timeout=args.grader_timeout,
        grader_version=JUDGE,
        revisions=revisions,
    )
    # Normalize tuples before comparing an existing JSON document.
    write(args.study / "plan.json", json.loads(json.dumps(plan)), frozen=True)
    for split in ("dev", "holdout", "pilot"):
        (args.study / f"{split}.ids").write_text("".join(i + "\n" for i in plan[split]))
    print(f"{args.study}/plan.json   dev={len(dev)} holdout={len(tasks) - len(dev)}")


def plan_for(args):
    p = read(args.study / "plan.json")
    if (
        sha(args.tasks) != p["tasks_sha256"]
        or sha(__file__) != p["study_sha256"]
        or sha(HERE / "run.py") != p["runner_sha256"]
    ):
        raise ValueError(
            "Task file, runner or analysis code changed after pre-registration. Use the frozen version or a new study."
        )
    if any(sha(HERE / f) != h for f, h in p["modules"].items()):
        raise ValueError(
            "Runner modules changed; restore the frozen version or create a new study."
        )
    if runner.protocol_hash(p["settings"]) != p["protocol_sha256"]:
        raise ValueError("Frozen runner protocol hash mismatch")
    for model in [p["baseline"], p["target"]] + p["api_models"]:
        lock = args.study / f"revision-{model}.json"
        if lock.exists():
            registration = read(lock)
            if registration.get("plan_sha256") != sha(args.study / "plan.json"):
                raise ValueError(f"Revision lock belongs to another plan: {lock}")
            p["revisions"][model] = registration["revision"]
    return p


def locked(args):
    selection = read(args.study / "selection.json")
    if selection["plan_sha256"] != sha(args.study / "plan.json"):
        raise ValueError("selection.json belongs to another plan.")
    return selection


def model_grid(p, model):
    if model in (p["baseline"], p["target"]):
        return p["configs"], p["seeds"]
    if model in p["api_models"]:
        return p["api_configs"], p["seeds"][:1]
    raise ValueError(f"Unknown model in study: {model}")


def run(args, p):
    configs, seeds = model_grid(p, args.model)
    issue_list = p[args.split]
    if args.pilot:
        if args.split != "dev" or args.model != p["baseline"]:
            raise ValueError("--pilot is limited to the baseline model on dev")
        issue_list = p["pilot"]
        configs = p["pilot_configs"]
    if args.split == "holdout":
        sel = locked(args)
        if args.paired_only:
            if args.model != p["target"]:
                raise ValueError("--paired-only is for the target model on holdout.")
            configs = list(dict.fromkeys([sel["A"], sel["B"]]))
    elif args.paired_only:
        raise ValueError("--paired-only requires --split holdout")
    env = [
        sys.executable,
        str(HERE / "run.py"),
        "--out",
        str(args.runs),
        "--tasks",
        str(args.tasks),
        "--model",
        args.model,
    ]
    for k, v in p["settings"].items():
        env.extend(["--" + k.replace("_", "-"), str(v)])
    if args.base_url:
        env += ["--base-url", args.base_url]
    revision = p["revisions"].get(args.model)
    supplied = os.getenv("HFIT_MODEL_REVISION")
    if supplied and revision and supplied != revision:
        raise ValueError("HFIT_MODEL_REVISION differs from the registered revision.")
    revision = revision or supplied
    if not revision or revision == "unspecified":
        raise ValueError(
            "Set HFIT_MODEL_REVISION to the served checkpoint revision before this model's first run."
        )
    write(
        args.study / f"revision-{args.model}.json",
        dict(plan_sha256=sha(args.study / "plan.json"), revision=revision),
        frozen=True,
    )
    runtime_env = dict(os.environ, HFIT_MODEL_REVISION=revision)
    if args.jobs < 1 or (args.limit is not None and args.limit < 0):
        raise ValueError("--jobs must be positive; --limit must be nonnegative.")
    jobs = itertools.product(issue_list, seeds, configs)
    if args.limit is not None:
        jobs = itertools.islice(jobs, args.limit)

    def run_one(job):
        i, seed, config = job
        print(args.model, config, seed, i, flush=True)
        subprocess.run(
            env + ["--instance-id", i, "--seed", str(seed), "--config", config],
            check=True,
            env=runtime_env,
        )

    # Feed at most jobs slots; stop issuing work when a runner reports infrastructure failure.
    # Already-running jobs finish and keep their atomic results for resumption.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        active = {
            pool.submit(run_one, job) for job in itertools.islice(jobs, args.jobs)
        }
        while active:
            finished, active = wait(active, return_when=FIRST_COMPLETED)
            for f in finished:
                f.result()
            active.update(
                pool.submit(run_one, job)
                for job in itertools.islice(jobs, len(finished))
            )


def outcome(root, model, cfg, seed, issue, p, tasks):
    slot = root / model / cfg / str(seed) / issue
    try:
        result = read(slot / "result.json")
        if any(
            result.get(k) != v
            for k, v in (
                ("instance_id", issue),
                ("config", cfg),
                ("seed", seed),
                ("model", model),
            )
        ):
            raise ValueError("result identity mismatch")
        if result.get("status") == "infra_error":
            raise ValueError("runner infrastructure error")
        if (
            result.get("base_commit") != tasks[issue]["base_commit"]
            or result.get("runner_sha256") != p["runner_sha256"]
        ):
            raise ValueError("commit or runner mismatch")
        if result.get("model_revision") != p["revisions"].get(model) or not p[
            "revisions"
        ].get(model):
            raise ValueError("unregistered or mismatched checkpoint revision")
        if result.get("protocol_sha256") != p["protocol_sha256"] or not result.get(
            "image_id"
        ):
            raise ValueError("missing runner protocol or image identity")
        actual_settings = result.get("settings", {})
        if any(actual_settings.get(k) != v for k, v in p["settings"].items()):
            raise ValueError("budget mismatch")
        verdict = read(slot / "verdict.json")
        digest = sha(slot / "prediction.json")
        if (
            digest != result.get("prediction_sha256")
            or digest != verdict["prediction_sha256"]
            or verdict.get("instance_id") != issue
        ):
            raise ValueError("verdict/prediction mismatch")
        public = {k: tasks[issue][k] for k in runner.PUBLIC}
        if result.get("task_sha256") != runner.sha(runner.stable(public)):
            raise ValueError("task content differs from locked cohort")
        if (
            verdict.get("grader_version") != p["grader_version"]
            or verdict.get("grader_timeout") != p["grader_timeout"]
            or verdict.get("eval_sha256") != ROW_HASH["evaluation.jsonl"]
        ):
            raise ValueError("different evaluator, timeout, or private test data")
        official = read(slot / verdict["report"])
        category = verdict["status"] + "_ids"
        if issue not in official["submitted_ids"] or issue not in official.get(
            category, []
        ):
            raise ValueError("reported label not corroborated by the official report")
        resolved = verdict["resolved"]
        if type(resolved) is not bool or verdict["status"] not in (
            "resolved",
            "unresolved",
            "empty_patch",
        ):
            raise ValueError(f"no official label: {verdict.get('status')}")
        if (verdict["status"] == "resolved") != resolved:
            raise ValueError("inconsistent verdict")
        return {"y": int(resolved), "r": result, "v": verdict, "dir": str(slot)}
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        return {"error": str(e), "dir": str(slot)}


def collect(args, p, tasks, model, split, configs=None):
    all_cfg, seeds = model_grid(p, model)
    return {
        (i, s, c): outcome(args.runs, model, c, s, i, p, tasks)
        for i, s, c in itertools.product(p[split], seeds, configs or all_cfg)
    }


def audit(args, model, split, cells):
    bad = [
        {"issue": i, "seed": s, "config": c, **x}
        for (i, s, c), x in cells.items()
        if "error" in x
    ]
    write(
        args.study / f"audit-{model}-{split}.json",
        {
            "valid": len(cells) - len(bad),
            "planned": len(cells),
            "missing_or_invalid": bad,
        },
    )
    if bad:
        raise ValueError(
            f"{len(bad)}/{len(cells)} missing/invalid slots. Inspect audit-{model}-{split}.json, repair and rerun."
        )


def summed_cost(rows, fields):
    parts = [x["r"].get(f) for x in rows for f in fields]
    if all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in parts):
        return sum(parts)
    return None


def summarize(rows):
    n = len(rows)
    solved = sum(x["y"] for x in rows)
    tokens = summed_cost(rows, ["prompt_tokens", "completion_tokens"])
    elapsed = summed_cost(rows, ["elapsed_seconds"])
    failures = [x for x in rows if not x["y"]]

    def diag(x):
        return x["r"].get("diagnostics", {})

    counters = {
        "no_file_edit": sum(diag(x).get("edits_seen") is False for x in failures),
        "repeated_action_observed": sum(
            diag(x).get("repeated_actions", 0) > 0 for x in failures
        ),
        "submitted_without_recorded_test_action": sum(
            x["r"]["status"] == "submitted" and diag(x).get("valid_test_runs") == 0
            for x in failures
        ),
    }
    failure_status = dict(Counter(x["r"]["status"] for x in failures))
    return dict(
        attempts=n,
        solved=solved,
        rate=solved / n,
        tokens=tokens,
        tokens_per_solve=tokens / solved if solved and tokens is not None else None,
        seconds_per_solve=elapsed / solved if solved and elapsed is not None else None,
        unresolved_by_stop_reason=failure_status,
        failure_indicators=counters,
    )


def rows_for_config(cells, config):
    return [x for (_, _, c), x in cells.items() if c == config]


def rates(cells, configs):
    return {c: summarize(rows_for_config(cells, c)) for c in configs}


def grid_output(args, p, tasks, model, split):
    cfg, seeds = model_grid(p, model)
    cells = collect(args, p, tasks, model, split)
    audit(args, model, split, cells)
    identities(cells)
    summary = rates(cells, cfg)
    by_seed = {
        s: {
            c: summarize([x for (i, t, k), x in cells.items() if t == s and k == c])
            for c in cfg
        }
        for s in seeds
    }
    points = [x["rate"] for x in summary.values()]
    out = dict(
        model=model,
        split=split,
        configs=summary,
        by_seed=by_seed,
        mean=statistics.mean(points),
        sd_population=statistics.pstdev(points),
    )
    write(args.study / f"grid-{model}-{split}.json", out)
    return cells, summary


def tau_b(a, b):
    """Kendall tau-b including tied harness solve counts; undefined for a constant ranking."""
    con = dis = t1 = t2 = 0
    for x, y in itertools.combinations(a, 2):
        da = a[x] - a[y]
        db = b[x] - b[y]
        if da * db > 0:
            con += 1
        elif da * db < 0:
            dis += 1
        elif da == 0 and db != 0:
            t1 += 1
        elif db == 0 and da != 0:
            t2 += 1
    denom = math.sqrt((con + dis + t1) * (con + dis + t2))
    return (con - dis) / denom if denom else None


def identities(cells):
    protocols = {x["r"]["protocol_sha256"] for x in cells.values()}
    if len(protocols) != 1:
        raise ValueError("Mixed runner protocols in one experimental grid")
    images = {}
    for (issue, _, _), x in cells.items():
        actual = x["r"]["image_id"]
        if images.setdefault(issue, actual) != actual:
            raise ValueError(f"Mixed image IDs for {issue}")
    return next(iter(protocols)), images


def choose(score):
    return min(score, key=lambda c: (-score[c], c))


def router(p, tasks, cells):
    # Empirical-Bayes smoothed per-repository solve rates. Four pseudo-issues at global mean.
    configs = p["configs"]
    seeds = p["seeds"]
    policy = {}
    for i in p["holdout"]:
        repo = tasks[i]["repo"]
        similar = [j for j in p["dev"] if tasks[j]["repo"] == repo]

        def estimate(c):
            global_rate = statistics.mean(x["y"] for x in rows_for_config(cells, c))
            local_solves = sum(
                cells[j, s, c]["y"] for j, s in itertools.product(similar, seeds)
            )
            strength = 4 * len(seeds)
            return (local_solves + strength * global_rate) / (
                len(similar) * len(seeds) + strength
            )

        policy[i] = choose({c: estimate(c) for c in configs})
    return policy


def selection(args, p, tasks):
    scores = {}
    common = None
    for m in dict.fromkeys((p["baseline"], p["target"])):
        cells, stats = grid_output(args, p, tasks, m, "dev")
        current = identities(cells)
        if common is not None and current != common:
            raise ValueError("9B and target dev grids differ in protocol or image IDs")
        common = current
        scores[m] = {c: v["solved"] for c, v in stats.items()}
    a = p.get("reference_config") or choose(scores[p["baseline"]])
    b = choose(scores[p["target"]])
    record = dict(
        plan_sha256=sha(args.study / "plan.json"),
        A=a,
        B=b,
        dev_solves=scores,
        protocol_sha256=common[0],
        comparison=p.get("comparison", "transfer_vs_reselection"),
        dev_tau_b=tau_b(*scores.values()) if len(scores) == 2 else None,
        target_holdout_router=router(p, tasks, cells),
    )
    write(args.study / "selection.json", record, frozen=True)
    print(f"A = {a}    B = {b}    dev tau-b = {record['dev_tau_b']}")


def paired_stats(p, cells, a, b, bootstrap_samples=10000):
    per_seed = []
    effects = []
    for s in p["seeds"]:
        a_count = sum(cells[i, s, a]["y"] for i in p["holdout"])
        b_count = sum(cells[i, s, b]["y"] for i in p["holdout"])
        per_seed.append(dict(seed=s, A=a_count, B=b_count, delta=b_count - a_count))
    for i in p["holdout"]:
        effects.append(
            statistics.mean(
                cells[i, s, b]["y"] - cells[i, s, a]["y"] for s in p["seeds"]
            )
        )
    n = len(effects)
    rng = random.Random(12761)
    draws = sorted(sum(rng.choices(effects, k=n)) for _ in range(bootstrap_samples))
    lower = draws[math.floor(0.025 * (len(draws) - 1))]
    upper = draws[math.ceil(0.975 * (len(draws) - 1))]
    delta = sum(effects)
    return dict(
        N=n,
        pairs=n * len(p["seeds"]),
        per_seed=per_seed,
        net_issues=delta,
        ci_issues=[lower, upper],
        delta_pp=100 * delta / n,
        ci_pp=[100 * lower / n, 100 * upper / n],
        bootstrap_draws=bootstrap_samples,
    )


def example(p, cells, a, b):
    # Prefer any discordant label without favoring a particular winner; break ties by ID/seed.
    i, s = min(
        itertools.product(p["holdout"], p["seeds"]),
        key=lambda t: (
            cells[t[0], t[1], a]["y"] == cells[t[0], t[1], b]["y"],
            t[0],
            t[1],
        ),
    )

    def side(c):
        x = cells[i, s, c]
        directory = Path(x["dir"])
        return {
            "config": c,
            "resolved": bool(x["y"]),
            "base_commit": x["r"]["base_commit"],
            "prediction": str(directory / "prediction.json"),
            "verdict": str(directory / "verdict.json"),
            "trace": str(directory / "trace.jsonl"),
            "origin": x["v"].get("origin"),
        }

    case = dict(instance_id=i, seed=s, A=side(a), B=side(b))
    for record in (case["A"], case["B"]):
        trace = Path(record["trace"])
        if sha(trace) != read(trace.with_name("result.json")).get("trace_sha256"):
            raise ValueError(f"Case trace digest mismatch: {trace}")
    return case


def pilot_report(args, p, tasks):
    # Reuse exactly the same clustered estimator on the pre-registered development subset.
    a, b = p["pilot_configs"]
    model = p["baseline"]
    cells = collect(args, p, tasks, model, "pilot", [a, b])
    audit(args, model, "pilot", cells)
    identities(cells)
    cohort = {**p, "holdout": p["pilot"]}
    case = example(cohort, cells, a, b)
    case["issue_text"] = tasks[case["instance_id"]]["problem_statement"]
    output = dict(
        model=model,
        revision=p["revisions"][model],
        split="development pilot",
        A=a,
        B=b,
        summary=rates(cells, [a, b]),
        case=case,
        **paired_stats(cohort, cells, a, b),
    )
    write(args.study / "pilot.json", output)
    print(
        f"{model} development pilot: {output['N']} issues, two seeds. See {args.study}/pilot.json"
    )


def report(args, p, tasks):
    sel = locked(args)
    m = p["target"]
    a, b = sel["A"], sel["B"]
    cells = collect(args, p, tasks, m, "holdout", list(dict.fromkeys((a, b))))
    audit(args, m, "paired", cells)
    if identities(cells)[0] != sel["protocol_sha256"]:
        raise ValueError("Runner protocol changed between dev and holdout")
    comp = paired_stats(p, cells, a, b)
    case = example(p, cells, a, b)
    case["issue_text"] = tasks[case["instance_id"]]["problem_statement"]
    write(args.study / "paired.json", dict(A=a, B=b, model=m, **comp, case=case))
    print(
        f"Slide 5: B − A = {comp['net_issues']:+g} issues out of {comp['N']}, 95% CI {comp['ci_issues']}"
    )
    print(f"Slide 4: {case['instance_id']} seed {case['seed']} (see paired.json).")
    if not args.full:
        return
    all_cells, _ = grid_output(args, p, tasks, m, "holdout")
    policies = {"best_fixed": [], "router": [], "oracle": []}
    for i, s in itertools.product(p["holdout"], p["seeds"]):
        policies["best_fixed"].append(all_cells[i, s, b])
        policies["router"].append(all_cells[i, s, sel["target_holdout_router"][i]])
        policies["oracle"].append(
            max((all_cells[i, s, c] for c in p["configs"]), key=lambda x: x["y"])
        )
    policy_summary = {k: summarize(v) for k, v in policies.items() if k != "oracle"}
    oracle_solves = sum(x["y"] for x in policies["oracle"])
    total_cost = summed_cost(all_cells.values(), ["prompt_tokens", "completion_tokens"])
    seconds = summed_cost(all_cells.values(), ["elapsed_seconds"])
    policy_summary["oracle"] = dict(
        solved=oracle_solves,
        attempts_per_issue_seed=len(p["configs"]),
        rate=oracle_solves / len(policies["oracle"]),
        all_candidate_tokens=total_cost,
        seconds_per_solve=seconds / oracle_solves
        if seconds is not None and oracle_solves
        else None,
        tokens_per_solve=(
            total_cost / oracle_solves
            if oracle_solves and total_cost is not None
            else None
        ),
    )
    write(args.study / "routing.json", policy_summary)
    write_heatmap(args, p, tasks, sel["protocol_sha256"])


def write_heatmap(args, p, tasks, protocol):
    models = list(dict.fromkeys([p["baseline"], p["target"]] + p["api_models"]))
    for split in ("dev", "holdout"):
        rows = []
        by_model = {}
        reference_images = None
        for m in models:
            cfg, seeds = model_grid(p, m)
            cells = collect(args, p, tasks, m, split)
            audit(args, m, split, cells)
            current, images = identities(cells)
            if current != protocol or (
                reference_images is not None and reference_images != images
            ):
                raise ValueError(f"Protocol/image mismatch for {m} {split}")
            reference_images = images
            by_model[m] = cells
            for c in cfg:
                items = rows_for_config(cells, c)
                rows.append(
                    [
                        m,
                        c,
                        split,
                        ",".join(map(str, seeds)),
                        sum(x["y"] for x in items),
                        len(items),
                    ]
                )
        with (args.study / f"heatmap-{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "model",
                    "config",
                    "split",
                    "seeds",
                    "solved",
                    "attempts",
                    "resolve_rate",
                ]
            )
            for row in rows:
                w.writerow(row + [row[-2] / row[-1]])
        correlations = []
        for m, n in itertools.combinations(models, 2):
            configs = sorted(set(model_grid(p, m)[0]) & set(model_grid(p, n)[0]))
            seeds = sorted(set(model_grid(p, m)[1]) & set(model_grid(p, n)[1]))

            def matched_scores(model):
                return {
                    c: sum(
                        by_model[model][i, s, c]["y"]
                        for i, s in itertools.product(p[split], seeds)
                    )
                    for c in configs
                }

            correlations.append(
                dict(
                    models=[m, n],
                    configs=configs,
                    seeds=seeds,
                    tau_b=tau_b(matched_scores(m), matched_scores(n)),
                )
            )
        write(args.study / f"rank-{split}.json", correlations)


def plot(args, p, tasks):
    """Draw only verified cells. Dependencies are optional on the analysis host."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.split == "holdout":
        locked(args)
    selected = args.model or list(dict.fromkeys((p["baseline"], p["target"])))
    grids = [model_grid(p, m) for m in selected]
    configs = sorted(set.intersection(*(set(cfg) for cfg, _ in grids)))
    seeds = sorted(set.intersection(*(set(seeds) for _, seeds in grids)))
    matrix = []
    first_identity = None
    for m in selected:
        cells = collect(args, p, tasks, m, args.split)
        audit(args, m, args.split, cells)
        fingerprint = identities(cells)
        if first_identity is not None and fingerprint != first_identity:
            raise ValueError("Model grids use different protocols or images")
        first_identity = fingerprint
        row = [
            sum(cells[i, s, c]["y"] for i, s in itertools.product(p[args.split], seeds))
            for c in configs
        ]
        matrix.append([v / (len(p[args.split]) * len(seeds)) for v in row])
    fig, ax = plt.subplots(
        figsize=(max(8, 0.65 * len(configs)), len(selected) * 0.65 + 2.3)
    )
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(configs)), labels=configs, rotation=45, ha="right")
    ax.set_yticks(range(len(selected)), labels=selected)
    for r, row in enumerate(matrix):
        for c, rate in enumerate(row):
            ax.text(
                c,
                r,
                f"{rate:.0%}",
                ha="center",
                va="center",
                color="white" if rate > 0.55 else "black",
            )
    ax.set_title(
        f"Official resolve rate · {args.split} · {len(p[args.split])} issues · seed(s) {', '.join(map(str, seeds))}"
    )
    ax.set_xlabel("Harness configuration")
    fig.colorbar(image, ax=ax, shrink=0.85, label="Resolve rate")
    fig.tight_layout()
    dest = args.study / f"heatmap-{args.split}-{'-'.join(selected)}.png"
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(dest)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    for name in ("init", "run", "grid", "select", "report", "plot", "pilot-report"):
        cmd = subs.add_parser(name)
        cmd.add_argument("--study", type=Path, default=Path("study"))
        cmd.add_argument("--tasks", type=Path, required=True)
        if name != "init":
            cmd.add_argument("--runs", type=Path, default=Path("runs"))
        if name == "init":
            cmd.add_argument(
                "--revision", action="append", default=[], metavar="MODEL=REVISION"
            )
            cmd.add_argument("--grader-timeout", type=int, default=1800)
            cmd.add_argument("--baseline", default="qwen9b")
            cmd.add_argument("--target", default="qwen27b")
            cmd.add_argument("--single-model", action="store_true",
                             help="Hold out a comparison of 00000 versus the baseline model's dev-selected configuration.")
            cmd.add_argument("--api-model", action="append", default=[])
            cmd.add_argument("--seeds", nargs=2, type=int, default=[1, 2])
            cmd.add_argument("--dev-size", type=int, default=30)
            cmd.add_argument("--split-seed", type=int, default=7607)
            exposure = cmd.add_mutually_exclusive_group(required=True)
            exposure.add_argument(
                "--seen-ids",
                type=Path,
                help="All IDs examined during prior tuning or pilot; one per line",
            )
            exposure.add_argument(
                "--no-seen-ids",
                action="store_true",
                help="Use only when no cohort issue informed tuning",
            )
            for k, v in BUDGET.items():
                cmd.add_argument(
                    "--" + k.replace("_", "-"), dest=k, type=type(v), default=v
                )
        if name in ("run", "grid"):
            cmd.add_argument("--model", required=True)
            cmd.add_argument("--split", choices=["dev", "holdout"], default="dev")
        if name == "plot":
            cmd.add_argument("--model", action="append", default=[])
            cmd.add_argument("--split", choices=["dev", "holdout"], default="dev")
        if name == "run":
            cmd.add_argument(
                "--pilot",
                action="store_true",
                help="Run pre-registered 9B development pilot",
            )
            cmd.add_argument(
                "--jobs",
                type=int,
                default=1,
                help="Maximum simultaneous instance containers and vLLM requests",
            )
            cmd.add_argument(
                "--base-url",
                help="Model server URL; same model name must resolve to the same checkpoint",
            )
            cmd.add_argument("--limit", type=int, help="Run first K planned slots")
            cmd.add_argument("--paired-only", action="store_true")
        if name == "report":
            cmd.add_argument(
                "--full",
                action="store_true",
                help="Also require all registered grids; write router/oracle and heatmap tables",
            )
    return parser.parse_args()


def main():
    args = cli()
    if args.cmd == "init":
        return init(args)
    p = plan_for(args)
    if args.cmd == "run":
        return run(args, p)
    tasks = task_rows(args.tasks)
    if args.cmd == "grid":
        if args.split == "holdout":
            locked(args)
        return grid_output(args, p, tasks, args.model, args.split)
    if args.cmd == "pilot-report":
        return pilot_report(args, p, tasks)
    if args.cmd == "plot":
        return plot(args, p, tasks)
    if args.cmd == "select":
        return selection(args, p, tasks)
    return report(args, p, tasks)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
