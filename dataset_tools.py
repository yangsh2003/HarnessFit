#!/usr/bin/env python3
"""Prepare the frozen Mini test split, load its per-issue images and grade predictions."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "work/harnessfit/data/swe-bench-verified-mini"
REVISION = "b316c349947c29963fce3f4a65967c9807a4b673"
REPO = "MariusHobbhahn/swe-bench-verified-mini"
PUBLIC = ("instance_id", "repo", "base_commit", "problem_statement", "version")
ROW_HASH = {
    "tasks.jsonl": "7e704bc2d1c9206b6167ecff192ca8ffc67c43a0026317a0dc4703f457373dd3",
    "evaluation.jsonl": "502b9406977574f0c759180e265cafddafc2ea48d23c9da3c03a4b952f6d6674",
}
PIN_HASH = "e95c88dacf70e95c5b35c848a4c5eca2b24ec6819b5c2278bb02c8713b7e1ba2"
CONFIG_HASH = "ce8d1354de0c9268fdd57ef9a288ddc51c968b95963ccf754f049b63d9393acb"
JUDGE = "4.1.0"


def hash_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def checked_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return json.loads(data)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
    os.replace(stream.name, path)


def as_json(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()


def prepare(args):
    # pyarrow is only required when rebuilding the transferred JSONL pair.
    import pyarrow.parquet as parquet
    rows = parquet.read_table(args.parquet).to_pylist()
    selected = [{k: r[k] for k in PUBLIC} for r in rows]
    files = {}
    for name, values in [("tasks.jsonl", selected), ("evaluation.jsonl", rows)]:
        blob = "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values).encode()
        if hashlib.sha256(blob).hexdigest() != ROW_HASH[name]:
            raise ValueError(f"The Parquet input does not reproduce the frozen {name}")
        files[name] = blob
    manifest = dict(repo_id=REPO, revision=REVISION, split="test", count=50,
                    public_fields=list(PUBLIC), sha256=ROW_HASH)
    for name, data in files.items():
        save(args.data / name, data)
    save(args.data / "manifest.json", as_json(manifest))
    print(f"Prepared 50 tasks in {args.data}")


def inventory(data):
    meta = json.loads((data / "manifest.json").read_bytes())
    if (meta.get("repo_id"), meta.get("revision"), meta.get("split"), meta.get("count"),
        meta.get("sha256")) != (REPO, REVISION, "test", 50, ROW_HASH):
        raise ValueError(f"Unexpected dataset manifest: {data}")
    for name, digest in ROW_HASH.items():
        if hash_file(data / name) != digest:
            raise ValueError(f"Checksum mismatch: {data / name}")
    ids = [json.loads(line)["instance_id"] for line in (data / "tasks.jsonl").read_text().splitlines()]
    local_lock = data / "environments/images.lock.json"
    pins = checked_json(local_lock if local_lock.exists() else HERE / "images.lock.json", PIN_HASH)
    configs = checked_json(HERE / "image_configs.json", CONFIG_HASH)
    if not len(set(ids)) == len(ids) == 50 or set(pins) != set(ids) or pins.keys() != configs.keys():
        raise ValueError("Dataset/image coverage mismatch")
    return pins, configs


def select_ids(args, pins):
    if args.ids_file:
        ids = args.ids_file.read_text().split()
        if len(ids) != len(set(ids)) or not ids or not set(ids) <= pins.keys():
            raise ValueError("--ids-file must contain unique Mini IDs separated by whitespace")
        return sorted(ids)
    return sorted(pins)


def blob_path(archive, digest):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"Malformed digest: {digest}")
    return archive / digest[7:]


def parts(archive, pin, config):
    digest = pin.split("@")[1]
    manifest = checked_json(blob_path(archive, digest), digest[7:])
    if manifest["config"]["digest"] != config:
        raise ValueError(f"Pinned config mismatch: {pin}")
    parts_list = [manifest["config"], *manifest["layers"]]
    for part in parts_list:
        path = blob_path(archive, part["digest"])
        if path.stat().st_size != part["size"]:
            raise ValueError(f"Truncated blob: {path}")
    return parts_list


def check(args):
    pins, configs = inventory(args.data)
    ids = select_ids(args, pins)
    missing = []
    archive = args.data / "environments/blobs/sha256"
    seen = set()
    for issue in ids:
        try:
            for part in parts(archive, pins[issue], configs[issue]):
                digest = part["digest"]
                if args.deep and digest not in seen and hash_file(blob_path(archive, digest)) != digest[7:]:
                    raise ValueError(f"Blob checksum mismatch: {issue} {digest}")
                seen.add(digest)
        except FileNotFoundError:
            missing.append(issue)
    print(f"Frozen public/evaluation rows: 50/50. Selected: {len(ids)}. "
          f"Local complete image manifests/layers: {len(ids) - len(missing)}/{len(ids)}.")
    if missing:
        print("Need pinned registry pull or copy environments/:", " ".join(missing))


def tag(pin):
    return pin.removeprefix("docker.io/").split("@")[0] + ":latest"


def docker_id(ref):
    check_docker = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", ref],
                                  text=True, capture_output=True)
    if check_docker.returncode:
        if "no such" not in check_docker.stderr.lower():
            raise RuntimeError(check_docker.stderr.strip() or "docker image inspect failed")
        return None
    return check_docker.stdout.strip()


def write_docker_archive(stream, archive, entries):
    # Each manifest records the full ordered layer chain. Shared files appear once.
    manifest = []
    files = {}
    for ref, descriptors in entries:
        config, *layers = [blob_path(archive, d["digest"]) for d in descriptors]
        config_name = config.name + ".json"
        files[config_name] = config
        layer_names = []
        for layer in layers:
            name = layer.name + ".tar.gz"
            layer_names.append(name)
            files[name] = layer
        manifest.append({"Config": config_name, "RepoTags": [ref], "Layers": layer_names})

    record = json.dumps(manifest).encode()
    with tarfile.open(fileobj=stream, mode="w|", bufsize=1024 * 1024, dereference=True) as tar:
        info = tarfile.TarInfo("manifest.json")
        info.size = len(record)
        tar.addfile(info, io.BytesIO(record))
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)


def docker_load(archive, entries):
    print(f"Streaming {len(entries)} local image(s) into Docker", flush=True)
    with subprocess.Popen(["docker", "load"], stdin=subprocess.PIPE) as docker:
        try:
            with docker.stdin:
                write_docker_archive(docker.stdin, archive, entries)
        except BrokenPipeError as exc:
            raise RuntimeError(f"docker load closed its input early (exit {docker.wait()})") from exc
        if docker.wait():
            raise RuntimeError("docker load failed")


def local_parts(args, archive, pin, config, issue):
    announced = False
    while True:
        try:
            return parts(archive, pin, config)
        except FileNotFoundError:
            if not args.wait_local:
                if args.pull_missing:
                    return None
                raise ValueError(f"Missing local blobs for {issue}. Supply --pull-missing to fetch the pinned image.")
            if not announced:
                print(f"Waiting for local image objects: {issue}", flush=True)
                announced = True
            time.sleep(30)


def verify_tag(ref, config):
    if docker_id(ref) != config:
        raise ValueError(f"Docker image {ref} differs from the frozen config")
    print(f"Ready {ref}", flush=True)


def images(args):
    pins, configs = inventory(args.data)
    ids = select_ids(args, pins)
    archive = args.data / "environments/blobs/sha256"
    checked = set()
    pending = []
    subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], check=True)
    for issue in ids:
        pin, config = pins[issue], configs[issue]
        ref = tag(pin)
        if docker_id(ref) == config:
            print(f"Ready {ref}", flush=True)
            continue
        descriptors = local_parts(args, archive, pin, config, issue)
        if descriptors is None:
            subprocess.run(["docker", "pull", "--platform", "linux/amd64", pin], check=True)
            subprocess.run(["docker", "tag", pin, ref], check=True)
        else:
            for d in descriptors:
                digest = d["digest"]
                if digest not in checked and hash_file(blob_path(archive, digest)) != digest[7:]:
                    raise ValueError(f"Local blob checksum mismatch: {issue} {digest}")
                checked.add(digest)
            if args.batch:
                pending.append((ref, descriptors))
                continue
            docker_load(archive, [(ref, descriptors)])
        verify_tag(ref, config)
    if pending:
        docker_load(archive, pending)
        for ref, descriptors in pending:
            verify_tag(ref, descriptors[0]["digest"])


def get_slots(group, allowed):
    slots = []
    for path in sorted(group.glob("*/prediction.json")):
        issue = path.parent.name
        if issue not in allowed:
            continue
        result = json.loads((path.parent / "result.json").read_bytes())
        if result.get("status") == "infra_error":
            print(f"Skipped runner error: {path.parent}", file=sys.stderr)
            continue
        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        if result.get("instance_id") != issue or result.get("prediction_sha256") != fingerprint:
            raise ValueError(f"Result metadata does not match prediction: {path}")
        pred = json.loads(raw)
        if pred.get("instance_id") != issue or not isinstance(pred.get("model_patch"), str):
            raise ValueError(f"Invalid prediction: {path}")
        model = pred.get("model_name_or_path", "")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", model) or any(p in ("", ".", "..") for p in model.split("/")):
            raise ValueError(f"Invalid model_name_or_path: {path}")
        slots.append((path, pred, fingerprint))
    if len({pred["model_name_or_path"] for _, pred, _ in slots}) > 1:
        raise ValueError(f"Multiple model IDs within the same model/config/seed group: {group}")
    return slots


def already_graded(slot, pin, timeout):
    path, pred, fingerprint = slot
    verdict_path = path.parent / "verdict.json"
    if not verdict_path.is_file():
        return False
    verdict = json.loads(verdict_path.read_bytes())
    if (verdict.get("prediction_sha256"), verdict.get("image"), verdict.get("eval_sha256"),
        verdict.get("grader_version"), verdict.get("grader_timeout")) != (
            fingerprint, pin, ROW_HASH["evaluation.jsonl"], JUDGE, timeout):
        return False
    report_path = path.parent / verdict.get("report", "")
    category = {"resolved": "resolved_ids", "unresolved": "unresolved_ids",
                "empty_patch": "empty_patch_ids"}.get(verdict.get("status"))
    if not category or not report_path.is_file():
        return False
    report = json.loads(report_path.read_bytes())
    return (verdict.get("instance_id") == pred["instance_id"]
            and verdict.get("resolved") is (category == "resolved_ids")
            and pred["instance_id"] in report[category])


def grade_group(args, group, pins, configs, allowed):
    slots = get_slots(group, allowed)
    slots = [s for s in slots if not already_graded(s, pins[s[1]["instance_id"]], args.timeout)]
    if not slots:
        print(f"{group}: no ungraded completed predictions for the selected IDs.")
        return 0
    for _, pred, _ in slots:
        issue = pred["instance_id"]
        if pred["model_patch"] and docker_id(tag(pins[issue])) != configs[issue]:
            raise ValueError(f"Prepare pinned image first: {issue}. Run the images subcommand.")
    # Different content gets a different official cache. A retry of the same content
    # reuses completed reports; missing per-instance reports are rerun by SWE-bench.
    inputs = [str(group), JUDGE, ROW_HASH, PIN_HASH, args.timeout, [s[2] for s in slots]]
    key = "hf-" + hashlib.sha256(as_json(inputs)).hexdigest()[:20]
    work = group / ".official" / key
    pred_file = work / "predictions.jsonl"
    save(pred_file, "".join(json.dumps(pred) + "\n" for _, pred, _ in slots).encode())
    run_official(args.data / "evaluation.jsonl", pred_file, [s[1]["instance_id"] for s in slots],
                 key, work, args.workers, args.timeout)
    model = slots[0][1]["model_name_or_path"].replace("/", "__")
    report_file = work / f"{model}.{key}.json"
    report = json.loads(report_file.read_bytes())
    if set(report["submitted_ids"]) != {s[1]["instance_id"] for s in slots}:
        raise ValueError(f"Official report does not cover the submitted group: {report_file}")
    errors = 0
    for path, pred, fingerprint in slots:
        issue = pred["instance_id"]
        if issue in report["resolved_ids"]:
            status, resolved = "resolved", True
        elif issue in report["unresolved_ids"]:
            status, resolved = "unresolved", False
        elif issue in report["empty_patch_ids"]:
            status, resolved = "empty_patch", False
        else:
            status, resolved = "evaluation_error", None
            errors += 1
        verdict = dict(instance_id=issue, resolved=resolved, status=status,
                       prediction_sha256=fingerprint, origin=f"swebench=={JUDGE}",
                       report=os.path.relpath(report_file, path.parent), image=pins[issue],
                       eval_sha256=ROW_HASH["evaluation.jsonl"], grader_timeout=args.timeout,
                       grader_version=JUDGE)
        if hash_file(path) != fingerprint:
            raise RuntimeError(f"Prediction changed while grading: {path}")
        save(path.parent / "verdict.json", as_json(verdict))
    print(f"{group}: {len(report['resolved_ids'])}/{len(slots)} resolved; {errors} evaluator errors.")
    return errors


def run_official(dataset, predictions, ids, key, cwd, workers, timeout):
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name",
           str(dataset), "--predictions_path", str(predictions), "--instance_ids", *ids,
           "--run_id", key, "--namespace", "swebench", "--cache_level", "instance",
           "--max_workers", str(workers), "--timeout", str(timeout)]
    subprocess.run(cmd, cwd=cwd, check=True)


def require_judge():
    if platform.machine().lower() not in {"x86_64", "amd64"} or platform.system() != "Linux":
        raise ValueError("Use a Linux x86_64 grading host matching the frozen image set.")
    if importlib.metadata.version("swebench") != JUDGE:
        raise ValueError(f"Install swebench=={JUDGE} in the grading environment")


def control(args):
    require_judge()
    pins, configs = inventory(args.data)
    issue = args.issue
    if issue not in (HERE / "known_seen.txt").read_text().split() or issue not in pins:
        raise ValueError("Use an issue already listed in known_seen.txt for the gold control")
    if docker_id(tag(pins[issue])) != configs[issue]:
        raise ValueError(f"Prepare the pinned image for {issue} first with the images command")
    rows = (args.data / "evaluation.jsonl").read_bytes().splitlines()
    row = next(line for line in rows if json.loads(line)["instance_id"] == issue)
    controls = args.output.resolve()
    controls.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="gold-", dir=controls))
    dataset = work / "control-dataset.jsonl"
    save(dataset, row + b"\n")
    run_official(dataset, "gold", [issue], work.name, work, 1, args.timeout)
    report_file = work / f"gold.{work.name}.json"
    report = json.loads(report_file.read_bytes())
    if report.get("resolved_ids") != [issue] or report.get("total_instances") != 1:
        raise RuntimeError(f"Gold control did not resolve; inspect {work} before scoring model runs.")
    print(f"Gold control resolved {issue}. Official report: {report_file}")


def grade(args):
    require_judge()
    pins, configs = inventory(args.data)
    allowed = set(select_ids(args, pins))
    groups = [args.group] if args.group else sorted({p.parent.parent for p in args.runs.glob("*/*/*/*/prediction.json")})
    if not groups or any(not g.is_dir() for g in groups):
        raise ValueError("No matching run group folders")
    errors = 0
    for g in groups:
        group = g.resolve()
        with (group / ".grade.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            errors += grade_group(args, group, pins, configs, allowed)
    if errors:
        raise RuntimeError(f"{errors} cases lack a definitive verdict. See the official logs and rerun grade.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for fun in [prepare, check, images, control, grade]:
        p = sub.add_parser(fun.__name__)
        p.set_defaults(func=fun)
        p.add_argument("--data", type=Path, default=DATA)
        if fun is prepare:
            p.add_argument("--parquet", type=Path, required=True)
        elif fun is not control:
            p.add_argument("--ids-file", type=Path)
        if fun is check:
            p.add_argument("--deep", action="store_true", help="Hash all shared image layers once")
        if fun is images:
            p.add_argument("--batch", action="store_true",
                           help="Import all missing local images in one Docker archive with shared layers stored once")
            source = p.add_mutually_exclusive_group()
            source.add_argument("--pull-missing", action="store_true")
            source.add_argument("--wait-local", action="store_true",
                                help="Wait for missing objects while a separate offline transfer publishes verified files")
        if fun is control:
            p.add_argument("--issue", default="django__django-12155")
            p.add_argument("--output", type=Path, default=Path("controls"))
        if fun in [control, grade]:
            p.add_argument("--timeout", type=int, default=1800)
        if fun is grade:
            arg = p.add_mutually_exclusive_group(required=True)
            arg.add_argument("--group", type=Path)
            arg.add_argument("--runs", type=Path)
            p.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    if min(getattr(args, "workers", 1), getattr(args, "timeout", 1800)) <= 0:
        parser.error("workers and timeout must be positive")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        sys.exit(str(error))
