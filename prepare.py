#!/usr/bin/env python3
"""Prepare the exact model, public tasks and Docker images used by the 9B study."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import dataset_tools as dataset

ROOT = Path(__file__).resolve().parent
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MS_REVISION = "e8885939589b1e032d291a1ceefd21b775e6277e"
MODEL = "Qwen/Qwen3.5-9B"
PARQUET = "data/test-00000-of-00001.parquet"
PARQUET_SHA = "f9ba19dea78884f1081355d2d8afb671899981f24180aa0c4c1aa14d2c23e855"
MODEL_MANIFEST_SHA = "37048cc496c8992ea778fc1395f10b3c1d2dcb434f5de066f9f5c4bbf832903a"


def model_files():
    manifest = (ROOT / "HFIT_SHA256SUMS").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MODEL_MANIFEST_SHA:
        raise ValueError("Wrong model checksum manifest")
    for line in manifest.decode().splitlines():
        digest, name = line.split()
        yield name, digest


def urls(resource, modelscope=None):
    # Honor the selected endpoint first. Each source must produce identical bytes.
    hosts = [os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/"),
             "https://hf-mirror.com", "https://huggingface.co"]
    sources = [f"{h}/{resource}" for h in dict.fromkeys(hosts)]
    if modelscope:
        sources.insert(1, modelscope)
    return list(dict.fromkeys(sources))


def download(path, expected, sources):
    if path.is_file() and dataset.hash_file(path) == expected:
        print(f"Verified {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for source in sources:
        print(f"Fetching {source}", flush=True)
        args = ["curl", "-q", "-fL", "--silent", "--show-error", "--connect-timeout", "12",
                "--speed-limit", "2048", "--speed-time", "60", "--retry", "2"]
        # A server without Range support gets one chance from byte zero.
        resume = subprocess.run([*args, "-C", "-", "-o", str(partial), source])
        status = resume.returncode
        if status == 33:
            status = subprocess.run([*args, "-o", str(partial), source]).returncode
        if status:
            continue
        if dataset.hash_file(partial) == expected:
            os.replace(partial, path)
            print(f"Verified {path.name}", flush=True)
            return
        partial.unlink()
        print(f"Wrong checksum from {source}, trying the next source.", file=sys.stderr)
    raise RuntimeError(f"Could not download {path.name} at its frozen SHA-256. "
                       "Retry or set HF_ENDPOINT=https://hf-mirror.com.")


def model(args):
    for name, expected in model_files():
        source = f"{MODEL}/resolve/{REVISION}/{name}"
        # ModelScope stores the same four weight blobs at a separate immutable commit.
        ms = f"https://modelscope.cn/models/{MODEL}/resolve/{MS_REVISION}/{name}" if name.endswith(".safetensors") else None
        download(args.model_dir / name, expected, urls(source, ms))


def data(args):
    try:
        dataset.inventory(args.data)
    except (FileNotFoundError, ValueError):
        args.parquet = args.data / "source" / PARQUET
        route = f"datasets/{dataset.REPO}/resolve/{dataset.REVISION}/{PARQUET}"
        download(args.parquet, PARQUET_SHA, urls(route))
        dataset.prepare(args)
    dataset.inventory(args.data)
    print(f"Verified 50 tasks and grading rows in {args.data}")


def images(args):
    args.ids_file = ROOT / "study_9b_baai" / f"{args.split}.ids" if args.split != "all" else None
    mirror = os.getenv("HFIT_DOCKER_MIRROR", "").removeprefix("https://").rstrip("/")
    if mirror:
        pins, configs = dataset.inventory(args.data)
        for issue in dataset.select_ids(args, pins):
            ref, config = dataset.tag(pins[issue]), configs[issue]
            if dataset.docker_id(ref) == config:
                continue
            try:
                dataset.parts(args.data / "environments/blobs/sha256", pins[issue], config)
                continue  # The existing local blob cache is verified and loaded below.
            except FileNotFoundError:
                pass
            alternative = mirror + "/" + pins[issue].removeprefix("docker.io/")
            if subprocess.run(["docker", "pull", "--platform", "linux/amd64", alternative]).returncode:
                print(f"Mirror unavailable for {issue}, trying the original registry.", file=sys.stderr)
                continue
            if dataset.docker_id(alternative) != config:
                raise ValueError(f"Wrong image config for {issue} from {mirror}")
            subprocess.run(["docker", "tag", alternative, ref], check=True)
    args.batch, args.pull_missing, args.wait_local = True, True, False
    dataset.images(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("model", help="Download and verify Qwen3.5-9B")
    p.add_argument("--model-dir", type=Path, default=os.getenv("HFIT_MODEL_DIR", ROOT / "models/Qwen3.5-9B"))
    for command in ("data", "images"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=os.getenv("HFIT_DATA", ROOT / "data/swe-bench-verified-mini"))
        if command == "images":
            p.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    args = parser.parse_args()
    destination = args.model_dir if args.action == "model" else args.data
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".prepare.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        {"model": model, "data": data, "images": images}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Preparation failed: {exc}")
