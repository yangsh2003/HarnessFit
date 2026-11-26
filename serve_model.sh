#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd)"
model_dir="${HFIT_MODEL_DIR:-$root/models/Qwen3.5-9B}"
model_dir="$(cd -- "$model_dir" && pwd -P)"
port="${HFIT_PORT:-8000}"
python - "$root" "$model_dir" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from prepare import model_files
from dataset_tools import hash_file
import torch

expected = {"vllm": "0.19.1", "transformers": "5.5.4", "torch": "2.10.0"}
for name, wanted in expected.items():
    actual = version(name)
    if actual.split("+", 1)[0] != wanted:
        sys.exit(f"{name} {actual} found; install {wanted} for this study.")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    sys.exit("An available CUDA GPU with BF16 support is required.")
print(f"GPU: {torch.cuda.get_device_name(0)}, torch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
for name, digest in model_files():
    if hash_file(Path(sys.argv[2]) / name) != digest:
        sys.exit(f"Wrong checksum for {name}. Run python prepare.py model.")
PY
exec python -m vllm.entrypoints.cli.main serve "$model_dir" \
    --host 127.0.0.1 --port "$port" --served-model-name qwen9b \
    --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 \
    --max-num-seqs 8 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization "${HFIT_GPU_UTIL:-0.38}" --enforce-eager \
    --language-model-only --reasoning-parser qwen3 --generation-config vllm \
    --distributed-executor-backend uni --mm-tensor-ipc direct_rpc
