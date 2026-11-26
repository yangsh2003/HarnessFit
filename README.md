# HarnessFit · Qwen3.5-9B 代码修复实验

本仓库用于三人并行完成 SWE-bench Verified Mini。50 道题分为 30 道开发题和 20 道留出题。每道题运行 16 种 harness 配置，每种使用 seed 1 和 2，合计 1,600 次。模型采用冻结权重进行推理。

**当前主实验使用 Qwen3.5-9B 的 thinking 模式。** 所有组的模型请求和 token 计数都显式使用 `enable_thinking=True`。每轮最多 2,048 个输出 token，思考和最终动作共用这份预算。工具只执行最终动作；轨迹记录每轮有没有产生思考、停止原因和可用的 token 计数。初始的 non-thinking 实验对应 Git 标签 `non-thinking-2026-11-26`，单独归档。

**B、C 同学的入口是 `python run_group.py`。** 首次参与本轮实验时，使用新的仓库目录或 Git worktree，复用已有权重、数据和模型服务。新的运行从零计数，每组先跑 300 个开发槽。旧版生成的 `progress.json`、`selection.json` 和预测都属于上一轮。

| 组别 | 负责配置（从左至右保留五位） | 开发集总量 | 留出集总量 |
|---|---|---:|---:|
| B | `00110` `01001` `10001` `10100` `11011` | 300 | 200 |
| C | `01100` `01111` `10010` `10111` `11000` | 300 | 200 |
| A（原机） | `00000` `00011` `00101` `01010` `11101` `11110` | 360 | 240 |

表中是总配额，实际待跑数由进度清单决定。五个位依次控制上下文压缩、显式规划、提交前测试验收、结构化文件工具和停滞后重试。`1` 表示开启。

## 1. 准备运行机

使用 Linux x86_64、可运行 Docker 容器的账号，以及支持 BF16、能容纳 9B 模型和 32,768 token 上下文的 NVIDIA GPU。原机使用单张 H100。四个权重分片合计约 18 GiB，逐题镜像的共享压缩层约 10 GB，Docker 解压、推理缓存及运行记录另外占用空间。

在 GPU 机器上克隆仓库并进入根目录。后文的命令都从这里执行。已有与原机相同的环境时，激活后从第 2 节开始。

```bash
git clone https://github.com/yangsh2003/HarnessFit.git HarnessFit-thinking
cd HarnessFit-thinking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install vllm==0.19.1 transformers==5.5.4 pyarrow==21.0.0
python -c 'import torch, vllm, vllm._C; print(vllm.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
docker version
```

原机的 PyTorch 为 `2.10.0+cu128`。上面的检查应能加载 vLLM 扩展，并显示 GPU 和 BF16 均为 `True`。驱动或平台与预编译包不兼容时，使用原机环境，或按对应版本的 vLLM 安装文档构建。开发集和留出集沿用同一套软件环境。[vLLM 0.19.1 GPU 安装](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)

## 2. 下载模型、数据集和逐题镜像

三项资源分别准备。`model` 下载模型并校验文件内容。`data` 下载冻结版本的 Parquet，生成用来推理的 `tasks.jsonl` 和供集中评分的 `evaluation.jsonl`。`images` 为每道题准备能实际检出代码和运行测试的 Docker 镜像。

```bash
python prepare.py model
python prepare.py data
python prepare.py images --split dev
```

从中国内地拉取 Hugging Face 资源较慢时，在下载前执行 `export HF_ENDPOINT=https://hf-mirror.com`。模型权重也可自动尝试固定版本的 ModelScope 镜像，所有源使用相同的 SHA-256 校验。Docker Hub 较慢时，可用 `HFIT_DOCKER_MIRROR=docker.1panel.live python prepare.py images --split dev` 尝试按同一 digest 拉取。已有 BAAI 共享缓存时，可以把数据目录中的整个 `environments/` 连同 `blobs/sha256/` 一起复制过来，再执行 `images` 从本地导入和校验。

| 本地目录 | 应有内容 | 上游版本 |
|---|---|---|
| `models/Qwen3.5-9B/` | 配置、tokenizer、分片索引、4 个 BF16 权重分片 | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `data/swe-bench-verified-mini/` | `tasks.jsonl`、`evaluation.jsonl`、`manifest.json`，可选 `environments/` | `b316c349947c29963fce3f4a65967c9807a4b673` |

已有大文件可以直接复用。如存放位置不同，在每个终端导出 `HFIT_MODEL_DIR` 和 `HFIT_DATA`，分别设为上表对应目录的绝对路径。镜像按仓库中的 `images.lock.json` 和 `image_configs.json` 验证。推理程序只读取公开题面。[Qwen3.5-9B 固定版本](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) · [SWE-bench Verified Mini 固定版本](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini/tree/b316c349947c29963fce3f4a65967c9807a4b673)

## 3. 启动模型并选择 B 或 C

先在 GPU 机器的一个终端启动模型服务。这里以分配到 GPU 0 为例，把编号换成自己的 GPU。保持这个进程运行。

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash serve_model.sh
```

服务沿用原 H100 实验预留 38% 显存的设置。在独占、容量较小的卡上，如 vLLM 提示可用 KV cache 无法支持 32,768 token，可增加 `HFIT_GPU_UTIL` 分配比例再启动，同时保留原上下文窗口和推理预算。

在同一机器的另一个终端进入仓库，激活环境。第一条运行命令只显示待跑量，第二条正式运行。正式运行也可以直接调用一次。输入 `B` 或 `C` 后，程序使用该组配置和两个种子，逐项调用原机相同的 `run.py`。

```bash
source .venv/bin/activate
python run_group.py --dry-run
python run_group.py
```

默认使用本机 `http://127.0.0.1:8000/v1`。已启动的服务使用其他端口时，设置 `HFIT_BASE_URL` 为其完整 `/v1` 地址。模型服务名为 `qwen9b`。在长时间运行的机器上，可在 `tmux` 会话中启动以上两个进程。要固定组别并设置并发，例如 B 组同时跑 2 个槽，可以执行 `python run_group.py --group B --jobs 2`。

## 4. 交接、断点续跑与留出集

1. **确认本轮进度。** 初次从新仓库运行时使用随代码冻结的空清单。A 后续可从本轮中央结果导出 `progress.json` 发给各组，用 `--progress /实际路径/progress.json` 接续。先用 `--dry-run` 核对组别、阶段和任务数，再正式派单。
2. **中途断开或服务出错。** 恢复模型服务或 Docker 后，使用同一组、同一结果目录重新执行原命令。中央清单里已有预测的槽会跳过。本地已完成的槽校验后复用。空补丁也构成一次已完成的预测。
3. **回传开发集。** 默认输出在 `runs_thinking/B/qwen9b/` 或 `runs_thinking/C/qwen9b/`，下一级依次为五位配置、种子、题目 ID。把整个组目录私下交给 A，保留每个槽的 `prediction.json`、`trace.jsonl` 和 `result.json`。A 将各组新完成的槽按原相对路径合并到中央结果目录，有同名完成记录时先核对校验值。
4. **开发集选定后再跑留出集。** A 对全部 960 个开发槽集中评分并冻结 `selection.json`，B、C 将该文件放入各自的 `study_9b_baai/`。先执行 `python prepare.py images --split holdout`，再执行 `python run_group.py --split holdout` 并选自己的组。两组都保留各自的 5 种配置，每组跑 200 个留出槽，并按同样方式回传。

```bash
# A 在能读取中央结果和公开题面的机器上执行
python export_progress.py --runs /path/to/harnessfit_thinking/runs_9b_baai \
  --tasks /path/to/data/swe-bench-verified-mini/tasks.jsonl --output progress.json
```

使用本轮新目录时，A 可以运行 `python run_group.py --group A --progress progress.json --runs /path/to/harnessfit_thinking/runs_9b_baai --base-url http://127.0.0.1:18000/v1` 接续自己的 6 个配置。沿用原机的 Docker 环境、`HFIT_DATA` 和推理端点。`full_9b.sh` 会对全部 16 种配置派题，三机分工期间统一使用 `run_group.py`。

### A 的集中评分命令

A 在已有 SWE-bench 4.1.0 的 Linux x86_64 评分环境运行下列命令。`RUNS` 指向合并后的目录，其中直接包含 `qwen9b/`。等全部留出预测回传后再运行第二阶段。

```bash
DATA="${HFIT_DATA:-$PWD/data/swe-bench-verified-mini}"
RUNS=/path/to/harnessfit_thinking/runs_9b_baai
python dataset_tools.py grade --data "$DATA" --runs "$RUNS" --ids-file study_9b_baai/dev.ids
python study.py select --study study_9b_baai --tasks "$DATA/tasks.jsonl" --runs "$RUNS"

# 640 个留出槽齐全后
python dataset_tools.py grade --data "$DATA" --runs "$RUNS" --ids-file study_9b_baai/holdout.ids
python study.py report --study study_9b_baai --tasks "$DATA/tasks.jsonl" --runs "$RUNS" --full
```

开发任务的「生成完毕」表示已有可核验的预测。解决率取自官方评分报告。比较三组结果时同时记录 GPU 型号、vLLM 版本和实际并发度。[SWE-bench 4.1.0](https://pypi.org/project/swebench/4.1.0/)
