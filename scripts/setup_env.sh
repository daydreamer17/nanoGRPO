#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-grpo-qwen05b}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk 'NR>2 {gsub(/\*/, "", $1); print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.12 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  trl==1.0.0 \
  vllm==0.17.1 \
  accelerate \
  datasets \
  peft \
  sentencepiece \
  wandb \
  --extra-index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import torch
import trl
import vllm
import wandb

print("torch", torch.__version__)
print("trl", trl.__version__)
print("vllm", vllm.__version__)
print("wandb", wandb.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY
