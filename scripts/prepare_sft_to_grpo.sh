#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/nano-grpo-qwen05b}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_SLICE="${TRAIN_SLICE:-train[:1000]}"
EVAL_SIZE="${EVAL_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SAVE_STEPS="${SAVE_STEPS:-20}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
OFFLINE="${OFFLINE:-true}"
GPU_ID="${GPU_ID:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-nano-grpo-qwen05b}"
RUN_NAME="${RUN_NAME:-sft-gsm8k-qwen25-05b-$(date -u +%Y%m%d-%H%M%S)}"

echo "[stage 1/2] training SFT adapter: ${RUN_NAME}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/train_sft_gsm8k.py" \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --train_slice "${TRAIN_SLICE}" \
  --eval_size "${EVAL_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --learning_rate "${LEARNING_RATE}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --offline "${OFFLINE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${RUN_NAME}"

RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"

echo "[stage 2/2] merging adapter into standalone model: ${RUN_DIR}/merged_model"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/merge_lora_adapter.py" \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_dir "${RUN_DIR}" \
  --offline "${OFFLINE}"

echo
echo "SFT run ready:"
echo "  run_dir=${RUN_DIR}"
echo "  final_adapter=${RUN_DIR}/final_adapter"
echo "  merged_model=${RUN_DIR}/merged_model"
