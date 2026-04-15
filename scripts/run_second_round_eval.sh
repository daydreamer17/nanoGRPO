#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT_DIR="${1:-/root/autodl-tmp/outputs/nano-grpo-qwen05b}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp}"
EVAL_SIZE="${EVAL_SIZE:-64}"
GPU_ID="${GPU_ID:-0}"
CHECKPOINT_STEPS_STRING="${CHECKPOINT_STEPS_STRING:-80 100 110 120 130 140 150 160 200}"

resolve_run_dir() {
  local experiment_dir="$1"
  if [[ -f "$experiment_dir/run_metadata.json" ]]; then
    printf '%s\n' "$experiment_dir"
    return 0
  fi

  local latest_run=""
  while IFS= read -r candidate; do
    latest_run="$candidate"
  done < <(
    find "$experiment_dir" -mindepth 1 -maxdepth 1 -type d \
      | while IFS= read -r dir; do
          if [[ -f "$dir/run_metadata.json" ]]; then
            basename "$dir"
          fi
        done \
      | sort
  )

  if [[ -z "$latest_run" ]]; then
    echo "No run directory with run_metadata.json found under: $experiment_dir" >&2
    return 1
  fi

  printf '%s/%s\n' "$experiment_dir" "$latest_run"
}

RUN_DIR="${RUN_DIR:-$(resolve_run_dir "$EXPERIMENT_DIR")}"
RUN_METADATA_PATH="$RUN_DIR/run_metadata.json"

MODEL_NAME_FROM_METADATA=""
if [[ -f "$RUN_METADATA_PATH" ]]; then
  MODEL_NAME_FROM_METADATA="$(python - <<'PY' "$RUN_METADATA_PATH"
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
print(metadata.get("model_name", ""))
PY
)"
fi

MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_FROM_METADATA}"

echo "Experiment dir: $EXPERIMENT_DIR"
echo "Run dir: $RUN_DIR"
echo "Base model: ${MODEL_NAME:-<default>}"
echo "Eval size: $EVAL_SIZE"
echo "GPU: $GPU_ID"

for step in $CHECKPOINT_STEPS_STRING; do
  adapter_path="$RUN_DIR/checkpoint-$step"
  if [[ ! -d "$adapter_path" ]]; then
    echo "Skipping checkpoint-$step (missing)"
    continue
  fi

  echo
  echo "Evaluating checkpoint-$step"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT_DIR/smoke_eval.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --adapter_path "$adapter_path" \
    --eval_size "$EVAL_SIZE" \
    --offline true \
    --write_jsonl false
done

if [[ -d "$RUN_DIR/final_adapter" ]]; then
  echo
  echo "Evaluating final_adapter"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT_DIR/smoke_eval.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --adapter_path "$RUN_DIR/final_adapter" \
    --eval_size "$EVAL_SIZE" \
    --offline true \
    --write_jsonl false
fi

echo
echo "Done. Local results:"
echo "  $RUN_DIR/run_results.jsonl"
if [[ "$RUN_DIR" != "$EXPERIMENT_DIR" ]]; then
  echo "Aggregate results:"
  echo "  $EXPERIMENT_DIR/run_results.jsonl"
fi
