#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT_DIR="${1:-/root/autodl-tmp/outputs/nano-grpo-qwen05b}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PHASE1_EVAL_SIZE="${PHASE1_EVAL_SIZE:-64}"
PHASE2_EVAL_SIZE="${PHASE2_EVAL_SIZE:-128}"
PHASE1_STEPS_STRING="${PHASE1_STEPS_STRING:-80 100 120 140 160}"
TOPK="${TOPK:-3}"

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

if [[ ! -f "$RUN_METADATA_PATH" ]]; then
  echo "Missing run_metadata.json under: $RUN_DIR" >&2
  exit 1
fi

RUN_NAME="$(python - <<'PY' "$RUN_METADATA_PATH"
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
print(metadata.get("wandb_run_name", ""))
PY
)"

MODEL_NAME_FROM_METADATA="$(python - <<'PY' "$RUN_METADATA_PATH"
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
print(metadata.get("model_name", ""))
PY
)"

MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_FROM_METADATA}"

REWARD_SCHEME="$(python - <<'PY' "$RUN_METADATA_PATH"
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
print(metadata.get("reward_scheme", "baseline"))
PY
)"

if [[ "$REWARD_SCHEME" != "dense_v2" ]]; then
  echo "Warning: run reward_scheme is '$REWARD_SCHEME', not dense_v2." >&2
fi

TIMESTAMP_TAG="$(date -u +%Y%m%d-%H%M%S)"
SUMMARY_PATH="${SUMMARY_PATH:-$RUN_DIR/dense_v2_eval_sweep_summary_$TIMESTAMP_TAG.md}"

run_eval() {
  local adapter_path="$1"
  local eval_size="$2"

  if [[ ! -d "$adapter_path" ]]; then
    echo "Skipping missing adapter path: $adapter_path"
    return 0
  fi

  echo
  echo "Evaluating $(basename "$adapter_path") with eval_size=$eval_size"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT_DIR/smoke_eval.py" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$RUN_DIR" \
    --model_name "$MODEL_NAME" \
    --adapter_path "$adapter_path" \
    --eval_size "$eval_size" \
    --batch_size "$BATCH_SIZE" \
    --offline true \
    --write_jsonl false
}

echo "Experiment dir: $EXPERIMENT_DIR"
echo "Run dir: $RUN_DIR"
echo "Run name: $RUN_NAME"
echo "Base model: $MODEL_NAME"
echo "Reward scheme: $REWARD_SCHEME"
echo "GPU: $GPU_ID"
echo "Phase 1 eval_size: $PHASE1_EVAL_SIZE"
echo "Phase 2 eval_size: $PHASE2_EVAL_SIZE"

for step in $PHASE1_STEPS_STRING; do
  run_eval "$RUN_DIR/checkpoint-$step" "$PHASE1_EVAL_SIZE"
done

run_eval "$RUN_DIR/final_adapter" "$PHASE1_EVAL_SIZE"

mapfile -t TOP_CHECKPOINTS < <(python - <<'PY' "$RUN_DIR/run_results.jsonl" "$RUN_NAME" "$PHASE1_EVAL_SIZE" "$TOPK"
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
run_name = sys.argv[2]
eval_size = int(sys.argv[3])
topk = int(sys.argv[4])

rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
latest = {}
for row in rows:
    if row.get("summary_type") != "eval":
        continue
    if row.get("run_name") != run_name:
        continue
    if row.get("eval_examples") != eval_size:
        continue
    adapter_label = row.get("adapter_label")
    if not adapter_label or adapter_label == "final_adapter":
        continue
    latest[adapter_label] = row

ranked = sorted(
    latest.values(),
    key=lambda row: (
        row.get("tuned_exact_match_rate", 0.0),
        row.get("tuned_exact_match", 0),
        row.get("tuned_format_hits", 0),
    ),
    reverse=True,
)

for row in ranked[:topk]:
    print(row["adapter_label"])
PY
)

if [[ "${#TOP_CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found for phase 2 reevaluation." >&2
  exit 1
fi

echo
echo "Top checkpoints for phase 2: ${TOP_CHECKPOINTS[*]}"

for adapter_label in "${TOP_CHECKPOINTS[@]}"; do
  run_eval "$RUN_DIR/$adapter_label" "$PHASE2_EVAL_SIZE"
done

python - <<'PY' "$RUN_DIR/run_results.jsonl" "$RUN_NAME" "$PHASE1_EVAL_SIZE" "$PHASE2_EVAL_SIZE" "$SUMMARY_PATH"
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
run_name = sys.argv[2]
phase1_eval_size = int(sys.argv[3])
phase2_eval_size = int(sys.argv[4])
summary_path = Path(sys.argv[5])

rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]

def latest_eval_rows(target_eval_size: int):
    latest = {}
    for row in rows:
        if row.get("summary_type") != "eval":
            continue
        if row.get("run_name") != run_name:
            continue
        if row.get("eval_examples") != target_eval_size:
            continue
        adapter_label = row.get("adapter_label")
        if not adapter_label:
            continue
        latest[adapter_label] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            row.get("tuned_exact_match_rate", 0.0),
            row.get("tuned_exact_match", 0),
            row.get("tuned_format_hits", 0),
        ),
        reverse=True,
    )

phase1_rows = latest_eval_rows(phase1_eval_size)
phase2_rows = latest_eval_rows(phase2_eval_size)

def render_table(eval_rows):
    if not eval_rows:
        return ["No results."]
    lines = ["| Adapter | Exact Match | Rate | Format Hits | Delta |", "| --- | --- | --- | --- | --- |"]
    for row in eval_rows:
        lines.append(
            "| {label} | {exact}/{total} | {rate:.4f} | {format_hits} | {delta} |".format(
                label=row.get("adapter_label", ""),
                exact=row.get("tuned_exact_match", 0),
                total=row.get("eval_examples", 0),
                rate=row.get("tuned_exact_match_rate", 0.0),
                format_hits=row.get("tuned_format_hits", 0),
                delta=row.get("delta_exact_match", 0),
            )
        )
    return lines

summary_lines = [
    f"# Dense V2 Eval Sweep Summary",
    "",
    f"- Run name: `{run_name}`",
    f"- Phase 1 eval size: `{phase1_eval_size}`",
    f"- Phase 2 eval size: `{phase2_eval_size}`",
    f"- Source results: `{results_path}`",
    "",
    "## Phase 1",
    "",
]
summary_lines.extend(render_table(phase1_rows))
summary_lines.extend(["", "## Phase 2", ""])
summary_lines.extend(render_table(phase2_rows))

summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(summary_path)
PY

echo
echo "Done."
echo "Run-local results: $RUN_DIR/run_results.jsonl"
echo "Aggregate results: $EXPERIMENT_DIR/run_results.jsonl"
echo "Summary doc: $SUMMARY_PATH"
