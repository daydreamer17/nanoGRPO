from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo_gsm8k_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    append_jsonl_record,
    build_eval_jsonl_path,
    build_run_results_jsonl_path,
    build_run_summary_path,
    completion_to_text,
    configure_runtime_env,
    extract_final_answer_from_text,
    parse_bool,
    preprocess_gsm8k_example,
    utc_now_iso,
    write_json_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare base vs tuned model on a small GSM8K eval slice.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--write_jsonl", type=parse_bool, default=True)
    parser.add_argument("--offline", type=parse_bool, default=False)
    return parser


def load_eval_dataset(eval_size: int):
    from datasets import load_dataset

    eval_dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{eval_size}]")
    return eval_dataset.map(
        lambda example, index: preprocess_gsm8k_example(example, index, "eval"),
        with_indices=True,
        remove_columns=eval_dataset.column_names,
        desc=f"Formatting GSM8K test[:{eval_size}]",
    )


def ensure_prompt_length_budget(dataset, tokenizer, max_prompt_length: int) -> int:
    max_seen = 0
    for example in dataset:
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_length = len(prompt_ids)
        max_seen = max(max_seen, prompt_length)
        if prompt_length > max_prompt_length:
            raise ValueError(
                f"Eval prompt length {prompt_length} exceeds max_prompt_length={max_prompt_length} "
                f"for question_id={example['question_id']}"
            )
    return max_seen


def load_model(model_name: str, adapter_path: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model_kwargs = {"device_map": "auto"}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate_predictions(model, tokenizer, dataset, batch_size: int, max_completion_length: int):
    import torch

    predictions = []
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    for start in range(0, len(dataset), batch_size):
        stop = min(len(dataset), start + batch_size)
        batch = [dataset[index] for index in range(start, stop)]
        prompt_texts = [
            tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
            for example in batch
        ]
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1)
        for row_index, sequence in enumerate(generated):
            completion_ids = sequence[int(prompt_lengths[row_index]) :]
            predictions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    return predictions


def merge_results(dataset, base_predictions, tuned_predictions):
    rows = []
    for example, base_text, tuned_text in zip(dataset, base_predictions, tuned_predictions):
        gold = example["solution"]
        base_answer = extract_final_answer_from_text(base_text)
        tuned_answer = extract_final_answer_from_text(tuned_text)
        rows.append(
            {
                "question_id": example["question_id"],
                "question": example["question"],
                "gold": gold,
                "base_text": completion_to_text(base_text),
                "base_answer": base_answer,
                "base_correct": base_answer == gold,
                "tuned_text": completion_to_text(tuned_text),
                "tuned_answer": tuned_answer,
                "tuned_correct": tuned_answer == gold,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_run_metadata(output_dir: Path) -> dict:
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def persist_eval_summary(output_dir: Path, experiment_dir: Path, adapter_path: str, summary: dict) -> None:
    label = f"eval-{Path(adapter_path).name}"
    summary_path = build_run_summary_path(output_dir, "eval", label)
    summary["summary_path"] = str(summary_path)
    summary["run_results_path"] = str(build_run_results_jsonl_path(output_dir))
    aggregate_results_path = build_run_results_jsonl_path(experiment_dir)
    summary["aggregate_run_results_path"] = str(aggregate_results_path)
    write_json_file(summary_path, summary)
    append_jsonl_record(summary["run_results_path"], summary)
    if aggregate_results_path != Path(summary["run_results_path"]):
        append_jsonl_record(aggregate_results_path, summary)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime_paths = configure_runtime_env(args.dataset_root, args.output_dir, offline=args.offline)
    output_dir = runtime_paths.output_dir
    adapter_path = args.adapter_path or str(output_dir / "final_adapter")
    if not Path(adapter_path).exists():
        raise SystemExit(f"Adapter path does not exist: {adapter_path}")

    run_metadata = load_run_metadata(output_dir)
    model_name = args.model_name or run_metadata.get("model_name") or DEFAULT_MODEL_NAME

    from transformers import AutoTokenizer
    import torch

    eval_dataset = load_eval_dataset(args.eval_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_prompt_tokens = ensure_prompt_length_budget(eval_dataset, tokenizer, args.max_prompt_length)

    base_model = load_model(model_name)
    base_predictions = generate_predictions(
        base_model,
        tokenizer,
        eval_dataset,
        batch_size=args.batch_size,
        max_completion_length=args.max_completion_length,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_model = load_model(model_name, adapter_path=adapter_path)
    tuned_predictions = generate_predictions(
        tuned_model,
        tokenizer,
        eval_dataset,
        batch_size=args.batch_size,
        max_completion_length=args.max_completion_length,
    )

    rows = merge_results(eval_dataset, base_predictions, tuned_predictions)
    base_correct = sum(int(row["base_correct"]) for row in rows)
    tuned_correct = sum(int(row["tuned_correct"]) for row in rows)
    summary = {
        "summary_type": "eval",
        "created_at": utc_now_iso(),
        "eval_examples": len(rows),
        "max_prompt_tokens": max_prompt_tokens,
        "base_exact_match": base_correct,
        "tuned_exact_match": tuned_correct,
        "delta_exact_match": tuned_correct - base_correct,
        "adapter_path": adapter_path,
        "adapter_label": Path(adapter_path).name,
        "base_exact_match_rate": base_correct / len(rows) if rows else 0.0,
        "tuned_exact_match_rate": tuned_correct / len(rows) if rows else 0.0,
        "base_format_hits": sum(row["base_answer"] is not None for row in rows),
        "tuned_format_hits": sum(row["tuned_answer"] is not None for row in rows),
        "model_name": model_name,
        "offline": args.offline,
        "batch_size": args.batch_size,
    }
    experiment_dir = output_dir
    if run_metadata:
        summary["run_name"] = run_metadata.get("wandb_run_name")
        summary["train_slice"] = run_metadata.get("train_slice")
        summary["train_examples"] = run_metadata.get("train_examples")
        summary["wandb_project"] = run_metadata.get("wandb_project")
        summary["experiment_dir"] = run_metadata.get("experiment_dir") or str(output_dir)
        experiment_dir = Path(summary["experiment_dir"]).expanduser().resolve()
    else:
        summary["experiment_dir"] = str(output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.write_jsonl:
        write_jsonl(build_eval_jsonl_path(output_dir), rows)
    persist_eval_summary(output_dir, experiment_dir, adapter_path, summary)


if __name__ == "__main__":
    main()
