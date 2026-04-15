from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanogrpo.grpo_gsm8k_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WANDB_PROJECT,
    append_jsonl_record,
    build_default_run_name,
    build_run_results_jsonl_path,
    build_run_summary_path,
    configure_runtime_env,
    has_wandb_auth_configured,
    parse_bool,
    preprocess_gsm8k_sft_example,
    utc_now_iso,
    write_json_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an SFT+LoRA baseline on GSM8K.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--use_run_subdir", type=parse_bool, default=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_slice", type=str, default="train[:1000]")
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--assistant_only_loss", type=parse_bool, default=False)
    parser.add_argument("--completion_only_loss", type=parse_bool, default=True)
    parser.add_argument("--offline", type=parse_bool, default=False)
    return parser


def ensure_online_wandb_login(project: str, run_name: str) -> None:
    if not has_wandb_auth_configured():
        raise SystemExit(
            "W&B online logging is required for this script. Run `wandb login` or set WANDB_API_KEY first."
        )

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_NAME"] = run_name

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed in the active environment.") from exc

    login_kwargs = {"anonymous": "never", "relogin": False}
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        login_kwargs["key"] = api_key

    try:
        logged_in = wandb.login(**login_kwargs)
    except Exception as exc:  # pragma: no cover - network/auth failure path
        raise SystemExit(f"W&B authentication failed: {exc}") from exc

    if logged_in is False:
        raise SystemExit("W&B authentication did not complete successfully.")


def load_and_prepare_dataset(args):
    from datasets import load_dataset

    train_dataset = load_dataset("openai/gsm8k", "main", split=args.train_slice)
    eval_dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{args.eval_size}]")

    train_dataset = train_dataset.map(
        lambda example, index: preprocess_gsm8k_sft_example(example, index, "train"),
        with_indices=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc=f"Formatting SFT GSM8K {args.train_slice}",
    )
    eval_dataset = eval_dataset.map(
        lambda example, index: preprocess_gsm8k_sft_example(example, index, "eval"),
        with_indices=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc=f"Formatting SFT GSM8K test[:{args.eval_size}]",
    )

    return train_dataset, eval_dataset


def ensure_sequence_length_budget(dataset, tokenizer, split_name: str, max_seq_length: int) -> int:
    max_seen = 0
    for example in dataset:
        input_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        sequence_length = len(input_ids)
        max_seen = max(max_seen, sequence_length)
        if sequence_length > max_seq_length:
            raise ValueError(
                f"{split_name} sequence length {sequence_length} exceeds max_seq_length={max_seq_length} "
                f"for question_id={example['question_id']}"
            )
    return max_seen


def build_training_args(args, output_dir: Path, run_name: str):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to="wandb",
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
        assistant_only_loss=args.assistant_only_loss,
        completion_only_loss=args.completion_only_loss,
        eval_strategy="no",
    )


def resolve_training_output_dir(experiment_dir: Path, run_name: str, use_run_subdir: bool) -> Path:
    if not use_run_subdir:
        return experiment_dir
    return experiment_dir / run_name


def write_run_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def extract_final_log_row(log_rows: list[dict]) -> dict:
    for row in reversed(log_rows):
        if "loss" in row:
            return row
    return log_rows[-1] if log_rows else {}


def select_min_row(log_rows: list[dict], metric_name: str) -> dict | None:
    rows_with_metric = [row for row in log_rows if metric_name in row]
    if not rows_with_metric:
        return None
    return min(rows_with_metric, key=lambda row: row[metric_name])


def build_training_summary(
    args,
    run_name: str,
    experiment_dir: Path,
    output_dir: Path,
    final_adapter_dir: Path,
    trainer,
    train_result,
) -> dict:
    log_rows = [row for row in trainer.state.log_history if "step" in row]
    final_row = extract_final_log_row(log_rows)
    best_loss_row = select_min_row(log_rows, "loss")
    checkpoint_dirs = sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir())
    train_metrics = dict(train_result.metrics)
    train_metrics["global_step"] = trainer.state.global_step

    return {
        "summary_type": "sft_train",
        "created_at": utc_now_iso(),
        "run_name": run_name,
        "wandb_project": args.wandb_project,
        "model_name": args.model_name,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "train_slice": args.train_slice,
        "train_examples": len(trainer.train_dataset),
        "eval_examples": len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0,
        "offline": args.offline,
        "train_mode": "sft_lora",
        "config": {
            "max_steps": args.max_steps,
            "max_seq_length": args.max_seq_length,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "assistant_only_loss": args.assistant_only_loss,
            "completion_only_loss": args.completion_only_loss,
        },
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "checkpoint_dirs": checkpoint_dirs,
        "train_metrics": train_metrics,
        "final_metrics": {
            "step": final_row.get("step"),
            "loss": final_row.get("loss"),
            "grad_norm": final_row.get("grad_norm"),
            "learning_rate": final_row.get("learning_rate"),
            "epoch": final_row.get("epoch"),
        },
        "best_metrics": {
            "loss": {
                "step": best_loss_row.get("step") if best_loss_row else None,
                "value": best_loss_row.get("loss") if best_loss_row else None,
            }
        },
    }


def persist_training_summary(experiment_dir: Path, output_dir: Path, run_name: str, summary: dict) -> None:
    summary_path = build_run_summary_path(output_dir, "sft_train", run_name)
    summary["summary_path"] = str(summary_path)
    summary["run_results_path"] = str(build_run_results_jsonl_path(output_dir))
    aggregate_results_path = build_run_results_jsonl_path(experiment_dir)
    summary["aggregate_run_results_path"] = str(aggregate_results_path)
    write_json_file(summary_path, summary)
    append_jsonl_record(summary["run_results_path"], summary)
    if aggregate_results_path != Path(summary["run_results_path"]):
        append_jsonl_record(aggregate_results_path, summary)


def cleanup_training_runtime(trainer=None, exit_code: int = 0) -> None:
    cleanup_errors = []

    accelerator = getattr(trainer, "accelerator", None) if trainer is not None else None
    if accelerator is not None and hasattr(accelerator, "end_training"):
        try:
            accelerator.end_training()
        except Exception as exc:  # pragma: no cover - best-effort cleanup path
            cleanup_errors.append(f"accelerator.end_training failed: {exc}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    except Exception as exc:  # pragma: no cover - best-effort cleanup path
        cleanup_errors.append(f"wandb.finish failed: {exc}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:  # pragma: no cover - best-effort cleanup path
        cleanup_errors.append(f"torch cleanup failed: {exc}")

    for message in cleanup_errors:
        print(f"[cleanup warning] {message}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    trainer = None
    exit_code = 0

    runtime_paths = configure_runtime_env(args.dataset_root, args.output_dir, offline=args.offline)
    run_name = args.wandb_run_name or build_default_run_name(prefix="sft-gsm8k-qwen25-05b")
    experiment_dir = runtime_paths.output_dir
    output_dir = resolve_training_output_dir(experiment_dir, run_name, args.use_run_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_online_wandb_login(args.wandb_project, run_name)

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    train_dataset, eval_dataset = load_and_prepare_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_seq_max = ensure_sequence_length_budget(train_dataset, tokenizer, "train", args.max_seq_length)
    eval_seq_max = ensure_sequence_length_budget(eval_dataset, tokenizer, "eval", args.max_seq_length)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = build_training_args(args, output_dir, run_name)

    trainer = SFTTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    metadata = {
        "model_name": args.model_name,
        "train_mode": "sft_lora",
        "train_slice": args.train_slice,
        "eval_size": args.eval_size,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "assistant_only_loss": args.assistant_only_loss,
        "completion_only_loss": args.completion_only_loss,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "use_run_subdir": args.use_run_subdir,
        "hf_home": os.environ["HF_HOME"],
        "hf_datasets_cache": os.environ["HF_DATASETS_CACHE"],
        "wandb_dir": os.environ["WANDB_DIR"],
        "wandb_project": args.wandb_project,
        "wandb_run_name": run_name,
        "offline": args.offline,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "max_train_sequence_tokens": train_seq_max,
        "max_eval_sequence_tokens": eval_seq_max,
    }
    write_run_metadata(output_dir, metadata)

    try:
        final_adapter_dir = output_dir / "final_adapter"
        train_result = trainer.train()
        trainer.save_model(str(final_adapter_dir))
        tokenizer.save_pretrained(final_adapter_dir)

        training_summary = build_training_summary(
            args=args,
            run_name=run_name,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            final_adapter_dir=final_adapter_dir,
            trainer=trainer,
            train_result=train_result,
        )
        persist_training_summary(experiment_dir, output_dir, run_name, training_summary)
    except Exception:
        exit_code = 1
        raise
    finally:
        cleanup_training_runtime(trainer=trainer, exit_code=exit_code)


if __name__ == "__main__":
    main()
