from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanogrpo.dense_reward_v2 import build_dense_reward_v2_funcs
from nanogrpo.grpo_gsm8k_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WANDB_PROJECT,
    append_jsonl_record,
    build_run_results_jsonl_path,
    build_run_summary_path,
    build_default_run_name,
    completion_to_text,
    configure_runtime_env,
    extract_final_answer_from_text,
    has_wandb_auth_configured,
    parse_bool,
    preprocess_gsm8k_example,
    utc_now_iso,
    write_json_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal-but-effective GRPO run on GSM8K.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--use_run_subdir", type=parse_bool, default=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reward_scheme", type=str, choices=("baseline", "dense_v2"), default="baseline")
    parser.add_argument("--train_slice", type=str, default="train[:512]")
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=160)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_vllm", type=parse_bool, default=True)
    parser.add_argument("--vllm_mode", type=str, default="server")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--offline", type=parse_bool, default=False)
    return parser


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(0.2 if extract_final_answer_from_text(text) is not None else 0.0)
    return rewards


def answer_reward(completions, solution, **kwargs):
    rewards = []
    for completion, gold in zip(completions, solution):
        text = completion_to_text(completion)
        rewards.append(1.0 if extract_final_answer_from_text(text) == gold else 0.0)
    return rewards


def resolve_reward_funcs(reward_scheme: str) -> list:
    if reward_scheme == "baseline":
        return [format_reward, answer_reward]
    if reward_scheme == "dense_v2":
        return build_dense_reward_v2_funcs()
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme}")


def resolve_primary_answer_metric_name(reward_scheme: str) -> str:
    if reward_scheme == "baseline":
        return "answer_reward"
    if reward_scheme == "dense_v2":
        return "exact_answer_reward"
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme}")


def resolve_primary_format_metric_name(reward_scheme: str) -> str:
    if reward_scheme == "baseline":
        return "format_reward"
    if reward_scheme == "dense_v2":
        return "strict_format_reward"
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme}")


def build_reward_metric_key(metric_name: str) -> str:
    return f"rewards/{metric_name}/mean"


def build_reward_metric_summary(log_rows: list[dict], reward_func_names: list[str]) -> dict[str, dict | None]:
    summary = {}
    for reward_name in reward_func_names:
        metric_key = build_reward_metric_key(reward_name)
        best_row = select_best_row(log_rows, metric_key)
        summary[reward_name] = {
            "step": best_row.get("step") if best_row else None,
            "value": best_row.get(metric_key) if best_row else None,
        }
    return summary


def extract_final_log_row(log_rows: list[dict]) -> dict:
    for row in reversed(log_rows):
        if "reward" in row:
            return row
    return log_rows[-1] if log_rows else {}


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
        lambda example, index: preprocess_gsm8k_example(example, index, "train"),
        with_indices=True,
        remove_columns=train_dataset.column_names,
        desc=f"Formatting GSM8K {args.train_slice}",
    )
    eval_dataset = eval_dataset.map(
        lambda example, index: preprocess_gsm8k_example(example, index, "eval"),
        with_indices=True,
        remove_columns=eval_dataset.column_names,
        desc=f"Formatting GSM8K test[:{args.eval_size}]",
    )

    return train_dataset, eval_dataset


def ensure_prompt_length_budget(dataset, tokenizer, split_name: str, max_prompt_length: int) -> int:
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
                f"{split_name} prompt length {prompt_length} exceeds max_prompt_length={max_prompt_length} "
                f"for question_id={example['question_id']}"
            )
    return max_seen


def build_training_args(args, output_dir: Path, run_name: str):
    from trl import GRPOConfig

    return GRPOConfig(
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
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        loss_type="dapo",
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_max_model_length=args.max_prompt_length + args.max_completion_length,
        log_completions=True,
        num_completions_to_print=2,
    )


def resolve_training_output_dir(experiment_dir: Path, run_name: str, use_run_subdir: bool) -> Path:
    if not use_run_subdir:
        return experiment_dir
    return experiment_dir / run_name


def write_run_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def select_best_row(log_rows: list[dict], metric_name: str) -> dict | None:
    rows_with_metric = [row for row in log_rows if metric_name in row]
    if not rows_with_metric:
        return None
    return max(rows_with_metric, key=lambda row: row[metric_name])


def build_training_summary(
    args,
    run_name: str,
    experiment_dir: Path,
    output_dir: Path,
    final_adapter_dir: Path,
    reward_func_names: list[str],
    trainer,
    train_result,
) -> dict:
    log_rows = [row for row in trainer.state.log_history if "step" in row]
    final_row = extract_final_log_row(log_rows)
    best_reward_row = select_best_row(log_rows, "reward")
    primary_answer_metric_name = resolve_primary_answer_metric_name(args.reward_scheme)
    primary_format_metric_name = resolve_primary_format_metric_name(args.reward_scheme)
    primary_answer_metric_key = build_reward_metric_key(primary_answer_metric_name)
    primary_format_metric_key = build_reward_metric_key(primary_format_metric_name)
    best_answer_reward_row = select_best_row(log_rows, primary_answer_metric_key)
    best_format_reward_row = select_best_row(log_rows, primary_format_metric_key)
    final_reward_metrics = {
        reward_name: final_row.get(build_reward_metric_key(reward_name)) for reward_name in reward_func_names
    }
    best_reward_metrics = build_reward_metric_summary(log_rows, reward_func_names)

    checkpoint_dirs = sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir())
    train_metrics = dict(train_result.metrics)
    train_metrics["global_step"] = trainer.state.global_step

    return {
        "summary_type": "train",
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
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "reward_scheme": args.reward_scheme,
        "reward_func_names": reward_func_names,
        "config": {
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "learning_rate": args.learning_rate,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "beta": args.beta,
        },
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "checkpoint_dirs": checkpoint_dirs,
        "train_metrics": train_metrics,
        "final_metrics": {
            "step": final_row.get("step"),
            "reward": final_row.get("reward"),
            "answer_reward_mean": final_row.get(primary_answer_metric_key),
            "format_reward_mean": final_row.get(primary_format_metric_key),
            "loss": final_row.get("loss"),
            "entropy": final_row.get("entropy"),
            "mean_completion_length": final_row.get("completions/mean_length"),
            "clipped_ratio": final_row.get("completions/clipped_ratio"),
            "reward_metrics": final_reward_metrics,
        },
        "best_metrics": {
            "reward": {
                "step": best_reward_row.get("step") if best_reward_row else None,
                "value": best_reward_row.get("reward") if best_reward_row else None,
            },
            "answer_reward_mean": {
                "step": best_answer_reward_row.get("step") if best_answer_reward_row else None,
                "value": best_answer_reward_row.get(primary_answer_metric_key) if best_answer_reward_row else None,
            },
            "format_reward_mean": {
                "step": best_format_reward_row.get("step") if best_format_reward_row else None,
                "value": best_format_reward_row.get(primary_format_metric_key) if best_format_reward_row else None,
            },
            "reward_metrics": best_reward_metrics,
        },
        "nonzero_reward_steps": sum(1 for row in log_rows if row.get("reward", 0.0) > 0.0),
        "nonzero_answer_reward_steps": sum(
            1 for row in log_rows if row.get(primary_answer_metric_key, 0.0) > 0.0
        ),
        "nonzero_format_reward_steps": sum(
            1 for row in log_rows if row.get(primary_format_metric_key, 0.0) > 0.0
        ),
    }


def persist_training_summary(experiment_dir: Path, output_dir: Path, run_name: str, summary: dict) -> None:
    summary_path = build_run_summary_path(output_dir, "train", run_name)
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
    run_name = args.wandb_run_name or build_default_run_name()
    experiment_dir = runtime_paths.output_dir
    output_dir = resolve_training_output_dir(experiment_dir, run_name, args.use_run_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_online_wandb_login(args.wandb_project, run_name)

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOTrainer

    train_dataset, eval_dataset = load_and_prepare_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_prompt_max = ensure_prompt_length_budget(train_dataset, tokenizer, "train", args.max_prompt_length)
    eval_prompt_max = ensure_prompt_length_budget(eval_dataset, tokenizer, "eval", args.max_prompt_length)

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

    reward_funcs = resolve_reward_funcs(args.reward_scheme)
    reward_func_names = [reward_func.__name__ for reward_func in reward_funcs]
    training_args = build_training_args(args, output_dir, run_name)

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    metadata = {
        "model_name": args.model_name,
        "reward_scheme": args.reward_scheme,
        "reward_func_names": reward_func_names,
        "train_slice": args.train_slice,
        "eval_size": args.eval_size,
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "temperature": args.temperature,
        "beta": args.beta,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
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
        "max_train_prompt_tokens": train_prompt_max,
        "max_eval_prompt_tokens": eval_prompt_max,
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
            reward_func_names=reward_func_names,
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
