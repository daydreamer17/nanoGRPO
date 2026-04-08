from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from grpo_gsm8k_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WANDB_PROJECT,
    build_default_run_name,
    completion_to_text,
    configure_runtime_env,
    extract_final_answer_from_text,
    has_wandb_auth_configured,
    parse_bool,
    preprocess_gsm8k_example,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal-but-effective GRPO run on GSM8K.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_slice", type=str, default="train[:512]")
    parser.add_argument("--eval_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=25)
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
        save_total_limit=4,
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


def write_run_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime_paths = configure_runtime_env(args.dataset_root, args.output_dir)
    output_dir = runtime_paths.output_dir
    run_name = args.wandb_run_name or build_default_run_name()

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

    training_args = build_training_args(args, output_dir, run_name)

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        reward_funcs=[format_reward, answer_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    metadata = {
        "model_name": args.model_name,
        "train_slice": args.train_slice,
        "eval_size": args.eval_size,
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "output_dir": str(output_dir),
        "hf_home": os.environ["HF_HOME"],
        "hf_datasets_cache": os.environ["HF_DATASETS_CACHE"],
        "wandb_dir": os.environ["WANDB_DIR"],
        "wandb_project": args.wandb_project,
        "wandb_run_name": run_name,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "max_train_prompt_tokens": train_prompt_max,
        "max_eval_prompt_tokens": eval_prompt_max,
    }
    write_run_metadata(output_dir, metadata)

    trainer.train()

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)


if __name__ == "__main__":
    main()
