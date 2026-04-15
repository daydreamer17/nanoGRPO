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
    append_jsonl_record,
    build_run_results_jsonl_path,
    build_run_summary_path,
    configure_runtime_env,
    utc_now_iso,
    write_json_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone model directory.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--base_model_name", type=str, default="")
    parser.add_argument("--merged_model_dir", type=str, default="")
    parser.add_argument("--offline", type=str, default="false")
    return parser


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def load_run_metadata(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_paths(args) -> tuple[Path | None, Path, str, Path]:
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None

    if args.adapter_path:
        adapter_path = Path(args.adapter_path).expanduser().resolve()
    elif run_dir is not None:
        adapter_path = run_dir / "final_adapter"
    else:
        raise SystemExit("Provide either --adapter_path or --run_dir so the adapter can be located.")

    if not adapter_path.exists():
        raise SystemExit(f"Adapter path does not exist: {adapter_path}")

    metadata = load_run_metadata(run_dir)
    base_model_name = args.base_model_name or metadata.get("model_name") or DEFAULT_MODEL_NAME

    if args.merged_model_dir:
        merged_model_dir = Path(args.merged_model_dir).expanduser().resolve()
    elif run_dir is not None:
        merged_model_dir = run_dir / "merged_model"
    else:
        merged_model_dir = adapter_path.parent / "merged_model"

    return run_dir, adapter_path, base_model_name, merged_model_dir


def build_merge_summary(
    run_dir: Path | None,
    adapter_path: Path,
    base_model_name: str,
    merged_model_dir: Path,
    experiment_dir: Path,
) -> dict:
    run_name = run_dir.name if run_dir is not None else adapter_path.parent.name
    return {
        "summary_type": "merge",
        "created_at": utc_now_iso(),
        "run_name": run_name,
        "train_mode": "adapter_merge",
        "base_model_name": base_model_name,
        "adapter_path": str(adapter_path),
        "merged_model_dir": str(merged_model_dir),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "experiment_dir": str(experiment_dir),
    }


def persist_merge_summary(experiment_dir: Path, output_dir: Path, label: str, summary: dict) -> None:
    summary_path = build_run_summary_path(output_dir, "merge", label)
    summary["summary_path"] = str(summary_path)
    summary["run_results_path"] = str(build_run_results_jsonl_path(output_dir))
    aggregate_results_path = build_run_results_jsonl_path(experiment_dir)
    summary["aggregate_run_results_path"] = str(aggregate_results_path)
    write_json_file(summary_path, summary)
    append_jsonl_record(summary["run_results_path"], summary)
    if aggregate_results_path != Path(summary["run_results_path"]):
        append_jsonl_record(aggregate_results_path, summary)


def main() -> None:
    args = build_parser().parse_args()
    args.offline = parse_bool(args.offline)

    runtime_paths = configure_runtime_env(args.dataset_root, args.output_dir, offline=args.offline)
    experiment_dir = runtime_paths.output_dir

    run_dir, adapter_path, base_model_name, merged_model_dir = resolve_paths(args)
    output_dir = run_dir if run_dir is not None else adapter_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import torch

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()

    tokenizer_source = str(adapter_path) if (adapter_path / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    merged_model.save_pretrained(str(merged_model_dir))
    tokenizer.save_pretrained(str(merged_model_dir))

    summary = build_merge_summary(
        run_dir=run_dir,
        adapter_path=adapter_path,
        base_model_name=base_model_name,
        merged_model_dir=merged_model_dir,
        experiment_dir=experiment_dir,
    )
    persist_merge_summary(experiment_dir, output_dir, merged_model_dir.name, summary)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - best-effort cleanup path
        print(f"[cleanup warning] torch.cuda.empty_cache failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
