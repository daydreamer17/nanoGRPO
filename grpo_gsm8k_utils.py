from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from netrc import NetrcParseError, netrc
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp")
DEFAULT_HF_HOME = DEFAULT_DATASET_ROOT / "hf"
DEFAULT_HF_DATASETS_CACHE = DEFAULT_HF_HOME / "datasets"
DEFAULT_TRANSFORMERS_CACHE = DEFAULT_HF_HOME / "transformers"
DEFAULT_WANDB_DIR = DEFAULT_DATASET_ROOT / "wandb"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "outputs" / "nano-grpo-qwen05b"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_WANDB_PROJECT = "nano-grpo-qwen05b"

GOLD_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*(.+)", re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem clearly and concisely. "
    "Use at most 3 short reasoning lines. "
    "The final line must be exactly `Final answer: <number>`. "
    "Do not add any text after the final answer line."
)


@dataclass(frozen=True)
class RuntimePaths:
    dataset_root: Path
    hf_home: Path
    hf_datasets_cache: Path
    wandb_dir: Path
    output_dir: Path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_runtime_env(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    offline: bool = False,
) -> RuntimePaths:
    dataset_root = Path(dataset_root).expanduser().resolve()
    hf_home = dataset_root / "hf"
    hf_datasets_cache = hf_home / "datasets"
    transformers_cache = hf_home / "transformers"
    wandb_dir = dataset_root / "wandb"
    output_dir = Path(output_dir).expanduser().resolve()

    for path in (dataset_root, hf_home, hf_datasets_cache, transformers_cache, wandb_dir, output_dir):
        ensure_directory(path)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return RuntimePaths(
        dataset_root=dataset_root,
        hf_home=hf_home,
        hf_datasets_cache=hf_datasets_cache,
        wandb_dir=wandb_dir,
        output_dir=output_dir,
    )


def build_default_run_name(prefix: str = "grpo-gsm8k-qwen25-05b") -> str:
    return f"{prefix}-{build_timestamp_tag()}"


def build_timestamp_tag() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return timestamp


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_filename(value: str) -> str:
    sanitized = SAFE_FILENAME_RE.sub("-", value).strip("-")
    return sanitized or "run"


def _normalize_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        normalized = format(value.quantize(Decimal(1)), "f")
    else:
        normalized = format(value.normalize(), "f").rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def canonicalize_numeric_text(text: str | None) -> str | None:
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    candidate = candidate.replace("$", "").replace(",", "")
    candidate = candidate.rstrip(". ")
    match = NUMBER_TOKEN_RE.search(candidate)
    if match is None:
        return None
    token = match.group(0)
    try:
        if "/" in token:
            fraction = Fraction(token)
            if fraction.denominator == 1:
                return str(fraction.numerator)
            decimal_value = Decimal(fraction.numerator) / Decimal(fraction.denominator)
            return _normalize_decimal(decimal_value)
        return _normalize_decimal(Decimal(token))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return token


def extract_gold_solution(answer: str) -> str:
    match = GOLD_ANSWER_RE.search(answer)
    if match is None:
        raise ValueError(f"Could not extract GSM8K final answer from: {answer!r}")
    normalized = canonicalize_numeric_text(match.group(1))
    if normalized is None:
        raise ValueError(f"Could not normalize GSM8K final answer from: {answer!r}")
    return normalized


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def preprocess_gsm8k_example(example: dict[str, Any], index: int, split_name: str) -> dict[str, Any]:
    return {
        "prompt": build_prompt(example["question"]),
        "question": example["question"].strip(),
        "solution": extract_gold_solution(example["answer"]),
        "question_id": index if split_name == "train" else 100_000 + index,
    }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if not completion:
            return ""
        if isinstance(completion[0], dict):
            return str(completion[-1].get("content", ""))
        return "".join(str(part) for part in completion)
    return str(completion)


def extract_final_answer_from_text(text: str) -> str | None:
    match = FINAL_ANSWER_RE.search(text)
    if match is None:
        return None
    return canonicalize_numeric_text(match.group(1))


def has_wandb_auth_configured() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        parsed = netrc(str(netrc_path))
    except (NetrcParseError, OSError):
        return False
    return any(host in parsed.hosts for host in ("api.wandb.ai", "wandb.ai"))


def build_eval_jsonl_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "smoke_eval.jsonl"


def build_run_results_jsonl_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "run_results.jsonl"


def build_run_summary_path(output_dir: str | Path, summary_type: str, label: str) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    summaries_dir = output_dir / "run_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{summary_type}_{sanitize_filename(label)}_{build_timestamp_tag()}.json"
    return summaries_dir / filename


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def append_jsonl_record(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path
