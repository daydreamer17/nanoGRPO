from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from nanogrpo.grpo_gsm8k_utils import (
    canonicalize_numeric_text,
    completion_to_text,
    extract_final_answer_from_text,
)

STRICT_FORMAT_REWARD = 0.15
BREVITY_FULL_REWARD = 0.10
BREVITY_PARTIAL_REWARD = 0.05
NUMERIC_PARSE_REWARD = 0.05
PROXIMITY_EXACT_REWARD = 0.25
PROXIMITY_CLOSE_REWARD = 0.20
PROXIMITY_MEDIUM_REWARD = 0.12
PROXIMITY_FAR_REWARD = 0.05
EXACT_ANSWER_REWARD = 0.65
STRICT_FINAL_ANSWER_PREFIX = "Final answer:"


def _split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _count_prefixed_lines(lines: list[str], prefix: str) -> int:
    return sum(1 for line in lines if line.startswith(prefix))


def extract_strict_final_answer_from_text(text: str) -> str | None:
    lines = _split_nonempty_lines(text)
    if not lines:
        return None
    if _count_prefixed_lines(lines, STRICT_FINAL_ANSWER_PREFIX) != 1:
        return None

    final_line = lines[-1]
    if not final_line.startswith(STRICT_FINAL_ANSWER_PREFIX):
        return None

    answer_text = final_line[len(STRICT_FINAL_ANSWER_PREFIX) :].strip()
    if not answer_text:
        return None
    return canonicalize_numeric_text(answer_text)


def has_trailing_text_after_final_answer(text: str) -> bool:
    lines = _split_nonempty_lines(text)
    if not lines:
        return False
    for index, line in enumerate(lines):
        if line.startswith(STRICT_FINAL_ANSWER_PREFIX):
            return index != len(lines) - 1
    return False


def count_reasoning_lines(text: str) -> int:
    lines = _split_nonempty_lines(text)
    if not lines:
        return 0
    strict_final_answer = extract_strict_final_answer_from_text(text)
    if strict_final_answer is not None:
        return max(len(lines) - 1, 0)
    return len(lines)


def parse_numeric_value(text: str | None) -> Decimal | None:
    normalized = canonicalize_numeric_text(text)
    if normalized is None:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def relative_error(predicted: Decimal, gold: Decimal) -> Decimal:
    if gold == 0:
        return abs(predicted - gold)
    return abs(predicted - gold) / abs(gold)


def score_completion_dense_reward_v2(completion: Any, solution: str | None = None) -> dict[str, Any]:
    text = completion_to_text(completion)
    lines = _split_nonempty_lines(text)
    strict_answer = extract_strict_final_answer_from_text(text)
    parsed_answer = extract_final_answer_from_text(text)
    gold_solution = canonicalize_numeric_text(solution)

    strict_format_reward = STRICT_FORMAT_REWARD if strict_answer is not None else 0.0

    if not lines:
        brevity_reward = 0.0
    else:
        reasoning_lines = count_reasoning_lines(text)
        if reasoning_lines <= 3:
            brevity_reward = BREVITY_FULL_REWARD
        elif reasoning_lines <= 5:
            brevity_reward = BREVITY_PARTIAL_REWARD
        else:
            brevity_reward = 0.0

    numeric_parse_reward = NUMERIC_PARSE_REWARD if parsed_answer is not None else 0.0

    proximity_reward = 0.0
    if parsed_answer is not None and gold_solution is not None:
        predicted_value = parse_numeric_value(parsed_answer)
        gold_value = parse_numeric_value(gold_solution)
        if predicted_value is not None and gold_value is not None:
            if predicted_value == gold_value:
                proximity_reward = PROXIMITY_EXACT_REWARD
            else:
                error_ratio = relative_error(predicted_value, gold_value)
                if error_ratio <= Decimal("0.01"):
                    proximity_reward = PROXIMITY_CLOSE_REWARD
                elif error_ratio <= Decimal("0.05"):
                    proximity_reward = PROXIMITY_MEDIUM_REWARD
                elif error_ratio <= Decimal("0.20"):
                    proximity_reward = PROXIMITY_FAR_REWARD

    exact_answer_reward = EXACT_ANSWER_REWARD if parsed_answer == gold_solution and gold_solution is not None else 0.0

    reward_breakdown = {
        "strict_format_reward": strict_format_reward,
        "brevity_reward": brevity_reward,
        "numeric_parse_reward": numeric_parse_reward,
        "proximity_reward": proximity_reward,
        "exact_answer_reward": exact_answer_reward,
    }
    return {
        "text": text,
        "line_count": len(lines),
        "reasoning_line_count": count_reasoning_lines(text) if lines else 0,
        "strict_answer": strict_answer,
        "parsed_answer": parsed_answer,
        "gold_solution": gold_solution,
        "has_trailing_text_after_final_answer": has_trailing_text_after_final_answer(text),
        "reward_breakdown": reward_breakdown,
        "total_reward": sum(reward_breakdown.values()),
    }


def strict_format_reward(completions, **kwargs):
    return [score_completion_dense_reward_v2(completion)["reward_breakdown"]["strict_format_reward"] for completion in completions]


def brevity_reward(completions, **kwargs):
    return [score_completion_dense_reward_v2(completion)["reward_breakdown"]["brevity_reward"] for completion in completions]


def numeric_parse_reward(completions, **kwargs):
    return [score_completion_dense_reward_v2(completion)["reward_breakdown"]["numeric_parse_reward"] for completion in completions]


def proximity_reward(completions, solution, **kwargs):
    return [
        score_completion_dense_reward_v2(completion, gold)["reward_breakdown"]["proximity_reward"]
        for completion, gold in zip(completions, solution)
    ]


def exact_answer_reward(completions, solution, **kwargs):
    return [
        score_completion_dense_reward_v2(completion, gold)["reward_breakdown"]["exact_answer_reward"]
        for completion, gold in zip(completions, solution)
    ]


def build_dense_reward_v2_funcs() -> list:
    return [
        strict_format_reward,
        brevity_reward,
        numeric_parse_reward,
        proximity_reward,
        exact_answer_reward,
    ]
