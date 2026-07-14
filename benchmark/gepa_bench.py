"""Reproduce the GPT-5.6 prompt-optimization numbers, with nothing of ours in the loop.

This talks to OpenAI and to DSPy. It does NOT need Apprentice's API, our database, or an
account. Bring your own key and check our arithmetic.

    export OPENAI_API_KEY=sk-...
    uv run python gepa_bench.py                 # gpt-5.6-luna, the cheapest 5.6
    uv run python gepa_bench.py --model gpt-5.6-terra

What it does, which is what Apprentice's optimizer does:

  1. Splits the 24-row invoice extraction set with seed 42 into train/val/holdout.
  2. Scores a deliberately weak baseline prompt on the HELD-OUT rows.
  3. Runs DSPy GEPA, which rewrites the prompt using the train and val rows only.
  4. Scores the rewritten prompt on the same held-out rows.

The score is deterministic JSON field F1: parse the output as JSON, compare leaf fields
against the gold answer. No LLM judge marks its own homework.

Expect the shape, not the digits. GEPA is stochastic and the baseline is a weak prompt, so
the starting score moves between runs (we have seen 50.2 and 56.0). What is stable is that
the optimized prompt scores 100 on the holdout. Our own run, on 2026-07-14:

    baseline 50.17  ->  optimized 100.00     12/12 held-out rows improved, 0 regressed
    67 seconds, $0.047, gpt-5.6-luna as both student and reflection model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy

GOLDEN = Path(__file__).parent / "golden.csv"
SEED = 42

# The prompt a developer actually starts with, before anyone has thought about it. The
# point of the exercise is that you should not have to think about it.
BASELINE_PROMPT = "Extract the fields from the text. Return JSON."


def leaf_pairs(obj: Any, prefix: str = "") -> set[tuple[str, str]]:
    """Flatten a JSON object to (path, value) pairs so nested answers score fairly."""
    pairs: set[tuple[str, str]] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            pairs |= leaf_pairs(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            pairs |= leaf_pairs(value, f"{prefix}[{i}]")
    else:
        # Numbers compare by value, not by formatting: 42.10 and 42.1 are the same answer.
        if isinstance(obj, float) and obj.is_integer():
            obj = int(obj)
        pairs.add((prefix, str(obj).strip().lower()))
    return pairs


def field_f1(expected: str, actual: str) -> float:
    try:
        want = json.loads(expected)
    except json.JSONDecodeError:
        return 0.0
    try:
        got = json.loads(extract_json(actual))
    except (json.JSONDecodeError, TypeError):
        # Not JSON at all is a zero. A model that ignores the format has not done the task.
        return 0.0
    if not isinstance(want, dict) or not isinstance(got, dict):
        return 0.0

    w, g = leaf_pairs(want), leaf_pairs(got)
    if not w and not g:
        return 1.0
    overlap = len(w & g)
    if overlap == 0:
        return 0.0
    precision = overlap / len(g)
    recall = overlap / len(w)
    return 2 * precision * recall / (precision + recall)


def extract_json(text: str) -> str:
    """Models like to wrap JSON in prose or fences. Take the outermost object."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class Extract(dspy.Signature):
    """Extract the requested fields from the text and return them as JSON."""

    text: str = dspy.InputField()
    extracted: str = dspy.OutputField()


def metric(example: dspy.Example, prediction: Any, *_: Any) -> Any:
    score = field_f1(example.expected, getattr(prediction, "extracted", ""))
    # GEPA reads this feedback to decide how to rewrite the prompt. A bare number teaches
    # it nothing; naming the specific failure is what makes the rewrite converge.
    if score == 1.0:
        feedback = "Every field matched."
    elif score == 0.0:
        feedback = "No field matched. Return ONE JSON object, no prose, no code fences."
    else:
        feedback = f"Partially correct (F1 {score:.2f}). Field names or values differ from the expected JSON."
    return dspy.Prediction(score=score, feedback=feedback)


def load_rows() -> list[dspy.Example]:
    with GOLDEN.open(newline="", encoding="utf-8") as handle:
        rows = [
            dspy.Example(text=r["input"], expected=r["output"]).with_inputs("text")
            for r in csv.DictReader(handle)
        ]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def evaluate(program: Any, rows: list[dspy.Example]) -> float:
    total = 0.0
    for row in rows:
        prediction = program(text=row.text)
        total += field_f1(row.expected, getattr(prediction, "extracted", ""))
    return 100 * total / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna", help="OpenAI model for both student and reflection")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    lm = dspy.LM(f"openai/{args.model}", cache=False)
    dspy.configure(lm=lm)

    rows = load_rows()
    holdout = rows[:12]
    train = rows[12:19]
    val = rows[19:]
    print(f"{len(rows)} rows -> train {len(train)}, val {len(val)}, holdout {len(holdout)} (seed {SEED})")
    print(f"model: {args.model}\n")

    program = dspy.Predict(Extract)
    program.signature = program.signature.with_instructions(BASELINE_PROMPT)

    started = time.monotonic()
    baseline = evaluate(program, holdout)
    print(f"baseline   {baseline:6.2f}   (prompt: {BASELINE_PROMPT!r})")

    gepa = dspy.GEPA(
        metric=metric,
        auto="light",
        reflection_lm=dspy.LM(f"openai/{args.model}", temperature=1.0, max_tokens=8000, cache=False),
    )
    optimized = gepa.compile(program, trainset=train, valset=val)

    score = evaluate(optimized, holdout)
    elapsed = time.monotonic() - started

    print(f"optimized  {score:6.2f}   ({elapsed:.0f}s)\n")
    print("optimized prompt:\n")
    print(optimized.signature.instructions)


if __name__ == "__main__":
    main()
