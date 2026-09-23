"""
scoring.py - machine-checkable success predicates for task_set.json.

A task's `success_criteria` is a JSON predicate:

  {"contains_amount": 250}                  a number in the answer equals 250 (+/- tol)
  {"contains_amount": 12980, "tol": 1}
  {"contains_any": ["cruciate", "IVDD"]}    case-insensitive substring, any one
  {"contains_all": ["Canada", "auto"]}
  {"refused": true}                         the run's structured refused flag
  {"language": "he"}                        >= 80% of the letters are Hebrew
  {"max_words": 30}
  {"no_dispatch": true}                     team: zero worker turns; single: zero tool calls
  {"judge": "<rubric>"}                     decided by the Sonnet judge (judge.py)
  {"all": [p, ...]}  {"any": [p, ...]}  {"not": p}

evaluate() returns True / False, or JUDGE when the outcome depends on a judge predicate.
Nothing here reads the task's type or its answerable label.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

JUDGE = "judge"
Outcome = Union[bool, str]

_NUMBER = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_HEBREW = re.compile(r"[֐-׿]")


@dataclass
class RunView:
    """What a predicate may look at: the run's outputs, never the task's labels."""
    answer: str
    refused: bool
    config: str                     # "single" | "team"
    worker_turns: Optional[int] = None
    tool_calls: int = 0


def numbers_in(text: str) -> list:
    return [float(m.group().replace(",", "")) for m in _NUMBER.finditer(text or "")]


def hebrew_share(text: str) -> float:
    letters = [c for c in text or "" if c.isalpha()]
    return sum(bool(_HEBREW.match(c)) for c in letters) / len(letters) if letters else 0.0


def judge_rubrics(pred: Dict[str, Any]) -> list:
    """All judge rubrics inside a predicate, in order."""
    if "judge" in pred:
        return [pred["judge"]]
    subs = pred.get("all") or pred.get("any") or ([pred["not"]] if "not" in pred else [])
    return [r for p in subs for r in judge_rubrics(p)]


def evaluate(pred: Dict[str, Any], run: RunView) -> Outcome:
    if "all" in pred:
        results = [evaluate(p, run) for p in pred["all"]]
        if False in results:
            return False
        return JUDGE if JUDGE in results else True
    if "any" in pred:
        results = [evaluate(p, run) for p in pred["any"]]
        if True in results:
            return True
        return JUDGE if JUDGE in results else False
    if "not" in pred:
        r = evaluate(pred["not"], run)
        return r if r == JUDGE else not r
    if "judge" in pred:
        return JUDGE
    if "contains_amount" in pred:
        target, tol = float(pred["contains_amount"]), float(pred.get("tol", 0.01))
        return any(abs(n - target) <= tol for n in numbers_in(run.answer))
    if "contains_any" in pred:
        low = (run.answer or "").lower()
        return any(s.lower() in low for s in pred["contains_any"])
    if "contains_all" in pred:
        low = (run.answer or "").lower()
        return all(s.lower() in low for s in pred["contains_all"])
    if "refused" in pred:
        return run.refused == bool(pred["refused"])
    if "language" in pred:
        if pred["language"] != "he":
            raise ValueError(f"unsupported language {pred['language']!r}")
        return hebrew_share(run.answer) >= 0.8
    if "max_words" in pred:
        return len((run.answer or "").split()) <= int(pred["max_words"])
    if "no_dispatch" in pred:
        dispatched = run.worker_turns if run.config == "team" else run.tool_calls
        return (dispatched == 0) == bool(pred["no_dispatch"])
    raise ValueError(f"unknown predicate {pred!r}")
