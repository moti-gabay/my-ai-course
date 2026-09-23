"""Offline tests for scoring.py predicates.  Run: .venv/bin/python tests/test_scoring.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import JUDGE, RunView, evaluate, judge_rubrics  # noqa: E402


def run(answer="", refused=False, config="team", worker_turns=1, tool_calls=1):
    return RunView(answer=answer, refused=refused, config=config, worker_turns=worker_turns, tool_calls=tool_calls)


def test_contains_amount_formats():
    for text in ["$12,980.00", "12980", "total: 12,980", "₪12,980.004"]:
        assert evaluate({"contains_amount": 12980}, run(text)), text
    assert not evaluate({"contains_amount": 12980}, run("12,890"))
    assert not evaluate({"contains_amount": 250}, run("$2,500"))
    assert evaluate({"contains_amount": 5180.2}, run("₪5,180.20"))


def test_hebrew_with_number():
    he = "הפוליסה משלמת עד $250 עבור ערבות בעקבות תאונה מכוסה."
    assert evaluate({"all": [{"language": "he"}, {"contains_amount": 250}, {"max_words": 30}]}, run(he))
    assert not evaluate({"language": "he"}, run("The policy pays up to $250."))


def test_max_words():
    assert evaluate({"max_words": 3}, run("one two three"))
    assert not evaluate({"max_words": 3}, run("one two three four"))


def test_refused_reads_the_structured_flag_only():
    assert evaluate({"refused": True}, run("any text at all", refused=True))
    assert not evaluate({"refused": True}, run("I cannot answer this, information missing", refused=False))


def test_no_dispatch_per_config():
    assert evaluate({"no_dispatch": True}, run(config="team", worker_turns=0, tool_calls=0))
    assert not evaluate({"no_dispatch": True}, run(config="team", worker_turns=1))
    assert evaluate({"no_dispatch": True}, run(config="single", worker_turns=None, tool_calls=0))
    assert not evaluate({"no_dispatch": True}, run(config="single", worker_turns=None, tool_calls=2))


def test_combinators_and_judge():
    p = {"all": [{"contains_amount": 250}, {"judge": "mentions the covered-accident condition"}]}
    assert evaluate(p, run("$250")) == JUDGE
    assert evaluate(p, run("$500")) is False          # code failure short-circuits the judge
    assert evaluate({"any": [{"refused": True}, {"judge": "x"}]}, run(refused=True)) is True
    assert evaluate({"not": {"contains_any": ["$"]}}, run("no dollars"))
    assert judge_rubrics(p) == ["mentions the covered-accident condition"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
