"""Safety-net unit tests. No API calls.  Run: .venv/bin/python tests/test_nets.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from team import SafetyNetChecker, detect_loop  # noqa: E402

R, A, W = "researcher", "analyst", "writer"


def test_ping_pong_is_a_loop():
    assert detect_loop([R, A, R, A])


def test_same_worker_four_times_is_a_loop():
    assert detect_loop([R, R, R, R])


def test_normal_routes_are_not_loops():
    for route in ([R, A, W], [R, R], [R, A, R], [R, R, A, W], [A, R, A, W]):
        assert not detect_loop(route), route


def test_old_bug_interleaved_orchestrator_hid_the_loop():
    # The old check compared the route with "orchestrator" between workers; that pair never repeats.
    o = "orchestrator"
    interleaved = [o, R, o, A, o, R, o, A]
    assert interleaved[-4:-2] != interleaved[-2:]
    assert detect_loop([x for x in interleaved if x != o])


def test_dispatch_breach_fires_before_the_looping_turn_runs():
    nets = SafetyNetChecker(max_turns=8)
    assert nets.dispatch_breach([R, A, R], A) == ("loop_detected", "repeating_agent_pair")
    assert nets.dispatch_breach([R, A], W) is None


def test_turn_cap():
    nets = SafetyNetChecker(max_turns=3)
    assert nets.dispatch_breach([R, A, W], R) == ("cap_breached", "max_turns")


def test_budget_nets():
    nets = SafetyNetChecker(max_tokens=1000, timeout_seconds=10)
    assert nets.budget_breach(999, time.time()) is None
    assert nets.budget_breach(1000, time.time()) == "token_budget"
    assert nets.budget_breach(0, time.time() - 11) == "wall_clock_timeout"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
