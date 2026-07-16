"""Tests for bench.py — the pass-rate harness. No model calls; the agent-running part
is exercised only through its pure helpers and the self-checking task fixtures."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_parse_signals():
    out = (
        "● read_file(mathx.py)\n"
        "  ⎿ def factorial…\n"
        "● edit_file(mathx.py)\n"
        "  ⎿ Edited mathx.py — 1 replacement.\n"
        "  (cut off: runaway repetition)\n"
        "  (nudge: narrated an action without calling a tool)\n"
        "● run_bash(python -m pytest -q)\n"
        "  ⎿ Wrote backup.py (3 bytes).\n"
        "● stopped: no progress in 3 turns\n"
    )
    s = bench.parse_signals(out)
    assert s["tools"] == 4 and s["edited"] == 1 and s["wrote"] == 1
    assert s["cuts"] == 1 and s["nudges"] == 1 and s["stalls"] == 1


def test_aggregate_rates():
    results = {
        "a": [{"passed": True, "duration": 2.0, "timed_out": False, "signals": _z(tools=3)},
              {"passed": False, "duration": 4.0, "timed_out": False, "signals": _z(tools=1, cuts=2)}],
        "b": [{"passed": True, "duration": 6.0, "timed_out": True, "signals": _z()}],
    }
    s = bench.aggregate(results)
    assert s["tasks"]["a"]["rate"] == 0.5 and s["tasks"]["a"]["signals"]["cuts"] == 2
    assert s["tasks"]["a"]["avg_duration"] == 3.0
    assert s["tasks"]["b"]["timeouts"] == 1
    assert s["overall"]["runs"] == 3 and s["overall"]["passed"] == 2
    assert abs(s["overall"]["rate"] - 2 / 3) < 1e-9


def test_bar():
    assert bench._bar(0.0, 10) == "░" * 10
    assert bench._bar(1.0, 10) == "█" * 10
    assert bench._bar(0.5, 10) == "█████░░░░░"


@pytest.mark.parametrize("name", list(bench.TASKS))
def test_unfixed_task_fails(tmp_path, name):
    # Sanity: each task's seed (plus its hidden grading test) must FAIL before any fix —
    # otherwise the task measures nothing.
    task = bench.TASKS[name]
    for rel, content in {**task["files"], **task.get("hidden_files", {})}.items():
        (tmp_path / rel).write_text(content)
    assert bench.check_pass(task["check"], str(tmp_path)) is False


def test_every_task_has_a_tier():
    assert all(t["tier"] in bench.TIERS for t in bench.TASKS.values())


def test_select_tasks():
    assert bench.select_tasks(None, "easy") == ["factorial", "max_of", "fizzbuzz"]
    assert bench.select_tasks(None, "hard") == ["palindrome", "money"]
    assert set(bench.select_tasks(None, "all")) == set(bench.TASKS)
    assert set(bench.select_tasks("all", "easy")) == set(bench.TASKS)      # --tasks all wins
    assert bench.select_tasks("factorial,money", "easy") == ["factorial", "money"]


def test_hidden_task_solved_passes(tmp_path):
    # The reference solution for a hard task must pass its hidden test.
    (tmp_path / "strutil.py").write_text(
        "def is_palindrome(s):\n    t = s.replace(' ', '').lower()\n    return t == t[::-1]\n")
    for rel, content in bench.TASKS["palindrome"].get("hidden_files", {}).items():
        (tmp_path / rel).write_text(content)
    assert bench.check_pass(bench.TASKS["palindrome"]["check"], str(tmp_path)) is True


def test_check_pass_true_on_green(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    assert bench.check_pass(["python", "-m", "pytest", "-q"], str(tmp_path)) is True


def _z(**kw):
    base = {"tools": 0, "edited": 0, "wrote": 0, "cuts": 0, "nudges": 0, "stalls": 0}
    base.update(kw)
    return base
