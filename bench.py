#!/usr/bin/env python3
"""bench.py — measure the SDSZcode harness's pass rate on self-checking tasks.

Each task seeds a fresh temp project and a check (a pytest that passes only when the
task is done). bench runs the agent (one-shot, auto-approve) N times per task and
reports how often it succeeds, plus timing and guard-signal stats — so parameter
changes (thinking level, nudges, …) can be compared with numbers instead of vibes.

Tasks come in three tiers:
  easy   — single file, single-function bug, a failing test points right at it.
  medium — 2+ files, the bug is NOT where the test fails; must trace across files.
  hard   — no visible test (implement to a spec) or a coordinated multi-file change;
           validated by a HIDDEN test dropped in only after the agent finishes.

    python3 bench.py                          # easy tier, 5 runs each
    python3 bench.py --tier all --runs 10
    python3 bench.py --tier hard --runs 5
    CODING_API_THINKING=high python3 bench.py --tier medium   # A/B a setting

Needs CODING_API_KEY in the env (or --key), and a reachable backend (CODING_API_BASE).
Standard library only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTEST = ["python", "-m", "pytest", "-q"]

# Each task: tier, seed `files` (with a planted bug or a stub), a `prompt`, a `check`
# whose exit 0 means "done", and optional `hidden_files` written only AFTER the agent
# runs (so hard tasks can hide the grading test from the model).
TASKS = {
    # ---- easy: single file, the failing test points straight at the bug ----
    "factorial": {
        "tier": "easy",
        "files": {
            "mathx.py": "def factorial(n):\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result\n",
            "test_mathx.py": "from mathx import factorial\n\n\ndef test_factorial():\n    assert factorial(5) == 120\n    assert factorial(0) == 1\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in mathx.py, "
                  "fix it with edit_file, then re-run pytest to confirm. Be concise.",
        "check": PYTEST,
    },
    "max_of": {
        "tier": "easy",
        "files": {
            "listutil.py": "def max_of(nums):\n    m = nums[0]\n    for x in nums:\n        if x < m:\n            m = x\n    return m\n",
            "test_listutil.py": "from listutil import max_of\n\n\ndef test_max_of():\n    assert max_of([3, 7, 2, 9, 4]) == 9\n    assert max_of([-1, -5, -2]) == -1\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in listutil.py, "
                  "fix it with edit_file, then re-run pytest to confirm. Be concise.",
        "check": PYTEST,
    },
    "fizzbuzz": {
        "tier": "easy",
        "files": {
            "fb.py": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append(\"FizzBuzz\")\n        elif i % 3 == 0:\n            out.append(\"Fizz\")\n        elif i % 5 == 0:\n            out.append(\"Buzz\")\n        else:\n            out.append(i)\n    return out\n",
            "test_fb.py": "from fb import fizzbuzz\n\n\ndef test_fizzbuzz():\n    assert fizzbuzz(5) == [\"1\", \"2\", \"Fizz\", \"4\", \"Buzz\"]\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in fb.py, "
                  "fix it with edit_file, then re-run pytest to confirm. Be concise.",
        "check": PYTEST,
    },

    # ---- medium: the bug is in a file the test does NOT import directly ----
    "cart": {
        "tier": "medium",
        "files": {
            "store.py": "from pricing import unit_price\n\n\ndef cart_total(items):\n    return sum(unit_price(name) * qty for name, qty in items)\n",
            "pricing.py": "PRICES = {\"apple\": 3, \"banana\": 2, \"cherry\": 5}\n\n\ndef unit_price(name):\n    return PRICES.get(name, 0) - 1\n",
            "test_store.py": "from store import cart_total\n\n\ndef test_total():\n    assert cart_total([(\"apple\", 2), (\"banana\", 3)]) == 12\n",
        },
        "prompt": "The test fails. Run `python -m pytest -q`, then TRACE the failure — the bug "
                  "is not in store.py (where the test points) but in a file it depends on. Find "
                  "and fix the real cause, then re-run pytest to confirm. Be concise.",
        "check": PYTEST,
    },
    "geometry": {
        "tier": "medium",
        "files": {
            "geometry.py": "from shapes import area\n\n\ndef total_area(shapes):\n    return sum(area(s) for s in shapes)\n",
            "shapes.py": "import math\n\n\ndef area(shape):\n    kind, size = shape\n    if kind == \"square\":\n        return size * size\n    if kind == \"circle\":\n        return math.pi * size\n    return 0\n",
            "test_geometry.py": "from geometry import total_area\n\n\ndef test_area():\n    assert total_area([(\"square\", 2)]) == 4\n    assert abs(total_area([(\"circle\", 3)]) - 28.274333882308138) < 1e-6\n",
        },
        "prompt": "The test fails. Run `python -m pytest -q`, then trace it — the failure surfaces "
                  "in geometry.py but the bug is in the module it calls. Fix the real cause and "
                  "re-run pytest to confirm. Be concise.",
        "check": PYTEST,
    },

    # ---- hard: no visible test / coordinated change; graded by a hidden test ----
    "palindrome": {
        "tier": "hard",
        "files": {
            "strutil.py": "def is_palindrome(s):\n    pass\n",
        },
        "prompt": "Implement is_palindrome(s) in strutil.py so it returns True iff s reads the "
                  "same forwards and backwards, IGNORING letter case and spaces "
                  "(e.g. 'Race car' -> True, 'hello' -> False, '' -> True). There is no test yet: "
                  "write one yourself, then run `python -m pytest -q` to confirm your implementation.",
        "check": PYTEST,
        "hidden_files": {
            "test_hidden_palindrome.py": "from strutil import is_palindrome\n\n\ndef test_palindrome():\n    assert is_palindrome(\"Race car\") is True\n    assert is_palindrome(\"hello\") is False\n    assert is_palindrome(\"\") is True\n    assert is_palindrome(\"Was it a car or a cat I saw\") is True\n",
        },
    },
    "money": {
        "tier": "hard",
        "files": {
            "money.py": "def fmt(amount):\n    return f\"${amount:.2f}\"\n",
            "report.py": "from money import fmt\n\n\ndef line(label, amount):\n    return f\"{label}: {fmt(amount)}\"\n",
        },
        "prompt": "Make a coordinated change across two files: add an optional `places` parameter "
                  "to fmt(amount, places=2) in money.py that controls the number of decimal places "
                  "(keeping current behavior when omitted), AND update report.py's line() to format "
                  "amounts with 4 decimal places. Then run `python -m pytest -q` to confirm.",
        "check": PYTEST,
        "hidden_files": {
            "test_hidden_money.py": "from money import fmt\nfrom report import line\n\n\ndef test_fmt_default():\n    assert fmt(3.5) == \"$3.50\"\n\n\ndef test_fmt_places():\n    assert fmt(3.5, 4) == \"$3.5000\"\n\n\ndef test_line_four_places():\n    assert line(\"x\", 3.5) == \"x: $3.5000\"\n",
        },
    },
}

TIERS = ("easy", "medium", "hard")


def _agent_cmd(workdir: str, steps: int, prompt: str) -> list[str]:
    """Prefer the installed `sdszcode` command; fall back to running agent.py directly."""
    base = ["sdszcode"] if shutil.which("sdszcode") else [sys.executable, str(HERE / "agent.py")]
    return base + ["-y", "-C", workdir, "-s", str(steps), prompt]


def parse_signals(out: str) -> dict:
    """Count guard/activity signals from the agent's stdout."""
    lines = out.splitlines()
    return {
        "tools": sum(1 for l in lines if l.startswith("● ")),
        "edited": out.count("⎿ Edited"),
        "wrote": out.count("⎿ Wrote"),
        "cuts": out.count("cut off: runaway"),
        "nudges": out.count("nudge:"),
        "stalls": out.count("no progress in"),
    }


def check_pass(check: list[str], workdir: str) -> bool:
    try:
        r = subprocess.run(check, cwd=workdir, capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def run_trial(name: str, task: dict, steps: int, timeout: int, env: dict) -> dict:
    """Seed a fresh project, run the agent once, drop any hidden test, and grade it.
    Returns {passed, duration, timed_out, signals}."""
    import tempfile

    d = tempfile.mkdtemp(prefix=f"bench_{name}_")
    try:
        for rel, content in task["files"].items():
            (Path(d) / rel).write_text(content)
        t0 = time.time()
        timed_out = False
        try:
            r = subprocess.run(_agent_cmd(d, steps, task["prompt"]),
                               capture_output=True, text=True, timeout=timeout, env=env)
            out = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
            so = e.stdout
            out = so.decode("utf-8", "replace") if isinstance(so, bytes) else (so or "")
        dur = time.time() - t0
        # Grade: hidden files are written only now, so the model never saw them.
        for rel, content in task.get("hidden_files", {}).items():
            (Path(d) / rel).write_text(content)
        passed = (not timed_out) and check_pass(task["check"], d)
        return {"passed": passed, "duration": dur, "timed_out": timed_out, "signals": parse_signals(out)}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def aggregate(results: dict) -> dict:
    """Turn per-task lists of trial dicts into a summary (pure — unit-testable)."""
    summary: dict = {"tasks": {}, "overall": {}}
    all_trials = []
    for name, trials in results.items():
        all_trials.extend(trials)
        n = len(trials)
        passed = sum(1 for t in trials if t["passed"])
        avg_dur = sum(t["duration"] for t in trials) / n if n else 0.0
        sig_tot = {k: sum(t["signals"][k] for t in trials) for k in
                   ("tools", "edited", "wrote", "cuts", "nudges", "stalls")}
        summary["tasks"][name] = {"tier": TASKS.get(name, {}).get("tier", "?"),
                                  "runs": n, "passed": passed,
                                  "rate": passed / n if n else 0.0,
                                  "avg_duration": avg_dur, "signals": sig_tot,
                                  "timeouts": sum(1 for t in trials if t["timed_out"])}
    n = len(all_trials)
    passed = sum(1 for t in all_trials if t["passed"])
    summary["overall"] = {"runs": n, "passed": passed, "rate": passed / n if n else 0.0,
                          "avg_duration": sum(t["duration"] for t in all_trials) / n if n else 0.0}
    return summary


def _bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 68)
    print(f"{'task':<12} {'tier':<7} {'pass':>7}  {'rate':>6}  {'avg s':>6}  signals")
    print("-" * 68)
    for name, s in summary["tasks"].items():
        sig = s["signals"]
        extra = f"tools={sig['tools']} edit={sig['edited']} cut={sig['cuts']} nudge={sig['nudges']} stall={sig['stalls']}"
        to = f" TIMEOUTS={s['timeouts']}" if s["timeouts"] else ""
        print(f"{name:<12} {s['tier']:<7} {s['passed']:>3}/{s['runs']:<3} {s['rate']*100:>5.0f}%  "
              f"{s['avg_duration']:>6.1f}  {extra}{to}")
    o = summary["overall"]
    print("-" * 68)
    print(f"{'OVERALL':<12} {'':<7} {o['passed']:>3}/{o['runs']:<3} {o['rate']*100:>5.0f}%  "
          f"{o['avg_duration']:>6.1f}  {_bar(o['rate'])}")
    print("=" * 68)


def select_tasks(tasks_arg, tier_arg) -> list[str]:
    """Resolve which task names to run from --tasks / --tier."""
    if tasks_arg and tasks_arg != "all":
        return [t.strip() for t in tasks_arg.split(",") if t.strip()]
    if tasks_arg == "all" or tier_arg == "all":
        return list(TASKS)
    return [n for n, t in TASKS.items() if t["tier"] == tier_arg]


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench", description="Pass-rate benchmark for the SDSZcode harness.")
    p.add_argument("--runs", type=int, default=5, help="trials per task (default 5)")
    p.add_argument("--tier", choices=(*TIERS, "all"), default="easy",
                   help="difficulty tier to run (default easy)")
    p.add_argument("--tasks", help="explicit comma-separated task names (overrides --tier), or 'all'")
    p.add_argument("--steps", type=int, default=14, help="max tool-call rounds per trial (default 14)")
    p.add_argument("--timeout", type=int, default=240, help="per-trial timeout in seconds (default 240)")
    p.add_argument("--key", help="API key (else CODING_API_KEY)")
    p.add_argument("--base", help="API base URL (else CODING_API_BASE)")
    p.add_argument("--model", help="model id (else CODING_API_MODEL)")
    p.add_argument("--thinking", help="reasoning depth for the trials (sets CODING_API_THINKING)")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    p.add_argument("--list", action="store_true", help="list tasks with their tier and exit")
    args = p.parse_args(argv)

    if args.list:
        for name, t in TASKS.items():
            print(f"{name:<12} {t['tier']}")
        return 0

    names = select_tasks(args.tasks, args.tier)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        print(f"unknown task(s): {', '.join(unknown)} — available: {', '.join(TASKS)}", file=sys.stderr)
        return 2
    if not names:
        print("no tasks selected", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env.setdefault("NO_PROXY", "*"); env.setdefault("no_proxy", "*")
    if args.key:
        env["CODING_API_KEY"] = args.key
    if args.base:
        env["CODING_API_BASE"] = args.base
    if args.model:
        env["CODING_API_MODEL"] = args.model
    if args.thinking:
        env["CODING_API_THINKING"] = args.thinking
    if not env.get("CODING_API_KEY"):
        print("No API key — set CODING_API_KEY or pass --key.", file=sys.stderr)
        return 1

    total = len(names) * args.runs
    print(f"Running {total} trials  ({len(names)} task(s) × {args.runs} runs, steps={args.steps}, "
          f"timeout={args.timeout}s, thinking={env.get('CODING_API_THINKING', 'low')})")
    results = {}
    done = 0
    for name in names:
        results[name] = []
        for i in range(args.runs):
            done += 1
            tier = TASKS[name]["tier"]
            print(f"  [{done}/{total}] {name} ({tier}) run {i + 1}… ", end="", flush=True)
            r = run_trial(name, TASKS[name], args.steps, args.timeout, env)
            mark = "PASS" if r["passed"] else ("TIMEOUT" if r["timed_out"] else "fail")
            print(f"{mark}  {r['duration']:.1f}s  (tools={r['signals']['tools']}, "
                  f"cut={r['signals']['cuts']}, nudge={r['signals']['nudges']})")
            results[name].append(r)

    summary = aggregate(results)
    if args.json:
        import json
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
