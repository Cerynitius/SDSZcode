#!/usr/bin/env python3
"""bench.py — measure the SDSZcode harness's pass rate on self-checking bug-fix tasks.

Each task seeds a fresh temp project with a planted bug and a pytest that fails until
it's fixed. bench runs the agent (one-shot, auto-approve) N times per task and reports
how often the tests end up green, plus timing and guard-signal stats — so parameter
changes (thinking level, nudges, etc.) can be compared with numbers instead of vibes.

    python3 bench.py                       # all tasks, 5 runs each
    python3 bench.py --runs 10 --tasks factorial
    CODING_API_THINKING=high python3 bench.py   # A/B a setting via env

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

# Each task: a set of files (with a planted bug), a prompt, and a shell check whose
# exit code 0 means "fixed". Keep them tiny and deterministic.
TASKS = {
    "factorial": {
        "files": {
            "mathx.py": "def factorial(n):\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result\n",
            "test_mathx.py": "from mathx import factorial\n\n\ndef test_factorial():\n    assert factorial(5) == 120\n    assert factorial(0) == 1\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in mathx.py, "
                  "fix it with edit_file, then re-run pytest to confirm all tests pass. Be concise.",
        "check": ["python", "-m", "pytest", "-q"],
    },
    "max_of": {
        "files": {
            "listutil.py": "def max_of(nums):\n    m = nums[0]\n    for x in nums:\n        if x < m:\n            m = x\n    return m\n",
            "test_listutil.py": "from listutil import max_of\n\n\ndef test_max_of():\n    assert max_of([3, 7, 2, 9, 4]) == 9\n    assert max_of([-1, -5, -2]) == -1\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in listutil.py, "
                  "fix it with edit_file, then re-run pytest to confirm. Be concise.",
        "check": ["python", "-m", "pytest", "-q"],
    },
    "fizzbuzz": {
        "files": {
            "fb.py": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append(\"FizzBuzz\")\n        elif i % 3 == 0:\n            out.append(\"Fizz\")\n        elif i % 5 == 0:\n            out.append(\"Buzz\")\n        else:\n            out.append(i)\n    return out\n",
            "test_fb.py": "from fb import fizzbuzz\n\n\ndef test_fizzbuzz():\n    assert fizzbuzz(5) == [\"1\", \"2\", \"Fizz\", \"4\", \"Buzz\"]\n",
        },
        "prompt": "The tests fail. Run `python -m pytest -q`, find the bug in fb.py, "
                  "fix it with edit_file, then re-run pytest to confirm. Be concise.",
        "check": ["python", "-m", "pytest", "-q"],
    },
}


def _agent_cmd(workdir: str, steps: int, prompt: str) -> list[str]:
    """Prefer the installed `sdszcode` command; fall back to running agent.py directly."""
    if shutil.which("sdszcode"):
        base = ["sdszcode"]
    else:
        base = [sys.executable, str(HERE / "agent.py")]
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
    """Seed a fresh project, run the agent once, return {passed, duration, timed_out, signals}."""
    import tempfile

    d = tempfile.mkdtemp(prefix=f"bench_{name}_")
    try:
        for rel, content in task["files"].items():
            (Path(d) / rel).write_text(content)
        t0 = time.time()
        timed_out = False
        out = ""
        try:
            r = subprocess.run(_agent_cmd(d, steps, task["prompt"]),
                               capture_output=True, text=True, timeout=timeout, env=env)
            out = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        dur = time.time() - t0
        passed = (not timed_out) and check_pass(task["check"], d)
        sig = parse_signals(out)
        return {"passed": passed, "duration": dur, "timed_out": timed_out, "signals": sig}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def aggregate(results: dict) -> dict:
    """Turn per-task lists of trial dicts into a summary (pure — unit-testable)."""
    summary = {"tasks": {}, "overall": {}}
    all_trials = []
    for name, trials in results.items():
        all_trials.extend(trials)
        n = len(trials)
        passed = sum(1 for t in trials if t["passed"])
        avg_dur = sum(t["duration"] for t in trials) / n if n else 0.0
        sig_tot = {k: sum(t["signals"][k] for t in trials) for k in
                   ("tools", "edited", "wrote", "cuts", "nudges", "stalls")}
        summary["tasks"][name] = {"runs": n, "passed": passed,
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
    print("\n" + "=" * 64)
    print(f"{'task':<12} {'pass':>7}  {'rate':>6}  {'avg s':>6}  signals")
    print("-" * 64)
    for name, s in summary["tasks"].items():
        sig = s["signals"]
        extra = f"tools={sig['tools']} edit={sig['edited']} cut={sig['cuts']} nudge={sig['nudges']} stall={sig['stalls']}"
        to = f" TIMEOUTS={s['timeouts']}" if s["timeouts"] else ""
        print(f"{name:<12} {s['passed']:>3}/{s['runs']:<3} {s['rate']*100:>5.0f}%  "
              f"{s['avg_duration']:>6.1f}  {extra}{to}")
    o = summary["overall"]
    print("-" * 64)
    print(f"OVERALL      {o['passed']:>3}/{o['runs']:<3} {o['rate']*100:>5.0f}%  {o['avg_duration']:>6.1f}  "
          f"{_bar(o['rate'])}")
    print("=" * 64)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench", description="Pass-rate benchmark for the SDSZcode harness.")
    p.add_argument("--runs", type=int, default=5, help="trials per task (default 5)")
    p.add_argument("--tasks", default="all",
                   help=f"comma-separated subset, or 'all' (available: {', '.join(TASKS)})")
    p.add_argument("--steps", type=int, default=12, help="max tool-call rounds per trial (default 12)")
    p.add_argument("--timeout", type=int, default=240, help="per-trial timeout in seconds (default 240)")
    p.add_argument("--key", help="API key (else CODING_API_KEY)")
    p.add_argument("--base", help="API base URL (else CODING_API_BASE)")
    p.add_argument("--model", help="model id (else CODING_API_MODEL)")
    p.add_argument("--thinking", help="reasoning depth for the trials (sets CODING_API_THINKING)")
    p.add_argument("--json", action="store_true", help="emit the summary as JSON")
    p.add_argument("--list", action="store_true", help="list task names and exit")
    args = p.parse_args(argv)

    if args.list:
        for name in TASKS:
            print(name)
        return 0

    names = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        print(f"unknown task(s): {', '.join(unknown)} — available: {', '.join(TASKS)}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["NO_COLOR"] = "1"                       # clean, parseable output
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
    print(f"Running {total} trials  ({len(names)} task(s) × {args.runs} runs, "
          f"steps={args.steps}, timeout={args.timeout}s, thinking={env.get('CODING_API_THINKING', 'low')})")
    results = {}
    done = 0
    for name in names:
        results[name] = []
        for i in range(args.runs):
            done += 1
            print(f"  [{done}/{total}] {name} run {i + 1}… ", end="", flush=True)
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
