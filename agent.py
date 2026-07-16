#!/usr/bin/env python3
"""A minimal coding-agent harness tuned for deepseek-v4-flash.

The stock model, under a generic harness (opencode), showed three failure modes:
  1. It loops re-reading the same file instead of acting.
  2. It claims "tests pass / code works" without ever running anything.
  3. It pads endings with repeated filler ("Goodbye.END Goodbye.END...").

This harness adapts to those:
  * a loop guard that short-circuits repeated/duplicate tool calls,
  * a system prompt + a soft check that pushes "run before you claim",
  * trailing-filler stripping,
  * deterministic sampling (temp 0) with reasoning_effort=low.

Usage:
  CODING_API_BASE=http://127.0.0.1:8000/v1 CODING_API_KEY=sk-... \
      python3 agent.py "your task"   # runs in the current directory
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = os.getenv("CODING_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
KEY = os.getenv("CODING_API_KEY", "")
MODEL = os.getenv("CODING_API_MODEL", "deepseek-v4-flash")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "16"))
WORKDIR = Path.cwd()

SYSTEM = """You are a focused coding agent working in the current directory.
Rules you MUST follow:
1. Be concise. Take one concrete action at a time.
2. NEVER claim code works, compiles, or that tests pass unless you actually ran it
   with run_bash in THIS session and saw the output. If you wrote or changed code,
   run it before concluding.
3. Do NOT read the same file more than once — you already have its content.
4. When the task is done, give ONE short final sentence and STOP. No goodbyes, no
   repetition, no filler.
Use the provided tools. Reason briefly, then act."""

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command in the working dir and return its output.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List files in a directory (default '.').",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]


def _api(messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": 2048, "reasoning_effort": "low", "temperature": 0}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "X-Thinking": "low"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]


# ---- tool implementations -------------------------------------------------
_read_hashes: dict[str, str] = {}   # path -> content hash, for the loop guard


def _safe(path: str) -> Path:
    p = (WORKDIR / path).resolve()
    return p


def do_read(args):
    p = _safe(args["path"])
    if not p.exists():
        return f"ERROR: {args['path']} does not exist."
    text = p.read_text(errors="replace")
    h = hashlib.sha1(text.encode()).hexdigest()
    if _read_hashes.get(str(p)) == h:
        # Loop guard: already read this exact content.
        return (f"(You already read {args['path']} earlier and it is UNCHANGED. Its content "
                f"is below — do NOT read it again; act on it or give your final answer.)\n{text}")
    _read_hashes[str(p)] = h
    return text


def do_write(args):
    p = _safe(args["path"]); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    _read_hashes.pop(str(p), None)  # content changed; allow a fresh read
    return f"Wrote {args['path']} ({len(args['content'])} bytes)."


# --- run_bash safety guard -------------------------------------------------
# NOT a real sandbox: it blocks casual/accidental damage from model output, but an
# allow-listed interpreter (python/node) can still do anything. Run genuinely
# untrusted tasks inside a container/VM. Commands execute only in WORKDIR.
# Set AGENT_ALLOW_ANY=1 to disable the guard (throwaway/sandboxed envs only).
_ALLOW = {"python", "python3", "pytest", "node", "ls", "cat", "head", "tail", "grep",
          "rg", "find", "wc", "diff", "sort", "uniq", "echo", "printf", "pwd", "stat",
          "file", "which", "true", "sed", "awk", "env", "date"}
_DENY = re.compile(r"""(?ix)
    \b(sudo|rm|rmdir|dd|mkfs|shutdown|reboot|halt|kill|pkill|killall|launchctl|brew|
       apt|yum|pip|curl|wget|nc|ncat|ssh|scp|sftp|telnet|chmod|chown|mount|umount|
       crontab|osascript|defaults|softwareupdate|systemsetup|git)\b
    | :\(\)\s*\{                 # fork bomb
    | \brm\s+-\w*r               # recursive delete
    | [>][>]?\s*/                # redirect into an absolute path
    | (^|\s)~/                   # reaching into $HOME
    | \.\.(/|\s|$)               # parent-directory escape
""")


def do_bash(args):
    cmd = (args.get("cmd") or "").strip()
    if os.getenv("AGENT_ALLOW_ANY") != "1":
        if _DENY.search(cmd):
            return ("REFUSED by the agent safety guard: the command looks destructive, networked, "
                    "privileged, or escapes the project directory. Only run/inspect code here.")
        first = (cmd.split() or [""])[0].split("/")[-1]
        if first not in _ALLOW:
            return (f"REFUSED by the agent safety guard: '{first}' is not allowed. Use write_file to "
                    f"create files; use run_bash only for running/inspecting code "
                    f"(python3, pytest, ls, cat, grep, ...).")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(WORKDIR), capture_output=True,
                           text=True, timeout=60)
        out = (r.stdout + r.stderr).strip()
        return f"(exit {r.returncode})\n{out[:4000]}" if out else f"(exit {r.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out (60s)."


def do_ls(args):
    p = _safe(args.get("path") or ".")
    if not p.exists():
        return f"ERROR: {p} not found."
    return "\n".join(sorted(x.name for x in p.iterdir())) or "(empty)"


DISPATCH = {"read_file": do_read, "write_file": do_write, "run_bash": do_bash, "list_dir": do_ls}


def _strip_filler(text: str) -> str:
    """Cut the model's trailing goodbye/END padding."""
    text = re.split(r"\b(?:Goodbye|GOODBYE|END)\b", text)[0]
    # collapse repeated trailing "Done." / "OK." style lines
    text = re.sub(r"(?:\s*(?:Done|OK|No further (?:needs|questions)|I'm satisfied)\.?){2,}\s*$", "", text.strip())
    return text.strip()


def run(task: str):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    recent_sigs: list[str] = []
    for step in range(1, MAX_STEPS + 1):
        ch = _api(messages)
        msg = ch["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            print(f"\n\033[1m● final answer\033[0m\n{_strip_filler(msg.get('content') or '')}")
            return
        # assistant turn (with tool_calls) goes back into the history
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        for tc in tcs:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except ValueError:
                # tolerate malformed tool JSON instead of crashing the loop
                result = f"ERROR: your tool arguments were not valid JSON: {raw[:120]}"
                print(f"  step {step}: {name}  \033[91m[bad JSON args]\033[0m")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            print(f"  step {step}: {name}({', '.join(f'{k}={str(v)[:30]!r}' for k,v in args.items())})")
            # generic loop guard: same tool+args 3x in a row -> intervene
            recent_sigs.append(sig)
            if recent_sigs[-3:].count(sig) >= 3:
                result = ("LOOP DETECTED: you have issued this exact tool call repeatedly. "
                          "Stop. Use what you already have and give your final answer now.")
            else:
                try:
                    result = DISPATCH[name](args)
                except Exception as e:  # noqa: BLE001
                    result = f"ERROR: {e}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    print("\n\033[93m● stopped: hit max steps\033[0m")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 agent.py \"<task>\"", file=sys.stderr); sys.exit(1)
    if not KEY:
        print("Set CODING_API_KEY (and optionally CODING_API_BASE, CODING_API_MODEL).", file=sys.stderr); sys.exit(1)
    print(f"\033[1m▶ task:\033[0m {sys.argv[1]}\n\033[1m▶ dir:\033[0m {WORKDIR}\n")
    run(sys.argv[1])
