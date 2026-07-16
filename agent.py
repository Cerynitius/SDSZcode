#!/usr/bin/env python3
"""SDSZcode — a tiny coding-agent harness tuned for small / local coding models.

Adapts to the failure modes small models show in agent loops (looping on re-reads,
claiming success without running, filler padding, malformed tool JSON, flaky/
capacity-limited backends) and gives them a clean interactive terminal UI.

Interactive:   python3 agent.py
One-shot:      python3 agent.py "your task"

Config via env: CODING_API_BASE, CODING_API_KEY (required), CODING_API_MODEL,
AGENT_MAX_STEPS, AGENT_RETRIES, AGENT_MAX_BACKOFF, AGENT_ALLOW_ANY.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.getenv("CODING_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
KEY = os.getenv("CODING_API_KEY", "")
MODEL = os.getenv("CODING_API_MODEL", "deepseek-v4-flash")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "16"))
# Shared-GPU backends return 503 while busy and can stay saturated ~a minute, so be
# patient: ~8 retries with backoff capped at 15s.
RETRIES = int(os.getenv("AGENT_RETRIES", "8"))
MAX_BACKOFF = int(os.getenv("AGENT_MAX_BACKOFF", "15"))
WORKDIR = Path.cwd()

SYSTEM = """You are a focused coding agent working in the current directory.
Rules you MUST follow:
1. Be concise. Take one concrete action at a time.
2. NEVER claim code works, compiles, or that tests pass unless you actually ran it
   with run_bash in THIS session and saw the output. If you wrote or changed code,
   run it before concluding.
3. To change an EXISTING file, use edit_file (a precise search/replace) — do NOT
   rewrite the whole file with write_file. Use write_file only to create new files.
4. Do NOT read the same file more than once — you already have its content. Use grep
   to locate things instead of reading everything.
5. Ground EVERY change in the actual file contents you read and the actual command
   output you saw. NEVER invent functions, classes, tests, or error messages you have
   not literally seen — if unsure, read or grep first.
6. Run Python tests with `python -m pytest` (bare `pytest` often can't import a local
   package). Always look at the real output before deciding what to fix.
7. When the task is done, give ONE short final sentence and STOP. No goodbyes, no
   repetition, no filler.
Use the provided tools. Reason briefly, then act."""

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a whole file (use for NEW files).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file",
        "description": "Precisely edit an existing file by replacing an exact substring. old_string must "
                       "match the file exactly (including whitespace) and be unique unless replace_all is true. "
                       "Prefer this over rewriting the whole file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search files for a regex/text pattern (recursive).",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command in the working dir and return its output.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List files in a directory (default '.').",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]

# ------------------------------------------------------------------ transport
_RETRY_CODES = {429, 500, 502, 503, 504}
_HDRS = {"Content-Type": "application/json", "X-Thinking": "low"}


def _post(body, timeout=180, retries=None):
    """Non-streaming POST with backoff (used for the JSON-repair helper)."""
    retries = RETRIES if retries is None else retries
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(BASE + "/chat/completions", data=data,
            headers={"Authorization": f"Bearer {KEY}", **_HDRS}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_CODES and attempt < retries:
                _backoff(e.code, attempt, retries, e.headers.get("Retry-After"))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries:
                time.sleep(min(2 ** attempt, MAX_BACKOFF))
                continue
            raise
    raise RuntimeError("unreachable")


def _backoff(code, attempt, retries, retry_after=None):
    wait = min(2 ** attempt, MAX_BACKOFF)
    ra = (retry_after or "").strip()
    if ra.isdigit():
        wait = max(wait, int(ra))
    sys.stderr.write(f"    \033[38;5;240m(backend {code}; retry {attempt+1}/{retries} in {wait}s)\033[0m\n")
    time.sleep(wait)


def _turn(messages, on_delta=None):
    """One streaming model turn. Streams content via on_delta and returns
    (content, tool_calls, finish_reason). Retries the connection on 5xx/429."""
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": 2048, "reasoning_effort": "low", "temperature": 0, "stream": True}
    data = json.dumps(body).encode()
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(BASE + "/chat/completions", data=data,
            headers={"Authorization": f"Bearer {KEY}", **_HDRS}, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_CODES and attempt < RETRIES:
                _backoff(e.code, attempt, RETRIES, e.headers.get("Retry-After"))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < RETRIES:
                time.sleep(min(2 ** attempt, MAX_BACKOFF))
                continue
            raise
        return _consume_stream(resp, on_delta)
    raise RuntimeError("unreachable")


# Small models leak chat-template control tokens into `content` — deepseek uses
# full-width-pipe markers like <｜DSML｜tool_result｜> and fake </tool_result> blocks.
# Left in, they pollute the display AND feed back into history, compounding confusion.
_LEAK = re.compile(r"<[^<>]*｜[^<>]*>|</?\s*tool_(?:result|call)\s*>", re.I)


class _StreamCleaner:
    """Strips leaked control tokens from streamed content, holding back a partial
    marker across chunks (a ｜-token or a prefix of a tool_result/call tag) so it is
    never printed half-formed. Ordinary '<'/'>' in code stream through untouched."""

    _MARKERS = ("</tool_result>", "<tool_result>", "</tool_call>", "<tool_call>")

    def __init__(self):
        self.hold = ""

    def feed(self, text: str) -> str:
        s = _LEAK.sub("", self.hold + text)
        self.hold = ""
        m = re.search(r"<[^>]*$", s)  # an unclosed '<…' at the very end
        if m:
            tail = s[m.start():]
            if "｜" in tail or any(k.startswith(tail) for k in self._MARKERS):
                self.hold, s = tail, s[:m.start()]
        return s

    def flush(self) -> str:
        s, self.hold = _LEAK.sub("", self.hold), ""
        return s


def _consume_stream(resp, on_delta):
    content, calls, finish = "", {}, None
    cleaner = _StreamCleaner()
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                ch = json.loads(d)["choices"][0]
            except (ValueError, KeyError, IndexError):
                continue
            delta = ch.get("delta") or {}
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if delta.get("content"):
                clean = cleaner.feed(delta["content"])
                if clean:
                    content += clean
                    if on_delta:
                        on_delta(clean)
            for tc in (delta.get("tool_calls") or []):
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                if tc.get("id"):
                    c["id"] = tc["id"]
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["args"] += fn.get("arguments") or ""
    finally:
        tail = cleaner.flush()
        if tail:
            content += tail
            if on_delta:
                on_delta(tail)
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
    tool_calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                   "function": {"name": c["name"], "arguments": c["args"]}}
                  for i, c in sorted(calls.items())]
    return content, tool_calls, finish


# --------------------------------------------------------------- tool impls
_read_hashes: dict[str, str] = {}


def _safe(path: str) -> Path:
    return (WORKDIR / path).resolve()


def do_read(args):
    p = _safe(args["path"])
    if not p.exists():
        return f"ERROR: {args['path']} does not exist."
    text = p.read_text(errors="replace")
    h = hashlib.sha1(text.encode()).hexdigest()
    if _read_hashes.get(str(p)) == h:
        # Stronger loop guard: refuse the re-read; the content is already in history.
        return (f"You already read {args['path']} and it is unchanged — its content is above in "
                f"the conversation. Do NOT read it again; act on it or give your final answer.")
    _read_hashes[str(p)] = h
    return text


def do_write(args):
    p = _safe(args["path"]); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    _read_hashes.pop(str(p), None)
    return f"Wrote {args['path']} ({len(args['content'])} bytes)."


def do_edit(args):
    p = _safe(args["path"])
    if not p.exists():
        return f"ERROR: {args['path']} does not exist — use write_file to create it."
    old, new = args.get("old_string", ""), args.get("new_string", "")
    if not old:
        return "ERROR: old_string is required and must be non-empty."
    if old == new:
        return "ERROR: old_string and new_string are identical — nothing to change."
    text = p.read_text(errors="replace")
    n = text.count(old)
    if n == 0:
        # The model often gets here by hallucinating file contents — re-ground it by
        # showing the ACTUAL current file, so it stops guessing.
        preview = text if len(text) <= 2500 else text[:2500] + "\n…(truncated)"
        return (f"ERROR: old_string not found in {args['path']}. Do NOT guess its contents — the "
                f"ACTUAL current file is below; copy old_string from it verbatim (whitespace "
                f"included):\n---8<---\n{preview}\n---8<---")
    if n > 1 and not args.get("replace_all"):
        return (f"ERROR: old_string appears {n} times — add surrounding context to make it "
                f"unique, or set replace_all=true.")
    p.write_text(text.replace(old, new))
    _read_hashes.pop(str(p), None)
    return f"Edited {args['path']} — {n} replacement{'s' if n != 1 else ''}."


def do_grep(args):
    pat = args.get("pattern", "")
    if not pat:
        return "ERROR: pattern is required."
    target = _safe(args.get("path") or ".")
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--", pat, str(target)]
    else:
        cmd = ["grep", "-rn", "--", pat, str(target)]
    try:
        r = subprocess.run(cmd, cwd=str(WORKDIR), capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out."
    out = (r.stdout or "").strip()
    # trim absolute prefix for readability
    out = out.replace(str(WORKDIR) + os.sep, "")
    return out[:4000] if out else "(no matches)"


# run_bash safety guard — NOT a real sandbox (an allow-listed interpreter can still do
# anything). Run untrusted tasks in a container/VM. AGENT_ALLOW_ANY=1 disables it.
_ALLOW = {"python", "python3", "pytest", "node", "ls", "cat", "head", "tail", "grep",
          "rg", "find", "wc", "diff", "sort", "uniq", "echo", "printf", "pwd", "stat",
          "file", "which", "true", "sed", "awk", "env", "date"}
_DENY = re.compile(r"""(?ix)
    \b(sudo|rm|rmdir|dd|mkfs|shutdown|reboot|halt|kill|pkill|killall|launchctl|brew|
       apt|yum|pip|curl|wget|nc|ncat|ssh|scp|sftp|telnet|chmod|chown|mount|umount|
       crontab|osascript|defaults|softwareupdate|systemsetup|git)\b
    | :\(\)\s*\{ | \brm\s+-\w*r | [>][>]?\s*/ | (^|\s)~/ | \.\.(/|\s|$)
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
                    f"create files; use run_bash only for running/inspecting code (python3, pytest, ...).")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(WORKDIR), capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr).strip()
        return f"(exit {r.returncode})\n{out[:4000]}" if out else f"(exit {r.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out (60s)."


def do_ls(args):
    p = _safe(args.get("path") or ".")
    if not p.exists():
        return f"ERROR: {p} not found."
    return "\n".join(sorted(x.name for x in p.iterdir())) or "(empty)"


DISPATCH = {"read_file": do_read, "write_file": do_write, "edit_file": do_edit,
            "grep": do_grep, "run_bash": do_bash, "list_dir": do_ls}


# ------------------------------------------------------------------- helpers
def _strip_filler(text: str) -> str:
    """Cut the model's trailing goodbye/meta/repetition padding."""
    t = _LEAK.sub("", text).strip()
    t = re.split(
        r"\b(?:Goodbye|Bye|farewell|Terminate)\b|\bSECOND FINAL\b|"
        r"\bNo further (?:needs|questions|actions?|issues?)\b|"
        r"\n?\s*Final (?:message|brief|statement)\b|\bEND\b",
        t, maxsplit=1, flags=re.I)[0].strip()
    t = re.sub(r"(?:\s*(?:Done|OK|Okay|That's it|All set|I'm satisfied|Task (?:finished|complete[d]?))[.!]?){1,}\s*$",
               "", t, flags=re.I).strip()
    return t


def _extract_json(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_args(name: str, raw: str):
    """JSON retry: ask the model to re-emit valid JSON args. Returns dict or None."""
    msgs = [
        {"role": "system", "content": "Output ONLY one valid minified JSON object — no prose, no fences."},
        {"role": "user", "content": f"This was meant to be the JSON arguments for tool `{name}` but is "
                                     f"invalid JSON:\n{raw}\nReturn the corrected JSON object only."},
    ]
    try:
        txt = _post({"model": MODEL, "messages": msgs, "max_tokens": 1024,
                     "temperature": 0, "reasoning_effort": "low"}, timeout=120)["choices"][0]["message"].get("content") or ""
        js = _extract_json(txt)
        out = json.loads(js) if js else None
        return out if isinstance(out, dict) else None
    except Exception:  # noqa: BLE001
        return None


_SKIP = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".mypy_cache", ".idea"}


def project_map(root: Path | None = None, max_entries: int = 200) -> str:
    root = root or WORKDIR
    lines, count = [], 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if rel != ".":
            lines.append("  " * (depth - 1) + os.path.basename(dirpath) + "/")
        for f in sorted(files):
            if f in _SKIP or f.startswith("."):
                continue
            lines.append("  " * depth + f)
            count += 1
            if count >= max_entries:
                lines.append("  …(truncated)")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty directory)"


# ---------------------------------------------------------------- UI / loop
_ICON = {"read_file": "○", "write_file": "✚", "edit_file": "✎", "grep": "⌕",
         "run_bash": "▸", "list_dir": "☰"}
_C = {"brand": "173", "tool": "173", "arg": "240", "ok": "108", "bad": "167", "stream": "180"}


def _c(code, s):
    return f"\033[38;5;{_C[code]}m{s}\033[0m"


def _render_tool(name, args, bad=False):
    icon = _ICON.get(name, "·")
    a = ", ".join(f"{k}={str(v)[:44]}" for k, v in args.items() if k != "content")
    if isinstance(args, dict) and "content" in args:
        a += (", " if a else "") + f"content=<{len(str(args['content']))}b>"
    col = _C["bad"] if bad else _C["tool"]
    sys.stdout.write(f"  \033[38;5;{col}m{icon} {name}\033[0m {_c('arg', a)}\n")


def _render_result(result):
    s = str(result)
    line = s.splitlines()[0] if s.strip() else "(empty)"
    col = _C["bad"] if s.startswith(("ERROR", "REFUSED", "LOOP")) else _C["ok"]
    sys.stdout.write(f"    \033[38;5;{col}m{line[:110]}\033[0m\n")


def run_turn(messages):
    """Run the agent loop over `messages` until a final (no-tool) answer, streaming
    output and rendering tool calls. Mutates `messages`. Returns the final text."""
    recent = []
    for _ in range(MAX_STEPS):
        printed = [False]

        def on_delta(t, printed=printed):
            if not printed[0]:
                sys.stdout.write(f"\033[38;5;{_C['stream']}m"); printed[0] = True
            sys.stdout.write(t); sys.stdout.flush()

        content, tcs, _finish = _turn(messages, on_delta)
        if printed[0]:
            sys.stdout.write("\033[0m\n")

        if not tcs:
            final = _strip_filler(content)
            messages.append({"role": "assistant", "content": final})
            if not printed[0] and final:
                print(_c("stream", final))
            return final

        messages.append({"role": "assistant", "content": content, "tool_calls": tcs})
        for tc in tcs:
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except ValueError:
                args = _repair_args(name, raw)
                if not isinstance(args, dict):
                    _render_tool(name, {"error": "bad JSON args"}, bad=True)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content":
                        "ERROR: invalid JSON args and unrepairable. Resend ONE valid JSON object."})
                    continue
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            _render_tool(name, args)
            recent.append(sig)
            if recent[-3:].count(sig) >= 3:
                result = "LOOP DETECTED: stop repeating this call; use what you have or answer."
            elif name not in DISPATCH:
                result = f"ERROR: unknown tool '{name}'. Available tools: {', '.join(DISPATCH)}."
            else:
                try:
                    result = DISPATCH[name](args)
                except Exception as e:  # noqa: BLE001
                    result = f"ERROR: {e}"
            _render_result(result)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    print(_c("bad", "● stopped: hit max steps"))
    return None


def _base_messages():
    return [{"role": "system", "content": SYSTEM},
            {"role": "system", "content": "Project layout (current directory):\n" + project_map()}]


def run(task: str):
    """One-shot: run a single task and exit."""
    messages = _base_messages() + [{"role": "user", "content": task}]
    print(_c("arg", f"▶ {WORKDIR}") + "\n")
    return run_turn(messages)


def _banner():
    print(_c("brand", "▐ SDSZcode") + "  " + _c("arg", f"{MODEL} · {WORKDIR}"))


def repl():
    _banner()
    tree = project_map().splitlines()
    print(_c("arg", "\n".join("  " + t for t in tree[:14]) + ("\n  …" if len(tree) > 14 else "")))
    print(_c("arg", "type a task · /map to show files · /exit to quit"))
    messages = _base_messages()
    while True:
        try:
            task = input("\n" + _c("brand", "› "))
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        t = task.strip()
        if not t:
            continue
        if t in ("/exit", "/quit", "/q", "exit", "quit"):
            print("bye"); return
        if t == "/map":
            print(_c("arg", project_map())); continue
        messages.append({"role": "user", "content": task})
        try:
            run_turn(messages)
        except urllib.error.HTTPError as e:
            print(_c("bad", f"● backend error HTTP {e.code}; the shared endpoint is overloaded — try again."))
        except KeyboardInterrupt:
            print("\n(interrupted)")


if __name__ == "__main__":
    if not KEY:
        print("Set CODING_API_KEY (and optionally CODING_API_BASE, CODING_API_MODEL).", file=sys.stderr)
        sys.exit(1)
    try:
        if len(sys.argv) >= 2:
            _banner()
            run(sys.argv[1])
        else:
            repl()
    except urllib.error.HTTPError as e:
        print(f"\n\033[38;5;167m● backend error: HTTP {e.code} {e.reason}\033[0m — overloaded even after "
              f"retries; try again shortly.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n(interrupted)", file=sys.stderr); sys.exit(130)
