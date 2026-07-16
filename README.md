# SDSZcode

A tiny, hackable **coding-agent harness tuned for small / local coding models**
(built and tested against `deepseek-v4-flash` on a self-hosted endpoint).

Generic agent harnesses (opencode, aider, …) assume a strong frontier model. Point
them at a small "fast" coding model and you hit predictable failure modes. SDSZcode
is a ~200-line single file that adapts to those failure modes instead of fighting
them — so a modest model becomes genuinely useful in an agent loop.

## Why

Real tests of `deepseek-v4-flash` under a stock harness showed three things:

| Failure mode | What SDSZcode does about it |
|---|---|
| **Loops re-reading the same file** instead of acting | Loop guard: repeated/identical tool calls are short-circuited with a "you already have this, act now" result |
| **Claims "tests pass" without ever running code** | System prompt forbids claiming success without a real `run_bash`; the model is pushed to run → see failure → fix |
| **Pads endings** with repeated `Goodbye.END`… filler | Trailing-filler stripping on the final answer |

Plus: deterministic sampling (`temperature=0`), `reasoning_effort=low`, tolerant
tool-argument parsing, a step cap, and a **safety guard** on shell execution.

In A/B tests on the same task and model, the stock harness produced buggy code and
falsely claimed the tests passed; under SDSZcode the model ran its own tests, hit the
failure, and fixed it to correct, verified code.

## Requirements

Python 3.9+ and an **OpenAI-compatible** chat-completions endpoint that supports tool
calling. Any provider works; the defaults target a local `deepseek-v4-flash` server.
No third-party dependencies — standard library only.

## Install

```bash
pip install -e .        # installs the `sdszcode` command
```

Or skip installing and run the script directly with `python3 agent.py`.

## Usage

```bash
export CODING_API_KEY="sk-..."                       # your API key (or pass --key)
export CODING_API_BASE="https://your-endpoint/v1"    # OpenAI-compatible base URL
export CODING_API_MODEL="deepseek-v4-flash"          # optional (this is the default)

sdszcode                            # interactive: a terminal REPL in the current dir
sdszcode "fix the failing tests"    # one-shot: run a single task and exit
```

The interactive UI streams the model's output live, renders each tool call with an
icon, prints a project file map on start, and takes follow-up tasks (multi-turn).
### Interactive session (Claude-Code style)

Running `sdszcode` with no task opens a session that feels like Claude Code:

- **Welcome box** with the model and working directory.
- **Slash commands:** `/help`, `/clear` (reset the conversation), `/undo` (revert the
  last file change), `/map` (file tree), `/model [name]` (show or switch the model),
  `/cwd`, `/exit`.
- **Multi-line input:** end a line with `\` to continue, or wrap a block in a
  ` ``` ` (or `"""`) fence — handy for pasting code. The whole block is sent as one task.
- **Tool calls** render as `● tool(arg)` with an indented `⎿` result summary.
- **Permission prompts before anything side-effecting.** Before `run_bash`,
  `write_file`, or `edit_file` runs, you get a `[Y]es / [n]o / [a]lways` prompt —
  and for edits/writes a **coloured diff preview** of exactly what will change.
  `a` (always) trusts that tool for the rest of the session; read-only tools
  (`read_file`, `grep`, `list_dir`) never prompt. `--yes` skips the prompts, and
  one-shot / piped runs auto-approve.
- **`/undo`** reverts the most recent `write_file` / `edit_file` — restoring the previous
  contents, or deleting the file if it was newly created. Repeat it to walk further back.

The underlying [safety guard](#safety) (the bash allow/deny list) still applies
regardless of what you approve.

### CLI flags

```
sdszcode [OPTIONS] [TASK]

  -C, --dir PATH       working directory the agent operates in (default: current)
  -m, --model NAME     model id (default: deepseek-v4-flash)
  -b, --base URL       API base URL
  -k, --key KEY        API key (else CODING_API_KEY)
  -s, --max-steps N    max tool-call rounds per task (default: 16)
  -y, --yes            skip permission prompts — auto-approve every write/command
      --allow-any      disable the shell safety guard (sandboxes/trusted dirs only)
      --no-color       plain output (also auto-off when piped or NO_COLOR is set)
  -V, --version        print version
  -h, --help           full help
```

Flags override the environment. Example: `sdszcode -C ~/proj -s 24 "add a --json flag"`.

### Tools the agent has

`read_file`, `write_file`, `edit_file` (precise search/replace), `grep` (search),
`run_bash`, `list_dir` — enough to explore, write code, patch it, run it, and iterate.

### Robustness built in

- **Streaming** output with a clean, coloured terminal UI.
- **Project map** injected on start so the model knows the layout without reading
  everything.
- **JSON retry** — malformed tool arguments are re-requested, not fatal.
- **503/backoff retry** — the endpoint may be a capacity-limited shared GPU; transient
  5xx/429 are retried (configurable via `AGENT_RETRIES` / `AGENT_MAX_BACKOFF`).
- **Loop guard** — re-reading an unchanged file is refused; a repeated tool call is
  broken with a nudge.
- **Unknown-tool guidance** — inventing a tool name returns the real tool list.
- **Anti-hallucination on** — every request carries `X-Anti-Hallucination: on` (and an
  `anti_hallucination` body field, so it survives header-stripping proxies). The backend
  then makes the model admit uncertainty instead of confabulating confident-sounding
  facts. This curbs *factual* hallucination; *code-structure* confabulation (inventing
  files/symbols) is caught by the exact-match `edit_file` re-grounding above.
- **Repetition breaker** — small models sometimes spiral into repeating a token or line
  until the budget is spent; the streamed output is watched and cut off the moment a
  spiral is detected (`(cut off: runaway repetition)`).
- **Stall guard** — stops after a few consecutive tool-taking turns that change nothing
  and reveal nothing new, rather than spinning to the step limit (`AGENT_STALL_LIMIT`).
- **Act-don't-narrate nudge** — if a turn *describes* an action ("Let me read…") without
  emitting the tool call, it's nudged to actually call the tool (up to `AGENT_MAX_NUDGES`).

### Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `CODING_API_BASE` | `http://127.0.0.1:8000/v1` | OpenAI-compatible base URL |
| `CODING_API_KEY` | – (required) | API key |
| `CODING_API_MODEL` | `deepseek-v4-flash` | model id |
| `AGENT_MAX_STEPS` | `16` | max tool-call rounds |
| `AGENT_STALL_LIMIT` | `3` | stop after N no-progress tool turns |
| `AGENT_MAX_NUDGES` | `2` | max "act, don't narrate" nudges per task |
| `CODING_API_THINKING` | `low` | reasoning depth (`low`/`high`/`max`); `low` works best here |
| `CODING_API_ANTI_HALLUCINATION` | `on` | `off` lets the model confabulate; leave `on` |
| `AGENT_ALLOW_ANY` | – | set to `1` to disable the shell safety guard (sandboxes only) |

## Safety

`run_bash` has a guard: an allow-list of run/inspect commands (`python3`, `pytest`,
`ls`, `cat`, …) plus a deny-list for destructive / networked / privileged ops and
directory escapes. **It is not a real sandbox** — an allow-listed interpreter can
still do anything. Run untrusted tasks inside a container or VM.

## What to expect (and why the guards matter)

Small coding models are useful in a loop but they misbehave, and SDSZcode is built to
make that *safe* rather than to pretend it doesn't happen. Observed with
`deepseek-v4-flash`:

- **Confabulation on multi-file tasks.** It will sometimes invent files, methods,
  tests, and error messages it never actually read, then confidently "fix" the
  fictional version. In one run it hallucinated an entire alternate module and tried
  to patch it — every one of those edits **failed harmlessly** because `edit_file`
  requires an exact, unique match against the real file. The real bug (in a different
  file it *did* read correctly) got fixed; nothing was corrupted.
  → *This is why the strict exact-match `edit_file` is the single most important safety
  feature.* A fuzzy or whole-file-rewrite editor would have let the hallucination
  overwrite correct code.
- **Leaked control tokens.** It emits chat-template markers (`<｜…｜>`, fake
  `</tool_result>` blocks) into its output. The harness strips these from the stream
  and from history so they don't pollute the display or compound.
- **Filler / repeated goodbyes.** Trimmed from the final answer.
- **Better with feedback than from scratch.** Give it failing tests to fix and it runs
  them, sees the error, and iterates; ask it to write correct code cold and verify the
  result yourself.

Practical guidance: keep tasks well-scoped, prefer a test-driven loop, and treat the
harness's guards (not the model's confidence) as the source of safety.

## Development

Run the regression tests — fully mocked (no network, no real model):

```bash
pytest -q
```

They lock in the harness behaviours: edit_file's uniqueness rules, the read loop
guard, the shell safety guard, JSON extraction/repair, 503 retry/backoff, filler
stripping, and the core run loop (tool dispatch, unknown-tool guidance, in-loop JSON
repair).

### Benchmarking pass rate

`bench.py` measures how reliably the harness completes tasks, across three difficulty
tiers. It seeds fresh temp projects, runs the agent N times per task against the **real**
backend, and reports the pass rate plus timing and guard-signal stats — so parameter
changes can be compared with numbers, not vibes:

- **easy** — single file, single-function bug; a failing test points right at it.
- **medium** — 2+ files; the bug is *not* where the test fails, so it must be traced.
- **hard** — no visible test (implement to a spec) or a coordinated multi-file change;
  graded by a **hidden** test dropped in only after the agent finishes.

```bash
export CODING_API_KEY=sk-...
python3 bench.py                         # easy tier, 5 runs each
python3 bench.py --tier all --runs 10
python3 bench.py --tier hard --runs 5
CODING_API_THINKING=high python3 bench.py --tier medium   # A/B a setting
python3 bench.py --list                 # tasks and their tiers
```

Example output:

```
task         tier       pass    rate   avg s  signals
factorial    easy      1/2      50%    52.1  tools=28 edit=3 cut=3 nudge=2 stall=0
OVERALL                1/2      50%    52.1  ██████████░░░░░░░░░░
```

## Status

Early and evolving — we're tuning it as we learn more about how small coding models
behave in agent loops. Issues and PRs welcome.

## License

MIT
