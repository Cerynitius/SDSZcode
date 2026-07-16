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

An **OpenAI-compatible** chat-completions endpoint that supports tool calling. Any
provider works; the defaults target a local `deepseek-v4-flash` server.

## Usage

```bash
export CODING_API_BASE="https://your-endpoint/v1"   # OpenAI-compatible base URL
export CODING_API_KEY="sk-..."                       # your API key
export CODING_API_MODEL="deepseek-v4-flash"          # optional (this is the default)

# runs in the current directory
python3 agent.py "Write fib.py with an iterative fib(n) plus asserts, and make the tests pass."
```

### Tools the agent has

`read_file`, `write_file`, `edit_file` (precise search/replace), `run_bash`,
`list_dir` — enough to write code, patch it, run it, and iterate.

Robustness: if the model emits malformed JSON tool arguments, the harness asks it to
re-emit valid JSON (a cheap retry) instead of wasting an agent turn.

### Configuration (env vars)

| var | default | meaning |
|---|---|---|
| `CODING_API_BASE` | `http://127.0.0.1:8000/v1` | OpenAI-compatible base URL |
| `CODING_API_KEY` | – (required) | API key |
| `CODING_API_MODEL` | `deepseek-v4-flash` | model id |
| `AGENT_MAX_STEPS` | `16` | max tool-call rounds |
| `AGENT_ALLOW_ANY` | – | set to `1` to disable the shell safety guard (sandboxes only) |

## Safety

`run_bash` has a guard: an allow-list of run/inspect commands (`python3`, `pytest`,
`ls`, `cat`, …) plus a deny-list for destructive / networked / privileged ops and
directory escapes. **It is not a real sandbox** — an allow-listed interpreter can
still do anything. Run untrusted tasks inside a container or VM.

## Development

Run the regression tests — fully mocked (no network, no real model):

```bash
pytest -q
```

They lock in the harness behaviours: edit_file's uniqueness rules, the read loop
guard, the shell safety guard, JSON extraction/repair, 503 retry/backoff, filler
stripping, and the core run loop (tool dispatch, unknown-tool guidance, in-loop JSON
repair).

## Status

Early and evolving — we're tuning it as we learn more about how small coding models
behave in agent loops. Issues and PRs welcome.

## License

MIT
