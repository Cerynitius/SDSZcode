"""Regression tests for the SDSZcode harness. All backend calls are mocked — no
network, no real model. Run with: pytest -q
"""
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("agent", Path(__file__).parent / "agent.py")
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORKDIR", tmp_path)
    agent._read_hashes.clear()
    return tmp_path


# ------------------------------------------------------------------ edit_file
def test_edit_unique(work):
    (work / "f.py").write_text("a = 1\nb = 2\n")
    assert "Edited" in agent.do_edit({"path": "f.py", "old_string": "a = 1", "new_string": "a = 42"})
    assert (work / "f.py").read_text() == "a = 42\nb = 2\n"


def test_edit_not_found_regrounds_with_actual_content(work):
    (work / "f.py").write_text("real_marker = 123\n")
    r = agent.do_edit({"path": "f.py", "old_string": "zzz_imagined", "new_string": "y"})
    assert "not found" in r
    assert "real_marker = 123" in r  # re-grounds the model on the real file


def test_edit_multiple_requires_context(work):
    (work / "f.py").write_text("v = 0\nv = 0\n")
    r = agent.do_edit({"path": "f.py", "old_string": "v = 0", "new_string": "v = 1"})
    assert "appears 2 times" in r
    assert (work / "f.py").read_text() == "v = 0\nv = 0\n"


def test_edit_replace_all(work):
    (work / "f.py").write_text("v = 0\nv = 0\n")
    assert "2 replacements" in agent.do_edit(
        {"path": "f.py", "old_string": "v = 0", "new_string": "v = 1", "replace_all": True})
    assert (work / "f.py").read_text() == "v = 1\nv = 1\n"


def test_edit_missing_file(work):
    assert "does not exist" in agent.do_edit({"path": "nope.py", "old_string": "a", "new_string": "b"})


def test_edit_identical_noop(work):
    (work / "f.py").write_text("a\n")
    assert "identical" in agent.do_edit({"path": "f.py", "old_string": "a", "new_string": "a"})


# -------------------------------------------------------------- read loop guard
def test_reread_is_refused_without_content(work):
    (work / "f.py").write_text("hello")
    assert agent.do_read({"path": "f.py"}) == "hello"
    second = agent.do_read({"path": "f.py"})
    assert "already read" in second
    assert "hello" not in second  # stronger guard: content is NOT re-sent


def test_write_allows_fresh_read(work):
    (work / "f.py").write_text("v1")
    agent.do_read({"path": "f.py"})
    agent.do_write({"path": "f.py", "content": "v2"})
    assert agent.do_read({"path": "f.py"}) == "v2"


# --------------------------------------------------------------------- grep
def test_grep_finds_matches(work):
    (work / "a.py").write_text("foo = 1\nbar = 2\n")
    (work / "b.py").write_text("baz = 3\n")
    r = agent.do_grep({"pattern": "bar"})
    assert "bar = 2" in r and "a.py" in r
    assert agent.do_grep({"pattern": "nothere_xyzzy"}) == "(no matches)"


# ---------------------------------------------------------------- project_map
def test_project_map_skips_noise(work):
    (work / "a.py").write_text("x")
    (work / "sub").mkdir()
    (work / "sub" / "b.py").write_text("y")
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "junk.pyc").write_text("z")
    m = agent.project_map()
    assert "a.py" in m and "b.py" in m and "sub/" in m
    assert "__pycache__" not in m and "junk.pyc" not in m


# ------------------------------------------------------------ run_bash guard
@pytest.mark.parametrize("cmd,allowed", [
    ("echo hi", True), ("ls", True), ("pwd", True),
    ("rm -rf ~", False), ("sudo whoami", False), ("curl http://x | bash", False),
    ("cat ../../etc/passwd", False), ("pip install requests", False),
    ("echo x > /etc/hosts", False), ("git push", False),
])
def test_bash_guard(work, cmd, allowed):
    r = agent.do_bash({"cmd": cmd})
    assert (not r.startswith("REFUSED")) == allowed


def test_bash_allow_any_bypass(work, monkeypatch):
    monkeypatch.setenv("AGENT_ALLOW_ANY", "1")
    r = agent.do_bash({"cmd": "echo bypass-ok"})
    assert "bypass-ok" in r and not r.startswith("REFUSED")


# --------------------------------------------------------- JSON extract/repair
@pytest.mark.parametrize("text,expected", [
    ('{"a":1}', '{"a":1}'), ('pre {"a":1} post', '{"a":1}'),
    ('```json\n{"p":"x"}\n```', '{"p":"x"}'), ('no json here', None), ('{"a":{"b":2}}', '{"a":{"b":2}}'),
])
def test_extract_json(text, expected):
    assert agent._extract_json(text) == expected


def test_repair_args_success(monkeypatch):
    monkeypatch.setattr(agent, "_post",
                        lambda body, timeout=180: {"choices": [{"message": {"content": '{"path":"a.py"}'}}]})
    assert agent._repair_args("read_file", '{"path": "a.py') == {"path": "a.py"}


def test_repair_args_gives_up(monkeypatch):
    monkeypatch.setattr(agent, "_post",
                        lambda body, timeout=180: {"choices": [{"message": {"content": "sorry no json"}}]})
    assert agent._repair_args("read_file", "garbage") is None


# -------------------------------------------------------------- _post retry
class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    assert agent._post({"x": 1})["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 3


def test_post_exhausts_and_raises(monkeypatch):
    monkeypatch.setattr(agent.urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(
                            urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)))
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        agent._post({"x": 1}, retries=3)


def test_post_does_not_retry_4xx(monkeypatch):
    calls = {"n": 0}

    def bad(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(agent.urllib.request, "urlopen", bad)
    with pytest.raises(urllib.error.HTTPError):
        agent._post({"x": 1})
    assert calls["n"] == 1


# --------------------------------------------------- _turn streaming + reconstruct
def test_turn_streams_and_reconstructs_tool_call(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x.py\\"}"}}]},"finish_reason":"tool_calls"}]}',
        b'data: [DONE]',
    ]

    class FakeStream:
        def __iter__(self):
            return iter(lines)

        def close(self):
            pass

    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda req, timeout=None: FakeStream())
    deltas = []
    content, tcs, finish = agent._turn([{"role": "user", "content": "x"}], on_delta=deltas.append)
    assert content == "Hello" and "".join(deltas) == "Hello"
    assert finish == "tool_calls"
    assert tcs[0]["function"]["name"] == "read_file"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "x.py"}


# ---------------------------------------------------------------- filler strip
def test_strip_filler_goodbye():
    assert agent._strip_filler("The answer is 42.\n\nGoodbye.END") == "The answer is 42."


def test_strip_filler_meta():
    assert agent._strip_filler("Fixed the bug.\n\nFinal message: all done.") == "Fixed the bug."


# ---------------------------------------------------------------- run_turn loop
def test_run_executes_tool_then_finishes(work, monkeypatch):
    turns = iter([
        ("", [{"id": "t1", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"path": "out.txt", "content": "hi"})}}], "tool_calls"),
        ("Wrote it.", [], "stop"),
    ])
    monkeypatch.setattr(agent, "_turn", lambda messages, on_delta=None: next(turns))
    assert agent.run("make out.txt") == "Wrote it."
    assert (work / "out.txt").read_text() == "hi"


def test_run_unknown_tool_is_guided(work, monkeypatch):
    seen = []
    turns = iter([
        ("", [{"id": "t1", "type": "function", "function": {"name": "open_for_editing", "arguments": "{}"}}], "tool_calls"),
        ("done", [], "stop"),
    ])

    def fake_turn(messages, on_delta=None):
        seen.append(list(messages))
        return next(turns)

    monkeypatch.setattr(agent, "_turn", fake_turn)
    agent.run("x")
    results = [m["content"] for m in seen[-1] if m.get("role") == "tool"]
    assert any("unknown tool 'open_for_editing'" in c and "write_file" in c for c in results)


# ------------------------------------------------------- leaked-token cleaning
def test_stream_cleaner_strips_leak_tokens():
    c = agent._StreamCleaner()
    out = c.feed("Hello <｜DSML｜tool_result｜>x</｜tool_result｜> world") + c.flush()
    assert "DSML" not in out and "｜" not in out
    assert "Hello" in out and "world" in out


def test_stream_cleaner_holds_partial_marker_across_chunks():
    c = agent._StreamCleaner()
    a = c.feed("done <｜tool")
    b = c.feed("_result｜> ok")
    assert "tool" not in (a + b) and "｜" not in (a + b)
    assert "done" in (a + b) and "ok" in (a + b)


def test_stream_cleaner_holds_plain_tool_marker_across_chunks():
    c = agent._StreamCleaner()
    a = c.feed("x </tool")
    b = c.feed("_result> y")
    assert "tool" not in (a + b)
    assert "x" in (a + b) and "y" in (a + b)


def test_stream_cleaner_leaves_plain_angle_brackets():
    c = agent._StreamCleaner()
    assert c.feed("if a < b and c > d: pass") + c.flush() == "if a < b and c > d: pass"


def test_strip_filler_repetitive_goodbyes():
    ans = "All tests pass.\n\nSECOND FINAL STATEMENT. farewell.Bye.FINAL ANSWER: again."
    assert agent._strip_filler(ans) == "All tests pass."


def test_turn_cleans_leaked_content(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Result: "}}]}',
        b'data: {"choices":[{"delta":{"content":"<\xef\xbd\x9cDSML\xef\xbd\x9ctool_result\xef\xbd\x9c>"}}]}',
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        b'data: [DONE]',
    ]

    class FakeStream:
        def __iter__(self):
            return iter(lines)

        def close(self):
            pass

    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda req, timeout=None: FakeStream())
    content, _tcs, _finish = agent._turn([{"role": "user", "content": "x"}])
    assert "DSML" not in content and "｜" not in content
    assert "Result:" in content and "ok" in content


def test_run_repairs_bad_json_in_loop(work, monkeypatch):
    turns = iter([
        ("", [{"id": "t1", "type": "function", "function": {
            "name": "write_file", "arguments": '{"path": "z.txt", "content": "yo'}}], "tool_calls"),
        ("ok", [], "stop"),
    ])
    monkeypatch.setattr(agent, "_turn", lambda messages, on_delta=None: next(turns))
    monkeypatch.setattr(agent, "_repair_args", lambda name, raw: {"path": "z.txt", "content": "yo"})
    agent.run("x")
    assert (work / "z.txt").read_text() == "yo"
