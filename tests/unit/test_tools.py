"""Tool system tests against a real temporary workspace (no network)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mao.config.schema import ApprovalRules, PermissionSet, PermissionsConfig, ToolsConfig
from mao.core.errors import PermissionDeniedError
from mao.core.events import EventBus
from mao.core.types import RunMode, ToolCall
from mao.messaging.blackboard import Blackboard
from mao.messaging.bus import MessageBus
from mao.messaging.hub import CollaborationHub
from mao.messaging.router import ContextRouter, RoutingRequest
from mao.security.approval import ApprovalDecision, ApprovalGateway, ApprovalRequest
from mao.security.redaction import Redactor
from mao.security.risk import ActionKind
from mao.security.sandbox import WorkspaceSandbox
from mao.tools.base import ToolContext
from mao.tools.executor import ToolExecutor
from mao.tools.registry import build_default_registry
from mao.tools.tests_tool import execute_tests, parse_test_output
from mao.tools.web import _DuckDuckGoParser, ensure_public_url
from mao.workspace.changes import ChangeTracker
from mao.workspace.git import GitRepo
from mao.workspace.workspace import scan_workspace

from tests.conftest import windows_only

FULL = PermissionSet(read=True, write=True, delete=True, execute=True, internet=False, git="write")


class Env:
    def __init__(self, root: Path, tmp: Path, *, mode: RunMode = RunMode.RUN, perms: PermissionSet = FULL, decision: ApprovalDecision = ApprovalDecision.ALLOW) -> None:
        self.bus = EventBus()
        self.requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> ApprovalDecision:
            self.requests.append(request)
            return decision

        permissions_config = PermissionsConfig()
        self.sandbox = WorkspaceSandbox(root, protected_patterns=permissions_config.protected_patterns, sensitive_patterns=permissions_config.sensitive_patterns)
        self.changes = ChangeTracker(self.sandbox, tmp / "backups")
        self.registry = build_default_registry()
        self.executor = ToolExecutor(self.registry, redactor=Redactor(), max_result_chars=12_000)
        self.board = Blackboard(Redactor())
        self.messages = MessageBus(self.bus, Redactor())
        self.ctx = ToolContext(
            agent_name="coder-1",
            agent_role="coder",
            permissions=perms,
            mode=mode,
            sandbox=self.sandbox,
            approval=ApprovalGateway(ApprovalRules(), handler, self.bus),
            changes=self.changes,
            bus=self.bus,
            tools_config=ToolsConfig(),
            permissions_config=permissions_config,
            collaboration=CollaborationHub(self.messages, self.board),
        )

    async def call(self, name: str, **args):  # type: ignore[no-untyped-def]
        allowed = {t.name for t in self.executor.tools_for(self.ctx, list(ToolsConfig().enabled_groups))}
        return await self.executor.execute(ToolCall(id="c1", name=name, arguments=args), self.ctx, allowed)


# ------------------------------------------------------------------ filesystem


async def test_read_file_with_range(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    (workspace / "long.txt").write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")
    result = await env.call("read_file", path="long.txt", start_line=10, end_line=12)
    assert result.ok
    assert "   10| line 10" in result.content and "   12| line 12" in result.content and "line 13" not in result.content
    (workspace / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert not (await env.call("read_file", path="bin.dat")).ok


async def test_write_edit_delete_and_rollback(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    (workspace / "crlf.txt").write_bytes(b"a = 1\r\nb = 2\r\n")
    assert (await env.call("write_file", path="new/module.py", content="print('new')\n")).ok
    assert (await env.call("edit_file", path="crlf.txt", old_text="b = 2", new_text="b = 3")).ok
    assert (workspace / "crlf.txt").read_bytes() == b"a = 1\r\nb = 3\r\n"
    result = await env.call("delete_path", path="README.md")
    assert result.ok and not (workspace / "README.md").exists()
    assert any(r.action.kind is ActionKind.DELETE for r in env.requests)

    summary = env.changes.summary()
    assert summary == {"created": ["new/module.py"], "modified": ["crlf.txt"], "deleted": ["README.md"]}
    diff = env.changes.diff()
    assert "-b = 2" in diff and "+b = 3" in diff and "+print('new')" in diff

    restored = env.changes.rollback()
    assert set(restored) >= {"new/module.py", "crlf.txt", "README.md"}
    assert (workspace / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    assert (workspace / "crlf.txt").read_bytes() == b"a = 1\r\nb = 2\r\n"
    assert not (workspace / "new" / "module.py").exists()


async def test_edit_file_ambiguity_and_missing(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    (workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    ambiguous = await env.call("edit_file", path="dup.py", old_text="x = 1", new_text="x = 2")
    assert not ambiguous.ok and "occurs 2 times" in ambiguous.content
    missing = await env.call("edit_file", path="dup.py", old_text="y = 1", new_text="y = 2")
    assert not missing.ok
    assert (await env.call("edit_file", path="dup.py", old_text="x = 1", new_text="x = 2", replace_all=True)).ok
    assert (workspace / "dup.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


async def test_sandbox_violation_and_denied_approval(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    outside = await env.call("write_file", path="../evil.txt", content="x")
    assert not outside.ok and "outside" in outside.content
    protected = await env.call("write_file", path=".git/config", content="x")
    assert not protected.ok
    denied_env = Env(workspace, tmp_path, decision=ApprovalDecision.DENY)
    denied = await denied_env.call("delete_path", path="README.md")
    assert not denied.ok and "rejected" in denied.content.lower()
    assert (workspace / "README.md").exists()


async def test_search_and_find(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    (workspace / ".env").write_text("SECRET_TOKEN=abc", encoding="utf-8")
    hits = await env.call("search_files", pattern=r"def \w+", glob="*.py")
    assert hits.ok and "src/app.py:1:" in hits.content
    secret_search = await env.call("search_files", pattern="SECRET_TOKEN")
    assert ".env" not in secret_search.content and "No matches" in secret_search.content
    found = await env.call("find_files", pattern="*.py")
    assert "src/app.py" in found.content


async def test_plan_mode_blocks_mutations(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path, mode=RunMode.PLAN)
    visible = {t.name for t in env.executor.tools_for(env.ctx, list(ToolsConfig().enabled_groups))}
    assert "read_file" in visible and "run_command" in visible
    assert "write_file" not in visible and "delete_path" not in visible and "run_tests" not in visible
    result = await env.executor.execute(ToolCall(id="x", name="write_file", arguments={"path": "a.txt", "content": "x"}), env.ctx, visible | {"write_file"})
    assert not result.ok and "PLAN" in result.content
    blocked = await env.call("run_command", command="echo hi")
    assert not blocked.ok and "PLAN" in blocked.content
    assert not (workspace / "a.txt").exists()


async def test_permissions_hide_tools(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path, perms=PermissionSet(read=True))
    visible = {t.name for t in env.executor.tools_for(env.ctx, list(ToolsConfig().enabled_groups))}
    assert "read_file" in visible and "post_finding" in visible
    assert not visible & {"write_file", "run_command", "git_commit", "web_search", "delete_path"}
    result = await env.executor.execute(ToolCall(id="x", name="write_file", arguments={"path": "a", "content": "b"}), env.ctx, {"write_file"})
    assert not result.ok and "No permission" in result.content


async def test_argument_validation(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    result = await env.call("read_file")
    assert not result.ok and "Required parameter" in result.content
    wrong = await env.call("read_file", path=5)
    assert not wrong.ok and "wrong type" in wrong.content


# ------------------------------------------------------------------ terminal


@windows_only
async def test_run_command_scrubs_secrets(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAO_DEMO_API_KEY", "super-secret-value-123")
    env = Env(workspace, tmp_path)
    result = await env.call("run_command", command="echo hello %MAO_DEMO_API_KEY%")
    assert result.ok, result.content
    assert "hello" in result.content and "super-secret-value-123" not in result.content


@windows_only
async def test_run_command_timeout_kills_process(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    result = await env.call("run_command", command="ping -n 30 127.0.0.1", timeout_s=1)
    assert not result.ok and "TIMEOUT" in result.content
    assert result.data["timed_out"] is True


async def test_denylisted_command(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    result = await env.call("run_command", command="format c:")
    assert not result.ok and "denylist" in result.content.lower()


async def test_high_risk_command_requires_approval(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path, decision=ApprovalDecision.DENY)
    result = await env.call("run_command", command="del /q README.md")
    assert not result.ok
    assert env.requests and env.requests[0].action.kind is ActionKind.EXECUTE
    assert (workspace / "README.md").exists()


# ------------------------------------------------------------------ tests tool


def test_parse_test_output_formats() -> None:
    assert parse_test_output("===== 2 failed, 10 passed, 1 skipped in 1.23s =====")["failed"] == 2
    jest = parse_test_output("Tests:       1 failed, 4 passed, 5 total")
    assert (jest["passed"], jest["failed"]) == (4, 1)
    cargo = parse_test_output("test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured")
    assert (cargo["passed"], cargo["failed"]) == (3, 1)
    unittest = parse_test_output("Ran 5 tests in 0.1s\n\nFAILED (failures=2)")
    assert (unittest["passed"], unittest["failed"]) == (3, 2)


async def test_execute_tests_real_pytest(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_sample.py").write_text("def test_ok():\n    assert 1 == 1\n\ndef test_bad():\n    assert 1 == 2\n", encoding="utf-8")
    env = Env(project, tmp_path)
    recorded = []
    env.ctx.on_test_result = recorded.append
    summary = await execute_tests(env.ctx, f'"{sys.executable}" -m pytest -q -p no:cacheprovider')
    assert summary.passed == 1 and summary.failed == 1 and not summary.success
    assert recorded and recorded[0].failed == 1


# ------------------------------------------------------------------ git


async def test_git_tools_and_checkpoint(workspace: Path, tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    repo = await GitRepo.detect(workspace)
    assert repo is not None
    env = Env(workspace, tmp_path)
    env.ctx.git = repo
    (workspace / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    status = await env.call("git_status")
    assert "app.py" in status.content
    diff = await env.call("git_diff")
    assert "-    return a + b" in diff.content
    sha = await repo.checkpoint("session_001")
    assert sha and (await repo.list_checkpoints())[0][0] == "session_001"
    commit = await env.call("git_commit", message="fix: subtraction bug")
    assert commit.ok, commit.content
    assert await repo.is_clean()


# ------------------------------------------------------------------ collaboration & routing


async def test_collaboration_hub_and_router(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    assert (await env.call("post_finding", title="Slow loop", content="src/app.py iterates in O(n^2)", files=["src/app.py"], importance=4)).ok
    assert (await env.call("send_message", to="reviewer", content="Please check app.py", kind="question")).ok
    env.board.add("architect", "decision", "Solution B", "We use solution B because of doc X")
    env.board.add("coder-1", "result", "step s1 done", "Details …", node_id="s1")

    router = ContextRouter(env.board, env.messages)
    text = router.build(RoutingRequest(agent_name="reviewer", role="reviewer", query_text="app.py loop", depends_on=["s1"], budget_tokens=4_000))
    assert "Solution B" in text and "Please check app.py" in text and "step s1 done" in text and "Slow loop" in text
    again = router.build(RoutingRequest(agent_name="reviewer", role="reviewer", budget_tokens=4_000))
    assert "Please check app.py" not in again  # already delivered

    board_text = await env.call("read_board", query="loop")
    assert "O(n^2)" in board_text.content


async def test_consult_limit(workspace: Path, tmp_path: Path) -> None:
    env = Env(workspace, tmp_path)
    calls = []

    async def handler(requester: str, target: str, question: str, node_id: str | None, depth: int) -> str:
        calls.append((requester, target, depth))
        return "Response"

    hub = CollaborationHub(env.messages, env.board, consult_handler=handler, max_consultations_per_task=1)
    env.ctx.collaboration = hub
    assert (await env.call("consult_agent", agent="architect", question="Which solution?")).content == "Response"
    limited = await env.call("consult_agent", agent="architect", question="Again?")
    assert not limited.ok and "Limit" in limited.content
    assert calls == [("coder-1", "architect", 1)]
    kinds = [m.kind for m in env.messages.all()]
    assert kinds == ["question", "answer"]


# ------------------------------------------------------------------ web helpers & workspace


def test_duckduckgo_parser() -> None:
    html = (
        '<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=x">Python <b>Docs</b></a>'
        '<a class="result__snippet" href="#">Official documentation</a></div>'
    )
    parser = _DuckDuckGoParser()
    parser.feed(html)
    assert parser.hits[0].url == "https://docs.python.org/3/"
    assert parser.hits[0].title == "Python Docs" and parser.hits[0].snippet == "Official documentation"


async def test_ssrf_protection() -> None:
    with pytest.raises(PermissionDeniedError):
        await ensure_public_url("http://127.0.0.1:11434/api/tags", allow_private=False)
    with pytest.raises(PermissionDeniedError):
        await ensure_public_url("file:///C:/Windows/win.ini", allow_private=False)
    await ensure_public_url("http://127.0.0.1:8080", allow_private=True)


def test_workspace_profile(workspace: Path) -> None:
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_app.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    profile = scan_workspace(workspace, ToolsConfig().filesystem.ignore_patterns)
    assert profile.languages.get("Python") == 2
    assert profile.test_command == "python -m pytest -q"
    assert "node_modules" not in profile.tree
    assert "README.md" in profile.key_files
    assert os.sep  # platform independent assertions above
