"""Offline demo mode (``mao --demo``).

Only the MODEL ANSWERS are simulated by a scripted mock provider. Everything
else is real: planning pipeline, agent tool loops, file changes in the demo
workspace, test runs, reviews, agent-to-agent messages, logs and sessions.
The UI labels this mode clearly as simulated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from mao.config.schema import AppConfig, ModelConfig, ProviderConfig
from mao.core.types import ChatMessage, CompletionRequest, CompletionResponse, MessageRole, ToolCall, new_id

DEMO_NOTICE = "DEMO MODE: model replies are simulated (mock provider). Tools, files, tests and logs are real."

DEMO_FILES = {
    "calculator.py": (
        '"""A small calculator for the orchestrator demo."""\n\n\n'
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a - b\n\n\n"
        "def divide(a, b):\n"
        '    """Return a divided by b."""\n'
        "    return a / b\n"
    ),
    "tests/test_calculator.py": (
        "from calculator import add, divide\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n\n"
        "def test_divide():\n"
        "    assert divide(6, 3) == 2\n"
    ),
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
    "README.md": "# Calculator demo\n\nA small project with a deliberate bug in `add()`.\n",
}


def create_demo_workspace(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for relative, content in DEMO_FILES.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return target


def apply_demo_config(config: AppConfig) -> None:
    providers = config.providers.providers
    for name, provider in providers.items():
        if name != "mock":
            provider.enabled = False
    providers["mock"] = ProviderConfig(
        type="mock",
        display_name="Mock (simuliert)",
        enabled=True,
        requires_api_key=False,
        selectable=True,
        local=True,
        max_retries=0,
        models={
            "scripted": ModelConfig(
                display_name="Demo model (simulated)",
                context_window=64_000,
                max_output_tokens=4_096,
                tier="strong",
                capabilities=["tools", "coding", "reasoning", "json"],
            )
        },
    )
    config.settings.orchestration.orchestrator_model = "mock/scripted"
    if config.tools.tests.command is None:
        # the demo project needs pytest, which is installed in mao's own environment
        config.tools.tests.command = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'


def install_demo_responder(hub: Any) -> None:
    provider = hub.providers.get("mock")
    if provider is not None and hasattr(provider, "set_responder"):
        provider.set_responder(DemoBrain())


def _call(name: str, **arguments: Any) -> CompletionResponse:
    return CompletionResponse(message=ChatMessage.assistant("", [ToolCall(id=new_id("demo_"), name=name, arguments=arguments)]))


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


PLAN = {
    "title": "(Demo) Fix the errors in the calculator",
    "summary": "Analyse the calculator, fix add(), guard divide(), run tests and review.",
    "approach": "Locate the bug with the tests, fix it minimally, verify with the test suite and review.",
    "steps": [
        {"id": "s1", "title": "Analyse code and tests", "description": "Read calculator.py and tests/, identify the faulty functions.", "kind": "analysis", "role": "architect", "depends_on": [], "modifies_files": False, "acceptance_criteria": ["Root cause named with file and line"], "complexity": 1},
        {"id": "s2", "title": "Fix add()", "description": "In calculator.py, add() must return the sum (a + b instead of a - b).", "kind": "implementation", "role": "coder", "depends_on": ["s1"], "modifies_files": True, "acceptance_criteria": ["add(2, 3) == 5"], "complexity": 1},
        {"id": "s3", "title": "Run the tests", "description": "Run the test suite; on failures, enter the test-fix cycle.", "kind": "test", "role": "tester", "depends_on": ["s2"], "modifies_files": False, "acceptance_criteria": ["All tests pass"], "complexity": 1},
        {"id": "s4", "title": "Review the changes", "description": "Check the diff.", "kind": "review", "role": "reviewer", "depends_on": ["s3"], "modifies_files": False, "acceptance_criteria": ["No blocking findings"], "complexity": 1},
        {"id": "s5", "title": "Final report", "description": "Summarise the results.", "kind": "synthesis", "role": "project_manager", "depends_on": ["s4"], "modifies_files": False, "acceptance_criteria": [], "complexity": 1},
    ],
    "risks": [{"risk": "divide() raises ZeroDivisionError when b == 0", "severity": "medium", "mitigation": "Document the behaviour or check it explicitly"}],
    "assumptions": ["The tests describe the intended behaviour"],
    "open_questions": [],
    "success_criteria": ["All tests pass"],
}


class DemoBrain:
    """Purpose-aware scripted answers that drive real tools."""

    def __call__(self, request: CompletionRequest) -> CompletionResponse | str:
        purpose = str(request.metadata.get("purpose") or "")
        handler = getattr(self, "on_" + purpose.replace(".", "_"), None)
        if handler is None:
            return self.on_generic(request)
        return handler(request)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _results(request: CompletionRequest) -> list[ChatMessage]:
        return [m for m in request.messages if m.role is MessageRole.TOOL]

    @staticmethod
    def _has_tool(request: CompletionRequest, name: str) -> bool:
        return any(t.name == name for t in request.tools)

    @staticmethod
    def _called(request: CompletionRequest, name: str) -> bool:
        return any(c.name == name for m in request.messages for c in m.tool_calls)

    def _fix_bug(self, request: CompletionRequest, label: str) -> CompletionResponse | str:
        results = self._results(request)
        if not self._called(request, "read_file") and self._has_tool(request, "read_file"):
            return _call("read_file", path="calculator.py")
        read = next((m for m in results if m.name == "read_file"), None)
        if read and "return a - b" in read.content and not self._called(request, "edit_file") and self._has_tool(request, "edit_file"):
            return _call("edit_file", path="calculator.py", old_text="return a - b", new_text="return a + b")
        edited = next((m for m in results if m.name == "edit_file"), None)
        if edited and not edited.is_error:
            if self._has_tool(request, "send_message") and not self._called(request, "send_message"):
                return _call("send_message", to="tester", kind="info", content="(Demo) add() fixed - please run the tests.")
            return _json({"status": "done", "summary": f"(Demo) {label}: add() now returns a + b.", "details": "changed calculator.py line 7", "files_changed": ["calculator.py"], "issues": [], "next_steps": ["Run the tests"]})
        return _json({"status": "done", "summary": f"(Demo) {label}: no change needed.", "files_changed": [], "issues": []})

    # ------------------------------------------------------------------ planning

    def on_plan_triage(self, request: CompletionRequest) -> CompletionResponse | str:
        if not self._called(request, "list_directory") and self._has_tool(request, "list_directory"):
            return _call("list_directory", path=".", depth=2)
        return _json(
            {
                "understanding": "(Demo) The calculator should work correctly; the existing tests define the expected behaviour.",
                "task_type": "code_change",
                "complexity": 3,
                "modifies_files": True,
                "needs_research": False,
                "investigations": [
                    {"role": "architect", "focus": "Find the faulty functions in calculator.py", "questions": ["Which function violates the tests?"]},
                    {"role": "security", "focus": "Edge cases and robust input handling", "questions": ["Are there unchecked edge cases?"]},
                ],
                "clarifications": [],
            }
        )

    def on_plan_investigate(self, request: CompletionRequest) -> CompletionResponse | str:
        if not self._called(request, "search_files") and self._has_tool(request, "search_files"):
            return _call("search_files", pattern=r"return a|assert", glob="*.py")
        if not self._called(request, "post_finding") and self._has_tool(request, "post_finding"):
            return _call(
                "post_finding",
                title="(Demo) add() subtracts instead of adding",
                content="calculator.py: add() returns a - b, tests/test_calculator.py expects add(2, 3) == 5.",
                kind="finding",
                files=["calculator.py:7", "tests/test_calculator.py:5"],
                importance=5,
            )
        return _json(
            {
                "summary": "(Demo) add() is wrong; divide() has no guard against division by zero.",
                "findings": [
                    {"title": "Wrong operator in add()", "detail": "return a - b instead of a + b", "files": ["calculator.py:7"], "severity": "high"},
                    {"title": "Division by zero", "detail": "divide(1, 0) raises ZeroDivisionError", "files": ["calculator.py:12"], "severity": "low"},
                ],
                "risks": ["Other callers might rely on the wrong behaviour"],
                "recommendations": ["Fix the operator and run the tests"],
            }
        )

    def on_plan_draft(self, request: CompletionRequest) -> str:
        return _json(PLAN)

    def on_plan_critique(self, request: CompletionRequest) -> str:
        agent = str(request.metadata.get("agent") or "")
        if agent.startswith("critic") and "round 1" in str(request.metadata.get("task_title") or ""):
            return _json(
                {
                    "verdict": "revise",
                    "score": 6,
                    "strengths": ["Clear ordering"],
                    "issues": [{"severity": "minor", "step": "s2", "issue": "The division-by-zero edge case is not considered", "suggestion": "Document it as a risk", "evidence": "calculator.py:12"}],
                    "missing_steps": [],
                }
            )
        return _json({"verdict": "approve", "score": 8, "strengths": ["Minimal, verifiable plan"], "issues": [], "missing_steps": []})

    def on_plan_revise(self, request: CompletionRequest) -> str:
        return _json({**PLAN, "changes": ["Division-by-zero risk documented"], "rejected_feedback": []})

    on_plan_feedback = on_plan_revise

    def on_plan_judge(self, request: CompletionRequest) -> str:
        return _json({**PLAN, "decision_rationale": "(Demo) A minimal fix verified by the tests is enough."})

    # ------------------------------------------------------------------ execution

    def on_execute_analysis(self, request: CompletionRequest) -> CompletionResponse | str:
        if not self._called(request, "read_file") and self._has_tool(request, "read_file"):
            return _call("read_file", path="calculator.py")
        return _json({"status": "done", "summary": "(Demo) Cause: calculator.py line 7 uses '-' instead of '+'.", "details": "add() must return a + b.", "files_changed": [], "issues": []})

    def on_execute_implementation(self, request: CompletionRequest) -> CompletionResponse | str:
        return self._fix_bug(request, "Implementation")

    def on_execute_fix(self, request: CompletionRequest) -> CompletionResponse | str:
        return self._fix_bug(request, "Fix")

    def on_execute_debug(self, request: CompletionRequest) -> str:
        return _json({"root_cause": "(Demo) add() subtracts", "affected_files": ["calculator.py:7"], "fix_instructions": "replace return a - b with return a + b", "confidence": 0.9})

    def on_execute_test(self, request: CompletionRequest) -> str:
        return _json({"status": "done", "summary": "(Demo) Checked manually.", "files_changed": [], "issues": []})

    def on_execute_review(self, request: CompletionRequest) -> str:
        return _json({"verdict": "approve", "summary": "(Demo) The change is correct and minimal.", "issues": [{"severity": "minor", "file": "calculator.py", "line": 12, "issue": "divide() has no guard for b == 0", "suggestion": "Handle it explicitly later"}]})

    on_execute_security = on_execute_review

    def on_execute_synthesis(self, request: CompletionRequest) -> str:
        text = request.messages[0].content if request.messages else ""
        fixed = 1 if "calculator.py" in text and "modified" in text else 0
        return _json(
            {
                "summary": "(Demo) The bug in add() is fixed and the test suite passes completely. divide() still does not handle division by zero explicitly.",
                "bugs_fixed": fixed,
                "highlights": ["add() fixed", "tests passed"],
                "remaining_issues": ["divide(…, 0) raises ZeroDivisionError"],
                "recommendations": ["Decide on the behaviour for division by zero"],
            }
        )

    # ------------------------------------------------------------------ debate

    def on_debate_propose(self, request: CompletionRequest) -> CompletionResponse | str:
        agent = str(request.metadata.get("agent") or "")
        if not self._called(request, "read_file") and self._has_tool(request, "read_file"):
            return _call("read_file", path="calculator.py")
        if len(agent) % 2 == 0:
            return _json({"title": "(Demo) Raise ValueError", "approach": "Raise ValueError with a clear message when b == 0", "rationale": "Explicit and testable", "pros": ["clear"], "cons": ["API change"], "risks": [], "evidence": ["calculator.py:12"], "confidence": 0.7})
        return _json({"title": "(Demo) Document ZeroDivisionError", "approach": "Keep the behaviour, document it in the docstring and add a test", "rationale": "No behaviour change", "pros": ["compatible"], "cons": ["less explicit"], "risks": [], "evidence": ["Python default behaviour"], "confidence": 0.6})

    def on_debate_critique(self, request: CompletionRequest) -> str:
        return _json(
            {
                "assessments": [
                    {"proposal_id": "P1", "stance": "support", "severity": "none", "arguments": ["(Demo) reasonable"], "evidence": [], "suggested_changes": []},
                    {"proposal_id": "P2", "stance": "amend", "severity": "minor", "arguments": ["(Demo) A test is still missing"], "evidence": [], "suggested_changes": ["Add a test"]},
                ]
            }
        )

    def on_debate_revise(self, request: CompletionRequest) -> str:
        return _json({"title": "(Demo) revised", "approach": "Raise ValueError when b == 0 and add a test", "rationale": "Critique taken into account", "confidence": 0.75, "withdrawn": False, "endorse": None})

    def on_debate_vote(self, request: CompletionRequest) -> str:
        return _json({"votes": [{"proposal_id": "P1", "score": 8, "rationale": "(Demo)"}, {"proposal_id": "P2", "score": 6, "rationale": "(Demo)"}], "preferred": "P1"})

    def on_debate_judge(self, request: CompletionRequest) -> str:
        return _json({"chosen_proposal": "P1", "decision": "(Demo) Raise ValueError when b == 0 and add a test", "rationale": "Explicit behaviour is easier to maintain", "evidence_checked": ["read calculator.py"], "rejected": [{"proposal_id": "P2", "reason": "less explicit"}], "confidence": 0.8, "open_risks": ["Callers must handle ValueError"]})

    # ------------------------------------------------------------------ misc

    def on_consult(self, request: CompletionRequest) -> str:
        return "(Demo) Answer: I recommend the minimal fix, verified by the tests afterwards."

    def on_compaction(self, request: CompletionRequest) -> str:
        return "- (Demo) Summary of the progress so far"

    def on_generic(self, request: CompletionRequest) -> str:
        if request.metadata.get("output") == "json":
            return _json({"status": "done", "summary": "(Demo) done", "files_changed": [], "issues": []})
        return "(Demo) done"


def looks_like_demo_text(text: str) -> bool:
    return bool(re.search(r"\(Demo\)", text or ""))
