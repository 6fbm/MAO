"""Instructions and JSON output structures for all orchestration phases.

Instructions are written in English (most reliable across models); agents are
told separately to write human-readable values in the configured language.
"""

from __future__ import annotations

import json

from mao.agents.agent import Agent

TRIAGE_SCHEMA = json.dumps(
    {
        "understanding": "What the user wants and what 'done' means (2-4 sentences)",
        "task_type": "code_change | analysis | research | documentation | mixed",
        "complexity": "1-5 (1 = trivial, 5 = very large)",
        "modifies_files": True,
        "needs_research": False,
        "investigations": [{"role": "architect", "focus": "What to investigate and why", "questions": ["..."]}],
        "clarifications": ["Questions for the user that block good planning (empty if none)"],
    },
    indent=1,
)

INVESTIGATION_SCHEMA = json.dumps(
    {
        "summary": "Key result of the investigation (3-6 sentences)",
        "findings": [{"title": "...", "detail": "...", "files": ["path/to/file.py:42"], "severity": "info | low | medium | high"}],
        "risks": ["..."],
        "recommendations": ["..."],
    },
    indent=1,
)

PLAN_SCHEMA = json.dumps(
    {
        "title": "Short plan title",
        "summary": "2-4 sentences",
        "approach": "How the task will be solved and why this approach",
        "steps": [
            {
                "id": "s1",
                "title": "Short step title",
                "description": "Self-contained, concrete instructions for the agent executing this step (files, functions, expected result)",
                "kind": "analysis | research | design | decision | implementation | test | review | security | documentation | synthesis",
                "role": "one of the team roles",
                "depends_on": [],
                "modifies_files": False,
                "acceptance_criteria": ["Verifiable criterion"],
                "complexity": "1-5",
            }
        ],
        "risks": [{"risk": "...", "severity": "low | medium | high", "mitigation": "..."}],
        "assumptions": ["..."],
        "open_questions": ["..."],
        "success_criteria": ["..."],
    },
    indent=1,
)

CRITIQUE_SCHEMA = json.dumps(
    {
        "verdict": "approve | revise | reject",
        "score": "1-10",
        "strengths": ["..."],
        "issues": [
            {"severity": "critical | major | minor", "step": "step id or null", "issue": "...", "suggestion": "...", "evidence": "file path, URL or reasoning"}
        ],
        "missing_steps": ["..."],
    },
    indent=1,
)

PROPOSAL_SCHEMA = json.dumps(
    {
        "title": "Short name of the proposal",
        "approach": "Concrete solution",
        "rationale": "Why this is the best option",
        "pros": ["..."],
        "cons": ["..."],
        "risks": ["..."],
        "evidence": ["file path, documentation URL, measurement"],
        "confidence": "0.0-1.0",
    },
    indent=1,
)

PROPOSAL_REVISION_SCHEMA = PROPOSAL_SCHEMA[:-2] + ',\n "withdrawn": false,\n "endorse": "proposal id you now support instead, or null"\n}'

ASSESSMENT_SCHEMA = json.dumps(
    {
        "assessments": [
            {
                "proposal_id": "P1",
                "stance": "support | oppose | amend",
                "severity": "critical | major | minor | none",
                "arguments": ["..."],
                "evidence": ["..."],
                "suggested_changes": ["..."],
            }
        ]
    },
    indent=1,
)

VOTE_SCHEMA = json.dumps(
    {"votes": [{"proposal_id": "P1", "score": "1-10", "rationale": "..."}], "preferred": "P1"},
    indent=1,
)

JUDGE_SCHEMA = json.dumps(
    {
        "chosen_proposal": "P1 | P2 | merged",
        "decision": "The final, concrete approach to implement",
        "rationale": "Why – based on the verified evidence",
        "evidence_checked": ["What you verified and the result"],
        "rejected": [{"proposal_id": "P2", "reason": "..."}],
        "confidence": "0.0-1.0",
        "open_risks": ["..."],
    },
    indent=1,
)

NODE_RESULT_SCHEMA = json.dumps(
    {
        "status": "done | blocked | failed",
        "summary": "What you did and the result (2-5 sentences)",
        "details": "Details the following steps need (decisions, exact locations, commands)",
        "files_changed": ["relative/path"],
        "issues": ["Remaining problems or risks"],
        "next_steps": ["..."],
    },
    indent=1,
)

DIAGNOSIS_SCHEMA = json.dumps(
    {
        "root_cause": "Precise root cause",
        "affected_files": ["path:line"],
        "fix_instructions": "Concrete steps to fix it",
        "confidence": "0.0-1.0",
    },
    indent=1,
)

REVIEW_SCHEMA = json.dumps(
    {
        "verdict": "approve | request_changes",
        "summary": "Overall assessment",
        "issues": [{"severity": "critical | major | minor", "file": "path", "line": None, "issue": "...", "suggestion": "..."}],
    },
    indent=1,
)

SYNTHESIS_SCHEMA = json.dumps(
    {
        "summary": "Final report for the user: what was done, what is verified, what is open (markdown allowed)",
        "bugs_fixed": 0,
        "highlights": ["..."],
        "remaining_issues": ["..."],
        "recommendations": ["..."],
    },
    indent=1,
)


def team_overview(agents: list[Agent]) -> str:
    lines = []
    for agent in agents:
        if not agent.available or agent.internal:
            continue
        lines.append(
            f"- {agent.name}: role={agent.role.name}; capabilities={', '.join(agent.capabilities)}; "
            f"permissions={agent.permissions.describe()}; model={agent.model_ref}"
        )
    return "\n".join(lines) or "- (no agents available)"


def triage_instructions(task: str, team: str, max_investigations: int) -> str:
    return f"""User task:
\"\"\"{task}\"\"\"

Team:
{team}

Understand the task. Take a quick look at the workspace (list_directory, key files) – do not analyze everything yet.
Then decide which specialist investigations are needed BEFORE a plan can be written (0 to {max_investigations}).
Use none for trivial tasks. Each investigation must name a team role and a precise focus. Prefer roles whose
capabilities fit (e.g. researcher only if external information is really needed)."""


def investigation_instructions(task: str, understanding: str, focus: str, questions: list[str]) -> str:
    question_text = "\n".join(f"- {q}" for q in questions) or "- (derive them from the focus)"
    return f"""Overall user task:
\"\"\"{task}\"\"\"

Understanding so far: {understanding or '-'}

Your investigation focus: {focus}
Questions:
{question_text}

Gather evidence with tools (read the relevant files; research only if you have internet access and it is needed).
Publish the most important findings with post_finding (with file paths / URLs). Do not change anything."""


def plan_instructions(task: str, understanding: str, team: str, *, max_steps: int = 12, feedback: str | None = None) -> str:
    feedback_text = f"\nUser feedback that the plan must address:\n\"\"\"{feedback}\"\"\"\n" if feedback else ""
    return f"""User task:
\"\"\"{task}\"\"\"

Understanding: {understanding or '-'}

Team (use these roles for the steps):
{team}
{feedback_text}
Create an executable plan using the investigation findings in the shared context (read_board for details).
Guidelines:
- At most {max_steps} steps. No busywork. Each description must be self-contained and concrete (files, functions, expected result).
- Steps that do not depend on each other must not declare dependencies, so they can run in parallel.
- Use kind "decision" only for a genuine choice between alternatives that deserves a team debate.
- If code changes, include implementation steps (modifies_files=true), a "test" step and a "review" step; add "security" if relevant.
- Choose a role for every step whose permissions allow it (implementation needs write permission, tests need execute).
- End with a "synthesis" step that summarizes the result for the user.
- State real risks with mitigations; list assumptions and open questions honestly."""


def critique_instructions(task: str, plan_json: str, round_number: int) -> str:
    return f"""User task:
\"\"\"{task}\"\"\"

Proposed plan (round {round_number}):
{plan_json}

Critically review this plan. Verify important assumptions against the actual workspace with tools.
Check: does it fully solve the task? Are steps concrete, correctly ordered and parallelized? Missing tests,
reviews, edge cases or risks? Unnecessary steps? Wrong roles/permissions?
Verdict "approve" only if the plan is good enough to execute; use "critical" only for issues that would make it fail."""


def revision_instructions(task: str, plan_json: str, critiques: str, feedback: str | None = None) -> str:
    feedback_text = f"\nUser feedback (highest priority):\n\"\"\"{feedback}\"\"\"\n" if feedback else ""
    return f"""User task:
\"\"\"{task}\"\"\"

Current plan:
{plan_json}

Critiques from the team:
{critiques or '(none)'}
{feedback_text}
Revise the plan. Address every valid point; reject invalid feedback with a short reason. Return the COMPLETE
revised plan in the required structure, plus "changes" (list of what you changed) and "rejected_feedback"
(list of objects with issue and reason)."""


def judge_plan_instructions(task: str, plan_json: str, critiques: str) -> str:
    return f"""User task:
\"\"\"{task}\"\"\"

The planning team did not reach consensus. Current plan:
{plan_json}

Unresolved critiques:
{critiques}

As the judge, weigh the arguments on evidence (verify disputed facts with tools). Produce the FINAL plan in the
required structure, incorporating the justified critiques, plus "decision_rationale" explaining the key choices."""


def proposal_instructions(question: str, context: str) -> str:
    return f"""Decision to make:
\"\"\"{question}\"\"\"

Context:
{context or '(see shared context)'}

Propose the best solution you can defend with evidence. Inspect the relevant files (and documentation if you
have internet access). Be concrete about how it would be implemented."""


def assessment_instructions(question: str, proposals: str) -> str:
    return f"""Decision under debate:
\"\"\"{question}\"\"\"

Proposals:
{proposals}

Assess EVERY proposal: support, oppose or amend, with arguments and evidence. Verify factual claims with tools
when they matter. Use severity "critical" only for flaws that make a proposal unacceptable."""


def proposal_revision_instructions(question: str, own: str, assessments: str, others: str) -> str:
    return f"""Decision under debate:
\"\"\"{question}\"\"\"

Your proposal:
{own}

Assessments by the critics:
{assessments}

Other proposals:
{others}

Revise your proposal to address valid criticism. If another proposal is clearly better, set withdrawn=true and
endorse its id. Return the full proposal structure."""


def vote_instructions(question: str, proposals: str, assessments: str) -> str:
    return f"""Decision:
\"\"\"{question}\"\"\"

Final proposals:
{proposals}

Summary of the critiques:
{assessments}

Score every proposal from 1 (bad) to 10 (excellent) based on correctness, risk, effort and fit to the project."""


def judge_instructions(question: str, context: str, proposals: str, assessments: str, votes: str) -> str:
    return f"""Decision:
\"\"\"{question}\"\"\"

Context:
{context or '(see shared context)'}

Proposals:
{proposals}

Critiques:
{assessments}

Votes (weighted totals):
{votes}

You are the judge. Do not simply follow the majority: check the decisive evidence yourself with tools (read the
cited files or documentation), consider the counter-arguments and choose – or merge – the best solution.
If you need expert input, you may use consult_agent once."""


def node_instructions(task: str, plan_title: str, approach: str, node_title: str, node_id: str, description: str, criteria: list[str], modifies_files: bool) -> str:
    criteria_text = "\n".join(f"- {c}" for c in criteria) or "- (none given – use common sense)"
    change_rule = (
        "This step is expected to change files. Make the changes directly with the file tools."
        if modifies_files
        else "This step should not change files unless it is strictly necessary to fulfil it."
    )
    return f"""Overall user task:
\"\"\"{task}\"\"\"

Plan: {plan_title}
Approach: {approach[:1500]}

Your step {node_id}: {node_title}
{description}

Acceptance criteria:
{criteria_text}

{change_rule}
Use the results of previous steps in the shared context. When done, report honestly; use status "blocked" or
"failed" if you could not complete the step."""


def diagnosis_instructions(command: str, output: str, attempt: int) -> str:
    return f"""The test suite fails (fix attempt {attempt}).
$ {command}
{output}

Find the root cause. Read the failing tests and the code under test. Do not change files – produce a precise diagnosis
and fix instructions for the coder."""


def fix_instructions(command: str, output: str, diagnosis: str) -> str:
    return f"""Fix the failing tests.

Diagnosis from the debugger:
{diagnosis or '(no diagnosis available – analyze yourself)'}

Test output:
$ {command}
{output}

Rules: fix the code under test (or genuinely wrong tests); never delete or weaken tests just to make them pass.
Run the tests yourself afterwards if you can."""


def review_instructions(node_title: str, description: str, diff: str, previous: str | None) -> str:
    previous_text = f"\nIssues raised in the previous review round:\n{previous}\n" if previous else ""
    return f"""Review task: {node_title}
{description}
{previous_text}
Diff of all changes made in this session:
{diff}

Read surrounding code where needed. Verdict "request_changes" only for critical or major problems that must be
fixed before the work can be accepted."""


def review_fix_instructions(issues: str) -> str:
    return f"""Reviewers requested changes. Fix these issues (keep changes focused):
{issues}

If an issue is not valid, explain why in your summary instead of changing code."""


def synthesis_instructions(task: str, plan_title: str, results: str, changes: str, tests: str, decisions: str) -> str:
    return f"""User task:
\"\"\"{task}\"\"\"

Plan: {plan_title}

Results of all steps:
{results}

File changes:
{changes}

Tests:
{tests}

Decisions:
{decisions}

Write the final report for the user. Be honest: distinguish verified results (e.g. passing tests) from unverified
claims, and list what remains open. Count bugs_fixed only for bugs that were actually fixed."""
