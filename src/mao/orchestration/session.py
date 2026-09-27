"""SessionRuntime: wires every per-session component together."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mao.agents.agent import AgentTask
from mao.agents.manager import AgentManager
from mao.agents.roles import RoleRegistry
from mao.agents.runtime import AgentRuntime, RuntimeServices
from mao.config.schema import AppConfig, PermissionSet, Settings
from mao.context.compaction import ConversationCompactor
from mao.core.errors import ConfigError, NoAvailableKeyError, ToolError
from mao.core.types import RunMode
from mao.messaging.blackboard import Blackboard
from mao.messaging.bus import MessageBus
from mao.messaging.hub import CollaborationHub
from mao.messaging.router import ContextRouter
from mao.models.selector import ModelSelector
from mao.orchestration.assignment import AgentAssigner
from mao.orchestration.control import RunControl
from mao.orchestration.debate import DebateEngine
from mao.orchestration.estimator import PlanEstimator
from mao.orchestration.plan import PlanDocument
from mao.providers.gateway import LLMGateway
from mao.providers.gemini import GeminiProvider
from mao.security.approval import ApprovalGateway, ApprovalHandler
from mao.security.sandbox import WorkspaceSandbox
from mao.sessions.logger import SessionLogger
from mao.sessions.store import Session
from mao.tokens.budget import BudgetGuard, LimitHandler
from mao.tokens.tracker import UsageRecord, UsageTracker, compute_cost
from mao.tools.base import ToolContext
from mao.tools.executor import ToolExecutor
from mao.tools.tests_tool import TestRunSummary
from mao.tools.web import WebSearcher
from mao.workspace.changes import ChangeTracker
from mao.workspace.git import GitRepo
from mao.workspace.workspace import ProjectProfile, scan_workspace

if TYPE_CHECKING:
    from mao.app import AppContext


class SessionRuntime:
    def __init__(
        self,
        app: AppContext,
        session: Session,
        workspace: Path,
        *,
        approval_handler: ApprovalHandler,
        limit_handler: LimitHandler | None,
    ) -> None:
        self.app = app
        self.session = session
        self.workspace = workspace
        config = app.config
        settings = config.settings
        hub = app.hub
        self.bus = app.bus
        self.bus.session_id = session.id
        self.redactor = app.redactor

        self.sandbox = WorkspaceSandbox(
            workspace,
            protected_patterns=config.permissions.protected_patterns,
            sensitive_patterns=config.permissions.sensitive_patterns,
        )
        self.tracker = UsageTracker()
        self._load_usage()
        self.control = RunControl()
        self.budget = BudgetGuard(settings.limits, self.tracker, self.bus, limit_handler)
        self.gateway = LLMGateway(
            providers=hub.providers,
            catalog=hub.catalog,
            keypools=hub.keypools,
            limiters=hub.limiters,
            tracker=self.tracker,
            bus=self.bus,
            budget=self.budget,
            control=self.control,
        )
        self.messages = MessageBus(self.bus, self.redactor, session.path("messages.jsonl"))
        self.messages.load()
        self.board = Blackboard(self.redactor, session.path("blackboard.json"))
        self.board.load()
        self.changes = ChangeTracker(self.sandbox, session.backups_dir, session.path("changes.json"))
        self.changes.load()
        self.approval = ApprovalGateway(
            config.permissions.approval, approval_handler, self.bus, auto_approve=config.permissions.auto_approve
        )
        self.selector = ModelSelector(hub.catalog, settings.orchestration.model_policy)
        self.roles = RoleRegistry(config.roles.roles)
        self.manager = AgentManager(
            agents_config=config.agents,
            roles=self.roles,
            catalog=hub.catalog,
            selector=self.selector,
            settings=settings,
            permissions=config.permissions,
            bus=self.bus,
        )
        self.agent_warnings = self.manager.build()
        self.compactor = ConversationCompactor(
            gateway=self.gateway, catalog=hub.catalog, selector=self.selector, config=settings.context, bus=self.bus
        )
        self.router = ContextRouter(self.board, self.messages)
        self.executor = ToolExecutor(app.tool_registry, redactor=self.redactor, max_result_chars=settings.context.max_tool_result_chars)
        self.collaboration = CollaborationHub(
            self.messages,
            self.board,
            consult_handler=self._consult,
            max_consultations_per_task=settings.limits.max_consultations_per_task,
        )
        self.web = WebSearcher(config.tools.web, app.http, grounded_search=self._grounded_search)
        self.git: GitRepo | None = None
        self.test_summaries: list[TestRunSummary] = []
        self.final_synthesis: dict | None = None
        self.profile: ProjectProfile | None = None
        self.services = RuntimeServices(
            gateway=self.gateway,
            executor=self.executor,
            router=self.router,
            compactor=self.compactor,
            manager=self.manager,
            bus=self.bus,
            control=self.control,
            settings=settings,
            tools_config=config.tools,
            permissions_config=config.permissions,
            sandbox=self.sandbox,
            approval=self.approval,
            changes=self.changes,
            http=app.http,
            collaboration=self.collaboration,
            web=self.web,
            on_test_result=self.record_test_summary,
            mode=RunMode.PLAN,
        )
        self.runtime = AgentRuntime(self.services)
        self.assigner = AgentAssigner()
        self.debate = DebateEngine(
            runtime=self.runtime,
            manager=self.manager,
            assigner=self.assigner,
            board=self.board,
            messages=self.messages,
            bus=self.bus,
            settings=settings,
            decisions_path=session.path("decisions.json"),
        )
        self.estimator = PlanEstimator(hub.catalog, self.manager, self.tracker)
        self.logger = SessionLogger(session.dir, session.id, self.bus, self.redactor, self.tracker, level=settings.logging.level)

        from mao.orchestration.planner import PlanningPipeline
        from mao.orchestration.quality import QualityLoops
        from mao.orchestration.scheduler import GraphScheduler

        self.planner = PlanningPipeline(self)
        self.quality = QualityLoops(self)
        self.scheduler = GraphScheduler(self)
        self.plan: PlanDocument | None = session.load_plan()
        self.opened_at = time.time()

    @classmethod
    async def open(
        cls,
        app: AppContext,
        session: Session,
        workspace: Path,
        *,
        approval_handler: ApprovalHandler,
        limit_handler: LimitHandler | None,
    ) -> SessionRuntime:
        runtime = cls(app, session, workspace, approval_handler=approval_handler, limit_handler=limit_handler)
        if app.config.tools.git.enabled:
            runtime.git = await GitRepo.detect(workspace)
            runtime.services.git = runtime.git
        return runtime

    # ------------------------------------------------------------------ properties

    @property
    def config(self) -> AppConfig:
        return self.app.config

    @property
    def settings(self) -> Settings:
        return self.app.config.settings

    @property
    def mode(self) -> RunMode:
        return self.services.mode

    def set_mode(self, mode: RunMode) -> None:
        self.services.mode = mode

    def reset_control(self) -> None:
        """A stopped run cannot be resumed with the same control object."""
        if self.control.stopped:
            self.control = RunControl()
            self.gateway.control = self.control
            self.services.control = self.control

    # ------------------------------------------------------------------ helpers

    def _load_usage(self) -> None:
        path = self.session.path("tokens.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.tracker.extend([UsageRecord.model_validate(r) for r in data.get("records", [])])
        except (json.JSONDecodeError, ValueError, OSError):
            pass

    def record_test_summary(self, summary: TestRunSummary) -> None:
        self.test_summaries.append(summary)

    async def ensure_profile(self) -> ProjectProfile:
        if self.profile is None:
            profile = await asyncio.to_thread(scan_workspace, self.workspace, self.config.tools.filesystem.ignore_patterns)
            if self.git is not None:
                try:
                    profile.git_branch = await self.git.current_branch()
                    status = await self.git.status()
                    profile.git_status = "\n".join(status.splitlines()[:30])
                except ToolError:
                    pass
            self.profile = profile
            if not self.board.entries("workspace"):
                self.board.add("orchestrator", "workspace", "Workspace overview", profile.render(), importance=3)
        return self.profile

    def system_tool_context(self) -> ToolContext:
        """Context for deterministic, orchestrator-driven tool use (e.g. test runs)."""
        return ToolContext(
            agent_name="orchestrator",
            agent_role="orchestrator",
            permissions=PermissionSet(read=True, execute=True, git="read"),
            mode=self.mode,
            sandbox=self.sandbox,
            approval=self.approval,
            changes=self.changes,
            bus=self.bus,
            tools_config=self.config.tools,
            permissions_config=self.config.permissions,
            http=self.app.http,
            collaboration=self.collaboration,
            git=self.git,
            web=self.web,
            on_test_result=self.record_test_summary,
        )

    async def _consult(self, requester: str, target: str, question: str, node_id: str | None, depth: int) -> str:
        agent = self.manager.find(target, exclude=requester)
        if agent is None:
            raise ToolError(f"No available agent '{target}' for a follow-up question")
        task = AgentTask(
            title=f"Question from {requester}",
            instructions=f'{requester} asks you:\n"""{question}"""\n\nAnswer precisely and briefly. Check facts with read-only tools when you need to.',
            purpose="consult",
            readonly=True,
            max_steps=6,
            node_id=node_id,
            consult_depth=depth,
            shared_context_query=question,
        )
        result = await self.runtime.run(agent, task, track_state=not agent.busy)
        if not result.text and result.error:
            return f"(question to {agent.name} failed: {result.error})"
        return f"Response from {agent.name}:\n{result.text}"

    async def _grounded_search(self, query: str) -> tuple[str, list[dict[str, str]]]:
        hub = self.app.hub
        model_ref = self.config.tools.web.gemini_search_model or "gemini"
        try:
            entry = hub.catalog.resolve(model_ref)
        except ConfigError as exc:
            raise ToolError(f"Gemini search is not configured: {exc}") from exc
        provider = hub.providers.get(entry.provider)
        if not isinstance(provider, GeminiProvider) or not hub.catalog.is_available(entry):
            raise ToolError("Gemini search unavailable (provider or key missing)")
        await self.budget.check(est_tokens=2_000)
        pool = hub.keypools[entry.provider]
        try:
            key = pool.acquire()
        except NoAvailableKeyError as exc:
            raise ToolError(str(exc)) from exc
        try:
            text, sources, usage = await provider.grounded_search(
                query, api_key=key.key if key else None, model_id=entry.api_id, timeout=entry.provider_config.timeout_s
            )
        finally:
            pool.release(key)
        self.tracker.record(
            UsageRecord(
                agent="web_search",
                provider=entry.provider,
                model=entry.api_id,
                purpose="research.grounded_search",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=compute_cost(usage, entry.config.pricing),
            )
        )
        return text, sources

    def save(self) -> None:
        totals = self.tracker.totals()
        self.session.meta.total_tokens = totals.total_tokens
        self.session.meta.cost_usd = totals.cost_usd
        self.session.save_meta()
        if self.plan is not None:
            self.session.save_plan(self.plan)
        self.logger.write_tokens()

    async def close(self) -> None:
        self.save()
        self.logger.close()
        if self.bus.session_id == self.session.id:
            self.bus.session_id = None
