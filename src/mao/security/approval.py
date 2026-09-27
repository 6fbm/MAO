"""Approval gateway: decides whether an action may run, asking the user if needed."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from mao.config.schema import ApprovalRules
from mao.core.errors import ApprovalDeniedError
from mao.core.events import ApprovalEvent, EventBus
from mao.security.risk import ActionKind, ProposedAction, RiskLevel


class ApprovalDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_SESSION = "allow_session"


@dataclass
class ApprovalRequest:
    action: ProposedAction
    rule: str


ApprovalHandler = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


async def deny_all(_request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.DENY


async def allow_all(_request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision.ALLOW


class ApprovalGateway:
    def __init__(
        self,
        rules: ApprovalRules,
        handler: ApprovalHandler,
        bus: EventBus,
        *,
        auto_approve: bool = False,
    ) -> None:
        self.rules = rules
        self.handler = handler
        self.bus = bus
        self.auto_approve = auto_approve
        self._lock = asyncio.Lock()
        self._session_allows: set[str] = set()

    def effective_rule(self, action: ProposedAction) -> str:
        if action.kind is ActionKind.READ:
            return "allow"
        rule = getattr(self.rules, action.kind.value, "ask")
        if rule == "ask_risky":
            rule = "ask" if action.risk >= RiskLevel.MEDIUM else "allow"
        # critical actions are never silently allowed
        if action.risk >= RiskLevel.CRITICAL and rule == "allow":
            rule = "ask"
        return rule

    def _publish(self, action: ProposedAction, decision: str) -> None:
        self.bus.publish(
            ApprovalEvent(
                agent=action.agent,
                action=action.kind.value,
                target=action.target,
                risk=action.risk.label,
                decision=decision,
            )
        )

    async def require(self, action: ProposedAction) -> None:
        """Return if the action may proceed, raise ``ApprovalDeniedError`` otherwise."""
        rule = self.effective_rule(action)
        if rule == "allow":
            return
        if rule == "deny":
            self._publish(action, "denied_by_policy")
            raise ApprovalDeniedError(
                f"Action forbidden by policy ({action.kind.value}): {action.target}"
            )
        session_key = action.kind.value
        critical = action.risk >= RiskLevel.CRITICAL
        if not critical and self.auto_approve:
            self._publish(action, "auto_approved")
            return
        if not critical and session_key in self._session_allows:
            return
        async with self._lock:
            if not critical and session_key in self._session_allows:
                return
            decision = await self.handler(ApprovalRequest(action=action, rule=rule))
            if decision is ApprovalDecision.ALLOW_SESSION and not critical:
                self._session_allows.add(session_key)
        self._publish(action, decision.value)
        if decision is ApprovalDecision.DENY:
            raise ApprovalDeniedError(f"Rejected by the user ({action.kind.value}): {action.target}")
