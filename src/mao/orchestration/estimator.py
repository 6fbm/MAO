"""Token and cost estimation for a plan before it is executed."""

from __future__ import annotations

from mao.agents.manager import AgentManager
from mao.core.errors import ConfigError
from mao.models.catalog import ModelCatalog
from mao.orchestration.plan import PlanDocument, PlanEstimate
from mao.orchestration.taskgraph import NodeKind
from mao.tokens.tracker import UsageTracker, estimate_cost

# rough (input, output) tokens for a step of complexity 3, including its tool loop
KIND_TOKENS: dict[NodeKind, tuple[int, int]] = {
    NodeKind.ANALYSIS: (30_000, 3_000),
    NodeKind.RESEARCH: (35_000, 4_000),
    NodeKind.DESIGN: (25_000, 4_000),
    NodeKind.DECISION: (90_000, 14_000),
    NodeKind.IMPLEMENTATION: (70_000, 12_000),
    NodeKind.TEST: (45_000, 7_000),
    NodeKind.REVIEW: (40_000, 5_000),
    NodeKind.SECURITY: (35_000, 4_000),
    NodeKind.DOCUMENTATION: (25_000, 6_000),
    NodeKind.SYNTHESIS: (15_000, 3_000),
}
COMPLEXITY_FACTOR = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.6, 5: 2.5}


class PlanEstimator:
    def __init__(self, catalog: ModelCatalog, manager: AgentManager, tracker: UsageTracker) -> None:
        self.catalog = catalog
        self.manager = manager
        self.tracker = tracker

    def estimate(self, plan: PlanDocument) -> PlanEstimate:
        estimate = PlanEstimate()
        unpriced: set[str] = set()
        for node in plan.graph.nodes:
            base_in, base_out = KIND_TOKENS.get(node.kind, (30_000, 4_000))
            factor = COMPLEXITY_FACTOR.get(node.complexity, 1.0)
            input_tokens, output_tokens = int(base_in * factor), int(base_out * factor)
            agent = self.manager.get(plan.assignments.get(node.id, "")) if plan.assignments.get(node.id) else None
            cost = 0.0
            if agent is not None and agent.model_ref:
                try:
                    entry = self.catalog.resolve(agent.model_ref)
                    if entry.config.pricing.is_free and not entry.local:
                        unpriced.add(entry.ref)
                    cost = estimate_cost(input_tokens, output_tokens, entry.config.pricing)
                except ConfigError:
                    pass
            estimate.input_tokens += input_tokens
            estimate.output_tokens += output_tokens
            estimate.cost_usd += cost
        estimate.low_tokens = int(estimate.total_tokens * 0.5)
        estimate.high_tokens = int(estimate.total_tokens * 2.0)
        estimate.low_cost_usd = estimate.cost_usd * 0.5
        estimate.high_cost_usd = estimate.cost_usd * 2.0
        totals = self.tracker.totals()
        estimate.planning_tokens_used = totals.total_tokens
        estimate.planning_cost_used = totals.cost_usd
        estimate.unpriced_models = sorted(unpriced)
        return estimate
