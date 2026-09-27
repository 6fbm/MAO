"""Token estimation for budgeting before a call and when providers report no usage.

Deliberately a conservative character heuristic: exact tokenizers differ per
provider and are not available offline for most of them. Reported usage from
the provider always replaces the estimate afterwards.
"""

from __future__ import annotations

import json
import math

from mao.core.types import ChatMessage, ToolSpec


class TokenEstimator:
    chars_per_token: float = 3.5
    message_overhead: int = 4

    def count_text(self, text: str | None) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)

    def count_message(self, message: ChatMessage) -> int:
        total = self.message_overhead + self.count_text(message.content)
        for call in message.tool_calls:
            total += 8 + self.count_text(call.name) + self.count_text(json.dumps(call.arguments, ensure_ascii=False))
        return total

    def count_tools(self, tools: list[ToolSpec]) -> int:
        return sum(
            12 + self.count_text(t.name) + self.count_text(t.description) + self.count_text(json.dumps(t.parameters))
            for t in tools
        )

    def count_request(
        self,
        messages: list[ChatMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        total = self.count_text(system) + self.message_overhead
        total += sum(self.count_message(m) for m in messages)
        if tools:
            total += self.count_tools(tools)
        return total


ESTIMATOR = TokenEstimator()
