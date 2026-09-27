"""Registry of all tools and filtering by agent groups, permissions and mode."""

from __future__ import annotations

from mao.config.schema import PermissionSet
from mao.core.errors import ConfigError
from mao.core.types import RunMode
from mao.security.permissions import missing_capabilities
from mao.tools.base import Tool

TOOL_GROUPS = ("filesystem", "terminal", "web", "git", "tests", "collaboration")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ConfigError(f"Tool registered twice: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def groups(self) -> dict[str, list[Tool]]:
        grouped: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            grouped.setdefault(tool.group, []).append(tool)
        return grouped

    def available_for(
        self,
        *,
        groups: list[str],
        permissions: PermissionSet,
        mode: RunMode,
        enabled_groups: list[str],
    ) -> list[Tool]:
        wanted = set(groups) & set(enabled_groups)
        return [
            tool
            for tool in self._tools.values()
            if tool.group in wanted
            and not missing_capabilities(permissions, tool.required)
            and tool.available_in(mode)
        ]


def build_default_registry() -> ToolRegistry:
    from mao.tools import collaboration, filesystem, git_tools, terminal, tests_tool, web

    registry = ToolRegistry()
    for module in (filesystem, terminal, web, git_tools, tests_tool, collaboration):
        for tool_cls in module.TOOLS:
            registry.register(tool_cls())
    return registry
