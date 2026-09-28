"""CLI commands. Importing this package registers every command."""

from mao.cli.commands import agents, chat, config_cmd, general, git_cmd, providers, sessions, task, usage  # noqa: F401
from mao.cli.commands.base import REGISTRY, CommandContext, CommandSpec, split_args

__all__ = ["REGISTRY", "CommandContext", "CommandSpec", "split_args"]
