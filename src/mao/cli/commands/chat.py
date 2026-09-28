"""Direct conversation with a single model, without the agent machinery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.text import Text

from mao.cli.commands.base import CommandContext, command
from mao.core.errors import MaoError
from mao.core.text import fmt_cost, fmt_int
from mao.core.types import ChatMessage, CompletionRequest, MessageRole
from mao.providers.gateway import LLMGateway
from mao.tokens.tracker import UsageTracker

if TYPE_CHECKING:
    from mao.app import AppContext

MAX_HISTORY_MESSAGES = 40


@dataclass
class ChatState:
    """One running conversation. Lives on the REPL, not in a session."""

    model_ref: str
    gateway: LLMGateway
    tracker: UsageTracker
    history: list[ChatMessage] = field(default_factory=list)
    turns: int = 0

    def trim(self) -> None:
        """Drop the oldest exchanges, never leaving a reply without its question.

        Anthropic and Gemini reject a history that does not start with a user
        message, so the cut moves forward to the next one instead of slicing
        blindly.
        """
        if len(self.history) <= MAX_HISTORY_MESSAGES:
            return
        cut = len(self.history) - MAX_HISTORY_MESSAGES
        while cut < len(self.history) and self.history[cut].role is not MessageRole.USER:
            cut += 1
        self.history = self.history[cut:]


def _first_available(app: AppContext) -> str:
    for entry in app.hub.catalog.entries(include_disabled=False):
        # The mock provider answers with canned text; only offer it in demo mode.
        if entry.provider == "mock" and not app.demo:
            continue
        if app.hub.catalog.is_available(entry):
            return entry.ref
    raise MaoError(
        "No model is available. Add an API key (/providers key add <provider>), "
        "start a local model server, or check /models."
    )


def _build_state(app: AppContext, model_ref: str) -> ChatState:
    entry = app.hub.catalog.resolve(model_ref)
    if not app.hub.catalog.is_available(entry):
        raise MaoError(f"{entry.ref} is not available: {app.hub.catalog.unavailable_reason(entry)}")
    tracker = UsageTracker()
    gateway = LLMGateway(
        providers=app.hub.providers,
        catalog=app.hub.catalog,
        keypools=app.hub.keypools,
        limiters=app.hub.limiters,
        tracker=tracker,
        bus=app.bus,
    )
    return ChatState(model_ref=entry.ref, gateway=gateway, tracker=tracker)


@command(
    "chat",
    summary="Talk to one model directly, without planning or agents",
    usage="/chat [<model>] | /chat off | /chat reset | /chat model <ref>",
    group="Tasks",
    subcommands=("off", "reset", "model"),
)
async def chat_cmd(ctx: CommandContext, args: list[str], raw: str) -> None:
    repl = ctx.repl
    sub = args[0].lower() if args else ""

    if sub == "off":
        if repl.chat is None:
            ctx.ui.info("Not in a chat.")
            return
        turns, cost = repl.chat.turns, repl.chat.tracker.totals().cost_usd
        repl.chat = None
        ctx.ui.success(f"Chat closed after {turns} turn(s), {fmt_cost(cost)}.")
        return

    if sub == "reset":
        if repl.chat is None:
            raise MaoError("Not in a chat. Start one with /chat [<model>].")
        repl.chat.history.clear()
        repl.chat.turns = 0
        ctx.ui.success("Conversation cleared - the model starts fresh.")
        return

    if sub == "model":
        if len(args) < 2:
            raise MaoError("Usage: /chat model <provider/model>")
        repl.chat = _build_state(ctx.app, args[1])
        ctx.ui.success(f"Chatting with {repl.chat.model_ref}. The conversation starts over.")
        return

    model_ref = args[0] if args else _first_available(ctx.app)
    repl.chat = _build_state(ctx.app, model_ref)
    ctx.ui.success(f"Chatting with {repl.chat.model_ref}.")
    ctx.ui.print(
        Text(
            "Everything you type now goes to the model. Commands still work: "
            "/chat off leaves, /chat reset forgets the conversation.",
            style="dim",
        )
    )


async def send(ctx: CommandContext, text: str) -> None:
    """Send one line to the model of the running chat and print the reply."""
    state = ctx.repl.chat
    if state is None:  # pragma: no cover - dispatch only calls this while a chat runs
        raise MaoError("Not in a chat.")
    app = ctx.app
    state.history.append(ChatMessage.user(text))
    state.trim()
    request = CompletionRequest(
        model="",
        system=f"You are a helpful assistant. Answer in {app.config.settings.response_language}.",
        messages=list(state.history),
        metadata={"agent": "chat", "purpose": "chat"},
    )
    try:
        with ctx.ui.console.status(f"{state.model_ref} …"):
            result = await state.gateway.complete(request, model_ref=state.model_ref, agent="chat", purpose="chat")
    except MaoError:
        # A failed turn must not poison the conversation.
        state.history.pop()
        raise
    reply = result.response.message.content or "(empty reply)"
    state.history.append(ChatMessage.assistant(reply))
    state.turns += 1
    usage = result.response.usage
    ctx.ui.print(Text(reply))
    ctx.ui.print(
        Text(
            f"  {result.entry.ref} · {fmt_int(usage.total_tokens)} tokens · "
            f"{fmt_cost(result.cost_usd)} · {fmt_cost(state.tracker.totals().cost_usd)} this chat",
            style="dim",
        )
    )
