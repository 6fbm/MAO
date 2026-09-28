"""The /chat conversation state."""

from __future__ import annotations

from mao.cli.commands.chat import MAX_HISTORY_MESSAGES, ChatState
from mao.core.types import ChatMessage, MessageRole


def _state() -> ChatState:
    return ChatState(model_ref="x/y", gateway=None, tracker=None)  # type: ignore[arg-type]


def test_history_is_trimmed_to_whole_exchanges() -> None:
    """A trimmed history must still open with a user message.

    Anthropic and Gemini reject a conversation that starts with an assistant turn.
    The trim runs in the same order as send() does it - question appended, then
    trimmed, then the reply - because that odd length is what used to slice the
    first question away and leave its answer dangling at the front.
    """
    state = _state()
    for turn in range(30):
        state.history.append(ChatMessage.user(f"q{turn}"))
        state.trim()
        assert len(state.history) <= MAX_HISTORY_MESSAGES
        assert state.history[0].role is MessageRole.USER, f"broke in turn {turn}"
        state.history.append(ChatMessage.assistant(f"a{turn}"))


def test_short_history_is_left_alone() -> None:
    state = _state()
    state.history.append(ChatMessage.user("hello"))
    state.history.append(ChatMessage.assistant("hi"))
    state.trim()
    assert len(state.history) == 2


def test_trim_keeps_the_most_recent_exchanges() -> None:
    state = _state()
    for turn in range(40):
        state.history.append(ChatMessage.user(f"q{turn}"))
        state.trim()
        state.history.append(ChatMessage.assistant(f"a{turn}"))
    assert state.history[-1].content == "a39"
    assert state.history[0].role is MessageRole.USER
