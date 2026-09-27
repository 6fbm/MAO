"""Exception hierarchy.

Provider errors are classified so the gateway can decide between retrying,
rotating API keys, compacting context or falling back to another model.
"""

from __future__ import annotations


class MaoError(Exception):
    """Base class for all orchestrator errors."""


class ConfigError(MaoError):
    """Invalid or missing configuration."""


# --------------------------------------------------------------------------- providers


class ProviderError(MaoError):
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        retry_after: float | None = None,
        body_excerpt: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        self.body_excerpt = body_excerpt

    def __str__(self) -> str:
        code = f" (HTTP {self.status_code})" if self.status_code else ""
        prefix = f"[{self.provider}] " if self.provider else ""
        return f"{prefix}{self.message}{code}"


class AuthenticationError(ProviderError):
    """401/403 – the key is invalid or lacks permission. Rotate key."""


class QuotaExceededError(ProviderError):
    """Billing quota exhausted for this key. Rotate key, do not retry."""


class RateLimitError(ProviderError):
    """429 – retry later or on another key."""

    retryable = True


class TransientProviderError(ProviderError):
    """5xx, overload, timeout or network problem – retry with backoff."""

    retryable = True


class ContextLengthError(ProviderError):
    """The prompt exceeds the model context window – compact and retry."""


class ModelNotFoundError(ProviderError):
    """The model id does not exist for this provider/key."""


class InvalidRequestError(ProviderError):
    """400 – the request was rejected (unsupported parameter etc.)."""


class ContentFilterError(ProviderError):
    """The provider blocked the prompt or response."""


class InvalidResponseError(ProviderError):
    """The provider returned something we could not parse."""

    retryable = True


class NoAvailableKeyError(ProviderError):
    """All keys are cooling down or invalid."""

    def __init__(self, message: str, *, provider: str = "", wait_seconds: float | None = None) -> None:
        super().__init__(message, provider=provider)
        self.wait_seconds = wait_seconds


class ProviderUnavailableError(MaoError):
    """All retries and fallback models failed."""

    def __init__(self, message: str, attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


# --------------------------------------------------------------------------- limits / control


class BudgetExceededError(MaoError):
    def __init__(self, metric: str, used: float, limit: float) -> None:
        super().__init__(f"Limit reached: {metric} ({used:,.2f} / {limit:,.2f})")
        self.metric = metric
        self.used = used
        self.limit = limit


class OperationCancelled(MaoError):
    """The user stopped the run."""


# --------------------------------------------------------------------------- security / tools


class PermissionDeniedError(MaoError):
    """An agent tried something its permissions or the policy do not allow."""


class SandboxViolationError(PermissionDeniedError):
    """A path resolved outside the workspace or is otherwise forbidden."""


class ApprovalDeniedError(PermissionDeniedError):
    """The user (or the approval policy) rejected an action."""


class ToolError(MaoError):
    """A tool failed in an expected way (message is shown to the agent)."""


# --------------------------------------------------------------------------- orchestration


class PlanValidationError(MaoError):
    pass


class SessionError(MaoError):
    pass
