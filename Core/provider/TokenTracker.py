"""Request-scoped token accounting with a backwards-compatible facade.

Historically :class:`TokenTracker` was a process-wide singleton whose counters
were reset at the beginning of a run.  A lock prevented torn writes, but two
concurrent indexing or query runs still reset and charged the same counters.

New code should create an explicit scope::

    tracker = TokenTracker.get_instance()
    with tracker.scope(name="request-123") as usage:
        ...  # existing providers continue to call TokenTracker.get_instance()
    persist(usage.get_usage(), usage.stage_history)

The active scope is carried by :mod:`contextvars`.  ``TokenUsageScope`` also
implements the accounting methods directly, so a scope can be injected into
new components without depending on the singleton facade.  Calls made without
an active scope deliberately use one locked legacy process-wide scope.  That
fallback preserves old scripts, but concurrent services must use an explicit
scope (and propagate the context when starting their own threads).
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from operator import index
import threading
from typing import Any, Dict


_ACTIVE_TOKEN_SCOPE: ContextVar["TokenUsageScope | None"] = ContextVar(
    "docarag_active_token_scope",
    default=None,
)


def current_token_scope() -> "TokenUsageScope | None":
    """Return only an explicitly active scope, never the legacy fallback."""

    return _ACTIVE_TOKEN_SCOPE.get()


def _token_count(value: Any, *, label: str) -> int:
    """Return a non-negative integer token count without accepting floats."""

    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative")
    return normalized


class TokenUsageScope:
    """A reusable, thread-safe token ledger for one logical request or run.

    A scope's counters remain readable after it is deactivated.  The same
    scope object may be shared by worker contexts; its internal lock protects
    concurrent provider callbacks.  Activations are context-local and may be
    nested safely.
    """

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name
        self._lock = threading.RLock()
        # Context-local activation stacks make ``with scope`` re-entrant while
        # avoiding a shared token attribute when the ledger is injected into
        # more than one execution context.
        self._activation_tokens: ContextVar[tuple[Token, ...]] = ContextVar(
            f"docarag_token_scope_activation_{id(self)}",
            default=(),
        )
        self.reset()

    def __enter__(self) -> "TokenUsageScope":
        token = _ACTIVE_TOKEN_SCOPE.set(self)
        stack = self._activation_tokens.get()
        self._activation_tokens.set((*stack, token))
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        stack = self._activation_tokens.get()
        if not stack:
            raise RuntimeError("TokenUsageScope exited without a matching entry")
        token = stack[-1]
        self._activation_tokens.set(stack[:-1])
        _ACTIVE_TOKEN_SCOPE.reset(token)

    def activate(self) -> "TokenUsageScope":
        """Return this scope as a context manager for dependency injection."""

        return self

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        prompt = _token_count(prompt_tokens, label="prompt_tokens")
        completion = _token_count(
            completion_tokens,
            label="completion_tokens",
        )
        with self._lock:
            self._prompt_tokens += prompt
            self._completion_tokens += completion
            self._total_tokens += prompt + completion

    def get_usage(self) -> dict[str, int]:
        """Return a consistent copy of the cumulative token counters."""

        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
            }

    @property
    def prompt_tokens(self) -> int:
        return self.get_usage()["prompt_tokens"]

    @property
    def completion_tokens(self) -> int:
        return self.get_usage()["completion_tokens"]

    @property
    def total_tokens(self) -> int:
        return self.get_usage()["total_tokens"]

    @property
    def stage_history(self) -> Dict[str, Dict[str, int]]:
        """Return a detached copy so persisted stats cannot mutate the ledger."""

        with self._lock:
            return {
                stage: dict(usage)
                for stage, usage in self._stage_history.items()
            }

    def snapshot(self) -> dict[str, object]:
        """Atomically copy cumulative counters and all recorded stage deltas."""

        with self._lock:
            return {
                "totals": {
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": self._total_tokens,
                },
                "stages": {
                    stage: dict(usage)
                    for stage, usage in self._stage_history.items()
                },
            }

    def reset(self) -> None:
        """Clear this scope's counters and stage baseline."""

        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0
            self._stage_history: Dict[str, Dict[str, int]] = {}
            self._last_stage_prompt_tokens = 0
            self._last_stage_completion_tokens = 0

    def begin_stage_sequence(self) -> None:
        """Set a new stage baseline without discarding cumulative usage.

        Application boundaries call this when reusing an outer request scope.
        It keeps the outer cumulative total intact while ensuring the first
        stage delta belongs only to the nested operation.
        """

        with self._lock:
            self._last_stage_prompt_tokens = self._prompt_tokens
            self._last_stage_completion_tokens = self._completion_tokens

    def record_stage(self, stage_name: str) -> Dict[str, int]:
        """Record usage since the preceding stage within *this* scope."""

        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string")
        with self._lock:
            prompt_tokens = (
                self._prompt_tokens - self._last_stage_prompt_tokens
            )
            completion_tokens = (
                self._completion_tokens - self._last_stage_completion_tokens
            )
            stage_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            self._stage_history[stage_name] = stage_usage
            self._last_stage_prompt_tokens = self._prompt_tokens
            self._last_stage_completion_tokens = self._completion_tokens
            return dict(stage_usage)

    def print_all_stages(self) -> None:
        """Print a stable snapshot of stage and cumulative usage."""

        with self._lock:
            history = {
                stage: dict(usage)
                for stage, usage in self._stage_history.items()
            }
            usage = {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
            }

        print("\n" + "=" * 50)
        print("📊 TOKEN USAGE REPORT 📊")
        print("=" * 50)
        if not history:
            print("No stages have been recorded yet.")
        else:
            print("\n--- Stage-by-Stage Breakdown ---")
            for stage, stage_usage in history.items():
                print(
                    f"  - Stage '{stage}':\n"
                    f"    Prompt: {stage_usage['prompt_tokens']:>6} | "
                    f"Completion: {stage_usage['completion_tokens']:>6} | "
                    f"Total: {stage_usage['total_tokens']:>7}"
                )
        print("\n--- Cumulative Total ---")
        print(
            "  Overall Usage | "
            f"Prompt: {usage['prompt_tokens']} | "
            f"Completion: {usage['completion_tokens']} | "
            f"Total: {usage['total_tokens']}"
        )
        print("=" * 50 + "\n")

    def __str__(self) -> str:
        usage = self.get_usage()
        return (
            "📊 Token Usage | "
            f"Prompt: {usage['prompt_tokens']} | "
            f"Completion: {usage['completion_tokens']} | "
            f"Total: {usage['total_tokens']}"
        )


class TokenTracker:
    """Singleton facade routing calls to the active token-usage scope.

    ``get_instance()`` and all historical accounting methods remain available.
    With no explicit scope, they target a process-wide legacy ledger.  Use
    :meth:`scope` for every concurrent request or indexing run.
    """

    _instance: "TokenTracker | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "TokenTracker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._legacy_scope = TokenUsageScope(name="legacy-global")
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TokenTracker":
        return cls()

    def scope(self, *, name: str | None = None) -> TokenUsageScope:
        """Create a fresh isolated ledger that can be used with ``with``."""

        return TokenUsageScope(name=name)

    def use_scope(self, scope: TokenUsageScope) -> TokenUsageScope:
        """Activate an injected ledger while preserving facade-based callers."""

        if not isinstance(scope, TokenUsageScope):
            raise TypeError("scope must be a TokenUsageScope")
        return scope.activate()

    def active_scope(self) -> TokenUsageScope:
        """Return the context scope or the documented legacy fallback."""

        return _ACTIVE_TOKEN_SCOPE.get() or self._legacy_scope

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.active_scope().add_usage(prompt_tokens, completion_tokens)

    def get_usage(self) -> dict[str, int]:
        return self.active_scope().get_usage()

    # Read-only compatibility attributes for callers that inspected the old
    # singleton's public counters.  Mutation must go through ``add_usage`` so
    # it cannot bypass scope selection or locking.
    @property
    def prompt_tokens(self) -> int:
        return self.active_scope().prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.active_scope().completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.active_scope().total_tokens

    @property
    def stage_history(self) -> Dict[str, Dict[str, int]]:
        return self.active_scope().stage_history

    def snapshot(self) -> dict[str, object]:
        return self.active_scope().snapshot()

    def reset(self) -> None:
        self.active_scope().reset()

    def begin_stage_sequence(self) -> None:
        self.active_scope().begin_stage_sequence()

    def record_stage(self, stage_name: str) -> Dict[str, int]:
        return self.active_scope().record_stage(stage_name)

    def print_all_stages(self) -> None:
        self.active_scope().print_all_stages()

    def __str__(self) -> str:
        return str(self.active_scope())
