from dataclasses import dataclass
import math
import re
from typing import Optional, Union
from urllib.parse import urlsplit


OllamaKeepAlive = Union[int, float, str]


def _validate_keep_alive(value: OllamaKeepAlive) -> None:
    """Validate the bounded subset of Ollama's keep-alive syntax we expose."""

    if isinstance(value, bool):
        raise ValueError("keep_alive must be a duration or a number of seconds")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or (value < 0 and value != -1):
            raise ValueError("keep_alive must be -1 or a finite non-negative value")
        return
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:-1|0|(?:\d+(?:\.\d+)?|\.\d+)(?:ns|us|µs|ms|s|m|h))",
        value,
    ):
        raise ValueError(
            "keep_alive must be -1, 0, seconds, or a duration such as '5m' or '1h'"
        )


def _normalize_api_base(value: Optional[str]) -> Optional[str]:
    """Return a safe HTTP(S) API base URL or reject it before SDK startup."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("api_base must be a non-empty HTTP(S) URL when provided")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "api_base must be an HTTP(S) base URL without credentials, query, or fragment"
        )
    return normalized


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen3-8B-AWQ"
    api_key: Optional[str] = "openai"
    api_base: str = "http://localhost:8003/v1"
    temperature: float = 0.1
    max_tokens: int = 5000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    backend: str = "openai"
    max_workers: int = 8
    max_in_flight: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    # Finite SDK network timeout for each request. Application-level retries
    # multiply the number of timed attempts, so size it for the slowest
    # expected generation rather than accepting an unbounded transport.
    request_timeout_seconds: float = 300.0
    # Ollama model residency. The server default is finite; keeping a model
    # forever must now be an explicit ``-1`` choice.
    keep_alive: OllamaKeepAlive = "5m"
    # Closing a client normally releases only its HTTP resources. Model
    # eviction affects every user of a shared Ollama server and is opt-in.
    unload_on_close: bool = False

    def __post_init__(self):
        if not isinstance(self.backend, str):
            raise ValueError("backend must be 'openai' or 'ollama'")
        self.backend = self.backend.strip().lower()
        if self.backend not in ["openai", "ollama"]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = self.model_name.strip()
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key
        ):
            raise ValueError("api_key must be a string when provided")
        self.api_base = _normalize_api_base(self.api_base)
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be a finite number between 0 and 2")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be a finite number between 0 and 2")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be at least 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        for field_name in ("frequency_penalty", "presence_penalty"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -2 <= value <= 2
            ):
                raise ValueError(
                    f"{field_name} must be a finite number between -2 and 2"
                )
        if self.backend == "ollama" and (
            self.frequency_penalty != 0 or self.presence_penalty != 0
        ):
            raise ValueError(
                "frequency_penalty and presence_penalty are not supported by "
                "the Ollama backend; leave both at 0"
            )
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be at least 1")
        if self.max_in_flight is None:
            self.max_in_flight = self.max_workers
        if (
            isinstance(self.max_in_flight, bool)
            or not isinstance(self.max_in_flight, int)
            or self.max_in_flight < 1
        ):
            raise ValueError("max_in_flight must be at least 1")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 1
            or self.max_retries > 10
        ):
            raise ValueError("max_retries must be between 1 and 10")
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be non-negative")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be a finite value above 0")
        self.request_timeout_seconds = float(self.request_timeout_seconds)
        _validate_keep_alive(self.keep_alive)
        if not isinstance(self.unload_on_close, bool):
            raise ValueError("unload_on_close must be a boolean")
        if self.backend != "ollama" and (
            self.keep_alive != "5m" or self.unload_on_close
        ):
            raise ValueError(
                "keep_alive and unload_on_close are only configurable for Ollama"
            )
