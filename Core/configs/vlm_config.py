from dataclasses import dataclass
import math
from typing import Optional

from Core.configs.llm_config import (
    OllamaKeepAlive,
    _normalize_api_base,
    _validate_keep_alive,
)


@dataclass
class VLMConfig:
    backend: str = "ollama"  # "qwen", "gpt", "ollama"
    model_name: str = "qwen2.5vl:6k"
    max_tokens: int = 6000
    temperature: float = 0.7
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_workers: int = 8
    max_in_flight: Optional[int] = None
    max_retries: int = 3
    retry_delay_seconds: float = 0.0
    # ``None`` is an explicitness sentinel: remote backends normalize it to a
    # finite default, while the in-process Qwen backend has no network request
    # to time out and rejects a user-supplied remote-only value.
    request_timeout_seconds: Optional[float] = None
    keep_alive: OllamaKeepAlive = "5m"
    unload_on_close: bool = False
    num_ctx: Optional[int] = None

    def __post_init__(self):
        if self.num_ctx is not None and (
            isinstance(self.num_ctx, bool)
            or not isinstance(self.num_ctx, int)
            or self.num_ctx < 1
        ):
            raise ValueError("num_ctx must be a positive integer when provided")
        if not isinstance(self.backend, str):
            raise ValueError("backend must be 'qwen', 'gpt', or 'ollama'")
        self.backend = self.backend.strip().lower()
        if self.backend not in {"qwen", "gpt", "ollama"}:
            raise ValueError(f"Unsupported VLM backend: {self.backend}")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = self.model_name.strip()
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key
        ):
            raise ValueError("api_key must be a string when provided")
        self.api_base = _normalize_api_base(self.api_base)
        if self.backend == "qwen" and (self.api_key or self.api_base):
            raise ValueError(
                "api_key and api_base are not supported by the in-process qwen backend"
            )
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
        if self.backend == "qwen":
            if self.request_timeout_seconds is not None:
                raise ValueError(
                    "request_timeout_seconds is only supported by remote GPT "
                    "and Ollama VLM backends, not in-process qwen"
                )
        else:
            if self.request_timeout_seconds is None:
                self.request_timeout_seconds = 300.0
            if (
                isinstance(self.request_timeout_seconds, bool)
                or not isinstance(self.request_timeout_seconds, (int, float))
                or not math.isfinite(self.request_timeout_seconds)
                or self.request_timeout_seconds <= 0
            ):
                raise ValueError(
                    "request_timeout_seconds must be a finite value above 0"
                )
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
