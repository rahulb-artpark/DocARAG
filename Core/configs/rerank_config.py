from dataclasses import dataclass
import math
from typing import Optional

from Core.configs.llm_config import _normalize_api_base


@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    # Safe default for machines without NVIDIA. Deployment presets may override it.
    device: Optional[str] = None
    backend: str = "local"  # Options: 'local', 'vllm'
    api_base: Optional[str] = None
    request_timeout_seconds: Optional[float] = None
    use_flash_attention: Optional[bool] = None
    flash_attention_fallback: Optional[bool] = None

    def __post_init__(self):
        if not isinstance(self.backend, str):
            raise ValueError("backend must be 'local' or 'vllm'")
        self.backend = self.backend.strip().lower()
        if self.backend not in {"local", "vllm"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = self.model_name.strip()
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise ValueError("max_length must be at least 1")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")
        if self.device is not None:
            if not isinstance(self.device, str) or not self.device.strip():
                raise ValueError("device must be a non-empty string when provided")
            self.device = self.device.strip()
        if self.api_base is not None:
            if not isinstance(self.api_base, str):
                raise ValueError("api_base must be a URL when provided")
            self.api_base = self.api_base.strip() or None
            self.api_base = _normalize_api_base(self.api_base)
        if self.backend == "vllm" and self.api_base is None:
            raise ValueError("api_base is required for the vllm reranker backend")
        if self.backend == "local":
            if self.api_base is not None:
                raise ValueError("api_base is only supported by the vllm backend")
            if self.request_timeout_seconds is not None:
                raise ValueError(
                    "request_timeout_seconds is only supported by the vllm backend"
                )
            self.device = self.device or "cpu"
            if self.use_flash_attention is None and self.device != "auto":
                self.use_flash_attention = "cuda" in self.device
            if self.use_flash_attention is True:
                if self.device != "auto" and "cuda" not in self.device:
                    raise ValueError(
                        "use_flash_attention requires a CUDA device or device='auto'"
                    )
                self.flash_attention_fallback = (
                    True
                    if self.flash_attention_fallback is None
                    else self.flash_attention_fallback
                )
            elif self.flash_attention_fallback is not None:
                raise ValueError(
                    "flash_attention_fallback requires use_flash_attention=true"
                )
        else:
            if self.device is not None:
                raise ValueError("device is only supported by the local reranker backend")
            if self.use_flash_attention is not None:
                raise ValueError(
                    "use_flash_attention is only supported by the local reranker backend"
                )
            if self.flash_attention_fallback is not None:
                raise ValueError(
                    "flash_attention_fallback is only supported by the local reranker backend"
                )
            if self.request_timeout_seconds is None:
                self.request_timeout_seconds = 30.0
            if (
                isinstance(self.request_timeout_seconds, bool)
                or not isinstance(self.request_timeout_seconds, (int, float))
                or not math.isfinite(self.request_timeout_seconds)
                or self.request_timeout_seconds <= 0
            ):
                raise ValueError(
                    "request_timeout_seconds must be finite and greater than 0"
                )
        if self.use_flash_attention is not None and not isinstance(
            self.use_flash_attention, bool
        ):
            raise ValueError("use_flash_attention must be a boolean")
        if self.flash_attention_fallback is not None and not isinstance(
            self.flash_attention_fallback, bool
        ):
            raise ValueError("flash_attention_fallback must be a boolean")
