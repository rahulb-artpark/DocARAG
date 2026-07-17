from dataclasses import dataclass
from typing import Optional

from Core.configs.llm_config import _normalize_api_base


@dataclass
class EmbeddingConfig:
    type: str = "text"
    backend: str = "local"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    # Local providers enforce this input bound. For remote providers it is the
    # declared served-model input contract and is included in vector identity.
    max_length: int = 8192
    # Remote providers own placement; only local providers accept a device.
    device: Optional[str] = None
    # Ollama sizes an embedding model's KV cache from its declared context
    # unless a request caps it, which costs gigabytes for a small model.
    num_ctx: Optional[int] = None
    
    
    def __post_init__(self):
        if not isinstance(self.type, str):
            raise ValueError("type must be 'text' or 'gme'")
        self.type = self.type.strip().lower()
        if self.type not in {"text", "gme"}:
            raise ValueError(f"Unsupported embedding type: {self.type}")
        if not isinstance(self.backend, str):
            raise ValueError("backend must be 'local', 'ollama', or 'openai'")
        self.backend = self.backend.strip().lower()
        if self.backend not in ["local", "ollama", "openai"]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.type == "gme" and self.backend != "local":
            raise ValueError("gme embeddings require the local backend")
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
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key
        ):
            raise ValueError("api_key must be a string when provided")
        if self.api_base is not None:
            self.api_base = _normalize_api_base(self.api_base)
        if self.backend == "local" and (self.api_key or self.api_base):
            raise ValueError(
                "api_key and api_base are not used by the local embedding backend"
            )
        if self.backend == "local":
            self.device = self.device or "cpu"
        elif self.device is not None:
            raise ValueError("device is only supported by the local embedding backend")
