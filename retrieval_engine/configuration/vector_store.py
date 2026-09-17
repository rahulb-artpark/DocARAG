from dataclasses import dataclass, field
import math
import os
import re
from urllib.parse import urlparse

from retrieval_engine.configuration.embeddings import EmbeddingConfig


@dataclass
class VDBConfig:
    mm_embedding: bool = True
    tenant_id: str = "default"
    corpus_id: str = "default"
    document_id: str | None = None
    document_version_id: str | None = None
    index_revision_id: str | None = None
    tenant_id_env: str = "DOCARAG_TENANT_ID"
    corpus_id_env: str = "DOCARAG_CORPUS_ID"
    document_id_env: str = "DOCARAG_DOCUMENT_ID"
    document_version_id_env: str = "DOCARAG_DOCUMENT_VERSION_ID"
    index_revision_id_env: str = "DOCARAG_INDEX_REVISION_ID"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_url_env: str = "DOCARAG_QDRANT_URL"
    collection_name: str = "default_collection"
    api_key_env: str = "QDRANT_API_KEY"
    timeout_seconds: float = 10.0
    manifest_dir_name: str = "./qdrant-attestation"
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    force_rebuild: bool = True
    index_batch_size: int = 256

    def __post_init__(self) -> None:
        environment_fields = {
            "tenant_id": self.tenant_id_env,
            "corpus_id": self.corpus_id_env,
            "document_id": self.document_id_env,
            "document_version_id": self.document_version_id_env,
            "index_revision_id": self.index_revision_id_env,
        }
        for field_name, env_name in environment_fields.items():
            if (
                not isinstance(env_name, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None
            ):
                raise ValueError(
                    f"vdb.{field_name}_env must be a valid environment variable name"
                )
            if env_name in os.environ:
                setattr(self, field_name, os.environ[env_name])

        for field_name in ("tenant_id", "corpus_id"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"vdb.{field_name} must be a non-empty trimmed string")
            if len(value) > 128:
                raise ValueError(f"vdb.{field_name} may not exceed 128 characters")
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None:
                raise ValueError(
                    f"vdb.{field_name} contains unsupported characters"
                )
        for field_name in (
            "document_id",
            "document_version_id",
            "index_revision_id",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or len(value) > 512
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None
            ):
                raise ValueError(
                    f"vdb.{field_name} must be a safe non-empty trimmed identifier"
                )
        if (
            not isinstance(self.qdrant_url_env, str)
            or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", self.qdrant_url_env
            )
            is None
        ):
            raise ValueError(
                "vdb.qdrant_url_env must be a valid environment variable name"
            )
        qdrant_url = os.environ.get(self.qdrant_url_env, self.qdrant_url)
        if not isinstance(qdrant_url, str) or not qdrant_url.strip():
            raise ValueError("vdb.qdrant_url must be a non-empty HTTP(S) URL")
        if qdrant_url != qdrant_url.strip():
            raise ValueError("vdb.qdrant_url may not have surrounding whitespace")
        parsed_url = urlparse(qdrant_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("vdb.qdrant_url must be an HTTP(S) URL with a host")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("vdb.qdrant_url may not contain credentials")
        self.qdrant_url = qdrant_url.rstrip("/")

        if not isinstance(self.collection_name, str) or not self.collection_name.strip():
            raise ValueError("vdb.collection_name must be a non-empty string")
        if self.collection_name != self.collection_name.strip():
            raise ValueError(
                "vdb.collection_name may not have surrounding whitespace"
            )
        if len(self.collection_name) > 255:
            raise ValueError("vdb.collection_name may not exceed 255 characters")

        if (
            not isinstance(self.api_key_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) is None
        ):
            raise ValueError(
                "vdb.api_key_env must be a valid environment variable name"
            )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("vdb.timeout_seconds must be a positive number")
        if (
            not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("vdb.timeout_seconds must be a positive finite number")
        self.timeout_seconds = float(self.timeout_seconds)

        if (
            not isinstance(self.manifest_dir_name, str)
            or not self.manifest_dir_name.strip()
        ):
            raise ValueError("vdb.manifest_dir_name must be a non-empty path")
        if self.manifest_dir_name != self.manifest_dir_name.strip():
            raise ValueError(
                "vdb.manifest_dir_name may not have surrounding whitespace"
            )
        if (
            isinstance(self.index_batch_size, bool)
            or not isinstance(self.index_batch_size, int)
            or self.index_batch_size < 1
        ):
            raise ValueError("vdb.index_batch_size must be a positive integer")
