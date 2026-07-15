from Core.provider.embedding import BaseEmbedder

from collections.abc import Iterable
from enum import Enum
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
from typing import List, Dict, Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid
import logging

log = logging.getLogger(__name__)


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_URL_ENV = "DOCARAG_QDRANT_URL"
DEFAULT_QDRANT_API_KEY_ENV = "QDRANT_API_KEY"
VECTOR_RECORD_ID_SCHEMA = "docarag.qdrant-record-id.v4"
VECTOR_PAYLOAD_SCHEMA = "docarag.qdrant-payload.v2"
_QDRANT_POINT_NAMESPACE = uuid.UUID("65dc7f26-27fe-5b93-8df8-76c53255e978")
_PAYLOAD_RECORD_ID = "_docarag_record_id"
_PAYLOAD_CONTENT = "_docarag_content"
_PAYLOAD_SCHEMA = "_docarag_schema"
_RESERVED_PAYLOAD_KEYS = frozenset(
    {_PAYLOAD_RECORD_ID, _PAYLOAD_CONTENT, _PAYLOAD_SCHEMA}
)
_KEYWORD_PAYLOAD_INDEXES = (
    "tenant_id",
    "corpus_id",
    "document_version_id",
    "index_revision_id",
    "source_sha256",
    "index_kind",
    "modality",
    "node_type",
    "canonical_revision_id",
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_IDENTITY_FIELDS = (
    "tenant_id",
    "corpus_id",
    "index_kind",
    "index_revision_id",
    "node_id",
    "derived_evidence_id",
    "canonical_revision_id",
    "canonical_block_id",
    "observation_id",
    "pdf_id",
)


class MissingVectorSourceIdentityError(ValueError):
    """A strict retrieval record cannot be bound to immutable source evidence."""


def _identity_scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return _identity_scalar(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("vector-store metadata cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Generic legacy callers historically supplied loosely typed metadata.  It
    # remains usable, but only in the explicitly non-source-bound ID namespace.
    return str(value)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representation_identity(
    value: str,
    *,
    modality: str,
    image_text: str | None,
    strict_source_identity: bool,
) -> Mapping[str, str]:
    if not isinstance(value, str):
        raise TypeError("vector-store representation must be a string")
    if modality == "text":
        return {"text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}

    image_path = Path(value)
    if image_path.is_file():
        image_sha256 = _file_sha256(image_path)
    elif strict_source_identity:
        raise MissingVectorSourceIdentityError(
            f"strict image representation is missing or unreadable: {value}"
        )
    else:
        # This is intentionally named as a path fallback: it must never be
        # confused with a digest of immutable image bytes.
        image_sha256 = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return {
            "legacy_image_path_sha256": image_sha256,
            "fused_text_sha256": hashlib.sha256(
                (image_text or "").encode("utf-8")
            ).hexdigest(),
        }
    return {
        "image_sha256": image_sha256,
        "fused_text_sha256": hashlib.sha256(
            (image_text or "").encode("utf-8")
        ).hexdigest(),
    }


def _normalized_metadata(metadata: Mapping[str, Any], *, modality: str) -> dict:
    if not isinstance(metadata, Mapping):
        raise TypeError("vector-store metadata must be a mapping")
    result = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("vector-store metadata keys must be non-empty strings")
        normalized = _identity_scalar(value)
        # Optional trace fields are omitted rather than serialized as an
        # ambiguous string sentinel.
        if normalized is not None:
            result[key] = normalized
    reserved = _RESERVED_PAYLOAD_KEYS.intersection(result)
    if reserved:
        raise ValueError(
            "vector-store metadata uses reserved payload keys: "
            + ", ".join(sorted(reserved))
        )
    existing_modality = result.get("modality")
    if existing_modality not in (None, modality):
        raise ValueError(
            f"metadata modality {existing_modality!r} conflicts with {modality!r} write"
        )
    existing_type = result.get("type")
    if existing_type in {"text", "image"} and existing_type != modality:
        raise ValueError(
            f"metadata type {existing_type!r} conflicts with {modality!r} write"
        )
    # Older callers sometimes used ``type`` for the semantic node type. Preserve
    # that information before maintaining the legacy MM-RAG modality field.
    if "node_type" not in result and existing_type not in (None, "text", "image"):
        result["node_type"] = existing_type
    result["modality"] = modality
    result["type"] = modality
    return result


def stable_vector_record_id(
    value: str,
    metadata: Mapping[str, Any],
    *,
    modality: str,
    image_text: str | None = None,
    strict_source_identity: bool = False,
) -> str:
    """Return a deterministic representation ID with an explicit legacy scope.

    Strict IDs bind the immutable document version, tree/evidence identity,
    modality, node type, and representation bytes/content. Generic callers that
    predate source provenance still receive deterministic IDs, but under a
    ``legacy_`` prefix so downstream code cannot mistake them for source truth.
    """

    if modality not in {"text", "image"}:
        raise ValueError("modality must be 'text' or 'image'")
    normalized = _normalized_metadata(metadata, modality=modality)
    representation = _representation_identity(
        value,
        modality=modality,
        image_text=image_text,
        strict_source_identity=strict_source_identity,
    )
    return _stable_vector_record_id_from_normalized(
        normalized,
        representation,
        modality=modality,
        strict_source_identity=strict_source_identity,
    )


def _stable_vector_record_id_from_normalized(
    normalized: Mapping[str, Any],
    representation: Mapping[str, str],
    *,
    modality: str,
    strict_source_identity: bool,
) -> str:
    """Build an ID from already-validated metadata and representation hashes."""

    document_version_id = normalized.get("document_version_id")
    source_sha256 = normalized.get("source_sha256")
    source_identity = {
        key: normalized[key]
        for key in _SOURCE_IDENTITY_FIELDS
        if normalized.get(key) not in (None, "")
    }
    if strict_source_identity:
        if not isinstance(document_version_id, str) or not document_version_id.strip():
            raise MissingVectorSourceIdentityError(
                "strict vector record requires document_version_id"
            )
        if document_version_id != document_version_id.strip():
            raise MissingVectorSourceIdentityError(
                "strict vector record document_version_id may not have surrounding "
                "whitespace"
            )
        if not isinstance(source_sha256, str) or _LOWER_SHA256.fullmatch(
            source_sha256
        ) is None:
            raise MissingVectorSourceIdentityError(
                "strict vector record requires lowercase source_sha256"
            )
        if not source_identity:
            raise MissingVectorSourceIdentityError(
                "strict vector record requires a node or evidence identity"
            )
        for field_name, field_value in source_identity.items():
            if isinstance(field_value, str) and field_value != field_value.strip():
                raise MissingVectorSourceIdentityError(
                    f"strict vector record {field_name} may not have surrounding "
                    "whitespace"
                )
        identity = {
            "schema": VECTOR_RECORD_ID_SCHEMA,
            "scope": "source-bound",
            "document_version_id": document_version_id,
            "source_sha256": source_sha256,
            "source_identity": source_identity,
            "node_type": normalized.get("node_type"),
            "modality": modality,
            "representation": representation,
        }
        return f"record_{modality}_{_canonical_sha256(identity)}"

    legacy_identity = {
        "schema": VECTOR_RECORD_ID_SCHEMA,
        "scope": "legacy-content",
        "metadata": normalized,
        "modality": modality,
        "representation": representation,
    }
    return f"legacy_{modality}_{_canonical_sha256(legacy_identity)}"


def _metadata_filter(
    metadata_filter: Mapping[str, Any] | None,
    document_version_id: str | None,
) -> Mapping[str, Any] | None:
    """Return one immutable, exact-match payload filter.

    The public retrieval contract intentionally accepts a flat mapping.  This
    keeps tenant/document isolation auditable and prevents backend-specific
    filter expressions from leaking through the BookRAG boundary.
    """

    if metadata_filter is not None and not isinstance(metadata_filter, Mapping):
        raise TypeError("metadata_filter must be a mapping")
    combined: dict[str, Any] = {}
    if metadata_filter:
        for key, value in metadata_filter.items():
            if not isinstance(key, str) or not key:
                raise ValueError("metadata_filter keys must be non-empty strings")
            if key in _RESERVED_PAYLOAD_KEYS:
                raise ValueError(f"metadata_filter may not use reserved key {key!r}")
            if not isinstance(value, (Enum, str, int, float, bool)):
                raise TypeError(
                    "metadata_filter values must be exact-match scalar values"
                )
            normalized = _identity_scalar(value)
            if normalized is None:
                raise ValueError("metadata_filter values may not be null")
            combined[key] = normalized
    if document_version_id is not None:
        if not isinstance(document_version_id, str) or not document_version_id.strip():
            raise ValueError("document_version_id must be a non-empty string")
        if document_version_id != document_version_id.strip():
            raise ValueError(
                "document_version_id may not have surrounding whitespace"
            )
        supplied = combined.get("document_version_id")
        if supplied is not None and supplied != document_version_id:
            raise ValueError(
                "metadata_filter document_version_id conflicts with the explicit "
                "document_version_id"
            )
        combined["document_version_id"] = document_version_id
    if not combined:
        return None
    return combined


def qdrant_point_id(record_id: str) -> str:
    """Map a stable logical record ID to a Qdrant-compatible UUID."""

    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, record_id))


def _validate_qdrant_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("qdrant_url must be a non-empty HTTP(S) URL")
    if url != url.strip():
        raise ValueError("qdrant_url may not have surrounding whitespace")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("qdrant_url must be an HTTP(S) URL with a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("qdrant_url may not contain credentials")
    return url.rstrip("/")


def _load_qdrant_components():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - exercised in packaging/doctor
        raise RuntimeError(
            "Qdrant support requires the 'qdrant-client' package"
        ) from exc
    return QdrantClient, models


def _normalize_vectors(raw_vectors: Any, expected_count: int) -> list[list[float]]:
    try:
        vectors = list(raw_vectors)
    except TypeError as exc:
        raise ValueError("embedding provider returned a non-iterable result") from exc
    if len(vectors) != expected_count:
        raise ValueError(
            "embedding provider returned "
            f"{len(vectors)} embeddings for {expected_count} inputs"
        )
    normalized: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if isinstance(vector, (str, bytes, bytearray)):
            raise ValueError("embedding vectors must be numeric sequences")
        try:
            values = list(vector)
        except TypeError as exc:
            raise ValueError("embedding vectors must be iterable") from exc
        if not values:
            raise ValueError("embedding vectors may not be empty")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise ValueError("embedding vectors must have one consistent dimension")
        output: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError("embedding vectors must contain only real numbers")
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError("embedding vectors may not contain NaN or infinity")
            output.append(scalar)
        normalized.append(output)
    return normalized


def _normalized_payload_patch(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("vector-store metadata must be a mapping")
    patch: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("vector-store metadata keys must be non-empty strings")
        if key in _RESERVED_PAYLOAD_KEYS:
            raise ValueError(f"vector-store metadata key {key!r} is reserved")
        normalized = _identity_scalar(value)
        if normalized is not None:
            patch[key] = normalized
    return patch


def _unique_record_indexes(
    ids: Sequence[str],
    values: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    auxiliary_texts: Sequence[str | None],
    *,
    compare_values: bool = True,
) -> tuple[int, ...]:
    """Coalesce exact duplicate records and reject stable-ID metadata collisions."""

    first_by_id: dict[str, int] = {}
    unique: list[int] = []
    for index, record_id in enumerate(ids):
        previous = first_by_id.get(record_id)
        if previous is None:
            first_by_id[record_id] = index
            unique.append(index)
            continue
        if (
            (compare_values and values[previous] != values[index])
            or dict(metadatas[previous]) != dict(metadatas[index])
            or auxiliary_texts[previous] != auxiliary_texts[index]
        ):
            raise ValueError(
                f"stable vector record ID collision for {record_id}; source identity "
                "or metadata is inconsistent"
            )
    return tuple(unique)


class _CollectionCountView:
    """Temporary count-only bridge for legacy KG callers.

    New code must call :meth:`VectorStore.count` directly.  Exposing no write
    methods prevents callers from bypassing scope enforcement.
    """

    def __init__(self, store: "VectorStore") -> None:
        self._store = store

    def count(self) -> int:
        return self._store.count()


class VectorStore:
    """Source-scoped BookRAG vector storage backed exclusively by Qdrant."""

    def __init__(
        self,
        embedding_model: BaseEmbedder,
        qdrant_url: str | None = None,
        collection_name: str = "multimodal_collection",
        *,
        api_key: str | None = None,
        api_key_env: str | None = DEFAULT_QDRANT_API_KEY_ENV,
        timeout_seconds: float = 10.0,
        scope_filter: Mapping[str, Any] | None = None,
        allow_collection_reset: bool = False,
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        if embedding_model is None:
            raise TypeError("embedding_model is required")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        if collection_name != collection_name.strip():
            raise ValueError("collection_name may not have surrounding whitespace")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, numbers.Real
        ):
            raise TypeError("timeout_seconds must be a positive number")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if not isinstance(allow_collection_reset, bool):
            raise TypeError("allow_collection_reset must be a boolean")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when supplied")
        if api_key_env is not None and (
            not isinstance(api_key_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env) is None
        ):
            raise ValueError("api_key_env must be a valid environment variable name")

        effective_qdrant_url = (
            os.environ.get(DEFAULT_QDRANT_URL_ENV, DEFAULT_QDRANT_URL)
            if qdrant_url is None
            else qdrant_url
        )
        qdrant_url = _validate_qdrant_url(effective_qdrant_url)
        normalized_scope = _metadata_filter(scope_filter, None)
        if normalized_scope is None:
            normalized_scope = {}
        if normalized_scope:
            missing_scope_fields = {"tenant_id", "corpus_id"}.difference(
                normalized_scope
            )
            if missing_scope_fields:
                raise ValueError(
                    "scope_filter must include tenant_id and corpus_id; missing "
                    + ", ".join(sorted(missing_scope_fields))
                )

        if models is None or client is None:
            qdrant_client_type, imported_models = _load_qdrant_components()
            if models is None:
                models = imported_models
        else:
            qdrant_client_type = None
        if client is None:
            resolved_api_key = api_key
            if resolved_api_key is None and api_key_env is not None:
                resolved_api_key = os.getenv(api_key_env) or None
            client = qdrant_client_type(
                url=qdrant_url,
                api_key=resolved_api_key,
                timeout=float(timeout_seconds),
            )
            owns_client = True
        else:
            owns_client = False

        self.embedding_model = embedding_model
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = client
        self._models = models
        self._owns_client = owns_client
        self._closed = False
        self._known_vector_size: int | None = None
        self._payload_indexes_ready = False
        self.scope_filter = dict(normalized_scope)
        self.allow_collection_reset = allow_collection_reset
        self.collection = _CollectionCountView(self)
        self._read_lease = None
        log.info("Qdrant collection '%s' configured at %s", collection_name, qdrant_url)

    def _require_open(self) -> None:
        if self._closed or self.client is None:
            raise RuntimeError("vector store is closed")

    @staticmethod
    def _is_conflict(error: Exception) -> bool:
        status = getattr(error, "status_code", getattr(error, "status", None))
        return status == 409 or "already exists" in str(error).lower()

    def _collection_exists(self) -> bool:
        self._require_open()
        exists = self.client.collection_exists(collection_name=self.collection_name)
        if not isinstance(exists, bool):
            raise RuntimeError("Qdrant returned an invalid collection-exists response")
        return exists

    @staticmethod
    def _vector_params(collection_info: Any) -> tuple[int, str]:
        try:
            vectors = collection_info.config.params.vectors
        except AttributeError as exc:
            raise RuntimeError("Qdrant returned an invalid collection configuration") from exc
        if isinstance(vectors, Mapping):
            raise RuntimeError(
                "DocARAG requires one unnamed dense vector; named-vector collection "
                "configuration is incompatible"
            )
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise RuntimeError("Qdrant collection has an invalid vector dimension")
        distance_value = getattr(distance, "value", distance)
        if not isinstance(distance_value, str) or distance_value.lower() != "cosine":
            raise RuntimeError("Qdrant collection must use cosine distance")
        return size, distance_value

    def _ensure_payload_indexes(self, collection_info: Any | None = None) -> None:
        if self._payload_indexes_ready:
            return
        if collection_info is None:
            collection_info = self.client.get_collection(self.collection_name)
        existing = getattr(collection_info, "payload_schema", {}) or {}
        existing_names = set(existing) if isinstance(existing, Mapping) else set()
        fields = set(_KEYWORD_PAYLOAD_INDEXES)
        fields.update(
            key
            for key in self.scope_filter
            if key
            in {
                "document_version_id",
                "index_revision_id",
                "source_sha256",
                "index_kind",
            }
        )
        for field_name in sorted(fields - existing_names):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=self._models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                if not self._is_conflict(exc):
                    raise
        self._payload_indexes_ready = True

    def _validate_existing_collection(
        self,
        expected_size: int | None = None,
        *,
        ensure_payload_indexes: bool = False,
    ) -> int:
        info = self.client.get_collection(self.collection_name)
        size, _distance = self._vector_params(info)
        if expected_size is not None and size != expected_size:
            raise ValueError(
                f"embedding dimension {expected_size} does not match Qdrant "
                f"collection dimension {size}"
            )
        self._known_vector_size = size
        if ensure_payload_indexes:
            self._ensure_payload_indexes(info)
        return size

    def _ensure_collection(self, vector_size: int) -> None:
        self._require_open()
        if isinstance(vector_size, bool) or not isinstance(vector_size, int) or vector_size < 1:
            raise ValueError("vector_size must be a positive integer")
        if self._collection_exists():
            self._validate_existing_collection(
                vector_size, ensure_payload_indexes=True
            )
            return
        self._payload_indexes_ready = False
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._models.VectorParams(
                    size=vector_size,
                    distance=self._models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            if not self._is_conflict(exc):
                raise
        if not self._collection_exists():
            raise RuntimeError("Qdrant collection creation was not durable")
        self._validate_existing_collection(
            vector_size, ensure_payload_indexes=True
        )

    def _qdrant_filter(
        self,
        metadata_filter: Mapping[str, Any] | None = None,
        document_version_id: str | None = None,
        *,
        point_ids: Sequence[str] | None = None,
    ) -> Any | None:
        supplied = _metadata_filter(metadata_filter, document_version_id) or {}
        combined = dict(self.scope_filter)
        for key, value in supplied.items():
            existing = combined.get(key)
            if existing is not None and existing != value:
                raise ValueError(
                    f"metadata filter field {key!r} conflicts with vector-store scope"
                )
            combined[key] = value
        conditions: list[Any] = []
        for key, value in sorted(combined.items()):
            if isinstance(value, float):
                conditions.append(
                    self._models.FieldCondition(
                        key=key,
                        range=self._models.Range(gte=value, lte=value),
                    )
                )
            else:
                conditions.append(
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchValue(value=value),
                    )
                )
        if point_ids is not None:
            conditions.append(self._models.HasIdCondition(has_id=list(point_ids)))
        return self._models.Filter(must=conditions) if conditions else None

    def _apply_scope(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        scoped = dict(metadata)
        for key, value in self.scope_filter.items():
            supplied = scoped.get(key)
            if supplied is not None and supplied != value:
                raise ValueError(
                    f"metadata field {key!r} conflicts with vector-store scope"
                )
            scoped[key] = value
        return scoped

    @staticmethod
    def _payload(record_id: str, content: str, metadata: Mapping[str, Any]) -> dict:
        return {
            **dict(metadata),
            _PAYLOAD_RECORD_ID: record_id,
            _PAYLOAD_CONTENT: content,
            _PAYLOAD_SCHEMA: VECTOR_PAYLOAD_SCHEMA,
        }

    def reset(self):
        """Delete this source scope; reject unscoped collection deletion by default."""

        self._require_open()
        if not self._collection_exists():
            return
        if self.scope_filter:
            selector = self._models.FilterSelector(filter=self._qdrant_filter())
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
                wait=True,
            )
            return
        if not self.allow_collection_reset:
            raise RuntimeError(
                "unscoped Qdrant collection reset is disabled; provide scope_filter "
                "or explicitly set allow_collection_reset=True"
            )
        self.client.delete_collection(collection_name=self.collection_name)
        self._known_vector_size = None
        self._payload_indexes_ready = False

    def count(self) -> int:
        """Return the exact durable record count in this store's source scope."""

        self._require_open()
        if not self._collection_exists():
            return 0
        self._validate_existing_collection()
        response = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._qdrant_filter(),
            exact=True,
        )
        count = getattr(response, "count", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError("Qdrant returned an invalid collection count")
        return count

    def close(self) -> None:
        """Drop client/model references and release any attached read lease."""

        lease = getattr(self, "_read_lease", None)
        self._read_lease = None
        client = getattr(self, "client", None)
        owns_client = getattr(self, "_owns_client", False)
        self._closed = True
        try:
            self.collection = None
            self.client = None
            self.embedding_model = None
            self._models = None
            if owns_client and client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        finally:
            if lease is not None:
                lease.close()

    def attach_read_lease(self, lease) -> None:
        """Transfer one acquired index read lease to this store's lifetime."""

        if lease is None or not callable(getattr(lease, "close", None)):
            raise TypeError("read lease must provide close()")
        if getattr(self, "_read_lease", None) is not None:
            raise RuntimeError("vector store already owns a read lease")
        self._read_lease = lease

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict] = None,
        *,
        batch_size: int = 256,
        strict_source_identity: bool = False,
    ):
        if not texts:
            return
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")
        if not isinstance(strict_source_identity, bool):
            raise TypeError("strict_source_identity must be a boolean")
        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must contain the same number of items")
        source_metadatas = (
            metadatas if metadatas is not None else [{} for _ in texts]
        )
        normalized_metadatas = [
            self._apply_scope(_normalized_metadata(metadata, modality="text"))
            for metadata in source_metadatas
        ]
        ids = [
            stable_vector_record_id(
                text,
                metadata,
                modality="text",
                strict_source_identity=strict_source_identity,
            )
            for text, metadata in zip(texts, normalized_metadatas)
        ]
        unique_indexes = _unique_record_indexes(
            ids,
            texts,
            normalized_metadatas,
            [None] * len(texts),
        )
        log.info("Upserting %d texts into Qdrant", len(unique_indexes))
        for i in range(0, len(unique_indexes), batch_size):
            batch_indexes = unique_indexes[i : i + batch_size]
            batch_texts = [texts[index] for index in batch_indexes]
            batch_embeddings = _normalize_vectors(
                self.embedding_model.embed_texts(batch_texts), len(batch_texts)
            )
            self._ensure_collection(len(batch_embeddings[0]))
            batch_ids = [ids[index] for index in batch_indexes]
            batch_metadatas = [normalized_metadatas[index] for index in batch_indexes]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    self._models.PointStruct(
                        id=qdrant_point_id(record_id),
                        vector=embedding,
                        payload=self._payload(record_id, text, metadata),
                    )
                    for record_id, text, metadata, embedding in zip(
                        batch_ids,
                        batch_texts,
                        batch_metadatas,
                        batch_embeddings,
                    )
                ],
                wait=True,
            )
        log.info("Texts upserted successfully")
        return ids

    def delete_text_by_metadata(self, metadata: dict):
        """Delete records matching metadata within this store's scope."""

        if not metadata:
            return
        self._require_open()
        if not self._collection_exists():
            return
        selector = self._models.FilterSelector(
            filter=self._qdrant_filter(metadata_filter=metadata)
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
            wait=True,
        )
        log.info("Deleted Qdrant records by scoped metadata")

    def delete_text_by_ids(self, ids: List[str]):
        """Delete logical record IDs without crossing this store's scope."""

        if not ids:
            return
        if any(not isinstance(record_id, str) or not record_id for record_id in ids):
            raise ValueError("ids must contain non-empty strings")
        self._require_open()
        if not self._collection_exists():
            return
        physical_ids = [qdrant_point_id(record_id) for record_id in ids]
        if self.scope_filter:
            selector = self._models.FilterSelector(
                filter=self._qdrant_filter(point_ids=physical_ids)
            )
        else:
            selector = self._models.PointIdsList(points=physical_ids)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
            wait=True,
        )
        log.info("Deleted %d Qdrant records by logical ID", len(ids))

    def update_text_metadatas(
        self, ids: List[str], metadatas: List[dict]
    ) -> None:
        """Update metadata without recomputing unchanged text embeddings."""

        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must contain the same number of items")
        if not ids:
            return
        if any(not isinstance(record_id, str) or not record_id for record_id in ids):
            raise ValueError("ids must contain non-empty strings")
        self._require_open()
        if not self._collection_exists():
            return
        for record_id, metadata in zip(ids, metadatas):
            patch = self._apply_scope(_normalized_payload_patch(metadata))
            physical_id = qdrant_point_id(record_id)
            points = (
                self._qdrant_filter(point_ids=[physical_id])
                if self.scope_filter
                else [physical_id]
            )
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=patch,
                points=points,
                wait=True,
            )
        log.info("Updated metadata for %d vector-store entries.", len(ids))

    def add_images(
        self,
        image_paths: List[str],
        metadatas: List[dict] = None,
        image_str: List[str] = None,
        *,
        batch_size: int = 256,
        strict_source_identity: bool = False,
    ):
        if not image_paths:
            return
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")
        if not isinstance(strict_source_identity, bool):
            raise TypeError("strict_source_identity must be a boolean")
        if metadatas is not None and len(image_paths) != len(metadatas):
            raise ValueError(
                "image_paths and metadatas must contain the same number of items"
            )
        if image_str is not None and len(image_paths) != len(image_str):
            raise ValueError(
                "image_paths and image_str must contain the same number of items"
            )
        if not self.embedding_model.MM_EMBEDDER:
            raise ValueError(
                "当前embedding_model不是多模态模型，无法添加图片。请使用支持图片向量化的模型。"
            )
        source_metadatas = (
            metadatas if metadatas is not None else [{} for _ in image_paths]
        )
        normalized_metadatas = [
            self._apply_scope(_normalized_metadata(metadata, modality="image"))
            for metadata in source_metadatas
        ]
        identity_texts = image_str if image_str is not None else [None] * len(image_paths)
        representations = [
            _representation_identity(
                path,
                modality="image",
                image_text=text,
                strict_source_identity=strict_source_identity,
            )
            for path, text in zip(image_paths, identity_texts)
        ]
        for metadata, representation in zip(normalized_metadatas, representations):
            image_sha256 = representation.get("image_sha256")
            if image_sha256 is not None:
                metadata["asset_sha256"] = image_sha256
        ids = [
            _stable_vector_record_id_from_normalized(
                metadata,
                representation,
                modality="image",
                strict_source_identity=strict_source_identity,
            )
            for metadata, representation in zip(
                normalized_metadatas, representations
            )
        ]
        unique_indexes = _unique_record_indexes(
            ids,
            image_paths,
            normalized_metadatas,
            identity_texts,
            compare_values=False,
        )
        log.info(f"Upserting {len(image_paths)} images into the database...")
        for i in range(0, len(unique_indexes), batch_size):
            batch_indexes = unique_indexes[i : i + batch_size]
            batch_paths = [image_paths[index] for index in batch_indexes]
            batch_texts = (
                [image_str[index] for index in batch_indexes]
                if image_str is not None
                else None
            )
            if hasattr(self.embedding_model, "embed_fused"):
                embeddings = self.embedding_model.embed_fused(
                    images=batch_paths, texts=batch_texts
                )
            else:
                embeddings = self.embedding_model.embed_images(batch_paths)

            embeddings = _normalize_vectors(embeddings, len(batch_paths))
            self._ensure_collection(len(embeddings[0]))

            batch_ids = [ids[index] for index in batch_indexes]
            batch_metadatas = [normalized_metadatas[index] for index in batch_indexes]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    self._models.PointStruct(
                        id=qdrant_point_id(record_id),
                        vector=embedding,
                        payload=self._payload(record_id, path, metadata),
                    )
                    for record_id, path, metadata, embedding in zip(
                        batch_ids,
                        batch_paths,
                        batch_metadatas,
                        embeddings,
                    )
                ],
                wait=True,
            )
        log.info("Images upserted successfully.")
        return ids

    @staticmethod
    def _retrieval_result(hit: Any) -> Dict[str, Any]:
        payload = getattr(hit, "payload", None)
        if not isinstance(payload, Mapping):
            raise RuntimeError("Qdrant retrieval hit has no payload")
        if payload.get(_PAYLOAD_SCHEMA) != VECTOR_PAYLOAD_SCHEMA:
            raise RuntimeError("Qdrant retrieval hit has an incompatible payload schema")
        record_id = payload.get(_PAYLOAD_RECORD_ID)
        content = payload.get(_PAYLOAD_CONTENT)
        if not isinstance(record_id, str) or not record_id:
            raise RuntimeError("Qdrant retrieval hit has no logical record ID")
        if not isinstance(content, str):
            raise RuntimeError("Qdrant retrieval hit has no representation content")
        score = getattr(hit, "score", None)
        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise RuntimeError("Qdrant retrieval hit has an invalid score")
        score = float(score)
        if not math.isfinite(score):
            raise RuntimeError("Qdrant retrieval hit has a non-finite score")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in _RESERVED_PAYLOAD_KEYS
        }
        return {
            "id": record_id,
            # Qdrant returns cosine similarity while BookRAG's established
            # retrieval contract exposes cosine distance (lower is better).
            "distance": 1.0 - score,
            "content": content,
            "metadata": metadata,
        }

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
        document_version_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search text and image representations in this source scope."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_filter = self._qdrant_filter(metadata_filter, document_version_id)
        if not self._collection_exists():
            return []
        query_embedding = _normalize_vectors(
            self.embedding_model.embed_texts([query_text]), 1
        )[0]
        self._validate_existing_collection(len(query_embedding))
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", None)
        if not isinstance(points, Iterable):
            raise RuntimeError("Qdrant returned an invalid query response")
        return [self._retrieval_result(hit) for hit in points]

    def batch_search(
        self,
        query_texts: List[str],
        top_k: int = 3,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
        document_version_id: str | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """Search several text queries through Qdrant's batch query API."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query_texts:
            return []
        query_filter = self._qdrant_filter(metadata_filter, document_version_id)
        if not self._collection_exists():
            return [[] for _query in query_texts]
        query_embeddings = _normalize_vectors(
            self.embedding_model.embed_texts(query_texts), len(query_texts)
        )
        self._validate_existing_collection(len(query_embeddings[0]))
        responses = self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=[
                self._models.QueryRequest(
                    query=embedding,
                    filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                for embedding in query_embeddings
            ],
        )
        try:
            response_list = list(responses)
        except TypeError as exc:
            raise RuntimeError("Qdrant returned an invalid batch query response") from exc
        if len(response_list) != len(query_texts):
            raise RuntimeError("Qdrant returned the wrong number of batch responses")
        batch_results: list[list[Dict[str, Any]]] = []
        for response in response_list:
            points = getattr(response, "points", None)
            if not isinstance(points, Iterable):
                raise RuntimeError("Qdrant returned an invalid batch query response")
            batch_results.append([self._retrieval_result(hit) for hit in points])
        return batch_results
