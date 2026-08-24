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
#: Tree points get their own ID and payload namespaces. The shared ones stay put
#: because the entity and chunk collections are written with them, and re-IDing
#: those would orphan live points this migration does not touch.
TREE_VECTOR_RECORD_ID_SCHEMA = "docarag.qdrant-tree-node-record-id.v1"
TREE_NODE_PAYLOAD_SCHEMA = "docarag.qdrant-tree-node-payload.v1"
_QDRANT_POINT_NAMESPACE = uuid.UUID("65dc7f26-27fe-5b93-8df8-76c53255e978")
_PAYLOAD_RECORD_ID = "_docarag_record_id"
_PAYLOAD_CONTENT = "_docarag_content"
#: One point now has a content per named vector, so the single content above
#: becomes the primary and this carries the rest keyed by vector name.
_PAYLOAD_CONTENTS = "_docarag_contents"
_PAYLOAD_SCHEMA = "_docarag_schema"
_RESERVED_PAYLOAD_KEYS = frozenset(
    {_PAYLOAD_RECORD_ID, _PAYLOAD_CONTENT, _PAYLOAD_CONTENTS, _PAYLOAD_SCHEMA}
)
#: Named dense vectors of the hybrid entity collection. ``name`` reproduces the
#: pre-hybrid behaviour exactly, so the schema adds evidence rather than
#: replacing the only signal ever measured against the quality bar.
_HYBRID_DENSE_VECTORS = ("name", "context")
#: Sparse vector carrying term frequencies; Qdrant applies IDF server-side.
_HYBRID_SPARSE_VECTOR = "lexical"
#: Named dense vectors of the tree-node collection. ``text`` is the node's own
#: words, ``context`` the words around or under it, ``image`` its crop. They are
#: separate rather than blended because that is where the separation was
#: measured -- see the equation branch of ``Core/pipelines/vdb_index.py``.
_TREE_DENSE_VECTORS = ("text", "context", "image")
#: The one dense name whose vector is a picture rather than prose.
_TREE_IMAGE_VECTOR = "image"
#: Sparse vector over everything the node says, for server-side lexical search.
_TREE_SPARSE_VECTOR = "lexical"
#: A tree collection holds two families of point. A semantic element is not a
#: node -- it is one cell, symbol or flowchart box inside one -- and it keeps its
#: own point because its element id, kind and box are per-element payload that a
#: shared point could not carry.
_TREE_POINT_KINDS = ("node", "semantic_element")
#: Marks the type-scoped twin of a vector's leg in a ``search_tree_vectors``
#: result. Not a vector name -- Qdrant only ever sees the base name -- so it
#: uses a character no vector name contains, and callers strip it with
#: ``tree_vector_base_name`` before reading anything off the row.
TREE_SCOPED_VECTOR_SUFFIX = "@type"


def tree_vector_base_name(leg_name: str) -> str:
    """The vector a leg queried, with any type-scoping marker removed."""

    if leg_name.endswith(TREE_SCOPED_VECTOR_SUFFIX):
        return leg_name[: -len(TREE_SCOPED_VECTOR_SUFFIX)]
    return leg_name


#: The dense names each schema requires. ``UNNAMED`` is absent on purpose: it is
#: the shape that has no names at all, and the absence is what the guards read.
_EXPECTED_DENSE_VECTORS = {}
_SCHEMA_LABELS = {}


class VectorSchema(str, Enum):
    """Which vector shape a collection carries.

    Qdrant cannot convert an unnamed vector into named ones, so a collection's
    shape is fixed at creation and a store has to know which one it is talking
    to. One VectorStore class serves all three; the guards below make a
    mismatch fail closed at the first call rather than write a point nothing
    can read.
    """

    UNNAMED = "unnamed"
    HYBRID_ENTITY = "hybrid-entity"
    TREE_NODE = "tree-node"


_EXPECTED_DENSE_VECTORS.update(
    {
        VectorSchema.HYBRID_ENTITY: _HYBRID_DENSE_VECTORS,
        VectorSchema.TREE_NODE: _TREE_DENSE_VECTORS,
    }
)
_SCHEMA_LABELS.update(
    {
        VectorSchema.UNNAMED: "unnamed",
        VectorSchema.HYBRID_ENTITY: "hybrid entity",
        VectorSchema.TREE_NODE: "tree-node",
    }
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
    # Grouped entity search and merge repointing both filter on this, so it is
    # indexed for the same reason the scope fields are.
    "canonical_id",
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
    "semantic_element_id",
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


def _scope_filter_values(scope_filter: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Validate a VectorStore's own scope, unlike the per-call metadata_filter
    contract: a field may be one scalar (MatchValue) or a non-empty list of
    scalars (MatchAny), for a scope spanning more than one document.
    """

    if scope_filter is not None and not isinstance(scope_filter, Mapping):
        raise TypeError("scope_filter must be a mapping")
    combined: dict[str, Any] = {}
    if scope_filter:
        for key, value in scope_filter.items():
            if not isinstance(key, str) or not key:
                raise ValueError("scope_filter keys must be non-empty strings")
            if key in _RESERVED_PAYLOAD_KEYS:
                raise ValueError(f"scope_filter may not use reserved key {key!r}")
            if isinstance(value, (list, tuple, set)):
                if not value:
                    raise ValueError(f"scope_filter field {key!r} may not be an empty list")
                normalized_values = []
                for item in value:
                    if not isinstance(item, (Enum, str, int, float, bool)):
                        raise TypeError(
                            "scope_filter list values must be exact-match scalars"
                        )
                    normalized_item = _identity_scalar(item)
                    if normalized_item is None:
                        raise ValueError("scope_filter values may not be null")
                    normalized_values.append(normalized_item)
                combined[key] = tuple(dict.fromkeys(normalized_values))
                continue
            if not isinstance(value, (Enum, str, int, float, bool)):
                raise TypeError("scope_filter values must be exact-match scalar values")
            normalized = _identity_scalar(value)
            if normalized is None:
                raise ValueError("scope_filter values may not be null")
            combined[key] = normalized
    return combined or None


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


def _normalized_metadata(metadata: Mapping[str, Any], *, modality: str | None) -> dict:
    """Coerce metadata to storable scalars, stamping the write's modality.

    ``modality=None`` is the tree-point mode: one point carries several named
    vectors, so there is no single modality to stamp. The fields are rejected
    rather than ignored, because a caller that still supplies one is describing
    a shape that no longer exists and its retrieval would silently read as text.
    """

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
    if modality is None:
        stamped = {key for key in ("modality", "type") if key in result}
        if stamped:
            raise ValueError(
                "tree-point metadata may not carry a point-level "
                + ", ".join(sorted(stamped))
                + "; a point holds several named vectors and each one's"
                " modality is the vector it was written under"
            )
        return result
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


def tree_vector_identity(
    name: str,
    value: str,
    *,
    image_text: str | None = None,
    strict_source_identity: bool = True,
) -> Mapping[str, str]:
    """The source digest for one named vector, reusing the existing rule.

    An image is hashed by its bytes, text by its characters -- the same
    ``_representation_identity`` the per-representation writer used, so a
    vector's identity does not change merely because it moved onto a shared
    point.
    """

    if name not in _TREE_DENSE_VECTORS:
        raise ValueError(f"unknown tree vector name: {name!r}")
    return _representation_identity(
        value,
        modality="image" if name == _TREE_IMAGE_VECTOR else "text",
        image_text=image_text,
        strict_source_identity=strict_source_identity,
    )


def tree_point_record_id(
    *,
    kind: str,
    metadata: Mapping[str, Any],
    representations: Mapping[str, Mapping[str, str]],
    lexical_source: str = "",
    strict_source_identity: bool = True,
) -> str:
    """One ID for a whole point, with no modality in it.

    The per-representation ID bakes the modality in twice, which is exactly what
    has to go: a node's vectors now share a point, so the ID has to name the
    node rather than one of its readings. Every representation digest stays
    inside it, so the property ``_unique_record_indexes`` relies on still holds
    -- the same ID implies the same content, and changing any one vector's
    source re-IDs the point.
    """

    if kind not in _TREE_POINT_KINDS:
        raise ValueError(f"unknown tree point kind: {kind!r}")
    if not representations:
        raise ValueError("a tree point requires at least one vector")
    normalized = _normalized_metadata(metadata, modality=None)
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
                "strict tree point requires document_version_id"
            )
        if not isinstance(source_sha256, str) or _LOWER_SHA256.fullmatch(
            source_sha256
        ) is None:
            raise MissingVectorSourceIdentityError(
                "strict tree point requires lowercase source_sha256"
            )
        if not source_identity:
            raise MissingVectorSourceIdentityError(
                "strict tree point requires a node or evidence identity"
            )
    identity = {
        "schema": TREE_VECTOR_RECORD_ID_SCHEMA,
        "scope": "source-bound-tree-point",
        "record_kind": kind,
        "document_version_id": document_version_id,
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "node_type": normalized.get("node_type"),
        "representations": {
            name: dict(digest) for name, digest in sorted(representations.items())
        },
        # Hashed from the text the terms were counted from rather than the term
        # map, so a tokenizer change is visible in the ID too.
        "lexical_source_sha256": hashlib.sha256(
            (lexical_source or "").encode("utf-8")
        ).hexdigest(),
    }
    return f"record_{kind}_{_canonical_sha256(identity)}"


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
        hybrid_entity_schema: bool = False,
        vector_schema: "VectorSchema | str | None" = None,
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        if embedding_model is None:
            raise TypeError("embedding_model is required")
        if not isinstance(hybrid_entity_schema, bool):
            raise TypeError("hybrid_entity_schema must be a boolean")
        if vector_schema is not None and hybrid_entity_schema:
            raise ValueError("pass vector_schema or hybrid_entity_schema, not both")
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
        normalized_scope = _scope_filter_values(scope_filter)
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
        # Each collection has its own vector shape and Qdrant cannot convert
        # between them, so this is per-instance: one VectorStore class serves
        # the unnamed chunk collections, the hybrid entity collection, and the
        # named tree-node collection. ``hybrid_entity_schema=True`` stays as the
        # spelling its three existing call sites use, one of which is the graph
        # stage this migration must not touch.
        if vector_schema is None:
            vector_schema = (
                VectorSchema.HYBRID_ENTITY
                if hybrid_entity_schema
                else VectorSchema.UNNAMED
            )
        try:
            self.vector_schema = VectorSchema(vector_schema)
        except ValueError as exc:
            raise ValueError(f"unknown vector_schema: {vector_schema!r}") from exc
        self.collection = _CollectionCountView(self)
        self._read_lease = None
        log.info("Qdrant collection '%s' configured at %s", collection_name, qdrant_url)

    @property
    def hybrid_entity_schema(self) -> bool:
        """Kept as a read-only view so the entity call sites read unchanged."""

        return self.vector_schema is VectorSchema.HYBRID_ENTITY

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

    def _require_sparse_vector(self, collection_info: Any) -> None:
        """A tree collection without its sparse vector cannot serve lexical search.

        Fail here rather than at the first keyword query, where the symptom
        would be an empty leg rather than a missing index.
        """

        sparse = getattr(collection_info.config.params, "sparse_vectors", None)
        if not isinstance(sparse, Mapping) or _TREE_SPARSE_VECTOR not in sparse:
            raise RuntimeError(
                f"tree-node collection is missing the {_TREE_SPARSE_VECTOR!r} sparse "
                "vector, so this collection predates hybrid search and must be rebuilt"
            )

    def _vector_params(self, collection_info: Any) -> tuple[int, str]:
        try:
            vectors = collection_info.config.params.vectors
        except AttributeError as exc:
            raise RuntimeError("Qdrant returned an invalid collection configuration") from exc
        expected_names = _EXPECTED_DENSE_VECTORS.get(self.vector_schema)
        if isinstance(vectors, Mapping):
            if expected_names is None:
                raise RuntimeError(
                    "DocARAG requires one unnamed dense vector; named-vector collection "
                    "configuration is incompatible"
                )
            label = _SCHEMA_LABELS[self.vector_schema]
            missing = set(expected_names).difference(vectors)
            if missing:
                raise RuntimeError(
                    f"{label} collection is missing dense vectors: "
                    + ", ".join(sorted(missing))
                )
            # Every dense vector comes from one embedder, so a collection whose
            # halves disagree cannot have been written by this code.
            sizes = {
                name: getattr(vectors[name], "size", None) for name in expected_names
            }
            if len(set(sizes.values())) != 1:
                raise RuntimeError(
                    f"{label} dense vectors disagree on dimension: {sizes}"
                )
            if self.vector_schema is VectorSchema.TREE_NODE:
                self._require_sparse_vector(collection_info)
            vectors = vectors[expected_names[0]]
        elif expected_names is not None:
            label = _SCHEMA_LABELS[self.vector_schema]
            raise RuntimeError(
                f"{label} collection requires named dense vectors; found an "
                f"unnamed vector, so this collection predates the {label} schema "
                "and must be rebuilt"
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

    def require_writable(self) -> None:
        """Raise now if an existing collection cannot accept this store's schema.

        A rebuild resets a scope before it writes, and a schema mismatch that
        surfaces at write time has already destroyed what it was replacing. An
        absent collection is writable: it will be created with this schema.
        """

        self._require_open()
        if not self._collection_exists():
            return
        self._validate_existing_collection()

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
            if self.hybrid_entity_schema:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        name: self._models.VectorParams(
                            size=vector_size,
                            distance=self._models.Distance.COSINE,
                        )
                        for name in _HYBRID_DENSE_VECTORS
                    },
                    sparse_vectors_config={
                        _HYBRID_SPARSE_VECTOR: self._models.SparseVectorParams(
                            index=self._models.SparseIndexParams(),
                            # IDF from collection statistics, so the client
                            # sends raw term frequencies and no vocabulary has
                            # to be persisted or kept in sync with the index.
                            modifier=self._models.Modifier.IDF,
                        )
                    },
                )
            elif self.vector_schema is VectorSchema.TREE_NODE:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        name: self._models.VectorParams(
                            size=vector_size,
                            distance=self._models.Distance.COSINE,
                        )
                        for name in _TREE_DENSE_VECTORS
                    },
                    sparse_vectors_config={
                        _TREE_SPARSE_VECTOR: self._models.SparseVectorParams(
                            index=self._models.SparseIndexParams(),
                            # Same reasoning as the entity collection: IDF is
                            # computed from collection statistics, so the client
                            # sends raw term frequencies and no vocabulary is
                            # persisted. Note those statistics span every
                            # document in the collection -- a scope filter
                            # restricts results, not the corpus IDF is drawn
                            # from.
                            modifier=self._models.Modifier.IDF,
                        )
                    },
                )
            else:
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
            if existing is None:
                combined[key] = value
                continue
            if isinstance(existing, tuple):
                if value not in existing:
                    raise ValueError(
                        f"metadata filter field {key!r} conflicts with vector-store scope"
                    )
                # keep the scope's own (wider) list -- value already verified a member
                continue
            if existing != value:
                raise ValueError(
                    f"metadata filter field {key!r} conflicts with vector-store scope"
                )
            combined[key] = value
        conditions: list[Any] = []
        for key, value in sorted(combined.items()):
            if isinstance(value, tuple):
                conditions.append(
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchAny(any=list(value)),
                    )
                )
            elif isinstance(value, float):
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
            if isinstance(value, tuple):
                # A multi-document (MatchAny) scope reads across documents;
                # a written point belongs to exactly one, so writing through
                # it would tag the point with an ambiguous scope.
                raise ValueError(
                    f"cannot write through a multi-value scope field {key!r}; "
                    "open a single-document-scoped VectorStore for writes"
                )
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
        if any(isinstance(value, tuple) for value in self.scope_filter.values()):
            raise RuntimeError(
                "cannot reset through a multi-value (MatchAny) scope -- it "
                "would delete every document the scope spans, not one"
            )
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

    def live_point_ids(self, *, page_size: int = 1024) -> set[str]:
        """Every point ID durably in this store's scope, and nothing else.

        Read without payload or vectors: this exists to be diffed against a
        projection, and a scope holding 23,274 points would otherwise pull
        every vector across the wire to answer a question about identity.

        The scope filter is what keeps a shared collection safe. The entity
        collection holds every document's entities, so an unscoped listing here
        would invite a caller to delete another document's points as "missing".
        """

        self._require_open()
        if not self._collection_exists():
            return set()
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or page_size < 1
        ):
            raise ValueError("page_size must be at least 1")
        self._validate_existing_collection()
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._qdrant_filter(),
                limit=page_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for point in points:
                point_id = getattr(point, "id", None)
                if point_id is None:
                    raise RuntimeError("Qdrant returned a point with no ID")
                found.add(str(point_id))
            if offset is None or not points:
                return found

    def delete_points(self, point_ids: Sequence[str]) -> int:
        """Delete named points, refusing any that are outside this scope.

        Deletion by ID alone would ignore ``scope_filter`` entirely, which on a
        collection shared between documents is one transposed argument away
        from deleting another document's index. The IDs are checked against the
        scope first, and the deletion itself is filtered as well.
        """

        self._require_open()
        wanted = [str(point_id) for point_id in point_ids]
        if not wanted:
            return 0
        if not self._collection_exists():
            raise RuntimeError("cannot delete from a collection that does not exist")
        self._validate_existing_collection()
        outside = sorted(set(wanted).difference(self.live_point_ids()))
        if outside:
            raise RuntimeError(
                "refusing to delete points outside this store's scope: "
                + ", ".join(outside[:5])
            )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.FilterSelector(
                filter=self._qdrant_filter(point_ids=wanted)
            ),
            wait=True,
        )
        log.info("Deleted %d points from Qdrant", len(set(wanted)))
        return len(set(wanted))

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
        self._require_unnamed()
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

    def _require_hybrid(self) -> None:
        if not self.hybrid_entity_schema:
            raise RuntimeError(
                "this operation requires a vector store opened with "
                "hybrid_entity_schema=True"
            )

    def _require_unnamed(self) -> None:
        """One vector per point, written without a name.

        A named collection would take such a point and silently store it under
        no name, where nothing queries it. Refusing is what makes a collection
        that was never renamed fail at the write rather than at the read.
        """

        if self.vector_schema is not VectorSchema.UNNAMED:
            raise RuntimeError(
                "this operation writes one unnamed vector per point and cannot "
                f"be used on a {_SCHEMA_LABELS[self.vector_schema]} collection"
            )

    def _require_tree_node(self) -> None:
        if self.vector_schema is not VectorSchema.TREE_NODE:
            raise RuntimeError(
                "this operation requires a vector store opened with "
                "vector_schema=VectorSchema.TREE_NODE"
            )

    def add_tree_node_points(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 256,
    ) -> List[str]:
        """Write one point per tree node, carrying all of its vectors.

        Each record supplies ``kind``, ``metadata``, a ``vectors`` map of
        name to source (text for ``text``/``context``, a crop path for
        ``image``), an optional ``image_text`` fused into the crop, and
        ``lexical_terms`` with the ``lexical_source`` they were counted from.

        A point is written in one upsert with every vector it has, because a
        plain upsert replaces a point rather than merging into it -- writing the
        halves separately would leave whichever went second as the whole point.
        Both dense halves go through one embedder call per batch for the same
        reason ``add_entity_points`` does: a second call could observe a changed
        model and produce vectors that are not comparable.
        """

        self._require_tree_node()
        self._require_open()
        if not records:
            return []
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")

        record_ids: List[str] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            record_ids.extend(self._write_tree_point_batch(batch))
        log.info("Upserted %d tree points into Qdrant", len(record_ids))
        return record_ids

    def tree_point_ids(self, records: Sequence[Mapping[str, Any]]) -> List[str]:
        """The Qdrant IDs these records would be written under, without writing.

        A caller diffing a projection against what the store holds has to
        derive IDs the way the writer does or the two sets never intersect.
        The metadata is scoped first, and the scope is part of the ID, so
        computing one from the caller's own unscoped metadata silently yields
        a different point -- every point looks new, every existing point looks
        deleted, and a reconcile rewrites the whole index believing it did not.
        """

        self._require_tree_node()
        return [
            qdrant_point_id(record_id)
            for record_id in self._tree_point_record_ids(records)
        ]

    def _tree_point_record_ids(
        self, records: Sequence[Mapping[str, Any]]
    ) -> List[str]:
        return [
            tree_point_record_id(
                kind=record.get("kind", "node"),
                metadata=self._apply_scope(
                    _normalized_metadata(record["metadata"], modality=None)
                ),
                representations={
                    name: tree_vector_identity(
                        name,
                        value,
                        image_text=record.get("image_text"),
                        strict_source_identity=True,
                    )
                    for name, value in dict(record.get("vectors") or {}).items()
                },
                lexical_source=record.get("lexical_source", ""),
                strict_source_identity=True,
            )
            for record in records
        ]

    def _write_tree_point_batch(self, batch: Sequence[Mapping[str, Any]]) -> List[str]:
        metadatas = [
            self._apply_scope(_normalized_metadata(record["metadata"], modality=None))
            for record in batch
        ]
        vector_maps = [dict(record.get("vectors") or {}) for record in batch]
        image_texts = [record.get("image_text") for record in batch]
        for vectors in vector_maps:
            unknown = set(vectors).difference(_TREE_DENSE_VECTORS)
            if unknown:
                raise ValueError(
                    "unknown tree vector names: " + ", ".join(sorted(unknown))
                )
            if not vectors:
                raise ValueError("a tree point requires at least one vector")

        ids = [
            tree_point_record_id(
                kind=record.get("kind", "node"),
                metadata=metadata,
                representations={
                    name: tree_vector_identity(
                        name, value, image_text=image_text, strict_source_identity=True
                    )
                    for name, value in vectors.items()
                },
                lexical_source=record.get("lexical_source", ""),
                strict_source_identity=True,
            )
            for record, metadata, vectors, image_text in zip(
                batch, metadatas, vector_maps, image_texts
            )
        ]
        # The ID already carries every vector's source digest, so two records
        # sharing one necessarily share their vectors; metadata is the only
        # thing left that can disagree.
        keep = _unique_record_indexes(
            ids, [None] * len(ids), metadatas, [None] * len(ids), compare_values=False
        )

        dense = self._embed_tree_vectors(
            [vector_maps[index] for index in keep],
            [image_texts[index] for index in keep],
        )
        if not dense:
            return []
        self._ensure_collection(len(next(iter(dense[0].values()))))

        points = []
        for position, index in enumerate(keep):
            record = batch[index]
            vectors: dict[str, Any] = dict(dense[position])
            terms = record.get("lexical_terms") or {}
            if terms:
                vectors[_TREE_SPARSE_VECTOR] = self._models.SparseVector(
                    indices=[int(term) for term in terms],
                    values=[float(value) for value in terms.values()],
                )
            contents = dict(vector_maps[index])
            metadata = dict(metadatas[index])
            metadata["record_kind"] = record.get("kind", "node")
            crop = contents.get(_TREE_IMAGE_VECTOR)
            if crop:
                digest = tree_vector_identity(
                    _TREE_IMAGE_VECTOR,
                    crop,
                    image_text=image_texts[index],
                    strict_source_identity=True,
                ).get("image_sha256")
                if digest is not None:
                    metadata["asset_sha256"] = digest
            points.append(
                self._models.PointStruct(
                    id=qdrant_point_id(ids[index]),
                    vector=vectors,
                    payload=self._tree_point_payload(ids[index], contents, metadata),
                )
            )
        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )
        return [ids[index] for index in keep]

    def _embed_tree_vectors(
        self,
        vector_maps: Sequence[Mapping[str, str]],
        image_texts: Sequence[str | None],
    ) -> List[dict[str, Any]]:
        """Embed a batch's vectors, one call per kind rather than per point."""

        text_slots: list[tuple[int, str]] = []
        text_values: list[str] = []
        for position, vectors in enumerate(vector_maps):
            for name in _TREE_DENSE_VECTORS:
                if name == _TREE_IMAGE_VECTOR or name not in vectors:
                    continue
                text_slots.append((position, name))
                text_values.append(vectors[name])

        embedded: List[dict[str, Any]] = [{} for _ in vector_maps]
        if text_values:
            vectors = _normalize_vectors(
                self.embedding_model.embed_texts(text_values), len(text_values)
            )
            for (position, name), vector in zip(text_slots, vectors):
                embedded[position][name] = vector

        crops = [
            (position, vectors[_TREE_IMAGE_VECTOR], image_texts[position])
            for position, vectors in enumerate(vector_maps)
            if _TREE_IMAGE_VECTOR in vectors
        ]
        if crops:
            if not getattr(self.embedding_model, "MM_EMBEDDER", False):
                raise ValueError(
                    "a crop vector requires a multi-modal embedding model"
                )
            # Split by whether a crop carries fused text: a pure image vector is
            # a different embedding call, not the same one with an empty string,
            # and embed_fused rejects empty text outright.
            fused = [item for item in crops if item[2]]
            pure = [item for item in crops if not item[2]]
            for group, vectors in (
                (
                    fused,
                    self.embedding_model.embed_fused(
                        images=[item[1] for item in fused],
                        texts=[item[2] for item in fused],
                    )
                    if fused
                    else [],
                ),
                (
                    pure,
                    self.embedding_model.embed_images([item[1] for item in pure])
                    if pure
                    else [],
                ),
            ):
                if not group:
                    continue
                for (position, _path, _text), vector in zip(
                    group, _normalize_vectors(vectors, len(group))
                ):
                    embedded[position][_TREE_IMAGE_VECTOR] = vector
        return embedded

    @staticmethod
    def _tree_point_payload(
        record_id: str, contents: Mapping[str, str], metadata: Mapping[str, Any]
    ) -> dict:
        # The primary content keeps the single-content invariant every existing
        # reader relies on; the per-name map is what a named-vector read uses to
        # return the text that actually matched.
        primary = ""
        for name in _TREE_DENSE_VECTORS:
            if contents.get(name):
                primary = contents[name]
                break
        return {
            **dict(metadata),
            _PAYLOAD_RECORD_ID: record_id,
            _PAYLOAD_CONTENT: primary,
            _PAYLOAD_CONTENTS: dict(contents),
            _PAYLOAD_SCHEMA: TREE_NODE_PAYLOAD_SCHEMA,
        }

    def add_entity_points(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 256,
    ) -> List[str]:
        """Write one point per entity carrying all three vectors.

        Each record supplies ``name_text`` (embedded as today), ``context_text``
        (embedded with the description), ``lexical_terms`` (raw frequencies for
        server-side IDF) and ``metadata``.  Both dense vectors go through one
        embedder call per batch so the collection's halves cannot drift apart.
        """

        self._require_hybrid()
        self._require_open()
        if not records:
            return []
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")

        record_ids: List[str] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            name_texts = [record["name_text"] for record in batch]
            context_texts = [record["context_text"] for record in batch]
            metadatas = [
                self._apply_scope(
                    _normalized_metadata(record["metadata"], modality="text")
                )
                for record in batch
            ]
            ids = [
                stable_vector_record_id(
                    name_text,
                    metadata,
                    modality="text",
                    strict_source_identity=True,
                )
                for name_text, metadata in zip(name_texts, metadatas)
            ]
            # One call over both halves: a second call could observe a changed
            # model and produce vectors that are not comparable.
            embeddings = _normalize_vectors(
                self.embedding_model.embed_texts(name_texts + context_texts),
                2 * len(batch),
            )
            name_vectors = embeddings[: len(batch)]
            context_vectors = embeddings[len(batch) :]
            self._ensure_collection(len(name_vectors[0]))
            points = []
            for index, record in enumerate(batch):
                terms = record["lexical_terms"] or {}
                points.append(
                    self._models.PointStruct(
                        id=qdrant_point_id(ids[index]),
                        vector={
                            _HYBRID_DENSE_VECTORS[0]: name_vectors[index],
                            _HYBRID_DENSE_VECTORS[1]: context_vectors[index],
                            _HYBRID_SPARSE_VECTOR: self._models.SparseVector(
                                indices=[int(term) for term in terms],
                                values=[float(value) for value in terms.values()],
                            ),
                        },
                        payload=self._payload(
                            ids[index], name_texts[index], metadatas[index]
                        ),
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
            record_ids.extend(ids)
        log.info("Upserted %d entity points into Qdrant", len(record_ids))
        return record_ids

    def search_entities(
        self,
        *,
        name_text: str,
        context_text: str,
        lexical_terms: Mapping[int, float],
        top_k: int,
        group_by: str,
        name_score_threshold: float | None = None,
        context_score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Fuse the three vectors and return one hit per distinct entity.

        ``group_by`` collapses a merged entity's several points server-side, so
        BookRAG 4.3.2 sees one row per entity and ``top_k`` counts entities
        rather than points -- no over-fetch, and duplicates cannot reach the
        gradient through a caller that forgot to collapse them.

        The score thresholds are the distance gate, pushed into the prefetches.
        A candidate enters fusion only by clearing a floor on some vector, so an
        empty result means nothing plausible was found -- the same outcome the
        pre-hybrid gate produced, now reachable through the context and lexical
        vectors as well.
        """

        self._require_hybrid()
        self._require_open()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not isinstance(group_by, str) or not group_by.strip():
            raise ValueError("group_by must be a non-empty string")
        if not self._collection_exists():
            return []

        query_filter = self._qdrant_filter(metadata_filter, None)
        embeddings = _normalize_vectors(
            self.embedding_model.embed_texts([name_text, context_text]), 2
        )
        self._validate_existing_collection(len(embeddings[0]))
        # Fusion needs a deeper pool than it returns: a point ranked outside a
        # prefetch's own limit cannot be recovered by the fusion step.
        prefetch_limit = max(top_k * 4, 32)
        dense = [
            self._models.Prefetch(
                query=embeddings[0],
                using=_HYBRID_DENSE_VECTORS[0],
                filter=query_filter,
                limit=prefetch_limit,
                score_threshold=name_score_threshold,
            ),
            self._models.Prefetch(
                query=embeddings[1],
                using=_HYBRID_DENSE_VECTORS[1],
                filter=query_filter,
                limit=prefetch_limit,
                score_threshold=context_score_threshold,
            ),
        ]
        prefetches = list(dense)
        if lexical_terms:
            # Nested, so the sparse vector reorders what the dense floors
            # already admitted rather than contributing candidates of its own.
            # IDF scores are unbounded and depend on term rarity, so there is no
            # threshold that would mean for it what a cosine floor means for the
            # dense vectors -- left flat it returns the nearest available match
            # for anything at all, and the distance gate can never fire.
            prefetches.append(
                self._models.Prefetch(
                    prefetch=list(dense),
                    query=self._models.SparseVector(
                        indices=[int(term) for term in lexical_terms],
                        values=[float(v) for v in lexical_terms.values()],
                    ),
                    using=_HYBRID_SPARSE_VECTOR,
                    limit=prefetch_limit,
                )
            )
        response = self.client.query_points_groups(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=self._models.FusionQuery(fusion=self._models.Fusion.RRF),
            group_by=group_by,
            group_size=1,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        groups = getattr(response, "groups", None)
        if not isinstance(groups, Iterable):
            raise RuntimeError("Qdrant returned an invalid grouped query response")
        results = []
        for group in groups:
            hits = getattr(group, "hits", None)
            if not hits:
                continue
            results.append(self._retrieval_result(hits[0]))
        return results

    def set_canonical_id(
        self, *, field_name: str, current_value: str, canonical_value: str
    ) -> None:
        """Repoint every point of a merged entity in one server-side call.

        Filtering on the field rather than on point ids makes a merge chain
        collapse correctly: repointing C to E moves the points that already
        followed A into C along with C's own, without the caller tracking them.
        """

        self._require_hybrid()
        self._require_open()
        for value, label in ((current_value, "current"), (canonical_value, "canonical")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}_value must be a non-empty string")
        if not self._collection_exists() or current_value == canonical_value:
            return
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={field_name: canonical_value},
            points=self._models.FilterSelector(
                filter=self._qdrant_filter({field_name: current_value})
            ).filter,
            wait=True,
        )

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
        self._require_unnamed()
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
                "embedding_model is not multimodal; images require a model "
                "that supports image vectorization"
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
    def _retrieval_result(hit: Any, *, vector_name: str | None = None) -> Dict[str, Any]:
        payload = getattr(hit, "payload", None)
        if not isinstance(payload, Mapping):
            raise RuntimeError("Qdrant retrieval hit has no payload")
        if payload.get(_PAYLOAD_SCHEMA) not in (
            VECTOR_PAYLOAD_SCHEMA,
            TREE_NODE_PAYLOAD_SCHEMA,
        ):
            raise RuntimeError("Qdrant retrieval hit has an incompatible payload schema")
        record_id = payload.get(_PAYLOAD_RECORD_ID)
        content = payload.get(_PAYLOAD_CONTENT)
        if vector_name is not None:
            contents = payload.get(_PAYLOAD_CONTENTS)
            if isinstance(contents, Mapping) and contents.get(vector_name):
                content = contents[vector_name]
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
        if vector_name is not None:
            # A tree point has no point-level modality; the modality of a hit is
            # the vector it was matched on. Reporting it here is what lets every
            # existing consumer of metadata["modality"] read unchanged.
            metadata["vector_name"] = vector_name
            metadata["modality"] = (
                "image" if vector_name == _TREE_IMAGE_VECTOR else "text"
            )
            metadata.setdefault("type", metadata["modality"])
            # One point can be hit on several vectors, and callers key audit
            # rows by this id -- a bare point id would let a node's image row
            # overwrite its text row.
            record_id = f"{record_id}#{vector_name}"
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
        if self.vector_schema is VectorSchema.TREE_NODE:
            # A named collection cannot answer an unnamed query. Merging the
            # dense names by score reproduces what the single unnamed vector
            # did -- one ranked list with pictures and prose scored against
            # each other -- so callers that never named a vector are unaffected.
            merged: List[Dict[str, Any]] = []
            for rows in self._tree_vector_rows(
                query_embedding, query_filter, top_k, _TREE_DENSE_VECTORS
            ).values():
                merged.extend(rows)
            merged.sort(key=lambda row: row["distance"])
            return merged[:top_k]
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

    def search_tree_vectors(
        self,
        query_text: str,
        top_k: int = 3,
        *,
        dense_names: Sequence[str] = _TREE_DENSE_VECTORS,
        lexical_terms: Mapping[int, float] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        scoped_node_types: Sequence[str] = (),
    ) -> Dict[str, List[Dict[str, Any]]]:
        """One ranked list per vector name, from a single round trip.

        Returned separately rather than fused server-side: which vector matched
        is the signal this schema exists to provide, and a server-side fusion
        would collapse it into one opaque list and leave the client's own RRF
        trace claiming an arithmetic it never performed.

        ``scoped_node_types`` adds a second list per name, restricted to those
        node types, under the ``name + TREE_SCOPED_VECTOR_SUFFIX`` key. Added
        rather than substituted: a question that asks to be shown a figure
        still needs the paragraph that explains it, so the unrestricted lists
        stay exactly as they were and the restricted ones give the requested
        kind a ranking it cannot be crowded out of.
        """

        self._require_tree_node()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be at least 1")
        unknown = set(dense_names).difference(_TREE_DENSE_VECTORS)
        if unknown:
            raise ValueError("unknown tree vector names: " + ", ".join(sorted(unknown)))
        query_filter = self._qdrant_filter(metadata_filter, None)
        if not self._collection_exists():
            return {}
        query_embedding = _normalize_vectors(
            self.embedding_model.embed_texts([query_text]), 1
        )[0]
        self._validate_existing_collection(len(query_embedding))
        return self._tree_vector_rows(
            query_embedding,
            query_filter,
            top_k,
            dense_names,
            lexical_terms,
            self._type_scoped_filter(query_filter, scoped_node_types),
        )

    def _type_scoped_filter(
        self, query_filter: Any, node_types: Sequence[str]
    ) -> Any | None:
        """The scope filter narrowed to a set of node types, or None.

        ``node_type`` carries a keyword payload index, so this costs a filtered
        search rather than a scan. The scope conditions are kept: a type-scoped
        leg must not reach outside the document the rest of the query is bound
        to.
        """

        wanted = [str(node_type) for node_type in node_types if str(node_type)]
        if not wanted:
            return None
        condition = self._models.FieldCondition(
            key="node_type",
            match=self._models.MatchAny(any=sorted(set(wanted))),
        )
        existing = list(getattr(query_filter, "must", None) or [])
        return self._models.Filter(must=[*existing, condition])

    def _tree_vector_rows(
        self,
        query_embedding: Any,
        query_filter: Any,
        top_k: int,
        dense_names: Sequence[str],
        lexical_terms: Mapping[int, float] | None = None,
        scoped_filter: Any | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        def sparse_query() -> Any:
            return self._models.SparseVector(
                indices=[int(term) for term in lexical_terms or {}],
                values=[float(value) for value in (lexical_terms or {}).values()],
            )

        names: list[str] = []
        requests: list[Any] = []
        # The unrestricted legs first, then the same legs restricted to the
        # requested node types. One batch either way, so scoping costs a round
        # trip's worth of extra requests rather than a second round trip.
        for scope, suffix in ((query_filter, ""), (scoped_filter, TREE_SCOPED_VECTOR_SUFFIX)):
            if suffix and scoped_filter is None:
                continue
            for name in dense_names:
                names.append(f"{name}{suffix}")
                requests.append(
                    self._models.QueryRequest(
                        query=query_embedding,
                        filter=scope,
                        limit=top_k,
                        with_payload=True,
                        with_vector=False,
                        using=name,
                    )
                )
            if lexical_terms:
                names.append(f"{_TREE_SPARSE_VECTOR}{suffix}")
                requests.append(
                    self._models.QueryRequest(
                        query=sparse_query(),
                        filter=scope,
                        limit=top_k,
                        with_payload=True,
                        with_vector=False,
                        using=_TREE_SPARSE_VECTOR,
                    )
                )
        responses = self.client.query_batch_points(
            collection_name=self.collection_name, requests=requests
        )
        if not isinstance(responses, Iterable):
            raise RuntimeError("Qdrant returned an invalid batch query response")
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for name, response in zip(names, responses):
            points = getattr(response, "points", None)
            if not isinstance(points, Iterable):
                raise RuntimeError("Qdrant returned an invalid query response")
            # A row records the vector it matched on, not the leg it arrived
            # under: a scoped and an unscoped hit on the same vector are the
            # same evidence, and giving them different row ids would stop them
            # deduplicating into one candidate.
            # The sparse name has no content of its own, so it falls through to
            # the point's primary text -- which is what it was matched on.
            rows[name] = [
                self._retrieval_result(hit, vector_name=tree_vector_base_name(name))
                for hit in points
            ]
        return rows

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
                    with_vector=False,
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
