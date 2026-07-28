import os
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterator, List, Literal, Tuple
from pathlib import Path
import hashlib
import stat
import pandas as pd
import numpy as np

from Core.provider.embedding import (
    TextEmbeddingProvider,
    GmeEmbeddingProvider,
)
from Core.provider.llm import LLM
from Core.provider.vdb import (
    VECTOR_RECORD_ID_SCHEMA,
    VectorStore,
    stable_vector_record_id,
)
from Core.Index.Tree import DocumentTree, NodeType
from Core.configs.vdb_config import VDBConfig
from Core.configs.system_config import SystemConfig
from Core.utils.utils import TextProcessor
from Core.utils.raptor_utils import raptor_tree
from Core.utils.bm25 import BM25
from Core.utils.artifact_paths import (
    QdrantCollectionState,
    qdrant_collection_state,
    qdrant_endpoint_sha256,
    remove_artifact_path,
    resolve_run_artifact_path,
    valid_lexical_index,
)
from Core.utils.vector_index_manifest import (
    invalidate_vector_index_attestation,
    publish_vector_index_attestation,
    read_vector_index_attestation,
    vector_index_lock,
    vector_index_attestation_matches,
)
import json
import logging

log = logging.getLogger(__name__)

save_path = "/home/wangshu/multimodal/GBC-RAG/test/sf/"


DEFAULT_TREE_INDEX_BATCH_SIZE = 256
TREE_VECTOR_INDEX_SCHEMA = "docarag.bookrag-tree-qdrant-index.v6"
TREE_VECTOR_PROJECTION_SCHEMA = "docarag.bookrag-tree-vector-projection.v6"
MAX_VECTOR_SEMANTIC_REFS = 128
MAX_VECTOR_SOURCE_EVIDENCE_REFS = 64
MAX_VECTOR_SEMANTIC_REFS_BYTES = 65_536
MAX_SEMANTIC_ELEMENT_EMBED_TEXT_CHARS = 16_384
_TREE_LOCAL_PATH_FIELDS = frozenset(
    {
        "file_path",
        "snapshot_path",
        "canonical_artifact_path",
        "routed_audit_artifact_path",
        "img_path",
        "page_path",
    }
)
_VECTOR_ATTESTATION_IDENTITY_FIELDS = (
    "schema",
    "source_sha256",
    "document_version_id",
    "index_revision_id",
    "tree_sha256",
    "tree_artifact_sha256",
    "projection_sha256",
    "node_count",
    "point_count",
    "collection_name",
    "scope_filter",
    "qdrant_endpoint_sha256",
    "vector_config_sha256",
    "embedding_identity_sha256",
    "store_seal_sha256",
)


def _store_revision_seal(revision_id: str) -> str | None:
    """Read the sealed tree digest for a revision from the store."""

    import psycopg

    url = os.environ.get("DOCARAG_DATABASE_URL")
    if not url:
        return None
    try:
        with psycopg.connect(url, connect_timeout=5) as connection:
            row = connection.execute(
                "SELECT seal.tree_sha256 FROM docarag.revision_seal seal "
                "JOIN docarag.index_revisions rev "
                "  ON rev.revision_key = seal.revision_key "
                "WHERE rev.revision_id = %s",
                (revision_id,),
            ).fetchone()
    except psycopg.Error:
        return None
    return str(row[0]) if row else None


class VectorIndexBindingError(RuntimeError):
    """A tree vector index cannot be bound to immutable source/config state."""


@dataclass(frozen=True)
class TreeIndexRecord:
    """One vector representation emitted from a BookRAG tree node.

    A node can produce both a textual and a visual representation.  Keeping
    these as individual records lets the index builder stream the tree without
    constructing document-sized text/image lists.
    """

    modality: Literal["text", "image"]
    value: str
    metadata: dict
    image_text: str | None = None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_stream_item(digest, value: object) -> None:
    encoded = _canonical_json_bytes(value)
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _asset_sha256(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bookrag_tree_artifact_sha256(
    path_value: str | os.PathLike[str],
    *,
    required: bool = True,
) -> str | None:
    """Hash a stable, regular tree artifact without following its final link."""

    path = Path(path_value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        if not required and not path.exists():
            return None
        raise VectorIndexBindingError(
            f"BookRAG tree artifact is unavailable: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VectorIndexBindingError(
                f"BookRAG tree artifact is not a regular file: {path}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise VectorIndexBindingError(
                f"BookRAG tree artifact changed while hashing: {path}"
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _embedding_identity(vdb_cfg: VDBConfig) -> dict[str, object]:
    embedding = vdb_cfg.embedding_config
    model_name = getattr(embedding, "model_name", None)
    if not isinstance(model_name, str) or not model_name.strip():
        raise VectorIndexBindingError("embedding model_name must be non-empty")
    if model_name != model_name.strip():
        raise VectorIndexBindingError(
            "embedding model_name may not have surrounding whitespace"
        )
    if vdb_cfg.mm_embedding:
        return {
            "provider": "gme",
            "model_name": model_name,
        }

    backend = getattr(embedding, "backend", None)
    if not isinstance(backend, str) or not backend.strip():
        raise VectorIndexBindingError("text embedding backend must be non-empty")
    if backend != backend.strip():
        raise VectorIndexBindingError(
            "text embedding backend may not have surrounding whitespace"
        )
    max_length = getattr(embedding, "max_length", None)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise VectorIndexBindingError("text embedding max_length must be positive")
    api_base = getattr(embedding, "api_base", None)
    api_base_sha256 = None
    if api_base is not None:
        if not isinstance(api_base, str) or not api_base.strip():
            raise VectorIndexBindingError("embedding api_base must be non-empty or null")
        if api_base != api_base.strip():
            raise VectorIndexBindingError(
                "embedding api_base may not have surrounding whitespace"
            )
        # The endpoint participates in identity without leaking deployment
        # topology into a portable run manifest.
        api_base_sha256 = hashlib.sha256(api_base.encode("utf-8")).hexdigest()
    return {
        "provider": "text",
        "backend": backend,
        "model_name": model_name,
        "max_length": max_length,
        "api_base_sha256": api_base_sha256,
    }


def _vector_config_identity(vdb_cfg: VDBConfig) -> dict[str, object]:
    collection_name = getattr(vdb_cfg, "collection_name", None)
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise VectorIndexBindingError("vector collection_name must be non-empty")
    if collection_name != collection_name.strip():
        raise VectorIndexBindingError(
            "vector collection_name may not have surrounding whitespace"
        )
    if not isinstance(vdb_cfg.mm_embedding, bool):
        raise VectorIndexBindingError("vdb.mm_embedding must be a boolean")
    endpoint_sha256 = qdrant_endpoint_sha256(vdb_cfg.qdrant_url)
    return {
        "projection_schema": TREE_VECTOR_PROJECTION_SCHEMA,
        "record_id_schema": VECTOR_RECORD_ID_SCHEMA,
        "backend": "qdrant",
        "collection_name": collection_name,
        "qdrant_endpoint_sha256": endpoint_sha256,
        "distance_metric": "cosine",
        "multimodal": vdb_cfg.mm_embedding,
        "embedding": _embedding_identity(vdb_cfg),
    }


def _validated_tenant_scope(vdb_cfg: VDBConfig) -> dict[str, str]:
    values = {
        "tenant_id": getattr(vdb_cfg, "tenant_id", None),
        "corpus_id": getattr(vdb_cfg, "corpus_id", None),
    }
    for field_name, value in values.items():
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise VectorIndexBindingError(
                f"vdb.{field_name} must be a non-empty trimmed string"
            )
    return values


def _validated_index_revision_id(
    vdb_cfg: VDBConfig,
    *,
    run_directory: str | os.PathLike[str],
) -> str:
    """Return an explicit run revision or a deterministic direct-call default."""

    configured = getattr(vdb_cfg, "index_revision_id", None)
    derived = Path(run_directory).expanduser().resolve().name
    value = configured if configured is not None else derived
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
    ):
        raise VectorIndexBindingError("index_revision_id must be a safe identifier")
    return value


def _document_vector_scope(
    *,
    source_sha256: str,
    document_version_id: str,
    vdb_cfg: VDBConfig,
    run_directory: str | os.PathLike[str],
    index_kind: str,
) -> dict[str, str]:
    """Derive one document's immutable payload scope.

    Writers and readers must derive this the same way or a read silently sees
    another document's points -- or, worse, none of its own.  Deriving it in one
    place is what keeps them from drifting apart.
    """

    if not _is_lower_sha256(source_sha256):
        raise VectorIndexBindingError("tree source_sha256 must be lowercase SHA-256")
    if (
        not isinstance(document_version_id, str)
        or not document_version_id
        or document_version_id != document_version_id.strip()
    ):
        raise VectorIndexBindingError("tree document_version_id must be non-empty")
    configured_document_version = getattr(vdb_cfg, "document_version_id", None)
    if (
        configured_document_version is not None
        and configured_document_version != document_version_id
    ):
        raise VectorIndexBindingError(
            "tree document_version_id differs from the configured PostgreSQL scope"
        )
    return {
        **_validated_tenant_scope(vdb_cfg),
        "document_version_id": document_version_id,
        "index_revision_id": _validated_index_revision_id(
            vdb_cfg, run_directory=run_directory
        ),
        "source_sha256": source_sha256,
        "index_kind": index_kind,
    }


def tree_vector_scope(tree: DocumentTree, vdb_cfg: VDBConfig) -> dict[str, str]:
    """Return the immutable payload scope shared by tree writes and reads."""

    return _document_vector_scope(
        source_sha256=tree.meta_info.source_sha256,
        document_version_id=tree.meta_info.document_version_id,
        vdb_cfg=vdb_cfg,
        run_directory=tree.save_dir,
        index_kind="tree",
    )


def entity_vector_scope(
    meta_info, vdb_cfg: VDBConfig, *, run_directory: str | os.PathLike[str]
) -> dict[str, str]:
    """Return the scope entity points are written with and must be read through.

    Takes the tree's metadata rather than the tree so the query path -- which
    holds a stored view, not a ``DocumentTree`` -- reaches the same scope the
    refiner wrote, through the same code.
    """

    return _document_vector_scope(
        source_sha256=getattr(meta_info, "source_sha256", ""),
        document_version_id=getattr(meta_info, "document_version_id", ""),
        vdb_cfg=vdb_cfg,
        run_directory=run_directory,
        index_kind="entity",
    )


def _qdrant_state(
    vdb_cfg: VDBConfig,
    scope_filter: dict[str, str],
) -> QdrantCollectionState:
    """Read the exact live Qdrant state using only secret references."""

    return qdrant_collection_state(
        qdrant_url=vdb_cfg.qdrant_url,
        collection_name=vdb_cfg.collection_name,
        scope_filter=scope_filter,
        api_key_env=vdb_cfg.api_key_env,
        timeout_seconds=vdb_cfg.timeout_seconds,
    )


def _tree_semantic_sha256(tree: DocumentTree) -> str:
    """Hash complete tree truth for manifest invalidation only.

    This full metadata snapshot is streamed into a local digest and is never
    passed to ``VectorStore``. Persisted record metadata is produced solely by
    :func:`_node_trace_metadata`, which projects semantic graphs to receipts.
    """

    digest = hashlib.sha256()
    _hash_stream_item(digest, {"schema": TREE_VECTOR_INDEX_SCHEMA})
    for node in tree.nodes:
        metadata = node.meta_info.model_dump(mode="json")
        sanitized = {
            key: value
            for key, value in metadata.items()
            if key not in _TREE_LOCAL_PATH_FIELDS
        }
        asset_digest = _asset_sha256(node.meta_info.img_path)
        if asset_digest is not None:
            sanitized["asset_sha256"] = asset_digest
        node_type = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        _hash_stream_item(
            digest,
            {
                "index_id": node.index_id,
                "parent_id": node.parent.index_id if node.parent is not None else None,
                "depth": node.depth,
                "node_type": node_type,
                "outline_node": bool(node.outline_node),
                "summary": node.summary,
                "metadata": sanitized,
            },
        )
    return digest.hexdigest()


def _tree_serialized_state_sha256(tree: DocumentTree) -> str:
    """Hash all serialized node state without reading referenced asset bytes.

    Like :func:`_tree_semantic_sha256`, this is a local integrity digest rather
    than vector-record metadata.
    """

    digest = hashlib.sha256()
    _hash_stream_item(digest, {"schema": TREE_VECTOR_INDEX_SCHEMA})
    for node in tree.nodes:
        node_type = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        _hash_stream_item(
            digest,
            {
                "index_id": node.index_id,
                "parent_id": node.parent.index_id if node.parent is not None else None,
                "depth": node.depth,
                "node_type": node_type,
                "outline_node": bool(node.outline_node),
                "summary": node.summary,
                "metadata": node.meta_info.model_dump(mode="json"),
            },
        )
    return digest.hexdigest()


def _validated_tree_artifact_sha256(tree: DocumentTree) -> str | None:
    """Bind an in-memory build input to the exact saved compatibility tree."""

    if getattr(tree, "store_backed", False):
        # The build input was read from the sealed store rows; the seal is the
        # saved-artifact identity, and there is no pickle to bind to.
        return getattr(tree, "store_seal", None)
    tree_artifact_path = DocumentTree.get_save_path(tree.save_dir)
    artifact_sha256 = bookrag_tree_artifact_sha256(
        tree_artifact_path,
        required=False,
    )
    if artifact_sha256 is None:
        return None
    saved_tree = DocumentTree.load_from_file(tree_artifact_path)
    artifact_sha256_after = bookrag_tree_artifact_sha256(tree_artifact_path)
    if artifact_sha256 != artifact_sha256_after:
        raise VectorIndexBindingError(
            "saved BookRAG tree changed while binding the vector build"
        )
    if _tree_serialized_state_sha256(saved_tree) != _tree_serialized_state_sha256(tree):
        raise VectorIndexBindingError(
            "in-memory BookRAG tree differs from the saved tree artifact"
        )
    return artifact_sha256_after


def expected_tree_vector_index_attestation(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
) -> dict[str, object]:
    """Compute the exact, secret-free identity required for index reuse."""

    if tree.root_node is None or not tree.nodes:
        raise VectorIndexBindingError("vector index requires a non-empty BookRAG tree")
    source_sha256 = tree.meta_info.source_sha256
    if not _is_lower_sha256(source_sha256):
        raise VectorIndexBindingError("tree source_sha256 must be lowercase SHA-256")
    document_version_id = tree.meta_info.document_version_id
    if not isinstance(document_version_id, str) or not document_version_id.strip():
        raise VectorIndexBindingError("tree document_version_id must be non-empty")
    if document_version_id != document_version_id.strip():
        raise VectorIndexBindingError(
            "tree document_version_id may not have surrounding whitespace"
        )
    bound_metadata = (("tree root", tree.root_node.meta_info),)
    node_metadata = (
        (f"tree node {node.index_id}", node.meta_info)
        for node in tree.get_nodes()
    )
    for label, metadata in chain(bound_metadata, node_metadata):
        if metadata.source_sha256 not in (None, source_sha256):
            raise VectorIndexBindingError(f"{label} belongs to another source SHA-256")
        if metadata.document_version_id not in (None, document_version_id):
            raise VectorIndexBindingError(
                f"{label} belongs to another document version"
            )
    if tree.root_node.meta_info.source_sha256 != source_sha256:
        raise VectorIndexBindingError("tree root is missing source SHA-256 binding")
    if tree.root_node.meta_info.document_version_id != document_version_id:
        raise VectorIndexBindingError("tree root is missing document-version binding")

    vector_config = _vector_config_identity(vdb_cfg)
    collection_name = str(vector_config["collection_name"])

    projection_digest = hashlib.sha256()
    _hash_stream_item(
        projection_digest,
        {"schema": TREE_VECTOR_PROJECTION_SCHEMA},
    )
    point_count = 0
    for record in iter_tree_index_records(tree):
        if record.modality == "image" and not vdb_cfg.mm_embedding:
            continue
        if record.metadata.get("document_version_id") != document_version_id:
            raise VectorIndexBindingError(
                f"tree node {record.metadata.get('node_id')} vector record has "
                "another document version"
            )
        if record.metadata.get("source_sha256") != source_sha256:
            raise VectorIndexBindingError(
                f"tree node {record.metadata.get('node_id')} vector record is not "
                "source-bound"
            )
        record_id = stable_vector_record_id(
            record.value,
            record.metadata,
            modality=record.modality,
            image_text=record.image_text,
            strict_source_identity=True,
        )
        _hash_stream_item(
            projection_digest,
            {
                "record_id": record_id,
                "metadata": record.metadata,
            },
        )
        point_count += 1
    # Visual paths are hashed for identity here and again by the writer, but
    # the current legacy crop store is path-based rather than CAS-backed. A
    # concurrent mutation between hashing and model decode remains detectable
    # only after the fact; verified visual readiness requires immutable
    # snapshots/content-addressed asset bytes passed to the embedder.
    if point_count < 1:
        raise VectorIndexBindingError("refusing to publish an empty vector index")

    embedding_identity = dict(vector_config["embedding"])
    scope_filter = tree_vector_scope(tree, vdb_cfg)
    identity = {
        "schema": TREE_VECTOR_INDEX_SCHEMA,
        "source_sha256": source_sha256,
        "document_version_id": document_version_id,
        "index_revision_id": scope_filter["index_revision_id"],
        "tree_sha256": _tree_semantic_sha256(tree),
        "tree_artifact_sha256": _validated_tree_artifact_sha256(tree),
        "projection_sha256": projection_digest.hexdigest(),
        "node_count": len(tree.nodes),
        "point_count": point_count,
        "collection_name": collection_name,
        "scope_filter": scope_filter,
        "qdrant_endpoint_sha256": vector_config["qdrant_endpoint_sha256"],
        "vector_config_sha256": _sha256_json(vector_config),
        "embedding_identity_sha256": _sha256_json(embedding_identity),
        "store_seal_sha256": _store_revision_seal(
            scope_filter["index_revision_id"]
        )
        or "0" * 64,
    }
    return {
        **identity,
        "identity_sha256": _sha256_json(identity),
        "state": "content_attested",
    }


def tree_vector_index_attestation_is_ready(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
    *,
    tree_artifact_sha256: str | None = None,
) -> bool:
    """Validate a build attestation without rehashing every indexed asset.

    The caller must hold a shared vector-index lease. The owner-only atomic
    local record attests the expensive semantic/projection hashes computed during
    the successful build; the current saved tree bytes, source/config identity,
    attestation self-hash, and live scoped Qdrant point count are checked here.
    """

    if tree_artifact_sha256 is not None and not _is_lower_sha256(
        tree_artifact_sha256
    ):
        raise VectorIndexBindingError("tree artifact SHA-256 is invalid")
    source_sha256 = tree.meta_info.source_sha256
    document_version_id = tree.meta_info.document_version_id
    if not _is_lower_sha256(source_sha256):
        return False
    if (
        not isinstance(document_version_id, str)
        or not document_version_id
        or document_version_id != document_version_id.strip()
    ):
        return False
    attestation = read_vector_index_attestation(vdb_cfg.manifest_dir_name)
    if attestation is None:
        return False
    if set(attestation) != {
        *_VECTOR_ATTESTATION_IDENTITY_FIELDS,
        "identity_sha256",
        "state",
    }:
        return False
    identity = {
        field_name: attestation.get(field_name)
        for field_name in _VECTOR_ATTESTATION_IDENTITY_FIELDS
    }
    if attestation.get("state") != "content_attested":
        return False
    digest_fields = (
        "source_sha256",
        "tree_sha256",
        "tree_artifact_sha256",
        "projection_sha256",
        "qdrant_endpoint_sha256",
        "vector_config_sha256",
        "embedding_identity_sha256",
        "identity_sha256",
    )
    if any(not _is_lower_sha256(attestation.get(key)) for key in digest_fields):
        return False
    attested_document_version_id = attestation.get("document_version_id")
    if (
        not isinstance(attested_document_version_id, str)
        or not attested_document_version_id
        or attested_document_version_id != attested_document_version_id.strip()
    ):
        return False
    if attestation.get("identity_sha256") != _sha256_json(identity):
        return False
    vector_config = _vector_config_identity(vdb_cfg)
    embedding_identity = dict(vector_config["embedding"])
    scope_filter = tree_vector_scope(tree, vdb_cfg)
    expected_scalars = {
        "schema": TREE_VECTOR_INDEX_SCHEMA,
        "source_sha256": source_sha256,
        "document_version_id": document_version_id,
        "index_revision_id": scope_filter["index_revision_id"],
        "collection_name": vector_config["collection_name"],
        "scope_filter": scope_filter,
        "qdrant_endpoint_sha256": vector_config["qdrant_endpoint_sha256"],
        "vector_config_sha256": _sha256_json(vector_config),
        "embedding_identity_sha256": _sha256_json(embedding_identity),
    }
    if any(attestation.get(key) != value for key, value in expected_scalars.items()):
        return False
    # The binding truth: the vectors attest to the sealed store rows.  When
    # the caller still holds the serialized tree, its artifact hash and node
    # count are checked as well; a store-backed caller has neither.
    if tree_artifact_sha256 is not None:
        if attestation.get("tree_artifact_sha256") != tree_artifact_sha256:
            return False
        if attestation.get("node_count") != len(tree.nodes):
            return False
    live_seal = _store_revision_seal(scope_filter["index_revision_id"])
    if live_seal is None or attestation.get("store_seal_sha256") != live_seal:
        return False
    node_count = attestation.get("node_count")
    if isinstance(node_count, bool) or not isinstance(node_count, int):
        return False
    if node_count < 1:
        return False
    point_count = attestation.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        return False
    if point_count < 1:
        return False
    remote = _qdrant_state(vdb_cfg, scope_filter)
    return remote.exists and remote.point_count == point_count


def _compact_semantic_refs(
    raw_refs: object,
    *,
    field_name: str,
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """Project graphs/receipts and identify digests proven by a full DTO."""

    if raw_refs is None:
        return {}, set()
    if not isinstance(raw_refs, list):
        raise VectorIndexBindingError(f"{field_name} must be an array")
    if len(raw_refs) > MAX_VECTOR_SEMANTIC_REFS:
        raise VectorIndexBindingError(
            f"{field_name} exceeds the vector semantic reference limit"
        )

    by_id: dict[str, dict[str, str]] = {}
    verified_by_full_graph: set[str] = set()
    for position, raw_ref in enumerate(raw_refs):
        if not isinstance(raw_ref, dict):
            raise VectorIndexBindingError(
                f"{field_name}[{position}] must be an object"
            )
        substructure_id = raw_ref.get("substructure_id")
        semantic_sha256 = raw_ref.get("semantic_sha256")
        comparison_sha256 = raw_ref.get("comparison_sha256")
        if "elements" in raw_ref:
            # A full graph is the only local source from which v3 can prove
            # both digests.  Parse it even when the serialized receipt already
            # contains comparison_sha256 so a tampered value cannot bypass the
            # contract validator.
            from docarag.domain import SemanticSubstructure

            try:
                full_graph = SemanticSubstructure.from_dict(raw_ref)
            except (TypeError, ValueError) as exc:
                raise VectorIndexBindingError(
                    f"{field_name}[{position}] is not a valid semantic graph"
                ) from exc
            substructure_id = full_graph.substructure_id
            semantic_sha256 = full_graph.semantic_sha256
            comparison_sha256 = full_graph.comparison_sha256
            verified_by_full_graph.add(substructure_id)
        if (
            not isinstance(substructure_id, str)
            or not substructure_id
            or substructure_id != substructure_id.strip()
            or len(substructure_id) > 256
        ):
            raise VectorIndexBindingError(
                f"{field_name}[{position}].substructure_id is invalid"
            )
        if not _is_lower_sha256(semantic_sha256):
            raise VectorIndexBindingError(
                f"{field_name}[{position}].semantic_sha256 is invalid"
            )
        # A compact pre-comparison receipt may omit comparison_sha256 here; it
        # can be upgraded below only from a matching, verified full graph.
        if comparison_sha256 is not None and not _is_lower_sha256(
            comparison_sha256
        ):
            raise VectorIndexBindingError(
                f"{field_name}[{position}].comparison_sha256 is invalid"
            )
        compact = {
            "substructure_id": substructure_id,
            "semantic_sha256": semantic_sha256,
        }
        if comparison_sha256 is not None:
            compact["comparison_sha256"] = comparison_sha256
        previous = by_id.get(substructure_id)
        if previous is not None and previous != compact:
            raise VectorIndexBindingError(
                f"{field_name} maps one substructure ID to multiple digests"
            )
        by_id[substructure_id] = compact
    return by_id, verified_by_full_graph


def _vector_semantic_substructure_refs(node_meta) -> list[dict[str, str]]:
    """Return bounded receipts without copying graph elements into vector payloads.

    Current trees carry full graphs once in ``semantic_substructures`` and
    small lineage receipts in each source reference. Historical trees may have
    only one of those shapes, so either can seed this compact projection. When
    both exist their ID/digest sets must agree.
    """

    top_level, verified_ids = _compact_semantic_refs(
        getattr(node_meta, "semantic_substructures", None),
        field_name="semantic_substructures",
    )
    source_level: dict[str, dict[str, str]] = {}
    source_refs = getattr(node_meta, "source_evidence_refs", None)
    if source_refs is not None:
        if not isinstance(source_refs, list):
            raise VectorIndexBindingError("source_evidence_refs must be an array")
        if len(source_refs) > MAX_VECTOR_SOURCE_EVIDENCE_REFS:
            raise VectorIndexBindingError(
                "source_evidence_refs exceeds the vector metadata source limit"
            )
        for position, source_ref in enumerate(source_refs):
            if not isinstance(source_ref, dict):
                raise VectorIndexBindingError(
                    f"source_evidence_refs[{position}] must be an object"
                )
            raw_semantic_refs = source_ref.get("semantic_substructure_refs")
            # Compatibility with the short-lived pre-receipt tree shape: read
            # IDs/digests from duplicated graphs, but never serialize the graph.
            if raw_semantic_refs is None:
                raw_semantic_refs = source_ref.get("semantic_substructures")
            compact_refs, source_verified_ids = _compact_semantic_refs(
                raw_semantic_refs,
                field_name=(
                    f"source_evidence_refs[{position}].semantic_substructure_refs"
                ),
            )
            verified_ids.update(source_verified_ids)
            for substructure_id, compact in compact_refs.items():
                previous = source_level.get(substructure_id)
                if previous is not None and previous != compact:
                    raise VectorIndexBindingError(
                        "source evidence maps one semantic substructure ID to "
                        "multiple digests"
                    )
                source_level[substructure_id] = compact
                if len(source_level) > MAX_VECTOR_SEMANTIC_REFS:
                    raise VectorIndexBindingError(
                        "source evidence exceeds the vector semantic reference limit"
                    )

    for substructure_id, compact in source_level.items():
        if "comparison_sha256" in compact:
            continue
        top_compact = top_level.get(substructure_id)
        comparison_sha256 = (
            top_compact.get("comparison_sha256")
            if top_compact is not None
            else None
        )
        if not _is_lower_sha256(comparison_sha256):
            raise VectorIndexBindingError(
                "legacy semantic receipt without comparison_sha256 requires a "
                "matching full top-level graph"
            )
        compact["comparison_sha256"] = comparison_sha256

    if top_level and source_level and top_level != source_level:
        raise VectorIndexBindingError(
            "top-level semantic substructures differ from source evidence receipts"
        )
    selected = top_level or source_level
    if any("comparison_sha256" not in compact for compact in selected.values()):
        raise VectorIndexBindingError(
            "semantic vector receipts require comparison_sha256"
        )
    unverified_ids = set(selected).difference(verified_ids)
    if unverified_ids:
        raise VectorIndexBindingError(
            "v3 semantic vector receipts require a matching full semantic "
            "substructure graph; source-only compact digests are unverified"
        )
    return [selected[substructure_id] for substructure_id in sorted(selected)]


def _node_trace_metadata(tree: DocumentTree, node) -> dict:
    """Return bounded scalar source-trace metadata for Qdrant payloads."""

    node_meta = node.meta_info
    tree_meta = tree.meta_info
    node_type = node.type.value if isinstance(node.type, NodeType) else str(node.type)
    metadata = {
        "node_id": node.index_id,
        "node_type": node_type,
        "index_kind": "tree",
    }

    scalar_fields = {
        "pdf_id": node_meta.pdf_id,
        "document_version_id": (
            node_meta.document_version_id or tree_meta.document_version_id
        ),
        "page_id": node_meta.page_id,
        "page_idx": node_meta.page_idx,
        "source_sha256": node_meta.source_sha256 or tree_meta.source_sha256,
    }
    for key, value in scalar_fields.items():
        if value is not None:
            metadata[key] = value

    geometry = getattr(node_meta, "source_geometry", None)
    if isinstance(geometry, dict):
        for source_key, target_key in (
            ("coordinate_space", "coordinate_space"),
            ("coordinate_origin", "coordinate_origin"),
            ("rotation_degrees", "rotation_degrees"),
            ("page_width", "page_width"),
            ("page_height", "page_height"),
        ):
            value = geometry.get(source_key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                metadata[target_key] = value
        bbox = geometry.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in bbox
            )
        ):
            for key, value in zip(
                ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"), bbox, strict=True
            ):
                metadata[key] = value

    # A transformed node must trace through its derived identity rather than
    # claiming to be an unmodified canonical block.  Canonical and derived
    # identities are therefore intentionally mutually exclusive here.
    if node_meta.derived_evidence_id:
        metadata["derived_evidence_id"] = node_meta.derived_evidence_id
    elif node_meta.canonical_block_id:
        metadata["canonical_block_id"] = node_meta.canonical_block_id
        if node_meta.canonical_revision_id:
            metadata["canonical_revision_id"] = node_meta.canonical_revision_id
        if node_meta.observation_id:
            metadata["observation_id"] = node_meta.observation_id

    semantic_refs = _vector_semantic_substructure_refs(node_meta)
    if semantic_refs:
        encoded_refs = _canonical_json_bytes(semantic_refs)
        if len(encoded_refs) > MAX_VECTOR_SEMANTIC_REFS_BYTES:
            raise VectorIndexBindingError(
                "compact semantic substructure references exceed the vector "
                "metadata byte limit"
            )
        metadata["semantic_substructure_count"] = len(semantic_refs)
        metadata["semantic_substructure_refs_json"] = encoded_refs.decode("utf-8")
        metadata["semantic_substructure_refs_sha256"] = hashlib.sha256(
            encoded_refs
        ).hexdigest()

    return metadata


def _representation_metadata(metadata: dict, modality: Literal["text", "image"]) -> dict:
    """Bind a semantic tree node to one explicit retrieval representation."""

    return {**metadata, "modality": modality, "type": modality}


def _semantic_element_embedding_text(structure, element) -> str:
    """Create bounded searchable text while canonical JSON remains source truth."""

    # ``SemanticElement.content`` is deeply frozen; its DTO projection is the
    # JSON-ready form accepted by both the embedder and deterministic hashing.
    content = element.to_dict()["content"]
    if isinstance(content, str):
        rendered_content = content
    else:
        rendered_content = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    context = [
        structure.kind.value,
        element.kind.value,
        element.role or "",
    ]
    if element.row_index is not None:
        context.extend(
            (
                f"row {element.row_index}",
                f"column {element.column_index}",
                f"row span {element.row_span}",
                f"column span {element.column_span}",
            )
        )
    rendered = " | ".join(item for item in context if item)
    if rendered_content:
        rendered = f"{rendered} | {rendered_content}" if rendered else rendered_content
    return rendered[:MAX_SEMANTIC_ELEMENT_EMBED_TEXT_CHARS]


def _iter_semantic_element_records(node_meta, base_metadata: dict) -> Iterator[TreeIndexRecord]:
    """Yield independently retrievable cells, symbols, and visual graph elements."""

    raw_structures = getattr(node_meta, "semantic_substructures", None)
    if not raw_structures:
        return
    if not isinstance(raw_structures, list):
        raise VectorIndexBindingError("semantic_substructures must be an array")

    from docarag.domain import SemanticSubstructure

    for structure_position, raw_structure in enumerate(raw_structures):
        try:
            structure = SemanticSubstructure.from_dict(raw_structure)
        except (TypeError, ValueError) as exc:
            raise VectorIndexBindingError(
                f"semantic_substructures[{structure_position}] is invalid"
            ) from exc
        if not structure.is_source_bound:
            raise VectorIndexBindingError(
                "semantic vector elements require source-bound substructures"
            )
        for element in structure.elements:
            element_content = element.to_dict()["content"]
            metadata = {
                **base_metadata,
                "semantic_substructure_id": structure.substructure_id,
                "semantic_structure_kind": structure.kind.value,
                "semantic_structure_availability": structure.availability.value,
                "semantic_structure_producer_kind": structure.producer_kind.value,
                "semantic_sha256": structure.semantic_sha256,
                "semantic_comparison_sha256": structure.comparison_sha256,
                "semantic_element_id": element.element_id,
                "semantic_element_kind": element.kind.value,
                "semantic_element_local_key": element.local_key,
                "semantic_element_content_sha256": _sha256_json(element_content),
                "semantic_element_source_selector_count": len(element.source_selectors),
                "semantic_element_relation_count": sum(
                    1
                    for relation in structure.relations
                    if element.element_id
                    in (relation.source_element_id, relation.target_element_id)
                ),
            }
            for key, value in (
                ("semantic_element_role", element.role),
                ("semantic_element_row_index", element.row_index),
                ("semantic_element_column_index", element.column_index),
                ("semantic_element_row_span", element.row_span),
                ("semantic_element_column_span", element.column_span),
                ("semantic_element_confidence", element.confidence),
            ):
                if value is not None:
                    metadata[key] = value
            if element.geometry is not None:
                element_geometry = element.geometry.to_dict()
                bbox = element_geometry.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    for key, value in zip(
                        (
                            "semantic_bbox_x0",
                            "semantic_bbox_y0",
                            "semantic_bbox_x1",
                            "semantic_bbox_y1",
                        ),
                        bbox,
                        strict=True,
                    ):
                        metadata[key] = value
                metadata["semantic_coordinate_space"] = element_geometry[
                    "coordinate_space"
                ]
                metadata["semantic_coordinate_origin"] = element_geometry[
                    "coordinate_origin"
                ]
            yield TreeIndexRecord(
                "text",
                _semantic_element_embedding_text(structure, element),
                _representation_metadata(metadata, "text"),
            )


def iter_tree_index_records(tree: DocumentTree) -> Iterator[TreeIndexRecord]:
    """Yield text and visual representations in stable tree order."""

    for node in tree.nodes:
        if node == tree.root_node:
            continue

        node_type = node.type
        meta_data = _node_trace_metadata(tree, node)

        if node_type == NodeType.IMAGE:
            image_path = node.meta_info.img_path
            image_str = (node.meta_info.caption or "") + (
                node.meta_info.footnote or ""
            )
            yield TreeIndexRecord(
                "text", image_str, _representation_metadata(meta_data, "text")
            )
            if image_path and os.path.exists(image_path):
                yield TreeIndexRecord(
                    "image",
                    image_path,
                    _representation_metadata(meta_data, "image"),
                    image_str,
                )
        elif node_type == NodeType.TABLE:
            table_str = node.meta_info.content or ""
            table_body = node.meta_info.table_body
            if table_body:
                table_str += table_body
            yield TreeIndexRecord(
                "text", table_str, _representation_metadata(meta_data, "text")
            )

            table_img = node.meta_info.img_path
            if table_img and os.path.exists(table_img):
                yield TreeIndexRecord(
                    "image",
                    table_img,
                    _representation_metadata(meta_data, "image"),
                    table_str,
                )
        elif node_type == NodeType.EQUATION:
            text_content = node.meta_info.content
            if text_content:
                yield TreeIndexRecord(
                    "text",
                    text_content,
                    _representation_metadata(meta_data, "text"),
                )

            # Equation crops share the node metadata with their LaTeX text so
            # retrieval can hydrate either representation from source truth.
            equation_img = node.meta_info.img_path
            if equation_img and os.path.exists(equation_img):
                yield TreeIndexRecord(
                    "image",
                    equation_img,
                    _representation_metadata(meta_data, "image"),
                    text_content or "",
                )
        elif node_type == NodeType.TEXT or node_type == NodeType.TITLE:
            text_content = node.meta_info.content
            if text_content:
                yield TreeIndexRecord(
                    "text",
                    text_content,
                    _representation_metadata(meta_data, "text"),
                )

        yield from _iter_semantic_element_records(node.meta_info, meta_data)


def process_tree_nodes(tree: DocumentTree) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Materialize tree records for compatibility with existing callers.

    New indexing code should consume :func:`iter_tree_index_records` directly.
    """

    text_list = []
    text_meta_data = []
    image_list = []
    image_meta_data = []
    image_str_list = []
    for record in iter_tree_index_records(tree):
        if record.modality == "text":
            text_list.append(record.value)
            text_meta_data.append(record.metadata)
        else:
            image_list.append(record.value)
            image_meta_data.append(record.metadata)
            image_str_list.append(record.image_text or "")

    text_dict = {"text": text_list, "meta": text_meta_data}
    image_dict = {
        "image": image_list,
        "meta": image_meta_data,
        "image_str": image_str_list,
    }
    return text_dict, image_dict


def _write_tree_vector_index(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
    *,
    expected_point_count: int,
    batch_size: int | None = None,
) -> None:
    """Reset and write one payload scope with bounded microbatches."""

    effective_batch_size = (
        batch_size
        if batch_size is not None
        else getattr(vdb_cfg, "index_batch_size", DEFAULT_TREE_INDEX_BATCH_SIZE)
    )
    if (
        isinstance(effective_batch_size, bool)
        or not isinstance(effective_batch_size, int)
        or effective_batch_size < 1
    ):
        raise ValueError("batch_size must be at least 1")

    embedder = None
    vdb = None
    active_error = False
    try:
        if vdb_cfg.mm_embedding:
            embedder = GmeEmbeddingProvider(
                model_name=vdb_cfg.embedding_config.model_name,
                device=vdb_cfg.embedding_config.device,
            )
            log.info("Using GME multi-modal embedding model for vector database.")
        else:
            embedder = TextEmbeddingProvider(
                model_name=vdb_cfg.embedding_config.model_name,
                device=vdb_cfg.embedding_config.device,
                backend=vdb_cfg.embedding_config.backend,
                api_base=vdb_cfg.embedding_config.api_base,
                api_key=getattr(vdb_cfg.embedding_config, "api_key", None),
                max_length=vdb_cfg.embedding_config.max_length,
            )
            log.info("Using text embedding model for vector database.")

        vdb = VectorStore(
            embedding_model=embedder,
            qdrant_url=vdb_cfg.qdrant_url,
            collection_name=vdb_cfg.collection_name,
            api_key_env=vdb_cfg.api_key_env,
            timeout_seconds=vdb_cfg.timeout_seconds,
            scope_filter=tree_vector_scope(tree, vdb_cfg),
        )
        # Deterministic upserts alone do not delete records removed from a
        # newer tree. A payload-scope reset reconciles stale records without
        # disturbing other documents, revisions, or index kinds in the collection.
        vdb.reset()

        text_batch: list[str] = []
        text_metadata_batch: list[dict] = []
        image_batch: list[str] = []
        image_metadata_batch: list[dict] = []
        image_text_batch: list[str] = []

        def flush_texts() -> None:
            if not text_batch:
                return
            # Pass snapshots because the reusable buffers are cleared after a
            # synchronous write.  This also keeps adapter/mock implementations
            # from retaining a subsequently mutated list.
            vdb.add_texts(
                texts=list(text_batch),
                metadatas=list(text_metadata_batch),
                strict_source_identity=True,
            )
            text_batch.clear()
            text_metadata_batch.clear()

        def flush_images() -> None:
            if not image_batch:
                return
            vdb.add_images(
                image_paths=list(image_batch),
                metadatas=list(image_metadata_batch),
                image_str=list(image_text_batch),
                strict_source_identity=True,
            )
            image_batch.clear()
            image_metadata_batch.clear()
            image_text_batch.clear()

        for record in iter_tree_index_records(tree):
            if record.modality == "text":
                text_batch.append(record.value)
                text_metadata_batch.append(record.metadata)
                if len(text_batch) >= effective_batch_size:
                    flush_texts()
            elif vdb_cfg.mm_embedding:
                image_batch.append(record.value)
                image_metadata_batch.append(record.metadata)
                image_text_batch.append(record.image_text or "")
                if len(image_batch) >= effective_batch_size:
                    flush_images()

        flush_texts()
        if vdb_cfg.mm_embedding:
            flush_images()
            log.info("Images added to vector database successfully.")

        observed_point_count = vdb.count()
        if observed_point_count != expected_point_count:
            raise VectorIndexBindingError(
                "tree vector scope contains "
                f"{observed_point_count} points; expected {expected_point_count}"
            )

        log.info("Vector database index built successfully.")
    except BaseException:
        active_error = True
        raise
    finally:
        cleanup_errors = []
        if vdb is not None:
            try:
                vdb.close()
            except Exception as exc:  # cleanup must continue to the model
                cleanup_errors.append(exc)
                log.exception("Failed to close tree vector store")
        if embedder is not None:
            try:
                embedder.close()
            except Exception as exc:
                cleanup_errors.append(exc)
                log.exception("Failed to close tree embedding provider")
        if cleanup_errors and not active_error:
            raise cleanup_errors[0]


def _prepare_tree_attestation_directory(directory: Path) -> None:
    """Prepare a safe run-local directory that never contains vector payloads."""

    if directory.is_symlink():
        raise VectorIndexBindingError(
            f"vector attestation directory may not be a symbolic link: {directory}"
        )
    if directory.exists() and not directory.is_dir():
        remove_artifact_path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
        return
    unexpected = [
        entry.name
        for entry in directory.iterdir()
        if entry.name != "docarag-qdrant-index.ready.json"
    ]
    if unexpected:
        raise VectorIndexBindingError(
            "vector attestation directory contains unexpected local/legacy "
            "artifacts: " + ", ".join(sorted(unexpected))
        )


def tree_vector_index_is_ready(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
) -> bool:
    """Return whether durable state exactly matches this tree and config."""

    directory = Path(vdb_cfg.manifest_dir_name)
    with vector_index_lock(directory, exclusive=False):
        if directory.is_symlink():
            return False
        expected = expected_tree_vector_index_attestation(tree, vdb_cfg)
        if read_vector_index_attestation(directory) != expected:
            return False
        remote = _qdrant_state(vdb_cfg, tree_vector_scope(tree, vdb_cfg))
        return vector_index_attestation_matches(
            directory,
            expected,
            observed_point_count=(remote.point_count if remote.exists else None),
        )


def build_vdb_index(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
    *,
    batch_size: int | None = None,
    reuse_if_ready: bool = False,
) -> bool:
    """Ensure or rebuild a source-bound BookRAG tree vector collection.

    Direct calls rebuild the selected payload scope by default. Controllers may set
    ``reuse_if_ready``; reuse then occurs only under the same process lock and
    only when the atomic attestation and live scoped point count match the exact
    current tree/source/vector/embedding identity.

    Returns ``True`` when a rebuild was performed and ``False`` when an exact
    ready index was reused.
    """

    if not isinstance(reuse_if_ready, bool):
        raise TypeError("reuse_if_ready must be a boolean")
    directory = Path(vdb_cfg.manifest_dir_name)
    with vector_index_lock(directory, exclusive=True):
        if directory.is_symlink():
            raise VectorIndexBindingError(
                f"vector attestation directory may not be a symbolic link: {directory}"
            )
        if reuse_if_ready:
            artifact_sha256 = _validated_tree_artifact_sha256(tree)
            if artifact_sha256 is not None and tree_vector_index_attestation_is_ready(
                tree,
                vdb_cfg,
                tree_artifact_sha256=artifact_sha256,
            ):
                log.info(
                    "Validated source-bound Qdrant index; skipping rebuild",
                )
                return False
        expected = expected_tree_vector_index_attestation(tree, vdb_cfg)
        if reuse_if_ready and read_vector_index_attestation(directory) == expected:
            remote = _qdrant_state(vdb_cfg, tree_vector_scope(tree, vdb_cfg))
            if vector_index_attestation_matches(
                directory,
                expected,
                observed_point_count=(remote.point_count if remote.exists else None),
            ):
                log.info("Validated source-bound Qdrant index; skipping rebuild")
                return False

        if directory.is_dir():
            invalidate_vector_index_attestation(directory)
        _prepare_tree_attestation_directory(directory)
        invalidate_vector_index_attestation(directory)
        try:
            _write_tree_vector_index(
                tree,
                vdb_cfg,
                expected_point_count=int(expected["point_count"]),
                batch_size=batch_size,
            )
            # The remote client and embedder close before local attestation is
            # published. Query readiness still requires a fresh live count.
            publish_vector_index_attestation(directory, expected)
        except BaseException:
            invalidate_vector_index_attestation(directory)
            raise
        return True


def get_input_text(cfg: SystemConfig) -> str:
    pdf_path = cfg.pdf_path
    file_name = str(Path(pdf_path).stem)
    md_file = os.path.join(cfg.save_path, "vlm", f"{file_name}.md")
    with open(md_file, "r", encoding="utf-8") as f:
        input_text = f.read()
    return input_text


def get_all_chunks(cfg: SystemConfig):
    corpus_text = get_input_text(cfg)
    chunks = TextProcessor.split_text_into_chunks(text=corpus_text, max_length=500)

    index_type = cfg.index_type
    if index_type == "vanilla" or index_type == "bm25":
        meta_datas = [{"source": "document", "chunk_id": i} for i in range(len(chunks))]
        return chunks, meta_datas
    elif index_type == "raptor":
        llm = LLM(cfg.llm)
        embed_cfg = cfg.vdb.embedding_config
        embedder = TextEmbeddingProvider(
            model_name=embed_cfg.model_name,
            device=embed_cfg.device,
            backend=embed_cfg.backend,
            api_base=embed_cfg.api_base,
            api_key=getattr(embed_cfg, "api_key", None),
            max_length=embed_cfg.max_length,
        )

        all_tree_text, all_meta_data = raptor_tree(chunks, embedder=embedder, llm=llm)
        embedder.close()
        return all_tree_text, all_meta_data


def _system_config_vector_scope(cfg: SystemConfig, *, index_kind: str) -> dict[str, str]:
    source = cfg.source
    if source is None or not _is_lower_sha256(source.sha256):
        raise VectorIndexBindingError(
            "Qdrant indexing requires cfg.source.sha256 bound to immutable input bytes"
        )
    if not cfg.save_path:
        raise VectorIndexBindingError("Qdrant indexing requires cfg.save_path")
    document_version_id = (
        getattr(source, "document_version_id", None)
        or getattr(cfg.vdb, "document_version_id", None)
        or Path(cfg.save_path).expanduser().resolve().name
    )
    if not isinstance(document_version_id, str) or not document_version_id.strip():
        raise VectorIndexBindingError(
            "Qdrant indexing requires the stable source document_version_id"
        )
    if document_version_id != document_version_id.strip():
        raise VectorIndexBindingError("document_version_id must be trimmed")
    configured_document_version = getattr(cfg.vdb, "document_version_id", None)
    if (
        configured_document_version is not None
        and configured_document_version != document_version_id
    ):
        raise VectorIndexBindingError(
            "source document_version_id differs from the configured PostgreSQL scope"
        )
    return {
        **_validated_tenant_scope(cfg.vdb),
        "document_version_id": document_version_id,
        "index_revision_id": _validated_index_revision_id(
            cfg.vdb, run_directory=cfg.save_path
        ),
        "source_sha256": source.sha256,
        "index_kind": index_kind,
    }


def _chunk_vector_attestation(
    cfg: SystemConfig,
    chunks: list[str],
    metadatas: list[dict],
    scope_filter: dict[str, str],
) -> dict[str, object]:
    projection = {
        "schema": "docarag.bookrag-chunk-vector-projection.v2",
        "chunks": [
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
            for text, metadata in zip(chunks, metadatas, strict=True)
        ],
    }
    vector_config = _vector_config_identity(cfg.vdb)
    identity = {
        "schema": "docarag.bookrag-chunk-qdrant-index.v2",
        "source_sha256": scope_filter["source_sha256"],
        "document_version_id": scope_filter["document_version_id"],
        "index_revision_id": scope_filter["index_revision_id"],
        "projection_sha256": _sha256_json(projection),
        "point_count": len(chunks),
        "collection_name": cfg.vdb.collection_name,
        "scope_filter": scope_filter,
        "qdrant_endpoint_sha256": vector_config["qdrant_endpoint_sha256"],
        "vector_config_sha256": _sha256_json(vector_config),
        "embedding_identity_sha256": _sha256_json(vector_config["embedding"]),
    }
    return {
        **identity,
        "identity_sha256": _sha256_json(identity),
        "state": "content_attested",
    }


def build_other_vdb_index(cfg: SystemConfig):
    local_dir = resolve_run_artifact_path(
        cfg.save_path,
        cfg.vdb.manifest_dir_name,
        label="vdb.manifest_dir_name",
    )
    cfg.vdb.manifest_dir_name = str(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    all_chunks, meta_datas = get_all_chunks(cfg)

    if cfg.index_type == "bm25":
        save_path = os.path.join(local_dir, "bm25_index.pkl")
        if not cfg.vdb.force_rebuild and valid_lexical_index(local_dir):
            log.info("Validated existing BM25 lexical index; skipping rebuild")
            return
        bm25 = BM25(all_chunks)
        bm25.initialize()
        # test
        query = "quick"
        results = bm25.search(query, top_k=2)
        log.info(f"BM25 search results for test query {query}: {results}")

        bm25.save(save_path)
        log.info(f"BM25 index saved to {save_path}")
    else:
        vdb_config = cfg.vdb
        scope_filter = _system_config_vector_scope(
            cfg,
            index_kind=str(cfg.index_type),
        )
        scoped_metadata = []
        for index, metadata in enumerate(meta_datas):
            scoped_metadata.append(
                {
                    **metadata,
                    **scope_filter,
                    "node_id": f"chunk-{index}",
                }
            )
        expected = _chunk_vector_attestation(
            cfg,
            all_chunks,
            scoped_metadata,
            scope_filter,
        )
        with vector_index_lock(local_dir, exclusive=True):
            _prepare_tree_attestation_directory(local_dir)
            if not cfg.vdb.force_rebuild and read_vector_index_attestation(local_dir) == expected:
                remote = _qdrant_state(vdb_config, scope_filter)
                if vector_index_attestation_matches(
                    local_dir,
                    expected,
                    observed_point_count=(remote.point_count if remote.exists else None),
                ):
                    log.info("Validated existing scoped Qdrant chunk index; skipping rebuild")
                    return
            invalidate_vector_index_attestation(local_dir)
            embedder = TextEmbeddingProvider(
                model_name=vdb_config.embedding_config.model_name,
                device=vdb_config.embedding_config.device,
                backend=vdb_config.embedding_config.backend,
                api_base=vdb_config.embedding_config.api_base,
                api_key=getattr(vdb_config.embedding_config, "api_key", None),
                max_length=vdb_config.embedding_config.max_length,
            )
            vdb = None
            try:
                vdb = VectorStore(
                    embedding_model=embedder,
                    qdrant_url=vdb_config.qdrant_url,
                    collection_name=vdb_config.collection_name,
                    api_key_env=vdb_config.api_key_env,
                    timeout_seconds=vdb_config.timeout_seconds,
                    scope_filter=scope_filter,
                )
                vdb.reset()
                vdb.add_texts(
                    texts=all_chunks,
                    metadatas=scoped_metadata,
                    strict_source_identity=True,
                )
                observed = vdb.count()
                if observed != len(all_chunks):
                    raise VectorIndexBindingError(
                        f"chunk vector scope contains {observed} points; "
                        f"expected {len(all_chunks)}"
                    )
            except BaseException:
                invalidate_vector_index_attestation(local_dir)
                raise
            finally:
                if vdb is not None:
                    vdb.close()
                embedder.close()
            publish_vector_index_attestation(local_dir, expected)
            log.info("Scoped Qdrant chunk index built successfully.")


def load_pdf_lists_from_dir(save_dir):
    res_list = []
    pdf_list_json_files = os.listdir(save_dir)
    for pdf_list_json_file in pdf_list_json_files:
        if not pdf_list_json_file.endswith(".json"):
            continue
        pdf_list_path = os.path.join(save_dir, pdf_list_json_file)
        with open(pdf_list_path, "r", encoding="utf-8") as f:
            pdf_list = json.load(f)
        tmp_dict = {"pdf_list": pdf_list, "pdf_list_path": pdf_list_path}
        res_list.append(tmp_dict)

    return res_list


def compute_mm_embedding(cfg: SystemConfig, tree_index: DocumentTree):
    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    text_only_group_data = []
    image_only_group_data = []
    fused_group_data = []

    all_node_data = []
    all_embeddings_values = []

    for i, node in enumerate(tree_index.nodes):
        if node == tree_index.root_node:
            continue

        node_id = node.index_id
        node_type = node.type
        content = node.meta_info.content
        img_path = (
            node.meta_info.img_path
            if node_type in (NodeType.IMAGE, NodeType.TABLE, NodeType.EQUATION)
            else None
        )

        node_info = {
            "node_id": node_id,
            "node_type": node_type,
            "content": content,
            "img_path": img_path,
            "embedding_idx": None,
        }
        current_node_data_idx = len(all_node_data)
        all_node_data.append(node_info)

        if content and img_path:
            fused_group_data.append(
                {
                    "original_node_data_idx": current_node_data_idx,
                    "text": content,
                    "image": img_path,
                }
            )
        elif content:
            text_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "text": content}
            )
        elif img_path:
            image_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "image": img_path}
            )

    if text_only_group_data:
        texts = [item["text"] for item in text_only_group_data]
        text_embeddings = embedder.embed_texts(texts)
        for i, item in enumerate(text_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = text_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if image_only_group_data:
        images = [item["image"] for item in image_only_group_data]
        image_embeddings = embedder.embed_images(images)
        for i, item in enumerate(image_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = image_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if fused_group_data:
        texts = [item["text"] for item in fused_group_data]
        images = [item["image"] for item in fused_group_data]
        fused_embeddings = embedder.embed_fused(texts=texts, images=images)
        for i, item in enumerate(fused_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = fused_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx
    embedder.clear_cache()

    # --- 保存所有节点元数据到JSON文件 ---
    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)  # 确保保存路径存在
    metadata_filepath = os.path.join(save_dir, "mm_node_metadata.json")
    embeddings_filepath = os.path.join(save_dir, "mm_embeddings.npy")

    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(all_node_data, f, ensure_ascii=False, indent=4)

    if all_embeddings_values:
        final_embeddings_array = np.array(all_embeddings_values)
        np.save(embeddings_filepath, final_embeddings_array)
        log.info(f"All embeddings saved to: {embeddings_filepath}")
    else:
        log.warning("No embeddings were computed, .npy file not saved.")

    log.info(f"All node metadata saved to: {metadata_filepath}")


def compute_mm_embedding_question(cfg: SystemConfig, group: pd.DataFrame):
    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    group_dedup = group.drop_duplicates(subset=["question"], keep="first")
    questions = group_dedup["question"].tolist()
    RERANKER_INSTRUCTION = "Retrieve the most relevant document for the given query."

    # add instruction for gme model
    question_embeddings_raw = embedder.embed_texts(
        questions, instruction=RERANKER_INSTRUCTION
    )

    all_question_embeddings = []
    question_embedding_indices = []

    for i, embedding in enumerate(question_embeddings_raw):
        all_question_embeddings.append(embedding)
        question_embedding_indices.append(len(all_question_embeddings) - 1)

    group_dedup["question_embedding_idx"] = question_embedding_indices

    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)

    question_metadata_filepath = os.path.join(save_dir, "mm_question_metadata.json")
    question_embeddings_filepath = os.path.join(save_dir, "mm_question_embeddings.npy")

    group_dedup.to_json(
        question_metadata_filepath, orient="records", force_ascii=False, indent=4
    )

    if all_question_embeddings:
        final_question_embeddings_array = np.array(all_question_embeddings)
        np.save(question_embeddings_filepath, final_question_embeddings_array)
        log.info(f"All question embeddings saved to: {question_embeddings_filepath}")
    else:
        log.warning("No question embeddings were computed, .npy file not saved.")

    log.info(f"All question metadata saved to: {question_metadata_filepath}")


if __name__ == "__main__":
    # tmp_tree_path = f"{save_path}/sftree.pkl"
    # tree_index = DocumentTree.load_from_file(tmp_tree_path)
    # print(f"Loaded tree index from: {tmp_tree_path}")
    # vector_store = build_vdb_from_tree(tree_index)
    print("test")
