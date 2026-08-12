import os
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterator, List, Literal, Mapping, Tuple
from pathlib import Path
import hashlib
import stat
import pandas as pd
import numpy as np

from Core.provider.embedding import (
    TextEmbeddingProvider,
    MultimodalEmbeddingProvider,
)
from Core.provider.llm import LLM
from Core.pipelines.entity_facets import BM25_B, BM25_K1, bm25_terms, lexical_terms
from Core.provider.vdb import (
    TREE_VECTOR_RECORD_ID_SCHEMA,
    VECTOR_RECORD_ID_SCHEMA,
    VectorSchema,
    VectorStore,
    _TREE_DENSE_VECTORS,
    _TREE_IMAGE_VECTOR,
    _TREE_SPARSE_VECTOR,
    qdrant_point_id,
    stable_vector_record_id,
    tree_point_record_id,
    tree_vector_identity,
)
from Core.Index.Tree import DocumentTree, NodeType
from Core.pipelines.node_context import context_contribution, node_content
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
#: v7 was one point per node carrying named vectors, where v6 was one point per
#: representation carrying a single unnamed one. v8 gives a figure, table,
#: listing and list the ``context`` vector only titles and equations had, so a
#: figure is reachable through the section it illustrates and not only through
#: its own words. Nothing dispatches on these -- they are domain separators
#: inside the digests, so bumping them is what makes every attestation written
#: under the old shape fail its exact comparison and forces a rebuild rather
#: than silently reading points that are not there.
#:
#: The bump is load-bearing here in a way it was not for v7. Readiness compares
#: the *configured* vector names, which v8 does not change; it never recomputes
#: the projection to compare against the attested ``projection_sha256``. So an
#: index whose points are missing every new context vector would otherwise pass
#: as ready, and the change would reach no document.
#:
#: v9 stores BM25 term weights where v8 stored raw counts. Qdrant's IDF
#: modifier is only the idf half of BM25, so v8's sparse leg was a tf-idf dot
#: product and ranked by length: the longest node in the document came first
#: for every query. The stored values change, the text they come from does not,
#: so nothing in the record identity moves and this bump is the only thing that
#: makes an existing index rebuild.
TREE_VECTOR_INDEX_SCHEMA = "docarag.bookrag-tree-qdrant-index.v9"
TREE_VECTOR_PROJECTION_SCHEMA = "docarag.bookrag-tree-vector-projection.v9"
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
    "vector_count",
    "sparse_vector_count",
    "lexical_average_length",
    "lexical_k1",
    "lexical_b",
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


@dataclass(frozen=True)
class TreePointRecord:
    """Everything one Qdrant point holds: a node, or one semantic element.

    A node's readings are named rather than separate -- ``text`` is its own
    words, ``context`` the words around or under it, ``image`` its crop -- and a
    point carries whichever of them that node has. The names are assigned by
    role and never by availability: an equation with no text of its own carries
    ``context`` alone rather than promoting its neighbourhood into ``text``,
    because the two are scored separately on purpose.
    """

    kind: Literal["node", "semantic_element"]
    metadata: dict
    vectors: Mapping[str, str]
    image_text: str | None = None

    @property
    def lexical_source(self) -> str:
        """The text the sparse vector's terms are counted from.

        Every text this point holds, joined. The dense split exists because
        blending *embeddings* measured worse; term frequencies are a bag of
        words and do not dilute that way, so lexical search gets the lot.
        """

        return "\n".join(
            self.vectors[name]
            for name in _TREE_DENSE_VECTORS
            if name != _TREE_IMAGE_VECTOR and self.vectors.get(name)
        )


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
        "record_id_schema": TREE_VECTOR_RECORD_ID_SCHEMA,
        "backend": "qdrant",
        "collection_name": collection_name,
        "qdrant_endpoint_sha256": endpoint_sha256,
        "distance_metric": "cosine",
        # The shape of the collection is part of what an attestation attests:
        # the same tree written under a different set of vector names is a
        # different index, and this is what says so.
        "vector_names": list(_TREE_DENSE_VECTORS),
        "sparse_vector": _TREE_SPARSE_VECTOR,
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
    *,
    frozen_average_length: float | None = None,
) -> dict[str, object]:
    """Compute the exact, secret-free identity required for index reuse.

    ``frozen_average_length`` is the BM25 normalizer an incremental reconcile
    kept rather than recomputed. ``lexical_average_length`` states the value
    the stored weights were computed with, not the corpus mean -- those are the
    same number after a full rebuild and drift apart across edits -- so a
    reconcile that re-weighted only the points it touched has to record the
    value those weights actually used, or the attestation describes an index
    that does not exist.
    """

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
    vector_count = 0
    sparse_vector_count = 0
    lexical_token_total = 0.0
    for point in iter_tree_point_records(tree):
        vectors = {
            name: value
            for name, value in point.vectors.items()
            if name != _TREE_IMAGE_VECTOR or vdb_cfg.mm_embedding
        }
        if not vectors:
            continue
        if point.metadata.get("document_version_id") != document_version_id:
            raise VectorIndexBindingError(
                f"tree node {point.metadata.get('node_id')} vector record has "
                "another document version"
            )
        if point.metadata.get("source_sha256") != source_sha256:
            raise VectorIndexBindingError(
                f"tree node {point.metadata.get('node_id')} vector record is not "
                "source-bound"
            )
        representations = {
            name: tree_vector_identity(
                name,
                value,
                image_text=point.image_text,
                strict_source_identity=True,
            )
            for name, value in vectors.items()
        }
        record_id = tree_point_record_id(
            kind=point.kind,
            metadata=point.metadata,
            representations=representations,
            lexical_source=point.lexical_source,
            strict_source_identity=True,
        )
        _hash_stream_item(
            projection_digest,
            {
                "record_id": record_id,
                "record_kind": point.kind,
                "metadata": point.metadata,
                # Redundant with the record ID, which already contains every
                # digest -- carried anyway so the attestation says which vectors
                # a point holds without having to recompute an ID to find out.
                "vectors": {
                    name: dict(digest) for name, digest in sorted(representations.items())
                },
            },
        )
        point_count += 1
        vector_count += len(vectors)
        if point.lexical_source:
            sparse_vector_count += 1
            # BM25 normalizes each document against the mean, so the mean has
            # to be known before the first point is written. This pass already
            # visits every point to count them; counting their tokens here is
            # what lets the writer stream in batches without a second walk.
            lexical_token_total += sum(lexical_terms(point.lexical_source).values())
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
        # point_count alone can no longer tell "every node indexed with all its
        # vectors" from "every node indexed with one": a bug that dropped every
        # crop would leave it correct. These two catch that. Neither is remotely
        # verifiable -- Qdrant's count API counts points, not vectors -- so they
        # guard the projection against the writer and nothing further.
        "vector_count": vector_count,
        "sparse_vector_count": sparse_vector_count,
        # The BM25 length normalizer, and the constants it was applied with.
        # Recorded because the stored sparse values cannot be reproduced from
        # the node text alone -- they depend on this mean -- so an attestation
        # without it could not say what the index actually holds. Rounded so
        # the digest does not turn on a float's last bits.
        "lexical_average_length": (
            round(frozen_average_length, 6)
            if frozen_average_length is not None
            else (
                round(lexical_token_total / sparse_vector_count, 6)
                if sparse_vector_count
                else 0.0
            )
        ),
        "lexical_k1": BM25_K1,
        "lexical_b": BM25_B,
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


#: How many siblings on each side of an equation carry its explanation.
#: Measured wider windows dilute: a two-sided span of 2 scored below a span of
#: 1 on every equation and question tried.
# Node types whose evidence is a crop, and those whose evidence is prose.
# A chart and a diagram index exactly like an image -- caption text plus the
# crop itself -- because what makes them findable is the same pair. Code,
# algorithms, abstracts and references index like prose. Keeping these as sets
# means adding a node type is one edit here rather than four scattered
# equality checks that silently exclude it from the index.
_VISUAL_NODE_TYPES = frozenset(
    {NodeType.IMAGE, NodeType.CHART, NodeType.DIAGRAM}
)
_TEXT_NODE_TYPES = frozenset(
    {
        NodeType.TEXT,
        NodeType.TITLE,
        NodeType.CODE,
        NodeType.ALGORITHM,
        NodeType.ABSTRACT,
        NodeType.REFERENCE,
    }
)
# Reachable node types deliberately kept out of the index. The dispatch below is
# an if/elif chain with no final else, so a type absent from every branch is
# simply never indexed -- no error, no log, and nothing downstream can tell the
# difference between "not in this document" and "silently dropped". Declaring the
# intentional omissions lets a test assert that every reachable type is one or the
# other, rather than leaving the chain to be read carefully by whoever adds the
# next type.
_NOT_INDEXED_NODE_TYPES = frozenset(
    {
        # Page furniture: a running header, a footer, a page number, marginal
        # aside or discarded block. Embedding it puts "Scanned with CamScanner"
        # into retrieval competing with the document's own prose, which is what
        # giving furniture its own node type was for.
        NodeType.HEADER,
        NodeType.FOOTER,
        NodeType.ASIDE,
        NodeType.DISCARDED,
    }
)

# Prose types whose "text" is not prose. A listing is indexed like a table --
# its own words in front of its syntax -- while the rest of _TEXT_NODE_TYPES
# index as themselves, because a paragraph, an abstract or a reference already
# is the sentence a question resembles. These stay inside _TEXT_NODE_TYPES so
# the "every reachable type is declared" check still sees them.
_LISTING_NODE_TYPES = frozenset({NodeType.CODE, NodeType.ALGORITHM})

# Types that may carry a materialized crop alongside their text.
_CROP_BEARING_NODE_TYPES = _VISUAL_NODE_TYPES | {
    NodeType.TABLE,
    NodeType.EQUATION,
    NodeType.LIST,
}

#: Types whose ``context`` slot is free to carry the section they sit under.
#: TITLE fills it with its own section summary and EQUATION with its
#: neighbourhood, so both are absent here. So is TEXT: a paragraph already is
#: the section's prose, and giving 12,000 of them a near-identical section
#: vector would crowd out the leg they share with the sections themselves.
_SECTION_CONTEXT_NODE_TYPES = (
    _VISUAL_NODE_TYPES | _LISTING_NODE_TYPES | {NodeType.TABLE, NodeType.LIST}
)

_EQUATION_CONTEXT_NEIGHBOURS = 1

#: Bounds one equation's context so a long sibling cannot crowd the embedder.
_EQUATION_CONTEXT_CHARS = 1500


_node_content = node_content
_context_contribution = context_contribution


def _table_text(node) -> str:
    """What a table says, in words and in cells.

    The caption names it and the grid holds the values, but neither is a
    sentence. The summary is, and it is the part a question resembles.
    """

    summary = str(getattr(node, "summary", "") or "").strip()
    caption = node_content(node)
    body = str(getattr(getattr(node, "meta_info", None), "table_body", "") or "").strip()
    return "\n".join(part for part in (summary, caption, body) if part)


def _listing_text(node) -> str:
    """What a listing does, in words, above what it says in syntax.

    A listing embeds as syntax for the same reason a grid embeds as markup:
    "! comment (optional)" and "<zerntype> <numterms> <datatype>" are the
    shape of a file format, not a sentence about one, and a question is a
    sentence. Every one of this document's 313 code nodes had a summary
    naming the subject -- "the INT Zernike file format consists of optional
    comment lines followed by the Zernike coefficient data" -- and indexed
    only the skeleton.

    Summary first, matching ``_table_text``: it is the part a question
    resembles and the part worth keeping if the window truncates.
    """

    summary = str(getattr(node, "summary", "") or "").strip()
    listing = _node_content(node)
    if summary and listing:
        return f"{summary}\n{listing}"
    return summary or listing


def _visual_text(node) -> str:
    """What a figure is about, in whatever words are attached to it.

    A caption is where a figure's subject is normally written, and reading the
    caption alone is enough for a producer that fills it. MinerU fills neither
    caption nor footnote: the vision model's account of the pixels lands in
    ``content`` and the section's own words land in ``summary``. So caption
    plus footnote indexed 1866 of one document's 2002 figures as the empty
    string -- every one of them reachable only through its pixels, and a
    figure whose pixels do not spell out its subject was reachable not at all.

    That is how "the image of Zernike Surface" returned three screenshots of
    Zemax coefficient windows -- the embedder could read "Zernike" rendered
    inside those screenshots -- while the Zernike Surface section's own figure,
    an abstract shaded form, was never a candidate. Its summary names the
    subject in the first line; nothing was reading it.

    The summary leads because it is the sentence a question resembles, the
    same order ``_table_text`` uses, and it is the part worth keeping if the
    embedder's window truncates.
    """

    summary = str(getattr(node, "summary", "") or "").strip()
    described = _context_contribution(node)
    if summary and described:
        return f"{summary}\n{described}"
    return summary or described


def _section_context_text(node) -> str:
    """The section a figure sits under, as a vector of its own.

    A figure's own words describe the figure. The word a question uses is in
    the heading above it, and nothing put the two together: asking for "the
    image of Wolter Surface" ranked that figure outside a top 200 whose first
    hit was its own section, because the figure's prose says "a 3D rendered,
    smooth, matte gray tapered cylinder" and never says Wolter.

    Blending the heading into ``text`` would answer that question and lose the
    other one. Measured on the same figure: the query naming the subject scored
    0.50 against today's ``text`` and 0.85 against heading-plus-section, while
    the query describing the picture -- "a tapered cylinder on a purple
    gradient background" -- scored 0.83 against the figure's own prose and 0.62
    once the heading was mixed into it. So the section is its own vector, for
    the reason a title keeps its heading apart from its section summary, and
    retrieval keeps whichever leg hits.

    This also reaches a figure whose own summary is wrong. The Zernike figure's
    summary argues the picture is *not* a Zernike surface, so no arrangement of
    its own words retrieves it; its section still scores 0.83.

    The direct parent, matching ``GBCRAG._with_section_titles``, which measured
    2956 of 2957 figures, equations and listings sitting straight under their
    heading on this corpus.
    """

    parent = getattr(node, "parent", None)
    if parent is None or getattr(parent, "type", None) != NodeType.TITLE:
        return ""
    heading = str(_node_content(parent) or "").strip()
    section = str(getattr(parent, "summary", "") or "").strip()
    if heading and section:
        return f"{heading}\n{section}"
    return heading or section


def _equation_self_text(node) -> str:
    """What the equation says, in words and in symbols."""

    summary = str(getattr(node, "summary", "") or "").strip()
    latex = _node_content(node)
    if summary and latex:
        return f"{summary}\n{latex}"
    return summary or latex


def _equation_context_text(node) -> str:
    """The caption a displayed equation never had.

    A figure carries an author-written caption stating what it shows, and that
    caption is why figures retrieve well.  A displayed equation carries none:
    the prose explaining it sits in the sentences on either side, often running
    straight through, so the sentence after an equation can begin mid-clause.

    Measured on a probability textbook, for an equation defining a worked
    example's event set, this window scored 0.589 against the question that
    needs it, where the equation's own LaTeX scored 0.377 and its summary 0.341
    -- the summary describes the mathematics, the neighbours describe the
    situation.  Neither text wins both cases, which is why this is a separate
    representation rather than a blend.
    """

    siblings = list(getattr(getattr(node, "parent", None), "children", ()) or ())
    position = next(
        (index for index, item in enumerate(siblings) if item is node), None
    )
    if position is None:
        return ""
    span = _EQUATION_CONTEXT_NEIGHBOURS
    window = siblings[max(0, position - span) : position + span + 1]
    parts = [text for item in window if (text := _context_contribution(item))]
    return "\n".join(parts)[:_EQUATION_CONTEXT_CHARS]


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


def _iter_semantic_element_points(
    node_meta, base_metadata: dict
) -> Iterator["TreePointRecord"]:
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
            yield TreePointRecord(
                kind="semantic_element",
                metadata=metadata,
                vectors={"text": _semantic_element_embedding_text(structure, element)},
            )


def _node_vectors(node, node_type) -> tuple[dict[str, str], str | None]:
    """The named vectors one node carries, and the text fused into its crop.

    Slots go by role. A figure's prose is ``text`` whether it came from a
    caption or a vision model; an equation's neighbourhood is ``context`` even
    when the equation has no text of its own. Nothing is promoted between slots
    to fill a gap, because the slots are scored separately on purpose.
    """

    vectors: dict[str, str] = {}
    image_text: str | None = None
    crop = node.meta_info.img_path
    has_crop = bool(crop) and os.path.exists(crop)

    if node_type in _VISUAL_NODE_TYPES:
        # An empty text is not a representation of anything: it embeds to
        # whatever the model returns for "", and every figure lacking a caption
        # landed on that same point.
        if _visual_text(node):
            vectors["text"] = _visual_text(node)
        if has_crop:
            # No text at all: whatever is passed beside a crop is fused *into*
            # its vector rather than stored next to it, so text there does not
            # annotate the picture, it displaces it. Feeding every figure its
            # ~945-character summary pushed the whole visual modality below the
            # text vectors -- 4 image vectors in a top 200, best at rank 36 --
            # and a question that said "show me the image" returned none.
            #
            # A caption was the cautious middle ground while a figure had one
            # vector. It has two now: the caption is already in ``text``, via
            # context_contribution, where it is searched as the prose it is.
            # So the crop is left a picture, which is the one thing only it can
            # be matched on.
            vectors[_TREE_IMAGE_VECTOR] = crop
    elif node_type in (NodeType.TABLE, NodeType.LIST):
        # A grid embeds as markup: "<table><tr><td>" is noise, and the cells
        # that carry the meaning are values, not sentences. The summary is the
        # only prose a table has. A list has the same shape of problem: its
        # items are fragments sharing a subject stated once, in the stem above.
        table_str = _table_text(node)
        vectors["text"] = table_str
        if has_crop:
            vectors[_TREE_IMAGE_VECTOR] = crop
            image_text = table_str
    elif node_type == NodeType.EQUATION:
        # Two readings of one equation, because no single text serves both. A
        # definition is found through its summary -- "convergence in
        # probability" -- while a worked example is found through the sentences
        # around it -- "Romeo and Juliet arrive within 15 minutes". Measured,
        # each text that rescues one case degrades the other by about as much,
        # so they stay separate vectors and retrieval keeps whichever hits.
        self_text = _equation_self_text(node)
        context_text = _equation_context_text(node)
        if self_text:
            vectors["text"] = self_text
        if context_text and context_text != self_text:
            vectors["context"] = context_text
        if has_crop:
            # The crop is paired with the neighbourhood rather than the LaTeX:
            # a picture of symbols and a transcription of the same symbols are
            # the same evidence twice, and neither says what the formula is for.
            vectors[_TREE_IMAGE_VECTOR] = crop
            image_text = context_text or self_text or ""
    elif node_type in _LISTING_NODE_TYPES:
        if _listing_text(node):
            vectors["text"] = _listing_text(node)
    elif node_type in _TEXT_NODE_TYPES:
        text_content = node.meta_info.content
        if text_content:
            vectors["text"] = text_content
        # A heading is a few words; the section under it is what the heading is
        # about. "2.3.2.2.94. Zernike Surface" and "defines the Zernike surface
        # ... characterized by a sag equation" answer different questions, and
        # only the first was reachable. A separate vector rather than a blend,
        # for the reason the equation keeps two: a heading is the exact string
        # someone searching for the section types.
        if node_type == NodeType.TITLE:
            section_summary = str(getattr(node, "summary", "") or "").strip()
            if section_summary and section_summary != (text_content or "").strip():
                vectors["context"] = section_summary

    # A figure, table, listing or list shows something without naming it: the
    # subject is in the heading above. Added last and never over an existing
    # context, so a title keeps its own section summary and an equation keeps
    # its neighbourhood -- both measured, both about something other than the
    # enclosing section.
    #
    # Gated on the node having contributed a vector of its own. A figure with
    # no prose, no crop and nothing else is not made findable by the section it
    # sits in; indexing it on the section alone would put a contentless point,
    # near-identical to its siblings', into the same leg as the section itself.
    if vectors and node_type in _SECTION_CONTEXT_NODE_TYPES and "context" not in vectors:
        section_text = _section_context_text(node)
        if section_text:
            vectors["context"] = section_text
    return vectors, image_text


def iter_tree_point_records(tree: DocumentTree) -> Iterator[TreePointRecord]:
    """Yield one record per Qdrant point, in stable tree order.

    Two families: one point per tree node, and one per semantic element. An
    element is not a node -- it is a cell, symbol or flowchart box inside one --
    and it keeps its own point because its element id, kind and box are
    per-element payload a shared point could not carry.
    """

    for node in tree.nodes:
        if node == tree.root_node:
            continue

        meta_data = _node_trace_metadata(tree, node)
        vectors, image_text = _node_vectors(node, node.type)
        if vectors:
            yield TreePointRecord(
                kind="node",
                metadata=meta_data,
                vectors=vectors,
                image_text=image_text,
            )

        yield from _iter_semantic_element_points(node.meta_info, meta_data)


def iter_tree_index_records(tree: DocumentTree) -> Iterator[TreeIndexRecord]:
    """Flatten the points back into one record per vector.

    The index writes points now; this is the per-representation view the
    projection digest and the older readers were written against. Vectors come
    out in ``_TREE_DENSE_VECTORS`` order, which is the order the per-branch
    dispatch used to yield them in.
    """

    for point in iter_tree_point_records(tree):
        for name in _TREE_DENSE_VECTORS:
            value = point.vectors.get(name)
            if not value:
                continue
            modality = "image" if name == _TREE_IMAGE_VECTOR else "text"
            yield TreeIndexRecord(
                modality,
                value,
                _representation_metadata(point.metadata, modality),
                point.image_text if modality == "image" else None,
            )


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
    lexical_average_length: float,
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
            embedder = MultimodalEmbeddingProvider(
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
            vector_schema=VectorSchema.TREE_NODE,
        )
        # Deterministic upserts alone do not delete records removed from a
        # newer tree. A payload-scope reset reconciles stale records without
        # disturbing other documents, revisions, or index kinds in the collection.
        vdb.reset()

        # Points, not representations. A plain upsert replaces a point rather
        # than merging into it, so a node's vectors have to reach Qdrant in one
        # call -- buffering them separately would leave whichever went second
        # as the whole point.
        point_batch: list[dict] = []
        pending_texts = 0
        pending_crops = 0

        def flush_points() -> None:
            nonlocal pending_texts, pending_crops
            if not point_batch:
                return
            # Pass a snapshot because the reusable buffer is cleared after a
            # synchronous write. This also keeps adapter/mock implementations
            # from retaining a subsequently mutated list.
            vdb.add_tree_node_points(records=list(point_batch))
            point_batch.clear()
            pending_texts = 0
            pending_crops = 0

        for point in iter_tree_point_records(tree):
            vectors = {
                name: value
                for name, value in point.vectors.items()
                if name != _TREE_IMAGE_VECTOR or vdb_cfg.mm_embedding
            }
            if not vectors:
                continue
            point_batch.append(
                {
                    "kind": point.kind,
                    "metadata": point.metadata,
                    "vectors": vectors,
                    "image_text": point.image_text,
                    # BM25 weights, not raw counts: Qdrant's IDF modifier is
                    # only the idf half, so the saturation and the length
                    # normalization have to be in the values it multiplies.
                    "lexical_terms": (
                        bm25_terms(
                            point.lexical_source,
                            average_length=lexical_average_length,
                        )
                        if point.lexical_source
                        else {}
                    ),
                    "lexical_source": point.lexical_source,
                }
            )
            crops = 1 if _TREE_IMAGE_VECTOR in vectors else 0
            pending_texts += len(vectors) - crops
            pending_crops += crops
            # Bound the batch by whichever embedder call it would make largest,
            # so a document of titles and a document of figures both keep their
            # per-call work at the configured size.
            if (
                len(point_batch) >= effective_batch_size
                or pending_texts >= effective_batch_size
                or pending_crops >= effective_batch_size
            ):
                flush_points()

        flush_points()

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


#: How far the frozen BM25 normalizer may drift from the corpus mean before a
#: reconcile refuses. Incremental edits keep the attested average rather than
#: re-weighting every point against a new one, so the two separate slowly. A
#: tenth is far past the point where it changes a ranking and far short of
#: where an edit becomes useless.
DEFAULT_MAX_AVERAGE_LENGTH_DRIFT = 0.10


@dataclass(frozen=True)
class TreeVectorReconciliation:
    """What one reconcile moved, and what it left alone."""

    added: int
    removed: int
    unchanged: int
    point_count: int
    #: The normalizer the stored weights use.
    average_length: float
    #: What a full rebuild would compute instead.
    corpus_average_length: float

    @property
    def drift(self) -> float:
        if not (self.average_length > 0):
            return 0.0
        return abs(self.corpus_average_length - self.average_length) / self.average_length

    def describe(self) -> str:
        return (
            f"{self.added} point(s) re-embedded, {self.removed} removed, "
            f"{self.unchanged} untouched, {self.point_count} attested"
        )


def _projected_tree_points(
    tree: DocumentTree, vdb_cfg: VDBConfig
) -> tuple[list[dict], float, int]:
    """Every point this tree projects to, in tree order, with its token count.

    The point's identity is content-addressed over each vector's source, so
    the resulting IDs *are* the dependency graph: a title whose summary changed
    re-IDs its own point and, through ``_section_context_text``, every child
    point that quoted it. No caller has to know which nodes an edit reached.

    The IDs themselves are not computed here. They depend on the store's scope,
    which is applied at write time, so only the store can say what a record
    will be keyed under -- see ``VectorStore.tree_point_ids``.
    """

    projected: list[dict] = []
    lexical_token_total = 0.0
    sparse_count = 0
    for point in iter_tree_point_records(tree):
        vectors = {
            name: value
            for name, value in point.vectors.items()
            if name != _TREE_IMAGE_VECTOR or vdb_cfg.mm_embedding
        }
        if not vectors:
            continue
        projected.append(
            {
                "kind": point.kind,
                "metadata": point.metadata,
                "vectors": vectors,
                "image_text": point.image_text,
                "lexical_source": point.lexical_source,
            }
        )
        if point.lexical_source:
            sparse_count += 1
            lexical_token_total += sum(lexical_terms(point.lexical_source).values())
    corpus_average = lexical_token_total / sparse_count if sparse_count else 0.0
    return projected, corpus_average, sparse_count


def reconcile_tree_vector_index(
    tree: DocumentTree,
    vdb_cfg: VDBConfig,
    *,
    frozen_average_length: float,
    batch_size: int | None = None,
    max_average_length_drift: float = DEFAULT_MAX_AVERAGE_LENGTH_DRIFT,
) -> TreeVectorReconciliation:
    """Bring one scope's points to what the tree now projects, and no further.

    The full builder resets the scope and rewrites every point, which for an
    edited summary means re-embedding a whole document to change four vectors.
    This diffs instead: what the tree projects against what Qdrant holds, by
    point ID. Only the difference is embedded, and only the points the
    projection no longer contains are deleted.

    ``frozen_average_length`` is the BM25 normalizer already in the
    attestation. Keeping it is what makes the write small: recomputing the mean
    would change every stored weight in the index, none of whose point IDs
    change with it, so the diff could not see the work and every point would
    have to be rewritten anyway. Holding it fixed keeps the index internally
    consistent -- every weight normalized the same way -- at the cost of the
    mean slowly ceasing to be the corpus mean, which is what the drift ceiling
    watches.
    """

    if not (frozen_average_length > 0):
        raise ValueError("frozen_average_length must be positive")
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

    projected, corpus_average, _sparse_count = _projected_tree_points(tree, vdb_cfg)
    if not projected:
        raise VectorIndexBindingError("refusing to publish an empty vector index")
    drift = (
        abs(corpus_average - frozen_average_length) / frozen_average_length
        if frozen_average_length > 0
        else 0.0
    )
    if drift > max_average_length_drift:
        raise VectorIndexBindingError(
            "BM25 average document length has drifted "
            f"{drift:.1%} from the attested {frozen_average_length:.3f} "
            f"(corpus is now {corpus_average:.3f}); rebuild the vdb stage "
            "instead of reconciling"
        )

    embedder = None
    vdb = None
    active_error = False
    try:
        if vdb_cfg.mm_embedding:
            embedder = MultimodalEmbeddingProvider(
                model_name=vdb_cfg.embedding_config.model_name,
                device=vdb_cfg.embedding_config.device,
            )
        else:
            embedder = TextEmbeddingProvider(
                model_name=vdb_cfg.embedding_config.model_name,
                device=vdb_cfg.embedding_config.device,
                backend=vdb_cfg.embedding_config.backend,
                api_base=vdb_cfg.embedding_config.api_base,
                api_key=getattr(vdb_cfg.embedding_config, "api_key", None),
                max_length=vdb_cfg.embedding_config.max_length,
            )
        vdb = VectorStore(
            embedding_model=embedder,
            qdrant_url=vdb_cfg.qdrant_url,
            collection_name=vdb_cfg.collection_name,
            api_key_env=vdb_cfg.api_key_env,
            timeout_seconds=vdb_cfg.timeout_seconds,
            scope_filter=tree_vector_scope(tree, vdb_cfg),
            vector_schema=VectorSchema.TREE_NODE,
        )

        # The store derives the IDs, because the scope it applies at write time
        # is part of them.
        point_ids = vdb.tree_point_ids(projected)
        live = vdb.live_point_ids()
        wanted = set(point_ids)
        to_add = [
            record
            for record, point_id in zip(projected, point_ids)
            if point_id not in live
        ]
        to_remove = sorted(live.difference(wanted))

        for start in range(0, len(to_add), effective_batch_size):
            vdb.add_tree_node_points(
                records=[
                    {
                        **record,
                        "lexical_terms": (
                            bm25_terms(
                                record["lexical_source"],
                                average_length=frozen_average_length,
                            )
                            if record["lexical_source"]
                            else {}
                        ),
                    }
                    for record in to_add[start : start + effective_batch_size]
                ],
                batch_size=effective_batch_size,
            )
        # Added before removed: a crash between the two leaves the scope holding
        # both readings of a node, which the next reconcile cleans up. The other
        # order leaves it holding neither.
        if to_remove:
            vdb.delete_points(to_remove)

        observed = vdb.count()
        if observed != len(wanted):
            raise VectorIndexBindingError(
                f"tree vector scope contains {observed} points after reconcile; "
                f"expected {len(wanted)}"
            )
        return TreeVectorReconciliation(
            added=len(to_add),
            removed=len(to_remove),
            unchanged=len(wanted) - len(to_add),
            point_count=len(wanted),
            average_length=frozen_average_length,
            corpus_average_length=corpus_average,
        )
    except BaseException:
        active_error = True
        raise
    finally:
        cleanup_errors = []
        if vdb is not None:
            try:
                vdb.close()
            except Exception as exc:
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
                lexical_average_length=float(expected["lexical_average_length"]),
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
    embedder = MultimodalEmbeddingProvider(
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
            if node_type in _CROP_BEARING_NODE_TYPES
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
    embedder = MultimodalEmbeddingProvider(
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
