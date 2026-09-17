"""Resolve BookRAG's configured artifact roles in the model interpreter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def _bind_identity_from_run_manifest(config, run: Path) -> None:
    """Populate the source/scope identity the driver recorded for this run.

    The runtime interpreter (``main.py``) binds ``config.source`` from the
    dataset row before building the index.  This out-of-band resolver runs in a
    separate subprocess against only the bare config snapshot, so it recovers
    the identical, already-validated identity from ``run-manifest.json``.  A
    missing or sourceless manifest is left untouched: the caller then raises the
    same explicit "source SHA-256 is required" error.
    """

    from retrieval_engine.configuration.system import SourceIdentity

    manifest_path = run / "run-manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    source = manifest.get("source")
    if not isinstance(source, dict):
        return
    sha256 = source.get("sha256")
    original_uri = source.get("uri")
    source_path = source.get("path")
    if not sha256 or not original_uri or not source_path:
        return

    document_id = manifest.get("document_id")
    document_version_id = manifest.get("document_version_id")
    revision_id = manifest.get("revision_id")
    # The manifest URI and file name are exactly the values the driver hashed to
    # derive these identifiers, so the SourceIdentity validator reproduces them.
    config.source = SourceIdentity(
        display_name=Path(source_path).name,
        original_uri=original_uri,
        sha256=sha256,
        snapshot_path=source.get("snapshot_path"),
        document_id=document_id,
        document_version_id=document_version_id,
        index_revision_id=revision_id,
    )

    tenant_id = manifest.get("tenant_id")
    corpus_id = manifest.get("corpus_id")
    if tenant_id:
        config.vdb.tenant_id = tenant_id
    if corpus_id:
        config.vdb.corpus_id = corpus_id
    if document_id:
        config.vdb.document_id = document_id
    if document_version_id:
        config.vdb.document_version_id = document_version_id
    if revision_id:
        config.vdb.index_revision_id = revision_id


def resolve_artifact_roles(config_path: Path, run_dir: Path) -> dict[str, object]:
    """Use the same helpers as BookRAG's readers/writers to name outputs."""

    # These imports intentionally stay local: this helper runs in the configured
    # BookRAG environment, while the controller CLI remains dependency-light.
    from retrieval_engine.indexes.graph import Graph
    from retrieval_engine.configuration.knowledge_graph import (
        entity_attestation_directory,
        graph_artifact_variant,
    )
    from retrieval_engine.configuration.system import load_system_config
    from retrieval_engine.utils.artifact_paths import (
        resolve_run_artifact_path,
        qdrant_endpoint_sha256,
    )
    from retrieval_engine.indexes.vector_manifest import VECTOR_INDEX_ATTESTATION_NAME

    run = run_dir.expanduser().resolve()
    config = load_system_config(str(config_path.expanduser().resolve()))
    # The immutable config snapshot is a byte-exact, content-addressed copy of the
    # user's configuration, which legitimately carries no per-document `source`.
    # The driver records the admitted source identity in the run manifest; bind it
    # here so this helper resolves the same Qdrant scope the runtime will publish.
    if config.source is None:
        _bind_identity_from_run_manifest(config, run)
    graph = run / Graph._get_filename(graph_artifact_variant(config.graph))
    entity_dir = Path(entity_attestation_directory(str(run), config.graph)).resolve()
    retrieval_dir = resolve_run_artifact_path(
        run,
        config.vdb.manifest_dir_name,
        label="vdb.manifest_dir_name",
    )
    for label, path in (
        ("graph", graph),
        ("entity attestation", entity_dir),
        ("retrieval artifact", retrieval_dir),
    ):
        try:
            path.relative_to(run)
        except ValueError as exc:
            raise ValueError(f"configured {label} escapes the run: {path}") from exc
    source_sha256 = config.source.sha256 if config.source is not None else None
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("configured source SHA-256 is required for Qdrant roles")
    if config.source is None:
        raise ValueError("configured source identity is required for Qdrant roles")
    from document_services.domain.evidence import DocumentRef

    expected_document = DocumentRef.identified(
        content_sha256=source_sha256,
        mime_type="application/pdf",
        source_uri=config.source.original_uri,
        display_name=config.source.display_name,
    )
    document_version_id = (
        config.source.document_version_id
        or config.vdb.document_version_id
        or expected_document.document_version_id
    )
    if document_version_id != expected_document.document_version_id:
        raise ValueError("configured Qdrant document version differs from its source")
    index_revision_id = (
        config.source.index_revision_id
        or config.vdb.index_revision_id
        or run.name
    )
    if index_revision_id != run.name:
        raise ValueError("configured Qdrant index revision differs from the run")
    scope = {
        "tenant_id": config.vdb.tenant_id,
        "corpus_id": config.vdb.corpus_id,
        "document_version_id": document_version_id,
        "index_revision_id": index_revision_id,
        "source_sha256": source_sha256,
    }
    roles: dict[str, object] = {
        "index_type": str(config.index_type),
        "graph": str(graph),
        "entity_attestation": str(entity_dir / VECTOR_INDEX_ATTESTATION_NAME),
        "entity_collection": config.graph.entity_collection_name,
        "entity_scope": {**scope, "index_kind": "entity"},
        "qdrant_endpoint_sha256": qdrant_endpoint_sha256(config.vdb.qdrant_url),
    }
    if config.index_type == "bm25":
        roles["lexical_store"] = str(retrieval_dir / "bm25_index.pkl")
    else:
        roles["retrieval_attestation"] = str(
            retrieval_dir / VECTOR_INDEX_ATTESTATION_NAME
        )
        roles["retrieval_collection"] = config.vdb.collection_name
        roles["retrieval_scope"] = {
            **scope,
            "index_kind": "tree" if config.index_type in {"gbc", "tree"} else str(config.index_type),
        }
    return roles


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve configured BookRAG artifacts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    roles = resolve_artifact_roles(args.config, args.run)
    _atomic_write(
        args.output.expanduser().resolve(),
        (json.dumps(roles, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
