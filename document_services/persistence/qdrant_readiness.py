"""Dependency-light live validation of one source-bound Qdrant attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import yaml

from retrieval_engine.utils.artifact_paths import (
    QdrantProtocolError,
    QdrantUnavailableError,
    qdrant_collection_state,
    qdrant_endpoint_sha256,
)


class QdrantAttestationMismatch(RuntimeError):
    """The local binding and live scoped Qdrant state are inconsistent."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_self_digest(attestation: dict[str, object]) -> None:
    """Reject mutable or fabricated readiness records before contacting Qdrant."""

    identity_sha256 = attestation.get("identity_sha256")
    if attestation.get("state") != "content_attested" or not _is_sha256(
        identity_sha256
    ):
        raise QdrantAttestationMismatch(
            "Qdrant attestation has no valid content identity"
        )
    identity = {
        key: value
        for key, value in attestation.items()
        if key not in {"identity_sha256", "state"}
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != identity_sha256:
        raise QdrantAttestationMismatch("Qdrant attestation self-digest mismatches")


def validate_qdrant_attestation(
    *,
    config_path: Path,
    attestation_path: Path,
    revision_id: str,
    expected_index_kind: str | None = None,
) -> int:
    """Return the exact live point count or raise a classified failure."""

    if not revision_id or revision_id != revision_id.strip():
        raise ValueError("revision_id must be a non-empty trimmed string")
    if (
        attestation_path.is_symlink()
        or not attestation_path.is_file()
        or not (0 < attestation_path.stat().st_size <= 64 * 1024)
    ):
        raise QdrantAttestationMismatch("Qdrant attestation is missing or unsafe")
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise QdrantAttestationMismatch(
            "Qdrant attestation or runtime configuration is invalid"
        ) from exc
    vdb = config.get("vdb") if isinstance(config, dict) else None
    source = config.get("source") if isinstance(config, dict) else None
    if not isinstance(attestation, dict) or not isinstance(vdb, dict):
        raise QdrantAttestationMismatch("Qdrant attestation binding is incomplete")
    _validate_self_digest(attestation)
    scope = attestation.get("scope_filter")
    required_scope = {
        "tenant_id",
        "corpus_id",
        "document_version_id",
        "index_revision_id",
        "source_sha256",
        "index_kind",
    }
    if not isinstance(scope, dict) or set(scope) != required_scope:
        raise QdrantAttestationMismatch("Qdrant attestation scope is invalid")
    if any(
        not isinstance(scope.get(key), str)
        or not scope[key]
        or scope[key] != scope[key].strip()
        for key in required_scope
    ):
        raise QdrantAttestationMismatch("Qdrant attestation scope values are invalid")
    configured_source_sha256 = source.get("sha256") if isinstance(source, dict) else None
    configured_document_version_id = (
        source.get("document_version_id") if isinstance(source, dict) else None
    ) or vdb.get("document_version_id")
    if (
        scope.get("tenant_id") != vdb.get("tenant_id")
        or scope.get("corpus_id") != vdb.get("corpus_id")
        or scope.get("document_version_id") != configured_document_version_id
        or scope.get("index_revision_id") != revision_id
        or scope.get("source_sha256") != configured_source_sha256
        or attestation.get("document_version_id") != configured_document_version_id
        or attestation.get("index_revision_id") != revision_id
    ):
        raise QdrantAttestationMismatch(
            "Qdrant attestation belongs to another tenant, corpus, document, or source"
        )
    if expected_index_kind is not None and scope.get("index_kind") != expected_index_kind:
        raise QdrantAttestationMismatch("Qdrant attestation index kind mismatches")
    qdrant_url_env = vdb.get("qdrant_url_env", "DOCARAG_QDRANT_URL")
    if (
        not isinstance(qdrant_url_env, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", qdrant_url_env) is None
    ):
        raise QdrantAttestationMismatch(
            "Qdrant endpoint environment reference is invalid"
        )
    qdrant_url = os.environ.get(qdrant_url_env, vdb.get("qdrant_url"))
    if attestation.get("qdrant_endpoint_sha256") != qdrant_endpoint_sha256(qdrant_url):
        raise QdrantAttestationMismatch("Qdrant endpoint binding mismatches")
    collection_name = attestation.get("collection_name")
    point_count = attestation.get("point_count")
    if (
        not isinstance(collection_name, str)
        or not collection_name
        or collection_name != collection_name.strip()
        or isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 1
    ):
        raise QdrantAttestationMismatch("Qdrant point binding is invalid")
    state = qdrant_collection_state(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        scope_filter=scope,
        api_key_env=vdb.get("api_key_env", "QDRANT_API_KEY"),
        timeout_seconds=vdb.get("timeout_seconds", 10.0),
    )
    if not state.exists or state.point_count != point_count:
        raise QdrantAttestationMismatch(
            "Qdrant scoped point count differs from the attestation"
        )
    return point_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--expected-index-kind")
    args = parser.parse_args(argv)
    try:
        count = validate_qdrant_attestation(
            config_path=args.config,
            attestation_path=args.attestation,
            revision_id=args.revision_id,
            expected_index_kind=args.expected_index_kind,
        )
    except QdrantUnavailableError as exc:
        print(f"unavailable: {exc}", file=sys.stderr)
        return 3
    except (QdrantProtocolError, QdrantAttestationMismatch, TypeError, ValueError) as exc:
        print(f"mismatch: {exc}", file=sys.stderr)
        return 2
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
