"""Application use case for a traceable routed canonical-document extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from collections.abc import Mapping, Sequence

from docarag.application.contracts import (
    DocumentExtractionRequest,
    DocumentExtractionResult,
)
from docarag.application.ports import (
    CanonicalExtractionPayload,
    DocumentExtractionBackend,
    RoutedAuditValidator,
    ValidatedRoutedAudit,
)
from docarag.canonical_artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_ARTIFACT_ROLE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION,
    ROUTED_PARSING_AUDIT_SCHEMA_VERSION,
)
from docarag.domain.base import (
    ContractValidationError,
    canonical_json_bytes,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
)
from docarag.domain.evidence import CanonicalBlock, DocumentRef, QualityStatus


class DocumentExtractionError(RuntimeError):
    """Base application-level extraction failure."""


class DocumentExtractionBindingError(DocumentExtractionError):
    """A backend result is not bound to the requested immutable source."""


class DocumentExtractionArtifactError(DocumentExtractionError):
    """The durable canonical artifact is unsafe, unstable, or out of bounds."""


@dataclass(frozen=True, slots=True)
class DocumentExtractionPolicy:
    """Application-level bounds independent of any parser implementation."""

    max_canonical_blocks: int = 1_000_000
    max_canonical_artifact_bytes: int = 128 * 1024 * 1024
    max_audit_artifact_bytes: int = 128 * 1024 * 1024
    max_artifact_json_depth: int = 256
    max_artifact_json_nodes: int = 5_000_000
    sha256_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for field_name, minimum, maximum in (
            ("max_canonical_blocks", 1, 10_000_000),
            (
                "max_canonical_artifact_bytes",
                1024,
                4 * 1024 * 1024 * 1024,
            ),
            (
                "max_audit_artifact_bytes",
                1024,
                4 * 1024 * 1024 * 1024,
            ),
            ("max_artifact_json_depth", 8, 512),
            ("max_artifact_json_nodes", 1_000, 10_000_000),
            ("sha256_chunk_bytes", 4096, 64 * 1024 * 1024),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{field_name} must be an integer between {minimum} and {maximum}"
                )


class ExtractDocumentUseCase:
    """Coordinate one backend and return a compact, content-bound receipt."""

    def __init__(
        self,
        backend: DocumentExtractionBackend,
        *,
        audit_validator: RoutedAuditValidator | None = None,
        policy: DocumentExtractionPolicy | None = None,
    ) -> None:
        if not isinstance(backend, DocumentExtractionBackend):
            raise TypeError("backend must implement DocumentExtractionBackend")
        active_policy = policy or DocumentExtractionPolicy()
        if not isinstance(active_policy, DocumentExtractionPolicy):
            raise TypeError("policy must be DocumentExtractionPolicy")
        active_validator = audit_validator or getattr(
            backend, "audit_validator", None
        )
        if not isinstance(active_validator, RoutedAuditValidator):
            raise TypeError(
                "audit_validator must implement RoutedAuditValidator; pass one "
                "explicitly or use a backend that exposes audit_validator"
            )
        try:
            configuration_sha256 = require_sha256(
                backend.configuration_sha256,
                "backend.configuration_sha256",
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise TypeError(str(exc)) from exc
        self._backend = backend
        self._audit_validator = active_validator
        self._policy = active_policy
        self._configuration_sha256 = configuration_sha256

    @property
    def policy(self) -> DocumentExtractionPolicy:
        return self._policy

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    def execute(
        self,
        request: DocumentExtractionRequest,
    ) -> DocumentExtractionResult:
        if not isinstance(request, DocumentExtractionRequest):
            raise TypeError("request must be DocumentExtractionRequest")
        if self._backend.configuration_sha256 != self._configuration_sha256:
            raise DocumentExtractionBindingError(
                "backend configuration identity changed after composition"
            )
        payload = self._backend.extract(request)
        if self._backend.configuration_sha256 != self._configuration_sha256:
            raise DocumentExtractionBindingError(
                "backend configuration identity changed during extraction"
            )
        if not isinstance(payload, CanonicalExtractionPayload):
            raise DocumentExtractionBindingError(
                "backend must return CanonicalExtractionPayload"
            )
        blocks = self._validate_payload(request, payload)
        observed_blocks_sha256 = _canonical_blocks_sha256(blocks)
        if not hmac.compare_digest(
            observed_blocks_sha256,
            payload.canonical_blocks_sha256,
        ):
            raise DocumentExtractionBindingError(
                "backend canonical_blocks_sha256 differs from canonical_blocks"
            )
        (
            artifact_path,
            artifact_sha256,
            artifact_size,
            artifact_payload,
            _artifact_content_sha256,
        ) = _inspect_artifact(
            payload.canonical_artifact_path,
            output_root=request.output_root,
            maximum_bytes=self._policy.max_canonical_artifact_bytes,
            chunk_bytes=self._policy.sha256_chunk_bytes,
            artifact_label="canonical artifact",
            max_json_depth=self._policy.max_artifact_json_depth,
            max_json_nodes=self._policy.max_artifact_json_nodes,
        )
        canonical_binding = _validate_canonical_artifact(
            artifact_payload,
            document=payload.document,
            blocks=blocks,
            configuration_sha256=self._configuration_sha256,
            canonical_blocks_sha256=payload.canonical_blocks_sha256,
        )
        del artifact_payload
        (
            audit_path,
            audit_sha256,
            audit_size,
            audit_payload,
            routed_audit_sha256,
        ) = _inspect_artifact(
            payload.audit_artifact_path,
            output_root=request.output_root,
            maximum_bytes=self._policy.max_audit_artifact_bytes,
            chunk_bytes=self._policy.sha256_chunk_bytes,
            artifact_label="routed audit artifact",
            max_json_depth=self._policy.max_artifact_json_depth,
            max_json_nodes=self._policy.max_artifact_json_nodes,
        )
        if not hmac.compare_digest(
            routed_audit_sha256,
            canonical_binding.audit_sha256,
        ):
            raise DocumentExtractionBindingError(
                "routed audit artifact differs from canonical artifact audit_sha256"
            )
        _validate_routed_audit_artifact(
            audit_payload,
            validator=self._audit_validator,
            binding=canonical_binding,
            document=payload.document,
            blocks=blocks,
        )
        del audit_payload
        return DocumentExtractionResult(
            request_id=request.request_id,
            backend_configuration_sha256=self._configuration_sha256,
            document=payload.document,
            canonical_artifact_uri=artifact_path.as_uri(),
            canonical_artifact_sha256=artifact_sha256,
            canonical_artifact_size_bytes=artifact_size,
            audit_artifact_uri=audit_path.as_uri(),
            audit_artifact_sha256=audit_sha256,
            audit_artifact_size_bytes=audit_size,
            routed_audit_sha256=routed_audit_sha256,
            canonical_block_count=len(blocks),
            canonical_blocks_sha256=payload.canonical_blocks_sha256,
        )

    def _validate_payload(
        self,
        request: DocumentExtractionRequest,
        payload: CanonicalExtractionPayload,
    ) -> tuple[CanonicalBlock, ...]:
        document = payload.document
        if not isinstance(document, DocumentRef):
            raise DocumentExtractionBindingError(
                "backend document must be a DocumentRef"
            )
        expected = {
            "content_sha256": request.source_sha256,
            "mime_type": request.media_type,
            "source_uri": request.source_uri,
            "display_name": request.display_name,
        }
        if any(getattr(document, name, None) != value for name, value in expected.items()):
            raise DocumentExtractionBindingError(
                "backend document identity differs from the extraction request"
            )
        if document.page_count is None or document.page_count < 1:
            raise DocumentExtractionBindingError(
                "backend document must declare a positive page_count"
            )
        try:
            blocks = tuple(payload.canonical_blocks)
        except TypeError as exc:
            raise DocumentExtractionBindingError(
                "backend canonical_blocks must be iterable"
            ) from exc
        if not blocks:
            raise DocumentExtractionBindingError(
                "backend returned no canonical blocks"
            )
        if len(blocks) > self._policy.max_canonical_blocks:
            raise DocumentExtractionBindingError(
                "backend exceeded max_canonical_blocks "
                f"({self._policy.max_canonical_blocks})"
            )
        if any(not isinstance(block, CanonicalBlock) for block in blocks):
            raise DocumentExtractionBindingError(
                "backend canonical_blocks must contain CanonicalBlock values"
            )
        if any(
            block.document_version_id != document.document_version_id
            or block.page_index >= document.page_count
            for block in blocks
        ):
            raise DocumentExtractionBindingError(
                "backend canonical blocks are outside the returned document"
            )
        if any(block.quality_status is not QualityStatus.ACCEPTED for block in blocks):
            raise DocumentExtractionBindingError(
                "only accepted canonical blocks may enter the application handoff"
            )
        for values, label in (
            ((block.block_id for block in blocks), "block_id"),
            ((block.revision_id for block in blocks), "revision_id"),
            ((block.canonical_key for block in blocks), "canonical_key"),
        ):
            materialized = tuple(values)
            if len(materialized) != len(set(materialized)):
                raise DocumentExtractionBindingError(
                    f"backend returned duplicate canonical {label} values"
                )
        return blocks


def _canonical_blocks_sha256(blocks: tuple[CanonicalBlock, ...]) -> str:
    """Hash the canonical JSON array with O(largest block) transient memory."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, block in enumerate(blocks):
        if index:
            digest.update(b",")
        digest.update(canonical_json_bytes(block.to_dict()))
    digest.update(b"]")
    return digest.hexdigest()


def _normalized_absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise DocumentExtractionArtifactError(
            f"{label} must be a filesystem path"
        )
    try:
        raw = os.fspath(value)
    except (TypeError, ValueError) as exc:
        raise DocumentExtractionArtifactError(f"{label} path is invalid") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise DocumentExtractionArtifactError(
            f"{label} must be a non-empty filesystem path"
        )
    path = Path(os.path.normpath(raw))
    if not path.is_absolute() or path.anchor != os.path.sep:
        raise DocumentExtractionArtifactError(
            f"{label} path must be an absolute local path"
        )
    return path


def _secure_directory_flags() -> int:
    """Return portable dirfd traversal flags or fail closed."""

    if (
        os.open not in getattr(os, "supports_dir_fd", ())
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise DocumentExtractionArtifactError(
            "secure artifact traversal requires openat, O_DIRECTORY, and O_NOFOLLOW"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_chain(path: Path, label: str) -> int:
    """Open every absolute directory component relative to a pinned parent."""

    flags = _secure_directory_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open(os.path.sep, flags)
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                flags,
                dir_fd=descriptor,
            )
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        return descriptor
    except (OSError, NotImplementedError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DocumentExtractionArtifactError(
            f"{label} could not be traversed without symlinks"
        ) from exc


def _secure_open_artifact(
    path_value: object,
    *,
    output_root: object,
    artifact_label: str,
) -> tuple[Path, int]:
    """Open a regular-file candidate beneath a pinned, symlink-free root.

    Every component is resolved with ``openat`` relative to an already opened
    directory.  This removes the check-then-open ancestor race inherent in
    ``Path.resolve`` plus a final-component ``O_NOFOLLOW``.
    """

    artifact_path = _normalized_absolute_path(path_value, artifact_label)
    root_path = _normalized_absolute_path(output_root, "output root")
    try:
        relative = artifact_path.relative_to(root_path)
    except ValueError as exc:
        raise DocumentExtractionArtifactError(
            f"{artifact_label} must be a regular file beneath output_root"
        ) from exc
    if not relative.parts:
        raise DocumentExtractionArtifactError(
            f"{artifact_label} must be a regular file beneath output_root"
        )

    parent_descriptor = _open_directory_chain(root_path, "output root")
    try:
        directory_flags = _secure_directory_flags()
        for component in relative.parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            previous_descriptor = parent_descriptor
            parent_descriptor = next_descriptor
            os.close(previous_descriptor)
        file_flags = (
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=parent_descriptor,
        )
    except (OSError, NotImplementedError) as exc:
        raise DocumentExtractionArtifactError(
            f"{artifact_label} could not be opened safely beneath output_root"
        ) from exc
    finally:
        os.close(parent_descriptor)
    return artifact_path, descriptor


def _inspect_artifact(
    path_value: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    maximum_bytes: int,
    chunk_bytes: int,
    artifact_label: str,
    max_json_depth: int,
    max_json_nodes: int,
) -> tuple[Path, str, int, Mapping[str, object], str]:
    artifact_path, descriptor = _secure_open_artifact(
        path_value,
        output_root=output_root,
        artifact_label=artifact_label,
    )
    reopened_descriptor: int | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DocumentExtractionArtifactError(
                f"{artifact_label} must be a regular file"
            )
        if before.st_size <= 0:
            raise DocumentExtractionArtifactError(
                f"{artifact_label} must not be empty"
            )
        if before.st_size > maximum_bytes:
            raise DocumentExtractionArtifactError(
                f"{artifact_label} exceeds its configured byte limit "
                f"({maximum_bytes})"
            )
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, chunk_bytes)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise DocumentExtractionArtifactError(
                    f"{artifact_label} grew beyond its configured byte limit"
                )
            digest.update(chunk)
        first_digest = digest.hexdigest()
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb") as source:
                artifact_payload = json.load(
                    _BoundedJSONReader(source, maximum_bytes, artifact_label),
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(
                            f"{artifact_label} contains a non-finite JSON value "
                            f"{value!r}"
                        )
                    ),
                )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise DocumentExtractionArtifactError(
                f"{artifact_label} must be strict UTF-8 JSON"
            ) from exc
        if not isinstance(artifact_payload, Mapping):
            raise DocumentExtractionArtifactError(
                f"{artifact_label} root must be a JSON object"
            )
        try:
            payload_content_sha256 = _streaming_canonical_json_sha256(
                artifact_payload,
                max_depth=max_json_depth,
                max_nodes=max_json_nodes,
            )
        except (TypeError, ValueError) as exc:
            raise DocumentExtractionArtifactError(
                f"{artifact_label} exceeds its JSON structure limits: {exc}"
            ) from exc
        os.lseek(descriptor, 0, os.SEEK_SET)
        verification_digest = hashlib.sha256()
        verification_size = 0
        while True:
            chunk = os.read(descriptor, chunk_bytes)
            if not chunk:
                break
            verification_size += len(chunk)
            if verification_size > maximum_bytes:
                raise DocumentExtractionArtifactError(
                    f"{artifact_label} grew beyond its configured byte limit"
                )
            verification_digest.update(chunk)
        after = os.fstat(descriptor)
        reopened_path, reopened_descriptor = _secure_open_artifact(
            path_value,
            output_root=output_root,
            artifact_label=artifact_label,
        )
        if reopened_path != artifact_path:
            raise DocumentExtractionArtifactError(
                f"{artifact_label} path identity changed while it was read"
            )
        path_after = os.fstat(reopened_descriptor)
    except OSError as exc:
        raise DocumentExtractionArtifactError(
            f"{artifact_label} could not be read"
        ) from exc
    finally:
        if reopened_descriptor is not None:
            os.close(reopened_descriptor)
        os.close(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_fingerprint_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        fingerprint_before != fingerprint_after
        or fingerprint_after != path_fingerprint_after
        or observed_size != before.st_size
        or verification_size != observed_size
        or not hmac.compare_digest(first_digest, verification_digest.hexdigest())
    ):
        raise DocumentExtractionArtifactError(
            f"{artifact_label} changed while its receipt was created"
        )
    return (
        artifact_path,
        first_digest,
        observed_size,
        artifact_payload,
        payload_content_sha256,
    )


class _BoundedJSONReader:
    """Prevent ``json.load`` from following concurrent unbounded file growth."""

    def __init__(self, source: object, maximum_bytes: int, label: str) -> None:
        self._source = source
        self._maximum_bytes = maximum_bytes
        self._label = label
        self._observed = 0

    def read(self, size: int = -1) -> bytes:
        remaining_with_sentinel = self._maximum_bytes - self._observed + 1
        requested = (
            remaining_with_sentinel
            if size is None or size < 0
            else min(size, remaining_with_sentinel)
        )
        chunk = self._source.read(requested)  # type: ignore[attr-defined]
        self._observed += len(chunk)
        if self._observed > self._maximum_bytes:
            raise ValueError(f"{self._label} grew beyond its configured byte limit")
        return chunk


def _streaming_canonical_json_sha256(
    value: object,
    *,
    max_depth: int,
    max_nodes: int,
) -> str:
    """Hash canonical JSON incrementally without a second whole-file buffer."""

    digest = hashlib.sha256()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"node count exceeds {max_nodes}")
        if depth > max_depth:
            raise ValueError(f"depth exceeds {max_depth}")
        if isinstance(item, Mapping):
            keys = sorted(item)
            if any(not isinstance(key, str) for key in keys):
                raise TypeError("JSON object keys must be strings")
            digest.update(b"{")
            for index, key in enumerate(keys):
                if index:
                    digest.update(b",")
                digest.update(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
                digest.update(b":")
                visit(item[key], depth + 1)
            digest.update(b"}")
            return
        if isinstance(item, list):
            digest.update(b"[")
            for index, child in enumerate(item):
                if index:
                    digest.update(b",")
                visit(child, depth + 1)
            digest.update(b"]")
            return
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"unsupported JSON value {type(item).__name__}")
        digest.update(
            json.dumps(
                item,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

    visit(value, 0)
    return digest.hexdigest()


def _reject_duplicate_json_keys(
    values: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(
                f"canonical artifact contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class _CanonicalArtifactBinding:
    audit_sha256: str
    parse_request_id: str
    projection_id: str
    routed_parsing_result_id: str


def _validate_canonical_artifact(
    artifact: Mapping[str, object],
    *,
    document: DocumentRef,
    blocks: Sequence[CanonicalBlock],
    configuration_sha256: str,
    canonical_blocks_sha256: str,
) -> _CanonicalArtifactBinding:
    """Prove that the artifact bytes contain the exact in-memory handoff.

    Hashing an artifact and hashing blocks independently is not a binding: a
    backend could otherwise return block set A and an unrelated file B.  The
    application therefore validates the canonical envelope and compares every
    serialized block to the already validated DTO without materializing a
    second block collection.
    """

    allowed = {
        "artifact_schema_version",
        "contract_schema_version",
        "artifact_role",
        "producer_schema_version",
        "configuration_sha256",
        "audit_sha256",
        "parse_request_id",
        "projection_id",
        "routed_parsing_result_id",
        "document_version_id",
        "source_sha256",
        "document",
        "block_count",
        "canonical_blocks_sha256",
        "canonical_blocks",
    }
    try:
        reject_unknown_fields(artifact, allowed, dto="CanonicalEvidenceArtifact")
        if artifact.get("artifact_schema_version") != CANONICAL_EVIDENCE_SCHEMA_VERSION:
            raise ContractValidationError("unsupported canonical artifact schema")
        if artifact.get("contract_schema_version") != CANONICAL_CONTRACT_SCHEMA_VERSION:
            raise ContractValidationError("canonical contract schema mismatch")
        if artifact.get("artifact_role") != CANONICAL_EVIDENCE_ARTIFACT_ROLE:
            raise ContractValidationError("canonical artifact role mismatch")
        if (
            artifact.get("producer_schema_version")
            != ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION
        ):
            raise ContractValidationError("canonical artifact producer mismatch")
        for field_name in (
            "parse_request_id",
            "projection_id",
            "routed_parsing_result_id",
        ):
            require_non_empty(artifact.get(field_name), field_name)
        for field_name in (
            "configuration_sha256",
            "audit_sha256",
            "source_sha256",
            "canonical_blocks_sha256",
        ):
            require_sha256(artifact.get(field_name), field_name)
        serialized_document = require_mapping(artifact.get("document"), "document")
        serialized_blocks = require_sequence(
            artifact.get("canonical_blocks"), "canonical_blocks"
        )
    except (ContractValidationError, TypeError, ValueError) as exc:
        raise DocumentExtractionArtifactError(
            f"canonical artifact envelope is invalid: {exc}"
        ) from exc

    expected_bindings = {
        "configuration_sha256": configuration_sha256,
        "document_version_id": document.document_version_id,
        "source_sha256": document.content_sha256,
        "canonical_blocks_sha256": canonical_blocks_sha256,
        "block_count": len(blocks),
    }
    for field_name, expected in expected_bindings.items():
        if artifact.get(field_name) != expected:
            raise DocumentExtractionBindingError(
                f"canonical artifact {field_name} differs from the backend handoff"
            )
    if dict(serialized_document) != document.to_dict():
        raise DocumentExtractionBindingError(
            "canonical artifact document differs from the backend handoff"
        )
    if len(serialized_blocks) != len(blocks):
        raise DocumentExtractionBindingError(
            "canonical artifact block count differs from the backend handoff"
        )
    for index, (serialized, block) in enumerate(zip(serialized_blocks, blocks)):
        if not isinstance(serialized, Mapping) or dict(serialized) != block.to_dict():
            raise DocumentExtractionBindingError(
                "canonical artifact blocks differ from the backend handoff "
                f"at index {index}"
            )
    return _CanonicalArtifactBinding(
        audit_sha256=str(artifact["audit_sha256"]),
        parse_request_id=str(artifact["parse_request_id"]),
        projection_id=str(artifact["projection_id"]),
        routed_parsing_result_id=str(artifact["routed_parsing_result_id"]),
    )


def _validate_routed_audit_artifact(
    artifact: Mapping[str, object],
    *,
    validator: RoutedAuditValidator,
    binding: _CanonicalArtifactBinding,
    document: DocumentRef,
    blocks: Sequence[CanonicalBlock],
) -> None:
    """Reconstruct and replay the complete routed audit before accepting it."""

    try:
        if artifact.get("schema_version") != ROUTED_PARSING_AUDIT_SCHEMA_VERSION:
            raise ContractValidationError("unsupported routed audit schema")
        audit = validator.validate(artifact)
        if not isinstance(audit, ValidatedRoutedAudit):
            raise ContractValidationError(
                "routed audit validator returned an invalid replay receipt"
            )
    except (ContractValidationError, TypeError, ValueError, RecursionError) as exc:
        raise DocumentExtractionArtifactError(
            f"routed audit artifact is not replayable: {exc}"
        ) from exc

    expected_identity = {
        "projection_id": (audit.projection_id, binding.projection_id),
        "parse_request_id": (audit.parse_request_id, binding.parse_request_id),
        "routed_parsing_result_id": (
            audit.routed_parsing_result_id,
            binding.routed_parsing_result_id,
        ),
    }
    for field_name, (actual, expected) in expected_identity.items():
        if actual != expected:
            raise DocumentExtractionBindingError(
                f"routed audit {field_name} differs from the canonical artifact"
            )
    if audit.document != document:
        raise DocumentExtractionBindingError(
            "routed audit document differs from the canonical artifact"
        )
    if audit.canonical_blocks != tuple(blocks):
        raise DocumentExtractionBindingError(
            "routed audit canonical blocks differ from the canonical artifact"
        )


__all__ = [
    "DocumentExtractionArtifactError",
    "DocumentExtractionBindingError",
    "DocumentExtractionError",
    "DocumentExtractionPolicy",
    "ExtractDocumentUseCase",
]
