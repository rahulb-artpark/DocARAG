"""Immutable publication for evidence-grounded graph releases.

``EvidenceGraphRelease`` is an immutable candidate/audit artifact.  This module
gives it a durable, content-addressed publication boundary; adoption as active
retrieval authority additionally requires an injected external authority bound
to the exact release, checkpoint, and declared selectors.  It never adopts
BookRAG's mutable NetworkX JSON as source data.

Checkpoint bytes are canonical JSON and are published through DocARAG's local
content-addressed store.  That store uses an fsync-ed temporary inode followed
by an atomic, no-overwrite hard link.  Existing bytes are accepted only after
their full SHA-256 digest is verified.  Reads additionally reconstruct the
domain release, which revalidates every anchor, mention, relation and GT-link.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from pathlib import Path
import re
from typing import ClassVar, Protocol, Self

from document_services.persistence.artifact_store import (
    ArtifactStoreError,
    LocalContentAddressedArtifactStore,
)
from document_services.evidence.assets import AssetMaterializationError
from document_services.domain.evidence import AssetKind, AssetRef
from document_services.domain.base import (
    ContractValidationError,
    canonical_json_bytes,
    content_sha256,
    parse_json_object,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
    stable_id,
)
from document_services.domain.knowledge import EvidenceGraphRelease


CHECKPOINT_REFERENCE_SCHEMA_VERSION = "docarag.evidence-graph-checkpoint-ref.v1"
ADOPTION_AUTHORIZATION_SCHEMA_VERSION = (
    "docarag.evidence-graph-adoption-authorization.v1"
)
CHECKPOINT_MEDIA_TYPE = "application/vnd.docarag.evidence-graph-release+json"
_RELEASE_ID = re.compile(r"kgrelease_[0-9a-f]{32}")


class EvidenceGraphCheckpointErrorCode(StrEnum):
    """Stable checkpoint failures suitable for logs and API translation."""

    INVALID_RELEASE = "invalid_release"
    INVALID_REFERENCE = "invalid_reference"
    STORAGE_ERROR = "storage_error"
    DIGEST_MISMATCH = "digest_mismatch"
    BINDING_MISMATCH = "binding_mismatch"
    RELEASE_MISMATCH = "release_mismatch"
    ADOPTION_AUTHORITY_REQUIRED = "adoption_authority_required"
    ADOPTION_AUTHORIZATION_ERROR = "adoption_authorization_error"
    ADOPTION_AUTHORIZATION_MISMATCH = "adoption_authorization_mismatch"


class EvidenceGraphCheckpointError(ValueError):
    """An evidence-graph checkpoint could not be safely used."""

    def __init__(self, code: EvidenceGraphCheckpointErrorCode, message: str) -> None:
        self.code = EvidenceGraphCheckpointErrorCode(code)
        super().__init__(message)


def _fail(code: EvidenceGraphCheckpointErrorCode, message: str) -> None:
    raise EvidenceGraphCheckpointError(code, message)


def _sorted_ids(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    result = tuple(require_non_empty(value, f"{field_name}[]") for value in values)
    if len(result) != len(set(result)):
        raise ContractValidationError(f"{field_name} contains duplicate IDs")
    return tuple(sorted(result))


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceGraphBinding:
    """Exact source/canonical and producer-declared tree/graph/map selectors.

    Equality proves that the release and reference agree; it does not prove the
    declared artifacts contain a complete BookRAG ``T/G/M``.  The external
    adoption authority must enforce the full-manifest and receipt policy.
    """

    document_id: str
    document_version_id: str
    source_sha256: str
    build_spec_sha256: str
    canonical_artifact_sha256: str
    canonical_revision_ids: tuple[str, ...]
    representation_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    tree_revision_id: str
    tree_artifact_sha256: str
    tree_node_ids: tuple[str, ...]
    graph_revision_id: str
    graph_artifact_sha256: str
    mapping_revision_id: str
    mapping_artifact_sha256: str
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "document_id",
            "document_version_id",
            "tree_revision_id",
            "graph_revision_id",
            "mapping_revision_id",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        for field_name in (
            "source_sha256",
            "build_spec_sha256",
            "canonical_artifact_sha256",
            "tree_artifact_sha256",
            "graph_artifact_sha256",
            "mapping_artifact_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        canonical_ids = _sorted_ids(
            self.canonical_revision_ids, "canonical_revision_ids"
        )
        if not canonical_ids:
            raise ContractValidationError("canonical_revision_ids must not be empty")
        representation_ids = _sorted_ids(self.representation_ids, "representation_ids")
        asset_ids = _sorted_ids(self.asset_ids, "asset_ids")
        if not representation_ids and not asset_ids:
            raise ContractValidationError(
                "binding requires at least one representation or asset ID"
            )
        tree_node_ids = _sorted_ids(self.tree_node_ids, "tree_node_ids")
        if not tree_node_ids:
            raise ContractValidationError("tree_node_ids must not be empty")
        object.__setattr__(self, "canonical_revision_ids", canonical_ids)
        object.__setattr__(self, "representation_ids", representation_ids)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "tree_node_ids", tree_node_ids)
        object.__setattr__(self, "binding_sha256", content_sha256(self._content_dict()))

    @classmethod
    def from_release(cls, release: EvidenceGraphRelease) -> Self:
        if not isinstance(release, EvidenceGraphRelease):
            raise ContractValidationError("release must be an EvidenceGraphRelease")
        return cls(
            document_id=release.document_id,
            document_version_id=release.document_version_id,
            source_sha256=release.source_sha256,
            build_spec_sha256=release.build_spec_sha256,
            canonical_artifact_sha256=release.canonical_artifact_sha256,
            canonical_revision_ids=release.canonical_revision_ids,
            representation_ids=release.representation_ids,
            asset_ids=release.asset_ids,
            tree_revision_id=release.tree_revision_id,
            tree_artifact_sha256=release.tree_artifact_sha256,
            tree_node_ids=release.tree_node_ids,
            graph_revision_id=release.graph_revision_id,
            graph_artifact_sha256=release.graph_artifact_sha256,
            mapping_revision_id=release.mapping_revision_id,
            mapping_artifact_sha256=release.mapping_artifact_sha256,
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "build_spec_sha256": self.build_spec_sha256,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "representation_ids": list(self.representation_ids),
            "asset_ids": list(self.asset_ids),
            "tree_revision_id": self.tree_revision_id,
            "tree_artifact_sha256": self.tree_artifact_sha256,
            "tree_node_ids": list(self.tree_node_ids),
            "graph_revision_id": self.graph_revision_id,
            "graph_artifact_sha256": self.graph_artifact_sha256,
            "mapping_revision_id": self.mapping_revision_id,
            "mapping_artifact_sha256": self.mapping_artifact_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        allowed = {
            "document_id",
            "document_version_id",
            "source_sha256",
            "build_spec_sha256",
            "canonical_artifact_sha256",
            "canonical_revision_ids",
            "representation_ids",
            "asset_ids",
            "tree_revision_id",
            "tree_artifact_sha256",
            "tree_node_ids",
            "graph_revision_id",
            "graph_artifact_sha256",
            "mapping_revision_id",
            "mapping_artifact_sha256",
            "binding_sha256",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ContractValidationError(
                f"EvidenceGraphBinding has unknown field(s): {', '.join(unknown)}"
            )
        result = cls(
            document_id=payload.get("document_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            build_spec_sha256=payload.get("build_spec_sha256"),  # type: ignore[arg-type]
            canonical_artifact_sha256=payload.get("canonical_artifact_sha256"),  # type: ignore[arg-type]
            canonical_revision_ids=tuple(
                require_sequence(
                    payload.get("canonical_revision_ids", []),
                    "canonical_revision_ids",
                )
            ),  # type: ignore[arg-type]
            representation_ids=tuple(
                require_sequence(payload.get("representation_ids", []), "representation_ids")
            ),  # type: ignore[arg-type]
            asset_ids=tuple(
                require_sequence(payload.get("asset_ids", []), "asset_ids")
            ),  # type: ignore[arg-type]
            tree_revision_id=payload.get("tree_revision_id"),  # type: ignore[arg-type]
            tree_artifact_sha256=payload.get("tree_artifact_sha256"),  # type: ignore[arg-type]
            tree_node_ids=tuple(
                require_sequence(payload.get("tree_node_ids", []), "tree_node_ids")
            ),  # type: ignore[arg-type]
            graph_revision_id=payload.get("graph_revision_id"),  # type: ignore[arg-type]
            graph_artifact_sha256=payload.get("graph_artifact_sha256"),  # type: ignore[arg-type]
            mapping_revision_id=payload.get("mapping_revision_id"),  # type: ignore[arg-type]
            mapping_artifact_sha256=payload.get("mapping_artifact_sha256"),  # type: ignore[arg-type]
        )
        if payload.get("binding_sha256") != result.binding_sha256:
            raise ContractValidationError("binding_sha256 does not match binding content")
        return result


@dataclass(frozen=True, slots=True)
class EvidenceGraphCheckpointRef:
    """Portable adoption token for one exact immutable checkpoint."""

    release_id: str
    release_content_sha256: str
    checkpoint_sha256: str
    size_bytes: int
    binding: EvidenceGraphBinding
    reference_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CHECKPOINT_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or _RELEASE_ID.fullmatch(
            self.release_id
        ) is None:
            raise ContractValidationError("release_id is not a canonical release ID")
        require_sha256(self.release_content_sha256, "release_content_sha256")
        require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _positive_integer(self.size_bytes, "size_bytes")
        if not isinstance(self.binding, EvidenceGraphBinding):
            raise ContractValidationError("binding must be an EvidenceGraphBinding")
        object.__setattr__(
            self,
            "reference_id",
            stable_id(
                "kgcheckpoint",
                self.release_id,
                self.release_content_sha256,
                self.checkpoint_sha256,
                self.size_bytes,
                self.binding.binding_sha256,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "reference_id": self.reference_id,
            "release_id": self.release_id,
            "release_content_sha256": self.release_content_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "size_bytes": self.size_bytes,
            "binding": self.binding.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        allowed = {
            "schema_version",
            "reference_id",
            "release_id",
            "release_content_sha256",
            "checkpoint_sha256",
            "size_bytes",
            "binding",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ContractValidationError(
                f"EvidenceGraphCheckpointRef has unknown field(s): {', '.join(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ContractValidationError(
                "unsupported evidence-graph checkpoint reference schema"
            )
        result = cls(
            release_id=payload.get("release_id"),  # type: ignore[arg-type]
            release_content_sha256=payload.get("release_content_sha256"),  # type: ignore[arg-type]
            checkpoint_sha256=payload.get("checkpoint_sha256"),  # type: ignore[arg-type]
            size_bytes=payload.get("size_bytes"),  # type: ignore[arg-type]
            binding=EvidenceGraphBinding.from_dict(
                require_mapping(payload.get("binding"), "binding")
            ),
        )
        if payload.get("reference_id") != result.reference_id:
            raise ContractValidationError("reference_id does not match reference content")
        return result

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


@dataclass(frozen=True, slots=True)
class EvidenceGraphAdoptionAuthorization:
    """An external authority's decision for one exact checkpoint binding.

    ``EvidenceGraphCheckpointStore`` accepts this value only as the direct
    result of its injected authority.  It is not a bearer token accepted from
    an ``adopt`` caller.
    """

    authority_id: str
    release_id: str
    release_content_sha256: str
    checkpoint_sha256: str
    binding_sha256: str
    authorization_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = ADOPTION_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.authority_id, "authority_id")
        if not isinstance(self.release_id, str) or _RELEASE_ID.fullmatch(
            self.release_id
        ) is None:
            raise ContractValidationError("release_id is not a canonical release ID")
        require_sha256(self.release_content_sha256, "release_content_sha256")
        require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        require_sha256(self.binding_sha256, "binding_sha256")
        object.__setattr__(
            self,
            "authorization_id",
            stable_id(
                "kgadoption",
                self.authority_id,
                self.release_id,
                self.release_content_sha256,
                self.checkpoint_sha256,
                self.binding_sha256,
            ),
        )

    @classmethod
    def from_reference(
        cls,
        reference: EvidenceGraphCheckpointRef,
        *,
        authority_id: str,
    ) -> Self:
        if not isinstance(reference, EvidenceGraphCheckpointRef):
            raise ContractValidationError(
                "reference must be an EvidenceGraphCheckpointRef"
            )
        return cls(
            authority_id=authority_id,
            release_id=reference.release_id,
            release_content_sha256=reference.release_content_sha256,
            checkpoint_sha256=reference.checkpoint_sha256,
            binding_sha256=reference.binding.binding_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "authorization_id": self.authorization_id,
            "authority_id": self.authority_id,
            "release_id": self.release_id,
            "release_content_sha256": self.release_content_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        allowed = {
            "schema_version",
            "authorization_id",
            "authority_id",
            "release_id",
            "release_content_sha256",
            "checkpoint_sha256",
            "binding_sha256",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ContractValidationError(
                "EvidenceGraphAdoptionAuthorization has unknown field(s): "
                f"{', '.join(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ContractValidationError(
                "unsupported evidence-graph adoption authorization schema"
            )
        result = cls(
            authority_id=payload.get("authority_id"),  # type: ignore[arg-type]
            release_id=payload.get("release_id"),  # type: ignore[arg-type]
            release_content_sha256=payload.get("release_content_sha256"),  # type: ignore[arg-type]
            checkpoint_sha256=payload.get("checkpoint_sha256"),  # type: ignore[arg-type]
            binding_sha256=payload.get("binding_sha256"),  # type: ignore[arg-type]
        )
        if payload.get("authorization_id") != result.authorization_id:
            raise ContractValidationError(
                "authorization_id does not match authorization content"
            )
        return result

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


class EvidenceGraphAdoptionAuthority(Protocol):
    """Trusted external decision source injected into a checkpoint store."""

    def authorization_for(
        self, reference: EvidenceGraphCheckpointRef
    ) -> EvidenceGraphAdoptionAuthorization | None:
        """Return exact authorization, or ``None`` to deny adoption."""

        ...


@dataclass(frozen=True, slots=True)
class PublishedEvidenceGraphCheckpoint:
    """Result of a verified idempotent publication."""

    reference: EvidenceGraphCheckpointRef
    uri: str
    path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.to_dict(),
            "uri": self.uri,
            "path": str(self.path),
        }


class EvidenceGraphCheckpointStore:
    """Publish and read exact evidence-graph releases through immutable CAS."""

    __slots__ = (
        "_adoption_authority",
        "_store",
        "_resolver",
        "max_checkpoint_bytes",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        create: bool = True,
        max_checkpoint_bytes: int = 128 * 1024 * 1024,
        adoption_authority: EvidenceGraphAdoptionAuthority | None = None,
    ) -> None:
        self.max_checkpoint_bytes = _positive_integer(
            max_checkpoint_bytes, "max_checkpoint_bytes"
        )
        if adoption_authority is not None and not callable(
            getattr(adoption_authority, "authorization_for", None)
        ):
            _fail(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_ERROR,
                "adoption_authority must provide authorization_for(reference)",
            )
        self._adoption_authority = adoption_authority
        try:
            self._store = LocalContentAddressedArtifactStore(
                root,
                create=create,
                max_artifact_bytes=self.max_checkpoint_bytes,
            )
            self._resolver = self._store.make_asset_resolver(
                max_assets=1,
                max_asset_bytes=self.max_checkpoint_bytes,
                max_total_bytes=self.max_checkpoint_bytes,
            )
        except (ArtifactStoreError, AssetMaterializationError) as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.STORAGE_ERROR,
                f"could not initialize evidence-graph checkpoint store: {exc}",
            ) from exc

    @staticmethod
    def _validated_release(release: EvidenceGraphRelease) -> EvidenceGraphRelease:
        if not isinstance(release, EvidenceGraphRelease):
            _fail(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                "release must be an EvidenceGraphRelease",
            )
        try:
            # Reconstruct at the boundary rather than trusting object provenance.
            return EvidenceGraphRelease.from_dict(release.to_dict())
        except (ContractValidationError, OverflowError, TypeError, ValueError) as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                f"evidence-graph release failed validation: {exc}",
            ) from exc

    @classmethod
    def reference_for(
        cls, release: EvidenceGraphRelease
    ) -> EvidenceGraphCheckpointRef:
        validated = cls._validated_release(release)
        encoded = canonical_json_bytes(validated.to_dict())
        return cls._reference_for_validated(validated, encoded)

    @staticmethod
    def _reference_for_validated(
        release: EvidenceGraphRelease,
        canonical_encoded: bytes,
    ) -> EvidenceGraphCheckpointRef:
        """Build a reference without cloning or encoding a validated release again."""

        return EvidenceGraphCheckpointRef(
            release_id=release.release_id,
            release_content_sha256=release.content_sha256,
            checkpoint_sha256=hashlib.sha256(canonical_encoded).hexdigest(),
            size_bytes=len(canonical_encoded),
            binding=EvidenceGraphBinding.from_release(release),
        )

    def publish(
        self, release: EvidenceGraphRelease
    ) -> PublishedEvidenceGraphCheckpoint:
        """Atomically publish once, or reuse only byte-identical content."""

        validated = self._validated_release(release)
        encoded = canonical_json_bytes(validated.to_dict())
        if len(encoded) > self.max_checkpoint_bytes:
            _fail(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                "evidence-graph checkpoint exceeds max_checkpoint_bytes before publication",
            )
        reference = self._reference_for_validated(validated, encoded)
        try:
            stored = self._store.publish(
                encoded,
                expected_sha256=reference.checkpoint_sha256,
                media_type=CHECKPOINT_MEDIA_TYPE,
            )
        except ArtifactStoreError as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.STORAGE_ERROR,
                f"could not publish evidence-graph checkpoint: {exc}",
            ) from exc
        if stored.sha256 != reference.checkpoint_sha256 or stored.size_bytes != len(
            encoded
        ):
            _fail(
                EvidenceGraphCheckpointErrorCode.DIGEST_MISMATCH,
                "published checkpoint identity differs from canonical release bytes",
            )
        restored = self.read(reference)
        if restored.to_dict() != validated.to_dict():
            _fail(
                EvidenceGraphCheckpointErrorCode.RELEASE_MISMATCH,
                "published checkpoint did not round-trip to the exact release",
            )
        return PublishedEvidenceGraphCheckpoint(
            reference=reference,
            uri=stored.uri,
            path=stored.path,
        )

    def read(self, reference: EvidenceGraphCheckpointRef) -> EvidenceGraphRelease:
        """Read only when bytes, generated IDs and all release bindings verify."""

        if not isinstance(reference, EvidenceGraphCheckpointRef):
            _fail(
                EvidenceGraphCheckpointErrorCode.INVALID_REFERENCE,
                "reference must be an EvidenceGraphCheckpointRef",
            )
        asset = AssetRef(
            kind=AssetKind.SOURCE,
            uri=f"cas://sha256/{reference.checkpoint_sha256}",
            media_type=CHECKPOINT_MEDIA_TYPE,
            sha256=reference.checkpoint_sha256,
            metadata={"role": "evidence_graph_checkpoint"},
        )
        try:
            material = self._resolver.resolve(asset, include_bytes=True)
        except AssetMaterializationError as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.STORAGE_ERROR,
                f"could not read evidence-graph checkpoint: {exc}",
            ) from exc
        if material.data is None:  # pragma: no cover - resolver contract guard
            _fail(
                EvidenceGraphCheckpointErrorCode.STORAGE_ERROR,
                "checkpoint resolver returned no bytes",
            )
        if material.size_bytes != reference.size_bytes:
            _fail(
                EvidenceGraphCheckpointErrorCode.DIGEST_MISMATCH,
                "checkpoint size differs from its adoption reference",
            )
        if hashlib.sha256(material.data).hexdigest() != reference.checkpoint_sha256:
            _fail(
                EvidenceGraphCheckpointErrorCode.DIGEST_MISMATCH,
                "checkpoint bytes differ from checkpoint_sha256",
            )
        try:
            release = EvidenceGraphRelease.from_json(material.data)
            canonical_encoded = canonical_json_bytes(release.to_dict())
            actual_reference = self._reference_for_validated(
                release, canonical_encoded
            )
        except (
            ContractValidationError,
            EvidenceGraphCheckpointError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                f"checkpoint does not contain a valid evidence-graph release: {exc}",
            ) from exc
        if actual_reference.checkpoint_sha256 != reference.checkpoint_sha256:
            _fail(
                EvidenceGraphCheckpointErrorCode.DIGEST_MISMATCH,
                "checkpoint canonical JSON digest differs from its reference",
            )
        if (
            actual_reference.release_id != reference.release_id
            or actual_reference.release_content_sha256
            != reference.release_content_sha256
        ):
            _fail(
                EvidenceGraphCheckpointErrorCode.RELEASE_MISMATCH,
                "checkpoint release identity differs from its adoption reference",
            )
        if actual_reference.binding != reference.binding:
            _fail(
                EvidenceGraphCheckpointErrorCode.BINDING_MISMATCH,
                "checkpoint source/canonical/tree/graph/map binding selectors "
                "differ from its reference",
            )
        if actual_reference != reference:
            _fail(
                EvidenceGraphCheckpointErrorCode.INVALID_REFERENCE,
                "checkpoint reference contains inconsistent derived fields",
            )
        return release

    def adopt(
        self,
        reference: EvidenceGraphCheckpointRef,
        *,
        expected_release: EvidenceGraphRelease,
    ) -> EvidenceGraphRelease:
        """Adopt an exact release only when the external authority permits it."""

        expected = self._validated_release(expected_release)
        builder_metadata = expected.metadata.get("builder")
        if (
            isinstance(builder_metadata, Mapping)
            and builder_metadata.get("authority_eligible") is False
        ):
            reason = builder_metadata.get("authority_ineligibility_reason")
            suffix = f": {reason}" if isinstance(reason, str) and reason else ""
            _fail(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                "evidence-graph release is inspection-only and cannot be adopted "
                f"as retrieval authority{suffix}",
            )
        expected_reference = self.reference_for(expected)
        if reference != expected_reference:
            _fail(
                EvidenceGraphCheckpointErrorCode.RELEASE_MISMATCH,
                "checkpoint reference does not identify the exact expected release",
            )
        if self._adoption_authority is None:
            _fail(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORITY_REQUIRED,
                "checkpoint adoption requires an external adoption authority",
            )
        try:
            authorization = self._adoption_authority.authorization_for(reference)
        except Exception as exc:
            raise EvidenceGraphCheckpointError(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_ERROR,
                f"external adoption authority failed: {type(exc).__name__}",
            ) from exc
        if not isinstance(authorization, EvidenceGraphAdoptionAuthorization):
            _fail(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_ERROR,
                "external adoption authority did not authorize this checkpoint",
            )
        if (
            authorization.release_id != reference.release_id
            or authorization.release_content_sha256
            != reference.release_content_sha256
            or authorization.checkpoint_sha256 != reference.checkpoint_sha256
            or authorization.binding_sha256 != reference.binding.binding_sha256
        ):
            _fail(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_MISMATCH,
                "external adoption authorization does not match the exact "
                "release/checkpoint/binding",
            )
        restored = self.read(reference)
        if restored.to_dict() != expected.to_dict():
            _fail(
                EvidenceGraphCheckpointErrorCode.RELEASE_MISMATCH,
                "checkpoint content does not equal the exact expected release",
            )
        return restored


__all__ = [
    "ADOPTION_AUTHORIZATION_SCHEMA_VERSION",
    "CHECKPOINT_MEDIA_TYPE",
    "CHECKPOINT_REFERENCE_SCHEMA_VERSION",
    "EvidenceGraphAdoptionAuthority",
    "EvidenceGraphAdoptionAuthorization",
    "EvidenceGraphBinding",
    "EvidenceGraphCheckpointError",
    "EvidenceGraphCheckpointErrorCode",
    "EvidenceGraphCheckpointRef",
    "EvidenceGraphCheckpointStore",
    "PublishedEvidenceGraphCheckpoint",
]
