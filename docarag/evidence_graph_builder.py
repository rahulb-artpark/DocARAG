"""Build an evidence-grounded graph release from BookRAG extraction checkpoints.

This module is the deliberately small compatibility boundary between the
dependency-light DocARAG contracts and BookRAG's legacy per-tree-node KG
checkpoint format.  It does not import BookRAG/Core, model SDKs, pydantic, or a
graph library.

The builder preserves checkpoint rows as extraction assertions.  It does not
perform entity resolution or graph refinement.  Relationship endpoints are
resolved only against the entity rows in the same checkpoint, using exact
legacy names.  Missing or ambiguous endpoints fail closed, which avoids an
accidental all-pairs name search and prevents evidence from being attached to
the wrong tree node.

Expected complexity is linear in canonical bytes/materials, tree descriptors,
checkpoint bytes/assertions and G-to-T links, plus
``O((A + E + R + L) log (A + E + R + L))`` for deterministic release ordering.
Exact byte ceilings are checked for canonical/tree semantic snapshots, selected
CAS assets, each and all legacy checkpoints, and the final publication
envelope.  No all-pairs operation is used.  Legacy checkpoints still provide
only coarse whole-node evidence; releases from this builder are therefore
inspection-only until an exact claim/material receipt path replaces them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import math
from typing import ClassVar, Protocol

from docarag.domain.base import (
    canonical_json_bytes,
    ContractValidationError,
    JsonObject,
    content_sha256,
    freeze_json,
    json_ready,
    parse_json_object,
    require_non_empty,
    require_sha256,
    stable_id,
)
from docarag.domain.evidence import (
    AssetRef,
    CanonicalBlock,
    DocumentRef,
    Representation,
)
from docarag.domain.knowledge import (
    EntityMention,
    EvidenceAnchor,
    EvidenceGraphRelease,
    GTLink,
    RelationAssertion,
)


BUILDER_SCHEMA_VERSION = "docarag.evidence-graph-builder.v1"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "docarag.kg-extraction-checkpoint.v1"
LEGACY_UNSPECIFIED_RELATION_TYPE = "legacy_unspecified"
DEFAULT_MAX_RELEASE_BYTES = 128 * 1024 * 1024
_MAX_SELECTIONS_PER_NODE = 4096
_ABSOLUTE_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_ABSOLUTE_MAX_SELECTED_ASSET_RECEIPTS = 400_000
_MAX_ARTIFACT_CHUNKS = 1_048_576
_ABSOLUTE_MAX_SEMANTIC_ARTIFACT_BYTES = 512 * 1024 * 1024
_ABSOLUTE_MAX_SEMANTIC_RECORD_BYTES = 64 * 1024 * 1024
_ABSOLUTE_MAX_SEMANTIC_LABEL_BYTES = 1024 * 1024
_MAX_ABSOLUTE_RELATION_WEIGHT = 1_000_000_000_000
_RELEASE_BUDGET_RESERVE_BYTES = 64 * 1024
_MIN_MENTION_RELEASE_BYTES = 192
_MIN_RELATION_RELEASE_BYTES = 320
_MIN_LINK_RELEASE_BYTES = 224


class EvidenceGraphBuildErrorCode(StrEnum):
    """Stable error categories suitable for logs and API translation."""

    INVALID_INPUT = "invalid_input"
    LIMIT_EXCEEDED = "limit_exceeded"
    DOCUMENT_BINDING = "document_binding"
    TREE_BINDING = "tree_binding"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    MATERIAL_BINDING = "material_binding"
    CHECKPOINT_INTEGRITY = "checkpoint_integrity"
    CHECKPOINT_BINDING = "checkpoint_binding"
    INVALID_ASSERTION = "invalid_assertion"
    AMBIGUOUS_ENDPOINT = "ambiguous_endpoint"
    MISSING_ENDPOINT = "missing_endpoint"


class EvidenceGraphBuildError(ContractValidationError):
    """A release could not be built without guessing or losing evidence."""

    def __init__(self, code: EvidenceGraphBuildErrorCode, message: str) -> None:
        self.code = EvidenceGraphBuildErrorCode(code)
        super().__init__(f"{self.code.value}: {message}")


def _error(code: EvidenceGraphBuildErrorCode, message: str) -> None:
    raise EvidenceGraphBuildError(code, message)


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            f"{field_name} must be a positive integer",
        )
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            f"{field_name} must be a non-negative integer",
        )
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
            f"{field_name} must be an array",
        )
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
            f"{field_name} must be an object",
        )
    if any(not isinstance(key, str) for key in value):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
            f"{field_name} must use string keys",
        )
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    field_name: str,
    *,
    code: EvidenceGraphBuildErrorCode = EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        _error(
            code,
            f"{field_name} has invalid fields ({'; '.join(details)})",
        )


def _required_text(value: object, field_name: str) -> str:
    try:
        return require_non_empty(value, field_name)
    except ContractValidationError as exc:
        _error(EvidenceGraphBuildErrorCode.INVALID_ASSERTION, str(exc))


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _checked_sha256(value: object, field_name: str) -> str:
    try:
        return require_sha256(value, field_name)
    except ContractValidationError as exc:
        _error(EvidenceGraphBuildErrorCode.INVALID_INPUT, str(exc))


@dataclass(frozen=True, slots=True)
class EvidenceGraphBuildLimits:
    """Hard resource ceilings applied before and during materialization."""

    max_canonical_blocks: int = 250_000
    max_canonical_input_bytes: int = 128 * 1024 * 1024
    max_canonical_artifact_bytes: int = 128 * 1024 * 1024
    max_tree_artifact_bytes: int = 128 * 1024 * 1024
    max_selected_assets: int = 100_000
    max_selected_asset_bytes: int = 256 * 1024 * 1024
    max_total_selected_asset_bytes: int = 2 * 1024 * 1024 * 1024
    max_materials: int = 250_000
    max_nodes: int = 100_000
    max_selections: int = 400_000
    max_checkpoints: int = 100_000
    max_checkpoint_bytes: int = 64 * 1024 * 1024
    max_total_checkpoint_bytes: int = 128 * 1024 * 1024
    max_anchors: int = 400_000
    max_mentions: int = 1_000_000
    max_relations: int = 2_000_000
    max_links: int = 3_000_000
    max_tree_descriptor_bytes: int = 64 * 1024 * 1024
    max_release_bytes: int = DEFAULT_MAX_RELEASE_BYTES

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "max_canonical_blocks",
            "max_canonical_input_bytes",
            "max_canonical_artifact_bytes",
            "max_tree_artifact_bytes",
            "max_selected_assets",
            "max_selected_asset_bytes",
            "max_total_selected_asset_bytes",
            "max_materials",
            "max_nodes",
            "max_selections",
            "max_checkpoints",
            "max_checkpoint_bytes",
            "max_total_checkpoint_bytes",
            "max_anchors",
            "max_mentions",
            "max_relations",
            "max_links",
            "max_tree_descriptor_bytes",
            "max_release_bytes",
        ):
            _positive_integer(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_canonical_blocks": self.max_canonical_blocks,
            "max_canonical_input_bytes": self.max_canonical_input_bytes,
            "max_canonical_artifact_bytes": self.max_canonical_artifact_bytes,
            "max_tree_artifact_bytes": self.max_tree_artifact_bytes,
            "max_selected_assets": self.max_selected_assets,
            "max_selected_asset_bytes": self.max_selected_asset_bytes,
            "max_total_selected_asset_bytes": self.max_total_selected_asset_bytes,
            "max_materials": self.max_materials,
            "max_nodes": self.max_nodes,
            "max_selections": self.max_selections,
            "max_checkpoints": self.max_checkpoints,
            "max_checkpoint_bytes": self.max_checkpoint_bytes,
            "max_total_checkpoint_bytes": self.max_total_checkpoint_bytes,
            "max_anchors": self.max_anchors,
            "max_mentions": self.max_mentions,
            "max_relations": self.max_relations,
            "max_links": self.max_links,
            "max_tree_descriptor_bytes": self.max_tree_descriptor_bytes,
            "max_release_bytes": self.max_release_bytes,
        }


@dataclass(frozen=True, slots=True)
class ImmutableArtifactRef:
    """A bounded content-addressed artifact that the builder must read itself."""

    uri: str
    sha256: str
    size_bytes: int

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        digest = _checked_sha256(self.sha256, "artifact.sha256")
        expected_uri = f"cas://sha256/{digest}"
        if self.uri != expected_uri:
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "artifact.uri must be the digest-consistent CAS URI",
            )
        _positive_integer(self.size_bytes, "artifact.size_bytes")


@dataclass(frozen=True, slots=True)
class SelectedAssetMaterializationReceipt:
    """Exact immutable bytes selected by BookRAG for one canonical asset."""

    asset_id: str
    artifact: ImmutableArtifactRef
    media_type: str | None = None

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.asset_id, "selected_asset_receipt.asset_id")
        if not isinstance(self.artifact, ImmutableArtifactRef):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selected_asset_receipt.artifact must be ImmutableArtifactRef",
            )
        if self.media_type is not None:
            _required_text(self.media_type, "selected_asset_receipt.media_type")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "asset_id": self.asset_id,
            "artifact": {
                "uri": self.artifact.uri,
                "sha256": self.artifact.sha256,
                "size_bytes": self.artifact.size_bytes,
            },
            "media_type": self.media_type,
        }


class ArtifactByteReader(Protocol):
    """Dependency-neutral port for bounded reads from an immutable store."""

    def iter_bytes(
        self,
        artifact: ImmutableArtifactRef,
        *,
        max_bytes: int,
    ) -> Iterable[bytes | bytearray | memoryview]: ...


@dataclass(frozen=True, slots=True)
class EvidenceGraphReleaseBinding:
    """Verified canonical-input and tree-evidence-descriptor selectors.

    These references are not arbitrary upstream files.  They address the exact
    deterministic length-framed byte streams returned by
    ``iter_canonical_input_artifact_bytes`` and
    ``iter_tree_descriptor_artifact_bytes``.  The original PDF, parser-native
    artifacts, the full BookRAG tree, and the legacy BookRAG pickle remain
    separately traceable inputs.  In particular, the tree descriptor stream
    binds selected evidence and checkpoint identities, not parent/child
    topology, sibling order, summaries, or complete tree-node content.  These
    snapshots are therefore bounded candidate-build inputs, not sufficient
    proof of an authoritative BookRAG ``T``.  ``build`` both recomputes their
    semantics and re-reads their CAS bytes, preventing an unrelated hash-valid
    object from being paired with a caller-supplied semantic digest.
    """

    build_spec_sha256: str
    canonical_artifact: ImmutableArtifactRef
    canonical_input_sha256: str
    tree_revision_id: str
    tree_artifact: ImmutableArtifactRef
    tree_descriptor_sha256: str
    selected_asset_receipts: tuple[SelectedAssetMaterializationReceipt, ...] = ()
    metadata: JsonObject = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _checked_sha256(self.build_spec_sha256, "build_spec_sha256")
        if not isinstance(self.canonical_artifact, ImmutableArtifactRef):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "canonical_artifact must be ImmutableArtifactRef",
            )
        _checked_sha256(self.canonical_input_sha256, "canonical_input_sha256")
        _required_text(self.tree_revision_id, "tree_revision_id")
        if not isinstance(self.tree_artifact, ImmutableArtifactRef):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "tree_artifact must be ImmutableArtifactRef",
            )
        _checked_sha256(self.tree_descriptor_sha256, "tree_descriptor_sha256")
        if not isinstance(self.selected_asset_receipts, Sequence) or isinstance(
            self.selected_asset_receipts, (str, bytes, bytearray)
        ):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selected_asset_receipts must be an ordered array",
            )
        receipts = tuple(self.selected_asset_receipts)
        if len(receipts) > _ABSOLUTE_MAX_SELECTED_ASSET_RECEIPTS:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                "selected_asset_receipts exceeds the absolute input ceiling",
            )
        if any(
            not isinstance(item, SelectedAssetMaterializationReceipt)
            for item in receipts
        ):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selected_asset_receipts must contain SelectedAssetMaterializationReceipt values",
            )
        receipt_ids = [item.asset_id for item in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selected_asset_receipts contains duplicate asset IDs",
            )
        try:
            metadata = freeze_json(self.metadata, field="binding.metadata")
        except ContractValidationError as exc:
            _error(EvidenceGraphBuildErrorCode.INVALID_INPUT, str(exc))
        if not isinstance(metadata, Mapping):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "binding.metadata must be an object",
            )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "selected_asset_receipts",
            tuple(sorted(receipts, key=lambda item: item.asset_id)),
        )

    @property
    def canonical_artifact_sha256(self) -> str:
        return self.canonical_artifact.sha256

    @property
    def tree_artifact_sha256(self) -> str:
        return self.tree_artifact.sha256


@dataclass(frozen=True, slots=True)
class CanonicalMaterialSelection:
    """One explicitly selected canonical locus used to extract a tree node.

    Exactly one material ID is required.  Legacy KG checkpoints attest only the
    whole BookRAG-selected material.  Version 1 therefore rejects finer spans,
    cells, boxes, or diagram elements instead of accepting caller-invented
    provenance.
    """

    canonical_revision_id: str
    representation_id: str | None = None
    asset_id: str | None = None
    char_span: tuple[int, int] | None = None
    table_cell: tuple[int, int] | None = None
    bbox: tuple[float, float, float, float] | None = None
    diagram_element_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.canonical_revision_id, "canonical_revision_id")
        representation_id = _optional_text(
            self.representation_id, "representation_id"
        )
        asset_id = _optional_text(self.asset_id, "asset_id")
        if (representation_id is None) == (asset_id is None):
            _error(
                EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                "a material selection requires exactly one representation_id or asset_id",
            )
        if any(
            value is not None
            for value in (
                self.char_span,
                self.table_cell,
                self.bbox,
                self.diagram_element_id,
            )
        ):
            _error(
                EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                "builder v1 supports whole selected materials only; fine locators "
                "are not attested by the legacy KG checkpoint",
            )
        try:
            metadata = freeze_json(self.metadata, field="selection.metadata")
        except ContractValidationError as exc:
            _error(EvidenceGraphBuildErrorCode.INVALID_INPUT, str(exc))
        if not isinstance(metadata, Mapping):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selection.metadata must be an object",
            )
        object.__setattr__(self, "representation_id", representation_id)
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "metadata", metadata)

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "canonical_revision_id": self.canonical_revision_id,
            "representation_id": self.representation_id,
            "asset_id": self.asset_id,
            "char_span": list(self.char_span) if self.char_span else None,
            "table_cell": list(self.table_cell) if self.table_cell else None,
            "bbox": list(self.bbox) if self.bbox else None,
            "diagram_element_id": self.diagram_element_id,
            "metadata": json_ready(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TreeNodeEvidenceDescriptor:
    """Dependency-free view of one evidence-bearing BookRAG tree node."""

    tree_node_id: str
    legacy_node_index: int
    node_type: str
    checkpoint_kind: str
    expected_checkpoint_identity_sha256: str
    selections: tuple[CanonicalMaterialSelection, ...]
    selected_representation_ids: tuple[str, ...]
    selected_asset_ids: tuple[str, ...]

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.tree_node_id, "tree_node_id")
        _non_negative_integer(self.legacy_node_index, "legacy_node_index")
        node_type = _required_text(self.node_type, "node_type")
        if node_type not in {"text", "image", "table", "equation", "title"}:
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "node_type must be text, image, table, equation, or title",
            )
        checkpoint_kind = _required_text(self.checkpoint_kind, "checkpoint_kind")
        if checkpoint_kind not in {"node", "title"}:
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "checkpoint_kind must be node or title",
            )
        if checkpoint_kind == "title" and node_type != "title":
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "title checkpoints require node_type='title'",
            )
        _checked_sha256(
            self.expected_checkpoint_identity_sha256,
            "expected_checkpoint_identity_sha256",
        )
        if not isinstance(self.selections, Sequence) or isinstance(
            self.selections, (str, bytes, bytearray)
        ):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selections must be an array",
            )
        if len(self.selections) > _MAX_SELECTIONS_PER_NODE:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                f"one tree node may select at most {_MAX_SELECTIONS_PER_NODE} materials",
            )
        selections = tuple(self.selections)
        if not selections:
            _error(
                EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                f"tree node {self.tree_node_id!r} requires a selected material",
            )
        if any(not isinstance(item, CanonicalMaterialSelection) for item in selections):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "selections must contain CanonicalMaterialSelection values",
            )
        identities = [content_sha256(item.identity_dict()) for item in selections]
        if len(identities) != len(set(identities)):
            _error(
                EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                f"tree node {self.tree_node_id!r} contains duplicate material selections",
            )
        for field_name in ("selected_representation_ids", "selected_asset_ids"):
            raw_ids = getattr(self, field_name)
            if not isinstance(raw_ids, Sequence) or isinstance(
                raw_ids, (str, bytes, bytearray)
            ):
                _error(
                    EvidenceGraphBuildErrorCode.TREE_BINDING,
                    f"{field_name} must be an ordered array",
                )
            values = tuple(_required_text(value, f"{field_name}[]") for value in raw_ids)
            if len(values) != len(set(values)):
                _error(
                    EvidenceGraphBuildErrorCode.TREE_BINDING,
                    f"{field_name} contains duplicate IDs",
                )
            object.__setattr__(self, field_name, values)
        selection_representation_ids = {
            item.representation_id
            for item in selections
            if item.representation_id is not None
        }
        selection_asset_ids = {
            item.asset_id for item in selections if item.asset_id is not None
        }
        if set(self.selected_representation_ids) != selection_representation_ids:
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "selected_representation_ids must identify exactly the representation selections",
            )
        if set(self.selected_asset_ids) != selection_asset_ids:
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "selected_asset_ids must identify exactly the asset selections",
            )
        object.__setattr__(self, "node_type", node_type)
        object.__setattr__(self, "checkpoint_kind", checkpoint_kind)
        object.__setattr__(self, "selections", selections)

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "tree_node_id": self.tree_node_id,
            "legacy_node_index": self.legacy_node_index,
            "node_type": self.node_type,
            "checkpoint_kind": self.checkpoint_kind,
            "expected_checkpoint_identity_sha256": self.expected_checkpoint_identity_sha256,
            "selections": [item.identity_dict() for item in self.selections],
            "selected_representation_ids": list(self.selected_representation_ids),
            "selected_asset_ids": list(self.selected_asset_ids),
        }


@dataclass(frozen=True, slots=True)
class LegacyKGCheckpointInput:
    """Bounded encoded checkpoint input assigned to an opaque T node.

    Keeping the envelope encoded lets the builder reject per-checkpoint and
    aggregate byte limits before JSON parsing or recursive freezing duplicates
    attacker-controlled structures in memory.
    """

    tree_node_id: str
    encoded_envelope: bytes

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.tree_node_id, "tree_node_id")
        if not isinstance(self.encoded_envelope, bytes):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "checkpoint.encoded_envelope must be bytes",
            )
        if not self.encoded_envelope:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "checkpoint.encoded_envelope must not be empty",
            )
        if len(self.encoded_envelope) > _ABSOLUTE_MAX_CHECKPOINT_BYTES:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                "checkpoint bytes exceed the absolute input ceiling",
            )

    @classmethod
    def from_mapping(
        cls,
        *,
        tree_node_id: str,
        envelope: Mapping[str, object],
    ) -> "LegacyKGCheckpointInput":
        """Encode a trusted in-memory producer result into the bounded input form."""

        try:
            encoded = canonical_json_bytes(envelope)
        except (ContractValidationError, TypeError, ValueError) as exc:
            _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
        return cls(tree_node_id=tree_node_id, encoded_envelope=encoded)


@dataclass(frozen=True, slots=True)
class _Material:
    block: CanonicalBlock
    representation: Representation | None
    asset: AssetRef | None


def _iter_framed_semantic_bytes(
    *,
    kind: str,
    header: object,
    records: Iterable[tuple[str, object]],
    maximum_bytes: int = DEFAULT_MAX_RELEASE_BYTES,
    maximum_record_bytes: int = _ABSOLUTE_MAX_SEMANTIC_RECORD_BYTES,
    limit_field: str = "semantic_artifact_bytes",
) -> Iterable[bytes]:
    """Yield the exact safe semantic-snapshot representation record by record."""

    selected_maximum = _positive_integer(maximum_bytes, "maximum_bytes")
    selected_record_maximum = _positive_integer(
        maximum_record_bytes, "maximum_record_bytes"
    )
    if selected_maximum > _ABSOLUTE_MAX_SEMANTIC_ARTIFACT_BYTES:
        _error(
            EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
            "maximum_bytes exceeds the absolute semantic-artifact ceiling",
        )
    if selected_record_maximum > _ABSOLUTE_MAX_SEMANTIC_RECORD_BYTES:
        _error(
            EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
            "maximum_record_bytes exceeds the absolute semantic-record ceiling",
        )
    total_bytes = 0

    def framed(label: str, value: object) -> Iterable[bytes]:
        nonlocal total_bytes
        label_bytes = label.encode("utf-8")
        if len(label_bytes) > _ABSOLUTE_MAX_SEMANTIC_LABEL_BYTES:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                "semantic record label exceeds the absolute byte ceiling",
            )
        encoded = canonical_json_bytes(value)
        if len(encoded) > selected_record_maximum:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                "semantic record bytes "
                f"{len(encoded)} exceed configured maximum {selected_record_maximum}",
            )
        for chunk in (
            len(label_bytes).to_bytes(8, "big"),
            label_bytes,
            len(encoded).to_bytes(8, "big"),
            encoded,
        ):
            total_bytes += len(chunk)
            if total_bytes > selected_maximum:
                _error(
                    EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                    f"{limit_field} count {total_bytes} exceeds configured maximum "
                    f"{selected_maximum}",
                )
            yield chunk

    yield from framed(
        "header",
        {
            "schema_version": BUILDER_SCHEMA_VERSION,
            "digest_algorithm": "sha256-length-framed-v1",
            "kind": kind,
            "value": header,
        },
    )
    for label, value in records:
        yield from framed(label, value)


def _framed_semantic_digest(
    *,
    kind: str,
    header: object,
    records: Iterable[tuple[str, object]],
    maximum_bytes: int | None = None,
    limit_field: str,
) -> tuple[str, int]:
    """Hash deterministic length-framed records without one aggregate JSON tree."""

    digest = hashlib.sha256()
    total_bytes = 0
    for chunk in _iter_framed_semantic_bytes(
        kind=kind,
        header=header,
        records=records,
        maximum_bytes=(
            maximum_bytes
            if maximum_bytes is not None
            else _ABSOLUTE_MAX_SEMANTIC_ARTIFACT_BYTES
        ),
        limit_field=limit_field,
    ):
        total_bytes += len(chunk)
        digest.update(chunk)
    return digest.hexdigest(), total_bytes


def iter_canonical_input_artifact_bytes(
    document: DocumentRef,
    canonical_blocks: Sequence[CanonicalBlock],
    *,
    maximum_bytes: int = DEFAULT_MAX_RELEASE_BYTES,
    maximum_record_bytes: int = _ABSOLUTE_MAX_SEMANTIC_RECORD_BYTES,
) -> Iterable[bytes]:
    """Yield bounded CAS bytes that bind canonical graph-input semantics."""

    if not isinstance(document, DocumentRef):
        _error(EvidenceGraphBuildErrorCode.INVALID_INPUT, "document must be DocumentRef")
    if any(not isinstance(block, CanonicalBlock) for block in canonical_blocks):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            "canonical_blocks must contain CanonicalBlock values",
        )
    ordered = sorted(canonical_blocks, key=lambda item: item.revision_id)
    return _iter_framed_semantic_bytes(
        kind="canonical-input",
        header=document.to_dict(),
        records=(
            (f"canonical_block:{block.revision_id}", block.to_dict())
            for block in ordered
        ),
        maximum_bytes=maximum_bytes,
        maximum_record_bytes=maximum_record_bytes,
        limit_field="canonical_input_bytes",
    )


def _canonical_input_digest(
    document: DocumentRef,
    canonical_blocks: Sequence[CanonicalBlock],
    *,
    maximum_bytes: int | None = None,
) -> tuple[str, int]:
    ordered = sorted(canonical_blocks, key=lambda item: item.revision_id)
    return _framed_semantic_digest(
        kind="canonical-input",
        header=document.to_dict(),
        records=(
            (f"canonical_block:{block.revision_id}", block.to_dict())
            for block in ordered
        ),
        maximum_bytes=maximum_bytes,
        limit_field="canonical_input_bytes",
    )


def iter_tree_descriptor_artifact_bytes(
    tree_nodes: Sequence[TreeNodeEvidenceDescriptor],
    *,
    tree_revision_id: str,
    maximum_bytes: int = DEFAULT_MAX_RELEASE_BYTES,
    maximum_record_bytes: int = _ABSOLUTE_MAX_SEMANTIC_RECORD_BYTES,
) -> Iterable[bytes]:
    """Yield CAS bytes for evidence descriptors, not a complete BookRAG-T."""

    if any(not isinstance(item, TreeNodeEvidenceDescriptor) for item in tree_nodes):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            "tree_nodes must contain TreeNodeEvidenceDescriptor values",
        )
    _required_text(tree_revision_id, "tree_revision_id")
    ordered = sorted(tree_nodes, key=lambda item: item.tree_node_id)
    return _iter_framed_semantic_bytes(
        kind="tree-descriptor",
        header={
            "tree_revision_id": tree_revision_id,
            "tree_node_count": len(ordered),
        },
        records=(
            (f"tree_node:{descriptor.tree_node_id}", descriptor.identity_dict())
            for descriptor in ordered
        ),
        maximum_bytes=maximum_bytes,
        maximum_record_bytes=maximum_record_bytes,
        limit_field="tree_descriptor_bytes",
    )


def compute_canonical_input_sha256(
    document: DocumentRef, canonical_blocks: Sequence[CanonicalBlock]
) -> str:
    """Return the semantic digest required by ``EvidenceGraphReleaseBinding``."""

    if not isinstance(document, DocumentRef):
        _error(EvidenceGraphBuildErrorCode.INVALID_INPUT, "document must be DocumentRef")
    if any(not isinstance(block, CanonicalBlock) for block in canonical_blocks):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            "canonical_blocks must contain CanonicalBlock values",
        )
    return _canonical_input_digest(document, canonical_blocks)[0]


def _tree_descriptor_digest(
    tree_nodes: Sequence[TreeNodeEvidenceDescriptor],
    *,
    tree_revision_id: str,
    maximum_bytes: int | None = None,
) -> tuple[str, int]:
    ordered = sorted(tree_nodes, key=lambda item: item.tree_node_id)
    return _framed_semantic_digest(
        kind="tree-descriptor",
        header={
            "tree_revision_id": tree_revision_id,
            "tree_node_count": len(ordered),
        },
        records=(
            (f"tree_node:{descriptor.tree_node_id}", descriptor.identity_dict())
            for descriptor in ordered
        ),
        maximum_bytes=maximum_bytes,
        limit_field="tree_descriptor_bytes",
    )


def compute_tree_descriptor_sha256(
    tree_nodes: Sequence[TreeNodeEvidenceDescriptor],
    *,
    tree_revision_id: str,
) -> str:
    """Return the ordered evidence-descriptor digest required by the binding."""

    if any(not isinstance(item, TreeNodeEvidenceDescriptor) for item in tree_nodes):
        _error(
            EvidenceGraphBuildErrorCode.INVALID_INPUT,
            "tree_nodes must contain TreeNodeEvidenceDescriptor values",
        )
    _required_text(tree_revision_id, "tree_revision_id")
    return _tree_descriptor_digest(
        tree_nodes,
        tree_revision_id=tree_revision_id,
    )[0]


_ENVELOPE_FIELDS = frozenset(
    {"schema_version", "identity_sha256", "payload_sha256", "identity", "payload"}
)
_PAYLOAD_FIELDS = frozenset({"entities", "relations", "node_idx"})
_ENTITY_FIELDS = frozenset(
    {"entity_name", "entity_type", "description", "source_ids"}
)
_RELATION_FIELDS = frozenset(
    {
        "src_entity_name",
        "tgt_entity_name",
        "relation_name",
        "weight",
        "description",
        "source_ids",
    }
)
_IDENTITY_FIELDS = frozenset(
    {"kind", "node", "models", "graph_extraction", "prompt_sha256"}
)
_TITLE_IDENTITY_FIELDS = _IDENTITY_FIELDS | {"title_context"}
_NODE_IDENTITY_FIELDS = frozenset({"node_idx", "node_type", "source", "page", "evidence"})
_SOURCE_IDENTITY_FIELDS = frozenset(
    {
        "source_sha256",
        "document_version_id",
        "canonical_block_id",
        "canonical_revision_id",
        "observation_id",
        "source_observation_ids",
        "derived_evidence_id",
        "source_canonical_block_ids",
        "source_canonical_revision_ids",
        "canonical_representation_ids",
        "canonical_asset_ids",
        "selected_representation_ids",
        "selected_asset_ids",
        "source_evidence_refs",
        "parser_provenance",
        "quality_status",
        "reconciliation_method",
        "transformations",
    }
)
_PAGE_IDENTITY_FIELDS = frozenset(
    {"page_idx", "page_id", "page_path", "source_geometry"}
)
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {"content", "table_body", "caption", "footnote"}
)
_IMAGE_EVIDENCE_FIELDS = _EVIDENCE_IDENTITY_FIELDS | {"image"}
_VALUE_DIGEST_FIELDS = frozenset({"present", "sha256"})
_MODEL_FIELDS = frozenset({"controller", "config"})
_MODELS_FIELDS = frozenset({"llm", "vlm"})
_GRAPH_EXTRACTION_FIELDS = frozenset(
    {
        "extractor_type",
        "local_model_name",
        "local_device",
        "image_description_force",
        "max_gleaning",
        "extraction_max_workers",
        "extraction_max_in_flight",
        "extraction_batch_size",
    }
)
_TITLE_CONTEXT_FIELDS = frozenset({"title_path", "sibling_nodes"})
_IMAGE_FILE_FIELDS = frozenset(
    {"status", "path_sha256", "bytes_sha256", "size"}
)


class _ReleaseByteBudget:
    """Conservative incremental guard before domain/release materialization."""

    __slots__ = ("maximum", "used")

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = _RELEASE_BUDGET_RESERVE_BYTES
        self._check("release_bytes")

    def _check(self, field_name: str) -> None:
        if self.used > self.maximum:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                f"{field_name} count {self.used} exceeds configured maximum "
                f"{self.maximum}",
            )

    def add_json(self, value: object) -> None:
        # One delimiter byte safely covers the surrounding array separator;
        # top-level keys, brackets and generated IDs are covered by the reserve.
        self.used += len(canonical_json_bytes(value)) + 1
        self._check("release_bytes")

    def require_minimum(self, additional_bytes: int) -> None:
        if self.used + additional_bytes > self.maximum:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                "release_bytes lower bound exceeds configured maximum "
                f"{self.maximum}",
            )


class EvidenceGraphReleaseBuilder:
    """Versioned, bounded builder for immutable evidence graph releases."""

    SCHEMA_VERSION: ClassVar[str] = BUILDER_SCHEMA_VERSION

    def __init__(
        self,
        limits: EvidenceGraphBuildLimits | None = None,
        *,
        artifact_reader: ArtifactByteReader,
    ) -> None:
        if limits is not None and not isinstance(limits, EvidenceGraphBuildLimits):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "limits must be EvidenceGraphBuildLimits",
            )
        self.limits = limits or EvidenceGraphBuildLimits()
        if not callable(getattr(artifact_reader, "iter_bytes", None)):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "artifact_reader must implement iter_bytes",
            )
        self.artifact_reader = artifact_reader

    @staticmethod
    def _limit(actual: int, maximum: int, field_name: str) -> None:
        if actual > maximum:
            _error(
                EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                f"{field_name} count {actual} exceeds configured maximum {maximum}",
            )

    @classmethod
    def _bounded_tuple(
        cls,
        value: object,
        *,
        maximum: int,
        field_name: str,
    ) -> tuple[object, ...]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                f"{field_name} must be a bounded sequence",
            )
        cls._limit(len(value), maximum, field_name)
        return tuple(value)

    def _verify_artifact(
        self,
        artifact: ImmutableArtifactRef,
        *,
        maximum_bytes: int,
        field_name: str,
    ) -> None:
        self._limit(artifact.size_bytes, maximum_bytes, f"{field_name}_bytes")
        digest = hashlib.sha256()
        total = 0
        chunk_count = 0
        try:
            chunks = self.artifact_reader.iter_bytes(
                artifact,
                max_bytes=min(maximum_bytes, artifact.size_bytes),
            )
            for position, chunk in enumerate(chunks):
                chunk_count += 1
                if chunk_count > _MAX_ARTIFACT_CHUNKS:
                    _error(
                        EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                        f"{field_name}_chunks exceeds configured maximum "
                        f"{_MAX_ARTIFACT_CHUNKS}",
                    )
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    _error(
                        EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                        f"{field_name} reader chunk {position} is not bytes-like",
                    )
                view = memoryview(chunk)
                if len(view) == 0:
                    _error(
                        EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                        f"{field_name} reader chunk {position} is empty",
                    )
                total += len(view)
                if total > maximum_bytes:
                    _error(
                        EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                        f"{field_name}_bytes count {total} exceeds configured maximum "
                        f"{maximum_bytes}",
                    )
                if total > artifact.size_bytes:
                    _error(
                        EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                        f"{field_name} contains more bytes than its immutable reference",
                    )
                digest.update(view)
        except EvidenceGraphBuildError:
            raise
        except Exception as exc:
            _error(
                EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                f"could not verify {field_name}: {type(exc).__name__}",
            )
        if total != artifact.size_bytes:
            _error(
                EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                f"{field_name} size differs from its immutable reference",
            )
        if digest.hexdigest() != artifact.sha256:
            _error(
                EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY,
                f"{field_name} bytes differ from its immutable reference",
            )

    @staticmethod
    def _canonical_materials(
        document: DocumentRef,
        blocks: Sequence[CanonicalBlock],
        *,
        max_materials: int,
    ) -> tuple[
        dict[tuple[str, str, str], _Material],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        materials: dict[tuple[str, str, str], _Material] = {}
        revision_ids: set[str] = set()
        representation_ids: set[str] = set()
        asset_ids: set[str] = set()
        representations_by_id: dict[str, Representation] = {}
        assets_by_id: dict[str, AssetRef] = {}
        material_count = 0
        for index, block in enumerate(blocks):
            if not isinstance(block, CanonicalBlock):
                _error(
                    EvidenceGraphBuildErrorCode.INVALID_INPUT,
                    f"canonical_blocks[{index}] must be CanonicalBlock",
                )
            if block.document_version_id != document.document_version_id:
                _error(
                    EvidenceGraphBuildErrorCode.DOCUMENT_BINDING,
                    f"canonical block {block.revision_id} belongs to another document version",
                )
            if block.revision_id in revision_ids:
                _error(
                    EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                    f"duplicate canonical revision {block.revision_id!r}",
                )
            revision_ids.add(block.revision_id)
            for representation in block.representations:
                material_count += 1
                if material_count > max_materials:
                    _error(
                        EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                        f"materials count exceeds configured maximum {max_materials}",
                    )
                key = (block.revision_id, "representation", representation.representation_id)
                if key in materials:
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"duplicate representation {representation.representation_id!r} "
                        f"in canonical revision {block.revision_id!r}",
                    )
                prior = representations_by_id.get(representation.representation_id)
                if prior is not None and prior.to_dict() != representation.to_dict():
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"representation ID {representation.representation_id!r} "
                        "identifies unequal canonical content",
                    )
                representations_by_id[representation.representation_id] = representation
                materials[key] = _Material(block, representation, None)
                representation_ids.add(representation.representation_id)
            for asset in block.assets:
                material_count += 1
                if material_count > max_materials:
                    _error(
                        EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED,
                        f"materials count exceeds configured maximum {max_materials}",
                    )
                key = (block.revision_id, "asset", asset.asset_id)
                if key in materials:
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"duplicate asset {asset.asset_id!r} in canonical revision "
                        f"{block.revision_id!r}",
                    )
                prior_asset = assets_by_id.get(asset.asset_id)
                if prior_asset is not None and prior_asset.to_dict() != asset.to_dict():
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"asset ID {asset.asset_id!r} identifies unequal canonical content",
                    )
                assets_by_id[asset.asset_id] = asset
                materials[key] = _Material(block, None, asset)
                asset_ids.add(asset.asset_id)
        if not revision_ids:
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "canonical_blocks must not be empty",
            )
        return (
            materials,
            tuple(sorted(revision_ids)),
            tuple(sorted(representation_ids)),
            tuple(sorted(asset_ids)),
        )

    @classmethod
    def _anchors_for_node(
        cls,
        descriptor: TreeNodeEvidenceDescriptor,
        materials: Mapping[tuple[str, str, str], _Material],
    ) -> tuple[EvidenceAnchor, ...]:
        anchors: list[EvidenceAnchor] = []
        for selection in descriptor.selections:
            kind = "representation" if selection.representation_id is not None else "asset"
            material_id = selection.representation_id or selection.asset_id
            assert material_id is not None
            material = materials.get(
                (selection.canonical_revision_id, kind, material_id)
            )
            if material is None:
                _error(
                    EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                    f"tree node {descriptor.tree_node_id!r} selected missing {kind} "
                    f"{material_id!r} in revision {selection.canonical_revision_id!r}",
                )
            if material.asset is not None:
                if material.asset.sha256 is None:
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"selected asset {material.asset.asset_id!r} has no immutable sha256",
                    )
                if material.asset.uri != f"cas://sha256/{material.asset.sha256}":
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"selected asset {material.asset.asset_id!r} is not bound to its "
                        "digest-consistent CAS URI",
                    )
            try:
                anchor = EvidenceAnchor(
                    canonical_revision_id=selection.canonical_revision_id,
                    representation_id=selection.representation_id,
                    asset_id=selection.asset_id,
                    page_id=material.block.page_id,
                    char_span=selection.char_span,
                    table_cell=selection.table_cell,
                    bbox=selection.bbox,
                    diagram_element_id=selection.diagram_element_id,
                    metadata={
                        "builder_schema_version": BUILDER_SCHEMA_VERSION,
                        "evidence_precision": "coarse_whole_node",
                        "authority_eligible": False,
                        "selection": json_ready(selection.metadata),
                        "asset_ref": (
                            material.asset.to_dict()
                            if material.asset is not None
                            else None
                        ),
                    },
                )
            except ContractValidationError as exc:
                _error(EvidenceGraphBuildErrorCode.MATERIAL_BINDING, str(exc))
            anchors.append(anchor)
        return tuple(anchors)

    @staticmethod
    def _validated_value_digest(
        value: object,
        field_name: str,
        *,
        expected_value: object | None = None,
        compare_expected: bool = False,
    ) -> Mapping[str, object]:
        digest = _mapping(value, field_name)
        _exact_fields(
            digest,
            _VALUE_DIGEST_FIELDS,
            field_name,
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        present = digest.get("present")
        if not isinstance(present, bool):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"{field_name}.present must be boolean",
            )
        try:
            require_sha256(digest.get("sha256"), f"{field_name}.sha256")
        except ContractValidationError as exc:
            _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
        if compare_expected:
            expected = {
                "present": True,
                "sha256": content_sha256(expected_value),
            }
            if dict(digest) != expected:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    f"{field_name} does not bind the exact ordered selected IDs",
                )
        return digest

    @staticmethod
    def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
        values = _sequence(value, field_name)
        result: list[str] = []
        for index, item in enumerate(values):
            if not isinstance(item, str) or not item:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                    f"{field_name}[{index}] must be a non-empty string",
                )
            result.append(item)
        return tuple(result)

    @classmethod
    def _validate_checkpoint_identity(
        cls,
        identity: Mapping[str, object],
        descriptor: TreeNodeEvidenceDescriptor,
        *,
        selected_assets: tuple[AssetRef, ...],
        selected_asset_receipts: tuple[SelectedAssetMaterializationReceipt, ...],
    ) -> Mapping[str, object]:
        kind = identity.get("kind")
        if kind != descriptor.checkpoint_kind:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"checkpoint kind does not match tree node {descriptor.tree_node_id!r}",
            )
        expected_identity_fields = (
            _TITLE_IDENTITY_FIELDS if kind == "title" else _IDENTITY_FIELDS
        )
        _exact_fields(
            identity,
            expected_identity_fields,
            "checkpoint.identity",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        node = _mapping(identity.get("node"), "checkpoint.identity.node")
        _exact_fields(
            node,
            _NODE_IDENTITY_FIELDS,
            "checkpoint.identity.node",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        if node.get("node_type") != descriptor.node_type:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"checkpoint node_type does not match tree node {descriptor.tree_node_id!r}",
            )
        node_index = node.get("node_idx")
        if (
            isinstance(node_index, bool)
            or not isinstance(node_index, int)
            or node_index < 0
        ):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "checkpoint.identity.node.node_idx must be a non-negative integer",
            )

        models = _mapping(identity.get("models"), "checkpoint.identity.models")
        _exact_fields(
            models,
            _MODELS_FIELDS,
            "checkpoint.identity.models",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        for model_name in ("llm", "vlm"):
            model = models.get(model_name)
            if model is None:
                continue
            model_identity = _mapping(model, f"checkpoint.identity.models.{model_name}")
            _exact_fields(
                model_identity,
                _MODEL_FIELDS,
                f"checkpoint.identity.models.{model_name}",
                code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
            )
            _required_text(
                model_identity.get("controller"),
                f"checkpoint.identity.models.{model_name}.controller",
            )

        graph_extraction = _mapping(
            identity.get("graph_extraction"), "checkpoint.identity.graph_extraction"
        )
        _exact_fields(
            graph_extraction,
            _GRAPH_EXTRACTION_FIELDS,
            "checkpoint.identity.graph_extraction",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        prompt = _mapping(identity.get("prompt_sha256"), "checkpoint.identity.prompt_sha256")
        for prompt_name, prompt_digest in prompt.items():
            if not prompt_name:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                    "checkpoint.identity.prompt_sha256 contains an empty prompt name",
                )
            try:
                require_sha256(prompt_digest, f"prompt_sha256.{prompt_name}")
            except ContractValidationError as exc:
                _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
        if kind == "title":
            title_context = _mapping(
                identity.get("title_context"), "checkpoint.identity.title_context"
            )
            _exact_fields(
                title_context,
                _TITLE_CONTEXT_FIELDS,
                "checkpoint.identity.title_context",
                code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
            )
            _sequence(title_context.get("title_path"), "title_context.title_path")
            _sequence(title_context.get("sibling_nodes"), "title_context.sibling_nodes")

        source = _mapping(node.get("source"), "checkpoint.identity.node.source")
        _exact_fields(
            source,
            _SOURCE_IDENTITY_FIELDS,
            "checkpoint.identity.node.source",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        try:
            require_sha256(source.get("source_sha256"), "source.source_sha256")
        except ContractValidationError as exc:
            _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
        _required_text(source.get("document_version_id"), "source.document_version_id")
        for field_name in (
            "canonical_block_id",
            "canonical_revision_id",
            "observation_id",
            "derived_evidence_id",
        ):
            value = source.get(field_name)
            if value is not None and (not isinstance(value, str) or not value):
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                    f"source.{field_name} must be null or a non-empty string",
                )
        for field_name in (
            "source_observation_ids",
            "source_canonical_block_ids",
            "source_canonical_revision_ids",
        ):
            cls._string_sequence(source.get(field_name), f"source.{field_name}")
        for field_name in (
            "canonical_representation_ids",
            "canonical_asset_ids",
            "source_evidence_refs",
            "parser_provenance",
            "transformations",
        ):
            cls._validated_value_digest(source.get(field_name), f"source.{field_name}")
        cls._validated_value_digest(
            source.get("selected_representation_ids"),
            "source.selected_representation_ids",
            expected_value=list(descriptor.selected_representation_ids),
            compare_expected=True,
        )
        cls._validated_value_digest(
            source.get("selected_asset_ids"),
            "source.selected_asset_ids",
            expected_value=list(descriptor.selected_asset_ids),
            compare_expected=True,
        )

        page = _mapping(node.get("page"), "checkpoint.identity.node.page")
        _exact_fields(
            page,
            _PAGE_IDENTITY_FIELDS,
            "checkpoint.identity.node.page",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        cls._validated_value_digest(page.get("page_path"), "page.page_path")
        cls._validated_value_digest(
            page.get("source_geometry"), "page.source_geometry"
        )

        evidence = _mapping(node.get("evidence"), "checkpoint.identity.node.evidence")
        evidence_fields = (
            _IMAGE_EVIDENCE_FIELDS
            if descriptor.node_type == "image"
            else _EVIDENCE_IDENTITY_FIELDS
        )
        _exact_fields(
            evidence,
            evidence_fields,
            "checkpoint.identity.node.evidence",
            code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
        )
        for field_name in ("content", "table_body", "caption", "footnote"):
            cls._validated_value_digest(
                evidence.get(field_name), f"evidence.{field_name}"
            )
        if descriptor.node_type == "image":
            if len(selected_assets) != 1 or len(selected_asset_receipts) != 1:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    "image KG checkpoints require exactly one selected immutable asset",
                )
            image = _mapping(evidence.get("image"), "evidence.image")
            _exact_fields(
                image,
                _IMAGE_FILE_FIELDS,
                "evidence.image",
                code=EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
            )
            if image.get("status") != "present":
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    "image KG checkpoint does not attest readable image bytes",
                )
            for digest_field in ("path_sha256", "bytes_sha256"):
                try:
                    require_sha256(image.get(digest_field), f"evidence.image.{digest_field}")
                except ContractValidationError as exc:
                    _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
            size = image.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                    "evidence.image.size must be a non-negative integer",
                )
            if image.get("bytes_sha256") != selected_assets[0].sha256:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    "image checkpoint bytes do not match the selected AssetRef sha256",
                )
            if image.get("size") != selected_asset_receipts[0].artifact.size_bytes:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    "image checkpoint size does not match verified selected asset bytes",
                )
        return source

    @staticmethod
    def _checkpoint(
        checkpoint: LegacyKGCheckpointInput,
        descriptor: TreeNodeEvidenceDescriptor,
        document: DocumentRef,
        *,
        selected_assets: tuple[AssetRef, ...],
        selected_asset_receipts: tuple[SelectedAssetMaterializationReceipt, ...],
    ) -> tuple[Mapping[str, object], Mapping[str, object], str, str]:
        try:
            parsed = parse_json_object(checkpoint.encoded_envelope)
            frozen = freeze_json(parsed, field="checkpoint.envelope")
        except (ContractValidationError, TypeError, ValueError, RecursionError) as exc:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} is invalid JSON: "
                f"{type(exc).__name__}",
            )
        if not isinstance(frozen, Mapping):  # pragma: no cover - parse/freeze guard
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "checkpoint envelope must be an object",
            )
        envelope = frozen
        if set(envelope) != _ENVELOPE_FIELDS:
            missing = sorted(_ENVELOPE_FIELDS - set(envelope))
            unknown = sorted(set(envelope) - _ENVELOPE_FIELDS)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unknown:
                details.append(f"unknown {', '.join(unknown)}")
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} has invalid envelope "
                f"fields ({'; '.join(details)})",
            )
        if envelope.get("schema_version") != LEGACY_CHECKPOINT_SCHEMA_VERSION:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} uses unsupported schema",
            )
        identity = _mapping(envelope.get("identity"), "checkpoint.identity")
        payload = _mapping(envelope.get("payload"), "checkpoint.payload")
        _exact_fields(payload, _PAYLOAD_FIELDS, "checkpoint.payload")
        identity_sha256 = envelope.get("identity_sha256")
        payload_sha256 = envelope.get("payload_sha256")
        try:
            require_sha256(identity_sha256, "checkpoint.identity_sha256")
            require_sha256(payload_sha256, "checkpoint.payload_sha256")
        except ContractValidationError as exc:
            _error(EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY, str(exc))
        if content_sha256(identity) != identity_sha256:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} has an invalid identity digest",
            )
        if identity_sha256 != descriptor.expected_checkpoint_identity_sha256:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} is not the "
                "KG-stage identity authorized by its tree descriptor",
            )
        if content_sha256(payload) != payload_sha256:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                f"checkpoint for tree node {checkpoint.tree_node_id!r} has an invalid payload digest",
            )
        source = EvidenceGraphReleaseBuilder._validate_checkpoint_identity(
            identity,
            descriptor,
            selected_assets=selected_assets,
            selected_asset_receipts=selected_asset_receipts,
        )
        node_identity = _mapping(identity.get("node"), "checkpoint.identity.node")
        if node_identity.get("node_idx") != descriptor.legacy_node_index:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"checkpoint node_idx does not match tree node {checkpoint.tree_node_id!r}",
            )
        payload_node_index = payload.get("node_idx")
        if (
            isinstance(payload_node_index, bool)
            or not isinstance(payload_node_index, int)
            or payload_node_index < 0
        ):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "checkpoint.payload.node_idx must be a non-negative integer",
            )
        if payload_node_index != descriptor.legacy_node_index:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"checkpoint payload node_idx does not match tree node {checkpoint.tree_node_id!r}",
            )
        if source.get("source_sha256") != document.content_sha256:
            _error(
                EvidenceGraphBuildErrorCode.DOCUMENT_BINDING,
                f"checkpoint source bytes do not match tree node {checkpoint.tree_node_id!r}",
            )
        if source.get("document_version_id") != document.document_version_id:
            _error(
                EvidenceGraphBuildErrorCode.DOCUMENT_BINDING,
                f"checkpoint document version does not match tree node {checkpoint.tree_node_id!r}",
            )
        declared_revisions: set[str] = set()
        direct_revision = source.get("canonical_revision_id")
        if direct_revision is not None and (
            not isinstance(direct_revision, str) or not direct_revision
        ):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY,
                "source.canonical_revision_id must be null or a non-empty string",
            )
        if isinstance(direct_revision, str):
            declared_revisions.add(direct_revision)
        raw_revisions = EvidenceGraphReleaseBuilder._string_sequence(
            source.get("source_canonical_revision_ids"),
            "source.source_canonical_revision_ids",
        )
        declared_revisions.update(raw_revisions)
        selected_revisions = {
            selection.canonical_revision_id for selection in descriptor.selections
        }
        if not selected_revisions.issubset(declared_revisions):
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"selected canonical revision is absent from checkpoint identity for "
                f"tree node {checkpoint.tree_node_id!r}",
            )
        return identity, payload, identity_sha256, payload_sha256  # type: ignore[return-value]

    @staticmethod
    def _source_ids(
        row: Mapping[str, object],
        descriptor: TreeNodeEvidenceDescriptor,
        field_name: str,
    ) -> tuple[int, ...]:
        raw = _sequence(row.get("source_ids"), f"{field_name}.source_ids")
        values: list[int] = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _error(
                    EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
                    f"{field_name}.source_ids[{index}] must be a non-negative integer",
                )
            values.append(value)
        if set(values) != {descriptor.legacy_node_index}:
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                f"{field_name}.source_ids must identify only legacy node "
                f"{descriptor.legacy_node_index}",
            )
        return tuple(values)

    @staticmethod
    def _raw_attributes(
        *,
        row: Mapping[str, object],
        ordinal: int,
        identity_sha256: str,
        payload_sha256: str,
        descriptor: TreeNodeEvidenceDescriptor,
    ) -> dict[str, object]:
        return {
            "legacy_assertion": json_ready(row),
            "legacy_record_ordinal": ordinal,
            "legacy_node_index": descriptor.legacy_node_index,
            "tree_node_id": descriptor.tree_node_id,
            "checkpoint_schema_version": LEGACY_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_identity_sha256": identity_sha256,
            "checkpoint_payload_sha256": payload_sha256,
            "builder_schema_version": BUILDER_SCHEMA_VERSION,
        }

    def build(
        self,
        *,
        document: DocumentRef,
        canonical_blocks: Sequence[CanonicalBlock],
        tree_nodes: Sequence[TreeNodeEvidenceDescriptor],
        checkpoints: Sequence[LegacyKGCheckpointInput],
        binding: EvidenceGraphReleaseBinding,
    ) -> EvidenceGraphRelease:
        """Build one deterministic release without graph refinement or merging."""

        if not isinstance(document, DocumentRef):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "document must be a validated DocumentRef",
            )
        if not isinstance(binding, EvidenceGraphReleaseBinding):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "binding must be EvidenceGraphReleaseBinding",
            )
        self._limit(
            len(binding.selected_asset_receipts),
            self.limits.max_selected_assets,
            "selected_asset_receipts",
        )
        declared_selected_asset_bytes = sum(
            receipt.artifact.size_bytes
            for receipt in binding.selected_asset_receipts
        )
        self._limit(
            declared_selected_asset_bytes,
            self.limits.max_total_selected_asset_bytes,
            "total_selected_asset_bytes",
        )
        self._verify_artifact(
            binding.canonical_artifact,
            maximum_bytes=self.limits.max_canonical_artifact_bytes,
            field_name="canonical_artifact",
        )
        self._verify_artifact(
            binding.tree_artifact,
            maximum_bytes=self.limits.max_tree_artifact_bytes,
            field_name="tree_artifact",
        )
        blocks = self._bounded_tuple(
            canonical_blocks,
            maximum=self.limits.max_canonical_blocks,
            field_name="canonical_blocks",
        )
        descriptors = self._bounded_tuple(
            tree_nodes,
            maximum=self.limits.max_nodes,
            field_name="tree_nodes",
        )
        checkpoint_inputs = self._bounded_tuple(
            checkpoints,
            maximum=self.limits.max_checkpoints,
            field_name="checkpoints",
        )
        if not descriptors:
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "tree_nodes must not be empty",
            )
        if any(not isinstance(item, TreeNodeEvidenceDescriptor) for item in descriptors):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "tree_nodes must contain TreeNodeEvidenceDescriptor values",
            )
        if any(not isinstance(item, LegacyKGCheckpointInput) for item in checkpoint_inputs):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "checkpoints must contain LegacyKGCheckpointInput values",
            )
        if any(not isinstance(item, CanonicalBlock) for item in blocks):
            _error(
                EvidenceGraphBuildErrorCode.INVALID_INPUT,
                "canonical_blocks must contain CanonicalBlock values",
            )
        typed_blocks = tuple(block for block in blocks if isinstance(block, CanonicalBlock))
        typed_descriptors = tuple(
            item for item in descriptors if isinstance(item, TreeNodeEvidenceDescriptor)
        )
        canonical_input_digest, _canonical_input_bytes = _canonical_input_digest(
            document,
            typed_blocks,
            maximum_bytes=self.limits.max_canonical_input_bytes,
        )
        if canonical_input_digest != binding.canonical_input_sha256:
            _error(
                EvidenceGraphBuildErrorCode.DOCUMENT_BINDING,
                "canonical semantic input digest differs from the release binding",
            )
        if (
            binding.canonical_artifact.sha256 != canonical_input_digest
            or binding.canonical_artifact.size_bytes != _canonical_input_bytes
        ):
            _error(
                EvidenceGraphBuildErrorCode.DOCUMENT_BINDING,
                "canonical CAS object is not the exact length-framed semantic input snapshot",
            )
        tree_descriptor_digest, _tree_descriptor_bytes = _tree_descriptor_digest(
            typed_descriptors,
            tree_revision_id=binding.tree_revision_id,
            maximum_bytes=self.limits.max_tree_descriptor_bytes,
        )
        if tree_descriptor_digest != binding.tree_descriptor_sha256:
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "tree semantic descriptor digest differs from the release binding",
            )
        if (
            binding.tree_artifact.sha256 != tree_descriptor_digest
            or binding.tree_artifact.size_bytes != _tree_descriptor_bytes
        ):
            _error(
                EvidenceGraphBuildErrorCode.TREE_BINDING,
                "tree CAS object is not the exact length-framed descriptor snapshot",
            )
        selection_count = sum(len(item.selections) for item in typed_descriptors)
        self._limit(selection_count, self.limits.max_selections, "selections")
        total_checkpoint_bytes = 0
        for checkpoint in checkpoint_inputs:
            assert isinstance(checkpoint, LegacyKGCheckpointInput)
            checkpoint_size = len(checkpoint.encoded_envelope)
            self._limit(
                checkpoint_size,
                self.limits.max_checkpoint_bytes,
                "checkpoint_bytes",
            )
            total_checkpoint_bytes += checkpoint_size
            self._limit(
                total_checkpoint_bytes,
                self.limits.max_total_checkpoint_bytes,
                "total_checkpoint_bytes",
            )

        descriptor_by_tree: dict[str, TreeNodeEvidenceDescriptor] = {}
        legacy_indexes: set[int] = set()
        for descriptor in typed_descriptors:
            if descriptor.tree_node_id in descriptor_by_tree:
                _error(
                    EvidenceGraphBuildErrorCode.TREE_BINDING,
                    f"duplicate tree_node_id {descriptor.tree_node_id!r}",
                )
            if descriptor.legacy_node_index in legacy_indexes:
                _error(
                    EvidenceGraphBuildErrorCode.TREE_BINDING,
                    f"duplicate legacy_node_index {descriptor.legacy_node_index}",
                )
            descriptor_by_tree[descriptor.tree_node_id] = descriptor
            legacy_indexes.add(descriptor.legacy_node_index)

        checkpoint_by_tree: dict[str, LegacyKGCheckpointInput] = {}
        for checkpoint in checkpoint_inputs:
            assert isinstance(checkpoint, LegacyKGCheckpointInput)
            if checkpoint.tree_node_id in checkpoint_by_tree:
                _error(
                    EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                    f"duplicate checkpoint for tree node {checkpoint.tree_node_id!r}",
                )
            checkpoint_by_tree[checkpoint.tree_node_id] = checkpoint
        if set(checkpoint_by_tree) != set(descriptor_by_tree):
            missing = sorted(set(descriptor_by_tree) - set(checkpoint_by_tree))
            unknown = sorted(set(checkpoint_by_tree) - set(descriptor_by_tree))
            details = []
            if missing:
                details.append(f"missing checkpoints for {', '.join(missing)}")
            if unknown:
                details.append(f"unknown checkpoint nodes {', '.join(unknown)}")
            _error(
                EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING,
                "; ".join(details),
            )

        (
            materials,
            canonical_revision_ids,
            representation_ids,
            asset_ids,
        ) = self._canonical_materials(
            document,
            typed_blocks,
            max_materials=self.limits.max_materials,
        )
        release_budget = _ReleaseByteBudget(self.limits.max_release_bytes)
        for identifier_values in (
            canonical_revision_ids,
            representation_ids,
            asset_ids,
            tuple(descriptor_by_tree),
        ):
            release_budget.add_json(list(identifier_values))

        anchors: list[EvidenceAnchor] = []
        anchor_by_id: dict[str, EvidenceAnchor] = {}
        anchors_by_tree: dict[str, tuple[EvidenceAnchor, ...]] = {}
        selected_assets_by_tree: dict[str, tuple[AssetRef, ...]] = {}
        for descriptor in sorted(typed_descriptors, key=lambda item: item.tree_node_id):
            values = self._anchors_for_node(descriptor, materials)
            for anchor in values:
                prior_anchor = anchor_by_id.get(anchor.anchor_id)
                if prior_anchor is not None and prior_anchor.to_dict() != anchor.to_dict():
                    _error(
                        EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                        f"anchor ID {anchor.anchor_id!r} identifies unequal evidence",
                    )
                if prior_anchor is None:
                    release_budget.add_json(anchor.to_dict())
                    anchor_by_id[anchor.anchor_id] = anchor
                    anchors.append(anchor)
            self._limit(len(anchors), self.limits.max_anchors, "anchors")
            anchors_by_tree[descriptor.tree_node_id] = values
            selected_assets_by_tree[descriptor.tree_node_id] = tuple(
                material.asset
                for selection in descriptor.selections
                if selection.asset_id is not None
                for material in (
                    materials[
                        (selection.canonical_revision_id, "asset", selection.asset_id)
                    ],
                )
                if material.asset is not None
            )

        selected_asset_by_id: dict[str, AssetRef] = {}
        for selected_assets in selected_assets_by_tree.values():
            for selected_asset in selected_assets:
                selected_asset_by_id[selected_asset.asset_id] = selected_asset
        self._limit(
            len(selected_asset_by_id),
            self.limits.max_selected_assets,
            "selected_assets",
        )
        receipt_by_asset_id = {
            receipt.asset_id: receipt for receipt in binding.selected_asset_receipts
        }
        if set(receipt_by_asset_id) != set(selected_asset_by_id):
            missing = sorted(set(selected_asset_by_id) - set(receipt_by_asset_id))
            unknown = sorted(set(receipt_by_asset_id) - set(selected_asset_by_id))
            details: list[str] = []
            if missing:
                details.append(f"missing receipts for {', '.join(missing)}")
            if unknown:
                details.append(f"unselected receipt assets {', '.join(unknown)}")
            _error(
                EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                "selected immutable asset receipts do not match tree selections ("
                + "; ".join(details)
                + ")",
            )
        for asset_id in sorted(selected_asset_by_id):
            selected_asset = selected_asset_by_id[asset_id]
            receipt = receipt_by_asset_id[asset_id]
            if (
                selected_asset.uri != receipt.artifact.uri
                or selected_asset.sha256 != receipt.artifact.sha256
                or selected_asset.media_type != receipt.media_type
            ):
                _error(
                    EvidenceGraphBuildErrorCode.MATERIAL_BINDING,
                    f"selected asset receipt for {asset_id!r} differs from its canonical AssetRef",
                )
            self._verify_artifact(
                receipt.artifact,
                maximum_bytes=self.limits.max_selected_asset_bytes,
                field_name=f"selected_asset[{asset_id}]",
            )
        selected_receipts_by_tree = {
            tree_node_id: tuple(
                receipt_by_asset_id[asset.asset_id] for asset in selected_assets
            )
            for tree_node_id, selected_assets in selected_assets_by_tree.items()
        }

        mentions: list[EntityMention] = []
        relations: list[RelationAssertion] = []
        links: list[GTLink] = []
        checkpoint_identity_digests: list[str] = []
        checkpoint_payload_digests: list[str] = []

        for descriptor in sorted(typed_descriptors, key=lambda item: item.tree_node_id):
            checkpoint = checkpoint_by_tree[descriptor.tree_node_id]
            (
                _identity,
                payload,
                identity_sha256,
                payload_sha256,
            ) = self._checkpoint(
                checkpoint,
                descriptor,
                document,
                selected_assets=selected_assets_by_tree[descriptor.tree_node_id],
                selected_asset_receipts=selected_receipts_by_tree[
                    descriptor.tree_node_id
                ],
            )
            checkpoint_identity_digests.append(identity_sha256)
            checkpoint_payload_digests.append(payload_sha256)
            raw_entities = _sequence(payload.get("entities"), "checkpoint.payload.entities")
            raw_relations = _sequence(
                payload.get("relations"), "checkpoint.payload.relations"
            )
            self._limit(
                len(mentions) + len(raw_entities), self.limits.max_mentions, "mentions"
            )
            self._limit(
                len(relations) + len(raw_relations),
                self.limits.max_relations,
                "relations",
            )
            self._limit(
                len(links) + len(raw_entities) + len(raw_relations),
                self.limits.max_links,
                "links",
            )
            release_budget.require_minimum(
                len(raw_entities) * _MIN_MENTION_RELEASE_BYTES
                + len(raw_relations) * _MIN_RELATION_RELEASE_BYTES
                + (len(raw_entities) + len(raw_relations))
                * _MIN_LINK_RELEASE_BYTES
            )
            extraction_revision_id = stable_id(
                "kgextract", identity_sha256, payload_sha256
            )
            anchor_ids = tuple(
                anchor.anchor_id for anchor in anchors_by_tree[descriptor.tree_node_id]
            )
            local_mentions: dict[str, list[EntityMention]] = {}
            for ordinal, raw_entity in enumerate(raw_entities):
                row = _mapping(raw_entity, f"entities[{ordinal}]")
                _exact_fields(row, _ENTITY_FIELDS, f"entities[{ordinal}]")
                name = _required_text(row.get("entity_name"), "entity_name")
                entity_type = _required_text(row.get("entity_type"), "entity_type")
                description = row.get("description")
                if not isinstance(description, str):
                    _error(
                        EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
                        f"entities[{ordinal}].description must be a string",
                    )
                self._source_ids(row, descriptor, f"entities[{ordinal}]")
                mention = EntityMention(
                    label=name,
                    entity_type=entity_type,
                    extraction_revision_id=extraction_revision_id,
                    evidence_anchor_ids=anchor_ids,
                    attributes={
                        **self._raw_attributes(
                            row=row,
                            ordinal=ordinal,
                            identity_sha256=identity_sha256,
                            payload_sha256=payload_sha256,
                            descriptor=descriptor,
                        ),
                        "description": description,
                    },
                )
                release_budget.add_json(mention.to_dict())
                mentions.append(mention)
                local_mentions.setdefault(name, []).append(mention)
                mention_link = GTLink(
                    graph_object_id=mention.mention_id,
                    tree_node_id=descriptor.tree_node_id,
                    role="extracted_from",
                    evidence_anchor_ids=anchor_ids,
                    metadata={
                        "builder_schema_version": BUILDER_SCHEMA_VERSION,
                        "evidence_precision": "coarse_whole_node",
                        "authority_eligible": False,
                        "checkpoint_payload_sha256": payload_sha256,
                        "legacy_record_ordinal": ordinal,
                    },
                )
                release_budget.add_json(mention_link.to_dict())
                links.append(mention_link)

            for ordinal, raw_relation in enumerate(raw_relations):
                row = _mapping(raw_relation, f"relations[{ordinal}]")
                _exact_fields(row, _RELATION_FIELDS, f"relations[{ordinal}]")
                source_name = _required_text(
                    row.get("src_entity_name"), "src_entity_name"
                )
                target_name = _required_text(
                    row.get("tgt_entity_name"), "tgt_entity_name"
                )
                raw_relation_name = row.get("relation_name")
                if not isinstance(raw_relation_name, str):
                    _error(
                        EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
                        f"relations[{ordinal}].relation_name must be a string",
                    )
                relation_name = (
                    raw_relation_name
                    if raw_relation_name.strip()
                    else LEGACY_UNSPECIFIED_RELATION_TYPE
                )
                description = row.get("description")
                if not isinstance(description, str):
                    _error(
                        EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
                        f"relations[{ordinal}].description must be a string",
                    )
                weight = row.get("weight")
                invalid_weight = isinstance(weight, bool) or not isinstance(
                    weight, (int, float)
                )
                if isinstance(weight, float):
                    invalid_weight = invalid_weight or not math.isfinite(weight)
                if isinstance(weight, (int, float)) and not isinstance(weight, bool):
                    invalid_weight = invalid_weight or abs(weight) > _MAX_ABSOLUTE_RELATION_WEIGHT
                if invalid_weight:
                    _error(
                        EvidenceGraphBuildErrorCode.INVALID_ASSERTION,
                        f"relations[{ordinal}].weight must be finite and have absolute "
                        f"value at most {_MAX_ABSOLUTE_RELATION_WEIGHT}",
                    )
                self._source_ids(row, descriptor, f"relations[{ordinal}]")
                source_candidates = local_mentions.get(source_name, [])
                target_candidates = local_mentions.get(target_name, [])
                for endpoint, name, candidates in (
                    ("source", source_name, source_candidates),
                    ("target", target_name, target_candidates),
                ):
                    if not candidates:
                        _error(
                            EvidenceGraphBuildErrorCode.MISSING_ENDPOINT,
                            f"relation {ordinal} {endpoint} {name!r} has no entity row "
                            f"in tree node {descriptor.tree_node_id!r}",
                        )
                    if len(candidates) > 1:
                        _error(
                            EvidenceGraphBuildErrorCode.AMBIGUOUS_ENDPOINT,
                            f"relation {ordinal} {endpoint} {name!r} matches "
                            f"{len(candidates)} entity rows in tree node "
                            f"{descriptor.tree_node_id!r}",
                        )
                relation = RelationAssertion(
                    source_mention_id=source_candidates[0].mention_id,
                    target_mention_id=target_candidates[0].mention_id,
                    relation_type=relation_name,
                    extraction_revision_id=extraction_revision_id,
                    evidence_anchor_ids=anchor_ids,
                    assertion_key=stable_id(
                        "legacyassertion",
                        payload_sha256,
                        descriptor.tree_node_id,
                        ordinal,
                    ),
                    attributes={
                        **self._raw_attributes(
                            row=row,
                            ordinal=ordinal,
                            identity_sha256=identity_sha256,
                            payload_sha256=payload_sha256,
                            descriptor=descriptor,
                        ),
                        "description": description,
                        "weight": weight,
                        "legacy_relation_name": raw_relation_name,
                    },
                )
                release_budget.add_json(relation.to_dict())
                relations.append(relation)
                relation_link = GTLink(
                    graph_object_id=relation.relation_id,
                    tree_node_id=descriptor.tree_node_id,
                    role="asserted_in",
                    evidence_anchor_ids=anchor_ids,
                    metadata={
                        "builder_schema_version": BUILDER_SCHEMA_VERSION,
                        "evidence_precision": "coarse_whole_node",
                        "authority_eligible": False,
                        "checkpoint_payload_sha256": payload_sha256,
                        "legacy_record_ordinal": ordinal,
                    },
                )
                release_budget.add_json(relation_link.to_dict())
                links.append(relation_link)

        def graph_records() -> Iterable[tuple[str, object]]:
            for value in sorted(anchors, key=lambda item: item.anchor_id):
                yield f"anchor:{value.anchor_id}", value.to_dict()
            for value in sorted(mentions, key=lambda item: item.mention_id):
                yield f"mention:{value.mention_id}", value.to_dict()
            for value in sorted(relations, key=lambda item: item.relation_id):
                yield f"relation:{value.relation_id}", value.to_dict()

        graph_artifact_sha256, _graph_bytes = _framed_semantic_digest(
            kind="evidence-graph-artifact",
            header={
                "document_version_id": document.document_version_id,
                "build_spec_sha256": binding.build_spec_sha256,
            },
            records=graph_records(),
            limit_field="graph_artifact_bytes",
        )
        graph_revision_id = stable_id(
            "graphrev",
            BUILDER_SCHEMA_VERSION,
            document.document_version_id,
            binding.build_spec_sha256,
            graph_artifact_sha256,
        )
        mapping_artifact_sha256, _mapping_bytes = _framed_semantic_digest(
            kind="evidence-mapping-artifact",
            header={
                "document_version_id": document.document_version_id,
                "graph_revision_id": graph_revision_id,
                "tree_revision_id": binding.tree_revision_id,
            },
            records=(
                (f"gt_link:{value.gt_link_id}", value.to_dict())
                for value in sorted(links, key=lambda item: item.gt_link_id)
            ),
            limit_field="mapping_artifact_bytes",
        )
        mapping_revision_id = stable_id(
            "mappingrev",
            BUILDER_SCHEMA_VERSION,
            document.document_version_id,
            binding.tree_revision_id,
            graph_revision_id,
            mapping_artifact_sha256,
        )
        checkpoint_set_sha256, _checkpoint_digest_bytes = _framed_semantic_digest(
            kind="checkpoint-set",
            header={"checkpoint_count": len(checkpoint_identity_digests)},
            records=(
                (f"identity:{index}", value)
                for index, value in enumerate(sorted(checkpoint_identity_digests))
            ),
            limit_field="checkpoint_digest_bytes",
        )
        payload_set_sha256, _payload_digest_bytes = _framed_semantic_digest(
            kind="checkpoint-payload-set",
            header={"checkpoint_count": len(checkpoint_payload_digests)},
            records=(
                (f"payload:{index}", value)
                for index, value in enumerate(sorted(checkpoint_payload_digests))
            ),
            limit_field="checkpoint_payload_digest_bytes",
        )
        checkpoint_set_sha256 = content_sha256(
            {
                "identity_set_sha256": checkpoint_set_sha256,
                "payload_set_sha256": payload_set_sha256,
            }
        )
        release_metadata = {
            "builder": {
                "schema_version": BUILDER_SCHEMA_VERSION,
                "algorithm": "indexed-local-endpoint-resolution",
                "limits": self.limits.to_dict(),
                "evidence_precision": "coarse_whole_node",
                "authority_eligible": False,
                "authority_ineligibility_reason": (
                    "legacy checkpoints do not yet carry exact direct or transitive "
                    "material-use receipts for each generated claim"
                ),
            },
            "input_digests": {
                "canonical_input_sha256": canonical_input_digest,
                "tree_descriptor_sha256": tree_descriptor_digest,
                "checkpoint_set_sha256": checkpoint_set_sha256,
            },
            "derived_semantic_digests": {
                "graph": {
                    "sha256": graph_artifact_sha256,
                    "size_bytes": _graph_bytes,
                    "materialized_in_cas": False,
                },
                "mapping": {
                    "sha256": mapping_artifact_sha256,
                    "size_bytes": _mapping_bytes,
                    "materialized_in_cas": False,
                },
            },
            "verified_artifacts": {
                "canonical": {
                    "artifact_kind": "canonical_input_semantic_snapshot",
                    "uri": binding.canonical_artifact.uri,
                    "sha256": binding.canonical_artifact.sha256,
                    "size_bytes": binding.canonical_artifact.size_bytes,
                },
                "tree": {
                    "artifact_kind": "tree_evidence_descriptor_snapshot",
                    "contains_full_tree_topology": False,
                    "uri": binding.tree_artifact.uri,
                    "sha256": binding.tree_artifact.sha256,
                    "size_bytes": binding.tree_artifact.size_bytes,
                },
                "selected_assets": [
                    receipt.to_dict()
                    for receipt in binding.selected_asset_receipts
                ],
            },
            "binding_metadata": json_ready(binding.metadata),
        }
        release_budget.add_json(release_metadata)
        release = EvidenceGraphRelease(
            document_id=document.document_id,
            document_version_id=document.document_version_id,
            source_sha256=document.content_sha256,
            build_spec_sha256=binding.build_spec_sha256,
            canonical_artifact_sha256=binding.canonical_artifact_sha256,
            tree_revision_id=binding.tree_revision_id,
            tree_artifact_sha256=binding.tree_artifact_sha256,
            graph_revision_id=graph_revision_id,
            graph_artifact_sha256=graph_artifact_sha256,
            mapping_revision_id=mapping_revision_id,
            mapping_artifact_sha256=mapping_artifact_sha256,
            canonical_revision_ids=canonical_revision_ids,
            representation_ids=representation_ids,
            asset_ids=asset_ids,
            tree_node_ids=tuple(descriptor_by_tree),
            anchors=tuple(anchors),
            mentions=tuple(mentions),
            relations=tuple(relations),
            gt_links=tuple(links),
            metadata=release_metadata,
        )
        release_bytes = canonical_json_bytes(release.to_dict())
        self._limit(
            len(release_bytes), self.limits.max_release_bytes, "release_bytes"
        )
        return release


def build_evidence_graph_release(
    *,
    document: DocumentRef,
    canonical_blocks: Sequence[CanonicalBlock],
    tree_nodes: Sequence[TreeNodeEvidenceDescriptor],
    checkpoints: Sequence[LegacyKGCheckpointInput],
    binding: EvidenceGraphReleaseBinding,
    limits: EvidenceGraphBuildLimits | None = None,
    artifact_reader: ArtifactByteReader,
) -> EvidenceGraphRelease:
    """Convenience entry point for callers that do not need a reusable builder."""

    return EvidenceGraphReleaseBuilder(
        limits,
        artifact_reader=artifact_reader,
    ).build(
        document=document,
        canonical_blocks=canonical_blocks,
        tree_nodes=tree_nodes,
        checkpoints=checkpoints,
        binding=binding,
    )


__all__ = [
    "BUILDER_SCHEMA_VERSION",
    "DEFAULT_MAX_RELEASE_BYTES",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_UNSPECIFIED_RELATION_TYPE",
    "ArtifactByteReader",
    "CanonicalMaterialSelection",
    "EvidenceGraphBuildError",
    "EvidenceGraphBuildErrorCode",
    "EvidenceGraphBuildLimits",
    "EvidenceGraphReleaseBinding",
    "EvidenceGraphReleaseBuilder",
    "ImmutableArtifactRef",
    "LegacyKGCheckpointInput",
    "SelectedAssetMaterializationReceipt",
    "TreeNodeEvidenceDescriptor",
    "build_evidence_graph_release",
    "compute_canonical_input_sha256",
    "compute_tree_descriptor_sha256",
    "iter_canonical_input_artifact_bytes",
    "iter_tree_descriptor_artifact_bytes",
]
