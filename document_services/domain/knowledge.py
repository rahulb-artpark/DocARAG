"""Immutable, dependency-free contracts for an evidence-grounded BookIndex graph.

The release envelope binds the current document/build to BookRAG's tree (T),
directed graph (G), and graph-to-tree mapping (M).  It deliberately models
extraction assertions, not entity-resolution decisions; aliases and canonical
entity revisions belong to a later contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import ClassVar, Self

from document_services.domain.base import (
    CONTRACT_SCHEMA_VERSION,
    ContractValidationError,
    JsonDTOMixin,
    JsonObject,
    content_sha256,
    freeze_json,
    json_ready,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_probability,
    require_schema_version,
    require_sequence,
    require_sha256,
    stable_id,
)


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return require_non_empty(value, field_name)


def _ids(
    values: Sequence[object], field_name: str, *, non_empty: bool = False
) -> tuple[str, ...]:
    result = tuple(sorted({require_non_empty(value, f"{field_name}[]") for value in values}))
    if non_empty and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return result


def _pair(
    value: Sequence[object] | None,
    field_name: str,
    *,
    strictly_increasing: bool = False,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ContractValidationError(f"{field_name} must contain exactly two integers")
    first, second = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ContractValidationError(f"{field_name} must contain exactly two integers")
    if first < 0 or second < 0:
        raise ContractValidationError(f"{field_name} values must be non-negative")
    if strictly_increasing and second <= first:
        raise ContractValidationError(f"{field_name} end must be greater than start")
    return (first, second)


def _bbox(value: Sequence[object] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ContractValidationError("bbox must contain exactly four coordinates")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ContractValidationError("bbox must contain finite numbers")
    x0, y0, x1, y1 = (float(item) for item in value)
    if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
        raise ContractValidationError("bbox uses normalized coordinates in [0, 1]")
    if x1 < x0 or y1 < y0:
        raise ContractValidationError("bbox must satisfy x0 <= x1 and y0 <= y1")
    return (x0, y0, x1, y1)


def _verify_generated(payload: Mapping[str, object], field_name: str, expected: str) -> None:
    if payload.get(field_name) != expected:
        raise ContractValidationError(f"{field_name} does not match canonical content")


@dataclass(frozen=True, slots=True)
class EvidenceAnchor(JsonDTOMixin):
    """An exact locus in one canonical representation or immutable asset.

    Character spans are half-open.  Table cells are zero-based ``(row,
    column)`` pairs.  Bounding boxes are normalized top-left page coordinates
    and therefore require ``page_id``.
    """

    canonical_revision_id: str
    representation_id: str | None = None
    asset_id: str | None = None
    page_id: str | None = None
    char_span: tuple[int, int] | None = None
    table_cell: tuple[int, int] | None = None
    bbox: tuple[float, float, float, float] | None = None
    diagram_element_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    anchor_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.canonical_revision_id, "canonical_revision_id")
        representation_id = _optional_id(self.representation_id, "representation_id")
        asset_id = _optional_id(self.asset_id, "asset_id")
        if (representation_id is None) == (asset_id is None):
            raise ContractValidationError(
                "EvidenceAnchor requires exactly one of representation_id or asset_id"
            )
        page_id = _optional_id(self.page_id, "page_id")
        char_span = _pair(self.char_span, "char_span", strictly_increasing=True)
        table_cell = _pair(self.table_cell, "table_cell")
        bbox = _bbox(self.bbox)
        if bbox is not None and page_id is None:
            raise ContractValidationError("bbox requires page_id")
        diagram_element_id = _optional_id(
            self.diagram_element_id, "diagram_element_id"
        )
        metadata = freeze_json(self.metadata, field="anchor.metadata")
        object.__setattr__(self, "representation_id", representation_id)
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "char_span", char_span)
        object.__setattr__(self, "table_cell", table_cell)
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "diagram_element_id", diagram_element_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "anchor_id",
            stable_id(
                "anchor",
                self.canonical_revision_id,
                representation_id,
                asset_id,
                page_id,
                char_span,
                table_cell,
                bbox,
                diagram_element_id,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "anchor_id": self.anchor_id,
            "canonical_revision_id": self.canonical_revision_id,
            "representation_id": self.representation_id,
            "asset_id": self.asset_id,
            "page_id": self.page_id,
            "char_span": list(self.char_span) if self.char_span else None,
            "table_cell": list(self.table_cell) if self.table_cell else None,
            "bbox": list(self.bbox) if self.bbox else None,
            "diagram_element_id": self.diagram_element_id,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "anchor_id",
                "canonical_revision_id",
                "representation_id",
                "asset_id",
                "page_id",
                "char_span",
                "table_cell",
                "bbox",
                "diagram_element_id",
                "metadata",
            },
            dto="EvidenceAnchor",
        )
        char_span = payload.get("char_span")
        table_cell = payload.get("table_cell")
        bbox = payload.get("bbox")
        result = cls(
            canonical_revision_id=payload.get("canonical_revision_id"),  # type: ignore[arg-type]
            representation_id=payload.get("representation_id"),  # type: ignore[arg-type]
            asset_id=payload.get("asset_id"),  # type: ignore[arg-type]
            page_id=payload.get("page_id"),  # type: ignore[arg-type]
            char_span=(
                tuple(require_sequence(char_span, "char_span"))
                if char_span is not None
                else None
            ),  # type: ignore[arg-type]
            table_cell=(
                tuple(require_sequence(table_cell, "table_cell"))
                if table_cell is not None
                else None
            ),  # type: ignore[arg-type]
            bbox=(
                tuple(require_sequence(bbox, "bbox")) if bbox is not None else None
            ),  # type: ignore[arg-type]
            diagram_element_id=payload.get("diagram_element_id"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "anchor.metadata"),
        )
        _verify_generated(payload, "anchor_id", result.anchor_id)
        return result


@dataclass(frozen=True, slots=True)
class EntityMention(JsonDTOMixin):
    """An extracted graph node before cross-mention entity resolution."""

    label: str
    entity_type: str
    extraction_revision_id: str
    evidence_anchor_ids: tuple[str, ...]
    extraction_confidence: float | None = None
    attributes: JsonObject = field(default_factory=dict)
    mention_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.label, "label")
        require_non_empty(self.entity_type, "entity_type")
        require_non_empty(self.extraction_revision_id, "extraction_revision_id")
        anchors = _ids(
            self.evidence_anchor_ids, "evidence_anchor_ids", non_empty=True
        )
        require_probability(self.extraction_confidence, "extraction_confidence")
        confidence = (
            float(self.extraction_confidence)
            if self.extraction_confidence is not None
            else None
        )
        attributes = freeze_json(self.attributes, field="mention.attributes")
        object.__setattr__(self, "evidence_anchor_ids", anchors)
        object.__setattr__(self, "extraction_confidence", confidence)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(
            self,
            "mention_id",
            stable_id(
                "mention",
                self.label,
                self.entity_type,
                self.extraction_revision_id,
                anchors,
                confidence,
                attributes,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "mention_id": self.mention_id,
            "label": self.label,
            "entity_type": self.entity_type,
            "extraction_revision_id": self.extraction_revision_id,
            "evidence_anchor_ids": list(self.evidence_anchor_ids),
            "extraction_confidence": self.extraction_confidence,
            "attributes": json_ready(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "mention_id",
                "label",
                "entity_type",
                "extraction_revision_id",
                "evidence_anchor_ids",
                "extraction_confidence",
                "attributes",
            },
            dto="EntityMention",
        )
        result = cls(
            label=payload.get("label"),  # type: ignore[arg-type]
            entity_type=payload.get("entity_type"),  # type: ignore[arg-type]
            extraction_revision_id=payload.get("extraction_revision_id"),  # type: ignore[arg-type]
            evidence_anchor_ids=tuple(
                require_sequence(
                    payload.get("evidence_anchor_ids", []), "evidence_anchor_ids"
                )
            ),  # type: ignore[arg-type]
            extraction_confidence=payload.get("extraction_confidence"),  # type: ignore[arg-type]
            attributes=require_mapping(
                payload.get("attributes", {}), "mention.attributes"
            ),
        )
        _verify_generated(payload, "mention_id", result.mention_id)
        return result


@dataclass(frozen=True, slots=True)
class RelationAssertion(JsonDTOMixin):
    """One directed, evidence-backed extraction assertion.

    Direction, relation label, evidence and optional producer assertion key are
    all part of identity.  Consequently opposite edges and parallel assertions
    are never collapsed by an undirected ``(source, target)`` key.
    """

    source_mention_id: str
    target_mention_id: str
    relation_type: str
    extraction_revision_id: str
    evidence_anchor_ids: tuple[str, ...]
    extraction_confidence: float | None = None
    assertion_key: str | None = None
    attributes: JsonObject = field(default_factory=dict)
    relation_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.source_mention_id, "source_mention_id")
        require_non_empty(self.target_mention_id, "target_mention_id")
        require_non_empty(self.relation_type, "relation_type")
        require_non_empty(self.extraction_revision_id, "extraction_revision_id")
        anchors = _ids(
            self.evidence_anchor_ids, "evidence_anchor_ids", non_empty=True
        )
        require_probability(self.extraction_confidence, "extraction_confidence")
        confidence = (
            float(self.extraction_confidence)
            if self.extraction_confidence is not None
            else None
        )
        assertion_key = _optional_id(self.assertion_key, "assertion_key")
        attributes = freeze_json(self.attributes, field="relation.attributes")
        object.__setattr__(self, "evidence_anchor_ids", anchors)
        object.__setattr__(self, "extraction_confidence", confidence)
        object.__setattr__(self, "assertion_key", assertion_key)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(
            self,
            "relation_id",
            stable_id(
                "kgrel",
                self.source_mention_id,
                self.target_mention_id,
                self.relation_type,
                self.extraction_revision_id,
                anchors,
                confidence,
                assertion_key,
                attributes,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "relation_id": self.relation_id,
            "source_mention_id": self.source_mention_id,
            "target_mention_id": self.target_mention_id,
            "relation_type": self.relation_type,
            "extraction_revision_id": self.extraction_revision_id,
            "evidence_anchor_ids": list(self.evidence_anchor_ids),
            "extraction_confidence": self.extraction_confidence,
            "assertion_key": self.assertion_key,
            "attributes": json_ready(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "relation_id",
                "source_mention_id",
                "target_mention_id",
                "relation_type",
                "extraction_revision_id",
                "evidence_anchor_ids",
                "extraction_confidence",
                "assertion_key",
                "attributes",
            },
            dto="RelationAssertion",
        )
        result = cls(
            source_mention_id=payload.get("source_mention_id"),  # type: ignore[arg-type]
            target_mention_id=payload.get("target_mention_id"),  # type: ignore[arg-type]
            relation_type=payload.get("relation_type"),  # type: ignore[arg-type]
            extraction_revision_id=payload.get("extraction_revision_id"),  # type: ignore[arg-type]
            evidence_anchor_ids=tuple(
                require_sequence(
                    payload.get("evidence_anchor_ids", []), "evidence_anchor_ids"
                )
            ),  # type: ignore[arg-type]
            extraction_confidence=payload.get("extraction_confidence"),  # type: ignore[arg-type]
            assertion_key=payload.get("assertion_key"),  # type: ignore[arg-type]
            attributes=require_mapping(
                payload.get("attributes", {}), "relation.attributes"
            ),
        )
        _verify_generated(payload, "relation_id", result.relation_id)
        return result


@dataclass(frozen=True, slots=True)
class GTLink(JsonDTOMixin):
    """One authoritative mapping from a graph object (G) to a tree node (T)."""

    graph_object_id: str
    tree_node_id: str
    role: str
    evidence_anchor_ids: tuple[str, ...]
    link_confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)
    gt_link_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_non_empty(self.graph_object_id, "graph_object_id")
        require_non_empty(self.tree_node_id, "tree_node_id")
        require_non_empty(self.role, "role")
        anchors = _ids(
            self.evidence_anchor_ids, "evidence_anchor_ids", non_empty=True
        )
        require_probability(self.link_confidence, "link_confidence")
        confidence = (
            float(self.link_confidence) if self.link_confidence is not None else None
        )
        metadata = freeze_json(self.metadata, field="gt_link.metadata")
        object.__setattr__(self, "evidence_anchor_ids", anchors)
        object.__setattr__(self, "link_confidence", confidence)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "gt_link_id",
            stable_id(
                "gtlink",
                self.graph_object_id,
                self.tree_node_id,
                self.role,
                anchors,
                confidence,
                metadata,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "gt_link_id": self.gt_link_id,
            "graph_object_id": self.graph_object_id,
            "tree_node_id": self.tree_node_id,
            "role": self.role,
            "evidence_anchor_ids": list(self.evidence_anchor_ids),
            "link_confidence": self.link_confidence,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "gt_link_id",
                "graph_object_id",
                "tree_node_id",
                "role",
                "evidence_anchor_ids",
                "link_confidence",
                "metadata",
            },
            dto="GTLink",
        )
        result = cls(
            graph_object_id=payload.get("graph_object_id"),  # type: ignore[arg-type]
            tree_node_id=payload.get("tree_node_id"),  # type: ignore[arg-type]
            role=payload.get("role"),  # type: ignore[arg-type]
            evidence_anchor_ids=tuple(
                require_sequence(
                    payload.get("evidence_anchor_ids", []), "evidence_anchor_ids"
                )
            ),  # type: ignore[arg-type]
            link_confidence=payload.get("link_confidence"),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "gt_link.metadata"),
        )
        _verify_generated(payload, "gt_link_id", result.gt_link_id)
        return result


@dataclass(frozen=True, slots=True)
class EvidenceGraphRelease(JsonDTOMixin):
    """Integrity-checked candidate G/M release bound to declared tree inputs.

    The generic ``tree_artifact_sha256`` field identifies whatever tree
    artifact the producer declares.  It does not by itself prove complete
    topology/content or adoption as the active BookRAG ``T``.  Retrieval must
    require a separately authorized release bound to an adopted full-tree
    manifest.
    """

    document_id: str
    document_version_id: str
    source_sha256: str
    build_spec_sha256: str
    canonical_artifact_sha256: str
    tree_revision_id: str
    tree_artifact_sha256: str
    graph_revision_id: str
    graph_artifact_sha256: str
    mapping_revision_id: str
    mapping_artifact_sha256: str
    canonical_revision_ids: tuple[str, ...]
    representation_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    tree_node_ids: tuple[str, ...]
    anchors: tuple[EvidenceAnchor, ...]
    mentions: tuple[EntityMention, ...]
    relations: tuple[RelationAssertion, ...]
    gt_links: tuple[GTLink, ...]
    metadata: JsonObject = field(default_factory=dict)
    content_sha256: str = field(init=False)
    release_id: str = field(init=False)

    SCHEMA_VERSION: ClassVar[str] = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "document_id",
            "document_version_id",
            "tree_revision_id",
            "graph_revision_id",
            "mapping_revision_id",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        require_sha256(self.source_sha256, "source_sha256")
        require_sha256(self.build_spec_sha256, "build_spec_sha256")
        require_sha256(
            self.canonical_artifact_sha256, "canonical_artifact_sha256"
        )
        require_sha256(self.tree_artifact_sha256, "tree_artifact_sha256")
        require_sha256(self.graph_artifact_sha256, "graph_artifact_sha256")
        require_sha256(self.mapping_artifact_sha256, "mapping_artifact_sha256")
        canonical_ids = _ids(
            self.canonical_revision_ids, "canonical_revision_ids", non_empty=True
        )
        representation_ids = _ids(self.representation_ids, "representation_ids")
        asset_ids = _ids(self.asset_ids, "asset_ids")
        if not representation_ids and not asset_ids:
            raise ContractValidationError(
                "release requires at least one declared representation or asset"
            )
        tree_node_ids = _ids(self.tree_node_ids, "tree_node_ids", non_empty=True)
        anchors = tuple(sorted(self.anchors, key=lambda item: item.anchor_id))
        mentions = tuple(sorted(self.mentions, key=lambda item: item.mention_id))
        relations = tuple(sorted(self.relations, key=lambda item: item.relation_id))
        gt_links = tuple(sorted(self.gt_links, key=lambda item: item.gt_link_id))
        self._reject_duplicate_ids(anchors, "anchor_id", "anchors")
        self._reject_duplicate_ids(mentions, "mention_id", "mentions")
        self._reject_duplicate_ids(relations, "relation_id", "relations")
        self._reject_duplicate_ids(gt_links, "gt_link_id", "gt_links")

        canonical_set = set(canonical_ids)
        representation_set = set(representation_ids)
        asset_set = set(asset_ids)
        anchor_set = {item.anchor_id for item in anchors}
        mention_set = {item.mention_id for item in mentions}
        relation_set = {item.relation_id for item in relations}
        tree_set = set(tree_node_ids)
        for anchor in anchors:
            if anchor.canonical_revision_id not in canonical_set:
                raise ContractValidationError(
                    f"anchor {anchor.anchor_id} has dangling canonical_revision_id"
                )
            if (
                anchor.representation_id is not None
                and anchor.representation_id not in representation_set
            ):
                raise ContractValidationError(
                    f"anchor {anchor.anchor_id} has dangling representation_id"
                )
            if anchor.asset_id is not None and anchor.asset_id not in asset_set:
                raise ContractValidationError(
                    f"anchor {anchor.anchor_id} has dangling asset_id"
                )
        for mention in mentions:
            self._require_subset(
                mention.evidence_anchor_ids,
                anchor_set,
                f"mention {mention.mention_id} evidence_anchor_ids",
            )
        for relation in relations:
            if relation.source_mention_id not in mention_set:
                raise ContractValidationError(
                    f"relation {relation.relation_id} has dangling source_mention_id"
                )
            if relation.target_mention_id not in mention_set:
                raise ContractValidationError(
                    f"relation {relation.relation_id} has dangling target_mention_id"
                )
            self._require_subset(
                relation.evidence_anchor_ids,
                anchor_set,
                f"relation {relation.relation_id} evidence_anchor_ids",
            )
        graph_objects = mention_set | relation_set
        link_keys: set[tuple[str, str, str]] = set()
        for link in gt_links:
            if link.graph_object_id not in graph_objects:
                raise ContractValidationError(
                    f"GTLink {link.gt_link_id} has dangling graph_object_id"
                )
            if link.tree_node_id not in tree_set:
                raise ContractValidationError(
                    f"GTLink {link.gt_link_id} has dangling tree_node_id"
                )
            self._require_subset(
                link.evidence_anchor_ids,
                anchor_set,
                f"GTLink {link.gt_link_id} evidence_anchor_ids",
            )
            semantic_key = (link.graph_object_id, link.tree_node_id, link.role)
            if semantic_key in link_keys:
                raise ContractValidationError(
                    "GTLinks must have one authoritative row per graph/tree/role key"
                )
            link_keys.add(semantic_key)

        metadata = freeze_json(self.metadata, field="release.metadata")
        object.__setattr__(self, "canonical_revision_ids", canonical_ids)
        object.__setattr__(self, "representation_ids", representation_ids)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "tree_node_ids", tree_node_ids)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "mentions", mentions)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "gt_links", gt_links)
        object.__setattr__(self, "metadata", metadata)
        digest = content_sha256(self._content_dict())
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(
            self,
            "release_id",
            stable_id(
                "kgrelease",
                self.document_version_id,
                self.build_spec_sha256,
                self.tree_revision_id,
                self.graph_revision_id,
                self.mapping_revision_id,
                digest,
            ),
        )

    @staticmethod
    def _reject_duplicate_ids(
        values: Sequence[object], attribute: str, field_name: str
    ) -> None:
        identifiers = [getattr(value, attribute) for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ContractValidationError(f"{field_name} contains duplicate IDs")

    @staticmethod
    def _require_subset(
        values: Sequence[str], declared: set[str], field_name: str
    ) -> None:
        missing = sorted(set(values) - declared)
        if missing:
            raise ContractValidationError(
                f"{field_name} contains dangling ID(s): {', '.join(missing)}"
            )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "build_spec_sha256": self.build_spec_sha256,
            "canonical_artifact_sha256": self.canonical_artifact_sha256,
            "tree_revision_id": self.tree_revision_id,
            "tree_artifact_sha256": self.tree_artifact_sha256,
            "graph_revision_id": self.graph_revision_id,
            "graph_artifact_sha256": self.graph_artifact_sha256,
            "mapping_revision_id": self.mapping_revision_id,
            "mapping_artifact_sha256": self.mapping_artifact_sha256,
            "canonical_revision_ids": list(self.canonical_revision_ids),
            "representation_ids": list(self.representation_ids),
            "asset_ids": list(self.asset_ids),
            "tree_node_ids": list(self.tree_node_ids),
            "anchors": [item.to_dict() for item in self.anchors],
            "mentions": [item.to_dict() for item in self.mentions],
            "relations": [item.to_dict() for item in self.relations],
            "gt_links": [item.to_dict() for item in self.gt_links],
            "metadata": json_ready(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "content_sha256": self.content_sha256,
            "release_id": self.release_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        require_schema_version(payload)
        reject_unknown_fields(
            payload,
            {
                "schema_version",
                "release_id",
                "content_sha256",
                "document_id",
                "document_version_id",
                "source_sha256",
                "build_spec_sha256",
                "canonical_artifact_sha256",
                "tree_revision_id",
                "tree_artifact_sha256",
                "graph_revision_id",
                "graph_artifact_sha256",
                "mapping_revision_id",
                "mapping_artifact_sha256",
                "canonical_revision_ids",
                "representation_ids",
                "asset_ids",
                "tree_node_ids",
                "anchors",
                "mentions",
                "relations",
                "gt_links",
                "metadata",
            },
            dto="EvidenceGraphRelease",
        )

        def dto_sequence(field_name: str, dto_type: type[JsonDTOMixin]) -> tuple[object, ...]:
            values = require_sequence(payload.get(field_name, []), field_name)
            return tuple(
                dto_type.from_dict(require_mapping(value, f"{field_name}[{index}]"))
                for index, value in enumerate(values)
            )

        result = cls(
            document_id=payload.get("document_id"),  # type: ignore[arg-type]
            document_version_id=payload.get("document_version_id"),  # type: ignore[arg-type]
            source_sha256=payload.get("source_sha256"),  # type: ignore[arg-type]
            build_spec_sha256=payload.get("build_spec_sha256"),  # type: ignore[arg-type]
            canonical_artifact_sha256=payload.get("canonical_artifact_sha256"),  # type: ignore[arg-type]
            tree_revision_id=payload.get("tree_revision_id"),  # type: ignore[arg-type]
            tree_artifact_sha256=payload.get("tree_artifact_sha256"),  # type: ignore[arg-type]
            graph_revision_id=payload.get("graph_revision_id"),  # type: ignore[arg-type]
            graph_artifact_sha256=payload.get("graph_artifact_sha256"),  # type: ignore[arg-type]
            mapping_revision_id=payload.get("mapping_revision_id"),  # type: ignore[arg-type]
            mapping_artifact_sha256=payload.get("mapping_artifact_sha256"),  # type: ignore[arg-type]
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
            tree_node_ids=tuple(
                require_sequence(payload.get("tree_node_ids", []), "tree_node_ids")
            ),  # type: ignore[arg-type]
            anchors=dto_sequence("anchors", EvidenceAnchor),  # type: ignore[arg-type]
            mentions=dto_sequence("mentions", EntityMention),  # type: ignore[arg-type]
            relations=dto_sequence("relations", RelationAssertion),  # type: ignore[arg-type]
            gt_links=dto_sequence("gt_links", GTLink),  # type: ignore[arg-type]
            metadata=require_mapping(payload.get("metadata", {}), "release.metadata"),
        )
        _verify_generated(payload, "content_sha256", result.content_sha256)
        _verify_generated(payload, "release_id", result.release_id)
        return result


__all__ = [
    "EntityMention",
    "EvidenceAnchor",
    "EvidenceGraphRelease",
    "GTLink",
    "RelationAssertion",
]
