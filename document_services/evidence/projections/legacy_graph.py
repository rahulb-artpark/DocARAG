"""Checked, read-only projection from evidence releases to legacy BookRAG JSON.

The active BookRAG graph uses an undirected ``networkx.Graph`` and therefore
cannot faithfully represent directed or parallel assertions.  This module
does not pretend otherwise.  Every source assertion remains embedded in an
``assertions`` list, while an explicit loss report describes the semantics a
legacy consumer will not honor.  The default policy rejects such a projection;
callers must deliberately opt in to a marked, read-only compatibility view.

The result contains ordinary JSON DTOs only.  It imports neither NetworkX nor
BookRAG classes, and the evidence release remains the sole authority.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math

from document_services.domain.base import (
    ContractValidationError,
    JsonObject,
    canonical_json_bytes,
    freeze_json,
    json_ready,
)
from document_services.domain.knowledge import EvidenceGraphRelease, GTLink, RelationAssertion


LEGACY_GRAPH_PROJECTION_SCHEMA_VERSION = "docarag.bookrag-legacy-projection.v2"

_MAX_ABS_LEGACY_WEIGHT = 1_000_000_000_000
_HARD_MAX_PROJECTION_RECORDS = 1_000_000
_HARD_MAX_PROJECTION_OUTPUT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LegacyProjectionLimits:
    """Explicit resource envelope for the inspection-only compatibility view.

    These defaults are intentionally below the evidence-release contract's
    broad interchange limits.  A legacy ``networkx.Graph`` projection is a
    diagnostic artifact, not the scalable retrieval representation.
    """

    max_nodes: int = 100_000
    max_relations: int = 250_000
    max_anchors: int = 250_000
    max_gt_links: int = 500_000
    max_tree_nodes: int = 100_000
    max_grouped_edges: int = 250_000
    max_anchor_references: int = 1_000_000
    max_release_metadata_bytes: int = 8 * 1024 * 1024
    max_estimated_output_bytes: int = 128 * 1024 * 1024
    max_output_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        record_fields = (
            "max_nodes",
            "max_relations",
            "max_anchors",
            "max_gt_links",
            "max_tree_nodes",
            "max_grouped_edges",
            "max_anchor_references",
        )
        byte_fields = (
            "max_release_metadata_bytes",
            "max_estimated_output_bytes",
            "max_output_bytes",
        )
        for field_name in (*record_fields, *byte_fields):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractValidationError(
                    f"legacy projection {field_name} must be a positive integer"
                )
        for field_name in record_fields:
            if getattr(self, field_name) > _HARD_MAX_PROJECTION_RECORDS:
                raise ContractValidationError(
                    f"legacy projection {field_name} exceeds the hard safety cap"
                )
        for field_name in byte_fields:
            if getattr(self, field_name) > _HARD_MAX_PROJECTION_OUTPUT_BYTES:
                raise ContractValidationError(
                    f"legacy projection {field_name} exceeds the hard byte cap"
                )
        if self.max_estimated_output_bytes > self.max_output_bytes:
            raise ContractValidationError(
                "max_estimated_output_bytes cannot exceed max_output_bytes"
            )


class LegacyGraphProjectionLimitError(ValueError):
    """The requested compatibility view exceeds its declared resource envelope."""


class LegacyProjectionPolicy(StrEnum):
    """Whether legacy semantic loss is rejected or explicitly marked."""

    REJECT = "reject"
    MARK = "mark"


class LegacyProjectionLossCode(StrEnum):
    """Legacy ``nx.Graph`` semantics absent from the source release."""

    DIRECTED_ASSERTIONS = "directed_assertions"
    PARALLEL_ASSERTIONS = "parallel_assertions"
    OPPOSITE_DIRECTIONS = "opposite_directions"
    MULTIPLE_RELATION_TYPES = "multiple_relation_types"
    RELATION_NAME_COLLAPSE = "relation_name_collapse"
    RELATION_DESCRIPTION_COLLAPSE = "relation_description_collapse"
    RELATION_WEIGHT_COLLAPSE = "relation_weight_collapse"
    INVALID_EDGE_COMPATIBILITY_FIELDS = "invalid_edge_compatibility_fields"
    RELATION_TREE_LINKS = "relation_tree_links"
    OPAQUE_NODE_IDS = "opaque_node_ids"


@dataclass(frozen=True, slots=True)
class LegacyProjectionLoss:
    code: LegacyProjectionLossCode
    message: str
    affected_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", LegacyProjectionLossCode(self.code))
        if not isinstance(self.message, str) or not self.message.strip():
            raise ContractValidationError("legacy projection loss message is empty")
        identifiers = tuple(sorted(set(self.affected_ids)))
        if not identifiers or any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise ContractValidationError(
                "legacy projection loss requires non-empty affected IDs"
            )
        object.__setattr__(self, "affected_ids", identifiers)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "affected_ids": list(self.affected_ids),
        }


class LossyLegacyGraphProjectionError(ValueError):
    """The requested projection would change legacy graph semantics."""

    def __init__(self, losses: tuple[LegacyProjectionLoss, ...]) -> None:
        self.losses = losses
        codes = ", ".join(loss.code.value for loss in losses)
        super().__init__(
            f"legacy BookRAG projection is lossy ({codes}); use policy='mark' "
            "only for a read-only compatibility view"
        )


class LegacyGraphProjectionCompatibilityError(ValueError):
    """The release cannot be linked to BookRAG's integer tree IDs exactly."""


@dataclass(frozen=True, slots=True)
class LegacyGraphProjectionResult:
    """Immutable, JSON-only derived view plus an explicit loss decision."""

    release_id: str
    release_content_sha256: str
    losses: tuple[LegacyProjectionLoss, ...]
    _payload: JsonObject

    def __post_init__(self) -> None:
        object.__setattr__(self, "losses", tuple(self.losses))
        object.__setattr__(
            self, "_payload", freeze_json(self._payload, field="legacy_projection")
        )

    @property
    def lossy(self) -> bool:
        return bool(self.losses)

    @property
    def retrieval_safe(self) -> bool:
        """The mention-ID compatibility graph is never a retrieval index."""

        return False

    def to_dict(self) -> dict[str, object]:
        payload = json_ready(self._payload)
        assert isinstance(payload, dict)  # frozen from a mapping above
        return payload

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def to_retrieval_dict(self) -> dict[str, object]:
        """Return the graph only when legacy retrieval cannot change semantics.

        ``policy='mark'`` deliberately creates an inspectable compatibility
        artifact.  It is not permission to load a lossy artifact into active
        retrieval; callers must pass this fail-closed boundary instead.
        """

        raise LossyLegacyGraphProjectionError(self.losses)


def _validated_release(release: EvidenceGraphRelease) -> EvidenceGraphRelease:
    if not isinstance(release, EvidenceGraphRelease):
        raise ContractValidationError("release must be an EvidenceGraphRelease")
    # Re-run generated-ID and referential-integrity validation at the boundary.
    return EvidenceGraphRelease.from_dict(release.to_dict())


def _description(attributes: Mapping[str, object]) -> str:
    value = attributes.get("description", "")
    return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class _LegacyRelationFields:
    relation_name: str
    description: str
    weight: int | float
    relation_name_valid: bool
    description_valid: bool
    weight_valid: bool

    @property
    def valid(self) -> bool:
        return (
            self.relation_name_valid
            and self.description_valid
            and self.weight_valid
        )


def _legacy_relation_fields(relation: RelationAssertion) -> _LegacyRelationFields:
    """Recover exact BookRAG fields preserved by the release builder.

    Builder-produced releases retain the original legacy name (including the
    meaningful empty string), description and numeric weight in attributes.
    Other producers may omit them, in which case the canonical type,
    extraction confidence and empty description are lossless compatibility
    defaults.  Present-but-invalid compatibility fields are never trusted.
    """

    attributes = relation.attributes
    relation_name_valid = True
    description_valid = True
    weight_valid = True

    if "legacy_relation_name" in attributes:
        raw_name = attributes["legacy_relation_name"]
        if isinstance(raw_name, str):
            relation_name = raw_name
        else:
            relation_name_valid = False
            relation_name = relation.relation_type
    else:
        relation_name = relation.relation_type

    if "description" in attributes:
        raw_description = attributes["description"]
        if isinstance(raw_description, str):
            description = raw_description
        else:
            description_valid = False
            description = ""
    else:
        description = ""

    if "weight" in attributes:
        raw_weight = attributes["weight"]
        if _valid_legacy_weight(raw_weight):
            weight = raw_weight
        else:
            weight_valid = False
            weight = (
                relation.extraction_confidence
                if relation.extraction_confidence is not None
                else 0.0
            )
    else:
        weight = (
            relation.extraction_confidence
            if relation.extraction_confidence is not None
            else 0.0
        )
    return _LegacyRelationFields(
        relation_name=relation_name,
        description=description,
        weight=weight,
        relation_name_valid=relation_name_valid,
        description_valid=description_valid,
        weight_valid=weight_valid,
    )


def _valid_legacy_weight(value: object) -> bool:
    """Validate without coercing an unbounded integer to ``float``.

    ``float(10**400)`` raises ``OverflowError``.  Integers are already finite,
    so compare them directly; floats additionally require IEEE finiteness.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return abs(value) <= _MAX_ABS_LEGACY_WEIGHT


def _tree_aliases(
    tree_node_ids: tuple[str, ...],
    supplied: Mapping[str, int] | None,
) -> dict[str, int]:
    """Return a total bijection to the integer IDs required by BookRAG."""

    declared = set(tree_node_ids)
    if supplied is None:
        aliases: dict[str, int] = {}
        for tree_node_id in tree_node_ids:
            try:
                alias = int(tree_node_id)
            except ValueError as exc:
                raise LegacyGraphProjectionCompatibilityError(
                    "opaque tree_node_ids require an explicit tree_node_aliases "
                    "mapping to existing BookRAG integer node IDs"
                ) from exc
            if alias < 0 or str(alias) != tree_node_id:
                raise LegacyGraphProjectionCompatibilityError(
                    "tree_node_ids must be canonical non-negative decimal integers "
                    "unless tree_node_aliases is supplied"
                )
            aliases[tree_node_id] = alias
    else:
        if not isinstance(supplied, Mapping):
            raise LegacyGraphProjectionCompatibilityError(
                "tree_node_aliases must be a mapping from release IDs to integers"
            )
        if any(not isinstance(key, str) for key in supplied):
            raise LegacyGraphProjectionCompatibilityError(
                "tree_node_aliases keys must be release tree-node IDs"
            )
        actual = set(supplied)
        if actual != declared:
            missing = sorted(declared - actual)
            extra = sorted(actual - declared)
            raise LegacyGraphProjectionCompatibilityError(
                "tree_node_aliases must cover exactly the release tree nodes "
                f"(missing={missing}, extra={extra})"
            )
        aliases = {}
        for tree_node_id in tree_node_ids:
            alias = supplied[tree_node_id]
            if isinstance(alias, bool) or not isinstance(alias, int) or alias < 0:
                raise LegacyGraphProjectionCompatibilityError(
                    "tree_node_aliases values must be non-negative integers"
                )
            aliases[tree_node_id] = alias
    if len(set(aliases.values())) != len(aliases):
        raise LegacyGraphProjectionCompatibilityError(
            "tree_node_aliases must be one-to-one; alias collisions lose T/G links"
        )
    return aliases


def _projection_policy(value: LegacyProjectionPolicy | str) -> LegacyProjectionPolicy:
    try:
        return LegacyProjectionPolicy(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in LegacyProjectionPolicy)
        raise ContractValidationError(
            f"legacy projection policy must be one of: {allowed}"
        ) from exc


def _record_json_size(value: object, *, kind: str) -> int:
    try:
        to_dict = getattr(value, "to_dict")
        return len(canonical_json_bytes(to_dict()))
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        raise LegacyGraphProjectionLimitError(
            f"legacy projection cannot safely encode {kind}"
        ) from exc


def _json_size(value: object, *, kind: str) -> int:
    try:
        return len(canonical_json_bytes(value))
    except (ContractValidationError, OverflowError, TypeError, ValueError) as exc:
        raise LegacyGraphProjectionLimitError(
            f"legacy projection cannot safely encode {kind}"
        ) from exc


def _preflight_release(
    release: EvidenceGraphRelease,
    limits: LegacyProjectionLimits,
) -> None:
    """Reject oversized input before DTO cloning or compatibility expansion."""

    if not isinstance(release, EvidenceGraphRelease):
        raise ContractValidationError("release must be an EvidenceGraphRelease")
    counts = {
        "nodes": (len(release.mentions), limits.max_nodes),
        "relations": (len(release.relations), limits.max_relations),
        "anchors": (len(release.anchors), limits.max_anchors),
        "GT-links": (len(release.gt_links), limits.max_gt_links),
        "tree nodes": (len(release.tree_node_ids), limits.max_tree_nodes),
    }
    for label, (actual, maximum) in counts.items():
        if actual > maximum:
            raise LegacyGraphProjectionLimitError(
                f"legacy projection {label} count {actual} exceeds limit {maximum}"
            )

    # Release metadata is intentionally omitted from the compatibility payload,
    # but ``_validated_release`` still clones and freezes it.  Bound it before
    # that clone so ignored diagnostic metadata cannot create an unaccounted
    # multi-megabyte allocation.
    metadata_bytes = _json_size(release.metadata, kind="release metadata")
    if metadata_bytes > limits.max_release_metadata_bytes:
        raise LegacyGraphProjectionLimitError(
            "legacy projection release metadata size "
            f"{metadata_bytes} exceeds limit {limits.max_release_metadata_bytes}"
        )

    grouped_edges = {
        tuple(sorted((relation.source_mention_id, relation.target_mention_id)))
        for relation in release.relations
    }
    if len(grouped_edges) > limits.max_grouped_edges:
        raise LegacyGraphProjectionLimitError(
            "legacy projection grouped edge count "
            f"{len(grouped_edges)} exceeds limit {limits.max_grouped_edges}"
        )

    anchor_references = sum(
        len(item.evidence_anchor_ids)
        for collection in (release.mentions, release.relations, release.gt_links)
        for item in collection
    )
    if anchor_references > limits.max_anchor_references:
        raise LegacyGraphProjectionLimitError(
            "legacy projection anchor reference count "
            f"{anchor_references} exceeds limit {limits.max_anchor_references}"
        )

    # The v2 payload stores each anchor DTO once and uses IDs everywhere else.
    # Add conservative per-record structural overhead and repeated loss-ID
    # allowance while measuring each source DTO independently, never building a
    # second whole release in memory during this preflight.
    estimated = 32 * 1024 + metadata_bytes
    categories = (
        (release.anchors, "anchor", 256),
        (release.mentions, "mention", 768),
        (release.relations, "relation", 1536),
        (release.gt_links, "GT-link", 512),
    )
    for records, kind, overhead in categories:
        for record in records:
            estimated += _record_json_size(record, kind=kind) + overhead
            if estimated > limits.max_estimated_output_bytes:
                raise LegacyGraphProjectionLimitError(
                    "legacy projection estimated output exceeds limit "
                    f"{limits.max_estimated_output_bytes} bytes"
                )
    estimated += sum(len(node_id.encode("utf-8")) + 128 for node_id in release.tree_node_ids)
    estimated += anchor_references * 96
    # An assertion ID can occur in several independent semantic-loss records.
    estimated += len(release.relations) * 8 * 96
    estimated += len(release.mentions) * 96
    if estimated > limits.max_estimated_output_bytes:
        raise LegacyGraphProjectionLimitError(
            "legacy projection estimated output exceeds limit "
            f"{limits.max_estimated_output_bytes} bytes"
        )


def _prescan_losses(release: EvidenceGraphRelease) -> tuple[LegacyProjectionLoss, ...]:
    """Find every known semantic loss without expanding the JSON projection."""

    losses: list[LegacyProjectionLoss] = [
        LegacyProjectionLoss(
            code=LegacyProjectionLossCode.OPAQUE_NODE_IDS,
            message=(
                "projected graph node IDs are evidence mention IDs, while the "
                "legacy BookRAG retriever addresses nodes by its Name/Type lookup "
                "key; this view is inspection-only"
            ),
            affected_ids=tuple(mention.mention_id for mention in release.mentions)
            or (release.release_id,),
        )
    ]
    grouped: dict[tuple[str, str], list[RelationAssertion]] = defaultdict(list)
    for relation in release.relations:
        pair = tuple(sorted((relation.source_mention_id, relation.target_mention_id)))
        grouped[(pair[0], pair[1])].append(relation)

    directed_ids = tuple(
        relation.relation_id
        for relation in release.relations
        if relation.source_mention_id != relation.target_mention_id
    )
    if directed_ids:
        losses.append(
            LegacyProjectionLoss(
                code=LegacyProjectionLossCode.DIRECTED_ASSERTIONS,
                message=(
                    "legacy BookRAG traverses these directed assertions as "
                    "undirected edges"
                ),
                affected_ids=directed_ids,
            )
        )

    relation_ids = {relation.relation_id for relation in release.relations}
    relation_link_ids = tuple(
        link.gt_link_id
        for link in release.gt_links
        if link.graph_object_id in relation_ids
    )
    if relation_link_ids:
        losses.append(
            LegacyProjectionLoss(
                code=LegacyProjectionLossCode.RELATION_TREE_LINKS,
                message=(
                    "legacy tree2kg maps entity nodes only; relation links are "
                    "retained in relation_tree2kg and assertion GT-links"
                ),
                affected_ids=relation_link_ids,
            )
        )

    for pair in sorted(grouped):
        relations = sorted(grouped[pair], key=lambda item: item.relation_id)
        identifiers = tuple(relation.relation_id for relation in relations)
        if len(relations) > 1:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.PARALLEL_ASSERTIONS,
                    message=(
                        "multiple source assertions share one legacy undirected "
                        "edge; all are retained in its assertions list"
                    ),
                    affected_ids=identifiers,
                )
            )
        directions = {
            (relation.source_mention_id, relation.target_mention_id)
            for relation in relations
        }
        if len(directions) > 1 and any(
            (target, source_id) in directions for source_id, target in directions
        ):
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.OPPOSITE_DIRECTIONS,
                    message=(
                        "opposite directed assertions share one legacy undirected "
                        "edge; their endpoints remain explicit in assertions"
                    ),
                    affected_ids=identifiers,
                )
            )
        if len({relation.relation_type for relation in relations}) > 1:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.MULTIPLE_RELATION_TYPES,
                    message=(
                        "different relation types share one legacy edge; each type "
                        "remains explicit in assertions"
                    ),
                    affected_ids=identifiers,
                )
            )

        fields = tuple(_legacy_relation_fields(relation) for relation in relations)
        invalid_ids = tuple(
            relation.relation_id
            for relation, compatibility in zip(relations, fields, strict=True)
            if not compatibility.valid
        )
        if invalid_ids:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.INVALID_EDGE_COMPATIBILITY_FIELDS,
                    message=(
                        "relation compatibility attributes are not bounded valid "
                        "legacy name/description/numeric-weight fields"
                    ),
                    affected_ids=invalid_ids,
                )
            )
        if len({field.relation_name for field in fields}) > 1:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.RELATION_NAME_COLLAPSE,
                    message=(
                        "parallel assertion relation names cannot occupy one legacy "
                        "edge field; exact values remain in assertions"
                    ),
                    affected_ids=identifiers,
                )
            )
        if len({field.description for field in fields}) > 1:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.RELATION_DESCRIPTION_COLLAPSE,
                    message=(
                        "parallel assertion descriptions cannot occupy one legacy "
                        "edge field; exact values remain in assertions"
                    ),
                    affected_ids=identifiers,
                )
            )
        if len({field.weight for field in fields}) > 1:
            losses.append(
                LegacyProjectionLoss(
                    code=LegacyProjectionLossCode.RELATION_WEIGHT_COLLAPSE,
                    message=(
                        "parallel assertion weights cannot occupy one legacy edge "
                        "field; exact values remain in assertions"
                    ),
                    affected_ids=identifiers,
                )
            )

    losses.sort(key=lambda loss: (loss.code.value, loss.affected_ids))
    return tuple(losses)


def project_bookrag_legacy_graph(
    release: EvidenceGraphRelease,
    *,
    policy: LegacyProjectionPolicy | str = LegacyProjectionPolicy.REJECT,
    tree_node_aliases: Mapping[str, int] | None = None,
    limits: LegacyProjectionLimits | None = None,
) -> LegacyGraphProjectionResult:
    """Project a release without silently discarding graph assertions.

    The ``graph`` and ``tree2kg`` members match BookRAG's persisted node-link
    shape.  Mention IDs are used as node IDs so equal labels/types are not
    accidentally entity-resolved.  Every grouped legacy edge carries the full
    directed assertion DTOs, evidence-anchor ID references and exact GT-links.
    A single top-level anchor catalog avoids repeated anchor DTO expansion.
    ``relation_tree2kg`` retains links the legacy entity-only map cannot express.

    Mention IDs cannot reproduce BookRAG's private Name/Type lookup keys, so
    every successful ``policy='mark'`` result is inspection-only and is never
    accepted by :meth:`LegacyGraphProjectionResult.to_retrieval_dict`.
    """

    selected_policy = _projection_policy(policy)
    selected_limits = limits if limits is not None else LegacyProjectionLimits()
    if not isinstance(selected_limits, LegacyProjectionLimits):
        raise ContractValidationError("limits must be LegacyProjectionLimits")
    _preflight_release(release, selected_limits)
    prescanned_losses = _prescan_losses(release)
    if selected_policy is LegacyProjectionPolicy.REJECT:
        # Fail before cloning the release or expanding any node/edge DTOs.
        raise LossyLegacyGraphProjectionError(prescanned_losses)

    source = _validated_release(release)
    aliases = _tree_aliases(source.tree_node_ids, tree_node_aliases)

    anchors = {anchor.anchor_id: anchor for anchor in source.anchors}
    mentions = {mention.mention_id: mention for mention in source.mentions}
    links_by_object: dict[str, list[GTLink]] = defaultdict(list)
    for link in source.gt_links:
        links_by_object[link.graph_object_id].append(link)

    nodes: list[dict[str, object]] = []
    mention_tree2kg: dict[str, set[str]] = {
        tree_node_id: set() for tree_node_id in source.tree_node_ids
    }
    relation_tree2kg: dict[str, set[str]] = {
        tree_node_id: set() for tree_node_id in source.tree_node_ids
    }
    captured_gt_link_ids: set[str] = set()
    for mention in source.mentions:
        mention_links = sorted(
            links_by_object.get(mention.mention_id, []),
            key=lambda item: item.gt_link_id,
        )
        for link in mention_links:
            mention_tree2kg[link.tree_node_id].add(mention.mention_id)
            captured_gt_link_ids.add(link.gt_link_id)
        nodes.append(
            {
                "id": mention.mention_id,
                "entity_name": mention.label,
                "entity_type": mention.entity_type,
                "description": _description(mention.attributes),
                "source_ids": sorted(
                    {aliases[link.tree_node_id] for link in mention_links}
                ),
                "mention": mention.to_dict(),
                "evidence_anchor_ids": list(mention.evidence_anchor_ids),
                "gt_links": [link.to_dict() for link in mention_links],
                "docarag_read_only": True,
            }
        )

    grouped: dict[tuple[str, str], list[RelationAssertion]] = defaultdict(list)
    for relation in source.relations:
        pair = tuple(sorted((relation.source_mention_id, relation.target_mention_id)))
        grouped[(pair[0], pair[1])].append(relation)

    edges: list[dict[str, object]] = []
    captured_relation_ids: set[str] = set()
    for pair in sorted(grouped):
        relations = sorted(grouped[pair], key=lambda item: item.relation_id)
        assertion_payloads: list[dict[str, object]] = []
        relation_source_ids: set[int] = set()
        legacy_fields: list[_LegacyRelationFields] = []
        for relation in relations:
            relation_links = sorted(
                links_by_object.get(relation.relation_id, []),
                key=lambda item: item.gt_link_id,
            )
            for link in relation_links:
                relation_tree2kg[link.tree_node_id].add(relation.relation_id)
                relation_source_ids.add(aliases[link.tree_node_id])
                captured_gt_link_ids.add(link.gt_link_id)
            fields = _legacy_relation_fields(relation)
            legacy_fields.append(fields)
            assertion_payloads.append(
                {
                    "relation": relation.to_dict(),
                    "legacy_compatibility_fields": {
                        "relation_name": fields.relation_name,
                        "description": fields.description,
                        "weight": fields.weight,
                        "exact": fields.valid,
                        "field_exactness": {
                            "relation_name": fields.relation_name_valid,
                            "description": fields.description_valid,
                            "weight": fields.weight_valid,
                        },
                    },
                    "evidence_anchor_ids": list(relation.evidence_anchor_ids),
                    "gt_links": [link.to_dict() for link in relation_links],
                }
            )
            captured_relation_ids.add(relation.relation_id)

        left, right = pair
        if len(relations) == 1:
            edge_source = relations[0].source_mention_id
            edge_target = relations[0].target_mention_id
        else:
            edge_source, edge_target = left, right
        source_mention = mentions[edge_source]
        target_mention = mentions[edge_target]

        names = {fields.relation_name for fields in legacy_fields}
        descriptions = {fields.description for fields in legacy_fields}
        weights = {fields.weight for fields in legacy_fields}
        name_exact = len(names) == 1 and all(
            fields.relation_name_valid for fields in legacy_fields
        )
        description_exact = (
            len(descriptions) == 1
            and all(fields.description_valid for fields in legacy_fields)
        )
        weight_exact = (
            len(weights) == 1
            and all(fields.weight_valid for fields in legacy_fields)
        )
        relation_name = next(iter(names)) if name_exact else "__see_assertions__"
        description = (
            next(iter(descriptions)) if description_exact else "__see_assertions__"
        )
        weight = next(iter(weights)) if weight_exact else 0.0
        edge_field_exact = name_exact and description_exact and weight_exact
        edges.append(
            {
                "source": edge_source,
                "target": edge_target,
                "src_entity_name": source_mention.label,
                "tgt_entity_name": target_mention.label,
                "relation_name": relation_name,
                "description": description,
                "weight": weight,
                "source_ids": sorted(relation_source_ids),
                "assertion_count": len(assertion_payloads),
                "assertions": assertion_payloads,
                "legacy_edge_fields_exact": edge_field_exact,
                "legacy_edge_field_exactness": {
                    "relation_name": name_exact,
                    "description": description_exact,
                    "weight": weight_exact,
                },
                "docarag_read_only": True,
            }
        )

    expected_relation_ids = {relation.relation_id for relation in source.relations}
    if captured_relation_ids != expected_relation_ids:
        raise RuntimeError("legacy projection failed to retain every relation assertion")
    expected_gt_link_ids = {link.gt_link_id for link in source.gt_links}
    if captured_gt_link_ids != expected_gt_link_ids:
        raise RuntimeError("legacy projection failed to retain every GT-link")

    loss_tuple = prescanned_losses
    retrieval_safe = False
    for edge in edges:
        edge["docarag_retrieval_safe"] = retrieval_safe

    payload: dict[str, object] = {
        "schema_version": LEGACY_GRAPH_PROJECTION_SCHEMA_VERSION,
        "authority": {
            "kind": "evidence_graph_release",
            "release_id": source.release_id,
            "release_content_sha256": source.content_sha256,
            "document_id": source.document_id,
            "document_version_id": source.document_version_id,
            "source_sha256": source.source_sha256,
            "build_spec_sha256": source.build_spec_sha256,
            "canonical_artifact_sha256": source.canonical_artifact_sha256,
            "tree_revision_id": source.tree_revision_id,
            "tree_artifact_sha256": source.tree_artifact_sha256,
            "graph_revision_id": source.graph_revision_id,
            "graph_artifact_sha256": source.graph_artifact_sha256,
            "mapping_revision_id": source.mapping_revision_id,
            "mapping_artifact_sha256": source.mapping_artifact_sha256,
        },
        "read_only_derived": True,
        "lossy": True,
        "losses": [loss.to_dict() for loss in loss_tuple],
        "retrieval_use": {
            "safe": retrieval_safe,
            "default_action": "reject",
            "authority_release_id": source.release_id,
            "marked_loss_is_not_retrieval_authorization": True,
        },
        "graph": {
            "directed": False,
            "multigraph": False,
            "graph": {
                "authority_release_id": source.release_id,
                "read_only_derived": True,
                "retrieval_safe": retrieval_safe,
            },
            "nodes": nodes,
            "links": edges,
        },
        "evidence_anchor_catalog": {
            anchor_id: anchors[anchor_id].to_dict() for anchor_id in sorted(anchors)
        },
        "tree2kg": {
            str(aliases[tree_node_id]): sorted(mention_ids)
            for tree_node_id, mention_ids in sorted(mention_tree2kg.items())
        },
        "relation_tree2kg": {
            str(aliases[tree_node_id]): sorted(relation_ids)
            for tree_node_id, relation_ids in sorted(relation_tree2kg.items())
        },
        "tree_node_aliases": {
            tree_node_id: aliases[tree_node_id] for tree_node_id in source.tree_node_ids
        },
        "variant": "evidence-sidecar-read-only",
    }
    try:
        output_size = len(canonical_json_bytes(payload))
    except (OverflowError, TypeError, ValueError) as exc:
        raise LegacyGraphProjectionLimitError(
            "legacy projection output cannot be encoded safely"
        ) from exc
    if output_size > selected_limits.max_output_bytes:
        raise LegacyGraphProjectionLimitError(
            "legacy projection encoded output size "
            f"{output_size} exceeds limit {selected_limits.max_output_bytes}"
        )
    return LegacyGraphProjectionResult(
        release_id=source.release_id,
        release_content_sha256=source.content_sha256,
        losses=loss_tuple,
        _payload=payload,
    )


__all__ = [
    "LEGACY_GRAPH_PROJECTION_SCHEMA_VERSION",
    "LegacyGraphProjectionLimitError",
    "LegacyGraphProjectionResult",
    "LegacyGraphProjectionCompatibilityError",
    "LegacyProjectionLimits",
    "LegacyProjectionLoss",
    "LegacyProjectionLossCode",
    "LegacyProjectionPolicy",
    "LossyLegacyGraphProjectionError",
    "project_bookrag_legacy_graph",
]
