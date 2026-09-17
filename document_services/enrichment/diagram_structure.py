"""Turn a recovered diagram into a canonical block-local semantic graph.

A diagram's meaning is its topology: which node connects to which, and in what
order a flow proceeds.  A prose description cannot answer "what feeds the
reconciliation step"; a typed node/edge graph can.  The canonical contract
already models these structures, so this maps an extracted graph onto them and
binds every node and edge to the exact crop it was read from.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging

from document_services.domain.base import ContractValidationError
from document_services.domain.evidence import (
    AssetRef,
    SemanticElement,
    SemanticElementKind,
    SemanticProducerKind,
    SemanticRelation,
    SemanticRelationKind,
    SemanticSourceSelector,
    SemanticStructureKind,
    SemanticSubstructure,
    normalize_semantic_edge_type,
)

log = logging.getLogger(__name__)

MAX_DIAGRAM_NODES = 64
MAX_DIAGRAM_EDGES = 128
MAX_LABEL_CHARS = 200

GRAPH_STRUCTURE_KINDS = (
    SemanticStructureKind.DIAGRAM,
    SemanticStructureKind.FLOWCHART,
)

_ELEMENT_KIND = {
    SemanticStructureKind.DIAGRAM: SemanticElementKind.DIAGRAM_ELEMENT,
    SemanticStructureKind.FLOWCHART: SemanticElementKind.FLOWCHART_ELEMENT,
}

_BRANCH_KIND = {
    "true": SemanticRelationKind.FLOW_BRANCH_TRUE,
    "false": SemanticRelationKind.FLOW_BRANCH_FALSE,
}


@dataclass(frozen=True, slots=True)
class DiagramNode:
    key: str
    label: str
    role: str | None = None
    inside: str | None = None


@dataclass(frozen=True, slots=True)
class DiagramEdge:
    source: str
    target: str
    label: str | None = None
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class DiagramGraph:
    kind: SemanticStructureKind
    nodes: tuple[DiagramNode, ...]
    edges: tuple[DiagramEdge, ...]


DiagramExtractor = Callable[[bytes, str], DiagramGraph | None]


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_LABEL_CHARS]


def _relation_kind(
    graph_kind: SemanticStructureKind, edge: DiagramEdge
) -> SemanticRelationKind:
    if graph_kind is SemanticStructureKind.DIAGRAM:
        return SemanticRelationKind.DIAGRAM_CONNECTS_TO
    branch = _clean(edge.branch).lower()
    return _BRANCH_KIND.get(branch, SemanticRelationKind.FLOW_NEXT)


def diagram_substructure(
    graph: DiagramGraph,
    *,
    asset: AssetRef,
    source_observation_ids: Sequence[str],
) -> SemanticSubstructure | None:
    """Return the graph as a source-bound substructure, or None if unusable."""

    if graph.kind not in _ELEMENT_KIND:
        raise ValueError(f"unsupported diagram structure kind: {graph.kind!r}")
    selector = SemanticSourceSelector(asset_id=asset.asset_id)
    element_kind = _ELEMENT_KIND[graph.kind]
    # A drawing can spend two boxes on one thing -- an outer boundary and an
    # inner label box with the same text.  Same-label nodes fold into one
    # element, and every key that named one of them resolves to it.
    elements: dict[str, SemanticElement] = {}
    by_label: dict[str, SemanticElement] = {}
    for node in graph.nodes[:MAX_DIAGRAM_NODES]:
        key = _clean(node.key)
        label = _clean(node.label)
        if not key or key in elements or not label:
            continue
        existing = by_label.get(label)
        if existing is not None:
            elements[key] = existing
            continue
        element = SemanticElement(
            local_key=key,
            kind=element_kind,
            source_selectors=(selector,),
            content=label,
            role=_clean(node.role) or None,
        )
        elements[key] = element
        by_label[label] = element
    if not elements:
        return None

    relations: list[SemanticRelation] = []
    seen: set[tuple[str, str, str]] = set()
    # Containment is drawn as a box inside a box, not as an arrow, so it
    # arrives on the node and leaves as a diagram_contains relation.  Mutual
    # or self containment is impossible to draw, so such claims are dropped.
    inside_of = {}
    for node in graph.nodes[:MAX_DIAGRAM_NODES]:
        child_key = _clean(node.key)
        parent_key = _clean(node.inside)
        if child_key and parent_key and child_key != parent_key:
            inside_of[child_key] = parent_key
    for node in graph.nodes[:MAX_DIAGRAM_NODES]:
        child_key = _clean(node.key)
        parent_key = _clean(node.inside)
        child = elements.get(child_key)
        parent = elements.get(parent_key)
        if child is None or parent is None or child is parent:
            continue
        if inside_of.get(parent_key) == child_key:
            continue
        identity = (
            parent.element_id,
            child.element_id,
            SemanticRelationKind.DIAGRAM_CONTAINS.value,
        )
        if identity in seen:
            continue
        seen.add(identity)
        relations.append(
            SemanticRelation(
                kind=SemanticRelationKind.DIAGRAM_CONTAINS,
                source_element_id=parent.element_id,
                target_element_id=child.element_id,
                source_selector_ids=(selector.selector_id,),
            )
        )
    for edge in graph.edges[:MAX_DIAGRAM_EDGES]:
        source = elements.get(_clean(edge.source))
        target = elements.get(_clean(edge.target))
        if source is None or target is None:
            continue
        if source.element_id == target.element_id:
            continue
        kind = _relation_kind(graph.kind, edge)
        identity = (source.element_id, target.element_id, kind.value)
        if identity in seen:
            continue
        label = _clean(edge.label) or None
        metadata: dict[str, object] = {}
        if label is not None:
            try:
                metadata["normalized_edge_type"] = normalize_semantic_edge_type(
                    label
                )
            except ContractValidationError:
                metadata = {}
        relations.append(
            SemanticRelation(
                kind=kind,
                source_element_id=source.element_id,
                target_element_id=target.element_id,
                source_selector_ids=(selector.selector_id,),
                label=label,
                metadata=metadata,
            )
        )
        seen.add(identity)

    unbound = SemanticSubstructure(
        kind=graph.kind,
        elements=tuple(by_label.values()),
        relations=tuple(relations),
        producer_kind=SemanticProducerKind.MODEL_DERIVED,
        source_field_paths=(asset.uri,),
        metadata={"asset_sha256": asset.sha256},
    )
    return unbound.with_source_lineage(
        source_observation_ids=tuple(source_observation_ids)
    )


__all__ = [
    "GRAPH_STRUCTURE_KINDS",
    "DiagramEdge",
    "DiagramExtractor",
    "DiagramGraph",
    "DiagramNode",
    "diagram_substructure",
]
