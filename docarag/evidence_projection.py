"""Turn one run's published artifacts into the records the store writes.

A run publishes three artifacts: the canonical blocks a parser produced, the
tree built over them, and the knowledge graph extracted from that tree.  This
reads all three and emits a single :class:`DocumentEvidence`, addressing
everything by the names the artifacts use -- a block's id, a node's index, an
entity's key -- and leaving the writer to resolve those to database surrogates.

Nothing here talks to a database, so a projection can be built, inspected and
compared without one.
"""

from __future__ import annotations

from collections.abc import Mapping

import hashlib
import json
from pathlib import Path

from docarag.persistence.contracts import (
    AssetRecord,
    BlockRecord,
    DocumentEvidence,
    ElementEdgeRecord,
    ElementRecord,
    EntityRecord,
    MentionRecord,
    RepresentationRecord,
    StatementEvidenceRecord,
    StatementRecord,
    StatementRoleRecord,
    StructureRecord,
    TreeNodeRecord,
)

CANONICAL_BLOCKS = "canonical_blocks.json"
TREE = "tree.json"
GRAPH = "graph_data.json"

MAX_ARTIFACT_BYTES = 512 * 1024 * 1024

# A statement's participants when the source only knows a source and a target.
SUBJECT_ROLE = "subject"
OBJECT_ROLE = "object"


class ProjectionError(RuntimeError):
    """A run's artifacts could not be read as evidence."""


def build_document_evidence(
    run_dir: str | Path,
    revision_id: str,
    *,
    asset_cas_root: str | Path | None = None,
    parts: frozenset[str] = frozenset({"tree", "graph"}),
) -> DocumentEvidence:
    """Project one run's stage outputs into evidence, one part at a time.

    Each stage persists only what it produced, so a part never loads the
    other parts' artifacts: the memory a projection needs follows the stage,
    not the document.
    """

    run = Path(run_dir).expanduser().resolve()
    cas_root = Path(asset_cas_root).expanduser().resolve() if asset_cas_root else None

    nodes = _project_tree(_read(run / TREE))
    blocks: tuple = ()
    assets: tuple = ()
    document = None
    if "tree" in parts:
        artifact = _read(run / CANONICAL_BLOCKS)
        blocks, assets = _project_blocks(artifact, cas_root)
        raw = artifact.get("document") if isinstance(artifact, dict) else None
        document = raw if isinstance(raw, dict) else None

    entities: tuple = ()
    mentions: tuple = ()
    statements: tuple = ()
    if "graph" in parts:
        entities, mentions, statements = _project_graph(_read(run / GRAPH), nodes)

    return DocumentEvidence(
        revision_id=revision_id,
        canonical_document=document,
        assets=assets,
        blocks=blocks,
        tree_nodes=nodes if "tree" in parts else (),
        entities=entities,
        mentions=mentions,
        statements=statements,
    )


# -- blocks -------------------------------------------------------------------


def _project_blocks(
    payload: object, cas_root: Path | None
) -> tuple[tuple[BlockRecord, ...], tuple[AssetRecord, ...]]:
    raw_blocks = _sequence(payload, "canonical_blocks")
    assets: dict[str, AssetRecord] = {}
    blocks: list[BlockRecord] = []

    for position, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            continue
        uid = _text(raw.get("block_id"))
        if not uid:
            continue

        asset_roles: list[tuple[str, str]] = []
        for asset in _sequence(raw.get("assets"), None):
            record = _asset_record(asset, cas_root)
            if record is None:
                continue
            assets.setdefault(record.sha256, record)
            role = _text(asset.get("kind")) or "asset"
            asset_roles.append((record.sha256, role))

        blocks.append(
            BlockRecord(
                uid=uid,
                kind=_text(raw.get("block_type")) or "unknown",
                block_no=position,
                revision_uid=_text(raw.get("revision_id")) or None,
                canonical=raw,
                page_uid=_text(raw.get("page_id")) or None,
                page_no=_integer(raw.get("page_index")),
                subtype=_text(raw.get("semantic_subtype")) or None,
                reading_order=_number(raw.get("reading_order")),
                quality=_text(raw.get("quality_status")) or None,
                reconciliation=_text(raw.get("reconciliation_method")) or None,
                content_sha256=_text(raw.get("content_sha256")) or None,
                geometry=raw.get("geometry") if isinstance(raw.get("geometry"), dict) else None,
                source_observation_ids=tuple(
                    _text(item) for item in _sequence(raw.get("source_observation_ids"), None)
                ),
                representations=_representations(raw.get("representations")),
                asset_roles=tuple(asset_roles),
                structures=_structures(raw.get("semantic_substructures")),
                attrs=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            )
        )
    return tuple(blocks), tuple(assets.values())


def _asset_record(asset: object, cas_root: Path | None) -> AssetRecord | None:
    """Describe an asset already published to the content-addressed store.

    Crops reach the canonical layer already published, so nothing is re-read
    here; only their size is recovered, and only when the store is reachable.
    """

    if not isinstance(asset, dict):
        return None
    digest = _text(asset.get("sha256"))
    if len(digest) != 64:
        return None
    byte_size = 0
    if cas_root is not None:
        try:
            byte_size = (cas_root / digest[:2] / digest).stat().st_size
        except OSError:
            byte_size = 0
    uri = _text(asset.get("uri"))
    return AssetRecord(
        sha256=digest,
        media_type=_text(asset.get("media_type")) or "application/octet-stream",
        byte_size=byte_size,
        cas_uri=uri if uri.startswith("cas://") else f"cas://sha256/{digest}",
        attrs=asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {},
    )


def _representations(payload: object) -> tuple[RepresentationRecord, ...]:
    records: list[RepresentationRecord] = []
    for raw in _sequence(payload, None):
        if not isinstance(raw, dict):
            continue
        kind = _text(raw.get("kind"))
        if not kind:
            continue
        content = raw.get("content")
        text = content if isinstance(content, str) else _render(content)
        if not text:
            continue
        producer = None
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            producer = _text(metadata.get("producer")) or None
        records.append(
            RepresentationRecord(
                kind=kind,
                text=text,
                lang=_text(raw.get("language")) or None,
                confidence=_probability(raw.get("confidence")),
                producer=producer,
                attrs=metadata if isinstance(metadata, dict) else {},
            )
        )
    return tuple(records)


def _structures(payload: object) -> tuple[StructureRecord, ...]:
    """Carry a block's semantic graphs: table cells, formula symbols, diagrams."""

    structures: list[StructureRecord] = []
    for raw in _sequence(payload, None):
        if not isinstance(raw, dict):
            continue
        elements: list[ElementRecord] = []
        by_id: dict[str, str] = {}
        for item in _sequence(raw.get("elements"), None):
            if not isinstance(item, dict):
                continue
            local_key = _text(item.get("local_key"))
            if not local_key:
                continue
            by_id[_text(item.get("element_id"))] = local_key
            content = item.get("content")
            elements.append(
                ElementRecord(
                    local_key=local_key,
                    kind=_text(item.get("kind")) or "element",
                    content=content if isinstance(content, str) else _render(content),
                    role=_text(item.get("role")) or None,
                    row_no=_integer(item.get("row_index")),
                    col_no=_integer(item.get("column_index")),
                    row_span=_integer(item.get("row_span")),
                    col_span=_integer(item.get("column_span")),
                    geometry=item.get("geometry") if isinstance(item.get("geometry"), dict) else None,
                    confidence=_probability(item.get("confidence")),
                    attrs=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                )
            )
        if not elements:
            continue

        edges: list[ElementEdgeRecord] = []
        seen: set[tuple[str, str, str, int]] = set()
        for item in _sequence(raw.get("relations"), None):
            if not isinstance(item, dict):
                continue
            src = by_id.get(_text(item.get("source_element_id")))
            dst = by_id.get(_text(item.get("target_element_id")))
            kind = _text(item.get("kind"))
            if not src or not dst or src == dst or not kind:
                continue
            identity = (src, dst, kind, 0)
            if identity in seen:
                continue
            seen.add(identity)
            edges.append(
                ElementEdgeRecord(
                    src_local_key=src,
                    dst_local_key=dst,
                    kind=kind,
                    label=_text(item.get("label")) or None,
                    attrs=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                )
            )

        source_paths = _sequence(raw.get("source_field_paths"), None)
        structures.append(
            StructureRecord(
                kind=_text(raw.get("kind")) or "layout",
                producer=_text(raw.get("producer_kind")) or "parser_native",
                elements=tuple(elements),
                edges=tuple(edges),
                availability=_text(raw.get("availability")) or "complete",
                source_uri=_text(source_paths[0]) if source_paths else None,
                attrs=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            )
        )
    return tuple(structures)


# -- tree ---------------------------------------------------------------------


_EVIDENCE_TRACE_FIELDS = (
    "page_id",
    "document_version_id",
    "canonical_block_id",
    "canonical_revision_id",
    "observation_id",
    "derived_evidence_id",
    "source_canonical_block_ids",
    "source_canonical_revision_ids",
    "source_observation_ids",
    "source_evidence_refs",
    "canonical_representation_ids",
    "canonical_asset_ids",
    "selected_representation_ids",
    "selected_asset_ids",
    "canonical_relations",
    "semantic_substructures",
    "transformations",
    "semantic_subtype",
    "title_level",
    "parser_provenance",
    "quality_status",
    "reconciliation_method",
    "source_uri",
    "source_sha256",
    "snapshot_path",
)


def _node_evidence(kind: str, meta: Mapping[str, object]) -> dict[str, object] | None:
    """The retrieval payload this node's evidence lineage is validated by.

    Mirrors the shape the tree publishes: the hydrator compares these fields
    against the canonical artifact, so they are stored verbatim, not rebuilt.
    """

    if not meta:
        return None
    payload: dict[str, object] = {
        "type": kind,
        "content": meta.get("content"),
        "page": meta.get("page_idx"),
    }
    for name in _EVIDENCE_TRACE_FIELDS:
        value = meta.get(name)
        if value is not None:
            payload[name] = value
    geometry = meta.get("source_geometry")
    if geometry is not None:
        payload["geometry"] = geometry
    if kind in ("image", "table", "equation"):
        payload["img_path"] = meta.get("img_path")
    if kind == "table":
        payload["table_body"] = meta.get("table_body")
        payload["caption"] = meta.get("caption")
        payload["footnote"] = meta.get("footnote")
    return payload


def _node_kind(value: object) -> str:
    kind = _text(value)
    # Trees serialized before the NodeType fix spelled the enum member name.
    if kind.startswith("NodeType."):
        kind = kind.removeprefix("NodeType.").lower()
    return kind or "node"


def _project_tree(payload: object) -> tuple[TreeNodeRecord, ...]:
    raw_nodes = _sequence(payload, "nodes")
    siblings: dict[int | None, int] = {}
    nodes: list[TreeNodeRecord] = []

    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_no = _integer(raw.get("index_id"))
        if node_no is None:
            continue
        parent_no = _integer(raw.get("parent_id"))
        if parent_no == node_no:
            parent_no = None
        meta = raw.get("meta_info") if isinstance(raw.get("meta_info"), dict) else {}
        summary = raw.get("summary")
        sibling_no = siblings.get(parent_no, 0)
        siblings[parent_no] = sibling_no + 1
        kind = _node_kind(raw.get("type"))
        evidence = _node_evidence(kind, meta)
        nodes.append(
            TreeNodeRecord(
                node_no=node_no,
                kind=kind,
                parent_no=parent_no,
                sibling_no=sibling_no,
                # The tree publishes its evidence binding under this name; it
                # is what joins a node to the block it was built from.
                block_uid=_text(meta.get("canonical_block_id")) or None,
                label=_clip(_text(meta.get("content"))) or None,
                summary=summary if isinstance(summary, str) and summary.strip() else None,
                attrs={"evidence": evidence} if evidence else {},
            )
        )
    return tuple(nodes)


# -- graph --------------------------------------------------------------------


def _project_graph(
    payload: object, nodes: tuple[TreeNodeRecord, ...]
) -> tuple[
    tuple[EntityRecord, ...], tuple[MentionRecord, ...], tuple[StatementRecord, ...]
]:
    if not isinstance(payload, dict):
        return (), (), ()
    graph = payload.get("graph") if isinstance(payload.get("graph"), dict) else {}

    entities: dict[str, EntityRecord] = {}
    ref_by_name: dict[str, str] = {}
    for raw in _sequence(graph.get("nodes"), None):
        if not isinstance(raw, dict):
            continue
        canonical = _text(raw.get("entity_name"))
        if not canonical:
            continue
        kind = _text(raw.get("entity_type")) or "UNKNOWN"
        ref = _entity_ref(kind, canonical)
        entities.setdefault(
            ref,
            EntityRecord(
                ref=ref,
                kind=kind,
                canonical=canonical,
                description=_text(raw.get("description")) or None,
            ),
        )
        ref_by_name[canonical] = ref
        node_id = _text(raw.get("id"))
        if node_id:
            ref_by_name[node_id] = ref

    known_nodes = {node.node_no for node in nodes}
    mentions: list[MentionRecord] = []
    seen_mentions: set[tuple[str, int]] = set()
    for node_key, entity_keys in (payload.get("tree2kg") or {}).items():
        node_no = _integer(node_key)
        if node_no is None or node_no not in known_nodes:
            continue
        for entity_key in _sequence(entity_keys, None):
            ref = ref_by_name.get(_text(entity_key))
            if ref is None or (ref, node_no) in seen_mentions:
                continue
            seen_mentions.add((ref, node_no))
            mentions.append(MentionRecord(entity_ref=ref, node_no=node_no))

    statements: list[StatementRecord] = []
    for raw in _sequence(graph.get("links"), None):
        if not isinstance(raw, dict):
            continue
        source = ref_by_name.get(_text(raw.get("src_entity_name")))
        target = ref_by_name.get(_text(raw.get("tgt_entity_name")))
        if source is None or target is None or source == target:
            continue
        predicate = _text(raw.get("relation_name")) or "related_to"
        statements.append(
            StatementRecord(
                predicate=predicate,
                roles=(
                    StatementRoleRecord(entity_ref=source, role=SUBJECT_ROLE),
                    StatementRoleRecord(entity_ref=target, role=OBJECT_ROLE, ordinal=1),
                ),
                weight=_number(raw.get("weight")),
                description=_text(raw.get("description")) or None,
                producer="kg_extraction",
                evidence=tuple(
                    StatementEvidenceRecord(node_no=node_no)
                    for node_no in _source_nodes(raw.get("source_ids"), known_nodes)
                ),
            )
        )
    return tuple(entities.values()), tuple(mentions), tuple(statements)


def _entity_ref(kind: str, canonical: str) -> str:
    """A stable local name for an entity within one projection."""

    digest = hashlib.sha256(f"{kind}\x00{canonical}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:24]}"


def _source_nodes(payload: object, known: set[int]) -> tuple[int, ...]:
    found = []
    for item in _sequence(payload, None):
        node_no = _integer(item)
        if node_no is not None and node_no in known:
            found.append(node_no)
    return tuple(dict.fromkeys(found))


# -- reading ------------------------------------------------------------------


def _read(path: Path) -> object:
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ProjectionError(f"{path.name} exceeds the artifact size limit")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProjectionError(f"{path.name} is not readable JSON") from exc


def _sequence(payload: object, key: str | None) -> list[object]:
    if key is not None:
        payload = payload.get(key) if isinstance(payload, dict) else None
    return [item for item in payload] if isinstance(payload, list) else []


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clip(value: str, limit: int = 512) -> str:
    collapsed = " ".join(value.split())
    return collapsed[:limit]


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _probability(value: object) -> float | None:
    number = _number(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def _render(value: object) -> str:
    """Flatten a structured representation, such as a table grid, into text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        rows = []
        for row in value:
            if isinstance(row, list):
                rows.append(" | ".join(_render(cell) for cell in row))
            else:
                rows.append(_render(row))
        return "\n".join(part for part in rows if part)
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = ["ProjectionError", "build_document_evidence"]
