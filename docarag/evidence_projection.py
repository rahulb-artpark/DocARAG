"""Project a completed run's canonical/tree/graph JSON into the evidence store.

Dependency-light: reads only the run's JSON artifacts (no BookRAG interpreter),
publishes figure/table/diagram bytes into a content-addressed asset store, and
returns a DocumentEvidence bundle for one index revision.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from docarag.persistence.contracts import (
    AssetRecord,
    BlockAssetLink,
    BlockRecord,
    DocumentEvidence,
    EntityRecord,
    GTLinkRecord,
    RelationRecord,
    RepresentationRecord,
    TreeNodeRecord,
)

_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_id(node_key: str) -> str:
    return "ent_" + hashlib.sha256(node_key.encode("utf-8")).hexdigest()[:24]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish_asset_bytes(data: bytes, cas_root: Path) -> str:
    """Store bytes at <cas_root>/<digest[:2]>/<digest>; return the digest."""

    digest = _sha256_bytes(data)
    target = cas_root / digest[:2] / digest
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return digest


def _blocks(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        blocks = payload.get("canonical_blocks")
        if isinstance(blocks, list):
            return [b for b in blocks if isinstance(b, dict)]
    return []


def _node_paths(nodes: list[dict]) -> dict[int, str]:
    parent = {}
    for node in nodes:
        nid = node.get("index_id")
        if isinstance(nid, int):
            parent[nid] = node.get("parent_id")
    paths: dict[int, str] = {}

    def path_for(nid: int) -> str:
        if nid in paths:
            return paths[nid]
        pid = parent.get(nid)
        if pid is None or pid == nid or pid not in parent:
            paths[nid] = str(nid)
        else:
            paths[nid] = path_for(pid) + "." + str(nid)
        return paths[nid]

    for nid in parent:
        path_for(nid)
    return paths


def build_document_evidence(
    run_dir: str | os.PathLike[str],
    index_revision_id: str,
    *,
    asset_cas_root: str | os.PathLike[str] | None = None,
) -> DocumentEvidence:
    run = Path(run_dir).expanduser().resolve()
    cas_root = Path(asset_cas_root).expanduser().resolve() if asset_cas_root else None

    assets: list[AssetRecord] = []
    blocks: list[BlockRecord] = []
    canonical_path = run / "canonical_blocks.json"
    if canonical_path.is_file():
        for block in _blocks(_load_json(canonical_path)):
            representations = tuple(
                RepresentationRecord(
                    kind=str(rep.get("kind") or "text"),
                    content_sha256=rep.get("content_sha256"),
                    content=rep.get("content"),
                    confidence=rep.get("confidence"),
                    metadata=rep.get("metadata") or {},
                )
                for rep in block.get("representations") or []
                if isinstance(rep, dict)
            )
            asset_links: list[BlockAssetLink] = []
            for asset in block.get("assets") or []:
                if not isinstance(asset, dict):
                    continue
                digest = asset.get("sha256") or asset.get("asset_id")
                if isinstance(digest, str) and len(digest) == 64:
                    asset_links.append(
                        BlockAssetLink(asset_id=digest, role=str(asset.get("role") or "asset"))
                    )
            blocks.append(
                BlockRecord(
                    block_id=str(block["block_id"]),
                    revision_id=str(block.get("revision_id") or block.get("revision") or ""),
                    block_type=str(block.get("block_type") or "text"),
                    page_index=int(block.get("page_index") or 0),
                    semantic_subtype=block.get("semantic_subtype"),
                    page_id=block.get("page_id"),
                    reading_order=block.get("reading_order"),
                    parent_block_id=block.get("parent_block_id"),
                    content_sha256=block.get("content_sha256"),
                    quality_status=block.get("quality_status"),
                    reconciliation_method=block.get("reconciliation_method"),
                    geometry=block.get("geometry"),
                    source_observation_ids=tuple(
                        str(x) for x in block.get("source_observation_ids") or []
                    ),
                    metadata=block.get("metadata") or {},
                    representations=representations,
                    asset_links=tuple(asset_links),
                )
            )

    images_dir = run / "vlm" / "images"
    if cas_root is not None and images_dir.is_dir():
        for image in sorted(images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in _IMAGE_MIME:
                continue
            data = image.read_bytes()
            digest = publish_asset_bytes(data, cas_root)
            assets.append(
                AssetRecord(
                    asset_id=digest,
                    mime_type=_IMAGE_MIME[image.suffix.lower()],
                    byte_size=len(data),
                    cas_uri=f"cas://sha256/{digest}",
                    metadata={"origin_name": image.name},
                )
            )

    tree_nodes: list[TreeNodeRecord] = []
    tree_path = run / "tree.json"
    if tree_path.is_file():
        tree = _load_json(tree_path)
        nodes = tree.get("nodes") if isinstance(tree, dict) else None
        nodes = [n for n in nodes or [] if isinstance(n, dict)]
        paths = _node_paths(nodes)
        for node in nodes:
            nid = node.get("index_id")
            if not isinstance(nid, int):
                continue
            meta = node.get("meta_info") or {}
            block_id = meta.get("block_id") if isinstance(meta, dict) else None
            summary = node.get("summary")
            tree_nodes.append(
                TreeNodeRecord(
                    node_id=nid,
                    node_path=paths.get(nid, str(nid)),
                    node_type=str(node.get("type") or "node"),
                    level=paths.get(nid, str(nid)).count(".") ,
                    parent_id=node.get("parent_id"),
                    block_id=block_id if isinstance(block_id, str) else None,
                    summary_sha256=(
                        _sha256_bytes(summary.encode("utf-8"))
                        if isinstance(summary, str) and summary
                        else None
                    ),
                    summary=summary if isinstance(summary, str) else None,
                )
            )

    entities: list[EntityRecord] = []
    relations: list[RelationRecord] = []
    gt_links: list[GTLinkRecord] = []
    graph_path = run / "graph_data.json"
    if graph_path.is_file():
        graph_doc = _load_json(graph_path)
        graph = graph_doc.get("graph") if isinstance(graph_doc, dict) else None
        id_map: dict[str, str] = {}
        if isinstance(graph, dict):
            for node in graph.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                key = str(node.get("id") or node.get("entity_name") or "")
                if not key:
                    continue
                entity_id = _entity_id(key)
                id_map[key] = entity_id
                description = node.get("description")
                entities.append(
                    EntityRecord(
                        entity_id=entity_id,
                        entity_name=str(node.get("entity_name") or ""),
                        entity_type=str(node.get("entity_type") or "UNKNOWN"),
                        description=description if isinstance(description, str) else None,
                        description_sha256=(
                            _sha256_bytes(description.encode("utf-8"))
                            if isinstance(description, str) and description
                            else None
                        ),
                    )
                )
            for position, link in enumerate(graph.get("links") or []):
                if not isinstance(link, dict):
                    continue
                src = id_map.get(str(link.get("source")))
                dst = id_map.get(str(link.get("target")))
                if src is None or dst is None:
                    continue
                weight = link.get("weight")
                relations.append(
                    RelationRecord(
                        relation_id="rel_"
                        + hashlib.sha256(
                            f"{src}|{dst}|{link.get('relation_name')}|{position}".encode()
                        ).hexdigest()[:24],
                        src_entity_id=src,
                        dst_entity_id=dst,
                        relation_type=str(link.get("relation_name") or "related"),
                        directed=True,
                        weight=float(weight) if _is_number(weight) else None,
                        description=(
                            link.get("description")
                            if isinstance(link.get("description"), str)
                            else None
                        ),
                    )
                )
        tree2kg = graph_doc.get("tree2kg") if isinstance(graph_doc, dict) else None
        if isinstance(tree2kg, dict):
            seen: set[tuple[str, int]] = set()
            for node_key, entity_keys in tree2kg.items():
                try:
                    node_id = int(node_key)
                except (TypeError, ValueError):
                    continue
                for entity_key in entity_keys or []:
                    entity_id = id_map.get(str(entity_key))
                    if entity_id is None or (entity_id, node_id) in seen:
                        continue
                    seen.add((entity_id, node_id))
                    gt_links.append(GTLinkRecord(entity_id=entity_id, node_id=node_id))

    return DocumentEvidence(
        index_revision_id=index_revision_id,
        assets=tuple(assets),
        blocks=tuple(blocks),
        tree_nodes=tuple(tree_nodes),
        entities=tuple(entities),
        relations=tuple(relations),
        gt_links=tuple(gt_links),
    )


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


__all__ = ["build_document_evidence", "publish_asset_bytes"]
