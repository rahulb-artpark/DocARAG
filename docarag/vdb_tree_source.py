"""Serve the vdb stage its tree from the store, one batch at a time.

The vector stage consumes a tree shape: nodes with typed meta, a root, and
document identity.  This source materializes that shape from PostgreSQL rows
instead of a pickle, so the stage's memory follows its batch size rather than
the document, and the vectors are built from exactly what queries will read.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from Core.Index.Tree import NodeType

from docarag.document_store import DocumentStore


class _NodeMeta:
    """The meta surface the vector stage reads, backed by store channels."""

    def __init__(self, view: Any, evidence: dict) -> None:
        meta = view.meta_info
        self.content = meta.content
        self.caption = meta.caption
        self.footnote = meta.footnote
        self.table_body = meta.table_body
        self.img_path = evidence.get("img_path") or meta.img_path
        self.page_idx = evidence.get("page")
        self.page_id = evidence.get("page_id")
        self.pdf_id = evidence.get("pdf_id")
        self.document_version_id = evidence.get("document_version_id")
        self.source_sha256 = evidence.get("source_sha256")
        self.source_sha = evidence.get("source_sha256")
        self.source_geometry = evidence.get("geometry")
        self.semantic_substructures = evidence.get("semantic_substructures")
        self.canonical_block_id = evidence.get("canonical_block_id")
        self.canonical_revision_id = evidence.get("canonical_revision_id")
        self.derived_evidence_id = evidence.get("derived_evidence_id")
        self.source_evidence_refs = evidence.get("source_evidence_refs")
        self._evidence = evidence

    def model_dump(self, mode: str = "python") -> dict:
        return dict(self._evidence)


class _Node:
    def __init__(self, view: Any, evidence: dict) -> None:
        self.index_id = view.index_id
        kind = str(view.type)
        try:
            self.type = NodeType(kind)
        except ValueError:
            self.type = kind
        self.depth = view.depth
        self.summary = view.summary
        self.outline_node = view.outline_node
        self.meta_info = _NodeMeta(view, evidence)
        self.parent: "_Node | None" = None
        self.children: list["_Node"] = []


class StoreTreeSource:
    """A tree the vector stage can walk, read from one revision's rows."""

    def __init__(self, store: DocumentStore) -> None:
        views = store.tree.nodes()
        by_id: dict[int, _Node] = {}
        for view in views:
            by_id[view.index_id] = _Node(view, dict(view.evidence or {}))
        self.root_node = None
        for view in views:
            node = by_id[view.index_id]
            if view.parent_id is None:
                self.root_node = node
            else:
                parent = by_id.get(view.parent_id)
                if parent is not None:
                    node.parent = parent
                    parent.children.append(node)
        self.nodes = [by_id[view.index_id] for view in views]
        self.save_dir = str(store.run_dir)
        scope = store.scope
        source = self.nodes and self.nodes[0].meta_info or None
        self.meta_info = SimpleNamespace(
            source_sha256=next(
                (
                    n.meta_info.source_sha256
                    for n in self.nodes
                    if n.meta_info.source_sha256
                ),
                None,
            ),
            document_version_id=scope.document_version_id,
        )

    @classmethod
    def open(cls, run_dir: str) -> "StoreTreeSource":
        store = DocumentStore.open(run_dir)
        try:
            return cls(store)
        finally:
            store.close()


__all__ = ["StoreTreeSource"]
