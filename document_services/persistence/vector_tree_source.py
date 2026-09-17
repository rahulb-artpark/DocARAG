"""Serve the vdb stage its tree from the store, one batch at a time.

The vector stage consumes a tree shape: nodes with typed meta, a root, and
document identity.  This source materializes that shape from PostgreSQL rows
instead of a pickle, so the stage's memory follows its batch size rather than
the document, and the vectors are built from exactly what queries will read.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from retrieval_engine.indexes.tree import NodeType

from document_services.persistence.document_store import DocumentStore


def asset_cas_root(run_dir: object) -> Path | None:
    """The content-addressed store a run's crops were published into."""

    configured = os.environ.get("DOCARAG_ASSET_CAS_ROOT")
    if configured:
        root = Path(configured).expanduser()
        return root if root.is_dir() else None
    if not isinstance(run_dir, (str, Path)) or not str(run_dir):
        return None
    root = Path(run_dir).expanduser().resolve() / "asset_cas"
    return root if root.is_dir() else None


def _from_cas(uri: str, root: Path | None) -> str:
    """Resolve ``cas://sha256/<digest>`` to the file the store published.

    ``document_services.evidence.crops`` publishes every visual crop into the run's
    content-addressed store and rewrites the block to carry the CAS URI, so for
    a chart or diagram the URI is frequently the *only* image reference the tree
    holds. The layout is ``<root>/<first-two-hex>/<digest>``.
    """

    if root is None:
        return ""
    parsed = urlparse(uri)
    if parsed.scheme != "cas" or parsed.netloc != "sha256":
        return ""
    digest = parsed.path.lstrip("/")
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        return ""
    published = root / digest[:2] / digest
    return str(published) if published.is_file() else ""


def openable_image(*candidates: object, cas_root: Path | None = None) -> str:
    """The first candidate something can actually open.

    An image reaches a node two ways: as a materialized local path, or as its
    canonical identity ``cas://sha256/<digest>``. A URI names a thing; it does
    not open one -- and ollama-python, handed one where a filename belonged,
    posted the URI itself as though it were the image. The server answered
    "illegal base64 data at input byte 3", the 3 being the colon in ``cas:``,
    and every chart and diagram in a document lost its entities that way.

    So a URI never passes through as a path. It is either resolved to the file
    the content-addressed store actually holds, or it is not an image here --
    because for a materialized crop the URI is often the only reference there
    is, and refusing it outright leaves the same figures with no image at all.

    A materialized path is preferred to a resolvable URI: both name the same
    bytes, and identity resolving ahead of a path is what went wrong here once
    already.
    """

    usable = [item for item in candidates if isinstance(item, str) and item]
    for candidate in usable:
        if "://" not in candidate:
            return candidate
    for candidate in usable:
        resolved = _from_cas(candidate, cas_root)
        if resolved:
            return resolved
    return ""


class _NodeMeta:
    """The meta surface the vector stage reads, backed by store channels."""

    def __init__(self, view: Any, evidence: dict, cas_root: Path | None = None) -> None:
        meta = view.meta_info
        self.content = meta.content
        self.caption = meta.caption
        self.footnote = meta.footnote
        self.table_body = meta.table_body
        self.img_path = openable_image(
            evidence.get("img_path"), meta.img_path, cas_root=cas_root
        )
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

    def __getattr__(self, name: str):
        # Every remaining meta field the stage may read lives in the evidence
        # payload; absence is None, exactly as the serialized meta behaved.
        try:
            return self.__dict__["_evidence"].get(name)
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self, mode: str = "python") -> dict:
        return dict(self._evidence)


class _Node:
    def __init__(self, view: Any, evidence: dict, cas_root: Path | None = None) -> None:
        self.index_id = view.index_id
        kind = str(view.type)
        try:
            self.type = NodeType(kind)
        except ValueError:
            self.type = kind
        self.depth = view.depth
        self.summary = view.summary
        self.outline_node = view.outline_node
        self.meta_info = _NodeMeta(view, evidence, cas_root)
        self.parent: "_Node | None" = None
        self.children: list["_Node"] = []


class StoreTreeSource:
    """A tree the vector stage can walk, read from one revision's rows."""

    def __init__(self, store: DocumentStore) -> None:
        views = store.tree.nodes()
        cas_root = asset_cas_root(store.run_dir)
        by_id: dict[int, _Node] = {}
        for view in views:
            by_id[view.index_id] = _Node(view, dict(view.evidence or {}), cas_root)
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
        self.store_backed = True
        row = store._borrow().execute(
            "SELECT tree_sha256 FROM docarag.revision_seal WHERE revision_key = %s",
            (store.revision_key,),
        ).fetchone()
        self.store_seal = str((row or {}).get("tree_sha256") or "") or None
        ledger_row = store._borrow().execute(
            "SELECT canonical_artifact_sha256 FROM docarag.index_revisions "
            "WHERE revision_key = %s",
            (store.revision_key,),
        ).fetchone()
        canonical_sha = str(
            (ledger_row or {}).get("canonical_artifact_sha256") or ""
        ) or None
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
            canonical_artifact_sha256=canonical_sha,
        )

    def get_nodes(self, hasRoot: bool = False) -> list:
        if hasRoot:
            return list(self.nodes)
        return [node for node in self.nodes if node is not self.root_node]

    def get_node_by_index_id(self, node_id: int):
        for node in self.nodes:
            if node.index_id == node_id:
                return node
        return None

    def get_path_from_root(self, node_id: int) -> list:
        node = self.get_node_by_index_id(node_id)
        path: list = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_sibling_nodes(self, node_id: int) -> list:
        node = self.get_node_by_index_id(node_id)
        if node is None or node.parent is None:
            return []
        return [child for child in node.parent.children if child is not node]

    @classmethod
    def open(cls, run_dir: str) -> "StoreTreeSource":
        store = DocumentStore.open(run_dir)
        try:
            return cls(store)
        finally:
            store.close()


__all__ = ["StoreTreeSource", "openable_image"]
