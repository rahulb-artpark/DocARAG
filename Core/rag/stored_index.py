"""The index retrieval reads, served from stored evidence.

Retrieval asks a document for a handful of things: the nodes mentioning an
entity, the subtree beneath a section, the graph those nodes induce, and the
evidence behind a final few nodes.  This answers all of them from the store,
one query each, so a question's memory follows the question rather than the
size of the document.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from typing import Any, Iterable, List, Sequence

import networkx as nx

from docarag.document_store import DocumentStore
from docarag.tree_store import GraphEdge, TreeNodeView


def _local_path(source_uri: str) -> str:
    """The filesystem path a file:// URI names, or nothing.

    A document's canonical identity is its URI; the path is a convenience for
    callers that want to show a filename. Any other scheme has no local path,
    and saying so beats inventing one.
    """

    if not isinstance(source_uri, str) or not source_uri.startswith("file://"):
        return ""
    return url2pathname(urlparse(source_uri).path)

MAX_SUBGRAPH_ENTITIES = 256


def entity_node_name(entity_name: str, entity_type: str) -> str:
    """The name a knowledge-graph node is known by."""

    return f"Name: {entity_name}\nType: {entity_type}"


def split_node_name(node_name: str) -> tuple[str, str | None]:
    """Recover the entity and, when present, the type from a node name."""

    text = str(node_name or "")
    if text.startswith("Name: ") and "\nType: " in text:
        name, _, kind = text[len("Name: ") :].partition("\nType: ")
        return name.strip(), kind.strip() or None
    return text.strip(), None


def _add_graph_edges(
    graph: nx.Graph,
    edges: Iterable[Any],
    *,
    entity_to_nodes: dict[str, set[int]],
    document_version_id: str | None,
    discount_factor: float,
) -> None:
    """Add edges to ``graph``, keyed by the combined "Name: x\\nType: y" form.

    Entity search (``entity_vdb``, ``NodeName2TreeNodes``, ``ent_map``) has
    always keyed entities this way; a subgraph edge's ``source``/``target``
    are bare canonicals straight from Postgres. Without this conversion, no
    query-matched entity name can ever equal a subgraph node name, so
    ``graph_reranker``'s personalization vector is always empty -- upstream
    of, and independent from, the ``source_ids`` gap this also fixes.
    """

    for edge in edges:
        source_name = entity_node_name(edge.source, edge.source_kind)
        target_name = entity_node_name(edge.target, edge.target_kind)
        for canonical, name in ((edge.source, source_name), (edge.target, target_name)):
            node_ids = entity_to_nodes.get(canonical, set())
            new_source_ids = (
                {(document_version_id, node_id) for node_id in node_ids}
                if document_version_id else set(node_ids)
            )
            # Union with whatever this node already carries, so a
            # cross-document edge (entity_to_nodes always empty) never
            # clobbers source_ids a per-document subgraph already populated.
            existing = graph.nodes[name].get("source_ids", set()) if graph.has_node(name) else set()
            graph.add_node(name, entity_name=name, source_ids=existing | new_source_ids)
        weight = edge.weight if edge.weight is not None else 1.0
        if getattr(edge, "link_origin", None) == "cross_document_inferred":
            weight *= discount_factor
        graph.add_edge(
            source_name, target_name,
            relation_name=edge.predicate, weight=weight,
            description=edge.description, link_origin=getattr(edge, "link_origin", None),
        )


class StoredTreeIndex:
    """The document tree, read a slice at a time."""

    __slots__ = ("_tree", "meta_info")

    def __init__(self, tree, meta_info=None) -> None:
        self._tree = tree
        self.meta_info = meta_info

    @property
    def root_node(self) -> TreeNodeView | None:
        return self._tree.root()

    def get_nodes(self, hasRoot: bool = False) -> List[TreeNodeView]:
        nodes = self._tree.nodes()
        if hasRoot:
            return nodes
        return [node for node in nodes if node.parent_id is not None]

    def get_node_by_index_id(self, node_id: int) -> TreeNodeView | None:
        return self._tree.node(node_id)

    def get_path_from_root(self, node_id: int) -> List[TreeNodeView]:
        return self._tree.path_from_root(node_id)

    def get_sibling_nodes(self, node_id: int) -> List[TreeNodeView]:
        return self._tree.siblings(node_id)

    def get_subtree_nodes(self, node_ids: Sequence[int] | int) -> List[TreeNodeView]:
        return self._tree.subtree(node_ids)

    def get_nodes_at_depth(self, depth: int) -> List[TreeNodeView]:
        return self._tree.nodes_at_depth(depth)

    def get_ancestor_at_depth(self, node_id: int, depth: int) -> TreeNodeView | None:
        return self._tree.ancestor_at_depth(node_id, depth)

    def get_filtered_nodes(self, node_type: Any) -> List[TreeNodeView]:
        return self._tree.nodes_of_kind(str(node_type))

    def get_max_depth(self) -> int:
        return self._tree.max_depth()

    def get_nodes_data(self, node_ids: Sequence[int] | None = None) -> List[dict]:
        """Return the evidence behind the nodes an answer will cite."""

        return [
            _node_payload(node, self.meta_info)
            for node in self._tree.nodes(node_ids)
        ]

    def retrieval_texts(self, node_ids: Sequence[int]) -> dict[int, str]:
        return self._tree.retrieval_texts(node_ids)


class StoredGraphIndex:
    """The knowledge graph, induced from the nodes a question reached."""

    __slots__ = ("_tree",)

    def __init__(self, tree) -> None:
        self._tree = tree

    @staticmethod
    def get_node_name_from_str(entity_name: str, entity_type: str) -> str:
        return entity_node_name(entity_name, entity_type)

    @staticmethod
    def get_node_name_from_entity(entity: Any) -> str:
        return entity_node_name(
            getattr(entity, "entity_name", ""), getattr(entity, "entity_type", "")
        )

    def NodeName2TreeNodes(self, node_name: str) -> List[int]:
        """Resolve a graph node name to the tree nodes that mention it."""

        canonical, kind = split_node_name(node_name)
        if not canonical:
            return []
        return self._tree.nodes_for_entity(canonical, kind)

    def get_full_graph(self, *, document_version_id: str | None = None):
        """The whole document's graph, not one subtree's -- GraphRAG's own shape.

        ``GraphRAG.__init__`` used to read ``self.gbc_index.GraphIndex.kg``,
        which only ever existed on the legacy in-memory ``Graph`` class; the
        store-backed ``StoredGraphIndex`` actually used at query time has no
        ``.kg`` at all, so this raised AttributeError before a query ever ran.
        Built from ``TreeStore.graph_statements`` (already existed, already
        unused by any RAG class) through the same node-naming/source_ids/
        discount path ``get_kg_subgraph`` uses.
        """

        rows = list(self._tree.graph_statements())
        entity_to_nodes: dict[str, set[int]] = {}
        for row in rows:
            node_ids = set(row.get("node_ids") or ())
            for canonical in (str(row["source"]), str(row["target"])):
                entity_to_nodes.setdefault(canonical, set()).update(node_ids)

        graph = nx.Graph()
        edges = (
            GraphEdge(
                source=str(row["source"]), target=str(row["target"]),
                predicate=str(row["predicate"]), weight=row["weight"],
                description=str(row.get("description") or ""),
                source_kind=str(row.get("source_kind") or ""),
                target_kind=str(row.get("target_kind") or ""),
            )
            for row in rows
        )
        _add_graph_edges(
            graph, edges, entity_to_nodes=entity_to_nodes,
            document_version_id=document_version_id, discount_factor=1.0,
        )
        return graph

    def get_kg_subgraph(
        self,
        tree_node_ids: Iterable[int],
        copy: bool = True,
        *,
        document_version_id: str | None = None,
    ):
        """Build the subgraph those nodes support, and nothing wider.

        Populates ``source_ids`` on each node: the tree nodes that entity is
        actually mentioned/evidenced in, among ``tree_node_ids``. Without this,
        ``graph_reranker``'s entity-score to tree-node join has nothing to read
        and always returns an empty ranking.
        """

        wanted = list(tree_node_ids)
        entity_to_nodes: dict[str, set[int]] = {}
        for node_no, canonicals in self._tree.entities_for_nodes(wanted).items():
            for canonical in canonicals:
                entity_to_nodes.setdefault(canonical, set()).add(node_no)

        graph = nx.Graph()
        _add_graph_edges(
            graph,
            self._tree.induced_subgraph(wanted),
            entity_to_nodes=entity_to_nodes,
            document_version_id=document_version_id,
            discount_factor=1.0,
        )
        return graph

    def add_cross_document_edges(
        self,
        graph: nx.Graph,
        edges: Iterable[Any],
        *,
        discount_factor: float = 1.0,
    ) -> None:
        """Merge cross-docs edges into an already-composed multi-document graph.

        No ``entity_to_nodes`` is passed: an edge's endpoints are expected to
        already be nodes in ``graph`` (populated by each document's own
        ``get_kg_subgraph``); a cross-docs edge only ever adds the edge and,
        for the endpoint kind of case where it doesn't, a bare node with no
        ``source_ids`` -- unreachable from a query-entity seed, but still able
        to carry weight through the graph.
        """

        _add_graph_edges(
            graph, edges, entity_to_nodes={}, document_version_id=None,
            discount_factor=discount_factor,
        )

    def get_subgraph_data(self, entities: Sequence[str], max_edges: int = 64) -> dict:
        """Return the entities and relations an answer trace should carry."""

        wanted = [str(item) for item in entities][:MAX_SUBGRAPH_ENTITIES]
        if not wanted:
            return {"nodes": [], "edges": []}
        names = {split_node_name(item)[0] for item in wanted}
        node_ids: list[int] = []
        for name in names:
            node_ids.extend(self._tree.nodes_for_entity(name))
        edges = self._tree.induced_subgraph(node_ids)[:max_edges]
        return {
            "nodes": [{"id": name, "entity_name": name} for name in sorted(names)],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation_name": edge.predicate,
                    "weight": edge.weight,
                    "description": edge.description,
                }
                for edge in edges
            ],
        }


@dataclass(frozen=True)
class StoredTreeMeta:
    """The identity retrieval binds a tree's vectors and evidence against.

    Field names mirror Core.Index.Tree.MetaInfo, which retrieval reads by name.
    """

    tenant_id: str = ""
    corpus_id: str = ""
    document_version_id: str = ""
    index_revision_id: str = ""
    source_sha256: str = ""
    source_uri: str = ""
    file_name: str = ""
    file_path: str = ""
    snapshot_path: str = ""
    canonical_artifact_path: str = ""
    canonical_artifact_sha256: str = ""
    canonical_artifact_size_bytes: int = 0

    @property
    def source_sha(self) -> str:
        return self.source_sha256

    @property
    def canonical_artifact_sha(self) -> str:
        return self.canonical_artifact_sha256

    @classmethod
    def from_store(cls, store: DocumentStore) -> "StoredTreeMeta":
        """Read the document's identity from the store, and only the store.

        This used to open run-manifest.json to learn the source's digest, URI
        and path -- a file read on the query path, for facts the ledger already
        holds. ``document_versions`` records the digest and URI at admission,
        and a ``file://`` URI is where the local path comes from, so no run
        directory has to exist for a query to answer.
        """

        scope = getattr(store, "scope", None)
        canonical_sha = ""
        source_sha256 = ""
        source_uri = ""
        try:
            row = store._borrow().execute(
                "SELECT r.canonical_artifact_sha256, v.source_sha256, v.source_uri "
                "FROM docarag.index_revisions r "
                "JOIN docarag.document_versions v "
                "  ON v.tenant_id = r.tenant_id "
                " AND v.corpus_id = r.corpus_id "
                " AND v.document_id = r.document_id "
                " AND v.document_version_id = r.document_version_id "
                "WHERE r.revision_key = %s",
                (store.revision_key,),
            ).fetchone()
            row = row or {}
            canonical_sha = str(row.get("canonical_artifact_sha256") or "")
            source_sha256 = str(row.get("source_sha256") or "")
            source_uri = str(row.get("source_uri") or "")
        except Exception:
            pass

        source_path = _local_path(source_uri)
        return cls(
            tenant_id=getattr(scope, "tenant_id", "") or "",
            corpus_id=getattr(scope, "corpus_id", "") or "",
            document_version_id=getattr(scope, "document_version_id", "") or "",
            index_revision_id=getattr(store, "revision_id", "") or "",
            source_sha256=source_sha256,
            source_uri=source_uri,
            file_name=Path(source_path).name if source_path else "document.pdf",
            file_path=source_path,
            snapshot_path="",
            canonical_artifact_path="",
            canonical_artifact_sha256=canonical_sha,
            canonical_artifact_size_bytes=0,
        )


class StoredIndex:
    """What a query holds open for one document."""

    __slots__ = ("GraphIndex", "TreeIndex", "_store", "embedder", "entity_vdb")

    def __init__(self, store: DocumentStore, config=None) -> None:
        self._store = store
        self.TreeIndex = StoredTreeIndex(store.tree, StoredTreeMeta.from_store(store))
        self.GraphIndex = StoredGraphIndex(store.tree)
        self.embedder = None
        self.entity_vdb = None
        if config is not None:
            self._open_entity_search(config)

    def _open_entity_search(self, config) -> None:
        # Retrieval scores query entities against graph node names and searches
        # the entity vectors, so both are opened with the index they serve.
        from Core.pipelines.vdb_index import entity_vector_scope
        from Core.provider.embedding import TextEmbeddingProvider
        from Core.provider.vdb import VectorStore

        embedding_config = config.graph.embedding_config
        # One collection holds every document's entities, so a read without the
        # scope the refiner wrote answers this document's question with another
        # document's graph.
        entity_scope = entity_vector_scope(
            self.TreeIndex.meta_info,
            config.vdb,
            run_directory=getattr(self._store, "run_dir", ""),
        )
        try:
            self.embedder = TextEmbeddingProvider(
                model_name=embedding_config.model_name,
                backend=embedding_config.backend,
                max_length=embedding_config.max_length,
                device=embedding_config.device,
                api_base=embedding_config.api_base,
                api_key=getattr(embedding_config, "api_key", None),
                num_ctx=getattr(embedding_config, "num_ctx", None),
            )
            self.entity_vdb = VectorStore(
                qdrant_url=config.vdb.qdrant_url,
                embedding_model=self.embedder,
                collection_name=config.graph.entity_collection_name,
                api_key_env=config.vdb.api_key_env,
                timeout_seconds=config.vdb.timeout_seconds,
                scope_filter=entity_scope,
                # Same collection the refiner writes, so the same schema. A
                # store opened without this reads named vectors as unnamed and
                # fails on the first query, not at open time.
                hybrid_entity_schema=True,
            )
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, run_dir, config=None, **kwargs) -> "StoredIndex":
        return cls(DocumentStore.open(run_dir, **kwargs), config=config)

    @property
    def store(self) -> DocumentStore:
        return self._store

    def close(self) -> None:
        for resource in (self.entity_vdb, self.embedder, self._store):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        self.entity_vdb = None
        self.embedder = None

    def __enter__(self) -> "StoredIndex":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def _node_payload(node: TreeNodeView, tree_meta: "StoredTreeMeta" = None) -> dict:
    """The retrieval payload for one node.

    The evidence-binding fields were validated against the canonical artifact
    when the tree was built and are stored verbatim, so they are replayed as
    stored -- rebuilding any of them here would break their digests.
    """

    meta = node.meta_info
    payload = dict(node.evidence) if node.evidence else {
        "type": node.type,
        "content": meta.content,
        "page": meta.page_idx,
    }
    payload["index_id"] = node.index_id
    payload["summary"] = node.summary
    payload["parent_id"] = node.parent_id
    payload["depth"] = node.depth
    if tree_meta:
        payload.setdefault("document_version_id", tree_meta.document_version_id)
        payload.setdefault("source_sha256", tree_meta.source_sha256)
        payload.setdefault("source_uri", tree_meta.source_uri)
        payload["index_revision_id"] = tree_meta.index_revision_id
    if meta.block_uid:
        payload["block_uid"] = meta.block_uid
    if meta.img_path:
        payload["crop_uri"] = meta.img_path
    return payload


__all__ = [
    "StoredGraphIndex",
    "StoredIndex",
    "StoredTreeIndex",
    "entity_node_name",
    "split_node_name",
]
