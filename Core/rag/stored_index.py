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
from typing import Any, Iterable, List, Sequence

import networkx as nx

from docarag.document_store import DocumentStore
from docarag.tree_store import TreeNodeView

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

    def get_kg_subgraph(self, tree_node_ids: Iterable[int], copy: bool = True):
        """Build the subgraph those nodes support, and nothing wider."""

        graph = nx.Graph()
        for edge in self._tree.induced_subgraph(list(tree_node_ids)):
            graph.add_node(edge.source, entity_name=edge.source)
            graph.add_node(edge.target, entity_name=edge.target)
            graph.add_edge(
                edge.source,
                edge.target,
                relation_name=edge.predicate,
                weight=edge.weight if edge.weight is not None else 1.0,
                description=edge.description,
            )
        return graph

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
        run_dir = getattr(store, "run_dir", None)
        manifest: dict = {}
        if run_dir is not None:
            try:
                manifest = json.loads(
                    (run_dir / "run-manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                manifest = {}
        source = manifest.get("source") or {}
        # The canonical identity was recorded at admission; the ledger is the
        # source of truth, and no artifact file is opened at query time.
        canonical_sha = ""
        try:
            row = store._borrow().execute(
                "SELECT canonical_artifact_sha256 FROM docarag.index_revisions "
                "WHERE revision_key = %s",
                (store.revision_key,),
            ).fetchone()
            canonical_sha = str((row or {}).get("canonical_artifact_sha256") or "")
        except Exception:
            canonical_sha = ""
        source_path = str(source.get("path") or "")

        scope = getattr(store, "scope", None)
        return cls(
            tenant_id=getattr(scope, "tenant_id", "") or "",
            corpus_id=getattr(scope, "corpus_id", "") or "",
            document_version_id=getattr(scope, "document_version_id", "") or "",
            index_revision_id=getattr(store, "revision_id", "") or "",
            source_sha256=str(source.get("sha256") or ""),
            source_uri=str(source.get("uri") or "")
            or (Path(source_path).as_uri() if source_path else ""),
            file_name=Path(source_path or "document.pdf").name,
            file_path=source_path,
            snapshot_path=str(manifest.get("source_snapshot") or ""),
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
        from Core.provider.embedding import TextEmbeddingProvider
        from Core.provider.vdb import VectorStore

        embedding_config = config.graph.embedding_config
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
