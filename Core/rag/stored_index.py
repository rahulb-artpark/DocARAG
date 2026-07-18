"""The index retrieval reads, served from stored evidence.

Retrieval asks a document for a handful of things: the nodes mentioning an
entity, the subtree beneath a section, the graph those nodes induce, and the
evidence behind a final few nodes.  This answers all of them from the store,
one query each, so a question's memory follows the question rather than the
size of the document.
"""

from __future__ import annotations

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

    __slots__ = ("_tree",)

    def __init__(self, tree) -> None:
        self._tree = tree

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

        return [_node_payload(node) for node in self._tree.nodes(node_ids)]

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


class StoredIndex:
    """What a query holds open for one document."""

    __slots__ = ("GraphIndex", "TreeIndex", "_store")

    def __init__(self, store: DocumentStore) -> None:
        self._store = store
        self.TreeIndex = StoredTreeIndex(store.tree)
        self.GraphIndex = StoredGraphIndex(store.tree)

    @classmethod
    def open(cls, run_dir, **kwargs) -> "StoredIndex":
        return cls(DocumentStore.open(run_dir, **kwargs))

    @property
    def store(self) -> DocumentStore:
        return self._store

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "StoredIndex":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def _node_payload(node: TreeNodeView) -> dict:
    """The retrieval payload for one node, with its evidence bindings."""

    meta = node.meta_info
    payload = {
        "index_id": node.index_id,
        "type": node.type,
        "content": meta.content,
        "page": meta.page_idx,
        "summary": node.summary,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "block_uid": meta.block_uid,
        "geometry": meta.geometry,
    }
    if meta.img_path:
        payload["img_path"] = meta.img_path
    if meta.caption:
        payload["caption"] = meta.caption
    if meta.footnote:
        payload["footnote"] = meta.footnote
    if meta.table_body:
        payload["table_body"] = meta.table_body
    return payload


__all__ = [
    "StoredGraphIndex",
    "StoredIndex",
    "StoredTreeIndex",
    "entity_node_name",
    "split_node_name",
]
