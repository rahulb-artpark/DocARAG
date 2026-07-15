import networkx as nx
from networkx.readwrite import json_graph
import os
import heapq
from collections import defaultdict
from typing import Iterable, Union, Set, List
from numpy import source
from pydantic import BaseModel, Field
import json

import logging

log = logging.getLogger(__name__)


class Entity(BaseModel):
    entity_name: str  # Primary key for entity
    entity_type: str = Field(default="")  # Entity type
    description: str = Field(default="")  # The description of this entity
    source_ids: Set[int] = Field(
        default_factory=set
    )  # Set of source IDs from which this entity is derived

    def __hash__(self):
        """
        Calculates the hash based on the composite key of entity_name and entity_type.
        """
        return hash((self.entity_name, self.entity_type))

    def __eq__(self, other):
        """
        Defines two Entity objects as equal if their entity_name and entity_type match.
        """
        if isinstance(other, Entity):
            return (self.entity_name, self.entity_type) == (
                other.entity_name,
                other.entity_type,
            )
        return False


class Relationship(BaseModel):
    src_entity_name: str  # Name of the entity on the left side of the edge
    tgt_entity_name: str  # Name of the entity on the right side of the edge
    relation_name: str = Field(default="")  # Name of the relation
    weight: float = Field(
        default=0.0
    )  # Weight of the edge, used in GraphRAG and LightRAG
    description: str = Field(
        default=""
    )  # Description of the edge, used in GraphRAG and LightRAG
    source_ids: Set[int] = Field(
        default_factory=set
    )  # Set of source IDs from which this edge is derived


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class Graph:
    _DATA_FILE = "graph_data.json"  # index data file
    _BASE_FILENAME = "graph_data"

    def __init__(self, save_path: str = None, variant: str = None):
        self.kg = nx.Graph()
        # 节点名采用 "entity_name (entity_type)"，确保唯一性
        self.tree2kg = defaultdict(set)  # Maps tree nodes id (int) to graph entities
        # self.name_to_nodes = defaultdict(set)  # entity_name -> set of node names
        self.save_dir = save_path
        self.variant = variant

        # dynamic filename based on variant
        self.data_filename = self._get_filename(variant)

    @classmethod
    def _get_filename(cls, variant: str = None) -> str:
        """内部辅助函数：根据 variant 生成对应的 json 文件名"""
        if variant:
            log.info("Using '%s' variant for graph filename.", variant)
            return f"{cls._BASE_FILENAME}_{variant}.json"
        return f"{cls._BASE_FILENAME}.json"

    def get_all_nodes(self) -> Set[str]:
        """Return all node names (entity_name (entity_type)) in the knowledge graph."""
        return set(self.kg.nodes)

    def _debug_check_add_node(self, node_name: str) -> None:
        """Debugging helper to check if a node can be added."""
        if node_name in self.kg.nodes:
            log.warning(
                f"Warning: Node '{node_name}' already exists in the knowledge graph."
            )
            print("warning here")

    def get_node_name_from_entity(self, entity: Entity) -> str:
        """Generate a node name from an Entity object."""
        return self.get_node_name_from_str(entity.entity_name, entity.entity_type)

    def get_node_name_from_str(self, entity_name: str, entity_type: str) -> str:
        return f"Name: {entity_name}\nType: {entity_type}"

    def add_kg_node(self, entity: Entity) -> None:
        """Add an entity/node to the KG with all its attributes."""
        node_name = self.get_node_name_from_entity(entity)

        self.kg.add_node(node_name, **entity.model_dump())
        # self.name_to_nodes[entity.entity_name].add(node_name)

    def add_kg_edge(self, rel: Relationship, src_type: str, tgt_type: str) -> None:
        """Add a relation/edge between two KG entities with all its attributes."""
        src_node_name = self.get_node_name_from_str(rel.src_entity_name, src_type)
        tgt_node_name = self.get_node_name_from_str(rel.tgt_entity_name, tgt_type)
        if src_node_name not in self.kg.nodes:
            raise KeyError(
                f"Source node '{src_node_name}' not found in knowledge graph."
            )
        if tgt_node_name not in self.kg.nodes:
            raise KeyError(
                f"Target node '{tgt_node_name}' not found in knowledge graph."
            )
        # Add the edge with all attributes from the Relationship model
        self.kg.add_edge(src_node_name, tgt_node_name, **rel.model_dump())

    def link(self, tree_node_id: int, entity_name: str, entity_type: str = "") -> None:
        """Create a bidirectional mapping between a tree node and a KG node."""
        node_name = self.get_node_name_from_str(entity_name, entity_type)
        if node_name not in self.kg:
            raise KeyError(f"KG node '{node_name}' not found in knowledge graph.")
        self.tree2kg[tree_node_id].add(node_name)

    def add_and_link(
        self,
        tree_node_id: int,
        entities: Union[Entity, List[Entity]],
    ) -> None:
        """Add one or more Entity nodes and link to tree node."""
        if isinstance(entities, Entity):
            entities = [entities]
        for entity in entities:
            node_name = self.get_node_name_from_entity(entity)
            source_ids = set(entity.source_ids)
            source_ids.add(tree_node_id)
            if node_name not in self.kg:
                self.add_kg_node(
                    entity.model_copy(update={"source_ids": source_ids})
                )
            else:
                existing = self.kg.nodes[node_name].get("source_ids", set())
                try:
                    source_ids.update(existing or ())
                except TypeError as exc:
                    raise ValueError(
                        f"Entity '{node_name}' has invalid source_ids"
                    ) from exc
                self.kg.nodes[node_name]["source_ids"] = source_ids
            self.link(tree_node_id, entity.entity_name, entity.entity_type)

    def update_entity(
        self, old_entity_name: str, old_entity_type: str, new_entity: Entity
    ) -> None:
        """
        if new entity already exists, it will be updated with new attributes.
        Args:
            old_entity_name (str): 需要被更新的实体节点名称。
            old_entity_type (str): 需要被更新的实体类型。
            new_entity (Entity): 新的实体对象。
        Raises:
            KeyError: 如果实体不存在。
        """
        old_node_name = self.get_node_name_from_str(old_entity_name, old_entity_type)
        new_node_name = self.get_node_name_from_entity(new_entity)
        if old_node_name not in self.kg:
            raise KeyError(f"Entity '{old_node_name}' not found in knowledge graph.")
        new_source_ids = new_entity.source_ids
        if new_node_name != old_node_name:
            # 1. Add new node and copy all edges
            self.kg.add_node(new_node_name, **new_entity.model_dump())
            for neighbor in list(self.kg.neighbors(old_node_name)):
                edge_data = self.kg.get_edge_data(old_node_name, neighbor)
                self.kg.add_edge(new_node_name, neighbor, **edge_data)
            # 2.1 update tree2kg
            for tree_id in new_source_ids:
                # If the old node is in the tree2kg, remove the old name
                self.tree2kg[tree_id].discard(old_node_name)
                self.tree2kg[tree_id].add(new_node_name)

            # 3. remove old node
            self.kg.remove_node(old_node_name)
        else:
            # only update attributes
            self.kg.nodes[old_node_name].update(new_entity.model_dump())
            # update tree2kg
            for tree_id in new_source_ids:
                self.tree2kg[tree_id].add(new_node_name)

    def get_entity(self, entity_name: str, entity_type: str = "") -> Entity:
        """
        Retrieve an entity from the knowledge graph by its name and type.
        Args:
            entity_name (str): The name of the entity to retrieve.
            entity_type (str): The type of the entity to retrieve.
        Returns:
            Entity: The entity object with all its attributes.
        Raises:
            KeyError: If the entity does not exist in the graph.
        """
        node_name = self.get_node_name_from_str(
            entity_name=entity_name, entity_type=entity_type
        )
        # node_name = f"{entity_name} ({entity_type})"
        if node_name not in self.kg.nodes:
            raise KeyError(f"Entity '{node_name}' not found in knowledge graph.")
        return Entity(**self.kg.nodes[node_name])

    def get_entity_by_node_name(self, node_name: str) -> Entity:
        """
        Retrieve an entity from the knowledge graph by its node name.
        Args:
            node_name (str): The name of the node to retrieve.
        Returns:
            Entity: The entity object with all its attributes.
        Raises:
            KeyError: If the node does not exist in the graph.
        """
        if node_name not in self.kg.nodes:
            raise KeyError(f"Node '{node_name}' not found in knowledge graph.")
        return Entity(**self.kg.nodes[node_name])

    def get_kg_subgraph(
        self, tree_node_ids: Iterable[int], copy: bool = True
    ) -> nx.Graph:
        """
        Given one or more tree node IDs, return the induced subgraph of the KG
        containing all linked entities. By default returns a deep copy; if copy=False,
        returns a lightweight view (faster slicing).

        Complexity: O(sum(degree(n)) + |nodes| + |edges|).
        For a few hundred nodes, this remains efficient even if KG has millions of edges.
        """
        # Collect all KG node names for the provided tree nodes
        kg_nodes = set().union(*(self.tree2kg.get(tid, set()) for tid in tree_node_ids))
        sub = self.kg.subgraph(kg_nodes)
        return sub.copy() if copy else sub

    def get_subgraph_data(self, entities: List[str], max_edges: int = 64) -> dict:
        """Return bounded, evidence-bearing entity and relation data.

        Relation selection is ``O(E log max_edges)`` time and ``O(max_edges)``
        memory, favoring relations with stronger weight and source provenance.
        Counts retain an explicit trace that the answer-facing view was bounded.
        """
        if max_edges < 0:
            raise ValueError("max_edges must be non-negative")
        subgraph = self.kg.subgraph(entities)

        def normalized_source_ids(value) -> list:
            if value is None:
                return []
            if isinstance(value, (str, bytes)):
                value = [value]
            try:
                return sorted(set(value), key=lambda item: (str(type(item)), str(item)))
            except TypeError:
                return [value]

        def numeric_weight(value) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        data = {
            "nodes": [],
            "edges": [],
            "edge_count_total": subgraph.number_of_edges(),
            "edges_truncated": False,
        }
        for node_name, attributes in subgraph.nodes(data=True):
            node_data = {
                "node_name": node_name,
                "entity_name": attributes.get("entity_name", ""),
                "entity_type": attributes.get("entity_type", ""),
                "description": attributes.get("description", ""),
                "source_ids": normalized_source_ids(attributes.get("source_ids")),
            }
            data["nodes"].append(node_data)
        selected_edges = []
        for ordinal, (source, target, attributes) in enumerate(
            subgraph.edges(data=True)
        ):
            edge_data = {
                "source_node": source,
                "target_node": target,
                "src_entity_name": attributes.get("src_entity_name", ""),
                "tgt_entity_name": attributes.get("tgt_entity_name", ""),
                "relation_name": attributes.get("relation_name", ""),
                "description": attributes.get("description", ""),
                "weight": attributes.get("weight", 0.0),
                "source_ids": normalized_source_ids(attributes.get("source_ids")),
            }
            source_count = len(edge_data["source_ids"])
            score = (
                numeric_weight(edge_data["weight"]),
                source_count,
                bool(edge_data["description"]),
                str(edge_data["relation_name"]),
                str(source),
                str(target),
            )
            candidate = (score, ordinal, edge_data)
            if len(selected_edges) < max_edges:
                heapq.heappush(selected_edges, candidate)
            elif max_edges and candidate[:2] > selected_edges[0][:2]:
                heapq.heapreplace(selected_edges, candidate)
        selected_edges.sort(key=lambda item: item[:2], reverse=True)
        data["edges"] = [edge for _, _, edge in selected_edges]
        data["edges_truncated"] = data["edge_count_total"] > len(selected_edges)
        return data

    def Entities2TreeNodes(self, entities: List[Entity]) -> List[int]:
        """
        Given KG node names, return all tree node IDs that link to them.
        """
        result = set()
        for ent in entities:
            result.update(ent.source_ids)
        return sorted(result)

    def Entity2TreeNodes(self, ent: Entity) -> List[int]:
        """
        Given an Entity object, return all tree node IDs that link to it.
        """
        res = ent.source_ids
        res = list(res)
        return res

    def NodeName2TreeNodes(self, node_name: str) -> Set[int]:
        """
        Given a node name (entity_name (entity_type)), return all tree node IDs that link to it.
        """
        ent = self.get_entity_by_node_name(node_name)
        res = ent.source_ids
        res = list(res)

        return res

    def remove_self_loops(self) -> int:
        """
        Returns:
        """
        nodes_with_selfloops = list(nx.nodes_with_selfloops(self.kg))

        if not nodes_with_selfloops:
            log.info("No self-loops found in the graph.")
            return 0

        self_loop_edges = [(node, node) for node in nodes_with_selfloops]

        num_removed = len(self_loop_edges)
        log.info(f"Found {num_removed} self-loops. Removing them...")
        self.kg.remove_edges_from(self_loop_edges)
        log.info("All self-loops have been removed.")

    def save_graph(self) -> None:

        if not self.save_dir:
            log.warning("Warning: save_dir is not set. Nothing will be saved.")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        # save_path = os.path.join(self.save_dir, self._DATA_FILE)

        # use dynamic filename based on variant
        save_path = os.path.join(self.save_dir, self.data_filename)

        graph_json_data = json_graph.node_link_data(self.kg, edges="links")

        data_to_save = {
            "graph": graph_json_data,
            "tree2kg": {k: list(v) for k, v in self.tree2kg.items()},
            "variant": self.variant,
        }

        # 3. 保存为格式化的JSON文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, cls=SetEncoder, indent=4, ensure_ascii=False)

        log.info(f"Graph data successfully saved to: {save_path}")

    @classmethod
    def load_from_dir(cls, load_dir: str, variant: str = None) -> "Graph":
        target_filename = cls._get_filename(variant)
        load_path = os.path.join(load_dir, target_filename)
        
        # load_path = os.path.join(load_dir, cls._DATA_FILE)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Error: Missing graph file: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        graph_instance = cls(save_path=load_dir, variant=variant)

        # Match ``save_graph(edges="links")`` explicitly. NetworkX 3.6 changes
        # the default edge-key name, so relying on the library default would
        # make existing BookRAG graph artifacts version-sensitive.
        graph_instance.kg = json_graph.node_link_graph(
            loaded_data["graph"], edges="links"
        )

        for _, node_data in graph_instance.kg.nodes(data=True):
            if "source_ids" in node_data and isinstance(node_data["source_ids"], list):
                node_data["source_ids"] = set(node_data["source_ids"])

        for _, _, edge_data in graph_instance.kg.edges(data=True):
            if "source_ids" in edge_data and isinstance(edge_data["source_ids"], list):
                edge_data["source_ids"] = set(edge_data["source_ids"])

        graph_instance.tree2kg = defaultdict(
            set, {int(k): set(v) for k, v in loaded_data["tree2kg"].items()}
        )

        log.info(f"Graph data successfully loaded from: {load_path}")
        log.info(
            f"Graph contains {len(graph_instance.kg.nodes)} nodes and {len(graph_instance.kg.edges)} edges."
        )
        return graph_instance


if __name__ == "__main__":
    # Example usage
    tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    graph = Graph(save_path=tmp_save_path)
    entity1 = Entity(
        entity_name="Entity1",
        entity_type="TypeA",
        description="First entity",
        source_ids={1},
    )
    entity2 = Entity(
        entity_name="Entity2",
        entity_type="TypeB",
        description="Second entity",
        source_ids={2},
    )

    graph.add_and_link(1, entity1)
    graph.add_and_link(2, entity2)

    relationship = Relationship(
        src_entity_name="Entity1", tgt_entity_name="Entity2", relation_name="related_to"
    )
    graph.add_kg_edge(relationship, src_type="TypeA", tgt_type="TypeB")

    graph.save_graph()

    loaded_graph = Graph.load_from_dir(tmp_save_path)
    print(loaded_graph.get_all_nodes())
    print(loaded_graph.get_entity("Entity1", "TypeA"))
    src_node_name = loaded_graph.get_node_name_from_str("Entity1", "TypeA")
    tgt_node_name = loaded_graph.get_node_name_from_str("Entity2", "TypeB")
    print(
        f"relation: {loaded_graph.kg.get_edge_data(src_node_name, tgt_node_name)['relation_name']}"
    )
