from __future__ import annotations

import unittest

from Core.Index.Graph import Entity, Graph, Relationship


class GraphEvidenceTests(unittest.TestCase):
    def test_add_and_link_unions_duplicate_entity_evidence(self):
        graph = Graph()
        graph.add_and_link(
            10,
            Entity(
                entity_name="Revenue",
                entity_type="metric",
                source_ids={10},
            ),
        )
        graph.add_and_link(
            11,
            Entity(
                entity_name="Revenue",
                entity_type="metric",
                source_ids={11, 12},
            ),
        )

        entity = graph.get_entity("Revenue", "metric")
        self.assertEqual({10, 11, 12}, entity.source_ids)
        node_name = graph.get_node_name_from_entity(entity)
        self.assertIn(node_name, graph.tree2kg[10])
        self.assertIn(node_name, graph.tree2kg[11])

    def test_entities_to_tree_nodes_unions_all_sources(self):
        graph = Graph()
        entities = [
            Entity(entity_name="A", entity_type="thing", source_ids={3, 1}),
            Entity(entity_name="B", entity_type="thing", source_ids={2, 3}),
        ]

        self.assertEqual([1, 2, 3], graph.Entities2TreeNodes(entities))

    def test_subgraph_data_preserves_relations_and_source_links(self):
        graph = Graph()
        left = Entity(
            entity_name="Revenue",
            entity_type="metric",
            description="Reported revenue",
            source_ids={10},
        )
        right = Entity(
            entity_name="FY2025", entity_type="period", source_ids={10, 11}
        )
        graph.add_kg_node(left)
        graph.add_kg_node(right)
        graph.add_kg_edge(
            Relationship(
                src_entity_name="Revenue",
                tgt_entity_name="FY2025",
                relation_name="reported_for",
                description="Revenue is reported for this period",
                weight=0.9,
                source_ids={10},
            ),
            src_type="metric",
            tgt_type="period",
        )

        result = graph.get_subgraph_data(list(graph.get_all_nodes()))

        self.assertEqual(2, len(result["nodes"]))
        self.assertEqual(1, len(result["edges"]))
        self.assertEqual("reported_for", result["edges"][0]["relation_name"])
        self.assertEqual([10], result["edges"][0]["source_ids"])

    def test_dense_subgraph_relation_evidence_is_bounded_and_traced(self):
        graph = Graph()
        for index in range(20):
            graph.add_kg_node(
                Entity(
                    entity_name=f"E{index}",
                    entity_type="entity",
                    source_ids={index},
                )
            )
        node_names = sorted(graph.get_all_nodes())
        for left_index, left in enumerate(node_names):
            for right_index, right in enumerate(node_names[left_index + 1 :], left_index + 1):
                graph.kg.add_edge(
                    left,
                    right,
                    relation_name="related",
                    weight=float(left_index + right_index),
                    source_ids={left_index},
                )

        result = graph.get_subgraph_data(node_names, max_edges=7)

        self.assertEqual(190, result["edge_count_total"])
        self.assertEqual(7, len(result["edges"]))
        self.assertTrue(result["edges_truncated"])
        weights = [edge["weight"] for edge in result["edges"]]
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertEqual(37.0, weights[0])

    def test_malformed_optional_edge_metadata_does_not_break_retrieval(self):
        graph = Graph()
        graph.kg.add_node("left", entity_name="Left", source_ids=None)
        graph.kg.add_node("right", entity_name="Right", source_ids={2})
        graph.kg.add_edge(
            "left",
            "right",
            relation_name="related",
            weight="not-a-number",
            source_ids=None,
        )

        result = graph.get_subgraph_data(["left", "right"])

        self.assertEqual([], result["nodes"][0]["source_ids"])
        self.assertEqual([], result["edges"][0]["source_ids"])


if __name__ == "__main__":
    unittest.main()
