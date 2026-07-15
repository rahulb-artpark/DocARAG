from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from Core.rag.gbc_utils import enhance_graph_with_semantic_links


class _FailIfCalledEmbedder:
    def embed_texts(self, texts):
        raise AssertionError("large scopes must not be embedded at query time")


class _SmallEmbedder:
    def embed_texts(self, texts):
        return np.eye(len(texts))


class RetrievalBoundsTests(unittest.TestCase):
    def test_large_graph_skips_quadratic_semantic_augmentation(self):
        graph = nx.path_graph(9)

        result = enhance_graph_with_semantic_links(
            graph, _FailIfCalledEmbedder(), max_nodes=8
        )

        self.assertIsNot(result, graph)
        self.assertEqual(set(graph.nodes), set(result.nodes))
        self.assertEqual(set(graph.edges), set(result.edges))

    def test_legacy_whole_number_percentile_is_normalized(self):
        graph = nx.path_graph(3)

        result = enhance_graph_with_semantic_links(
            graph, _SmallEmbedder(), x_percentile=90
        )

        self.assertEqual(set(graph.nodes), set(result.nodes))

    def test_invalid_percentile_is_rejected_before_embedding(self):
        with self.assertRaises(ValueError):
            enhance_graph_with_semantic_links(
                nx.path_graph(3), _FailIfCalledEmbedder(), x_percentile=101
            )


if __name__ == "__main__":
    unittest.main()
