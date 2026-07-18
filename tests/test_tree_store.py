from __future__ import annotations

import unittest

from docarag.tree_store import (
    MAX_NODES_PER_READ,
    GraphEdge,
    NodeMeta,
    TreeNodeView,
    TreeStore,
    _bounded_ids,
    _view,
)


class _Result:
    def __init__(self, rows):
        self.rows = list(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _RecordingConnection:
    """A connection that records what was asked and never closes itself."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.entered = 0

    def execute(self, query, params=None):
        self.calls.append((" ".join(query.split()), tuple(params or ())))
        return _Result(self.rows)

    def __enter__(self):  # pragma: no cover - must never be used
        self.entered += 1
        return self

    def __exit__(self, *_exc):  # pragma: no cover
        return False


def row(**overrides):
    base = {
        "node_no": 7,
        "node_kind": "NodeType.IMAGE",
        "depth": 3,
        "parent_no": 2,
        "label": "Figure 1",
        "summary": "A summary.",
        "block_uid": "blk-7",
        "page_no": 4,
        "geometry": {"bbox": [0, 0, 1, 1]},
        "is_outline": False,
        "content": "recovered text",
        "caption": "FIGURE 1.",
        "footnote": "",
        "table_body": "",
        "cas_uri": "cas://sha256/" + "a" * 64,
    }
    base.update(overrides)
    return base


class TreeStoreReadTests(unittest.TestCase):
    def store(self, rows=()):
        connection = _RecordingConnection(rows)
        return TreeStore(lambda: connection, 42), connection

    def test_a_connection_is_borrowed_and_never_entered(self):
        # Entering a psycopg connection ends the caller's transaction on exit,
        # which would close it between two reads of the same slice.
        store, connection = self.store([row()])

        store.nodes([7])
        store.node(7)

        self.assertEqual(0, connection.entered)
        self.assertEqual(2, len(connection.calls))

    def test_every_read_is_scoped_to_one_revision(self):
        store, connection = self.store()

        store.nodes()
        store.subtree([1])
        store.nodes_at_depth(2)
        store.nodes_for_entity("speaker")
        store.induced_subgraph([1, 2])

        for query, params in connection.calls:
            self.assertIn("revision_key = %s", query)
            self.assertEqual(42, params[0])

    def test_ancestors_and_descendants_use_containment_not_recursion(self):
        store, connection = self.store([row()])

        store.path_from_root(7)
        store.subtree([1])

        ancestors, subtree = (query for query, _ in connection.calls)
        self.assertIn("path @>", ancestors)
        self.assertIn("path <@", subtree)
        for query in (ancestors, subtree):
            self.assertNotIn("RECURSIVE", query.upper())

    def test_naming_no_nodes_asks_the_database_nothing(self):
        store, connection = self.store([row()])

        self.assertEqual([], store.nodes([]))
        self.assertEqual([], store.subtree([]))
        self.assertEqual({}, store.retrieval_texts([]))
        self.assertEqual([], store.induced_subgraph([]))
        self.assertEqual({}, store.entities_for_nodes([]))

        self.assertEqual([], connection.calls)

    def test_a_read_is_bounded_however_many_nodes_are_named(self):
        store, connection = self.store([row()])

        store.nodes(list(range(MAX_NODES_PER_READ * 3)))

        _query, params = connection.calls[0]
        self.assertEqual(MAX_NODES_PER_READ, len(params[1]))

    def test_node_ids_are_deduplicated_and_validated(self):
        self.assertEqual([3, 1], _bounded_ids([3, 1, 3, -2, True, "x", None]))

    def test_a_row_becomes_the_view_retrieval_reads(self):
        view = _view(row())

        self.assertIsInstance(view, TreeNodeView)
        self.assertEqual(7, view.index_id)
        self.assertEqual("NodeType.IMAGE", view.type)
        self.assertEqual(2, view.parent_id)
        self.assertEqual("recovered text", view.meta_info.content)
        self.assertEqual(4, view.meta_info.page_idx)
        self.assertTrue(view.meta_info.img_path.startswith("cas://"))
        self.assertEqual("A summary.", view.summary)

    def test_a_node_without_recovered_text_falls_back_to_its_label(self):
        view = _view(row(content=None))

        self.assertEqual("Figure 1", view.meta_info.content)

    def test_meta_dumps_the_channels_an_answer_may_cite(self):
        dumped = _view(row()).meta_info.model_dump()

        self.assertEqual(
            {
                "content",
                "page_idx",
                "img_path",
                "caption",
                "footnote",
                "table_body",
                "block_uid",
                "geometry",
            },
            set(dumped),
        )

    def test_a_missing_root_is_reported_rather_than_invented(self):
        store, _connection = self.store([])

        self.assertIsNone(store.root())
        self.assertIsNone(store.node(1))
        self.assertEqual(0, store.max_depth())

    def test_an_entity_lookup_may_be_narrowed_by_kind(self):
        store, connection = self.store([{"node_no": 3}])

        store.nodes_for_entity("speaker", "HARDWARE")

        _query, params = connection.calls[0]
        self.assertEqual((42, "speaker", "HARDWARE", "HARDWARE"), params)

    def test_a_subgraph_row_becomes_a_binary_edge(self):
        store, _connection = self.store(
            [
                {
                    "src": "amplifier",
                    "dst": "speaker",
                    "predicate": "drives",
                    "weight": 8.0,
                    "description": "the amplifier drives the speaker",
                }
            ]
        )

        edges = store.induced_subgraph([1])

        self.assertEqual(
            GraphEdge("amplifier", "speaker", "drives", 8.0, "the amplifier drives the speaker"),
            edges[0],
        )

    def test_the_subgraph_is_capped_so_one_question_cannot_load_a_corpus(self):
        store, connection = self.store([])

        store.induced_subgraph([1, 2, 3])

        _query, params = connection.calls[0]
        self.assertIsInstance(params[2], int)
        self.assertGreater(params[2], 0)

    def test_the_reranker_feed_is_returned_by_node(self):
        store, _connection = self.store(
            [{"node_no": 5, "retrieval_text": "a caption"}]
        )

        self.assertEqual({5: "a caption"}, store.retrieval_texts([5]))

    def test_construction_rejects_an_unusable_scope(self):
        with self.assertRaises(TypeError):
            TreeStore(None, 1)
        with self.assertRaises(TypeError):
            TreeStore(lambda: None, "not-a-key")


if __name__ == "__main__":
    unittest.main()
