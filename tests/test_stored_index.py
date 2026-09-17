from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval_engine.rag.stored_indexes import (
    StoredTreeMeta,
    StoredGraphIndex,
    StoredIndex,
    StoredTreeIndex,
    entity_node_name,
    split_node_name,
)
from document_services.persistence.tree_store import GraphEdge, NodeMeta, TreeNodeView


def node(index_id: int, parent_id: int | None = 0, **meta) -> TreeNodeView:
    return TreeNodeView(
        index_id=index_id,
        type=meta.pop("type", "NodeType.TEXT"),
        depth=meta.pop("depth", 1),
        parent_id=parent_id,
        summary=meta.pop("summary", ""),
        meta_info=NodeMeta(**meta),
    )


class _FakeTree:
    """Records what retrieval asked for, and answers with fixed slices."""

    def __init__(self, nodes=(), edges=(), entity_nodes=()):
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._entity_nodes = list(entity_nodes)
        self.calls: list[tuple[str, tuple]] = []
        self.revision_key = 1

    def _record(self, name, *args):
        self.calls.append((name, args))

    def nodes(self, node_ids=None):
        self._record("nodes", node_ids)
        if node_ids is None:
            return self._nodes
        wanted = set(node_ids)
        return [item for item in self._nodes if item.index_id in wanted]

    def node(self, node_id):
        self._record("node", node_id)
        return next((i for i in self._nodes if i.index_id == node_id), None)

    def root(self):
        self._record("root")
        return next((i for i in self._nodes if i.parent_id is None), None)

    def siblings(self, node_id):
        self._record("siblings", node_id)
        return []

    def path_from_root(self, node_id):
        self._record("path_from_root", node_id)
        return [i for i in self._nodes if i.index_id <= node_id]

    def subtree(self, node_ids):
        self._record("subtree", node_ids)
        return self._nodes

    def nodes_at_depth(self, depth):
        self._record("nodes_at_depth", depth)
        return [i for i in self._nodes if i.depth == depth]

    def nodes_of_kind(self, kind):
        self._record("nodes_of_kind", kind)
        return [i for i in self._nodes if i.type == kind]

    def ancestor_at_depth(self, node_id, depth):
        self._record("ancestor_at_depth", node_id, depth)
        return next((i for i in self._nodes if i.depth == depth), None)

    def max_depth(self):
        self._record("max_depth")
        return max((i.depth for i in self._nodes), default=0)

    def retrieval_texts(self, node_ids):
        self._record("retrieval_texts", tuple(node_ids))
        return {i: "text" for i in node_ids}

    def nodes_for_entity(self, canonical, entity_kind=None):
        self._record("nodes_for_entity", canonical, entity_kind)
        return list(self._entity_nodes)

    def entities_for_nodes(self, node_ids):
        canonicals = {edge.source for edge in self._edges} | {edge.target for edge in self._edges}
        return {node_id: canonicals for node_id in node_ids if node_id in self._entity_nodes}

    def induced_subgraph(self, node_ids):
        self._record("induced_subgraph", tuple(node_ids))
        return self._edges


class EntityNameTests(unittest.TestCase):
    def test_a_node_name_carries_the_entity_and_its_type(self):
        self.assertEqual(
            "Name: speaker\nType: HARDWARE", entity_node_name("speaker", "HARDWARE")
        )

    def test_a_node_name_is_recovered_into_its_parts(self):
        self.assertEqual(
            ("speaker", "HARDWARE"), split_node_name("Name: speaker\nType: HARDWARE")
        )

    def test_a_bare_name_is_read_as_an_entity_without_a_type(self):
        self.assertEqual(("speaker", None), split_node_name("speaker"))


class StoredTreeIndexTests(unittest.TestCase):
    def index(self):
        tree = _FakeTree(
            nodes=[
                node(0, parent_id=None, type="root", depth=0),
                node(1, depth=1),
                node(2, parent_id=1, depth=2, type="NodeType.IMAGE",
                     content="Speaker", page_idx=4, img_path="cas://sha256/" + "a" * 64),
            ]
        )
        return StoredTreeIndex(tree), tree

    def test_the_root_is_the_node_without_a_parent(self):
        index, _tree = self.index()

        self.assertEqual(0, index.root_node.index_id)

    def test_listing_nodes_can_exclude_the_root(self):
        index, _tree = self.index()

        self.assertEqual([1, 2], [n.index_id for n in index.get_nodes(hasRoot=False)])
        self.assertEqual(3, len(index.get_nodes(hasRoot=True)))

    def test_a_node_payload_carries_the_evidence_an_answer_cites(self):
        index, _tree = self.index()

        payload = index.get_nodes_data([2])[0]

        self.assertEqual(2, payload["index_id"])
        self.assertEqual("Speaker", payload["content"])
        self.assertEqual(4, payload["page"])
        self.assertTrue(payload["crop_uri"].startswith("cas://"))

    def test_empty_channels_are_omitted_rather_than_sent_blank(self):
        index, _tree = self.index()

        payload = index.get_nodes_data([1])[0]

        self.assertNotIn("crop_uri", payload)
        self.assertNotIn("caption", payload)
        self.assertNotIn("table_body", payload)

    def test_each_operation_is_one_question_to_the_store(self):
        index, tree = self.index()

        index.get_subtree_nodes([1])
        index.get_path_from_root(2)
        index.get_max_depth()
        index.get_filtered_nodes("NodeType.IMAGE")

        self.assertEqual(
            ["subtree", "path_from_root", "max_depth", "nodes_of_kind"],
            [name for name, _ in tree.calls],
        )


class StoredGraphIndexTests(unittest.TestCase):
    def index(self):
        tree = _FakeTree(
            entity_nodes=[3, 7],
            edges=[
                GraphEdge("amplifier", "speaker", "drives", 8.0, "it drives it"),
                GraphEdge("speaker", "cabinet", "mounted_in", 2.0, ""),
            ],
        )
        return StoredGraphIndex(tree), tree

    def test_a_graph_name_resolves_to_the_nodes_that_mention_it(self):
        index, tree = self.index()

        found = index.NodeName2TreeNodes("Name: speaker\nType: HARDWARE")

        self.assertEqual([3, 7], found)
        self.assertEqual(("nodes_for_entity", ("speaker", "HARDWARE")), tree.calls[0])

    def test_an_unnamed_entity_asks_the_store_nothing(self):
        index, tree = self.index()

        self.assertEqual([], index.NodeName2TreeNodes(""))
        self.assertEqual([], tree.calls)

    def test_the_subgraph_is_a_graph_ranking_can_run_over(self):
        index, _tree = self.index()

        graph = index.get_kg_subgraph([3, 7])

        self.assertEqual(3, graph.number_of_nodes())
        self.assertEqual(2, graph.number_of_edges())
        self.assertEqual(
            "drives", graph.edges[entity_node_name("amplifier", ""), entity_node_name("speaker", "")]["relation_name"]
        )

    def test_subgraph_data_is_bounded_by_the_caller(self):
        index, _tree = self.index()

        data = index.get_subgraph_data(["Name: speaker\nType: HARDWARE"], max_edges=1)

        self.assertEqual(1, len(data["edges"]))
        self.assertEqual(["speaker"], [item["id"] for item in data["nodes"]])

    def test_asking_about_no_entities_returns_an_empty_trace(self):
        index, tree = self.index()

        self.assertEqual({"nodes": [], "edges": []}, index.get_subgraph_data([]))
        self.assertEqual([], tree.calls)


class StoredIndexTests(unittest.TestCase):
    class _Store:
        def __init__(self):
            self.tree = _FakeTree()
            self.closed = 0

        def close(self):
            self.closed += 1

    def test_an_index_exposes_both_views_over_one_store(self):
        store = self._Store()

        index = StoredIndex(store)

        self.assertIsInstance(index.TreeIndex, StoredTreeIndex)
        self.assertIsInstance(index.GraphIndex, StoredGraphIndex)
        self.assertIs(store, index.store)

    def test_closing_the_index_releases_the_document(self):
        store = self._Store()

        with StoredIndex(store):
            self.assertEqual(0, store.closed)

        self.assertEqual(1, store.closed)


class _ScopedStore:
    """A store carrying the identity the entity scope is derived from."""

    def __init__(self, run_dir, source_sha256=None):
        self.tree = _FakeTree()
        self.run_dir = run_dir
        self.revision_id = "rev-1"
        self.revision_key = "rev-key"
        self.scope = SimpleNamespace(
            tenant_id="tenant-a",
            corpus_id="corpus-a",
            document_version_id="docv_1",
        )
        self._row = {
            "canonical_artifact_sha256": "",
            "source_sha256": source_sha256 or "",
            "source_uri": f"file://{run_dir}/doc.pdf",
        }
        self.closed = 0

    def _borrow(self):
        # The identity comes from index_revisions joined to document_versions;
        # a run directory is not consulted.
        row = self._row

        class _Cursor:
            def execute(self, _sql, _params=None):
                return self

            def fetchone(self):
                return row

        return _Cursor()

    def close(self):
        self.closed += 1


class StoredEntitySearchTests(unittest.TestCase):
    """How the query path opens the collection the refiner wrote.

    Both facts pinned here were wrong in production and no test held either:
    the store was opened without the hybrid schema, so every entity search
    raised; and without the writer's scope, so it answered one document's
    question from another document's graph.
    """

    SHA = "e2be53beeeab9232eb51badbe5982796f14c128304b9e7254027e94ad69515df"

    def _config(self):
        embedding = SimpleNamespace(
            model_name="m", backend="ollama", max_length=512,
            device="cpu", api_base="http://localhost:11434",
            api_key=None, num_ctx=None,
        )
        return SimpleNamespace(
            graph=SimpleNamespace(
                embedding_config=embedding,
                entity_collection_name="entities",
            ),
            vdb=SimpleNamespace(
                qdrant_url="http://localhost:6335",
                api_key_env="QDRANT_API_KEY",
                timeout_seconds=10.0,
                tenant_id="tenant-a",
                corpus_id="corpus-a",
                document_version_id="docv_1",
                index_revision_id="rev-1",
            ),
        )

    def _open(self, directory):
        store = _ScopedStore(Path(directory), source_sha256=self.SHA)
        opened = {}

        def vector_store(**kwargs):
            opened.update(kwargs)
            return Mock()

        with patch("retrieval_engine.providers.embeddings.TextEmbeddingProvider", Mock()), patch(
            "retrieval_engine.indexes.vector_store.VectorStore", vector_store
        ):
            index = StoredIndex(store, config=self._config())
        return index, opened

    def test_the_entity_store_is_opened_with_the_hybrid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            _index, opened = self._open(directory)

        # Named dense vectors plus a sparse one cannot be read by an
        # unqualified search; without this the first question raises.
        self.assertIs(True, opened["hybrid_entity_schema"])

    def test_the_entity_store_is_opened_with_the_writers_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            _index, opened = self._open(directory)

        self.assertEqual(
            {
                "tenant_id": "tenant-a",
                "corpus_id": "corpus-a",
                "document_version_id": "docv_1",
                "index_revision_id": "rev-1",
                "source_sha256": self.SHA,
                "index_kind": "entity",
            },
            opened["scope_filter"],
        )


if __name__ == "__main__":
    unittest.main()


class TheQueryPathReadsIdentityFromTheStore(unittest.TestCase):
    """A query must not need a run directory to know what it is answering about.

    StoredTreeMeta.from_store opened run-manifest.json for the source's digest,
    URI and path -- a file read on the query path for facts the ledger already
    records at admission. document_versions holds the digest and the URI, and a
    file:// URI is where a local path comes from.
    """

    SHA = "c" * 64

    def _meta(self, uri):
        with tempfile.TemporaryDirectory() as directory:
            store = _ScopedStore(Path(directory), source_sha256=self.SHA)
            store._row["source_uri"] = uri
            return StoredTreeMeta.from_store(store)

    def test_the_digest_comes_from_the_ledger(self):
        self.assertEqual(self.SHA, self._meta("file:///docs/report.pdf").source_sha256)

    def test_a_file_uri_yields_its_path_and_name(self):
        meta = self._meta("file:///docs/report.pdf")
        self.assertEqual("/docs/report.pdf", meta.file_path)
        self.assertEqual("report.pdf", meta.file_name)

    def test_a_non_file_uri_has_no_local_path(self):
        meta = self._meta("s3://bucket/report.pdf")
        self.assertEqual("", meta.file_path)
        self.assertEqual("s3://bucket/report.pdf", meta.source_uri)

    def test_a_percent_encoded_path_is_decoded(self):
        self.assertEqual(
            "/docs/annual report.pdf",
            self._meta("file:///docs/annual%20report.pdf").file_path,
        )

    def test_no_manifest_is_ever_opened(self):
        # The run directory is empty; the identity still resolves.
        meta = self._meta("file:///docs/report.pdf")
        self.assertEqual(self.SHA, meta.source_sha256)
