"""Editing a summary re-embeds what that summary reached, and nothing else.

Rewriting one summary used to cost a whole document: the revision resealed,
the vector attestation stopped matching, and the next question failed closed
with "TreeVDB attestation does not match the sealed store revision". The only
cure was a full vdb rebuild -- 23,274 points and every dense vector for
test.pdf, to change four.

The reconciler diffs instead. A tree point's ID is content-addressed over each
vector's source (``tree_point_record_id``: "changing any one vector's source
re-IDs the point"), so re-projecting the tree and diffing IDs against Qdrant
finds exactly the points whose text moved -- including the ones an edit reached
indirectly. A section heading's summary is quoted into the ``context`` vector
of every figure and table beneath it, and those points re-ID themselves without
anyone having to write down the dependency.

Two properties carry the weight here. Editing a *paragraph's* summary must move
**zero** points, because a paragraph's vector is its content and its summary is
never embedded -- if that test fails, the diff is being driven by the write
rather than by the text. And the BM25 normalizer stays frozen: recomputing it
would change every stored weight in the index while changing no point ID at
all, so the diff could not see the work and the whole index would need
rewriting anyway.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Core.Index.Tree import NodeType
from Core.pipelines.vdb_index import (
    DEFAULT_MAX_AVERAGE_LENGTH_DRIFT,
    TreeVectorReconciliation,
    VectorIndexBindingError,
    _projected_tree_points,
    reconcile_tree_vector_index,
)
from Core.provider.vdb import VectorSchema, VectorStore
from tests.qdrant_fakes import FakeModels, FakeQdrantClient

SOURCE = "a" * 64
DOCUMENT = "docv_1"
REVISION = "rev_1"
#: Every scope carries tenant and corpus: the collection is shared, and the
#: store refuses to open without them for exactly that reason.
SCOPE = {
    "tenant_id": "default",
    "corpus_id": "default",
    "document_version_id": DOCUMENT,
    "index_revision_id": REVISION,
}


class _Embedder:
    MM_EMBEDDER = False

    def __init__(self):
        self.embedded: list[str] = []

    def embed_texts(self, texts):
        self.embedded.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        pass


class _Meta:
    """Node meta the way the store serves it: named fields, else None.

    ``vdb_tree_source._NodeMeta`` answers any field it was not given with None
    -- "absence is None, exactly as the serialized meta behaved" -- so a
    fixture that raises instead would be testing a shape the stage never sees.
    """

    def __init__(self, node):
        self.content = node["content"]
        self.summary = node.get("summary", "")
        self.caption = ""
        self.footnote = ""
        self.table_body = ""
        self.img_path = ""
        self.page_idx = 1
        self.page_id = "page_1"
        self.document_version_id = DOCUMENT
        self.source_sha256 = SOURCE
        self.source_sha = SOURCE
        self.pdf_id = node["id"]
        self.canonical_block_id = f"block_{node['id']}"
        self.canonical_revision_id = f"blockrev_{node['id']}"

    def __getattr__(self, name):
        return None


def _meta(node):
    return _Meta(node)


def _tree(nodes):
    """A tree shaped the way the vector stage reads one, and no further."""

    built = []
    by_id = {}
    for node in nodes:
        item = SimpleNamespace(
            index_id=node["id"],
            type=node["type"],
            summary=node.get("summary", ""),
            depth=node.get("depth", 1),
            outline_node=node["type"] is NodeType.TITLE,
            parent=None,
            children=[],
            meta_info=_meta(node),
        )
        by_id[node["id"]] = item
        built.append(item)
    for node in nodes:
        parent_id = node.get("parent")
        if parent_id is not None:
            by_id[node["id"]].parent = by_id[parent_id]
            by_id[parent_id].children.append(by_id[node["id"]])
    root = SimpleNamespace(
        index_id=0,
        type=NodeType.ROOT,
        summary="",
        depth=0,
        outline_node=False,
        parent=None,
        children=[],
        meta_info=_meta({"id": 0, "content": "", "summary": ""}),
    )
    return SimpleNamespace(
        nodes=[root, *built],
        root_node=root,
        get_nodes=lambda hasRoot=False: built,
        meta_info=SimpleNamespace(
            source_sha256=SOURCE, document_version_id=DOCUMENT
        ),
        save_dir="",
        store_backed=True,
        store_seal="b" * 64,
    )


def _vdb_cfg(**extra):
    return SimpleNamespace(
        qdrant_url="http://localhost:6333",
        collection_name="c",
        api_key_env=None,
        timeout_seconds=5.0,
        mm_embedding=False,
        index_batch_size=64,
        manifest_dir_name="unused",
        embedding_config=SimpleNamespace(
            model_name="m", device=None, backend="local",
            api_base=None, max_length=512,
        ),
        **extra,
    )


#: A heading, a figure beneath it, and a paragraph. The figure's context vector
#: quotes the heading's summary; the paragraph's vector is its own content.
BASE = [
    {"id": 1, "type": NodeType.TITLE, "content": "2.3 Wolter Surface",
     "summary": "Defines the Wolter surface."},
    {"id": 2, "type": NodeType.IMAGE, "content": "A tapered cylinder.",
     "summary": "The Wolter surface rendered in three dimensions.", "parent": 1},
    {"id": 3, "type": NodeType.TEXT, "content": "The shell is hyperbolic.",
     "summary": "Describes the shell.", "parent": 1},
]


def _edited(node_id, summary):
    return [
        {**node, "summary": summary} if node["id"] == node_id else dict(node)
        for node in BASE
    ]


class TheProjectionIsTheDependencyGraph(unittest.TestCase):
    """Which points an edit reaches, decided by content-addressed IDs alone."""

    def _ids(self, nodes):
        # Through the store, because the scope it applies is part of the ID and
        # a projection alone cannot say what a point will be keyed under.
        projected, _average, _count = _projected_tree_points(
            _tree(nodes), _vdb_cfg()
        )
        store = VectorStore(
            _Embedder(),
            collection_name="c",
            client=FakeQdrantClient(),
            models=FakeModels,
            vector_schema=VectorSchema.TREE_NODE,
            scope_filter=SCOPE,
        )
        return set(store.tree_point_ids(projected))

    def test_editing_a_figures_summary_moves_that_figure(self):
        before = self._ids(BASE)
        after = self._ids(_edited(2, "A rendered conic, not a cylinder."))

        self.assertEqual(1, len(after - before))
        self.assertEqual(1, len(before - after))

    def test_editing_a_heading_summary_moves_the_figure_beneath_it(self):
        # Nobody wrote this dependency down: _section_context_text quotes the
        # parent's summary into the child's context vector, so the child's ID
        # changes with it.
        before = self._ids(BASE)
        after = self._ids(_edited(1, "Defines the grazing-incidence surface."))

        self.assertEqual(2, len(after - before), "heading and its figure")

    def test_editing_a_paragraphs_summary_moves_nothing(self):
        # A paragraph's text vector is its content. Its summary is read live by
        # the reranker and never embedded, so no point may move -- this is what
        # proves the diff follows the text and not the write.
        self.assertEqual(
            self._ids(BASE), self._ids(_edited(3, "A completely different gloss."))
        )

    def test_every_point_keeps_a_stable_id_when_nothing_changes(self):
        self.assertEqual(self._ids(BASE), self._ids(BASE))


class OnlyTheDifferenceIsEmbedded(unittest.TestCase):
    def _reconcile(self, before, after, **kwargs):
        client = FakeQdrantClient()
        embedder = _Embedder()
        # What a full rebuild of ``before`` would have attested, which is what
        # the frozen normalizer is: the corpus mean as of the last rebuild.
        frozen = _projected_tree_points(_tree(before), _vdb_cfg())[1]

        def _open(*_args, **_kw):
            return VectorStore(
                embedder,
                collection_name="c",
                client=client,
                models=FakeModels,
                vector_schema=VectorSchema.TREE_NODE,
                scope_filter=SCOPE,
            )

        with patch("Core.pipelines.vdb_index.VectorStore", _open), patch(
            "Core.pipelines.vdb_index.TextEmbeddingProvider", lambda **_k: embedder
        ), patch(
            "Core.pipelines.vdb_index.tree_vector_scope",
            lambda *_a, **_k: SCOPE,
        ):
            first = reconcile_tree_vector_index(
                _tree(before), _vdb_cfg(), frozen_average_length=frozen
            )
            embedder.embedded.clear()
            second = reconcile_tree_vector_index(
                _tree(after),
                _vdb_cfg(),
                frozen_average_length=frozen,
                # Three short nodes: editing one moves their mean by more than
                # any real document's, where one summary is a rounding error
                # against thousands. The ceiling has its own tests below.
                **{"max_average_length_drift": 1.0, **kwargs},
            )
        return first, second, embedder, client

    def test_an_unchanged_tree_embeds_nothing_and_deletes_nothing(self):
        _first, second, embedder, _client = self._reconcile(BASE, BASE)

        self.assertEqual(0, second.added)
        self.assertEqual(0, second.removed)
        self.assertEqual([], embedder.embedded)

    def test_one_edited_figure_re_embeds_one_point(self):
        _first, second, embedder, _client = self._reconcile(
            BASE, _edited(2, "A rendered conic, not a cylinder.")
        )

        self.assertEqual(1, second.added)
        self.assertEqual(1, second.removed)
        self.assertEqual(2, second.unchanged)
        self.assertIn("A rendered conic, not a cylinder.", " ".join(embedder.embedded))

    def test_the_replaced_point_is_gone_not_merely_shadowed(self):
        _first, second, _embedder, client = self._reconcile(
            BASE, _edited(2, "A rendered conic, not a cylinder.")
        )
        stored = client.collections["c"]["records"]

        self.assertEqual(second.point_count, len(stored))
        self.assertNotIn(
            "The Wolter surface rendered in three dimensions.",
            " ".join(str(point.payload) for point in stored.values()),
        )

    def test_a_paragraph_edit_touches_qdrant_not_at_all(self):
        _first, second, embedder, client = self._reconcile(
            BASE, _edited(3, "A completely different gloss.")
        )

        self.assertEqual((0, 0), (second.added, second.removed))
        self.assertEqual([], embedder.embedded)
        self.assertEqual([], client.delete_calls)


class TheNormalizerIsHeldStill(unittest.TestCase):
    def test_the_reported_average_is_the_one_the_weights_used(self):
        outcome = TreeVectorReconciliation(
            added=1, removed=1, unchanged=2, point_count=3,
            average_length=8.0, corpus_average_length=9.0,
        )

        self.assertEqual(8.0, outcome.average_length)
        self.assertAlmostEqual(0.125, outcome.drift)

    def test_drift_past_the_ceiling_refuses_instead_of_reconciling(self):
        with self.assertRaisesRegex(VectorIndexBindingError, "drifted"):
            reconcile_tree_vector_index(
                _tree(BASE),
                _vdb_cfg(),
                # Far below any real corpus mean, so the live mean is a long
                # way from it and the ceiling has to bite.
                frozen_average_length=0.5,
            )

    def test_the_ceiling_is_a_tenth(self):
        self.assertEqual(0.10, DEFAULT_MAX_AVERAGE_LENGTH_DRIFT)

    def test_a_non_positive_average_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            reconcile_tree_vector_index(
                _tree(BASE), _vdb_cfg(), frozen_average_length=0.0
            )


class DeletionStaysInsideTheScope(unittest.TestCase):
    """The entity collection is shared between documents; this must never roam."""

    def _store(self, client):
        return VectorStore(
            _Embedder(),
            collection_name="c",
            client=client,
            models=FakeModels,
            vector_schema=VectorSchema.TREE_NODE,
            scope_filter={**SCOPE, "index_revision_id": REVISION},
        )

    def _seed(self, store, node_id, document_version_id):
        store.add_tree_node_points(
            records=[
                {
                    "kind": "node",
                    "metadata": {
                        "document_version_id": document_version_id,
                        "source_sha256": SOURCE,
                        "node_id": node_id,
                        "node_type": "text",
                        "index_kind": "tree",
                        "index_revision_id": REVISION,
                    },
                    "vectors": {"text": f"node {node_id}"},
                    "lexical_terms": {1: 1.0},
                    "lexical_source": f"node {node_id}",
                }
            ]
        )

    def test_live_ids_see_only_this_documents_points(self):
        client = FakeQdrantClient()
        mine = self._store(client)
        self._seed(mine, 1, DOCUMENT)
        theirs = VectorStore(
            _Embedder(), collection_name="c", client=client, models=FakeModels,
            vector_schema=VectorSchema.TREE_NODE,
            scope_filter={**SCOPE, "document_version_id": "docv_other"},
        )
        self._seed(theirs, 2, "docv_other")

        self.assertEqual(1, len(mine.live_point_ids()))
        self.assertEqual(2, len(client.collections["c"]["records"]))

    def test_deleting_another_documents_point_is_refused(self):
        client = FakeQdrantClient()
        mine = self._store(client)
        self._seed(mine, 1, DOCUMENT)
        theirs = VectorStore(
            _Embedder(), collection_name="c", client=client, models=FakeModels,
            vector_schema=VectorSchema.TREE_NODE,
            scope_filter={**SCOPE, "document_version_id": "docv_other"},
        )
        self._seed(theirs, 2, "docv_other")
        outsider = next(iter(theirs.live_point_ids()))

        with self.assertRaisesRegex(RuntimeError, "outside this store's scope"):
            mine.delete_points([outsider])
        self.assertEqual(2, len(client.collections["c"]["records"]))

    def test_paging_returns_every_point_not_just_the_first_page(self):
        client = FakeQdrantClient()
        store = self._store(client)
        for node_id in range(1, 8):
            self._seed(store, node_id, DOCUMENT)

        self.assertEqual(7, len(store.live_point_ids(page_size=2)))
        self.assertGreater(len(client.scroll_calls), 1)


if __name__ == "__main__":
    unittest.main()
