"""One point per tree node, carrying every vector that node has.

The index used to emit a point per representation, so a figure's picture and its
prose were separate candidates in one undifferentiated ranked list and nothing
could ask for one rather than the other. They now share a point under named
vectors, and a query names the vector it wants.

Two properties carry most of the weight. A point may hold only *some* of the
names -- a paragraph has no picture -- and a query for a name it lacks must not
reach it. And a hit's id is suffixed with the vector it matched, because one
point can be hit on several and callers key audit rows by that id.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Core.provider.vdb import (
    VectorSchema,
    VectorStore,
    _PAYLOAD_CONTENTS,
    _TREE_SPARSE_VECTOR,
    tree_point_record_id,
    tree_vector_identity,
)
from tests.qdrant_fakes import FakeModels, FakeQdrantClient

SOURCE = "a" * 64


class _Embedder:
    """Text embeds by length; images embed to a constant, so the two differ."""

    MM_EMBEDDER = True

    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0
        self.fused_calls = 0

    def embed_texts(self, texts):
        self.text_calls += 1
        return [[float(len(t)), 1.0] for t in texts]

    def embed_images(self, images):
        self.image_calls += 1
        return [[0.0, 9.0] for _ in images]

    def embed_fused(self, images, texts):
        self.fused_calls += 1
        return [[1.0, 9.0] for _ in images]

    def close(self):
        pass


def _store(embedder=None):
    client = FakeQdrantClient()
    store = VectorStore(
        embedder or _Embedder(),
        collection_name="c",
        client=client,
        models=FakeModels,
        vector_schema=VectorSchema.TREE_NODE,
        scope_filter={"tenant_id": "t", "corpus_id": "c"},
    )
    return store, client


def _meta(node_id, **extra):
    return {
        "document_version_id": "docv_1",
        "source_sha256": SOURCE,
        "node_id": node_id,
        "node_type": "image",
        "index_kind": "tree",
        "index_revision_id": "rev_1",
        **extra,
    }


class ANodeIsOnePoint(unittest.TestCase):
    def test_three_vectors_become_one_point(self):
        store, client = _store()
        with tempfile.TemporaryDirectory() as directory:
            crop = f"{directory}/f.jpg"
            Path(crop).write_bytes(b"\xff\xd8jpeg")
            store.add_tree_node_points(
                records=[
                    {
                        "kind": "node",
                        "metadata": _meta(1),
                        "vectors": {
                            "text": "the Zernike surface",
                            "context": "section 2.3",
                            "image": crop,
                        },
                        "lexical_terms": {7: 1.0},
                        "lexical_source": "the Zernike surface section 2.3",
                    }
                ]
            )

        self.assertEqual(1, len(client.upsert_calls[0]["points"]))
        vectors = client.upsert_calls[0]["vectors"][0]
        self.assertEqual(
            {"text", "context", "image", _TREE_SPARSE_VECTOR}, set(vectors)
        )

    def test_a_point_carries_only_the_vectors_it_has(self):
        store, client = _store()
        store.add_tree_node_points(
            records=[
                {
                    "kind": "node",
                    "metadata": _meta(2, node_type="text"),
                    "vectors": {"text": "a paragraph"},
                    "lexical_terms": {3: 1.0},
                    "lexical_source": "a paragraph",
                }
            ]
        )

        self.assertEqual(
            {"text", _TREE_SPARSE_VECTOR}, set(client.upsert_calls[0]["vectors"][0])
        )

    def test_a_point_with_no_lexical_terms_carries_no_sparse_vector(self):
        store, client = _store()
        store.add_tree_node_points(
            records=[
                {
                    "kind": "node",
                    "metadata": _meta(3, node_type="text"),
                    "vectors": {"text": "x"},
                    "lexical_terms": {},
                }
            ]
        )

        self.assertEqual({"text"}, set(client.upsert_calls[0]["vectors"][0]))

    def test_every_vector_is_embedded_in_one_call_per_kind(self):
        embedder = _Embedder()
        store, _ = _store(embedder)
        with tempfile.TemporaryDirectory() as directory:
            crop = f"{directory}/f.jpg"
            Path(crop).write_bytes(b"\xff\xd8jpeg")
            store.add_tree_node_points(
                records=[
                    {
                        "kind": "node",
                        "metadata": _meta(index),
                        "vectors": {"text": f"t{index}", "context": f"c{index}",
                                    "image": crop},
                        "lexical_terms": {},
                    }
                    for index in range(4)
                ]
            )

        # Eight texts and four crops, but one call each -- a second call could
        # observe a changed model and produce vectors that do not compare.
        self.assertEqual(1, embedder.text_calls)
        self.assertEqual(1, embedder.image_calls)

    def test_a_crop_without_fused_text_is_a_pure_image_embedding(self):
        embedder = _Embedder()
        store, _ = _store(embedder)
        with tempfile.TemporaryDirectory() as directory:
            crop = f"{directory}/f.jpg"
            Path(crop).write_bytes(b"\xff\xd8jpeg")
            store.add_tree_node_points(
                records=[
                    {"kind": "node", "metadata": _meta(1),
                     "vectors": {"image": crop}, "lexical_terms": {}}
                ]
            )

        self.assertEqual(1, embedder.image_calls)
        self.assertEqual(0, embedder.fused_calls)

    def test_a_crop_with_fused_text_goes_through_the_fused_call(self):
        embedder = _Embedder()
        store, _ = _store(embedder)
        with tempfile.TemporaryDirectory() as directory:
            crop = f"{directory}/f.jpg"
            Path(crop).write_bytes(b"\xff\xd8jpeg")
            store.add_tree_node_points(
                records=[
                    {"kind": "node", "metadata": _meta(1),
                     "vectors": {"image": crop}, "image_text": "Figure 4",
                     "lexical_terms": {}}
                ]
            )

        self.assertEqual(1, embedder.fused_calls)
        self.assertEqual(0, embedder.image_calls)

    def test_the_payload_keeps_a_content_per_vector(self):
        store, client = _store()
        store.add_tree_node_points(
            records=[
                {
                    "kind": "node",
                    "metadata": _meta(1, node_type="title"),
                    "vectors": {"text": "heading", "context": "the section summary"},
                    "lexical_terms": {},
                }
            ]
        )

        payload = client.upsert_calls[0]["points"][0].payload
        self.assertEqual(
            {"text": "heading", "context": "the section summary"},
            payload[_PAYLOAD_CONTENTS],
        )
        # The single content stays populated for readers that name no vector.
        self.assertEqual("heading", payload["_docarag_content"])

    def test_the_point_carries_no_point_level_modality(self):
        store, client = _store()
        store.add_tree_node_points(
            records=[
                {"kind": "node", "metadata": _meta(1, node_type="text"),
                 "vectors": {"text": "x"}, "lexical_terms": {}}
            ]
        )

        payload = client.upsert_calls[0]["points"][0].payload
        self.assertNotIn("modality", payload)
        self.assertEqual("node", payload["record_kind"])

    def test_metadata_claiming_a_modality_is_refused(self):
        store, _ = _store()
        with self.assertRaisesRegex(ValueError, "point-level modality"):
            store.add_tree_node_points(
                records=[
                    {"kind": "node", "metadata": _meta(1, modality="text"),
                     "vectors": {"text": "x"}, "lexical_terms": {}}
                ]
            )

    def test_an_unknown_vector_name_is_refused(self):
        store, _ = _store()
        with self.assertRaisesRegex(ValueError, "unknown tree vector names: summary"):
            store.add_tree_node_points(
                records=[
                    {"kind": "node", "metadata": _meta(1),
                     "vectors": {"summary": "x"}, "lexical_terms": {}}
                ]
            )


class ThePointIdNamesTheNodeNotAReading(unittest.TestCase):
    def _id(self, **overrides):
        base = {
            "kind": "node",
            "metadata": _meta(1),
            "representations": {"text": tree_vector_identity("text", "hello")},
        }
        base.update(overrides)
        return tree_point_record_id(**base)

    def test_the_same_node_and_vectors_give_the_same_id(self):
        self.assertEqual(self._id(), self._id())

    def test_changing_one_vector_changes_the_id(self):
        other = self._id(
            representations={"text": tree_vector_identity("text", "goodbye")}
        )
        self.assertNotEqual(self._id(), other)

    def test_adding_a_vector_changes_the_id(self):
        other = self._id(
            representations={
                "text": tree_vector_identity("text", "hello"),
                "context": tree_vector_identity("context", "around it"),
            }
        )
        self.assertNotEqual(self._id(), other)

    def test_changing_the_lexical_source_changes_the_id(self):
        self.assertNotEqual(self._id(), self._id(lexical_source="different terms"))

    def test_a_node_and_a_semantic_element_never_share_an_id(self):
        self.assertNotEqual(self._id(), self._id(kind="semantic_element"))

    def test_the_id_does_not_name_a_modality(self):
        # The whole point: the old id carried the modality twice.
        self.assertTrue(self._id().startswith("record_node_"))

    def test_a_point_without_source_identity_is_refused(self):
        from Core.provider.vdb import MissingVectorSourceIdentityError

        with self.assertRaises(MissingVectorSourceIdentityError):
            tree_point_record_id(
                kind="node",
                metadata={"document_version_id": "docv_1", "source_sha256": SOURCE},
                representations={"text": tree_vector_identity("text", "x")},
            )

    def test_a_point_with_no_vectors_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one vector"):
            tree_point_record_id(kind="node", metadata=_meta(1), representations={})


class AQueryNamesTheVectorItWants(unittest.TestCase):
    def _populated(self):
        store, client = _store()
        with tempfile.TemporaryDirectory() as directory:
            crop = f"{directory}/f.jpg"
            Path(crop).write_bytes(b"\xff\xd8jpeg")
            store.add_tree_node_points(
                records=[
                    {
                        "kind": "node",
                        "metadata": _meta(1),
                        "vectors": {"text": "a much longer piece of prose", "image": crop},
                        "lexical_terms": {11: 1.0},
                        "lexical_source": "zernike",
                    },
                    {
                        "kind": "node",
                        "metadata": _meta(2, node_type="text"),
                        "vectors": {"text": "short"},
                        "lexical_terms": {22: 1.0},
                        "lexical_source": "unrelated",
                    },
                ]
            )
        return store, client

    def test_each_name_returns_its_own_ranked_list(self):
        store, _ = self._populated()

        rows = store.search_tree_vectors("query", top_k=5)

        self.assertEqual({"text", "context", "image"}, set(rows))
        # Only node 1 has a crop, so the image list cannot contain node 2.
        self.assertEqual([1], [r["metadata"]["node_id"] for r in rows["image"]])
        self.assertEqual([], rows["context"])

    def test_a_hit_reports_the_vector_it_matched(self):
        store, _ = self._populated()

        rows = store.search_tree_vectors("query", top_k=5)

        self.assertEqual("image", rows["image"][0]["metadata"]["vector_name"])
        self.assertEqual("image", rows["image"][0]["metadata"]["modality"])
        self.assertEqual("text", rows["text"][0]["metadata"]["modality"])

    def test_the_row_id_is_suffixed_with_the_vector(self):
        # One point can be hit on several vectors and callers key audit rows by
        # this id; a bare point id would let one overwrite the other.
        store, _ = self._populated()

        rows = store.search_tree_vectors("query", top_k=5)
        image_id = rows["image"][0]["id"]
        text_id = next(r["id"] for r in rows["text"] if r["metadata"]["node_id"] == 1)

        self.assertTrue(image_id.endswith("#image"))
        self.assertTrue(text_id.endswith("#text"))
        self.assertNotEqual(image_id, text_id)
        self.assertEqual(image_id.split("#")[0], text_id.split("#")[0])

    def test_an_image_hit_returns_the_crop_as_its_content(self):
        store, _ = self._populated()

        rows = store.search_tree_vectors("query", top_k=5)

        self.assertTrue(rows["image"][0]["content"].endswith(".jpg"))
        self.assertEqual(
            "a much longer piece of prose",
            next(r for r in rows["text"] if r["metadata"]["node_id"] == 1)["content"],
        )

    def test_a_lexical_query_is_one_more_ranked_list(self):
        store, _ = self._populated()

        rows = store.search_tree_vectors("query", top_k=5, lexical_terms={11: 1.0})

        self.assertIn(_TREE_SPARSE_VECTOR, rows)
        scored = [r for r in rows[_TREE_SPARSE_VECTOR] if r["distance"] < 1.0]
        self.assertEqual([1], [r["metadata"]["node_id"] for r in scored])

    def test_every_name_is_asked_in_one_round_trip(self):
        embedder = _Embedder()
        client = FakeQdrantClient()
        store = VectorStore(
            embedder, collection_name="c", client=client, models=FakeModels,
            vector_schema=VectorSchema.TREE_NODE,
            scope_filter={"tenant_id": "t", "corpus_id": "c"},
        )
        store.add_tree_node_points(
            records=[{"kind": "node", "metadata": _meta(1, node_type="text"),
                      "vectors": {"text": "x"}, "lexical_terms": {5: 1.0},
                      "lexical_source": "x"}]
        )
        before = embedder.text_calls

        store.search_tree_vectors("query", top_k=3, lexical_terms={5: 1.0})

        self.assertEqual(1, len(client.batch_query_calls))
        self.assertEqual(4, len(client.batch_query_calls[0]["requests"]))
        self.assertEqual(before + 1, embedder.text_calls)

    def test_an_unnamed_search_merges_the_dense_names(self):
        # mm_rag names no vector; it must still get one ranked list, the way the
        # single unnamed vector used to answer.
        store, _ = self._populated()

        rows = store.search("query", top_k=10)

        self.assertTrue(rows)
        self.assertEqual(
            sorted(row["distance"] for row in rows),
            [row["distance"] for row in rows],
        )


if __name__ == "__main__":
    unittest.main()
