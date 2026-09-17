"""A store knows which vector shape its collection has, and refuses the others.

Qdrant cannot convert an unnamed vector into named ones, so a collection's shape
is fixed at creation. Three shapes now share one VectorStore: the unnamed chunk
collections, the hybrid entity collection, and the named tree-node collection.

The guards matter more than the dispatch. A tree collection that was never
renamed still holds unnamed vectors; writing named points into it, or unnamed
points into the new one, would store vectors under a name nothing queries and
the symptom would be an empty result rather than an error. Every mismatch fails
closed at the first call.
"""

from __future__ import annotations

import unittest

from retrieval_engine.indexes.vector_store import (
    VectorSchema,
    _TREE_DENSE_VECTORS,
    _TREE_SPARSE_VECTOR,
)
from tests.qdrant_fakes import FakeModels, FakeQdrantClient, qdrant_store


class _Embedder:
    MM_EMBEDDER = False

    def embed_texts(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        pass


def _store(schema=None, **kwargs):
    from retrieval_engine.indexes.vector_store import VectorStore

    client = FakeQdrantClient()
    store = VectorStore(
        _Embedder(),
        collection_name="c",
        client=client,
        models=FakeModels,
        vector_schema=schema,
        **kwargs,
    )
    return store, client


class TheSchemaIsChosenOnce(unittest.TestCase):
    def test_the_default_is_the_unnamed_shape(self):
        store, _ = _store()
        self.assertIs(VectorSchema.UNNAMED, store.vector_schema)
        self.assertFalse(store.hybrid_entity_schema)

    def test_the_entity_flag_still_selects_the_hybrid_shape(self):
        # Its three call sites are unchanged by this migration, one of them in
        # the graph stage.
        store, _ = _store(hybrid_entity_schema=True)
        self.assertIs(VectorSchema.HYBRID_ENTITY, store.vector_schema)
        self.assertTrue(store.hybrid_entity_schema)

    def test_the_tree_schema_is_selected_explicitly(self):
        store, _ = _store(VectorSchema.TREE_NODE)
        self.assertIs(VectorSchema.TREE_NODE, store.vector_schema)
        self.assertFalse(store.hybrid_entity_schema)

    def test_a_string_names_a_schema(self):
        store, _ = _store("tree-node")
        self.assertIs(VectorSchema.TREE_NODE, store.vector_schema)

    def test_two_spellings_of_the_same_thing_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            _store(VectorSchema.TREE_NODE, hybrid_entity_schema=True)

    def test_an_unknown_schema_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown vector_schema"):
            _store("multi-vector")


class TheCollectionIsCreatedInItsOwnShape(unittest.TestCase):
    def test_a_tree_collection_carries_named_dense_vectors_and_a_sparse_one(self):
        store, client = _store(VectorSchema.TREE_NODE)
        store._ensure_collection(2)

        created = client.collections["c"]
        self.assertEqual(set(_TREE_DENSE_VECTORS), set(created["vectors_config"]))
        self.assertIn(_TREE_SPARSE_VECTOR, created["sparse_vectors_config"])
        self.assertEqual(
            "idf", created["sparse_vectors_config"][_TREE_SPARSE_VECTOR].modifier
        )

    def test_the_unnamed_collection_is_unchanged(self):
        store, client = _store()
        store._ensure_collection(2)

        created = client.collections["c"]
        self.assertEqual(2, created["vectors_config"].size)
        self.assertIsNone(created["sparse_vectors_config"])

    def test_the_entity_collection_is_unchanged(self):
        store, client = _store(hybrid_entity_schema=True)
        store._ensure_collection(2)

        created = client.collections["c"]
        self.assertEqual({"name", "context"}, set(created["vectors_config"]))
        self.assertIn("lexical", created["sparse_vectors_config"])


class AMismatchedCollectionFailsClosed(unittest.TestCase):
    def _existing(self, schema, vectors_config, sparse=None):
        store, client = _store(schema)
        client.create_collection(
            collection_name="c",
            vectors_config=vectors_config,
            sparse_vectors_config=sparse,
        )
        return store

    def test_a_tree_store_refuses_a_collection_that_was_never_renamed(self):
        store = self._existing(
            VectorSchema.TREE_NODE, FakeModels.VectorParams(size=2, distance="Cosine")
        )
        with self.assertRaisesRegex(RuntimeError, "predates the tree-node schema"):
            store._validate_existing_collection(2)

    def test_an_unnamed_store_refuses_the_renamed_collection(self):
        store = self._existing(
            None,
            {n: FakeModels.VectorParams(size=2, distance="Cosine")
             for n in _TREE_DENSE_VECTORS},
        )
        with self.assertRaisesRegex(RuntimeError, "one unnamed dense vector"):
            store._validate_existing_collection(2)

    def test_a_tree_collection_missing_a_dense_vector_is_refused(self):
        store = self._existing(
            VectorSchema.TREE_NODE,
            {"text": FakeModels.VectorParams(size=2, distance="Cosine")},
            sparse={_TREE_SPARSE_VECTOR: FakeModels.SparseVectorParams()},
        )
        with self.assertRaisesRegex(RuntimeError, "missing dense vectors: context, image"):
            store._validate_existing_collection(2)

    def test_a_tree_collection_without_its_sparse_vector_is_refused(self):
        # Otherwise the symptom is an empty lexical leg, not a missing index.
        store = self._existing(
            VectorSchema.TREE_NODE,
            {n: FakeModels.VectorParams(size=2, distance="Cosine")
             for n in _TREE_DENSE_VECTORS},
        )
        with self.assertRaisesRegex(RuntimeError, "predates hybrid search"):
            store._validate_existing_collection(2)

    def test_tree_dense_vectors_must_agree_on_dimension(self):
        store = self._existing(
            VectorSchema.TREE_NODE,
            {
                "text": FakeModels.VectorParams(size=2, distance="Cosine"),
                "context": FakeModels.VectorParams(size=2, distance="Cosine"),
                "image": FakeModels.VectorParams(size=4, distance="Cosine"),
            },
            sparse={_TREE_SPARSE_VECTOR: FakeModels.SparseVectorParams()},
        )
        with self.assertRaisesRegex(RuntimeError, "disagree on dimension"):
            store._validate_existing_collection(2)


class TheUnnamedWritersRefuseANamedCollection(unittest.TestCase):
    def test_add_texts_is_refused_on_a_tree_store(self):
        store, _ = _store(VectorSchema.TREE_NODE)
        with self.assertRaisesRegex(RuntimeError, "one unnamed vector per point"):
            store.add_texts(texts=["a"], metadatas=[{}])

    def test_add_images_is_refused_on_a_tree_store(self):
        store, _ = _store(VectorSchema.TREE_NODE)
        with self.assertRaisesRegex(RuntimeError, "one unnamed vector per point"):
            store.add_images(image_paths=["/tmp/x.jpg"], metadatas=[{}])

    def test_add_texts_still_works_on_an_unnamed_store(self):
        store, client = _store()
        store.add_texts(texts=["hello"], metadatas=[{}])
        self.assertEqual(1, len(client.upsert_calls))

    def test_the_entity_writer_is_refused_on_a_tree_store(self):
        store, _ = _store(VectorSchema.TREE_NODE)
        with self.assertRaisesRegex(RuntimeError, "hybrid_entity_schema=True"):
            store.add_entity_points(records=[{}])


if __name__ == "__main__":
    unittest.main()
