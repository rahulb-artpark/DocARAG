"""The fake's model signatures have to match the client we actually ship.

A fake that accepts a field the real model rejects turns a live failure into a
green suite. That happened: QueryRequest spells it ``with_vector`` while
``client.query_points`` takes ``with_vectors``, the fake accepted the plural in
both, and every named-vector query failed against a real Qdrant with

    ValidationError: with_vectors  Extra inputs are not permitted

after the whole suite had passed. The pre-existing ``batch_search`` carried the
same mistake, latent only because nothing calls it.

These check the shapes against the installed qdrant-client rather than against
the fake, so the next divergence fails here instead of in production.
"""

from __future__ import annotations

import inspect
import unittest

from tests.qdrant_fakes import FakeModels


def _real_models():
    from qdrant_client import models

    return models


class TheRequestModelMatches(unittest.TestCase):
    def test_query_request_field_names_exist_on_the_real_model(self):
        real = set(_real_models().QueryRequest.model_fields)
        fake = {f.name for f in FakeModels.QueryRequest.__dataclass_fields__.values()}

        self.assertEqual(
            set(), fake - real, "the fake accepts fields qdrant-client rejects"
        )

    def test_the_singular_spelling_is_the_one_query_request_uses(self):
        real = _real_models().QueryRequest.model_fields

        self.assertIn("with_vector", real)
        self.assertNotIn("with_vectors", real)

    def test_query_points_really_does_take_the_plural(self):
        # Not a typo on our side: the client is genuinely inconsistent, which is
        # why one call site keeps the plural.
        from qdrant_client import QdrantClient

        parameters = inspect.signature(QdrantClient.query_points).parameters

        self.assertIn("with_vectors", parameters)
        self.assertNotIn("with_vector", parameters)

    def test_named_vector_targeting_is_expressible(self):
        self.assertIn("using", _real_models().QueryRequest.model_fields)


class TheCollectionModelsMatch(unittest.TestCase):
    def test_sparse_vector_params_take_an_index_and_a_modifier(self):
        real = _real_models().SparseVectorParams.model_fields

        self.assertIn("index", real)
        self.assertIn("modifier", real)

    def test_idf_is_a_real_modifier(self):
        self.assertTrue(hasattr(_real_models().Modifier, "IDF"))

    def test_a_sparse_vector_carries_indices_and_values(self):
        real = set(_real_models().SparseVector.model_fields)
        fake = {f.name for f in FakeModels.SparseVector.__dataclass_fields__.values()}

        self.assertEqual(set(), fake - real)

    def test_create_collection_accepts_a_sparse_config(self):
        from qdrant_client import QdrantClient

        parameters = inspect.signature(QdrantClient.create_collection).parameters

        self.assertIn("sparse_vectors_config", parameters)
        self.assertIn("vectors_config", parameters)


class OurOwnCallsAreValid(unittest.TestCase):
    """Build the requests the tree path builds, against the real models."""

    def test_a_dense_named_query_request_validates(self):
        models = _real_models()

        request = models.QueryRequest(
            query=[0.1, 0.2],
            filter=None,
            limit=8,
            with_payload=True,
            with_vector=False,
            using="image",
        )

        self.assertEqual("image", request.using)

    def test_a_sparse_query_request_validates(self):
        models = _real_models()

        request = models.QueryRequest(
            query=models.SparseVector(indices=[7], values=[1.0]),
            filter=None,
            limit=8,
            with_payload=True,
            with_vector=False,
            using="lexical",
        )

        self.assertEqual("lexical", request.using)

    def test_search_tree_vectors_builds_requests_the_real_models_accept(self):
        """The decisive one: production code, real models, fake transport.

        Every earlier test builds a request by hand. This runs the code that
        actually ships and lets qdrant-client validate what it produced -- which
        is what turns "the fake agreed with us" into "the client agrees with
        us".
        """

        from qdrant_client import models

        from Core.provider.vdb import VectorSchema, VectorStore
        from tests.qdrant_fakes import FakeQdrantClient

        class _Embedder:
            MM_EMBEDDER = True

            def embed_texts(self, texts):
                return [[0.1, 0.2] for _ in texts]

            def close(self):
                pass

        client = FakeQdrantClient()
        store = VectorStore(
            _Embedder(),
            collection_name="c",
            client=client,
            models=models,
            vector_schema=VectorSchema.TREE_NODE,
        )
        client.create_collection(
            collection_name="c",
            vectors_config={
                name: models.VectorParams(size=2, distance=models.Distance.COSINE)
                for name in ("text", "context", "image")
            },
            sparse_vectors_config={
                "lexical": models.SparseVectorParams(
                    index=models.SparseIndexParams(), modifier=models.Modifier.IDF
                )
            },
        )

        rows = store.search_tree_vectors("a query", top_k=3, lexical_terms={7: 1.0})

        self.assertEqual({"text", "context", "image", "lexical"}, set(rows))
        requests = client.batch_query_calls[0]["requests"]
        self.assertEqual(4, len(requests))
        self.assertEqual(
            ["context", "image", "lexical", "text"],
            sorted(r.using for r in requests),
        )

    def test_the_tree_collection_config_validates(self):
        models = _real_models()
        from Core.provider.vdb import _TREE_DENSE_VECTORS, _TREE_SPARSE_VECTOR

        vectors = {
            name: models.VectorParams(size=2048, distance=models.Distance.COSINE)
            for name in _TREE_DENSE_VECTORS
        }
        sparse = {
            _TREE_SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(), modifier=models.Modifier.IDF
            )
        }

        self.assertEqual({"text", "context", "image"}, set(vectors))
        self.assertEqual("idf", sparse[_TREE_SPARSE_VECTOR].modifier.value)


if __name__ == "__main__":
    unittest.main()
