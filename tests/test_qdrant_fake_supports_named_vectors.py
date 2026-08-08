"""The Qdrant fake has to model the schema the tree index is moving to.

Every later stage of the named-vector migration is tested against this fake, so
the properties it has to reproduce are worth pinning on their own: a point
carrying several vectors by name, a point carrying only *some* of them, a query
that targets one name, and a sparse vector scored on shared terms.

Before this, `create_collection` took no `sparse_vectors_config` at all and
`_cosine` assumed one flat vector per point -- so neither the entity collection's
hybrid schema nor the tree collection's future one could be faked.
"""

from __future__ import annotations

import unittest

from tests.qdrant_fakes import FakeModels, FakeQdrantClient


def _point(point_id, vectors, **payload):
    payload.setdefault("_docarag_record_id", point_id)
    payload.setdefault("_docarag_content", point_id)
    return FakeModels.PointStruct(id=point_id, vector=vectors, payload=payload)


def _client_with(points, *, sparse=None):
    client = FakeQdrantClient()
    client.create_collection(
        collection_name="c",
        vectors_config={
            "text": FakeModels.VectorParams(size=2, distance="Cosine"),
            "image": FakeModels.VectorParams(size=2, distance="Cosine"),
        },
        sparse_vectors_config=sparse,
    )
    client.upsert(collection_name="c", points=points, wait=True)
    return client


def _ids(response):
    return [hit.id for hit in response.points]


class APointCarriesVectorsByName(unittest.TestCase):
    def test_a_query_reaches_the_vector_it_names(self):
        client = _client_with(
            [
                _point("a", {"text": [1.0, 0.0], "image": [0.0, 1.0]}),
                _point("b", {"text": [0.0, 1.0], "image": [1.0, 0.0]}),
            ]
        )

        by_text = client.query_points(
            collection_name="c", query=[1.0, 0.0], query_filter=None,
            limit=2, with_payload=True, with_vectors=False, using="text",
        )
        by_image = client.query_points(
            collection_name="c", query=[1.0, 0.0], query_filter=None,
            limit=2, with_payload=True, with_vectors=False, using="image",
        )

        # Same query vector, opposite winners -- which is the whole point of
        # naming: the node matched on its picture is not the one matched on its
        # prose.
        self.assertEqual("a", _ids(by_text)[0])
        self.assertEqual("b", _ids(by_image)[0])

    def test_a_point_without_the_named_vector_is_not_a_candidate(self):
        client = _client_with(
            [
                _point("has-image", {"text": [1.0, 0.0], "image": [1.0, 0.0]}),
                _point("text-only", {"text": [1.0, 0.0]}),
            ]
        )

        response = client.query_points(
            collection_name="c", query=[1.0, 0.0], query_filter=None,
            limit=10, with_payload=True, with_vectors=False, using="image",
        )

        self.assertEqual(["has-image"], _ids(response))

    def test_an_unnamed_point_still_answers_an_unnamed_query(self):
        client = FakeQdrantClient()
        client.create_collection(
            collection_name="flat",
            vectors_config=FakeModels.VectorParams(size=2, distance="Cosine"),
        )
        client.upsert(collection_name="flat", points=[_point("a", [1.0, 0.0])], wait=True)

        response = client.query_points(
            collection_name="flat", query=[1.0, 0.0], query_filter=None,
            limit=1, with_payload=True, with_vectors=False,
        )

        self.assertEqual(["a"], _ids(response))

    def test_the_upsert_record_reports_vectors_by_name(self):
        client = _client_with([_point("a", {"text": [1.0, 0.0]})])

        self.assertEqual([{"text": [1.0, 0.0]}], client.upsert_calls[0]["vectors"])
        self.assertEqual({"text": [1.0, 0.0]}, client.records["a"]["vectors"])

    def test_an_unnamed_point_reports_its_vector_under_none(self):
        client = FakeQdrantClient()
        client.create_collection(
            collection_name="flat",
            vectors_config=FakeModels.VectorParams(size=2, distance="Cosine"),
        )
        client.upsert(collection_name="flat", points=[_point("a", [1.0, 0.0])], wait=True)

        self.assertEqual([{None: [1.0, 0.0]}], client.upsert_calls[0]["vectors"])


class ASparseVectorIsScoredOnSharedTerms(unittest.TestCase):
    def _sparse_client(self):
        return _client_with(
            [
                _point("zernike", {"lexical": FakeModels.SparseVector([11, 22], [1.0, 1.0])}),
                _point("unrelated", {"lexical": FakeModels.SparseVector([33, 44], [1.0, 1.0])}),
            ],
            sparse={"lexical": FakeModels.SparseVectorParams(
                index=FakeModels.SparseIndexParams(), modifier=FakeModels.Modifier.IDF
            )},
        )

    def test_a_keyword_query_reaches_the_point_holding_the_term(self):
        client = self._sparse_client()

        response = client.query_points(
            collection_name="c",
            query=FakeModels.SparseVector([11], [1.0]),
            query_filter=None, limit=10,
            with_payload=True, with_vectors=False, using="lexical",
        )

        self.assertEqual(["zernike"], [h.id for h in response.points if h.score > 0])

    def test_the_sparse_config_survives_collection_creation(self):
        client = self._sparse_client()
        params = client.get_collection("c").config.params

        self.assertIn("lexical", params.sparse_vectors)
        self.assertEqual("idf", params.sparse_vectors["lexical"].modifier)

    def test_a_dense_query_does_not_score_a_sparse_vector(self):
        client = self._sparse_client()

        response = client.query_points(
            collection_name="c", query=[1.0, 0.0], query_filter=None,
            limit=10, with_payload=True, with_vectors=False, using="lexical",
        )

        self.assertEqual([0.0, 0.0], [hit.score for hit in response.points])


class ABatchQueryTargetsEachNameSeparately(unittest.TestCase):
    def test_each_request_uses_its_own_vector_name(self):
        client = _client_with(
            [
                _point("a", {"text": [1.0, 0.0], "image": [0.0, 1.0]}),
                _point("b", {"text": [0.0, 1.0], "image": [1.0, 0.0]}),
            ]
        )

        responses = client.query_batch_points(
            collection_name="c",
            requests=[
                FakeModels.QueryRequest(query=[1.0, 0.0], limit=1, using="text"),
                FakeModels.QueryRequest(query=[1.0, 0.0], limit=1, using="image"),
            ],
        )

        self.assertEqual(["a"], _ids(responses[0]))
        self.assertEqual(["b"], _ids(responses[1]))
        self.assertEqual(1, len(client.batch_query_calls))


if __name__ == "__main__":
    unittest.main()
