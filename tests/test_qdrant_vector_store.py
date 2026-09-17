from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from unittest.mock import Mock, patch

from retrieval_engine.configuration.vector_store import VDBConfig
from retrieval_engine.indexes.vector_store import VectorStore, qdrant_point_id
from tests.qdrant_fakes import FakeModels, FakeQdrantClient


def scope(document: str) -> dict[str, str]:
    return {
        "tenant_id": "default",
        "corpus_id": "default",
        "document_version_id": document,
        "index_revision_id": f"revision-{document}",
        "source_sha256": ("a" if document == "doc-1" else "b") * 64,
        "index_kind": "tree",
    }


def store(client: FakeQdrantClient, document: str) -> VectorStore:
    embedder = SimpleNamespace(
        MM_EMBEDDER=False,
        embed_texts=lambda texts: [
            [float(len(text)), float(text.count("a") + 1)] for text in texts
        ],
    )
    return VectorStore(
        embedder,
        collection_name="bookrag",
        scope_filter=scope(document),
        client=client,
        models=FakeModels,
    )


class QdrantConfigTests(unittest.TestCase):
    def test_qdrant_and_cross_tenant_defaults_are_explicit(self):
        config = VDBConfig()

        self.assertEqual("http://127.0.0.1:6333", config.qdrant_url)
        self.assertEqual("DOCARAG_QDRANT_URL", config.qdrant_url_env)
        self.assertEqual("QDRANT_API_KEY", config.api_key_env)
        self.assertEqual("default", config.tenant_id)
        self.assertEqual("default", config.corpus_id)
        self.assertEqual("./qdrant-attestation", config.manifest_dir_name)

    def test_config_rejects_credentials_in_url_and_invalid_names(self):
        with self.assertRaisesRegex(ValueError, "credentials"):
            VDBConfig(qdrant_url="http://user:secret@localhost:6333")
        with self.assertRaisesRegex(ValueError, "tenant_id"):
            VDBConfig(tenant_id="../escape")
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            VDBConfig(api_key_env="NOT VALID")
        with self.assertRaisesRegex(ValueError, "qdrant_url_env"):
            VDBConfig(qdrant_url_env="NOT VALID")
        with self.assertRaisesRegex(ValueError, "collection_name"):
            VDBConfig(collection_name=" padded ")

    def test_environment_endpoint_override_is_validated_and_materialized(self):
        with patch.dict(
            os.environ,
            {"TEST_QDRANT_URL": "https://qdrant.internal.example:7443/"},
            clear=False,
        ):
            config = VDBConfig(
                qdrant_url="http://configured.example:6333",
                qdrant_url_env="TEST_QDRANT_URL",
            )

        self.assertEqual(
            "https://qdrant.internal.example:7443", config.qdrant_url
        )
        with (
            patch.dict(
                os.environ,
                {"TEST_QDRANT_URL": "http://user:secret@qdrant:6333"},
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "credentials"),
        ):
            VDBConfig(qdrant_url_env="TEST_QDRANT_URL")

    def test_postgres_scope_environment_overrides_are_materialized(self):
        environment = {
            "DOCARAG_TENANT_ID": "tenant-a",
            "DOCARAG_CORPUS_ID": "corpus-a",
            "DOCARAG_DOCUMENT_ID": "doc-a",
            "DOCARAG_DOCUMENT_VERSION_ID": "docv-a",
            "DOCARAG_INDEX_REVISION_ID": "index-a",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = VDBConfig()

        self.assertEqual("tenant-a", config.tenant_id)
        self.assertEqual("corpus-a", config.corpus_id)
        self.assertEqual("doc-a", config.document_id)
        self.assertEqual("docv-a", config.document_version_id)
        self.assertEqual("index-a", config.index_revision_id)


class QdrantVectorStoreTests(unittest.TestCase):
    def test_direct_default_endpoint_honors_environment_and_explicit_wins(self):
        captured: list[str] = []

        class CapturingClient(FakeQdrantClient):
            def __init__(self, **kwargs):
                super().__init__()
                captured.append(kwargs["url"])

        with (
            patch(
                "retrieval_engine.indexes.vector_store._load_qdrant_components",
                return_value=(CapturingClient, FakeModels),
            ),
            patch.dict(
                os.environ,
                {"DOCARAG_QDRANT_URL": "https://runtime-qdrant:7443"},
                clear=False,
            ),
        ):
            inherited = VectorStore(SimpleNamespace(embed_texts=lambda texts: []))
            explicit = VectorStore(
                SimpleNamespace(embed_texts=lambda texts: []),
                qdrant_url="http://explicit-qdrant:6333",
            )
        inherited.close()
        explicit.close()
        self.assertEqual(
            ["https://runtime-qdrant:7443", "http://explicit-qdrant:6333"],
            captured,
        )

    def test_default_client_reads_only_the_named_api_key_environment_variable(self):
        captured: dict[str, object] = {}

        class CapturingClient(FakeQdrantClient):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        with (
            patch(
                "retrieval_engine.indexes.vector_store._load_qdrant_components",
                return_value=(CapturingClient, FakeModels),
            ),
            patch.dict(os.environ, {"TEST_QDRANT_KEY": "secret-value"}, clear=False),
        ):
            vector_store = VectorStore(
                SimpleNamespace(embed_texts=lambda texts: [[1.0] for _ in texts]),
                api_key_env="TEST_QDRANT_KEY",
            )

        self.assertEqual("http://127.0.0.1:6333", captured["url"])
        self.assertEqual("secret-value", captured["api_key"])
        self.assertEqual(10.0, captured["timeout"])
        client = vector_store.client
        vector_store.close()
        self.assertTrue(client.closed)

    def test_logical_ids_are_deterministic_and_physical_ids_are_uuid(self):
        client = FakeQdrantClient()
        vector_store = store(client, "doc-1")

        first = vector_store.add_texts(["alpha"], [{"node_id": 1}])
        second = vector_store.add_texts(["alpha"], [{"node_id": 1}])

        self.assertEqual(first, second)
        physical_id = qdrant_point_id(first[0])
        self.assertIn(physical_id, client.collections["bookrag"]["records"])
        result = vector_store.search("alpha", top_k=1)[0]
        self.assertEqual(first[0], result["id"])
        self.assertNotIn("_docarag_record_id", result["metadata"])
        self.assertEqual("alpha", result["content"])

    def test_scope_is_injected_and_conflicts_fail_before_embedding(self):
        client = FakeQdrantClient()
        vector_store = store(client, "doc-1")
        embedder = vector_store.embedding_model

        ids = vector_store.add_texts(["alpha"], [{"node_id": 1}])
        metadata = client.records[ids[0]]["metadata"]
        self.assertEqual("default", metadata["tenant_id"])
        self.assertEqual("doc-1", metadata["document_version_id"])

        original = embedder.embed_texts
        embedder.embed_texts = Mock(wraps=original)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            vector_store.add_texts(
                ["wrong"],
                [{"node_id": 2, "tenant_id": "another"}],
            )
        embedder.embed_texts.assert_not_called()

    def test_scoped_count_reset_delete_and_update_do_not_cross_documents(self):
        client = FakeQdrantClient()
        first = store(client, "doc-1")
        second = store(client, "doc-2")
        first_id = first.add_texts(["alpha"], [{"node_id": 1}])[0]
        second_id = second.add_texts(["beta"], [{"node_id": 2}])[0]

        second.delete_text_by_ids([first_id])
        second.update_text_metadatas([first_id], [{"description": "wrong"}])
        self.assertEqual(1, first.count())
        self.assertNotIn("description", client.records[first_id]["metadata"])

        first.reset()
        self.assertEqual(0, first.count())
        self.assertEqual(1, second.count())
        self.assertIn(second_id, client.records)

    def test_unscoped_reset_fails_closed(self):
        client = FakeQdrantClient()
        vector_store = VectorStore(
            SimpleNamespace(embed_texts=lambda texts: [[1.0] for _ in texts]),
            client=client,
            models=FakeModels,
        )
        vector_store.add_texts(["alpha"], [{"node_id": 1}])

        with self.assertRaisesRegex(RuntimeError, "unscoped"):
            vector_store.reset()

    def test_existing_collection_dimension_and_distance_are_validated(self):
        client = FakeQdrantClient()
        client.create_collection(
            collection_name="bookrag",
            vectors_config=FakeModels.VectorParams(size=3, distance="Cosine"),
        )
        vector_store = store(client, "doc-1")

        with self.assertRaisesRegex(ValueError, "dimension"):
            vector_store.add_texts(["alpha"], [{"node_id": 1}])

        client.collections["bookrag"]["vectors_config"] = FakeModels.VectorParams(
            size=2, distance="Dot"
        )
        with self.assertRaisesRegex(RuntimeError, "cosine"):
            vector_store.search("alpha")

    def test_batch_query_uses_one_qdrant_batch_request_and_source_filter(self):
        client = FakeQdrantClient()
        vector_store = store(client, "doc-1")
        vector_store.add_texts(
            ["alpha", "beta"],
            [{"node_id": 1}, {"node_id": 2}],
        )

        results = vector_store.batch_search(["alpha", "beta"], top_k=1)

        self.assertEqual([1, 1], [len(items) for items in results])
        self.assertEqual(1, len(client.batch_query_calls))
        request_filter = client.batch_query_calls[0]["requests"][0].filter
        conditions = {
            condition.key: condition.match.value
            for condition in request_filter.must
        }
        self.assertEqual(scope("doc-1"), conditions)

    def test_partial_scope_without_tenant_and_corpus_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tenant_id and corpus_id"):
            VectorStore(
                SimpleNamespace(embed_texts=lambda texts: [[1.0] for _ in texts]),
                scope_filter={"document_version_id": "doc-1"},
                client=FakeQdrantClient(),
                models=FakeModels,
            )


if __name__ == "__main__":
    unittest.main()
