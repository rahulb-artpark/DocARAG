"""Opt-in integration test for the real Qdrant service.

Set ``DOCARAG_TEST_QDRANT_URL`` to run this test.  The test creates an isolated
collection with a random name and always removes it afterward.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
import uuid

from Core.provider.vdb import VectorStore


QDRANT_URL = os.environ.get("DOCARAG_TEST_QDRANT_URL")


@unittest.skipUnless(QDRANT_URL, "set DOCARAG_TEST_QDRANT_URL for live Qdrant")
class LiveQdrantVectorStoreTests(unittest.TestCase):
    def test_scoped_round_trip_and_cleanup(self):
        from qdrant_client import QdrantClient, models

        collection_name = f"docarag_live_{uuid.uuid4().hex}"
        client = QdrantClient(url=QDRANT_URL, timeout=10.0)
        store = VectorStore(
            SimpleNamespace(
                MM_EMBEDDER=False,
                embed_texts=lambda texts: [
                    [float(len(text)), float(text.lower().count("a") + 1)]
                    for text in texts
                ],
            ),
            qdrant_url=QDRANT_URL,
            collection_name=collection_name,
            scope_filter={
                "tenant_id": "integration-test",
                "corpus_id": "integration-test",
                "document_version_id": "document-v1",
                "index_revision_id": "index-r1",
                "source_sha256": "a" * 64,
                "index_kind": "tree",
            },
            client=client,
            models=models,
        )
        try:
            record_ids = store.add_texts(
                ["alpha document evidence"],
                [{"node_id": "node-1", "node_type": "text"}],
                strict_source_identity=True,
            )

            self.assertEqual(1, store.count())
            results = store.search("alpha evidence", top_k=1)
            self.assertEqual(record_ids[0], results[0]["id"])
            self.assertEqual("alpha document evidence", results[0]["content"])
            self.assertEqual("document-v1", results[0]["metadata"]["document_version_id"])

            store.reset()
            self.assertEqual(0, store.count())
        finally:
            store.close()
            if client.collection_exists(collection_name=collection_name):
                client.delete_collection(collection_name=collection_name)
            client.close()


if __name__ == "__main__":
    unittest.main()
