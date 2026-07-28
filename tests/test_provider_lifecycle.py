from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Core.Index.GBCIndex import GBC
from Core.Index.Graph import Entity
from Core.Index.Tree import NodeType
from Core.configs.graph_config import GraphConfig
from Core.configs.vdb_config import VDBConfig
from Core.pipelines import kg_builder
from Core.pipelines.kg_extractor import KGExtractionError
from Core.pipelines.kg_refiner import KGRefiner
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore
from tests.qdrant_fakes import qdrant_store


class TextEmbeddingLifecycleTests(unittest.TestCase):
    def test_local_close_releases_weights_and_is_idempotent(self):
        provider = object.__new__(TextEmbeddingProvider)
        provider._closed = False
        provider.backend = "local"
        provider.device = "cpu"
        provider.model_name = "test-model"
        provider.model = object()
        provider.tokenizer = object()

        provider.close()
        provider.close()

        self.assertTrue(provider._closed)
        self.assertFalse(hasattr(provider, "model"))
        self.assertFalse(hasattr(provider, "tokenizer"))

    def test_remote_close_closes_client_once(self):
        provider = object.__new__(TextEmbeddingProvider)
        provider._closed = False
        provider.backend = "openai"
        provider.model_name = "remote-model"
        provider.client = Mock()
        client = provider.client

        provider.close()
        provider.close()

        client.close.assert_called_once_with()
        self.assertIsNone(provider.client)


class VectorStoreLifecycleTests(unittest.TestCase):
    def test_text_embedding_and_writes_are_chunked(self):
        embedder = Mock()
        embedder.embed_texts.side_effect = (
            lambda texts: [[float(index)] for index, _text in enumerate(texts)]
        )
        store, client = qdrant_store(embedder)
        texts = [f"text-{index}" for index in range(5)]
        metadatas = [{"index": index} for index in range(5)]

        ids = store.add_texts(texts, metadatas, batch_size=2)

        self.assertEqual(
            [call.args[0] for call in embedder.embed_texts.call_args_list],
            [texts[:2], texts[2:4], texts[4:]],
        )
        self.assertEqual(len(client.upsert_calls), 3)
        self.assertTrue(
            all(
                metadata["modality"] == "text" and metadata["type"] == "text"
                for call in client.upsert_calls
                for metadata in call["metadatas"]
            )
        )
        self.assertEqual(len(ids), 5)
        with self.assertRaises(ValueError):
            store.add_texts(["one"], [], batch_size=2)

    def test_metadata_update_does_not_invoke_embedding_provider(self):
        embedder = Mock()
        embedder.embed_texts.return_value = [[1.0]]
        store, client = qdrant_store(embedder)
        ids = store.add_texts(["entity"], [{"description": "old"}])
        embedder.reset_mock()

        store.update_text_metadatas(ids, [{"description": "updated"}])

        self.assertEqual("updated", client.set_payload_calls[0]["payload"]["description"])
        embedder.embed_texts.assert_not_called()
        with self.assertRaises(ValueError):
            store.update_text_metadatas(["one"], [])

    def test_close_breaks_provider_and_qdrant_references(self):
        store = object.__new__(VectorStore)
        store.collection = object()
        store.client = object()
        store.embedding_model = object()

        store.close()

        self.assertIsNone(store.collection)
        self.assertIsNone(store.client)
        self.assertIsNone(store.embedding_model)


class KGResourceLifecycleTests(unittest.TestCase):
    def test_constructor_failure_closes_already_created_provider(self):
        embedder = Mock()
        config = GraphConfig()

        with (
            patch(
                "Core.pipelines.kg_refiner.TextEmbeddingProvider",
                return_value=embedder,
            ),
            patch(
                "Core.pipelines.kg_refiner.TextRerankerProvider",
                side_effect=RuntimeError("reranker failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "reranker failed"),
        ):
            KGRefiner(
                llm=object(),
                graph_config=config,
                graph_index=object(),
                save_path="/tmp/not-created",
                vdb_config=VDBConfig(),
                scope_filter={
                    "tenant_id": "test",
                    "corpus_id": "test",
                    "document_version_id": "document-v1",
                    "source_sha256": "a" * 64,
                    "index_kind": "entity",
                },
            )

        embedder.close.assert_called_once_with()

    def test_close_continues_after_one_resource_close_fails(self):
        refiner = object.__new__(KGRefiner)
        refiner.reranker = Mock()
        refiner.reranker.close.side_effect = OSError("close failed")
        vdb = Mock()
        embedder = Mock()
        refiner.vdb = vdb
        refiner.embedder = embedder
        refiner.entity_to_vdb_id = {"entity": "id"}
        refiner.entity_alias_map = {"old": "new"}

        refiner.close()

        vdb.close.assert_called_once_with()
        embedder.close.assert_called_once_with()
        self.assertIsNone(refiner.reranker)
        self.assertIsNone(refiner.vdb)
        self.assertIsNone(refiner.embedder)
        self.assertEqual(refiner.entity_to_vdb_id, {})
        self.assertEqual(refiner.entity_alias_map, {})

    def test_builder_closes_refiner_when_extraction_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = SimpleNamespace(index_id=0, type=NodeType.ROOT)
            node = SimpleNamespace(index_id=1, type=NodeType.TEXT)
            tree = SimpleNamespace(root_node=root, nodes=[root, node])
            config = SimpleNamespace(
                save_path=directory,
                graph=GraphConfig(extraction_batch_size=1),
                vdb=VDBConfig(),
                llm=object(),
                vlm=object(),
            )
            extractor = Mock()
            extractor.iter_extract_kg.side_effect = KGExtractionError(
                1, "text", RuntimeError("model failed")
            )
            refiner = Mock()

            with (
                patch.object(kg_builder, "LLM", return_value=object()),
                patch.object(kg_builder, "KGExtractor", return_value=extractor),
                patch.object(kg_builder, "KGRefiner", return_value=refiner),
                patch.object(
                    kg_builder,
                    "tree_vector_scope",
                    return_value={
                        "tenant_id": "test",
                        "corpus_id": "test",
                        "document_version_id": "document-v1",
                        "source_sha256": "a" * 64,
                        "index_kind": "tree",
                    },
                ),
                self.assertRaises(KGExtractionError),
            ):
                kg_builder.build_knowledge_graph(tree, config)

            refiner.close.assert_called_once_with()


class GBCResourceLifecycleTests(unittest.TestCase):
    def test_rebuild_streams_graph_nodes_in_bounded_batches(self):
        entities = {
            f"node-{index}": Entity(
                entity_name=f"entity-{index}",
                entity_type="TEST",
                description=f"description-{index}",
            )
            for index in range(5)
        }
        with tempfile.TemporaryDirectory() as directory:
            gbc = object.__new__(GBC)
            gbc.GraphIndex = SimpleNamespace(
                kg=SimpleNamespace(nodes=entities.keys()),
                get_entity_by_node_name=lambda name: entities[name],
            )
            gbc.entity_scope = {
                "tenant_id": "test",
                "corpus_id": "test",
                "document_version_id": "document-v1",
                "source_sha256": "a" * 64,
                "index_kind": "entity",
            }
            gbc.entity_attestation_path = str(Path(directory) / "entity-attestation")
            gbc.entity_vdb = Mock()
            gbc.entity_vdb.count.return_value = 5

            with (
                patch.object(
                    GBC,
                    "_expected_entity_attestation",
                    return_value={"point_count": 5},
                ),
                patch(
                    "Core.Index.GBCIndex.vector_index_lock",
                    return_value=nullcontext(),
                ),
            ):
                gbc.rebuild_vdb(batch_size=2)

            gbc.entity_vdb.reset.assert_called_once_with()
            # Entity points now, each carrying the three vectors the fused
            # query reads; the bound being tested is unchanged.
            batch_lengths = [
                len(call.args[0])
                for call in gbc.entity_vdb.add_entity_points.call_args_list
            ]
            self.assertEqual(batch_lengths, [2, 2, 1])

    def test_close_breaks_store_reference_before_closing_embedder(self):
        gbc = object.__new__(GBC)
        store = Mock()
        embedder = Mock()
        gbc.entity_vdb = store
        gbc.embedder = embedder

        gbc.close()
        gbc.close()

        store.close.assert_called_once_with()
        embedder.close.assert_called_once_with()
        self.assertIsNone(gbc.entity_vdb)
        self.assertIsNone(gbc.embedder)


if __name__ == "__main__":
    unittest.main()
