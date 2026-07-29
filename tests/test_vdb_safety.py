from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from Core.provider.vdb import (
    MissingVectorSourceIdentityError,
    VectorStore,
    stable_vector_record_id,
)
from Core.provider.embedding import MultimodalEmbeddingProvider
from Core.rag.mm_rag import MMRAG, NO_USABLE_EVIDENCE_ANSWER
from tests.qdrant_fakes import FakeModels, FakeQdrantClient, qdrant_store


SOURCE_SHA256 = "a" * 64


def text_store() -> tuple[VectorStore, FakeQdrantClient, Mock]:
    embedder = Mock()
    embedder.embed_texts.side_effect = lambda texts: [[float(len(text))] for text in texts]
    store, client = qdrant_store(embedder)
    return store, client, embedder


class StableVectorIdentityTests(unittest.TestCase):
    def test_metadata_omits_nulls_and_rejects_nonfinite_values(self):
        store, collection, embedder = text_store()

        ids = store.add_texts(
            ["source text"],
            [{"optional": None, "quality": 0.75}],
        )

        self.assertNotIn("optional", collection.records[ids[0]]["metadata"])
        self.assertEqual(0.75, collection.records[ids[0]]["metadata"]["quality"])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            store.add_texts(["bad"], [{"quality": math.nan}])
        self.assertEqual(1, embedder.embed_texts.call_count)

    def test_two_semantic_elements_with_the_same_text_get_separate_records(self):
        base = {
            "document_version_id": "document-v1",
            "source_sha256": SOURCE_SHA256,
            "node_id": 17,
            "canonical_block_id": "block_" + "a" * 32,
        }
        value = "diagram | diagram_element | output | Speaker"

        first = stable_vector_record_id(
            value,
            {**base, "semantic_element_id": "selem_" + "a" * 32},
            modality="text",
            strict_source_identity=True,
        )
        second = stable_vector_record_id(
            value,
            {**base, "semantic_element_id": "selem_" + "b" * 32},
            modality="text",
            strict_source_identity=True,
        )

        self.assertNotEqual(
            first,
            second,
            "a diagram may draw the same label twice; each element is its own record",
        )

    def test_records_without_a_semantic_element_keep_their_identity(self):
        metadata = {
            "document_version_id": "document-v1",
            "source_sha256": SOURCE_SHA256,
            "node_id": 17,
        }

        self.assertEqual(
            stable_vector_record_id(
                "revenue table", metadata, modality="text", strict_source_identity=True
            ),
            stable_vector_record_id(
                "revenue table",
                {**metadata, "semantic_element_id": None},
                modality="text",
                strict_source_identity=True,
            ),
        )

    def test_strict_text_upsert_is_idempotent_and_source_content_bound(self):
        store, collection, _embedder = text_store()
        metadata = {
            "document_version_id": "document-v1",
            "source_sha256": SOURCE_SHA256,
            "node_id": 17,
            "canonical_revision_id": "block:r1",
            "node_type": "table",
        }

        first = store.add_texts(
            ["revenue table"],
            [metadata],
            strict_source_identity=True,
        )
        second = store.add_texts(
            ["revenue table"],
            [metadata],
            strict_source_identity=True,
        )

        self.assertEqual(first, second)
        self.assertTrue(first[0].startswith("record_text_"))
        self.assertEqual(1, len(collection.records))
        self.assertEqual(2, len(collection.upsert_calls))
        stored = collection.records[first[0]]["metadata"]
        self.assertEqual("text", stored["modality"])
        self.assertEqual("text", stored["type"])
        self.assertEqual("table", stored["node_type"])

        changed_content = stable_vector_record_id(
            "changed revenue table",
            metadata,
            modality="text",
            strict_source_identity=True,
        )
        changed_document = stable_vector_record_id(
            "revenue table",
            {**metadata, "document_version_id": "document-v2"},
            modality="text",
            strict_source_identity=True,
        )
        changed_source = stable_vector_record_id(
            "revenue table",
            {**metadata, "source_sha256": "b" * 64},
            modality="text",
            strict_source_identity=True,
        )
        self.assertNotEqual(first[0], changed_content)
        self.assertNotEqual(first[0], changed_document)
        self.assertNotEqual(first[0], changed_source)

    def test_strict_ids_do_not_collide_across_sources_with_same_local_ids(self):
        common = {
            "document_version_id": "document-v1",
            "node_id": 17,
            "canonical_revision_id": "block:r1",
            "node_type": "table",
        }

        first = stable_vector_record_id(
            "identical content",
            {**common, "source_sha256": "a" * 64},
            modality="text",
            strict_source_identity=True,
        )
        second = stable_vector_record_id(
            "identical content",
            {**common, "source_sha256": "b" * 64},
            modality="text",
            strict_source_identity=True,
        )

        self.assertNotEqual(first, second)

    def test_strict_ids_do_not_collide_across_tenants_or_index_revisions(self):
        common = {
            "tenant_id": "tenant-a",
            "corpus_id": "corpus-a",
            "document_version_id": "document-v1",
            "index_revision_id": "index-r1",
            "index_kind": "tree",
            "source_sha256": SOURCE_SHA256,
            "node_id": 17,
        }
        first = stable_vector_record_id(
            "identical content",
            common,
            modality="text",
            strict_source_identity=True,
        )
        another_tenant = stable_vector_record_id(
            "identical content",
            {**common, "tenant_id": "tenant-b"},
            modality="text",
            strict_source_identity=True,
        )
        another_revision = stable_vector_record_id(
            "identical content",
            {**common, "index_revision_id": "index-r2"},
            modality="text",
            strict_source_identity=True,
        )
        self.assertNotEqual(first, another_tenant)
        self.assertNotEqual(first, another_revision)

    def test_strict_identity_rejects_missing_source_before_embedding(self):
        store, collection, embedder = text_store()

        with self.assertRaisesRegex(
            MissingVectorSourceIdentityError, "document_version_id"
        ):
            store.add_texts(
                ["text"],
                [{"source_sha256": SOURCE_SHA256, "node_id": 1}],
                strict_source_identity=True,
            )
        with self.assertRaisesRegex(
            MissingVectorSourceIdentityError, "source_sha256"
        ):
            store.add_texts(
                ["text"],
                [{"document_version_id": "document-v1", "node_id": 1}],
                strict_source_identity=True,
            )
        with self.assertRaisesRegex(
            MissingVectorSourceIdentityError, "node or evidence"
        ):
            store.add_texts(
                ["text"],
                [
                    {
                        "document_version_id": "document-v1",
                        "source_sha256": SOURCE_SHA256,
                    }
                ],
                strict_source_identity=True,
            )

        for invalid_digest in ("A" * 64, f" {SOURCE_SHA256}", SOURCE_SHA256[:-1]):
            with self.subTest(source_sha256=invalid_digest):
                with self.assertRaisesRegex(
                    MissingVectorSourceIdentityError, "source_sha256"
                ):
                    store.add_texts(
                        ["text"],
                        [
                            {
                                "document_version_id": "document-v1",
                                "source_sha256": invalid_digest,
                                "node_id": 1,
                            }
                        ],
                        strict_source_identity=True,
                    )

        with self.assertRaisesRegex(
            MissingVectorSourceIdentityError, "surrounding whitespace"
        ):
            store.add_texts(
                ["text"],
                [
                    {
                        "document_version_id": " document-v1 ",
                        "source_sha256": SOURCE_SHA256,
                        "node_id": 1,
                    }
                ],
                strict_source_identity=True,
            )

        embedder.embed_texts.assert_not_called()
        self.assertEqual({}, collection.records)

    def test_generic_legacy_fallback_is_explicit_and_deterministic(self):
        store, collection, _embedder = text_store()

        first = store.add_texts(["entity"], [{"entity_id": 3}])
        second = store.add_texts(["entity"], [{"entity_id": 3}])

        self.assertEqual(first, second)
        self.assertTrue(first[0].startswith("legacy_text_"))
        self.assertEqual(1, len(collection.records))

        duplicate_ids = store.add_texts(
            ["duplicate", "duplicate"],
            [{"entity_id": 4}, {"entity_id": 4}],
        )
        self.assertEqual(duplicate_ids[0], duplicate_ids[1])
        self.assertEqual(1, len(collection.upsert_calls[-1]["ids"]))

    def test_same_strict_identity_with_inconsistent_metadata_fails_closed(self):
        store, collection, embedder = text_store()
        base = {
            "document_version_id": "document-v1",
            "source_sha256": SOURCE_SHA256,
            "node_id": 2,
            "node_type": "text",
        }

        with self.assertRaisesRegex(ValueError, "ID collision"):
            store.add_texts(
                ["same", "same"],
                [{**base, "page_idx": 1}, {**base, "page_idx": 2}],
                strict_source_identity=True,
            )

        embedder.embed_texts.assert_not_called()
        self.assertEqual({}, collection.records)

    def test_image_identity_hashes_bytes_and_fused_text_not_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.png"
            second_path = root / "second.png"
            first_path.write_bytes(b"same image bytes")
            second_path.write_bytes(b"same image bytes")
            metadata = {
                "document_version_id": "document-v1",
                "source_sha256": SOURCE_SHA256,
                "node_id": 9,
                "type": "equation",
            }

            first = stable_vector_record_id(
                str(first_path),
                metadata,
                modality="image",
                image_text="E=mc^2",
                strict_source_identity=True,
            )
            moved = stable_vector_record_id(
                str(second_path),
                metadata,
                modality="image",
                image_text="E=mc^2",
                strict_source_identity=True,
            )
            changed_fusion = stable_vector_record_id(
                str(second_path),
                metadata,
                modality="image",
                image_text="E=mc^3",
                strict_source_identity=True,
            )

            self.assertEqual(first, moved)
            self.assertNotEqual(first, changed_fusion)

    def test_image_upsert_preserves_visual_modality_and_semantic_node_type(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "table.png"
            image.write_bytes(b"table image")
            embedder = SimpleNamespace(
                MM_EMBEDDER=True,
                embed_fused=lambda images, texts: [[1.0, 2.0] for _ in images],
            )
            store, collection = qdrant_store(embedder)
            metadata = {
                "document_version_id": "document-v1",
                "source_sha256": SOURCE_SHA256,
                "node_id": 4,
                # Legacy semantic type is retained as node_type before `type`
                # becomes the MM-RAG representation discriminator.
                "type": "table",
            }

            ids = store.add_images(
                [str(image)],
                [metadata],
                ["rows and columns"],
                strict_source_identity=True,
            )

            stored = collection.records[ids[0]]["metadata"]
            self.assertEqual("image", stored["modality"])
            self.assertEqual("image", stored["type"])
            self.assertEqual("table", stored["node_type"])
            self.assertEqual(
                hashlib.sha256(b"table image").hexdigest(),
                stored["asset_sha256"],
            )


class VectorStoreIsolationAndFilterTests(unittest.TestCase):
    def test_store_close_releases_attached_read_lease_once(self):
        store = object.__new__(VectorStore)
        store.collection = object()
        store.client = object()
        store.embedding_model = object()
        store._read_lease = None
        lease = Mock()

        store.attach_read_lease(lease)
        store.close()
        store.close()

        lease.close.assert_called_once_with()
        self.assertIsNone(store.collection)
        self.assertIsNone(store.client)
        self.assertIsNone(store.embedding_model)

    def test_search_limits_fail_before_embedding_and_empty_batch_is_a_noop(self):
        store, collection, embedder = text_store()

        for invalid in (0, -1, True, 1.5):
            with self.subTest(top_k=invalid):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    store.search("question", top_k=invalid)
        self.assertEqual([], store.batch_search([], top_k=3))

        embedder.embed_texts.assert_not_called()
        self.assertEqual([], collection.query_calls)

    def test_reset_deletes_only_the_source_scope(self):
        client = FakeQdrantClient()
        embedder = SimpleNamespace(embed_texts=lambda texts: [[1.0] for _ in texts])
        first = VectorStore(
            embedder,
            collection_name="tree",
            scope_filter={
                "tenant_id": "default",
                "corpus_id": "default",
                "document_version_id": "doc-1",
                "index_kind": "tree",
            },
            client=client,
            models=FakeModels,
        )
        second = VectorStore(
            embedder,
            collection_name="tree",
            scope_filter={
                "tenant_id": "default",
                "corpus_id": "default",
                "document_version_id": "doc-2",
                "index_kind": "tree",
            },
            client=client,
            models=FakeModels,
        )
        first.add_texts(["first"], [{"node_id": 1}])
        second.add_texts(["second"], [{"node_id": 2}])

        first.reset()

        self.assertEqual(0, first.count())
        self.assertEqual(1, second.count())
        self.assertIn("tree", client.collections)

        unscoped = VectorStore(
            embedder,
            collection_name="tree",
            client=client,
            models=FakeModels,
        )
        with self.assertRaisesRegex(RuntimeError, "unscoped"):
            unscoped.reset()

    def test_search_and_batch_search_compose_metadata_and_document_filters(self):
        store, collection, _embedder = text_store()
        store.add_texts(
            ["formula"],
            [{"node_type": "table", "document_version_id": "document-v1"}],
        )
        supplied = {"node_type": "table"}

        matches = store.search(
            "formula",
            top_k=7,
            metadata_filter=supplied,
            document_version_id="document-v1",
        )
        self.assertEqual(1, len(matches))
        store.batch_search(
            ["formula", "table"],
            metadata_filter={"modality": "image"},
            document_version_id="document-v1",
        )
        store.search("unfiltered")

        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            store.search("padded", document_version_id=" document-v1 ")

        first_conditions = collection.query_calls[0]["query_filter"].must
        self.assertEqual(
            {"document_version_id": "document-v1", "node_type": "table"},
            {condition.key: condition.match.value for condition in first_conditions},
        )
        self.assertEqual(7, collection.query_calls[0]["limit"])
        batch_request = collection.batch_query_calls[0]["requests"][0]
        self.assertEqual(
            {"document_version_id": "document-v1", "modality": "image"},
            {
                condition.key: condition.match.value
                for condition in batch_request.filter.must
            },
        )
        self.assertIsNone(collection.query_calls[-1]["query_filter"])
        self.assertEqual({"node_type": "table"}, supplied)


class MultimodalRetrievalMetadataTests(unittest.TestCase):
    def test_changed_image_bytes_are_not_sent_to_vlm_context(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "figure.png"
            image.write_bytes(b"original bytes")
            expected = hashlib.sha256(b"original bytes").hexdigest()
            document = {
                "id": "record_image_figure",
                "content": str(image),
                "metadata": {
                    "node_id": 3,
                    "modality": "image",
                    "asset_sha256": expected,
                },
            }
            image.write_bytes(b"changed bytes")
            rag = object.__new__(MMRAG)

            prompt, images = rag._create_augmented_prompt("question", [document])

            self.assertEqual([], images)
            self.assertNotIn(str(image), prompt)

    def test_changed_only_image_does_not_call_models_or_report_used_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "figure.png"
            image.write_bytes(b"changed bytes")
            document = {
                "id": "record_image_figure",
                "content": str(image),
                "metadata": {
                    "node_id": 3,
                    "modality": "image",
                    "asset_sha256": hashlib.sha256(b"original bytes").hexdigest(),
                },
            }
            rag = object.__new__(MMRAG)
            rag.topk = 3
            rag.vdb = SimpleNamespace(search=Mock(return_value=[document]))
            rag.llm = Mock()
            rag.vlm = Mock()

            answer, retrieved_ids = rag.generation("question", root / "audit")

            self.assertEqual(NO_USABLE_EVIDENCE_ANSWER, answer)
            self.assertEqual([], retrieved_ids)
            rag.llm.get_completion.assert_not_called()
            rag.vlm.generate.assert_not_called()
            self.assertFalse((root / "audit").exists())

    def test_prompt_audit_and_ids_include_only_images_actually_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for node_id in range(1, 4):
                image = root / f"image-{node_id}.png"
                image.write_bytes(f"image-{node_id}".encode("utf-8"))
                records.append(
                    {
                        "id": f"record_image_{node_id}",
                        "content": str(image),
                        "metadata": {"node_id": node_id, "modality": "image"},
                    }
                )
            records.append(
                {
                    "id": "record_text_4",
                    "content": "supporting text",
                    "metadata": {"node_id": 4, "modality": "text"},
                }
            )
            rag = object.__new__(MMRAG)
            rag.topk = 4
            rag.vdb = SimpleNamespace(search=Mock(return_value=records))
            rag.llm = Mock()
            rag.vlm = Mock()
            rag.vlm.generate.return_value = "answer"

            answer, retrieved_ids = rag.generation("question", root / "audit")

            self.assertEqual("answer", answer)
            self.assertEqual([1, 2, 4], retrieved_ids)
            call = rag.vlm.generate.call_args
            self.assertEqual(
                [str(root / "image-1.png"), str(root / "image-2.png")],
                call.kwargs["images"],
            )
            self.assertNotIn(
                str(root / "image-3.png"), call.kwargs["prompt_or_memory"]
            )
            saved = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (root / "audit").glob("retrieval-*.json")
            ]
            self.assertEqual(3, len(saved))
            self.assertEqual(
                {"record_image_1", "record_image_2", "record_text_4"},
                {item["vector_record_id"] for item in saved},
            )

    def test_close_unloads_owned_mm_provider_once(self):
        provider = object.__new__(MultimodalEmbeddingProvider)
        provider.close = Mock()
        provider.clear_cache = Mock()
        rag = object.__new__(MMRAG)
        rag.vdb = SimpleNamespace(embedding_model=provider)

        rag.close()
        rag.close()

        provider.close.assert_called_once_with()
        provider.clear_cache.assert_not_called()
        self.assertIsNone(rag.vdb)

    def test_table_and_equation_visual_records_reach_vlm_and_keep_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "table.png"
            equation = root / "equation.png"
            table.write_bytes(b"table")
            equation.write_bytes(b"equation")
            documents = [
                {
                    "content": str(table),
                    "id": "record_image_table",
                    "metadata": {
                        "node_id": 4,
                        "modality": "image",
                        "type": "text",  # explicit modality must win
                        "node_type": "table",
                        "document_version_id": "document-v1",
                        "canonical_revision_id": "table:r1",
                    },
                },
                {
                    "content": str(equation),
                    "id": "record_image_equation",
                    "metadata": {
                        "node_id": 5,
                        "modality": "image",
                        "type": "image",
                        "node_type": "equation",
                        "document_version_id": "document-v1",
                        "canonical_revision_id": "equation:r1",
                    },
                },
            ]
            rag = object.__new__(MMRAG)
            rag.topk = 3
            rag.vdb = SimpleNamespace(search=Mock(return_value=documents))
            rag.llm = Mock()
            rag.vlm = Mock()
            rag.vlm.generate.return_value = "visual answer"

            answer, retrieved_ids = rag.generation("What does this show?", root)

            self.assertEqual("visual answer", answer)
            self.assertEqual([4, 5], retrieved_ids)
            rag.vlm.generate.assert_called_once()
            self.assertEqual(
                [str(table), str(equation)],
                rag.vlm.generate.call_args.kwargs["images"],
            )
            rag.llm.get_completion.assert_not_called()
            saved_files = sorted(root.glob("retrieval-*.json"))
            self.assertEqual(2, len(saved_files))
            saved_payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in saved_files
            ]
            saved = next(item for item in saved_payloads if item["node_id"] == 4)
            self.assertEqual("image", saved["modality"])
            self.assertEqual("table", saved["node_type"])
            self.assertEqual("document-v1", saved["document_version_id"])
            self.assertEqual("table:r1", saved["canonical_revision_id"])
            self.assertEqual("record_image_table", saved["vector_record_id"])

    def test_same_node_modalities_do_not_overwrite_audit_or_duplicate_node_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "table.png"
            image.write_bytes(b"table")
            rag = object.__new__(MMRAG)
            records = [
                {
                    "id": "record_text_table",
                    "content": "structured table",
                    "metadata": {
                        "node_id": 7,
                        "modality": "text",
                        "node_type": "table",
                    },
                },
                {
                    "id": "record_image_table",
                    "content": str(image),
                    "metadata": {
                        "node_id": 7,
                        "modality": "image",
                        "node_type": "table",
                    },
                },
            ]

            retrieval_ids = rag._save_retrieval_res(records, root)

            self.assertEqual([7], retrieval_ids)
            payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in root.glob("retrieval-*.json")
            ]
            self.assertEqual(2, len(payloads))
            self.assertEqual(
                {"record_text_table", "record_image_table"},
                {item["vector_record_id"] for item in payloads},
            )

    def test_mm_retrieve_forwards_source_filter(self):
        rag = object.__new__(MMRAG)
        rag.vdb = SimpleNamespace(search=Mock(return_value=[]))

        rag._retrieve(
            "question",
            top_k=9,
            metadata_filter={"node_type": "table"},
            document_version_id="document-v1",
        )

        rag.vdb.search.assert_called_once_with(
            query_text="question",
            top_k=9,
            metadata_filter={"node_type": "table"},
            document_version_id="document-v1",
        )


if __name__ == "__main__":
    unittest.main()
