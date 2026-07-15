from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.configs.vdb_config import VDBConfig
from Core.pipelines import vdb_index
from Core.provider.vdb import VectorStore
from tests.qdrant_fakes import qdrant_store


class StreamingTreeVectorIndexTests(unittest.TestCase):
    def test_vdb_batch_size_is_validated_in_configuration(self):
        self.assertEqual(VDBConfig(index_batch_size=17).index_batch_size, 17)
        for value in (0, -1, True, "17"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "index_batch_size"
            ):
                VDBConfig(index_batch_size=value)

    def _tree(self, directory: str) -> tuple[DocumentTree, list[str]]:
        tree = DocumentTree(
            meta_dict={
                "file_name": "source.pdf",
                "source_sha256": "a" * 64,
                "document_version_id": "document-version-1",
            },
            cfg=SimpleNamespace(save_path=directory),
        )
        image_paths = []

        def add_node(node_type: NodeType, **metadata):
            node = TreeNode(metadata)
            node.type = node_type
            tree.add_node(node)
            tree.root_node.add_child(node)
            return node

        add_node(
            NodeType.TEXT,
            pdf_id=11,
            page_idx=0,
            page_id="page-0",
            canonical_block_id="canonical-text",
            canonical_revision_id="canonical-text:r1",
            observation_id="observation-text",
            content="alpha",
            parser_provenance={"nested": "must not reach vector payloads"},
        )

        table_path = str(Path(directory) / "table.png")
        Path(table_path).write_bytes(b"table")
        image_paths.append(table_path)
        add_node(
            NodeType.TABLE,
            pdf_id=12,
            page_idx=1,
            page_id="page-1",
            canonical_block_id="stale-direct-id",
            derived_evidence_id="derived-table",
            source_canonical_block_ids=["canonical-table-a", "canonical-table-b"],
            content="table heading",
            table_body="|a|b|",
            img_path=table_path,
        )

        figure_path = str(Path(directory) / "figure.png")
        Path(figure_path).write_bytes(b"figure")
        image_paths.append(figure_path)
        add_node(
            NodeType.IMAGE,
            pdf_id=13,
            page_idx=2,
            page_id="page-2",
            canonical_block_id="canonical-image",
            caption="figure caption",
            footnote="figure footnote",
            img_path=figure_path,
        )

        equation_path = str(Path(directory) / "equation.png")
        Path(equation_path).write_bytes(b"equation")
        image_paths.append(equation_path)
        add_node(
            NodeType.EQUATION,
            pdf_id=14,
            page_idx=3,
            page_id="page-3",
            canonical_block_id="canonical-equation",
            content=r"E=mc^2",
            img_path=equation_path,
        )

        add_node(
            NodeType.TITLE,
            pdf_id=15,
            page_idx=4,
            page_id="page-4",
            canonical_block_id="canonical-title",
            content="Conclusion",
        )
        return tree, image_paths

    @staticmethod
    def _config(directory: str, *, multimodal: bool = True):
        return SimpleNamespace(
            mm_embedding=multimodal,
            embedding_config=SimpleNamespace(
                model_name="test-embedding",
                device="cpu",
                backend="local",
                api_base=None,
                max_length=128,
            ),
            tenant_id="default",
            corpus_id="default",
            qdrant_url="http://127.0.0.1:6333",
            api_key_env="QDRANT_API_KEY",
            timeout_seconds=10.0,
            manifest_dir_name=str(Path(directory) / "vector_attestation"),
            collection_name="tree",
        )

    def test_build_streams_bounded_batches_with_complete_stable_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, expected_images = self._tree(directory)
            embedder = Mock()
            store = Mock()
            store.count.return_value = 8

            with (
                patch.object(vdb_index, "GmeEmbeddingProvider", return_value=embedder),
                patch.object(vdb_index, "VectorStore", return_value=store),
            ):
                vdb_index.build_vdb_index(
                    tree, self._config(directory), batch_size=2
                )

            text_batches = [
                call.kwargs["texts"] for call in store.add_texts.call_args_list
            ]
            image_batches = [
                call.kwargs["image_paths"]
                for call in store.add_images.call_args_list
            ]
            self.assertTrue(all(len(batch) <= 2 for batch in text_batches))
            self.assertTrue(all(len(batch) <= 2 for batch in image_batches))
            self.assertEqual(
                [
                    "alpha",
                    "table heading|a|b|",
                    "figure captionfigure footnote",
                    r"E=mc^2",
                    "Conclusion",
                ],
                [item for batch in text_batches for item in batch],
            )
            self.assertEqual(
                expected_images, [item for batch in image_batches for item in batch]
            )
            self.assertEqual(
                ["table heading|a|b|", "figure captionfigure footnote", r"E=mc^2"],
                [
                    item
                    for call in store.add_images.call_args_list
                    for item in call.kwargs["image_str"]
                ],
            )
            store.close.assert_called_once_with()
            store.reset.assert_called_once_with()
            embedder.close.assert_called_once_with()

    def test_metadata_is_scalar_and_binds_canonical_or_derived_source(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = self._tree(directory)
            records = list(vdb_index.iter_tree_index_records(tree))

            text = next(record for record in records if record.value == "alpha")
            self.assertEqual(
                {
                    "node_id": 1,
                    "node_type": "text",
                    "index_kind": "tree",
                    "pdf_id": 11,
                    "document_version_id": "document-version-1",
                    "page_id": "page-0",
                    "page_idx": 0,
                    "source_sha256": "a" * 64,
                    "canonical_block_id": "canonical-text",
                    "canonical_revision_id": "canonical-text:r1",
                    "observation_id": "observation-text",
                    "modality": "text",
                    "type": "text",
                },
                text.metadata,
            )
            table_records = [
                record
                for record in records
                if record.metadata.get("pdf_id") == 12
            ]
            self.assertEqual(2, len(table_records))
            for record in table_records:
                self.assertEqual("derived-table", record.metadata["derived_evidence_id"])
                self.assertNotIn("canonical_block_id", record.metadata)
                self.assertTrue(
                    all(
                        isinstance(value, (str, int, float, bool))
                        for value in record.metadata.values()
                    )
                )
                self.assertNotIn("source_canonical_block_ids", record.metadata)

    def test_failure_still_closes_store_and_embedding_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = self._tree(directory)
            embedder = Mock()
            store = Mock()
            store.add_texts.side_effect = RuntimeError("write failed")

            with (
                patch.object(vdb_index, "GmeEmbeddingProvider", return_value=embedder),
                patch.object(vdb_index, "VectorStore", return_value=store),
                self.assertRaisesRegex(RuntimeError, "write failed"),
            ):
                vdb_index.build_vdb_index(
                    tree, self._config(directory), batch_size=2
                )

            store.close.assert_called_once_with()
            embedder.close.assert_called_once_with()

    def test_store_constructor_failure_closes_embedding_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = self._tree(directory)
            embedder = Mock()

            with (
                patch.object(vdb_index, "GmeEmbeddingProvider", return_value=embedder),
                patch.object(
                    vdb_index, "VectorStore", side_effect=RuntimeError("open failed")
                ),
                self.assertRaisesRegex(RuntimeError, "open failed"),
            ):
                vdb_index.build_vdb_index(tree, self._config(directory))

            embedder.close.assert_called_once_with()


class BatchedImageVectorStoreTests(unittest.TestCase):
    def _store(self):
        embedder = SimpleNamespace(
            MM_EMBEDDER=True,
            embed_fused=Mock(side_effect=lambda images, texts: [[1.0] for _ in images]),
        )
        store, client = qdrant_store(embedder)
        store._test_client = client
        return store

    def test_image_embeddings_and_writes_are_batched_in_order(self):
        store = self._store()
        paths = [f"image-{index}.png" for index in range(5)]
        texts = [f"caption-{index}" for index in range(5)]
        metadata = [{"node_id": index} for index in range(5)]

        ids = store.add_images(
            paths, metadata, texts, batch_size=2
        )

        self.assertEqual(
            [paths[:2], paths[2:4], paths[4:]],
            [
                call.kwargs["images"]
                for call in store.embedding_model.embed_fused.call_args_list
            ],
        )
        self.assertEqual(
            [texts[:2], texts[2:4], texts[4:]],
            [
                call.kwargs["texts"]
                for call in store.embedding_model.embed_fused.call_args_list
            ],
        )
        self.assertEqual(
            paths,
            [
                item
                for call in store._test_client.upsert_calls
                for item in call["documents"]
            ],
        )
        self.assertEqual(5, len(ids))

    def test_image_input_and_embedding_counts_are_validated(self):
        store = self._store()

        with self.assertRaisesRegex(ValueError, "batch_size"):
            store.add_images(["one.png"], batch_size=0)
        with self.assertRaisesRegex(ValueError, "metadatas"):
            store.add_images(["one.png"], [], ["one"])
        with self.assertRaisesRegex(ValueError, "image_str"):
            store.add_images(["one.png"], [{"node_id": 1}], [])

        store.embedding_model.embed_fused.side_effect = lambda images, texts: []
        with self.assertRaisesRegex(ValueError, "0 embeddings for 1 inputs"):
            store.add_images(
                ["one.png"], [{"node_id": 1}], ["one"], batch_size=1
            )
        self.assertEqual([], store._test_client.upsert_calls)


class BatchedTextVectorStoreTests(unittest.TestCase):
    def test_text_embedding_count_is_validated_before_write(self):
        store, client = qdrant_store(
            SimpleNamespace(embed_texts=lambda _texts: [])
        )

        with self.assertRaisesRegex(ValueError, "0 embeddings for 1 inputs"):
            store.add_texts(["one"], [{"node_id": 1}], batch_size=1)

        self.assertEqual([], client.upsert_calls)


if __name__ == "__main__":
    unittest.main()
