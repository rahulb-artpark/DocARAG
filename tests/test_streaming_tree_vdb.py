from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from retrieval_engine.indexes.tree import DocumentTree, NodeType, TreeNode
from retrieval_engine.configuration.vector_store import VDBConfig
from retrieval_engine.indexing.vector import builder as vdb_index
from retrieval_engine.indexes.vector_store import VectorStore
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
            # Five content nodes, five points -- carrying the nine vectors the
            # index used to spread across nine points.
            store.count.return_value = 5

            with (
                patch.object(vdb_index, "MultimodalEmbeddingProvider", return_value=embedder),
                patch.object(vdb_index, "VectorStore", return_value=store),
            ):
                vdb_index.build_vdb_index(
                    tree, self._config(directory), batch_size=2
                )

            batches = [
                call.kwargs["records"]
                for call in store.add_tree_node_points.call_args_list
            ]
            self.assertTrue(all(len(batch) <= 2 for batch in batches))
            points = [record for batch in batches for record in batch]

            self.assertEqual(
                [
                    {"text": "alpha"},
                    {"text": "table heading\n|a|b|", "image": expected_images[0]},
                    {
                        "text": "figure caption figure footnote",
                        "image": expected_images[1],
                    },
                    {
                        "text": r"E=mc^2",
                        # The neighbouring figure contributes its caption, which
                        # is where a figure's prose lives; content alone would
                        # lose it.
                        "context": "figure caption figure footnote\nE=mc^2\nConclusion",
                        "image": expected_images[2],
                    },
                    {"text": "Conclusion"},
                ],
                [record["vectors"] for record in points],
            )
            self.assertEqual(9, sum(len(r["vectors"]) for r in points))
            self.assertEqual(
                [
                    None,
                    "table heading\n|a|b|",
                    # A figure's crop carries no text: the caption is already in
                    # its text vector, and fusing prose into a picture displaces
                    # the one signal only the picture has.
                    None,
                    # A table and an equation still fuse, because for them the
                    # crop is secondary to the text it depicts.
                    "figure caption figure footnote\nE=mc^2\nConclusion",
                    None,
                ],
                [record["image_text"] for record in points],
            )
            # Every point that holds text also holds terms for the lexical leg.
            self.assertTrue(all(record["lexical_terms"] for record in points))
            store.close.assert_called_once_with()
            store.reset.assert_called_once_with()
            embedder.close.assert_called_once_with()

    def _equation_in_prose(self, directory: str, *, summary: str = ""):
        """A displayed equation with the sentences that explain it around it."""

        tree = DocumentTree(
            meta_dict={
                "file_name": "book.pdf",
                "source_sha256": "b" * 64,
                "document_version_id": "document-version-2",
            },
            cfg=SimpleNamespace(save_path=directory),
        )
        section = TreeNode({"pdf_id": 1, "page_idx": 0, "content": "Probability Laws"})
        section.type = NodeType.TITLE
        tree.add_node(section)
        tree.root_node.add_child(section)

        def child(node_type, **meta):
            node = TreeNode(meta)
            node.type = node_type
            tree.add_node(node)
            section.add_child(node)
            return node

        child(NodeType.TEXT, pdf_id=2, page_idx=0, content="Far earlier sentence.")
        # The store keeps a figure's caption in its own field and leaves
        # content as the vision model's account of the pixels, so a neighbour
        # that reads content alone loses the half naming the subject.
        child(
            NodeType.IMAGE,
            pdf_id=3,
            page_idx=0,
            caption="Figure 1.5: the event M that Romeo and Juliet arrive within 15 minutes.",
            content="A 2D plot with a shaded band around the diagonal.",
        )
        equation = child(
            NodeType.EQUATION,
            pdf_id=4,
            page_idx=0,
            content="M = {(x,y) | |x-y| <= 1/4}",
        )
        equation.summary = summary
        child(NodeType.TEXT, pdf_id=5, page_idx=0, content="and is shaded in the figure.")
        child(NodeType.TEXT, pdf_id=6, page_idx=0, content="Unrelated later sentence.")
        return tree

    def test_an_equation_is_indexed_with_the_prose_that_explains_it(self):
        # A figure carries an author-written caption; a displayed equation
        # carries none, and its explanation is in the sentences either side --
        # which is why the following one can begin mid-clause.
        with tempfile.TemporaryDirectory() as directory:
            tree = self._equation_in_prose(directory)

            texts = [
                record.value
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "text"
            ]

            self.assertIn(
                "Figure 1.5: the event M that Romeo and Juliet arrive within 15 minutes. "
                "A 2D plot with a shaded band around the diagonal.\n"
                "M = {(x,y) | |x-y| <= 1/4}\n"
                "and is shaded in the figure.",
                texts,
            )

    def test_a_caption_already_inside_the_content_is_not_repeated(self):
        # Some producers fold the caption into content; joining both would
        # feed the embedder the same sentence twice.
        with tempfile.TemporaryDirectory() as directory:
            tree = self._equation_in_prose(directory)
            figure = tree.root_node.children[0].children[1]
            figure.meta_info.content = (
                "Figure 1.5: the event M that Romeo and Juliet arrive within 15 "
                "minutes. A 2D plot."
            )

            context = [
                record.value
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "text" and "Romeo" in record.value
                and "shaded in the figure" in record.value
            ]

            self.assertEqual(1, len(context), context)
            self.assertEqual(1, context[0].count("Figure 1.5:"))

    def test_the_equation_context_window_stops_at_one_sibling_each_side(self):
        # Wider windows measured worse on every equation and question tried.
        with tempfile.TemporaryDirectory() as directory:
            tree = self._equation_in_prose(directory)

            context = [
                record.value
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "text"
                and record.value.startswith("Figure 1.5:")
                and "shaded" in record.value
                # The figure now carries its own text record, which opens with
                # the same caption; the equation's LaTeX is what tells the
                # context window apart from the figure it starts with.
                and "M = {(x,y)" in record.value
            ]

            self.assertEqual(1, len(context), context)
            self.assertNotIn("Far earlier sentence.", context[0])
            self.assertNotIn("Unrelated later sentence.", context[0])

    def test_an_equation_summary_is_indexed_apart_from_its_neighbourhood(self):
        # The summary states the mathematics and finds a definition; the
        # neighbours state the situation and find a worked example. Blending
        # them measured worse than either, so they stay separate records.
        with tempfile.TemporaryDirectory() as directory:
            tree = self._equation_in_prose(
                directory, summary="The set M of points within 1/4 of the diagonal."
            )

            texts = [
                record.value
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "text"
            ]

            self.assertIn(
                "The set M of points within 1/4 of the diagonal.\n"
                "M = {(x,y) | |x-y| <= 1/4}",
                texts,
            )
            self.assertTrue(
                any(t.startswith("Figure 1.5:") and "shaded" in t for t in texts),
                f"the neighbourhood must survive alongside the summary: {texts}",
            )

    def test_a_table_is_indexed_with_the_only_prose_it_has(self):
        # A grid embeds as markup. "<table><tr><td>" is noise and the cells are
        # values, so the summary is the one part a question resembles -- and it
        # was the one part left out.
        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "file_name": "book.pdf",
                    "source_sha256": "c" * 64,
                    "document_version_id": "document-version-3",
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            node = TreeNode(
                {
                    "pdf_id": 1,
                    "page_idx": 0,
                    "content": "Table 8.1",
                    "table_body": "<table><tr><td>sin A</td><td>1/2</td></tr></table>",
                }
            )
            node.type = NodeType.TABLE
            node.summary = "Trigonometric ratios for 0, 30, 45, 60 and 90 degrees."
            tree.add_node(node)
            tree.root_node.add_child(node)

            texts = [
                record.value
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "text"
            ]

            self.assertEqual(
                [
                    "Trigonometric ratios for 0, 30, 45, 60 and 90 degrees.\n"
                    "Table 8.1\n"
                    "<table><tr><td>sin A</td><td>1/2</td></tr></table>"
                ],
                texts,
            )

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
            store.add_tree_node_points.side_effect = RuntimeError("write failed")

            with (
                patch.object(vdb_index, "MultimodalEmbeddingProvider", return_value=embedder),
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
                patch.object(vdb_index, "MultimodalEmbeddingProvider", return_value=embedder),
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
