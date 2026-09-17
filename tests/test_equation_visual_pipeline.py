from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from retrieval_engine.indexes.tree import DocumentTree, NodeType
from retrieval_engine.indexing.tree.nodes import create_node_by_type
from retrieval_engine.indexing.vector import builder as vdb_index
from retrieval_engine.rag.gbc.answer import AnswerAgent


class EquationVisualPipelineTests(unittest.TestCase):
    def _build_and_reload_tree(self, directory: str) -> tuple[DocumentTree, Path, str]:
        crop = Path(directory) / "equation-crop.jpg"
        crop.write_bytes(b"representative equation crop")
        latex = r"$$E = mc^2$$"

        tree = DocumentTree(
            meta_dict={
                "file_name": "equations.pdf",
                "file_path": "/source.pdf",
                "source_sha256": "a" * 64,
                "document_version_id": "equations-v1",
            },
            cfg=SimpleNamespace(save_path=directory),
        )
        equation = create_node_by_type(
            {
                "type": "equation",
                "text": latex,
                "text_format": "latex",
                "img_path": str(crop),
                "pdf_id": 17,
                "page_idx": 2,
                "canonical_block_id": "equation-17",
                "middle_json": {"type": "interline_equation"},
            },
            isTitle=False,
        )
        tree.add_node(equation)
        tree.root_node.add_child(equation)
        tree.save_to_file()

        return DocumentTree.load_from_file(DocumentTree.get_save_path(directory)), crop, latex

    def test_equation_crop_survives_real_tree_and_retrieval_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, crop, latex = self._build_and_reload_tree(directory)

            equation = tree.get_node_by_pdf_id(17)
            self.assertEqual(NodeType.EQUATION, equation.type)
            self.assertEqual(str(crop), equation.meta_info.img_path)

            serialized = tree.get_nodes_data([equation.index_id])
            self.assertEqual(
                {
                    "index_id": equation.index_id,
                    "type": NodeType.EQUATION,
                    "content": latex,
                    "page": 2,
                    "canonical_block_id": "equation-17",
                    "source_sha256": "a" * 64,
                    "img_path": str(crop),
                },
                serialized[0],
            )

            tree_json = json.loads((Path(directory) / "tree.json").read_text())
            json_equation = next(
                node for node in tree_json["nodes"] if node["index_id"] == equation.index_id
            )
            self.assertEqual(str(crop), json_equation["meta_info"]["img_path"])

            # Exercise the real retrieval payload rather than a hand-built
            # answer dictionary: an equation remains text evidence and also
            # becomes visual evidence for the answer VLM.
            agent = AnswerAgent.__new__(AnswerAgent)
            text_nodes, image_nodes = agent._prepare_evidence(serialized)
            self.assertEqual(latex, text_nodes[0]["content"])
            self.assertEqual(str(crop), image_nodes[0]["img_path"])
            self.assertEqual(3, text_nodes[0]["page"])
            self.assertEqual(2, serialized[0]["page"])

    def test_equation_latex_and_crop_reach_multimodal_vector_store(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, crop, latex = self._build_and_reload_tree(directory)
            embedding = Mock()
            store = Mock()
            store.embedding_model = embedding
            # The LaTeX and the crop are one node, so they are one point.
            store.count.return_value = 1
            cfg = SimpleNamespace(
                mm_embedding=True,
                tenant_id="test",
                corpus_id="test",
                qdrant_url="http://qdrant:6333",
                api_key_env="QDRANT_API_KEY",
                timeout_seconds=1.0,
                embedding_config=SimpleNamespace(
                    model_name="test-gme",
                    device="cpu",
                ),
                manifest_dir_name=str(Path(directory) / "vector-attestation"),
                collection_name="test",
                index_batch_size=256,
            )

            with (
                patch.object(vdb_index, "MultimodalEmbeddingProvider", return_value=embedding),
                patch.object(vdb_index, "VectorStore", return_value=store),
            ):
                vdb_index.build_vdb_index(tree, cfg)

            store.add_tree_node_points.assert_called_once()
            records = store.add_tree_node_points.call_args.kwargs["records"]
            self.assertEqual(1, len(records))
            record = records[0]

            # Both readings on one point, each under its own name, and no
            # point-level modality: a point holds several vectors and each
            # one's modality is the vector it was written under.
            self.assertEqual({"text": latex, "image": str(crop)}, record["vectors"])
            self.assertEqual(latex, record["image_text"])
            self.assertEqual("node", record["kind"])
            self.assertEqual(
                {
                    "node_id": 1,
                    "node_type": "equation",
                    "index_kind": "tree",
                    "pdf_id": 17,
                    "document_version_id": "equations-v1",
                    "page_idx": 2,
                    "source_sha256": "a" * 64,
                    "canonical_block_id": "equation-17",
                },
                record["metadata"],
            )
            store.close.assert_called_once_with()
            store.reset.assert_called_once_with()
            embedding.close.assert_called_once_with()

    def test_equation_crop_is_used_by_multimodal_reranker_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, crop, latex = self._build_and_reload_tree(directory)
            embedding = Mock()
            embedding.embed_fused.return_value = [[0.1, 0.2, 0.3]]
            cfg = SimpleNamespace(
                save_path=directory,
                vdb=SimpleNamespace(
                    embedding_config=SimpleNamespace(
                        model_name="test-gme", device="cpu"
                    )
                ),
            )

            with patch.object(
                vdb_index, "MultimodalEmbeddingProvider", return_value=embedding
            ):
                vdb_index.compute_mm_embedding(cfg, tree)

            embedding.embed_fused.assert_called_once_with(
                texts=[latex], images=[str(crop)]
            )
            metadata = json.loads(
                (Path(directory) / "mm_node_metadata.json").read_text()
            )
            self.assertEqual(NodeType.EQUATION, metadata[0]["node_type"])
            self.assertEqual(str(crop), metadata[0]["img_path"])
            self.assertEqual(0, metadata[0]["embedding_idx"])


if __name__ == "__main__":
    unittest.main()
