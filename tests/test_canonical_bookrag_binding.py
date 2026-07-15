from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from Core.Index.Tree import DocumentTree
from Core.pipelines.doc_tree_builder import (
    CANONICAL_EVIDENCE_FILE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    PARSER_OBSERVATION_SCHEMA_VERSION,
    CanonicalEvidenceBindingError,
    bind_canonical_evidence,
)
from Core.pipelines.tree_node_builder import create_node_by_type
from docarag.domain import CanonicalBlock, DocumentRef, ParseResult, RelationType


class CanonicalBookRAGBindingTests(unittest.TestCase):
    def make_cfg(self, root: Path):
        source_bytes = b"canonical BookRAG fixture"
        root.mkdir(parents=True, exist_ok=True)
        source_path = root / "source.pdf"
        source_path.write_bytes(source_bytes)
        return SimpleNamespace(
            source=SimpleNamespace(
                display_name="original-name.pdf",
                original_uri="s3://source-bucket/original-name.pdf",
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                snapshot_path=str(source_path),
            ),
            pdf_path=str(source_path),
            save_path=str(root / "run-123"),
            mineru=SimpleNamespace(
                backend="pipeline",
                method="auto",
                lang="en",
                server_url="https://secret:credential@example.invalid/v1",
            ),
        )

    @staticmethod
    def blocks() -> list[dict]:
        # Deliberately not ordered by page. The typed adapter sorts observations,
        # while BookRAG must retain the exact legacy list order.
        return [
            {
                "type": "text",
                "text": "Second-page heading",
                "text_level": 0,
                "page_idx": 1,
                "middle_json": {
                    "type": "title",
                    "bbox": [20, 30, 300, 60],
                    "index": 0,
                },
            },
            {
                "type": "table",
                "table_body": "<table><tr><td>42</td></tr></table>",
                "table_caption": ["Measured value"],
                "table_footnote": [],
                "img_path": "/evidence/table-1.png",
                "page_idx": 0,
                "middle_json": {
                    "type": "table",
                    "bbox": [10, 100, 400, 260],
                    "index": 3,
                },
                "_reconciliation": {
                    "status": "unmatched_content",
                    "reason": "fixture exercises review quality",
                },
            },
        ]

    def bind(self, root: Path, blocks: list[dict]):
        cfg = self.make_cfg(root)
        merged_path = Path(cfg.save_path) / "auto" / "source_merged_content.json"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(blocks), encoding="utf-8")
        request, canonical = bind_canonical_evidence(
            cfg,
            blocks,
            merged_content_path=merged_path,
        )
        return cfg, request, canonical

    def test_canonical_artifacts_tree_node_and_retrieval_share_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocks = self.blocks()
            cfg, request, canonical = self.bind(root, blocks)

            self.assertEqual(2, len(canonical))
            self.assertEqual(
                [item.block_id for item in canonical],
                [item["canonical_block_id"] for item in blocks],
            )
            self.assertEqual([1, 0], [item.page_index for item in canonical])
            self.assertEqual(
                [1, 0],
                [item["page_idx"] for item in blocks],
                "legacy block order changed",
            )

            canonical_path = Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE
            canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
            self.assertEqual(
                CANONICAL_EVIDENCE_SCHEMA_VERSION,
                canonical_payload["artifact_schema_version"],
            )
            self.assertEqual(2, canonical_payload["block_count"])
            self.assertEqual(
                request.document.document_version_id,
                canonical_payload["document"]["document_version_id"],
            )
            self.assertEqual(
                request.document,
                DocumentRef.from_dict(canonical_payload["document"]),
            )
            restored_canonical = tuple(
                CanonicalBlock.from_dict(item)
                for item in canonical_payload["canonical_blocks"]
            )
            self.assertEqual(canonical, restored_canonical)

            observation_path = (
                Path(cfg.save_path) / "auto" / "source_parser_observations.json"
            )
            observation_payload = json.loads(
                observation_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                PARSER_OBSERVATION_SCHEMA_VERSION,
                observation_payload["artifact_schema_version"],
            )
            restored_result = ParseResult.from_dict(observation_payload["result"])
            observations_by_id = {
                item.observation_id: item for item in restored_result.observations
            }
            observations = tuple(
                observations_by_id[item]
                for item in observation_payload["legacy_order_observation_ids"]
            )
            self.assertEqual(
                [item["observation_id"] for item in blocks],
                [item.observation_id for item in observations],
            )
            serialized_artifacts = canonical_path.read_text(
                encoding="utf-8"
            ) + observation_path.read_text(encoding="utf-8")
            self.assertNotIn("credential", serialized_artifacts)
            self.assertNotIn("example.invalid", serialized_artifacts)

            tree = DocumentTree(
                meta_dict={
                    "file_name": request.document.display_name,
                    "source_uri": request.document.source_uri,
                    "source_sha256": request.document.content_sha256,
                    "snapshot_path": cfg.source.snapshot_path,
                    "document_version_id": request.document.document_version_id,
                    "canonical_artifact_path": str(canonical_path),
                },
                cfg=cfg,
            )
            node = create_node_by_type(blocks[1], isTitle=False)
            self.assertEqual(cfg.source.sha256, node.meta_info.source_sha256)
            tree.add_node(node)
            tree.root_node.add_child(node)
            tree.save_to_file()
            restored_tree = DocumentTree.load_from_file(
                DocumentTree.get_save_path(cfg.save_path)
            )
            retrieval = restored_tree.get_nodes_data([node.index_id])[0]
            self.assertEqual(canonical[1].block_id, retrieval["canonical_block_id"])
            self.assertEqual(
                canonical[1].revision_id, retrieval["canonical_revision_id"]
            )
            self.assertEqual(
                canonical[1].source_observation_ids,
                tuple(retrieval["source_observation_ids"]),
            )
            self.assertEqual(
                [item.representation_id for item in canonical[1].representations],
                retrieval["canonical_representation_ids"],
            )
            self.assertEqual(
                blocks[1]["selected_representation_ids"],
                retrieval["selected_representation_ids"],
            )
            self.assertEqual(
                blocks[1]["selected_asset_ids"],
                retrieval["selected_asset_ids"],
            )
            self.assertEqual(
                canonical[1].document_version_id,
                retrieval["document_version_id"],
            )
            self.assertEqual(canonical[1].page_id, retrieval["page_id"])
            self.assertEqual([10.0, 100.0, 400.0, 260.0], retrieval["geometry"]["bbox"])
            self.assertEqual(cfg.source.original_uri, retrieval["source_uri"])
            self.assertEqual(cfg.source.sha256, retrieval["source_sha256"])
            self.assertEqual("mineru", retrieval["parser_provenance"]["parser_name"])
            self.assertEqual("needs_review", retrieval["quality_status"])
            self.assertEqual(
                "mineru_page_reconciliation:unmatched_content",
                retrieval["reconciliation_method"],
            )

    def test_artifacts_are_byte_deterministic_for_the_same_parser_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.blocks()
            cfg, _request, _canonical = self.bind(root, deepcopy(raw))
            canonical_path = Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE
            observation_path = (
                Path(cfg.save_path) / "auto" / "source_parser_observations.json"
            )
            first = (canonical_path.read_bytes(), observation_path.read_bytes())

            self.bind(root, deepcopy(raw))
            second = (canonical_path.read_bytes(), observation_path.read_bytes())
            self.assertEqual(first, second)

    def test_complete_observation_reconciliation_preserves_structural_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocks = [
                {
                    "type": "text",
                    "text": "First paragraph",
                    "page_idx": 0,
                    "middle_json": {
                        "type": "text",
                        "bbox": [10, 10, 300, 50],
                        "index": 0,
                    },
                },
                {
                    "type": "text",
                    "text": "Second paragraph",
                    "page_idx": 0,
                    "middle_json": {
                        "type": "text",
                        "bbox": [10, 60, 300, 100],
                        "index": 1,
                    },
                },
            ]

            _cfg, _request, canonical = self.bind(Path(tmp), blocks)

            self.assertEqual(2, len(canonical))
            self.assertEqual(
                [(RelationType.READING_NEXT, canonical[1].block_id)],
                [
                    (relation.relation_type, relation.target_block_id)
                    for relation in canonical[0].relations
                ],
            )
            self.assertEqual(
                [(RelationType.READING_PREVIOUS, canonical[0].block_id)],
                [
                    (relation.relation_type, relation.target_block_id)
                    for relation in canonical[1].relations
                ],
            )
            self.assertEqual(
                [relation.to_dict() for relation in canonical[0].relations],
                blocks[0]["canonical_relations"],
            )
            self.assertEqual(
                [item.representation_id for item in canonical[0].representations],
                blocks[0]["canonical_representation_ids"],
            )

    def test_canonical_block_identity_ignores_run_local_crop_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_blocks = self.blocks()
            second_blocks = deepcopy(first_blocks)
            second_blocks[1]["img_path"] = "/another-run/crops/table-1.png"

            _cfg, _request, first = self.bind(root / "first", first_blocks)
            _cfg, _request, second = self.bind(root / "second", second_blocks)
            self.assertEqual(
                [item.block_id for item in first],
                [item.block_id for item in second],
            )
            self.assertNotEqual(
                first[1].revision_id,
                second[1].revision_id,
                "asset provenance changes must create a new evidence revision",
            )

    def test_invalid_block_fails_before_canonical_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocks = [
                {
                    "type": "text",
                    "text": "",
                    "page_idx": 0,
                    "middle_json": {"type": "text", "bbox": [0, 0, 10, 10]},
                }
            ]
            cfg = self.make_cfg(root)
            merged_path = Path(cfg.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(blocks), encoding="utf-8")

            with self.assertRaisesRegex(
                CanonicalEvidenceBindingError, "parser-contract translation failed"
            ):
                bind_canonical_evidence(
                    cfg,
                    blocks,
                    merged_content_path=merged_path,
                )
            self.assertFalse((Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE).exists())
            self.assertNotIn("canonical_block_id", blocks[0])
            observation_path = (
                Path(cfg.save_path) / "auto" / "source_parser_observations.json"
            )
            failed_result = ParseResult.from_dict(
                json.loads(observation_path.read_text(encoding="utf-8"))["result"]
            )
            self.assertEqual("failed", failed_result.status.value)
            self.assertTrue(failed_result.issues)

    def test_source_hash_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            cfg.source.sha256 = None
            with self.assertRaisesRegex(
                CanonicalEvidenceBindingError, "cfg.source.sha256"
            ):
                bind_canonical_evidence(
                    cfg,
                    self.blocks(),
                    merged_content_path=root / "not-needed.json",
                )


if __name__ == "__main__":
    unittest.main()
