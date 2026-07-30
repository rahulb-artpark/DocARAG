from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from Core.Index.Tree import DocumentTree, NodeType
from Core.configs.figure_config import FigureConfig
from Core.pipelines.doc_tree_builder import (
    CANONICAL_EVIDENCE_FILE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    PARSER_OBSERVATION_SCHEMA_VERSION,
    CanonicalEvidenceBindingError,
    bind_canonical_evidence,
    merge_canonical_evidence_shards,
)
from Core.pipelines.tree_node_builder import create_node_by_type
from docarag.bookrag_projection import project_canonical_blocks
from docarag.domain import CanonicalBlock, DocumentRef, ParseResult, RelationType
from docarag.evidence_hydration import CanonicalEvidenceCatalog


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
            figures=FigureConfig(
                ocr=False,
                describe=False,
                transcribe_formulas=False,
                transcribe_tables=False,
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

    def bind(self, root: Path, blocks: list[dict], **kwargs):
        cfg = self.make_cfg(root)
        merged_path = Path(cfg.save_path) / "auto" / "source_merged_content.json"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(blocks), encoding="utf-8")
        request, canonical, _has_unresolved_cross_page_relation = bind_canonical_evidence(
            cfg,
            blocks,
            merged_content_path=merged_path,
            **kwargs,
        )
        return cfg, request, canonical

    def test_image_row_takes_its_caption_channels_from_the_canonical_projection(
        self,
    ):
        # BookRAG builds an image node's content from these channels while the
        # evidence hydrator re-derives it from the canonical projection.  The
        # row must carry the projection's text, not MinerU's own caption list.
        blocks = [
            {
                "type": "image",
                "img_path": "/evidence/figure-1.png",
                "image_caption": ["Figure 1: system overview"],
                "image_footnote": [],
                "page_idx": 0,
                "middle_json": {
                    "type": "image",
                    "bbox": [10, 10, 200, 200],
                    "index": 1,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, canonical = self.bind(Path(tmp), blocks)

            projected = project_canonical_blocks(request.document, canonical)[0]
            self.assertEqual(
                list(projected["image_caption"]), blocks[0]["image_caption"]
            )
            self.assertEqual(
                list(projected["image_footnote"]), blocks[0]["image_footnote"]
            )

    def test_native_visual_types_keep_text_channels_and_tree_types(self):
        cases = (
            (
                "image_body",
                {
                    "image_caption": ["Figure 1: eye anatomy"],
                    "image_footnote": ["Retina highlighted"],
                },
                NodeType.IMAGE,
                "Figure 1: eye anatomy Retina highlighted",
            ),
            (
                "chart_body",
                {
                    "content": "Myopia: 40%\nHypermetropia: 25%",
                    "chart_caption": ["Prevalence by condition"],
                    "chart_footnote": ["Illustrative values"],
                },
                NodeType.CHART,
                (
                    "Prevalence by condition Myopia: 40%\n"
                    "Hypermetropia: 25% Illustrative values"
                ),
            ),
        )
        for block_type, channels, expected_type, expected_content in cases:
            with self.subTest(block_type=block_type), tempfile.TemporaryDirectory() as tmp:
                blocks = [
                    {
                        "type": block_type,
                        **channels,
                        "page_idx": 0,
                        "middle_json": {
                            "type": block_type,
                            "bbox": [10, 10, 300, 200],
                            "index": 1,
                        },
                        "_reconciliation": {
                            "status": "unmatched_content",
                            "reason": "native visual type fixture",
                        },
                    }
                ]
                _cfg, request, canonical = self.bind(Path(tmp), blocks)

                projected = project_canonical_blocks(request.document, canonical)[0]
                self.assertEqual(expected_type.value, projected["type"])
                self.assertEqual(projected["image_caption"], blocks[0]["image_caption"])
                self.assertEqual(projected["image_footnote"], blocks[0]["image_footnote"])

                node = create_node_by_type(blocks[0], isTitle=False)
                self.assertEqual(expected_type, node.type)
                self.assertEqual(expected_content, node.meta_info.content)

    def test_equation_row_text_follows_the_projection_selection(self):
        # When a vision transcription is preferred over the recognizer output,
        # a tree built from the recognizer text would fail hydration.
        blocks = [
            {
                "type": "equation",
                "text": "$$\nv _ { 2 }\n$$",
                "text_format": "latex",
                "page_idx": 0,
                "middle_json": {
                    "type": "interline_equation",
                    "bbox": [10, 10, 200, 60],
                    "index": 1,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, canonical = self.bind(Path(tmp), blocks)

            projected = project_canonical_blocks(request.document, canonical)[0]
            self.assertEqual(projected["text"], blocks[0]["text"])
            self.assertEqual(projected["text_format"], blocks[0]["text_format"])

    def test_native_interline_equation_becomes_an_equation_node(self):
        blocks = [
            {
                "type": "interline_equation",
                "text": r"E=mc^2",
                "text_format": "latex",
                "page_idx": 0,
                "middle_json": {
                    "type": "interline_equation",
                    "bbox": [10, 10, 200, 60],
                    "index": 1,
                },
                "_reconciliation": {
                    "status": "unmatched_content",
                    "reason": "native equation type fixture",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, canonical = self.bind(Path(tmp), blocks)

            projected = project_canonical_blocks(request.document, canonical)[0]
            self.assertEqual("equation", projected["type"])
            self.assertEqual("equation", blocks[0]["type"])
            node = create_node_by_type(blocks[0], isTitle=False)
            self.assertEqual(NodeType.EQUATION, node.type)
            self.assertEqual(r"E=mc^2", node.meta_info.content)

    def test_list_items_reach_the_tree_as_renderable_markdown(self):
        blocks = [
            {
                "type": "list",
                "sub_type": "text",
                "list_items": [
                    "• Cannot see nearby objects distinctly.",
                    "Can see distant objects clearly.",
                    "3. Image forms behind the retina.",
                ],
                "page_idx": 0,
                "middle_json": {
                    "type": "list",
                    "bbox": [10, 10, 300, 100],
                    "index": 1,
                },
                "_reconciliation": {
                    "status": "unmatched_content",
                    "reason": "fixture models MinerU list-only content",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, canonical = self.bind(Path(tmp), blocks)

            projected = project_canonical_blocks(request.document, canonical)[0]
            expected = (
                "- Cannot see nearby objects distinctly.\n"
                "- Can see distant objects clearly.\n"
                "- Image forms behind the retina."
            )
            self.assertEqual(expected, projected["text"])
            self.assertEqual("markdown", projected["text_format"])
            self.assertEqual(expected, blocks[0]["text"])

            node = create_node_by_type(blocks[0], isTitle=False)
            self.assertEqual(NodeType.LIST, node.type)
            self.assertEqual(expected, node.meta_info.content)

    def test_specialized_text_blocks_reach_their_typed_tree_nodes(self):
        cases = (
            ("code_body", {"code_body": "def answer():\n    return 42"}, NodeType.CODE),
            ("algorithm", {"text": "1. Read input\n2. Return output"}, NodeType.ALGORITHM),
            ("abstract", {"text": "This paper presents a bounded method."}, NodeType.ABSTRACT),
            ("ref_text", {"text": "Doe, J. Example Systems, 2026."}, NodeType.REFERENCE),
        )
        for block_type, payload, expected_type in cases:
            with self.subTest(block_type=block_type), tempfile.TemporaryDirectory() as tmp:
                blocks = [
                    {
                        "type": block_type,
                        **payload,
                        "page_idx": 0,
                        "middle_json": {
                            "type": block_type,
                            "bbox": [10, 10, 300, 100],
                            "index": 1,
                        },
                        "_reconciliation": {
                            "status": "unmatched_content",
                            "reason": "typed textual block fixture",
                        },
                    }
                ]
                _cfg, request, canonical = self.bind(Path(tmp), blocks)

                projected = project_canonical_blocks(request.document, canonical)[0]
                expected_content = payload.get("text", payload.get("code_body"))
                self.assertEqual(expected_type.value, projected["type"])
                self.assertEqual(expected_content, projected["text"])
                self.assertEqual(expected_content, blocks[0]["text"])

                node = create_node_by_type(blocks[0], isTitle=False)
                self.assertEqual(expected_type, node.type)
                self.assertEqual(expected_content, node.meta_info.content)

    def test_table_row_channels_follow_the_projection(self):
        # MinerU's raw caption and footnote strings differ from the canonical
        # ones in whitespace, and the hydrator compares exactly.
        blocks = [
            {
                "type": "table",
                "table_body": "<table><tr><td>42</td></tr></table>",
                "table_caption": [],
                "table_footnote": ["The \" symbol means arcseconds. "],
                "img_path": "/evidence/table-1.png",
                "page_idx": 0,
                "middle_json": {
                    "type": "table",
                    "bbox": [10, 100, 400, 260],
                    "index": 1,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, canonical = self.bind(Path(tmp), blocks)

            projected = project_canonical_blocks(request.document, canonical)[0]
            self.assertEqual(
                list(projected["table_caption"]), blocks[0]["table_caption"]
            )
            self.assertEqual(
                list(projected["table_footnote"]), blocks[0]["table_footnote"]
            )
            self.assertEqual(projected["table_body"], blocks[0]["table_body"])

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

    def test_document_page_count_override_is_not_truncated_to_this_calls_blocks(
        self,
    ):
        # A per-shard caller only ever sees its own shard's blocks (here,
        # pages 0-1), so deriving page_count from them would silently
        # truncate it for every shard but the one containing the document's
        # last page. document_page_count lets the caller report the real
        # total instead.
        blocks = self.blocks()
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, _canonical = self.bind(
                Path(tmp), blocks, document_page_count=50
            )
            self.assertEqual(50, request.document.page_count)

    def test_document_page_count_defaults_to_derived_value_when_omitted(self):
        blocks = self.blocks()
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, request, _canonical = self.bind(Path(tmp), blocks)
            self.assertEqual(2, request.document.page_count)

    def test_debug_artifact_suffix_prevents_shard_dumps_from_overwriting_each_other(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            merged_path = Path(cfg.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(self.blocks()), encoding="utf-8")

            bind_canonical_evidence(
                cfg,
                [self.blocks()[0]],
                merged_content_path=merged_path,
                debug_artifact_suffix="_shard000",
            )
            bind_canonical_evidence(
                cfg,
                [self.blocks()[1]],
                merged_content_path=merged_path,
                debug_artifact_suffix="_shard001",
            )

            dump_dir = Path(cfg.save_path) / "auto"
            self.assertTrue(
                (dump_dir / "source_shard000_parser_observations.json").is_file()
            )
            self.assertTrue(
                (dump_dir / "source_shard001_parser_observations.json").is_file()
            )

    def test_omitted_debug_artifact_suffix_matches_todays_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _request, _canonical = self.bind(root, self.blocks())
            self.assertTrue(
                (Path(cfg.save_path) / "auto" / "source_parser_observations.json")
                .is_file()
            )

    def test_merging_shard_fragments_matches_a_whole_document_call(self):
        # Two separate bind_canonical_evidence calls, one per block, each
        # writing to its own fragment filename instead of the shared
        # canonical_blocks.json -- simulating two shards of the same
        # document. Merging them must produce exactly what one
        # whole-document call would have.
        blocks = self.blocks()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            merged_path = Path(cfg.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(blocks), encoding="utf-8")

            whole_request, whole_canonical, _has_unresolved = bind_canonical_evidence(
                cfg,
                deepcopy(blocks),
                merged_content_path=merged_path,
                canonical_artifact_filename="whole_document_reference.json",
            )

            fragment_names = []
            for index, block in enumerate(deepcopy(blocks)):
                fragment_name = f"canonical_blocks_shard{index:03d}.json"
                bind_canonical_evidence(
                    cfg,
                    [block],
                    merged_content_path=merged_path,
                    debug_artifact_suffix=f"_shard{index:03d}",
                    canonical_artifact_filename=fragment_name,
                )
                fragment_names.append(fragment_name)

            canonical_path = merge_canonical_evidence_shards(
                cfg, shard_fragment_filenames=fragment_names
            )
            self.assertEqual(Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE, canonical_path)

            catalog = CanonicalEvidenceCatalog.load(canonical_path)
            self.assertEqual(whole_request.document, catalog.document)
            self.assertEqual(
                sorted(item.block_id for item in whole_canonical),
                sorted(block.block_id for block in catalog.blocks),
            )
            self.assertEqual(
                sorted(item.revision_id for item in whole_canonical),
                sorted(block.revision_id for block in catalog.blocks),
            )

            payload = json.loads(canonical_path.read_text(encoding="utf-8"))
            self.assertEqual(len(blocks), payload["block_count"])
            self.assertEqual(len(blocks), len(payload["canonical_blocks"]))

    def test_merge_rejects_an_empty_shard_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.make_cfg(Path(tmp))
            with self.assertRaises(CanonicalEvidenceBindingError):
                merge_canonical_evidence_shards(cfg, shard_fragment_filenames=[])

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

    def test_missing_table_continuation_target_is_flagged_for_fallback(self):
        # Simulates what a per-shard bind_canonical_evidence call sees when a
        # table's continuation landed in a different shard: only table-1 is
        # present, so its next_table_id can't resolve within this call.
        blocks = [
            {
                "type": "table",
                "block_id": "table-1",
                "table_body": "<table><tr><td>1</td></tr></table>",
                "page_idx": 0,
                "next_table_id": "table-2",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg = self.make_cfg(Path(tmp))
            merged_path = Path(_cfg.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(blocks), encoding="utf-8")
            _request, _canonical, has_unresolved = bind_canonical_evidence(
                _cfg, blocks, merged_content_path=merged_path,
            )
            self.assertTrue(has_unresolved)

    def test_table_continuation_resolved_within_one_call_is_not_flagged(self):
        # Control: both halves present in the same call (as a whole-document
        # bind, or two blocks that happen to land in the same shard) resolve
        # normally and must not trigger the fallback signal.
        blocks = [
            {
                "type": "table",
                "block_id": "table-1",
                "table_body": "<table><tr><td>1</td></tr></table>",
                "page_idx": 0,
                "next_table_id": "table-2",
            },
            {
                "type": "table",
                "block_id": "table-2",
                "table_body": "<table><tr><td>2</td></tr></table>",
                "page_idx": 1,
                "previous_table_id": "table-1",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _cfg = self.make_cfg(Path(tmp))
            merged_path = Path(_cfg.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(blocks), encoding="utf-8")
            _request, canonical, has_unresolved = bind_canonical_evidence(
                _cfg, blocks, merged_content_path=merged_path,
            )
            self.assertFalse(has_unresolved)
            self.assertEqual(2, len(canonical))

    def test_an_unrelated_needs_review_reason_does_not_trigger_fallback(self):
        # A page-reconciliation mismatch also produces NEEDS_REVIEW quality,
        # but it is not a cross-page relation gap and must not be conflated
        # with one -- the fallback signal is specifically about relations
        # whose target this call's own blocks didn't include.
        blocks = self.blocks()  # includes one block with _reconciliation unmatched_content
        with tempfile.TemporaryDirectory() as tmp:
            _cfg, _request, canonical = self.bind(Path(tmp), blocks)
            self.assertTrue(
                any(item.quality_status.value == "needs_review" for item in canonical)
            )
        with tempfile.TemporaryDirectory() as tmp2:
            _cfg2 = self.make_cfg(Path(tmp2))
            merged_path = Path(_cfg2.save_path) / "auto" / "source_merged_content.json"
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_text(json.dumps(blocks), encoding="utf-8")
            _request2, _canonical2, has_unresolved = bind_canonical_evidence(
                _cfg2, blocks, merged_content_path=merged_path,
            )
            self.assertFalse(has_unresolved)

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
