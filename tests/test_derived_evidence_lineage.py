from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import tempfile
import unittest

from retrieval_engine.indexes.tree import DocumentTree
from retrieval_engine.parsing.pdf_refinement import (
    dash_line_refiner,
    merge_tables_and_mark_invalid,
    merge_text_and_mark_invalid,
    truncate_ocr_error_refiner,
)
from retrieval_engine.indexing.tree.nodes import create_node_by_type


def canonical_block(
    suffix: str,
    text: str,
    *,
    content_type: str = "text",
    page_idx: int = 0,
) -> dict:
    block = {
        "type": content_type,
        "text": text,
        "pdf_id": int(suffix),
        "page_idx": page_idx,
        "middle_json": {"index": int(suffix)},
        "document_version_id": "document-revision-1",
        "source_sha256": "a" * 64,
        "page_id": f"page-{page_idx}",
        "geometry": {
            "coordinate_space": "pdf_points",
            "bbox": [10.0, 20.0, 30.0, 40.0],
        },
        "canonical_block_id": f"block-{suffix}",
        "canonical_revision_id": f"revision-{suffix}",
        "observation_id": f"observation-{suffix}",
        "source_observation_ids": [f"observation-{suffix}"],
        "canonical_representation_ids": [f"representation-{suffix}"],
        "canonical_asset_ids": [],
        "selected_representation_ids": [f"representation-{suffix}"],
        "selected_asset_ids": [],
    }
    if content_type == "table":
        block.pop("text")
        block.update(
            {
                "table_body": text,
                "table_caption": [],
                "table_footnote": [],
            }
        )
    return block


class DerivedEvidenceLineageTests(unittest.TestCase):
    def test_noop_refinement_retains_direct_canonical_identity(self):
        block = canonical_block("1", "No dash normalization is needed.")

        dash_line_refiner([block])

        self.assertEqual("block-1", block["canonical_block_id"])
        self.assertEqual("revision-1", block["canonical_revision_id"])
        self.assertEqual("observation-1", block["observation_id"])
        self.assertNotIn("derived_evidence_id", block)
        self.assertNotIn("transformations", block)

    def test_single_block_changes_are_deterministic_and_composable(self):
        original = canonical_block(
            "1",
            "well- formed opening sentence here a b c d",
        )
        first = deepcopy(original)
        second = deepcopy(original)

        for block in (first, second):
            dash_line_refiner([block])
            truncate_ocr_error_refiner(
                [block],
                window_size=4,
                single_char_ratio_threshold=0.75,
            )

        self.assertEqual(first["derived_evidence_id"], second["derived_evidence_id"])
        self.assertEqual("well-formed opening sentence", first["text"])
        self.assertIsNone(first["canonical_block_id"])
        self.assertIsNone(first["canonical_revision_id"])
        self.assertIsNone(first["observation_id"])
        self.assertEqual(["block-1"], first["source_canonical_block_ids"])
        self.assertEqual(["revision-1"], first["source_canonical_revision_ids"])
        self.assertEqual(["observation-1"], first["source_observation_ids"])
        self.assertEqual(1, len(first["source_evidence_refs"]))
        self.assertEqual(
            [
                "bookrag.dash_line_normalization.v1",
                "bookrag.ocr_garbage_truncation.v1",
            ],
            first["transformations"],
        )
        self.assertEqual("block-1", first["source_evidence_refs"][0]["canonical_block_id"])
        self.assertEqual("page-0", first["source_evidence_refs"][0]["page_id"])
        self.assertEqual(
            [10.0, 20.0, 30.0, 40.0],
            first["source_evidence_refs"][0]["geometry"]["bbox"],
        )

        changed_output = deepcopy(original)
        changed_output["text"] += " extra"
        dash_line_refiner([changed_output])
        self.assertNotEqual(
            first["derived_evidence_id"],
            changed_output["derived_evidence_id"],
        )

    def test_legacy_sources_without_revision_ids_do_not_collide(self):
        first = {
            "type": "text",
            "text": "cross- page",
            "document_version_id": "document-one",
            "page_id": "document-one:page:0",
            "page_idx": 0,
            "geometry": {"bbox": [0, 0, 10, 10]},
        }
        second = {
            "type": "text",
            "text": "cross- page",
            "document_version_id": "document-two",
            "page_id": "document-two:page:0",
            "page_idx": 0,
            "geometry": {"bbox": [0, 0, 10, 10]},
        }

        dash_line_refiner([first, second])

        self.assertEqual([], first["source_canonical_revision_ids"])
        self.assertNotEqual(
            first["derived_evidence_id"],
            second["derived_evidence_id"],
        )

    def test_text_merge_unions_text_and_equation_source_evidence(self):
        target = canonical_block("1", "The measured value is-")
        continuation = canonical_block("2", "consistent across samples", page_idx=1)
        equation = canonical_block(
            "3",
            r"E=mc^2",
            content_type="equation",
            page_idx=1,
        )

        merge_text_and_mark_invalid(target, [continuation, equation])

        self.assertEqual(
            "The measured value isconsistent across samples E=mc^2",
            target["text"],
        )
        self.assertTrue(continuation["invalid"])
        self.assertNotIn("invalid", equation)
        self.assertIsNone(target["canonical_block_id"])
        self.assertEqual(
            ["block-1", "block-2", "block-3"],
            target["source_canonical_block_ids"],
        )
        self.assertEqual(
            ["revision-1", "revision-2", "revision-3"],
            target["source_canonical_revision_ids"],
        )
        self.assertEqual(
            ["observation-1", "observation-2", "observation-3"],
            target["source_observation_ids"],
        )
        self.assertEqual(
            ["representation-1", "representation-2", "representation-3"],
            target["canonical_representation_ids"],
        )
        self.assertEqual(
            ["representation-1", "representation-2", "representation-3"],
            target["selected_representation_ids"],
        )
        self.assertEqual(
            ["page-0", "page-1", "page-1"],
            [ref["page_id"] for ref in target["source_evidence_refs"]],
        )
        self.assertEqual(
            ["bookrag.text_merge.v1"],
            target["transformations"],
        )

    def test_table_merge_unions_source_revisions(self):
        target = canonical_block(
            "1",
            "<table><tbody><tr><td>A</td></tr></tbody></table>",
            content_type="table",
        )
        continuation = canonical_block(
            "2",
            "<table><tbody><tr><td>B</td></tr></tbody></table>",
            content_type="table",
            page_idx=1,
        )

        merge_tables_and_mark_invalid(target, continuation)

        self.assertTrue(continuation["invalid"])
        self.assertIn("<td>B</td>", target["table_body"])
        self.assertIsNone(target["canonical_block_id"])
        self.assertEqual(
            ["revision-1", "revision-2"],
            target["source_canonical_revision_ids"],
        )
        self.assertEqual(
            ["bookrag.table_merge.v1"],
            target["transformations"],
        )

    def test_tree_retrieval_exposes_derived_lineage_without_direct_id(self):
        block = canonical_block("1", "cross- page")
        dash_line_refiner([block])

        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(save_path=directory)
            tree = DocumentTree(
                meta_dict={
                    "source_uri": "s3://documents/source.pdf",
                    "source_sha256": "a" * 64,
                },
                cfg=cfg,
            )
            node = create_node_by_type(block, isTitle=False)
            tree.add_node(node)
            tree.root_node.add_child(node)
            tree.save_to_file()
            restored = DocumentTree.load_from_file(
                DocumentTree.get_save_path(directory)
            )
            retrieval = restored.get_nodes_data([node.index_id])[0]

        self.assertNotIn("canonical_block_id", retrieval)
        self.assertNotIn("canonical_revision_id", retrieval)
        self.assertNotIn("observation_id", retrieval)
        self.assertEqual(block["derived_evidence_id"], retrieval["derived_evidence_id"])
        self.assertEqual(["block-1"], retrieval["source_canonical_block_ids"])
        self.assertEqual(["revision-1"], retrieval["source_canonical_revision_ids"])
        self.assertEqual(block["source_evidence_refs"], retrieval["source_evidence_refs"])
        self.assertEqual(block["transformations"], retrieval["transformations"])
        self.assertEqual("s3://documents/source.pdf", retrieval["source_uri"])


if __name__ == "__main__":
    unittest.main()
