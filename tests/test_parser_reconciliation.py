from __future__ import annotations

import json
import subprocess
import sys
import unittest

from document_services.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    ContractValidationError,
    CoordinateSpace,
    PageRef,
    ParserBlockObservation,
    ParserProvenance,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
    content_sha256,
)
from document_services.parsing.reconciliation import (
    ReconciliationError,
    ReconciliationIssueCode,
    ReconciliationPolicy,
    ReconciliationResult,
    reconcile_observations,
)


DOCUMENT_VERSION = "document-version-reconciliation-fixture"


def representation(
    content: object,
    *,
    kind: RepresentationKind = RepresentationKind.TEXT,
    media_type: str = "text/plain",
) -> Representation:
    return Representation(kind=kind, content=content, media_type=media_type)


def observation(
    parser: str,
    native_id: str,
    *,
    page: int = 0,
    block_type: BlockType = BlockType.PARAGRAPH,
    text: str | None = "Quarterly revenue increased",
    representations: tuple[Representation, ...] | None = None,
    bbox: tuple[float, float, float, float] | None = (0.1, 0.1, 0.7, 0.2),
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    page_size: tuple[float, float] | None = None,
    confidence: float | None = 0.9,
    assets: tuple[AssetRef, ...] = (),
    document_version: str = DOCUMENT_VERSION,
) -> ParserBlockObservation:
    page_ref = PageRef(document_version, page)
    if representations is None:
        representations = (
            (representation(text) if text is not None else None),
        )
        representations = tuple(item for item in representations if item is not None)
    geometry = None
    if bbox is not None:
        geometry = SourceGeometry(
            coordinate_space=coordinate_space,
            bbox=BoundingBox(*bbox),
            page_width=page_size[0] if page_size else None,
            page_height=page_size[1] if page_size else None,
        )
    return ParserBlockObservation(
        document_version_id=document_version,
        page_id=page_ref.page_id,
        page_index=page,
        provenance=ParserProvenance(
            parser_name=parser,
            parser_version="1.0",
            model_version=f"{parser}-model",
            config_sha256=content_sha256({"parser": parser}),
            parser_native_id=native_id,
            raw_artifact_uri=f"artifact://fixture/{parser}.json",
        ),
        block_type=block_type,
        representations=representations,
        geometry=geometry,
        reading_order=1.0,
        confidence=confidence,
        assets=assets,
    )


class ParserReconciliationTests(unittest.TestCase):
    def test_fuses_same_page_geometry_and_text_without_losing_sources(self):
        docling = observation("docling", "text-1", confidence=0.97)
        mineru = observation(
            "mineru",
            "text-a",
            bbox=(0.11, 0.1, 0.69, 0.2),
            confidence=0.93,
            representations=(
                representation("Quarterly revenue increased"),
                representation(
                    "Quarterly **revenue** increased",
                    kind=RepresentationKind.MARKDOWN,
                    media_type="text/markdown",
                ),
            ),
        )

        result = reconcile_observations((mineru, docling))

        self.assertEqual(1, len(result.canonical_blocks))
        block = result.canonical_blocks[0]
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            set(block.source_observation_ids),
        )
        self.assertEqual(QualityStatus.ACCEPTED, block.quality_status)
        self.assertEqual(2, len(block.representations))
        self.assertEqual(docling.observation_id, block.metadata["anchor_observation_id"])
        self.assertTrue(block.metadata["match_evidence"])
        self.assertEqual(1, result.matched_pair_count)
        self.assertEqual(1, result.cluster_merge_count)

    def test_never_merges_across_pages_or_unrelated_regions(self):
        first = observation("docling", "a", page=0)
        other_page = observation("mineru", "b", page=1)
        other_region = observation("paddleocr-vl", "c", bbox=(0.1, 0.7, 0.7, 0.8))

        result = reconcile_observations((first, other_page, other_region))

        self.assertEqual(3, len(result.canonical_blocks))
        self.assertTrue(all(len(block.source_observation_ids) == 1 for block in result.canonical_blocks))
        self.assertEqual(0, result.matched_pair_count)
        self.assertEqual(0, result.cluster_merge_count)

    def test_normalizes_explicit_page_coordinates_before_matching(self):
        normalized = observation("docling", "a", bbox=(0.1, 0.1, 0.7, 0.2))
        pixels = observation(
            "mineru",
            "b",
            bbox=(100, 100, 700, 200),
            coordinate_space=CoordinateSpace.PAGE_PIXELS,
            page_size=(1000, 1000),
        )
        result = reconcile_observations((normalized, pixels))
        self.assertEqual(1, len(result.canonical_blocks))

    def test_geometry_free_blocks_require_exact_normalized_text(self):
        first = observation("docling", "a", bbox=None, text="  Total\nRevenue ")
        same = observation("mineru", "b", bbox=None, text="total revenue")
        different = observation("paddleocr-vl", "c", bbox=None, text="total expenses")
        result = reconcile_observations((first, same, different))
        self.assertEqual(2, len(result.canonical_blocks))
        self.assertIn(
            {first.observation_id, same.observation_id},
            [set(block.source_observation_ids) for block in result.canonical_blocks],
        )

    def test_table_visual_cross_type_preserves_grid_crop_and_marks_disagreement(self):
        table_crop = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri="artifact://fixture/table.png",
            media_type="image/png",
        )
        table = observation(
            "mineru",
            "table",
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    {"cells": [["Q1", "10"]]},
                    kind=RepresentationKind.TABLE_GRID,
                    media_type="application/json",
                ),
                representation(
                    "|Q1|10|",
                    kind=RepresentationKind.MARKDOWN,
                    media_type="text/markdown",
                ),
            ),
            assets=(table_crop,),
            confidence=0.93,
        )
        visual = observation(
            "paddleocr-vl",
            "visual",
            block_type=BlockType.FIGURE,
            representations=(
                representation(
                    "table showing Q1 10",
                    kind=RepresentationKind.OCR_TEXT,
                ),
            ),
            confidence=0.82,
        )

        result = reconcile_observations((visual, table))

        block = result.canonical_blocks[0]
        self.assertEqual(BlockType.TABLE, block.block_type)
        self.assertEqual(2, len(block.source_observation_ids))
        self.assertEqual(table_crop.asset_id, block.assets[0].asset_id)
        self.assertIn(
            ReconciliationIssueCode.TYPE_DISAGREEMENT,
            {issue.code for issue in result.issues},
        )

    def test_equation_and_plain_text_merge_only_with_latex_and_strong_geometry(self):
        equation = observation(
            "mineru",
            "equation",
            block_type=BlockType.EQUATION,
            representations=(
                representation(
                    "E=mc^2",
                    kind=RepresentationKind.LATEX,
                    media_type="application/x-latex",
                ),
            ),
        )
        text = observation("docling", "plain", text="E=mc^2")
        result = reconcile_observations((text, equation))
        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(BlockType.EQUATION, result.canonical_blocks[0].block_type)

    def test_overlapping_siblings_from_same_parser_are_not_collapsed(self):
        first = observation("docling", "a")
        second = observation("docling", "b")
        result = reconcile_observations((first, second))
        self.assertEqual(2, len(result.canonical_blocks))

    def test_complete_link_prevents_transitive_multi_parser_collapse(self):
        first = observation("docling", "a", bbox=(0.0, 0.0, 0.5, 0.2))
        bridge = observation("mineru", "b", bbox=(0.2, 0.0, 0.7, 0.2))
        last = observation("paddleocr-vl", "c", bbox=(0.4, 0.0, 0.9, 0.2))
        policy = ReconciliationPolicy(minimum_iou=0.4, minimum_text_similarity=0.0)
        result = reconcile_observations((first, bridge, last), policy=policy)
        self.assertGreaterEqual(len(result.canonical_blocks), 2)

    def test_candidate_bound_is_visible_and_marks_quality_for_review(self):
        values = [observation("docling", "anchor")]
        values.extend(
            observation(f"parser-{index}", f"block-{index}") for index in range(5)
        )
        policy = ReconciliationPolicy(max_candidates_per_observation=1)
        result = reconcile_observations(values, policy=policy)
        self.assertIn(
            ReconciliationIssueCode.CANDIDATE_LIMIT,
            {issue.code for issue in result.issues},
        )
        self.assertIn(
            QualityStatus.NEEDS_REVIEW,
            {block.quality_status for block in result.canonical_blocks},
        )

    def test_input_order_does_not_change_result_identity(self):
        values = (
            observation("docling", "a"),
            observation("mineru", "b"),
            observation("paddleocr-vl", "c", page=1),
        )
        forward = reconcile_observations(values)
        reverse = reconcile_observations(reversed(values))
        self.assertEqual(forward.result_id, reverse.result_id)
        self.assertEqual(forward.to_dict(), reverse.to_dict())

    def test_policy_and_result_have_strict_verified_json_roundtrip(self):
        policy = ReconciliationPolicy(max_candidates_per_observation=9)
        self.assertEqual(policy, ReconciliationPolicy.from_json(policy.to_json()))
        result = reconcile_observations(
            (observation("docling", "a"), observation("mineru", "b")),
            policy=policy,
        )
        restored = ReconciliationResult.from_json(result.to_json())
        self.assertEqual(result, restored)

        tampered = result.to_dict()
        tampered["result_id"] = "reconciliation_tampered"
        with self.assertRaises(ContractValidationError):
            ReconciliationResult.from_dict(tampered)

    def test_identical_duplicate_is_deduplicated_and_reported(self):
        item = observation("docling", "a")
        result = reconcile_observations((item, item))
        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(
            ReconciliationIssueCode.DUPLICATE_OBSERVATION,
            result.issues[0].code,
        )

    def test_same_identity_with_changed_content_is_rejected(self):
        first = observation("docling", "a", text="first")
        changed = observation("docling", "a", text="changed")
        self.assertEqual(first.observation_id, changed.observation_id)
        with self.assertRaises(ReconciliationError):
            reconcile_observations((first, changed))

    def test_cross_document_input_is_rejected(self):
        with self.assertRaises(ReconciliationError):
            reconcile_observations(
                (
                    observation("docling", "a"),
                    observation("mineru", "b", document_version="other-version"),
                )
            )

    def test_policy_is_validated_and_content_addressed(self):
        first = ReconciliationPolicy()
        second = ReconciliationPolicy()
        self.assertEqual(first.policy_sha256, second.policy_sha256)
        with self.assertRaises(ContractValidationError):
            ReconciliationPolicy(minimum_iou=1.1)
        with self.assertRaises(ContractValidationError):
            ReconciliationPolicy(high_iou=0.1, minimum_iou=0.5)

    def test_import_does_not_load_parser_or_model_runtimes(self):
        script = """
import json, sys
import document_services.parsing.reconciliation
forbidden = ("retrieval_engine", 'torch', 'mineru', 'paddle', 'paddleocr', 'docling', 'transformers', 'vllm')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual([], json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
