"""Adversarial merge gates for parser reconciliation.

These tests encode source-truth invariants rather than implementation details:
coordinates must share one physical frame, conflicting evidence must never be
silently accepted, candidate bounds must fail closed without starving the best
match, and canonical identity/lineage must survive incremental reconciliation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from unittest import mock
import unittest

import document_services.parsing.reconciliation as reconciliation_module
from document_services.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    ContractValidationError,
    CoordinateOrigin,
    CoordinateSpace,
    PageRef,
    ParserBlockObservation,
    ParserProvenance,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
    content_sha256,
    stable_id,
)
from document_services.parsing.reconciliation import (
    ReconciliationIssueCode,
    ReconciliationPolicy,
    ReconciliationResult,
    reconcile_observations,
)


DOCUMENT_VERSION = "document-version-reconciliation-adversarial"


def representation(
    content: object,
    *,
    kind: RepresentationKind = RepresentationKind.TEXT,
    media_type: str = "text/plain",
    confidence: float | None = None,
) -> Representation:
    return Representation(
        kind=kind,
        content=content,
        media_type=media_type,
        confidence=confidence,
    )


def observation(
    parser: str,
    native_id: str,
    *,
    page: int = 0,
    block_type: BlockType = BlockType.PARAGRAPH,
    text: str | None = "same source content",
    representations: tuple[Representation, ...] | None = None,
    bbox: tuple[float, float, float, float] | None = (0.1, 0.1, 0.7, 0.2),
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    page_size: tuple[float, float] | None = None,
    coordinate_origin: str | None = None,
    reading_order: float | None = 1.0,
    confidence: float | None = 0.9,
    assets: tuple[AssetRef, ...] = (),
) -> ParserBlockObservation:
    if representations is None:
        representations = (
            (representation(text, confidence=confidence) if text is not None else None),
        )
        representations = tuple(item for item in representations if item is not None)
    geometry = None
    if bbox is not None:
        geometry = SourceGeometry(
            coordinate_space=coordinate_space,
            bbox=BoundingBox(*bbox),
            page_width=page_size[0] if page_size else None,
            page_height=page_size[1] if page_size else None,
            coordinate_origin=(
                CoordinateOrigin.TOP_LEFT
                if coordinate_origin is None
                or coordinate_origin.casefold() in {"top_left", "topleft"}
                else CoordinateOrigin.BOTTOM_LEFT
            ),
        )
    metadata: dict[str, object] = {}
    if coordinate_origin is not None:
        metadata["coordinate_origin"] = coordinate_origin
    page_ref = PageRef(DOCUMENT_VERSION, page)
    return ParserBlockObservation(
        document_version_id=DOCUMENT_VERSION,
        page_id=page_ref.page_id,
        page_index=page,
        provenance=ParserProvenance(
            parser_name=parser,
            parser_version="1.0",
            model_version=f"{parser}-model",
            config_sha256=content_sha256({"parser": parser}),
            parser_native_id=native_id,
            raw_artifact_uri=f"artifact://fixture/{parser}/{native_id}.json",
            transformations=(f"{parser}_to_observation.v1",),
        ),
        block_type=block_type,
        representations=representations,
        geometry=geometry,
        reading_order=reading_order,
        confidence=confidence,
        assets=assets,
        metadata=metadata,
    )


def source_sets(result) -> set[frozenset[str]]:
    return {
        frozenset(block.source_observation_ids) for block in result.canonical_blocks
    }


class CountingCandidateBucket:
    """Sequence-like probe counter for a single dense spatial-index bucket."""

    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.yield_count = 0

    def __iter__(self):
        for value in self.values:
            self.yield_count += 1
            yield value

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


class GeometryAndConflictMergeGates(unittest.TestCase):
    def test_bottom_left_docling_and_top_left_paddle_same_region_merge(self):
        # Both boxes identify 10%-20% from the top of the same 1000x1000 page.
        docling = observation(
            "docling",
            "top-region",
            bbox=(100, 800, 700, 900),
            coordinate_space=CoordinateSpace.PAGE_POINTS,
            page_size=(1000, 1000),
            coordinate_origin="BOTTOMLEFT",
        )
        paddle = observation(
            "paddleocr-vl",
            "top-region",
            bbox=(100, 100, 700, 200),
            coordinate_space=CoordinateSpace.PAGE_PIXELS,
            page_size=(1000, 1000),
            coordinate_origin="TOPLEFT",
        )

        result = reconcile_observations((docling, paddle))

        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(
            {docling.observation_id, paddle.observation_id},
            set(result.canonical_blocks[0].source_observation_ids),
        )
        self.assertEqual(1, result.cluster_merge_count)

    def test_contradictory_numeric_text_is_contested_not_accepted(self):
        docling = observation(
            "docling", "revenue", text="Revenue was USD 10 million", confidence=0.99
        )
        mineru = observation(
            "mineru", "revenue", text="Revenue was USD 90 million", confidence=0.99
        )

        result = reconcile_observations((docling, mineru))

        self.assertEqual(1, len(result.canonical_blocks))
        block = result.canonical_blocks[0]
        self.assertEqual(QualityStatus.NEEDS_REVIEW, block.quality_status)
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            set(block.source_observation_ids),
        )
        self.assertIn(
            ReconciliationIssueCode.CONFLICTING_OBSERVATION,
            {issue.code for issue in result.issues},
        )

    def test_conflicting_table_grids_are_contested(self):
        docling = observation(
            "docling",
            "table",
            block_type=BlockType.TABLE,
            text=None,
            representations=(
                representation(
                    {"rows": [["Metric", "Value"], ["Revenue", "10"]]},
                    kind=RepresentationKind.TABLE_GRID,
                    media_type="application/json",
                    confidence=0.98,
                ),
            ),
            confidence=0.98,
        )
        mineru = observation(
            "mineru",
            "table",
            block_type=BlockType.TABLE,
            text=None,
            representations=(
                representation(
                    {"rows": [["Metric", "Value"], ["Revenue", "90"]]},
                    kind=RepresentationKind.TABLE_GRID,
                    media_type="application/json",
                    confidence=0.97,
                ),
            ),
            confidence=0.97,
        )

        result = reconcile_observations((docling, mineru))

        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(
            QualityStatus.NEEDS_REVIEW,
            result.canonical_blocks[0].quality_status,
        )
        self.assertIn(
            ReconciliationIssueCode.CONFLICTING_OBSERVATION,
            {issue.code for issue in result.issues},
        )

    def test_conflicting_formulae_are_contested(self):
        docling = observation(
            "docling",
            "formula",
            block_type=BlockType.EQUATION,
            text=None,
            representations=(
                representation(
                    r"E=mc^2",
                    kind=RepresentationKind.LATEX,
                    media_type="application/x-latex",
                    confidence=0.99,
                ),
            ),
            confidence=0.99,
        )
        mineru = observation(
            "mineru",
            "formula",
            block_type=BlockType.EQUATION,
            text=None,
            representations=(
                representation(
                    r"E=mc^3",
                    kind=RepresentationKind.LATEX,
                    media_type="application/x-latex",
                    confidence=0.99,
                ),
            ),
            confidence=0.99,
        )

        result = reconcile_observations((docling, mineru))

        self.assertEqual(1, len(result.canonical_blocks))
        self.assertEqual(
            QualityStatus.NEEDS_REVIEW,
            result.canonical_blocks[0].quality_status,
        )
        self.assertIn(
            ReconciliationIssueCode.CONFLICTING_OBSERVATION,
            {issue.code for issue in result.issues},
        )


class IncrementalIdentityMergeGates(unittest.TestCase):
    def test_prior_canonical_retains_block_id_and_advances_revision_on_change(self):
        mineru = observation("mineru", "region-1")
        initial = reconcile_observations((mineru,))
        prior = initial.canonical_blocks[0]

        replay = reconcile_observations(
            (mineru,), prior_blocks=initial.canonical_blocks
        )
        replayed = replay.canonical_blocks[0]
        self.assertEqual(prior.block_id, replayed.block_id)
        self.assertEqual(prior.revision, replayed.revision)
        self.assertEqual(prior.revision_id, replayed.revision_id)

        docling = observation("docling", "region-1")
        expanded = reconcile_observations(
            (mineru, docling), prior_blocks=initial.canonical_blocks
        )
        revised = expanded.canonical_blocks[0]
        self.assertEqual(prior.block_id, revised.block_id)
        self.assertEqual(prior.canonical_key, revised.canonical_key)
        self.assertEqual(prior.revision + 1, revised.revision)
        self.assertNotEqual(prior.revision_id, revised.revision_id)

    def test_exact_source_identity_survives_geometry_and_content_correction(self):
        original = observation(
            "mineru",
            "stable-source-region",
            text="uncorrected value",
            bbox=(0.10, 0.10, 0.30, 0.20),
        )
        prior = reconcile_observations((original,)).canonical_blocks[0]
        corrected = observation(
            "mineru",
            "stable-source-region",
            text="corrected value",
            bbox=(0.70, 0.70, 0.90, 0.80),
        )
        self.assertEqual(original.observation_id, corrected.observation_id)

        result = reconcile_observations((corrected,), prior_blocks=(prior,))
        revised = result.canonical_blocks[0]

        self.assertEqual(prior.block_id, revised.block_id)
        self.assertEqual(prior.canonical_key, revised.canonical_key)
        self.assertEqual(prior.revision + 1, revised.revision)

    def test_unmatched_current_block_cannot_hijack_a_reserved_prior_key(self):
        current_observation = observation(
            "mineru",
            "new-region",
            text="current content",
            bbox=(0.10, 0.10, 0.30, 0.20),
        )
        current_without_prior = reconcile_observations(
            (current_observation,)
        ).canonical_blocks[0]
        unrelated_prior = reconcile_observations(
            (
                observation(
                    "docling",
                    "retired-region",
                    text="unrelated historical content",
                    bbox=(0.70, 0.70, 0.90, 0.80),
                ),
            )
        ).canonical_blocks[0]
        reserved_prior = replace(
            unrelated_prior,
            canonical_key=current_without_prior.canonical_key,
            revision=7,
        )

        result = reconcile_observations(
            (current_observation,), prior_blocks=(reserved_prior,)
        )
        current = result.canonical_blocks[0]

        self.assertNotEqual(reserved_prior.canonical_key, current.canonical_key)
        self.assertNotEqual(reserved_prior.block_id, current.block_id)
        self.assertEqual(1, current.revision)
        self.assertIn(
            ReconciliationIssueCode.CANONICAL_IDENTITY_COLLISION,
            {issue.code for issue in result.issues},
        )
        self.assertIn(
            ReconciliationIssueCode.CANONICAL_IDENTITY_COLLISION,
            {issue.code for issue in result.issues},
        )

    def test_prior_identity_probes_are_page_local_not_quadratic(self):
        page_count = 240
        initial_observations = tuple(
            observation("docling", f"page-{page}", page=page)
            for page in range(page_count)
        )
        initial = reconcile_observations(initial_observations)

        with mock.patch(
            "document_services.parsing.reconciliation._prior_match_score",
            wraps=reconciliation_module._prior_match_score,
        ) as scorer:
            replay = reconcile_observations(
                initial_observations,
                prior_blocks=reversed(initial.canonical_blocks),
            )

        self.assertEqual(page_count, len(replay.canonical_blocks))
        self.assertLessEqual(scorer.call_count, page_count)
        self.assertEqual(
            {block.block_id for block in initial.canonical_blocks},
            {block.block_id for block in replay.canonical_blocks},
        )

    def test_crowded_prior_identity_candidates_fail_closed(self):
        priors = reconcile_observations(
            tuple(
                observation("docling", f"prior-{index}", text=f"prior {index}")
                for index in range(5)
            )
        ).canonical_blocks
        current = observation("mineru", "current", text="new value")

        result = reconcile_observations(
            (current,),
            prior_blocks=priors,
            policy=ReconciliationPolicy(max_candidates_per_observation=2),
        )

        self.assertIn(
            ReconciliationIssueCode.PRIOR_IDENTITY_CANDIDATE_LIMIT,
            {issue.code for issue in result.issues},
        )
        self.assertNotIn(
            result.canonical_blocks[0].canonical_key,
            {block.canonical_key for block in priors},
        )

    def test_repeated_prior_text_in_distinct_regions_does_not_hide_spatial_identity(self):
        prior_result = reconcile_observations(
            (
                observation(
                    "docling",
                    "left",
                    text="N/A",
                    bbox=(0.02, 0.02, 0.04, 0.04),
                    reading_order=1.0,
                ),
                observation(
                    "docling",
                    "middle",
                    text="N/A",
                    bbox=(0.45, 0.45, 0.47, 0.47),
                    reading_order=2.0,
                ),
                observation(
                    "docling",
                    "right",
                    text="N/A",
                    bbox=(0.85, 0.85, 0.87, 0.87),
                    reading_order=3.0,
                ),
            )
        )
        middle_prior = next(
            block for block in prior_result.canonical_blocks if block.reading_order == 2.0
        )
        current = observation(
            "mineru",
            "middle-new-parser",
            text="N/A",
            bbox=(0.45, 0.45, 0.47, 0.47),
            reading_order=2.0,
        )

        result = reconcile_observations(
            (current,),
            prior_blocks=prior_result.canonical_blocks,
            policy=ReconciliationPolicy(max_candidates_per_observation=1),
        )

        revised = result.canonical_blocks[0]
        self.assertEqual(middle_prior.block_id, revised.block_id)
        self.assertEqual(middle_prior.revision + 1, revised.revision)
        self.assertNotIn(
            ReconciliationIssueCode.PRIOR_IDENTITY_CANDIDATE_LIMIT,
            {issue.code for issue in result.issues},
        )

    def test_tied_prior_identity_candidates_do_not_depend_on_order(self):
        priors = reconcile_observations(
            (
                observation("docling", "left", text="alpha"),
                observation("docling", "right", text="beta"),
            )
        ).canonical_blocks
        current = observation("mineru", "current", text="gamma")

        forward = reconcile_observations((current,), prior_blocks=priors)
        reverse = reconcile_observations((current,), prior_blocks=reversed(priors))

        self.assertEqual(forward.to_dict(), reverse.to_dict())
        self.assertIn(
            ReconciliationIssueCode.AMBIGUOUS_PRIOR_IDENTITY,
            {issue.code for issue in forward.issues},
        )
        self.assertNotIn(
            forward.canonical_blocks[0].canonical_key,
            {block.canonical_key for block in priors},
        )

    def test_fuzzy_one_to_one_competition_never_churns_silently(self):
        priors = reconcile_observations(
            (
                observation(
                    "docling", "prior-primary", bbox=(0.10, 0.10, 0.50, 0.20)
                ),
                observation(
                    "docling", "prior-alternative", bbox=(0.40, 0.10, 0.80, 0.20)
                ),
            )
        ).canonical_blocks
        current = (
            observation(
                "mineru", "current-strong", bbox=(0.10, 0.10, 0.50, 0.20)
            ),
            observation(
                "mineru", "current-contender", bbox=(0.10, 0.10, 0.45, 0.20)
            ),
        )

        result = reconcile_observations(
            current,
            prior_blocks=priors,
            policy=ReconciliationPolicy(minimum_iou=0.14),
        )

        self.assertEqual(2, len(result.canonical_blocks))
        self.assertIn(
            ReconciliationIssueCode.AMBIGUOUS_PRIOR_IDENTITY,
            {issue.code for issue in result.issues},
        )
        self.assertTrue(
            any(
                "fuzzy_candidate_lost_one_to_one_assignment"
                in issue.metadata.get("reasons", ())
                for issue in result.issues
                if issue.code is ReconciliationIssueCode.AMBIGUOUS_PRIOR_IDENTITY
            )
        )


class CandidateBoundMergeGates(unittest.TestCase):
    def test_dense_spatial_bucket_has_a_hard_probe_bound_and_fails_closed(self):
        limit = 4
        bbox = (0.10, 0.10, 0.11, 0.11)
        priors = tuple(
            observation("docling", f"dense-prior-{index}", bbox=bbox)
            for index in range(64)
        )
        current = observation("mineru", "dense-current", bbox=bbox)
        ordered = (*priors, current)
        bucket = CountingCandidateBucket(tuple(range(len(priors))))
        policy = ReconciliationPolicy(
            grid_size=16,
            max_candidates_per_observation=limit,
        )
        cells = reconciliation_module._grid_cells(bbox, policy.grid_size)
        self.assertEqual(1, len(cells))

        with mock.patch(
            "document_services.parsing.reconciliation._iou",
            wraps=reconciliation_module._iou,
        ) as iou:
            selection = reconciliation_module._bounded_candidates(
                len(priors),
                ordered=ordered,
                normalized_boxes=(bbox,) * len(ordered),
                normalized_texts=("dense content",) * len(ordered),
                grid={(*cells[0], "text"): {"docling": bucket}},
                exact_text={},
                geometry_free_text={},
                policy=policy,
            )

        # K+1 plausible candidates prove overflow.  Scanning the remainder is
        # quadratic page work and retaining an arbitrary K corrupts lineage.
        self.assertGreater(selection.plausible_count_lower_bound, limit)
        self.assertEqual((), selection.entries)
        self.assertLessEqual(bucket.yield_count, limit + 1)
        self.assertLessEqual(iou.call_count, limit + 1)

    def test_exact_text_budget_does_not_starve_best_spatial_match(self):
        wrong_region = observation(
            "docling",
            "wrong-region",
            text="N/A",
            bbox=(0.05, 0.05, 0.15, 0.10),
            reading_order=1.0,
        )
        correct_region = observation(
            "paddleocr-vl",
            "correct-region",
            text="N/A",
            bbox=(0.70, 0.70, 0.90, 0.80),
            reading_order=2.0,
        )
        current = observation(
            "mineru",
            "correct-region",
            text="N/A",
            bbox=(0.70, 0.70, 0.90, 0.80),
            reading_order=3.0,
        )

        result = reconcile_observations(
            (wrong_region, correct_region, current),
            policy=ReconciliationPolicy(max_candidates_per_observation=1),
        )

        self.assertIn(
            frozenset({correct_region.observation_id, current.observation_id}),
            source_sets(result),
        )
        self.assertIn(
            frozenset({wrong_region.observation_id}), source_sets(result)
        )

    def test_same_parser_siblings_do_not_consume_candidate_budget(self):
        siblings = tuple(
            observation("docling", f"sibling-{index}") for index in range(4)
        )

        result = reconcile_observations(
            siblings,
            policy=ReconciliationPolicy(max_candidates_per_observation=1),
        )

        self.assertEqual(4, len(result.canonical_blocks))
        self.assertEqual(0, result.candidate_pair_count)
        self.assertEqual(0, result.matched_pair_count)
        self.assertNotIn(
            ReconciliationIssueCode.CANDIDATE_LIMIT,
            {issue.code for issue in result.issues},
        )
        self.assertEqual(
            {QualityStatus.ACCEPTED},
            {block.quality_status for block in result.canonical_blocks},
        )

    def test_repeated_geometric_text_in_distinct_regions_is_not_candidate_overflow(self):
        values = (
            observation(
                "docling",
                "top-left",
                text="N/A",
                bbox=(0.02, 0.02, 0.04, 0.04),
            ),
            observation(
                "mineru",
                "middle",
                text="N/A",
                bbox=(0.45, 0.45, 0.47, 0.47),
            ),
            observation(
                "paddleocr-vl",
                "bottom-right",
                text="N/A",
                bbox=(0.85, 0.85, 0.87, 0.87),
            ),
        )

        result = reconcile_observations(
            values,
            policy=ReconciliationPolicy(max_candidates_per_observation=1),
        )

        self.assertEqual(3, len(result.canonical_blocks))
        self.assertNotIn(
            ReconciliationIssueCode.CANDIDATE_LIMIT,
            {issue.code for issue in result.issues},
        )
        self.assertEqual(
            {QualityStatus.ACCEPTED},
            {block.quality_status for block in result.canonical_blocks},
        )

    def test_repeated_geometry_free_text_is_left_ambiguous(self):
        values = (
            observation("docling", "left", text="Approved", bbox=None),
            observation("docling", "right", text="Approved", bbox=None),
            observation("mineru", "left", text="Approved", bbox=None),
            observation("mineru", "right", text="Approved", bbox=None),
        )

        result = reconcile_observations(values)

        self.assertEqual(4, len(result.canonical_blocks))
        self.assertEqual(
            {QualityStatus.NEEDS_REVIEW},
            {block.quality_status for block in result.canonical_blocks},
        )
        self.assertTrue(result.issues)
        self.assertTrue(
            any("ambiguous" in issue.message.casefold() for issue in result.issues)
        )

    def test_geometry_free_text_does_not_choose_between_repeated_regions(self):
        values = (
            observation(
                "paddleocr-vl",
                "left",
                text="Approved",
                bbox=(0.10, 0.10, 0.30, 0.20),
            ),
            observation(
                "paddleocr-vl",
                "right",
                text="Approved",
                bbox=(0.70, 0.10, 0.90, 0.20),
            ),
            observation("mineru", "unknown-region", text="Approved", bbox=None),
        )

        result = reconcile_observations(values)

        self.assertEqual(3, len(result.canonical_blocks))
        self.assertEqual(
            {QualityStatus.NEEDS_REVIEW},
            {block.quality_status for block in result.canonical_blocks},
        )
        self.assertIn(
            ReconciliationIssueCode.AMBIGUOUS_GEOMETRY_FREE,
            {issue.code for issue in result.issues},
        )


class QualityAndLineageMergeGates(unittest.TestCase):
    def test_unscored_single_observation_is_degraded(self):
        unscored = observation("docling", "unscored", confidence=None)

        result = reconcile_observations((unscored,))

        self.assertEqual(
            QualityStatus.DEGRADED,
            result.canonical_blocks[0].quality_status,
        )
        self.assertIn(
            ReconciliationIssueCode.LOW_CONFIDENCE,
            {issue.code for issue in result.issues},
        )

    def test_representation_and_asset_lineage_names_every_source(self):
        shared_text = representation(
            "Revenue 10",
            kind=RepresentationKind.OCR_TEXT,
            confidence=0.95,
        )
        table_grid = representation(
            {"rows": [["Revenue", "10"]]},
            kind=RepresentationKind.TABLE_GRID,
            media_type="application/json",
            confidence=0.96,
        )
        shared_crop = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri="artifact://fixture/shared-table.png",
            media_type="image/png",
            sha256="a" * 64,
        )
        docling = observation(
            "docling",
            "table",
            block_type=BlockType.TABLE,
            text=None,
            representations=(shared_text,),
            assets=(shared_crop,),
            confidence=0.95,
        )
        mineru = observation(
            "mineru",
            "table",
            block_type=BlockType.TABLE,
            text=None,
            representations=(shared_text, table_grid),
            assets=(shared_crop,),
            confidence=0.96,
        )

        result = reconcile_observations((docling, mineru))

        self.assertEqual(1, len(result.canonical_blocks))
        block = result.canonical_blocks[0]
        representation_sources = block.metadata.get("representation_sources")
        asset_sources = block.metadata.get("asset_sources")
        self.assertIsInstance(representation_sources, Mapping)
        self.assertIsInstance(asset_sources, Mapping)
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            set(representation_sources[shared_text.representation_id]),
        )
        self.assertEqual(
            {mineru.observation_id},
            set(representation_sources[table_grid.representation_id]),
        )
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            set(asset_sources[shared_crop.asset_id]),
        )


class StrictResultValidationMergeGates(unittest.TestCase):
    def setUp(self):
        self.base = reconcile_observations((observation("docling", "base"),))

    def _construct_with_counts(
        self,
        *,
        candidate_pair_count: object,
        matched_pair_count: object,
        cluster_merge_count: object,
    ) -> ReconciliationResult:
        return ReconciliationResult(
            document_version_id=self.base.document_version_id,
            policy_sha256=self.base.policy_sha256,
            canonical_blocks=self.base.canonical_blocks,
            issues=self.base.issues,
            candidate_pair_count=candidate_pair_count,  # type: ignore[arg-type]
            matched_pair_count=matched_pair_count,  # type: ignore[arg-type]
            cluster_merge_count=cluster_merge_count,  # type: ignore[arg-type]
        )

    def test_boolean_and_float_counters_are_rejected(self):
        invalid_counts = (
            (False, 0, 0),
            (1.5, 0, 0),
            (1, False, 0),
            (1, 0.5, 0),
            (1, 1, False),
            (1, 1, 0.5),
        )
        for candidate_count, matched_count, merge_count in invalid_counts:
            with self.subTest(
                candidate=candidate_count,
                matched=matched_count,
                merged=merge_count,
            ):
                with self.assertRaises(ContractValidationError):
                    self._construct_with_counts(
                        candidate_pair_count=candidate_count,
                        matched_pair_count=matched_count,
                        cluster_merge_count=merge_count,
                    )

    def test_forged_json_hash_does_not_bypass_counter_type_validation(self):
        payload = self.base.to_dict()
        payload["candidate_pair_count"] = False
        payload["matched_pair_count"] = False
        payload["cluster_merge_count"] = False
        payload["result_id"] = stable_id(
            "reconciliation",
            self.base.document_version_id,
            self.base.policy_sha256,
            [block.revision_id for block in self.base.canonical_blocks],
            [issue.issue_id for issue in self.base.issues],
            False,
            False,
            False,
        )

        with self.assertRaises(ContractValidationError):
            ReconciliationResult.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
