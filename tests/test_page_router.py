from __future__ import annotations

import subprocess
import sys
import unittest

from docarag.domain import (
    AcceleratorKind,
    BlockType,
    ContractValidationError,
    PageRef,
    ParserCapabilityManifest,
    ParserModality,
)
from docarag.ports import ParserRegistry
from docarag.routing import (
    AdapterRoutingProfile,
    DeterministicPageRouter,
    NoCapableParserError,
    PageClass,
    PageFeatures,
    PageRoutingPolicy,
    RouteDecision,
    RouteReasonCode,
    SecondaryRouteRole,
)


ALL_BLOCK_TYPES = (
    BlockType.PARAGRAPH,
    BlockType.TABLE,
    BlockType.EQUATION,
    BlockType.FIGURE,
    BlockType.CHART,
    BlockType.DIAGRAM,
)


def manifest(
    adapter_name: str,
    parser_name: str,
    modalities: tuple[ParserModality, ...],
    *,
    mime_types: tuple[str, ...] = ("application/pdf",),
    block_types: tuple[BlockType, ...] = ALL_BLOCK_TYPES,
    supports_page_selection: bool = True,
) -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name=adapter_name,
        adapter_version="1.0.0",
        parser_name=parser_name,
        parser_version="2.0.0",
        model_version="fixture-v1",
        supported_mime_types=mime_types,
        modalities=modalities,
        block_types=block_types,
        accelerator=AcceleratorKind.CPU,
        supports_page_selection=supports_page_selection,
        deterministic=True,
    )


class NeverParseAdapter:
    def __init__(self, value: ParserCapabilityManifest) -> None:
        self._manifest = value
        self.parse_calls = 0

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, _request):
        self.parse_calls += 1
        raise AssertionError("the route planner must not invoke parser runtimes")


def adapters() -> tuple[NeverParseAdapter, ...]:
    return (
        NeverParseAdapter(
            manifest(
                "docling",
                "docling",
                (
                    ParserModality.NATIVE_TEXT,
                    ParserModality.LAYOUT,
                    ParserModality.TABLE,
                    ParserModality.FORMULA,
                    ParserModality.IMAGE,
                ),
            )
        ),
        NeverParseAdapter(
            manifest(
                "mineru",
                "mineru",
                (
                    ParserModality.NATIVE_TEXT,
                    ParserModality.OCR_TEXT,
                    ParserModality.LAYOUT,
                    ParserModality.TABLE,
                    ParserModality.FORMULA,
                    ParserModality.IMAGE,
                    ParserModality.CHART,
                    ParserModality.DIAGRAM,
                ),
            )
        ),
        NeverParseAdapter(
            manifest(
                "paddleocr-vl",
                "paddleocr-vl",
                (
                    ParserModality.OCR_TEXT,
                    ParserModality.LAYOUT,
                    ParserModality.TABLE,
                    ParserModality.FORMULA,
                    ParserModality.IMAGE,
                    ParserModality.CHART,
                    ParserModality.DIAGRAM,
                ),
            )
        ),
    )


def profiles() -> tuple[AdapterRoutingProfile, ...]:
    return (
        AdapterRoutingProfile(
            "docling",
            native_text_rank=0,
            technical_rank=1,
            visual_rank=2,
            ambiguous_rank=1,
            estimated_latency_ms=100,
            estimated_cost_microusd=10,
        ),
        AdapterRoutingProfile(
            "mineru",
            native_text_rank=1,
            technical_rank=0,
            visual_rank=1,
            ambiguous_rank=0,
            estimated_latency_ms=200,
            estimated_cost_microusd=20,
            heavy=True,
        ),
        AdapterRoutingProfile(
            "paddleocr-vl",
            native_text_rank=2,
            technical_rank=2,
            visual_rank=0,
            ambiguous_rank=1,
            estimated_latency_ms=300,
            estimated_cost_microusd=30,
            heavy=True,
        ),
    )


def policy(**overrides) -> PageRoutingPolicy:
    values = {
        "policy_name": "fixture-policy",
        "policy_version": "1.0.0",
        "adapter_profiles": profiles(),
    }
    values.update(overrides)
    return PageRoutingPolicy(**values)


def complete_stats(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "native_character_count": 1200,
        "native_word_count": 180,
        "text_coverage_ratio": 0.55,
        "raster_image_coverage_ratio": 0.05,
        "replacement_character_ratio": 0.0,
        "control_character_ratio": 0.0,
        "duplicated_text_ratio": 0.01,
        "ocr_native_disagreement_ratio": 0.01,
        "connected_component_density": 0.0,
        "font_count": 4,
        "vector_path_density": 0.0,
        "skew_degrees": 0.0,
        "candidate_table_regions": 0,
        "candidate_equation_regions": 0,
        "candidate_figure_regions": 0,
        "candidate_chart_regions": 0,
        "candidate_diagram_regions": 0,
        "column_count": 1,
        "is_scanned_page": False,
        "is_slide_like": False,
        "is_photographed_page": False,
    }
    values.update(overrides)
    return values


def page(**stats) -> PageRef:
    return PageRef(
        "docv_route_fixture",
        0,
        width=612,
        height=792,
        native_stats=stats,
    )


def router(
    *,
    adapter_values: tuple[NeverParseAdapter, ...] | None = None,
    route_policy: PageRoutingPolicy | None = None,
) -> DeterministicPageRouter:
    return DeterministicPageRouter(
        ParserRegistry(adapter_values or adapters()), route_policy or policy()
    )


class PageFeatureContractTests(unittest.TestCase):
    def test_from_page_ref_preserves_unknowns_and_unmapped_stats(self):
        source = page(unmapped_counter=7)
        features = PageFeatures.from_page_ref(source, "application/pdf")

        self.assertIsNone(features.native_character_count)
        self.assertIsNone(features.is_scanned_page)
        self.assertEqual(
            7, features.metadata["unmapped_native_stats"]["unmapped_counter"]
        )
        self.assertEqual(features, PageFeatures.from_json(features.to_json()))

    def test_explicit_overrides_win_and_invalid_probe_values_are_rejected(self):
        source = page(**complete_stats(native_character_count=5))
        features = PageFeatures.from_page_ref(
            source,
            "application/pdf",
            native_character_count=99,
            metadata={"probe_revision": "fixture-v2"},
        )
        self.assertEqual(99, features.native_character_count)
        self.assertEqual("fixture-v2", features.metadata["probe_revision"])

        with self.assertRaisesRegex(ContractValidationError, "unknown field"):
            PageFeatures.from_page_ref(source, "application/pdf", typo_count=1)
        with self.assertRaisesRegex(ContractValidationError, "between 0 and 1"):
            PageFeatures.from_page_ref(
                source, "application/pdf", text_coverage_ratio=1.1
            )

    def test_feature_profile_and_policy_round_trip_and_detect_tampering(self):
        features = PageFeatures.from_page_ref(
            page(**complete_stats()), "application/pdf"
        )
        route_policy = policy()
        self.assertEqual(features, PageFeatures.from_json(features.to_json()))
        for profile in route_policy.adapter_profiles:
            self.assertEqual(
                profile, AdapterRoutingProfile.from_json(profile.to_json())
            )
        self.assertEqual(
            route_policy, PageRoutingPolicy.from_json(route_policy.to_json())
        )

        tampered = features.to_dict()
        tampered["feature_id"] = "pagefeat_tampered"
        with self.assertRaisesRegex(ContractValidationError, "feature_id"):
            PageFeatures.from_dict(tampered)


class DeterministicPageRouterTests(unittest.TestCase):
    def test_incomplete_native_stats_routes_ambiguous_without_type_error(self):
        features = PageFeatures.from_page_ref(page(), "application/pdf")
        decision = router().route(features)

        self.assertEqual(PageClass.AMBIGUOUS, decision.page_class)
        self.assertIn(RouteReasonCode.PROBE_FEATURES_INCOMPLETE, decision.reason_codes)
        self.assertEqual(SecondaryRouteRole.ADJUDICATOR, decision.secondary_role)

    def test_complete_core_signals_do_not_require_optional_probe_hints(self):
        core_stats = {
            "native_character_count": 1200,
            "native_word_count": 180,
            "text_coverage_ratio": 0.55,
            "raster_image_coverage_ratio": 0.05,
            "replacement_character_ratio": 0.0,
            "control_character_ratio": 0.0,
            "duplicated_text_ratio": 0.01,
        }
        features = PageFeatures.from_page_ref(
            page(**core_stats), "application/pdf"
        )
        decision = router().route(features)

        self.assertIsNone(features.candidate_table_regions)
        self.assertIsNone(features.is_scanned_page)
        self.assertEqual(PageClass.NATIVE_TEXT, decision.page_class)
        self.assertNotIn(
            RouteReasonCode.PROBE_FEATURES_INCOMPLETE, decision.reason_codes
        )

        known_table = PageFeatures.from_page_ref(
            page(**core_stats),
            "application/pdf",
            candidate_table_regions=1,
        )
        self.assertEqual(PageClass.TECHNICAL, router().route(known_table).page_class)

    def test_native_technical_and_visual_pages_choose_specialized_primaries(self):
        cases = (
            (
                complete_stats(),
                PageClass.NATIVE_TEXT,
                "docling",
                SecondaryRouteRole.FALLBACK,
            ),
            (
                complete_stats(candidate_table_regions=2, candidate_equation_regions=1),
                PageClass.TECHNICAL,
                "mineru",
                SecondaryRouteRole.FALLBACK,
            ),
            (
                complete_stats(
                    native_character_count=10,
                    native_word_count=2,
                    text_coverage_ratio=0.01,
                    raster_image_coverage_ratio=0.92,
                    is_slide_like=True,
                ),
                PageClass.VISUAL,
                "paddleocr-vl",
                SecondaryRouteRole.FALLBACK,
            ),
        )
        for stats, page_class, primary, secondary_role in cases:
            with self.subTest(page_class=page_class):
                decision = router().route(
                    PageFeatures.from_page_ref(page(**stats), "application/pdf")
                )
                self.assertEqual(page_class, decision.page_class)
                self.assertEqual(primary, decision.primary_adapter_name)
                self.assertEqual(secondary_role, decision.secondary_role)

    def test_disagreement_plans_one_adjudicator_but_never_calls_any_parser(self):
        adapter_values = adapters()
        features = PageFeatures.from_page_ref(
            page(**complete_stats(ocr_native_disagreement_ratio=0.80)),
            "application/pdf",
        )
        decision = router(adapter_values=adapter_values).route(features)

        self.assertEqual(PageClass.AMBIGUOUS, decision.page_class)
        self.assertEqual(SecondaryRouteRole.ADJUDICATOR, decision.secondary_role)
        self.assertLessEqual(len(decision.planned_adapter_names), 2)
        self.assertEqual(3, len(decision.candidates))
        self.assertEqual([0, 0, 0], [item.parse_calls for item in adapter_values])
        self.assertIn(RouteReasonCode.ADJUDICATOR_PLANNED, decision.reason_codes)

    def test_structure_requirements_filter_incapable_adapters_from_manifest(self):
        features = PageFeatures.from_page_ref(
            page(**complete_stats(candidate_diagram_regions=1)), "application/pdf"
        )
        decision = router().route(features)
        by_name = {item.adapter_name: item for item in decision.candidates}

        self.assertEqual("mineru", decision.primary_adapter_name)
        self.assertFalse(by_name["docling"].eligible)
        self.assertIn(
            RouteReasonCode.REQUIRED_MODALITY_UNSUPPORTED,
            by_name["docling"].rejection_reason_codes,
        )
        self.assertIn(ParserModality.DIAGRAM, decision.required_modalities)
        self.assertIn(BlockType.DIAGRAM, decision.required_block_types)

    def test_primary_and_total_budgets_gate_work_without_hidden_relaxation(self):
        features = PageFeatures.from_page_ref(
            page(**complete_stats()), "application/pdf"
        )
        route_policy = policy(
            max_primary_cost_microusd=15,
            max_total_cost_microusd=15,
        )
        decision = router(route_policy=route_policy).route(features)
        by_name = {item.adapter_name: item for item in decision.candidates}

        self.assertEqual(("docling",), decision.planned_adapter_names)
        self.assertIn(
            RouteReasonCode.PRIMARY_COST_BUDGET_EXCEEDED,
            by_name["mineru"].rejection_reason_codes,
        )
        self.assertIn(RouteReasonCode.SINGLE_CAPABLE_ADAPTER, decision.reason_codes)

        with self.assertRaises(NoCapableParserError) as raised:
            router(
                route_policy=policy(
                    max_primary_latency_ms=50,
                    max_total_latency_ms=50,
                )
            ).route(features)
        self.assertEqual(features.page.page_id, raised.exception.page_id)
        self.assertIn("primary_latency_budget_exceeded", str(raised.exception))

    def test_total_budget_can_suppress_secondary_that_is_primary_eligible(self):
        features = PageFeatures.from_page_ref(
            page(**complete_stats()), "application/pdf"
        )
        decision = router(
            route_policy=policy(max_total_cost_microusd=25)
        ).route(features)

        self.assertEqual("docling", decision.primary_adapter_name)
        self.assertIsNone(decision.secondary_adapter_name)
        self.assertIn(
            RouteReasonCode.SECONDARY_COST_BUDGET_EXCEEDED,
            decision.reason_codes,
        )

    def test_explicit_no_capable_error_audits_registration_and_mime_failures(self):
        only_docling = adapters()[:1]
        route_policy = policy(
            adapter_profiles=(
                profiles()[0],
                AdapterRoutingProfile(
                    "not-running",
                    0,
                    0,
                    0,
                    0,
                    estimated_latency_ms=1,
                    estimated_cost_microusd=1,
                ),
            )
        )
        features = PageFeatures.from_page_ref(
            page(**complete_stats()), "image/tiff"
        )
        with self.assertRaises(NoCapableParserError) as raised:
            router(
                adapter_values=only_docling, route_policy=route_policy
            ).route(features)
        by_name = {item.adapter_name: item for item in raised.exception.candidates}
        self.assertIn(
            RouteReasonCode.MIME_TYPE_UNSUPPORTED,
            by_name["docling"].rejection_reason_codes,
        )
        self.assertIn(
            RouteReasonCode.ADAPTER_NOT_REGISTERED,
            by_name["not-running"].rejection_reason_codes,
        )

    def test_ties_and_decision_ids_are_stable_across_registration_order(self):
        tied_profiles = tuple(
            AdapterRoutingProfile(
                item.adapter_name,
                native_text_rank=0,
                technical_rank=0,
                visual_rank=0,
                ambiguous_rank=0,
                estimated_latency_ms=100,
                estimated_cost_microusd=10,
            )
            for item in profiles()
        )
        route_policy = policy(adapter_profiles=tied_profiles)
        features = PageFeatures.from_page_ref(
            page(**complete_stats()), "application/pdf"
        )
        forward = router(route_policy=route_policy).route(features)
        reverse = router(
            adapter_values=tuple(reversed(adapters())), route_policy=route_policy
        ).route(features)

        self.assertEqual("docling", forward.primary_adapter_name)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.decision_id, reverse.decision_id)
        self.assertEqual(forward, RouteDecision.from_json(forward.to_json()))

        tampered = forward.to_dict()
        tampered["decision_id"] = "route_tampered"
        with self.assertRaisesRegex(ContractValidationError, "decision_id"):
            RouteDecision.from_dict(tampered)

    def test_import_is_free_of_vendor_model_and_bookrag_core_modules(self):
        script = """
import json, sys
import docarag.routing
forbidden = ('Core', 'torch', 'mineru', 'paddle', 'paddleocr', 'docling', 'transformers', 'vllm')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual([], __import__("json").loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
