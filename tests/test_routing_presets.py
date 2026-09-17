from __future__ import annotations

import json
import subprocess
import sys
import unittest

from document_services.parser_adapters.docling import DoclingExportAdapter
from document_services.parser_adapters.mineru_compat import MinerUNormalizedAdapter
from document_services.parser_adapters.paddleocr_vl import PaddleOCRVLPageAdapter
from document_services.domain import ContractValidationError, PageRef
from document_services.ports import ParserRegistry
from document_services.parsing.routing_presets import (
    BALANCED_PDF_PRESET,
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
    balanced_pdf_routing_policy,
    routing_policy_preset,
)
from document_services.parsing.routing import DeterministicPageRouter, PageFeatures


def stats(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "native_character_count": 1000,
        "native_word_count": 150,
        "text_coverage_ratio": 0.50,
        "raster_image_coverage_ratio": 0.05,
        "replacement_character_ratio": 0.0,
        "control_character_ratio": 0.0,
        "duplicated_text_ratio": 0.0,
        "ocr_native_disagreement_ratio": 0.0,
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


def features(**overrides: object) -> PageFeatures:
    page = PageRef("docv_preset_fixture", 0, native_stats=stats(**overrides))
    return PageFeatures.from_page_ref(page, "application/pdf")


def registry() -> ParserRegistry:
    return ParserRegistry(
        (
            DoclingExportAdapter(
                {},
                parser_version="fixture",
                model_version="fixture",
                config={},
            ),
            MinerUNormalizedAdapter(
                lambda _request: (),
                parser_version="fixture",
                model_version="fixture",
                config={},
            ),
            PaddleOCRVLPageAdapter(
                lambda _request: (),
                parser_version="fixture",
                model_version="fixture",
                config={},
            ),
        )
    )


class RoutingPresetTests(unittest.TestCase):
    def test_balanced_preset_names_match_shipped_adapter_manifests(self):
        active_registry = registry()
        self.assertEqual(
            {DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER},
            {manifest.adapter_name for manifest in active_registry.manifests()},
        )

        router = DeterministicPageRouter(
            active_registry, balanced_pdf_routing_policy()
        )
        cases = (
            (features(), DOCLING_ADAPTER),
            (
                features(candidate_table_regions=2, candidate_equation_regions=1),
                MINERU_ADAPTER,
            ),
            (
                features(
                    native_character_count=2,
                    native_word_count=1,
                    text_coverage_ratio=0.0,
                    raster_image_coverage_ratio=0.90,
                    is_slide_like=True,
                ),
                PADDLEOCR_VL_ADAPTER,
            ),
        )
        for page_features, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expected, router.route(page_features).primary_adapter_name)

    def test_preset_is_versioned_serializable_and_has_explicit_overrides(self):
        policy = routing_policy_preset(
            BALANCED_PDF_PRESET,
            estimated_latency_ms={DOCLING_ADAPTER: 25},
            max_primary_latency_ms=50_000,
            max_total_latency_ms=90_000,
        )
        self.assertEqual(policy, type(policy).from_json(policy.to_json()))
        self.assertEqual(
            25, policy.profile_by_adapter[DOCLING_ADAPTER].estimated_latency_ms
        )

        with self.assertRaises(ContractValidationError):
            routing_policy_preset("magic-auto")
        with self.assertRaises(ContractValidationError):
            balanced_pdf_routing_policy(
                estimated_latency_ms={"misspelled-adapter": 1}
            )

    def test_importing_preset_does_not_load_model_runtimes(self):
        script = """
import json, sys
import document_services.parsing.routing_presets
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
