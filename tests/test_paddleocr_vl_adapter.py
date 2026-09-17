from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import unittest

from document_services.parser_adapters.paddleocr_vl import PaddleOCRVLPageAdapter
from document_services.domain import (
    AssetKind,
    BlockType,
    CoordinateSpace,
    DocumentRef,
    ErrorClassification,
    IssueSeverity,
    PageRef,
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParseErrorCode,
    ParseIssue,
    ParserExecutionError,
    ParserCapabilityManifest,
    ParserModality,
    RepresentationKind,
)
from document_services.ports import ParserRegistry


def request_fixture() -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=hashlib.sha256(b"paddle fixture pdf").hexdigest(),
        mime_type="application/pdf",
        source_uri="file:///fixtures/paddle.pdf",
        document_id="paddle-fixture",
        display_name="paddle.pdf",
        page_count=2,
    )
    return ParseRequest(
        document=document,
        pages=(
            PageRef(document.document_version_id, 0, width=500, height=400),
            PageRef(document.document_version_id, 1, width=500, height=400),
        ),
        options=ParseOptions(languages=("en",)),
        run_id="run-paddle-fixture",
        trace_id="trace-paddle-fixture",
    )


def polygon(bbox):
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def block(label, content, bbox, block_id, order=None, **extra):
    return {
        "block_label": label,
        "block_content": content,
        "block_bbox": bbox,
        "block_polygon_points": polygon(bbox),
        "block_id": block_id,
        "block_order": order,
        "group_id": block_id,
        **extra,
    }


def layout_box(item, score):
    return {
        "cls_id": 1,
        "label": item["block_label"],
        "score": score,
        "coordinate": item["block_bbox"],
        "order": item["block_order"],
        "polygon_points": item["block_polygon_points"],
    }


def page_results_fixture():
    title = block("doc_title", "A multimodal report", [20, 20, 360, 55], 1, 1)
    text = block(
        "text",
        "OCR main text",
        [20, 70, 470, 130],
        2,
        2,
        native_text="Native PDF text",
        ocr_text="OCR alternate text",
        content_source="ocr_text",
    )
    table = block(
        "table",
        "<table><tr><td>A</td><td>1</td></tr></table>",
        [20, 150, 450, 270],
        3,
        3,
        table_grid={"rows": [["A", "1"]]},
        table_caption="Table 1",
        asset_uri="artifact://paddle/page-1/table-3.png",
        asset_sha256="a" * 64,
    )
    formula = block(
        "display_formula",
        "$$ E = mc^2 $$",
        [100, 290, 300, 340],
        4,
        4,
        mathml="<math><mi>E</mi></math>",
        asset_uri="artifact://paddle/page-1/formula-4.png",
    )
    page_zero = {
        "input_path": "/fixtures/paddle.pdf",
        "page_index": 0,
        "page_count": 2,
        "width": 1000,
        "height": 800,
        "model_settings": {
            "use_layout_detection": True,
            "return_layout_polygon_points": True,
        },
        "raw_artifact_uri": "artifact://paddle/page-1/result.json",
        "layout_det_res": {
            "boxes": [
                layout_box(title, 0.99),
                layout_box(text, 0.96),
                layout_box(table, 0.93),
                layout_box(formula, 0.91),
            ]
        },
        "parsing_res_list": [title, text, table, formula],
    }

    image = block("image", "", [30, 40, 280, 240], 5, None)
    chart = block(
        "chart",
        "Quarterly revenue chart",
        [310, 40, 760, 300],
        6,
        None,
        chart_data={"quarters": ["Q1", "Q2"], "values": [3, 5]},
    )
    diagram = block(
        "workflow",
        "Source to answer workflow",
        [30, 330, 760, 600],
        7,
        1,
        ocr_text="Upload -> parse -> retrieve",
        diagram_graph={
            "nodes": ["upload", "parse", "retrieve"],
            "edges": [["upload", "parse"], ["parse", "retrieve"]],
        },
        assets=[{"uri": "diagram-7.webp", "metadata": {"saved_by": "runner"}}],
    )
    caption = block(
        "figure_caption", "Figure 2: ingestion workflow", [30, 620, 600, 660], 8, 2
    )
    page_one = {
        "input_path": "/fixtures/paddle.pdf",
        "page_index": 1,
        "page_count": 2,
        "width": 1000,
        "height": 800,
        "model_settings": {"use_chart_recognition": True},
        "asset_base_uri": "artifact://paddle/page-2",
        "saved_assets": {
            "6": {
                "uri": "imgs/chart-6.jpg",
                "sha256": "b" * 64,
                "metadata": {"saved_by": "save_to_markdown"},
            }
        },
        "layout_det_res": {
            "boxes": [
                layout_box(image, 0.88),
                layout_box(chart, 0.86),
                layout_box(diagram, 0.84),
                layout_box(caption, 0.82),
            ]
        },
        "parsing_res_list": [image, chart, diagram, caption],
    }
    return [page_zero, page_one]


class PaddleOCRVLPageAdapterTests(unittest.TestCase):
    def setUp(self):
        self.request = request_fixture()
        self.pages = page_results_fixture()

    def adapter(self, pages=None):
        values = self.pages if pages is None else pages
        return PaddleOCRVLPageAdapter(
            lambda _request: values,
            parser_version="paddleocr-3.4.0",
            model_version="PaddleOCR-VL-1.6-0.9B",
            config={
                "pipeline_version": "v1.6",
                "vl_rec_backend": "vllm-server",
                "use_chart_recognition": True,
            },
            raw_artifact_uri="artifact://paddle/run",
        )

    def test_manifest_and_result_json_roundtrip(self):
        adapter = self.adapter()
        manifest = adapter.manifest()
        result = ParserRegistry((adapter,)).parse("paddleocr-vl-page", self.request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual(8, len(result.observations))
        self.assertEqual(manifest, ParserCapabilityManifest.from_json(manifest.to_json()))
        self.assertEqual(result, ParseResult.from_json(result.to_json()))
        self.assertEqual("paddleocr-vl", manifest.parser_name)
        self.assertIn(ParserModality.OCR_TEXT, manifest.modalities)
        self.assertIn(ParserModality.TABLE, manifest.modalities)
        self.assertIn(ParserModality.FORMULA, manifest.modalities)
        self.assertIn(ParserModality.CHART, manifest.modalities)
        self.assertIn(ParserModality.DIAGRAM, manifest.modalities)

    def test_preserves_text_tables_formulas_visuals_captions_and_geometry(self):
        result = self.adapter().parse(self.request)
        by_native = {
            item.provenance.parser_native_id: item for item in result.observations
        }

        title = by_native["page:0:block:1"]
        self.assertEqual(BlockType.TITLE, title.block_type)
        self.assertEqual(RepresentationKind.OCR_TEXT, title.representations[0].kind)
        self.assertEqual(0.99, title.confidence)

        text = by_native["page:0:block:2"]
        self.assertEqual(
            {RepresentationKind.TEXT, RepresentationKind.OCR_TEXT},
            {item.kind for item in text.representations},
        )
        self.assertEqual(CoordinateSpace.PAGE_PIXELS, text.geometry.coordinate_space)
        self.assertEqual(4, len(text.geometry.polygon))

        table = by_native["page:0:block:3"]
        self.assertEqual(BlockType.TABLE, table.block_type)
        self.assertEqual(
            {
                RepresentationKind.HTML,
                RepresentationKind.TABLE_GRID,
                RepresentationKind.CAPTION,
            },
            {item.kind for item in table.representations},
        )
        self.assertEqual(AssetKind.REGION_CROP, table.assets[0].kind)
        self.assertEqual("a" * 64, table.assets[0].sha256)

        formula = by_native["page:0:block:4"]
        self.assertEqual(BlockType.EQUATION, formula.block_type)
        self.assertEqual(
            {RepresentationKind.OCR_TEXT, RepresentationKind.MATHML},
            {item.kind for item in formula.representations},
        )
        self.assertEqual("unknown", next(
            item.metadata["format"]
            for item in formula.representations
            if item.kind is RepresentationKind.OCR_TEXT
        ))
        self.assertEqual(1, len(formula.semantic_substructures))
        self.assertTrue(formula.assets)

        image = by_native["page:1:block:5"]
        self.assertEqual(BlockType.FIGURE, image.block_type)
        self.assertEqual(
            "artifact://paddle/page-2/imgs/img_in_image_box_30_40_280_240.jpg",
            image.assets[0].uri,
        )

        chart = by_native["page:1:block:6"]
        self.assertEqual(BlockType.CHART, chart.block_type)
        self.assertEqual(
            {
                RepresentationKind.TEXT,
                RepresentationKind.STRUCTURED_DATA,
            },
            {item.kind for item in chart.representations},
        )
        self.assertEqual("b" * 64, chart.assets[0].sha256)

        diagram = by_native["page:1:block:7"]
        self.assertEqual(BlockType.DIAGRAM, diagram.block_type)
        self.assertEqual(
            {
                RepresentationKind.TEXT,
                RepresentationKind.OCR_TEXT,
                RepresentationKind.DIAGRAM_GRAPH,
            },
            {item.kind for item in diagram.representations},
        )
        self.assertEqual(1, len(diagram.semantic_substructures))
        self.assertEqual(
            "artifact://paddle/page-2/diagram-7.webp", diagram.assets[0].uri
        )

        caption = by_native["page:1:block:8"]
        self.assertEqual(BlockType.CAPTION, caption.block_type)
        self.assertEqual(RepresentationKind.CAPTION, caption.representations[0].kind)
        for observation in result.observations:
            self.assertEqual(
                self.request.document.document_version_id,
                observation.document_version_id,
            )
            self.assertEqual("paddleocr-vl", observation.provenance.parser_name)

    def test_ids_and_content_hashes_are_stable_when_pages_and_blocks_reorder(self):
        reordered = copy.deepcopy(self.pages)
        reordered.reverse()
        for page in reordered:
            page["parsing_res_list"].reverse()
        forward = self.adapter().parse(self.request)
        reverse = self.adapter(reordered).parse(self.request)
        self.assertEqual(
            {(item.observation_id, item.content_sha256) for item in forward.observations},
            {(item.observation_id, item.content_sha256) for item in reverse.observations},
        )

    def test_bad_block_yields_partial_result_without_fake_content(self):
        pages = copy.deepcopy(self.pages)
        pages[0]["parsing_res_list"][1]["block_bbox"] = [20, 70, 2000, 130]
        result = self.adapter(pages).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertEqual(7, len(result.observations))
        self.assertTrue(
            any("outside page dimensions" in issue.message for issue in result.issues)
        )
        self.assertFalse(
            any(
                representation.content == "OCR main text"
                for observation in result.observations
                for representation in observation.representations
            )
        )

    def test_empty_visual_without_saved_asset_is_explicit_partial_failure(self):
        pages = copy.deepcopy(self.pages)
        del pages[1]["asset_base_uri"]
        del pages[1]["saved_assets"]
        result = self.adapter(pages).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertTrue(any("representation or asset" in i.message for i in result.issues))
        self.assertNotIn(
            "page:1:block:5",
            {item.provenance.parser_native_id for item in result.observations},
        )

    def test_source_mismatch_and_missing_page_are_not_silently_accepted(self):
        pages = copy.deepcopy(self.pages)
        pages[1]["input_path"] = "/different/document.pdf"
        result = self.adapter(pages).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertEqual({0}, {item.page_index for item in result.observations})
        self.assertTrue(any("not bound" in issue.message for issue in result.issues))

        missing = self.adapter(self.pages[:1]).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, missing.status)
        self.assertTrue(any("no result" in issue.message for issue in missing.issues))

    def test_staged_input_is_bound_by_source_hash(self):
        pages = copy.deepcopy(self.pages)
        for page in pages:
            page["input_path"] = "/worker/staged/input.pdf"
            page["source_sha256"] = self.request.document.content_sha256
        result = self.adapter(pages).parse(self.request)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)

        pages[1]["source_sha256"] = "f" * 64
        failed_page = self.adapter(pages).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, failed_page.status)
        self.assertTrue(any("source_sha256" in i.message for i in failed_page.issues))

    def test_loader_and_stream_failures_are_typed_and_retryable(self):
        def fail_loader(_request):
            raise RuntimeError("vLLM endpoint unavailable")

        loader_result = PaddleOCRVLPageAdapter(
            fail_loader,
            parser_version="3.4.0",
            model_version="PaddleOCR-VL-1.6-0.9B",
            config={"backend": "vllm-server"},
        ).parse(self.request)
        self.assertEqual(ParseStatus.FAILED, loader_result.status)
        self.assertEqual(
            ErrorClassification.RETRYABLE, loader_result.issues[0].classification
        )

        def fail_after_first():
            yield self.pages[0]
            raise ConnectionError("worker disconnected")

        stream_result = self.adapter(fail_after_first()).parse(self.request)
        self.assertEqual(ParseStatus.PARTIAL, stream_result.status)
        self.assertEqual({0}, {item.page_index for item in stream_result.observations})
        self.assertTrue(
            any(
                issue.classification == ErrorClassification.RETRYABLE
                for issue in stream_result.issues
            )
        )

    def test_loader_can_preserve_permanent_service_contract_failure(self):
        issue = ParseIssue(
            ParseErrorCode.UNSUPPORTED_INPUT,
            ErrorClassification.PERMANENT,
            IssueSeverity.ERROR,
            "raw VLM service is not a complete PaddleOCR-VL parser",
        )

        def fail_loader(_request):
            raise ParserExecutionError(issue)

        result = PaddleOCRVLPageAdapter(
            fail_loader,
            parser_version="3.7.0",
            model_version="PaddleOCR-VL-1.6-0.9B",
            config={"endpoint_kind": "complete-pipeline"},
        ).parse(self.request)
        self.assertEqual((issue,), result.issues)

    def test_import_does_not_load_paddle_or_model_runtimes(self):
        script = """
import json, sys
import document_services.parser_adapters.paddleocr_vl
forbidden = ('paddle', 'paddleocr', 'torch', 'transformers', 'vllm')
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
