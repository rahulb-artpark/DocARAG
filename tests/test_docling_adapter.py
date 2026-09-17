from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from document_services.parser_adapters.docling import (
    DEFAULT_MAX_EMBEDDED_ASSET_BYTES,
    DoclingExportAdapter,
)
from document_services.domain import (
    AssetKind,
    BlockType,
    CoordinateOrigin,
    CoordinateSpace,
    DocumentRef,
    ErrorClassification,
    IssueSeverity,
    PageRef,
    ParseErrorCode,
    ParseIssue,
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserExecutionError,
    RepresentationKind,
)
from document_services.ports import ParserRegistry


SOURCE_BYTES = b"docling adapter source fixture"
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


def request_fixture(
    *, pages: tuple[int, ...] = (0, 1), options: ParseOptions | None = None
) -> ParseRequest:
    document = DocumentRef.identified(
        content_sha256=SOURCE_SHA256,
        mime_type="application/pdf",
        source_uri="file:///fixtures/docling.pdf",
        document_id="docling-fixture",
        display_name="docling.pdf",
        page_count=2,
    )
    return ParseRequest(
        document=document,
        pages=tuple(
            PageRef(document.document_version_id, page, width=612, height=792)
            for page in pages
        ),
        options=options or ParseOptions(languages=("en",)),
        run_id="run-docling",
        trace_id="trace-docling",
        idempotency_key="docling-fixture-v1",
    )


def provenance(page_no: int, bbox: tuple[float, float, float, float], end: int = 1):
    left, top, right, bottom = bbox
    return [
        {
            "page_no": page_no,
            "bbox": {
                "l": left,
                "t": top,
                "r": right,
                "b": bottom,
                "coord_origin": "BOTTOMLEFT",
            },
            "charspan": [0, end],
        }
    ]


def export_fixture() -> dict[str, object]:
    table_image_hash = hashlib.sha256(b"table-image").hexdigest()
    formula_image_hash = hashlib.sha256(b"formula-image").hexdigest()
    picture_image_hash = hashlib.sha256(b"picture-image").hexdigest()
    cells = [
        {
            "text": "Metric <name>",
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 1,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 2,
            "row_span": 1,
            "col_span": 2,
            "column_header": True,
            "row_header": False,
        },
        {
            "text": "Accuracy",
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 1,
            "row_span": 1,
            "col_span": 1,
            "column_header": False,
            "row_header": True,
        },
        {
            "text": "98|7%",
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "start_col_offset_idx": 1,
            "end_col_offset_idx": 2,
            "row_span": 1,
            "col_span": 1,
            "column_header": False,
            "row_header": False,
        },
    ]
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "docling-fixture",
        "source_sha256": SOURCE_SHA256,
        "origin": {"mimetype": "application/pdf", "filename": "docling.pdf"},
        "pages": {
            "1": {"page_no": 1, "size": {"width": 612, "height": 792}},
            "2": {"page_no": 2, "size": {"width": 612, "height": 792}},
        },
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/groups/0"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/pictures/0"},
            ],
        },
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "list",
                "children": [{"$ref": "#/texts/1"}],
            }
        ],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "content_layer": "body",
                "label": "section_header",
                "level": 1,
                "text": "Results",
                "orig": "Results",
                "prov": provenance(1, (50, 750, 200, 720), len("Results")),
            },
            {
                "self_ref": "#/texts/1",
                "parent": {"$ref": "#/groups/0"},
                "content_layer": "body",
                "label": "list_item",
                "marker": "1.",
                "enumerated": True,
                "text": "Measure every parser",
                "orig": "Measure every parser",
                "prov": provenance(1, (50, 700, 260, 680), 20),
            },
            {
                "self_ref": "#/texts/2",
                "parent": {"$ref": "#/body"},
                "content_layer": "body",
                "label": "formula",
                "text": "E = m c^2",
                "orig": "E=mc2",
                # This span intentionally describes the shorter original text.
                "prov": provenance(1, (60, 640, 180, 600), len("E=mc2")),
                "confidence": 0.97,
                "image": {
                    "uri": "artifact://docling/formula.png",
                    "mimetype": "image/png",
                    "sha256": formula_image_hash,
                    "size": {"width": 120, "height": 40},
                    "dpi": 144,
                },
            },
            {
                "self_ref": "#/texts/3",
                "parent": {"$ref": "#/tables/0"},
                "content_layer": "body",
                "label": "caption",
                "text": "Table 1: benchmark scores",
                "orig": "Table 1: benchmark scores",
                "prov": provenance(2, (50, 690, 280, 670), 25),
            },
            {
                "self_ref": "#/texts/4",
                "parent": {"$ref": "#/tables/0"},
                "content_layer": "body",
                "label": "footnote",
                "text": "Scores are illustrative.",
                "orig": "Scores are illustrative.",
                "prov": provenance(2, (50, 400, 260, 380), 24),
            },
            {
                "self_ref": "#/texts/5",
                "parent": {"$ref": "#/pictures/0"},
                "content_layer": "body",
                "label": "text",
                "text": "Load PDF → route page → index evidence",
                "orig": "Load PDF → route page → index evidence",
                "prov": provenance(2, (90, 330, 500, 300), 38),
            },
            {
                "self_ref": "#/texts/6",
                "parent": {"$ref": "#/pictures/0"},
                "content_layer": "body",
                "label": "caption",
                "text": "Figure 1: routed workflow",
                "orig": "Figure 1: routed workflow",
                "prov": provenance(2, (80, 250, 320, 230), 25),
            },
            {
                "self_ref": "#/texts/7",
                "parent": {"$ref": "#/pictures/0"},
                "content_layer": "body",
                "label": "footnote",
                "text": "Source: architecture team",
                "orig": "Source: architecture team",
                "prov": provenance(2, (80, 220, 320, 200), 25),
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "parent": {"$ref": "#/body"},
                "content_layer": "body",
                "label": "table",
                "prov": provenance(2, (50, 650, 560, 420), 0),
                "captions": [{"$ref": "#/texts/3"}],
                "footnotes": [{"$ref": "#/texts/4"}],
                "references": [],
                "image": {
                    "uri": "artifact://docling/table.png",
                    "mimetype": "image/png",
                    "sha256": table_image_hash,
                    "size": {"width": 1020, "height": 460},
                    "dpi": 144,
                },
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": cells,
                    "grid": [],
                },
                "annotations": [],
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "parent": {"$ref": "#/body"},
                "content_layer": "body",
                "label": "picture",
                "prov": provenance(2, (70, 370, 550, 260), 0),
                "children": [{"$ref": "#/texts/5"}],
                "captions": [{"$ref": "#/texts/6"}],
                "footnotes": [{"$ref": "#/texts/7"}],
                "references": [],
                "image": {
                    "uri": "artifact://docling/workflow.png",
                    "mimetype": "image/png",
                    "sha256": picture_image_hash,
                    "size": {"width": 960, "height": 220},
                    "dpi": 144,
                },
                "annotations": [
                    {
                        "kind": "classification",
                        "provenance": "DocumentPictureClassifier",
                        "predicted_classes": [
                            {"class_name": "flow_chart", "confidence": 0.93},
                            {"class_name": "screenshot_from_computer", "confidence": 0.05},
                        ],
                    },
                    {
                        "kind": "description",
                        "provenance": "fixture-vlm",
                        "text": "A three-step document routing workflow.",
                    },
                ],
            }
        ],
    }


def adapter(
    export=None,
    *,
    max_embedded_asset_bytes: int = DEFAULT_MAX_EMBEDDED_ASSET_BYTES,
) -> DoclingExportAdapter:
    return DoclingExportAdapter(
        export_fixture() if export is None else export,
        parser_version="2.57.0",
        model_version="docling-layout-fixture",
        config={"ocr": "auto", "tables": True, "formula_enrichment": True},
        raw_artifact_uri="artifact://runs/docling/document.json",
        max_embedded_asset_bytes=max_embedded_asset_bytes,
    )


class DoclingExportAdapterTests(unittest.TestCase):
    def test_preserves_text_table_formula_image_and_source_binding(self):
        request = request_fixture()
        parser = adapter()
        result = ParserRegistry((parser,)).parse("docling-export", request)

        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertFalse(result.issues)
        self.assertEqual(request.request_id, result.request_id)
        self.assertEqual(result, ParseResult.from_json(result.to_json()))

        by_ref = {
            observation.provenance.parser_native_id: observation
            for observation in result.observations
        }
        title = by_ref["#/texts/0"]
        list_item = by_ref["#/texts/1"]
        equation = by_ref["#/texts/2"]
        table = by_ref["#/tables/0"]
        picture = by_ref["#/pictures/0"]
        picture_caption = by_ref["#/texts/6"]
        picture_footnote = by_ref["#/texts/7"]

        self.assertEqual(BlockType.TITLE, title.block_type)
        self.assertEqual("# Results", title.representations[1].content)
        self.assertEqual(BlockType.LIST_ITEM, list_item.block_type)
        self.assertEqual("1. Measure every parser", list_item.representations[1].content)

        self.assertEqual(BlockType.EQUATION, equation.block_type)
        self.assertEqual(
            (RepresentationKind.TEXT,),
            tuple(representation.kind for representation in equation.representations),
        )
        self.assertEqual("E = m c^2", equation.representations[0].content)
        self.assertEqual("unknown", equation.representations[0].metadata["format"])
        self.assertEqual(AssetKind.REGION_CROP, equation.assets[0].kind)

        table_kinds = {representation.kind for representation in table.representations}
        self.assertEqual(
            {
                RepresentationKind.TABLE_GRID,
                RepresentationKind.HTML,
                RepresentationKind.MARKDOWN,
                RepresentationKind.CAPTION,
                RepresentationKind.FOOTNOTE,
            },
            table_kinds,
        )
        table_html = next(
            representation.content
            for representation in table.representations
            if representation.kind == RepresentationKind.HTML
        )
        table_markdown = next(
            representation.content
            for representation in table.representations
            if representation.kind == RepresentationKind.MARKDOWN
        )
        self.assertIn('colspan="2"', table_html)
        self.assertIn("Metric &lt;name&gt;", table_html)
        self.assertIn(r"98\|7%", table_markdown)
        self.assertEqual(AssetKind.REGION_CROP, table.assets[0].kind)
        self.assertEqual(1, len(table.semantic_substructures))
        table_structure = table.semantic_substructures[0]
        self.assertEqual(3, len(table_structure.elements))
        self.assertTrue(table_structure.relations)
        self.assertTrue(
            all(
                selector.representation_id
                == next(
                    item.representation_id
                    for item in table.representations
                    if item.kind is RepresentationKind.TABLE_GRID
                )
                for element in table_structure.elements
                for selector in element.source_selectors
            )
        )

        self.assertEqual(BlockType.DIAGRAM, picture.block_type)
        self.assertEqual("flow_chart", picture.semantic_subtype)
        self.assertEqual(0.93, picture.confidence)
        picture_kinds = {representation.kind for representation in picture.representations}
        self.assertEqual(
            {
                RepresentationKind.VISUAL_DESCRIPTION,
                RepresentationKind.OCR_TEXT,
                RepresentationKind.STRUCTURED_DATA,
                RepresentationKind.CAPTION,
                RepresentationKind.FOOTNOTE,
            },
            picture_kinds,
        )
        ocr = next(
            representation.content
            for representation in picture.representations
            if representation.kind == RepresentationKind.OCR_TEXT
        )
        self.assertIn("route page", ocr)
        self.assertEqual(AssetKind.EXTRACTED_IMAGE, picture.assets[0].kind)
        self.assertEqual(
            RepresentationKind.CAPTION, picture_caption.representations[0].kind
        )
        self.assertEqual(
            RepresentationKind.FOOTNOTE, picture_footnote.representations[0].kind
        )

        for observation in result.observations:
            self.assertEqual(
                request.document.document_version_id,
                observation.document_version_id,
            )
            self.assertEqual(
                request.page_by_index[observation.page_index].page_id,
                observation.page_id,
            )
            self.assertEqual("docling", observation.provenance.parser_name)
            self.assertEqual("2.57.0", observation.provenance.parser_version)
        self.assertEqual(
            "artifact://runs/docling/document.json#/tables/0",
            table.provenance.raw_artifact_uri,
        )
        self.assertEqual(CoordinateSpace.PAGE_POINTS, table.geometry.coordinate_space)
        self.assertEqual(CoordinateOrigin.TOP_LEFT, table.geometry.coordinate_origin)
        self.assertEqual([50.0, 142.0, 560.0, 372.0], table.geometry.bbox.to_list())
        self.assertEqual(
            "BOTTOMLEFT", table.metadata["source_coordinate_origin"]
        )

    def test_manifest_is_explicit_and_import_is_dependency_free(self):
        manifest = adapter().manifest()
        self.assertEqual("docling-export", manifest.adapter_name)
        self.assertEqual("docling", manifest.parser_name)
        self.assertIn("application/pdf", manifest.supported_mime_types)
        self.assertIn(BlockType.TABLE, manifest.block_types)
        self.assertIn(BlockType.DIAGRAM, manifest.block_types)
        self.assertFalse(manifest.supports_region_selection)
        self.assertTrue(manifest.supports_batch)
        self.assertEqual(False, manifest.metadata["vendor_imports"])
        self.assertEqual(
            DEFAULT_MAX_EMBEDDED_ASSET_BYTES,
            manifest.metadata["max_embedded_asset_bytes"],
        )
        self.assertIn("externalize", manifest.metadata["embedded_asset_policy"])

        script = """
import json, sys
import document_services.parser_adapters.docling
forbidden = ('torch', 'mineru', 'paddle', 'paddleocr', 'docling', 'transformers', 'vllm')
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in forbidden)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual([], json.loads(completed.stdout))

    def test_one_loader_export_covers_a_multi_page_batch(self):
        calls: list[str] = []

        def load(request: ParseRequest):
            calls.append(request.request_id)
            return export_fixture()

        parser = adapter(load)
        request = request_fixture()
        result = parser.parse(request)
        self.assertTrue(parser.manifest().supports_batch)
        self.assertEqual([request.request_id], calls)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual({0, 1}, {item.page_index for item in result.observations})

    def test_bottom_left_and_top_left_geometry_are_physically_equivalent(self):
        request = request_fixture()
        bottom_left = export_fixture()
        bottom_provenance = bottom_left["texts"][0]["prov"][0]
        bottom_provenance["coord_origin"] = "BOTTOMLEFT"
        bottom_provenance["polygon"] = [
            {"x": 50, "y": 720},
            {"x": 200, "y": 720},
            {"x": 200, "y": 750},
            {"x": 50, "y": 750},
        ]

        top_left = copy.deepcopy(bottom_left)
        top_provenance = top_left["texts"][0]["prov"][0]
        top_provenance["bbox"] = {
            "l": 50,
            "t": 42,
            "r": 200,
            "b": 72,
            "coord_origin": "TOPLEFT",
        }
        top_provenance["coord_origin"] = "TOPLEFT"
        top_provenance["polygon"] = [
            {"x": 50, "y": 72},
            {"x": 200, "y": 72},
            {"x": 200, "y": 42},
            {"x": 50, "y": 42},
        ]

        bottom_result = adapter(bottom_left).parse(request)
        top_result = adapter(top_left).parse(request)
        self.assertEqual(ParseStatus.SUCCEEDED, bottom_result.status)
        self.assertEqual(ParseStatus.SUCCEEDED, top_result.status)
        bottom_geometry = next(
            item.geometry
            for item in bottom_result.observations
            if item.provenance.parser_native_id == "#/texts/0"
        )
        top_geometry = next(
            item.geometry
            for item in top_result.observations
            if item.provenance.parser_native_id == "#/texts/0"
        )
        self.assertEqual(bottom_geometry, top_geometry)
        self.assertEqual(CoordinateOrigin.TOP_LEFT, bottom_geometry.coordinate_origin)
        self.assertEqual([50.0, 42.0, 200.0, 72.0], bottom_geometry.bbox.to_list())
        self.assertEqual(
            [(50.0, 72.0), (200.0, 72.0), (200.0, 42.0), (50.0, 42.0)],
            [(point.x, point.y) for point in bottom_geometry.polygon],
        )

    def test_unknown_coordinate_origin_is_a_typed_partial_issue(self):
        export = export_fixture()
        export["pictures"][0]["prov"][0]["bbox"]["coord_origin"] = "CENTER"
        result = adapter(export).parse(request_fixture())
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertTrue(result.observations)
        self.assertEqual(1, len(result.issues))
        self.assertEqual("#/pictures/0", result.issues[0].parser_native_id)
        self.assertIn("unsupported coordinate origin", result.issues[0].message)

    def test_bottom_left_geometry_without_page_height_is_rejected(self):
        export = export_fixture()
        export["body"]["children"] = [{"$ref": "#/texts/0"}]
        export["groups"] = []
        export["texts"] = [export["texts"][0]]
        export["tables"] = []
        export["pictures"] = []
        template = request_fixture()
        request = ParseRequest(
            document=template.document,
            pages=tuple(
                PageRef(template.document.document_version_id, page_index)
                for page_index in (0, 1)
            ),
            options=template.options,
        )
        result = adapter(export).parse(request)
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertEqual(1, len(result.issues))
        self.assertIn("authoritative request page height", result.issues[0].message)

    def test_oversized_data_uri_is_rejected_before_decode_as_partial_issue(self):
        export = export_fixture()
        payload = base64.b64encode(b"12345").decode("ascii")
        export["pictures"][0]["image"] = {
            "uri": f"DATA:image/png;BASE64,{payload}",
            "mimetype": "image/png",
        }
        with patch(
            "document_services.parser_adapters.docling.base64.b64decode",
            side_effect=AssertionError("oversized payload must not be decoded"),
        ) as decoder:
            result = adapter(export, max_embedded_asset_bytes=4).parse(
                request_fixture()
            )
        decoder.assert_not_called()
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertEqual(1, len(result.issues))
        self.assertEqual("#/pictures/0", result.issues[0].parser_native_id)
        self.assertIn("externalize the asset", result.issues[0].message)

    def test_embedded_data_uri_at_limit_is_hashed_and_recorded(self):
        export = export_fixture()
        image_bytes = b"12345"
        payload = base64.b64encode(image_bytes).decode("ascii")
        export["pictures"][0]["image"] = {
            "uri": f"data:image/png;base64,{payload}",
            "mimetype": "image/png",
        }
        result = adapter(export, max_embedded_asset_bytes=len(image_bytes)).parse(
            request_fixture()
        )
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        picture = next(
            item
            for item in result.observations
            if item.provenance.parser_native_id == "#/pictures/0"
        )
        self.assertEqual(hashlib.sha256(image_bytes).hexdigest(), picture.assets[0].sha256)
        self.assertEqual(len(image_bytes), picture.assets[0].metadata["embedded_bytes"])

    def test_native_ids_are_stable_when_collection_arrays_are_reordered(self):
        request = request_fixture()
        forward = adapter().parse(request)
        reversed_export = export_fixture()
        reversed_export["texts"] = list(reversed(reversed_export["texts"]))
        reverse = adapter(reversed_export).parse(request)
        self.assertEqual(ParseStatus.SUCCEEDED, reverse.status)
        self.assertEqual(
            {
                (item.provenance.parser_native_id, item.observation_id)
                for item in forward.observations
            },
            {
                (item.provenance.parser_native_id, item.observation_id)
                for item in reverse.observations
            },
        )

    def test_full_export_is_cleanly_filtered_to_requested_pages(self):
        request = request_fixture(pages=(1,))
        result = adapter().parse(request)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertFalse(result.issues)
        self.assertTrue(result.observations)
        self.assertEqual({1}, {item.page_index for item in result.observations})

    def test_options_disable_modalities_without_fake_observations(self):
        request = request_fixture(
            options=ParseOptions(
                languages=("en",),
                parse_tables=False,
                parse_formulas=False,
                parse_images=False,
            )
        )
        result = adapter().parse(request)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        block_types = {item.block_type for item in result.observations}
        self.assertNotIn(BlockType.TABLE, block_types)
        self.assertNotIn(BlockType.EQUATION, block_types)
        self.assertNotIn(BlockType.DIAGRAM, block_types)
        self.assertTrue(all(not item.assets for item in result.observations))

    def test_invalid_item_returns_typed_partial_result(self):
        export = export_fixture()
        invalid = copy.deepcopy(export["texts"][0])
        invalid["self_ref"] = "#/texts/99"
        invalid["prov"][0]["page_no"] = 99
        export["texts"].append(invalid)
        result = adapter(export).parse(request_fixture())
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertTrue(result.observations)
        self.assertEqual(1, len(result.issues))
        issue = result.issues[0]
        self.assertEqual(ParseErrorCode.INVALID_NORMALIZED_BLOCK, issue.code)
        self.assertEqual(ErrorClassification.PERMANENT, issue.classification)
        self.assertEqual(IssueSeverity.ERROR, issue.severity)
        self.assertEqual("#/texts/99", issue.parser_native_id)

    def test_source_mismatch_is_permanent_contract_failure(self):
        export = export_fixture()
        export["source_sha256"] = "0" * 64
        result = adapter(export).parse(request_fixture())
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertFalse(result.observations)
        self.assertEqual(ParseErrorCode.CONTRACT_VIOLATION, result.issues[0].code)
        self.assertEqual(
            ErrorClassification.PERMANENT, result.issues[0].classification
        )

    def test_loader_failure_is_retryable_typed_failure(self):
        def fail(_request):
            raise RuntimeError("worker disconnected")

        parser = DoclingExportAdapter(
            fail,
            parser_version="2.57.0",
            model_version="docling-layout-fixture",
            config={"ocr": "auto"},
        )
        result = parser.parse(request_fixture())
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertFalse(result.observations)
        self.assertEqual(ParseErrorCode.PARSER_FAILURE, result.issues[0].code)
        self.assertEqual(
            ErrorClassification.RETRYABLE, result.issues[0].classification
        )

    def test_loader_can_preserve_permanent_service_input_failure(self):
        issue = ParseIssue(
            ParseErrorCode.INVALID_REQUEST,
            ErrorClassification.PERMANENT,
            IssueSeverity.ERROR,
            "verified source binding is invalid",
        )

        def fail(_request):
            raise ParserExecutionError(issue)

        result = DoclingExportAdapter(
            fail,
            parser_version="2.57.0",
            model_version="docling-layout-fixture",
            config={"ocr": "auto"},
        ).parse(request_fixture())
        self.assertEqual((issue,), result.issues)


if __name__ == "__main__":
    unittest.main()
