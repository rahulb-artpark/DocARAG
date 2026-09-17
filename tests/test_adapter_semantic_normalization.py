from __future__ import annotations

from collections.abc import Mapping
import copy
import unittest

from document_services.parser_adapters.mineru_compat import MinerUNormalizedAdapter
from document_services.parser_adapters.paddleocr_vl import PaddleOCRVLPageAdapter
from document_services.parser_adapters.semantic_normalization import (
    MAX_MATHML_BYTES,
    MAX_MATHML_DEPTH,
    MAX_MATHML_ELEMENTS,
    explicit_diagram_substructure,
)
from document_services.domain import (
    ParseErrorCode,
    ParseStatus,
    RelationType,
    Representation,
    RepresentationKind,
    SemanticProducerKind,
    SemanticRelationKind,
    SemanticStructureKind,
)
from tests.test_docling_adapter import (
    adapter as docling_adapter,
    export_fixture as docling_export_fixture,
    request_fixture as docling_request_fixture,
)
from tests.test_paddleocr_vl_adapter import (
    page_results_fixture,
    request_fixture as paddle_request_fixture,
)
from tests.test_parser_contracts import request_fixture as generic_request_fixture


class AdapterSemanticNormalizationTests(unittest.TestCase):
    @staticmethod
    def mineru(blocks):
        return MinerUNormalizedAdapter(
            lambda _request: blocks,
            parser_version="3.4.4-fixture",
            model_version="mineru-fixture",
            config={"backend": "pipeline"},
        )

    @staticmethod
    def paddle(pages):
        return PaddleOCRVLPageAdapter(
            lambda _request: pages,
            parser_version="paddleocr-3.4.0",
            model_version="PaddleOCR-VL-1.6-0.9B",
            config={"pipeline_version": "v1.6", "vl_rec_backend": "vllm-server"},
            raw_artifact_uri="artifact://paddle/run",
        )

    def test_docling_cells_spans_headers_and_exact_selectors_are_typed(self) -> None:
        result = docling_adapter().parse(docling_request_fixture())
        table = next(
            item
            for item in result.observations
            if item.provenance.parser_native_id == "#/tables/0"
        )
        structure = table.semantic_substructures[0]
        self.assertEqual(SemanticStructureKind.TABLE, structure.kind)
        self.assertEqual(SemanticProducerKind.DETERMINISTIC_DERIVED, structure.producer_kind)
        self.assertEqual(("data.table_cells",), structure.source_field_paths)
        self.assertFalse(structure.source_observation_ids)
        self.assertEqual(
            {"column_header", "row_header", "body"},
            {element.role for element in structure.elements},
        )
        self.assertIn(
            (0, 0, 1, 2),
            {
                (
                    element.row_index,
                    element.column_index,
                    element.row_span,
                    element.column_span,
                )
                for element in structure.elements
            },
        )
        self.assertEqual(
            {"/table_cells/0", "/table_cells/1", "/table_cells/2"},
            {
                selector.native_selector
                for element in structure.elements
                for selector in element.source_selectors
            },
        )
        self.assertTrue(
            {
                SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                SemanticRelationKind.TABLE_ADJACENT_BELOW,
            }.issubset({relation.kind for relation in structure.relations})
        )

    def test_docling_rejects_hostile_dimensions_before_dense_allocation(self) -> None:
        export = docling_export_fixture()
        export["tables"][0]["data"]["num_rows"] = 10**9
        export["tables"][0]["data"]["num_cols"] = 10**9
        result = docling_adapter(export).parse(docling_request_fixture())
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        self.assertFalse(
            any(
                item.provenance.parser_native_id == "#/tables/0"
                for item in result.observations
            )
        )
        self.assertTrue(
            any(
                issue.code is ParseErrorCode.INVALID_NORMALIZED_BLOCK
                and "table-position limit" in issue.message
                for issue in result.issues
            )
        )

    def test_explicit_grids_and_graph_edges_keep_representation_selectors(self) -> None:
        result = self.paddle(page_results_fixture()).parse(paddle_request_fixture())
        by_native = {
            item.provenance.parser_native_id: item for item in result.observations
        }
        table = by_native["page:0:block:3"]
        table_structure = table.semantic_substructures[0]
        table_representation = next(
            item
            for item in table.representations
            if item.kind is RepresentationKind.TABLE_GRID
        )
        self.assertEqual(
            {"/rows/0/0", "/rows/0/1"},
            {
                selector.native_selector
                for element in table_structure.elements
                for selector in element.source_selectors
            },
        )
        self.assertTrue(
            all(
                selector.representation_id == table_representation.representation_id
                for element in table_structure.elements
                for selector in element.source_selectors
            )
        )

        diagram = by_native["page:1:block:7"]
        graph = diagram.semantic_substructures[0]
        self.assertEqual(SemanticStructureKind.FLOWCHART, graph.kind)
        self.assertEqual(3, len(graph.elements))
        self.assertEqual(2, len(graph.relations))
        selector_by_id = {
            selector.selector_id: selector
            for element in graph.elements
            for selector in element.source_selectors
        }
        self.assertEqual(
            {"/edges/0", "/edges/1"},
            {
                selector_by_id[relation.source_selector_ids[0]].native_selector
                for relation in graph.relations
            },
        )
        self.assertTrue(
            all(
                relation.kind is SemanticRelationKind.FLOW_NEXT
                for relation in graph.relations
            )
        )

    def test_cross_parser_graph_comparison_ignores_native_ids_but_keeps_edge_type(self) -> None:
        docling_graph = {
            "nodes": [
                {
                    "id": "docling-node-17",
                    "label": "Manager",
                    "type": "actor",
                    "confidence": 0.99,
                    "metadata": {"vendor": "docling", "object_id": 17},
                },
                {
                    "id": "docling-node-23",
                    "label": "Report",
                    "type": "document",
                    "confidence": 0.98,
                    "metadata": {"vendor": "docling", "object_id": 23},
                },
            ],
            "edges": [
                {
                    "source": "docling-node-17",
                    "target": "docling-node-23",
                    "type": "Depends-On",
                    "label": "document dependency",
                    "confidence": 0.97,
                }
            ],
        }
        mineru_graph = {
            "nodes": [
                {
                    "node_id": "mineru-A",
                    "name": "Manager",
                    "kind": "actor",
                    "confidence": 0.71,
                    "metadata": {"vendor": "mineru", "native_key": "A"},
                },
                {
                    "node_id": "mineru-B",
                    "name": "Report",
                    "kind": "document",
                    "confidence": 0.69,
                    "metadata": {"vendor": "mineru", "native_key": "B"},
                },
            ],
            "edges": [
                {
                    "from": "mineru-A",
                    "to": "mineru-B",
                    "kind": "depends on",
                    "label": "document dependency",
                    "confidence": 0.62,
                }
            ],
        }
        conflicting_graph = copy.deepcopy(mineru_graph)
        conflicting_graph["edges"][0]["kind"] = "reports_to"

        for flowchart in (False, True):
            with self.subTest(flowchart=flowchart):
                docling_representation = Representation(
                    kind=RepresentationKind.DIAGRAM_GRAPH,
                    content=docling_graph,
                    media_type="application/json",
                    confidence=0.99,
                )
                mineru_representation = Representation(
                    kind=RepresentationKind.DIAGRAM_GRAPH,
                    content=mineru_graph,
                    media_type="application/json",
                    confidence=0.71,
                )
                conflicting_representation = Representation(
                    kind=RepresentationKind.DIAGRAM_GRAPH,
                    content=conflicting_graph,
                    media_type="application/json",
                    confidence=0.71,
                )
                docling = explicit_diagram_substructure(
                    docling_graph,
                    docling_representation,
                    source_field="diagram_graph",
                    parser_name="docling",
                    confidence=0.99,
                    flowchart=flowchart,
                )
                mineru = explicit_diagram_substructure(
                    mineru_graph,
                    mineru_representation,
                    source_field="diagram_graph",
                    parser_name="mineru",
                    confidence=0.71,
                    flowchart=flowchart,
                )
                conflicting = explicit_diagram_substructure(
                    conflicting_graph,
                    conflicting_representation,
                    source_field="diagram_graph",
                    parser_name="mineru",
                    confidence=0.71,
                    flowchart=flowchart,
                )

                self.assertNotEqual(docling.semantic_sha256, mineru.semantic_sha256)
                self.assertNotEqual(docling.substructure_id, mineru.substructure_id)
                self.assertEqual(docling.comparison_sha256, mineru.comparison_sha256)
                self.assertEqual(
                    "depends_on",
                    docling.relations[0].metadata["normalized_edge_type"],
                )
                self.assertEqual(
                    "depends_on",
                    mineru.relations[0].metadata["normalized_edge_type"],
                )
                self.assertNotEqual(
                    mineru.comparison_sha256,
                    conflicting.comparison_sha256,
                )
                # Source-neutral comparison never mutates or unions the exact
                # parser graph and its representation-bound selectors.
                self.assertTrue(
                    any(
                        element.content.get("id") == "docling-node-17"
                        for element in docling.elements
                        if isinstance(element.content, Mapping)
                    )
                )
                self.assertTrue(
                    all(
                        selector.representation_id
                        == docling_representation.representation_id
                        for element in docling.elements
                        for selector in element.source_selectors
                    )
                )

    def test_mathml_tree_is_bounded_typed_and_latex_is_never_inferred(self) -> None:
        result = self.mineru(
            [
                {
                    "type": "equation",
                    "block_id": "equation",
                    "text": "x^2",
                    "mathml": "<math><msup><mi>x</mi><mn>2</mn></msup></math>",
                    "page_idx": 0,
                }
            ]
        ).parse(generic_request_fixture())
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        equation = result.observations[0]
        self.assertEqual(
            {RepresentationKind.TEXT, RepresentationKind.MATHML},
            {item.kind for item in equation.representations},
        )
        text = next(
            item for item in equation.representations if item.kind is RepresentationKind.TEXT
        )
        self.assertEqual("unknown", text.metadata["format"])
        structure = equation.semantic_substructures[0]
        self.assertEqual(SemanticStructureKind.FORMULA, structure.kind)
        self.assertEqual(("mathml",), structure.source_field_paths)
        self.assertIn(
            SemanticRelationKind.FORMULA_SUPERSCRIPT_OF,
            {relation.kind for relation in structure.relations},
        )
        self.assertTrue(
            all(
                selector.representation_id
                == next(
                    item.representation_id
                    for item in equation.representations
                    if item.kind is RepresentationKind.MATHML
                )
                for element in structure.elements
                for selector in element.source_selectors
            )
        )

    def test_explicit_latex_fields_and_formats_remain_authoritative_latex(self) -> None:
        export = docling_export_fixture()
        export["texts"][2]["latex"] = r"E = mc^2"
        docling_result = docling_adapter(export).parse(docling_request_fixture())
        docling_formula = next(
            item
            for item in docling_result.observations
            if item.provenance.parser_native_id == "#/texts/2"
        )
        self.assertEqual(
            {RepresentationKind.LATEX},
            {item.kind for item in docling_formula.representations},
        )

        mineru_result = self.mineru(
            [
                {
                    "type": "equation",
                    "block_id": "latex",
                    "text": r"x^2",
                    "text_format": "latex",
                    "page_idx": 0,
                }
            ]
        ).parse(generic_request_fixture())
        self.assertEqual(
            {RepresentationKind.LATEX},
            {item.kind for item in mineru_result.observations[0].representations},
        )

        pages = page_results_fixture()
        pages[0]["parsing_res_list"][3]["content_format"] = "latex"
        paddle_result = self.paddle(pages).parse(paddle_request_fixture())
        paddle_formula = next(
            item
            for item in paddle_result.observations
            if item.provenance.parser_native_id == "page:0:block:4"
        )
        self.assertIn(
            RepresentationKind.LATEX,
            {item.kind for item in paddle_formula.representations},
        )

    def test_hostile_json_xml_and_undeclared_edges_become_typed_issues(self) -> None:
        nested: object = {"nodes": ["a"], "edges": []}
        for _ in range(40):
            nested = {"nested": nested}
        too_deep_mathml = "<math>" + "<mrow>" * (MAX_MATHML_DEPTH + 1)
        too_deep_mathml += "x" + "</mrow>" * (MAX_MATHML_DEPTH + 1) + "</math>"
        too_many_elements = (
            "<math>"
            + "<mi>x</mi>" * MAX_MATHML_ELEMENTS
            + "</math>"
        )
        too_many_bytes = (
            "<math><mtext>"
            + "x" * MAX_MATHML_BYTES
            + "</mtext></math>"
        )
        blocks = [
            {
                "type": "image",
                "sub_type": "diagram",
                "block_id": "deep-graph",
                "diagram_graph": nested,
                "page_idx": 0,
            },
            {
                "type": "image",
                "sub_type": "diagram",
                "block_id": "bad-edge",
                "diagram_graph": {
                    "nodes": ["declared"],
                    "edges": [["declared", "missing"]],
                },
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "dtd",
                "text": "x",
                "mathml": '<!DOCTYPE math [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><math>&xxe;</math>',
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "deep-xml",
                "text": "x",
                "mathml": too_deep_mathml,
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "many-xml-elements",
                "text": "x",
                "mathml": too_many_elements,
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "large-xml",
                "text": "x",
                "mathml": too_many_bytes,
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "empty-xml",
                "text": "x",
                "mathml": "<math/>",
                "page_idx": 0,
            },
        ]
        result = self.mineru(blocks).parse(generic_request_fixture())
        self.assertEqual(ParseStatus.FAILED, result.status)
        self.assertFalse(result.observations)
        self.assertEqual(7, len(result.issues))
        messages = "\n".join(issue.message for issue in result.issues)
        self.assertIn("JSON depth", messages)
        self.assertIn("undeclared node", messages)
        self.assertIn("DTD or entity", messages)
        self.assertIn("XML depth", messages)
        self.assertIn("XML elements", messages)
        self.assertIn("UTF-8 bytes", messages)
        self.assertIn("no formula content", messages)
        self.assertTrue(
            all(issue.code is ParseErrorCode.INVALID_NORMALIZED_BLOCK for issue in result.issues)
        )

    def test_paddle_visual_content_uses_exact_ocr_setting_and_explicit_description(self) -> None:
        pages = page_results_fixture()
        pages[1]["model_settings"]["docarag_inference_options"] = {
            "useOcrForImageBlock": True
        }
        pages[1]["parsing_res_list"][1]["visual_description"] = "Explicit chart description"
        result = self.paddle(pages).parse(paddle_request_fixture())
        by_native = {
            item.provenance.parser_native_id: item for item in result.observations
        }
        chart = by_native["page:1:block:6"]
        self.assertTrue(
            any(
                item.kind is RepresentationKind.OCR_TEXT
                and item.content == "Quarterly revenue chart"
                for item in chart.representations
            )
        )
        self.assertTrue(
            any(
                item.kind is RepresentationKind.VISUAL_DESCRIPTION
                and item.content == "Explicit chart description"
                for item in chart.representations
            )
        )

    def test_paddle_incomplete_native_order_emits_no_page_reading_chain(self) -> None:
        pages = page_results_fixture()
        pages[0]["parsing_res_list"][2]["block_order"] = None
        result = self.paddle(pages).parse(paddle_request_fixture())
        page_zero = [item for item in result.observations if item.page_index == 0]
        self.assertTrue(
            all(
                relation.relation_type
                not in {RelationType.READING_NEXT, RelationType.READING_PREVIOUS}
                for item in page_zero
                for relation in item.relations
            )
        )
        self.assertTrue(
            all(
                any(
                    record["status"] == "incomplete_page_reading_order"
                    for record in item.metadata["paddle_structural_ref_resolution"]
                )
                for item in page_zero
            )
        )


if __name__ == "__main__":
    unittest.main()
