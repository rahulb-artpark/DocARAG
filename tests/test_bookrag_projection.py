from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from docarag.artifact_store import LocalContentAddressedArtifactStore
from docarag.asset_material import AssetMaterializationError
from docarag.bookrag_projection import (
    BOOKRAG_ASSET_LEASE_SCHEMA_VERSION,
    PROJECTION_NAME,
    BookRAGAssetLeaseError,
    BookRAGAssetMaterializationLease,
    BookRAGImageMaterialPolicy,
    BookRAGProjectionError,
    ProjectionBindingError,
    ProjectionErrorCode,
    ProjectionEvidenceLossError,
    ProjectionTableError,
    project_canonical_blocks,
    project_to_bookrag_pdf_list,
)
from docarag.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    CanonicalBlock,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    Point,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "7" * 64


def document(*, source: str = SOURCE_SHA256, pages: int = 4) -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=source,
        mime_type="application/pdf",
        source_uri="file:///source/report.pdf",
        display_name="report.pdf",
        page_count=pages,
    )


def geometry(
    bbox: tuple[float, float, float, float],
    *,
    polygon_only: bool = False,
) -> SourceGeometry:
    x0, y0, x1, y1 = bbox
    return SourceGeometry(
        coordinate_space=CoordinateSpace.NORMALIZED,
        bbox=None if polygon_only else BoundingBox(*bbox),
        polygon=(
            Point(x0, y0),
            Point(x1, y0),
            Point(x1, y1),
            Point(x0, y1),
        )
        if polygon_only
        else (),
        page_width=1.0,
        page_height=1.0,
    )


def representation(
    kind: RepresentationKind,
    content: object,
    media_type: str | None = None,
    *,
    confidence: float | None = 0.9,
    metadata: dict[str, object] | None = None,
) -> Representation:
    defaults = {
        RepresentationKind.HTML: "text/html",
        RepresentationKind.MARKDOWN: "text/markdown",
        RepresentationKind.CSV: "text/csv",
        RepresentationKind.LATEX: "application/x-latex",
        RepresentationKind.MATHML: "application/mathml+xml",
        RepresentationKind.TABLE_GRID: "application/json",
        RepresentationKind.DIAGRAM_GRAPH: "application/json",
        RepresentationKind.STRUCTURED_DATA: "application/json",
    }
    return Representation(
        kind=kind,
        content=content,  # type: ignore[arg-type]
        media_type=media_type or defaults.get(kind, "text/plain"),
        confidence=confidence,
        metadata=metadata or {},
    )


def source_parser(observation_id: str, parser_name: str) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "parser_name": parser_name,
        "parser_version": "1.2.3",
        "model_version": f"{parser_name}-model",
        "config_sha256": "a" * 64,
        "parser_native_id": f"native-{observation_id}",
        "raw_artifact_uri": f"artifact://{parser_name}/{observation_id}.json",
        "transformations": [f"{parser_name}.translate.v1"],
    }


def canonical_block(
    doc: DocumentRef,
    key: str,
    *,
    page: int,
    block_type: BlockType,
    representations: tuple[Representation, ...] = (),
    assets: tuple[AssetRef, ...] = (),
    reading_order: float | None = None,
    semantic_subtype: str | None = None,
    source_ids: tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
    block_geometry: SourceGeometry | None = None,
) -> CanonicalBlock:
    source_ids = source_ids or (f"obs_{key}",)
    return CanonicalBlock(
        canonical_key=key,
        revision=1,
        document_version_id=doc.document_version_id,
        page_id=PageRef(doc.document_version_id, page).page_id,
        page_index=page,
        block_type=block_type,
        representations=representations,
        assets=assets,
        source_observation_ids=source_ids,
        reconciliation_method="fixture_reconciliation",
        quality_status=QualityStatus.ACCEPTED,
        geometry=block_geometry or geometry((0.1, 0.2, 0.8, 0.4)),
        semantic_subtype=semantic_subtype,
        reading_order=reading_order,
        metadata=metadata or {},
    )


class RichBookRAGProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = document()

    def test_rich_table_formula_visual_and_text_project_without_losing_lineage(self):
        title_text = representation(RepresentationKind.TEXT, "Results")
        title = canonical_block(
            self.document,
            "title",
            page=0,
            block_type=BlockType.TITLE,
            representations=(title_text,),
            reading_order=1,
            metadata={"level": 2, "source_sha256": SOURCE_SHA256},
        )

        formula = canonical_block(
            self.document,
            "formula",
            page=0,
            block_type=BlockType.EQUATION,
            representations=(
                representation(RepresentationKind.MATHML, "<math><mi>E</mi></math>"),
                representation(RepresentationKind.LATEX, "E = mc^2", confidence=0.95),
            ),
            assets=(
                AssetRef(
                    AssetKind.REGION_CROP,
                    "file:///tmp/formula%20crop.png",
                    "image/png",
                    "b" * 64,
                ),
            ),
            reading_order=3,
        )

        table_html = representation(
            RepresentationKind.HTML,
            "<table><tbody><tr><td>Q1</td><td>10</td></tr></tbody></table>",
            confidence=0.94,
        )
        table_markdown = representation(
            RepresentationKind.MARKDOWN,
            "| Quarter | Revenue |\n| --- | --- |\n| Q1 | 10 |",
        )
        table_grid = representation(
            RepresentationKind.TABLE_GRID,
            {"rows": [["Quarter", "Revenue"], ["Q1", 10]]},
        )
        caption = representation(RepresentationKind.CAPTION, "Table 1: Revenue")
        footnote = representation(RepresentationKind.FOOTNOTE, "USD millions")
        table_sources = ("obs_table_docling", "obs_table_mineru")
        table = canonical_block(
            self.document,
            "table",
            page=1,
            block_type=BlockType.TABLE,
            representations=(
                table_grid,
                footnote,
                table_markdown,
                table_html,
                caption,
            ),
            assets=(
                AssetRef(
                    AssetKind.REGION_CROP,
                    "artifact://parsers/table.png",
                    "image/png",
                ),
            ),
            reading_order=2,
            source_ids=table_sources,
            metadata={
                "source_sha256": SOURCE_SHA256,
                "selected_observation_id": "obs_table_mineru",
                "anchor_observation_id": "obs_table_docling",
                "primary_representations": {
                    RepresentationKind.HTML.value: table_html.representation_id,
                },
                "source_parsers": [
                    source_parser("obs_table_mineru", "mineru"),
                    source_parser("obs_table_docling", "docling"),
                ],
            },
            block_geometry=geometry((0.05, 0.1, 0.95, 0.6), polygon_only=True),
        )

        diagram = canonical_block(
            self.document,
            "diagram",
            page=1,
            block_type=BlockType.DIAGRAM,
            semantic_subtype="workflow",
            representations=(
                representation(RepresentationKind.OCR_TEXT, "Upload -> parse -> answer"),
                representation(
                    RepresentationKind.DIAGRAM_GRAPH,
                    {"nodes": ["upload", "parse", "answer"], "edges": [[0, 1], [1, 2]]},
                ),
                representation(
                    RepresentationKind.VISUAL_DESCRIPTION,
                    "A three-step ingestion workflow",
                ),
                representation(RepresentationKind.CAPTION, "Figure 2"),
                representation(RepresentationKind.FOOTNOTE, "Arrows show data flow"),
            ),
            assets=(
                AssetRef(
                    AssetKind.EXTRACTED_IMAGE,
                    "images/workflow.png",
                    "image/png",
                    "c" * 64,
                ),
            ),
            reading_order=None,
        )

        projected = project_canonical_blocks(
            self.document, [diagram, table, formula, title]
        )

        self.assertEqual([1, 2, 3, 4], [row["pdf_id"] for row in projected])
        self.assertEqual(
            [title.block_id, formula.block_id, table.block_id, diagram.block_id],
            [row["canonical_block_id"] for row in projected],
        )
        self.assertEqual(
            ["text", "equation", "table", "image"],
            [row["type"] for row in projected],
        )

        title_row, formula_row, table_row, diagram_row = projected
        self.assertEqual(2, title_row["text_level"])
        self.assertEqual("title", title_row["semantic_subtype"])

        self.assertEqual("E = mc^2", formula_row["text"])
        self.assertEqual("latex", formula_row["text_format"])
        self.assertEqual("", formula_row["img_path"])
        self.assertEqual(
            [formula.representations[1].representation_id],
            formula_row["selected_representation_ids"],
        )
        self.assertEqual([], formula_row["selected_asset_ids"])
        self.assertEqual(
            BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY.value,
            formula_row["image_material_policy"],
        )
        self.assertEqual(
            "no_verified_local_image_selected",
            formula_row["image_material_status"],
        )
        self.assertFalse(formula_row["legacy_inspection_only"])

        self.assertEqual(table_html.content, table_row["table_body"])
        self.assertEqual("html", table_row["table_body_source_kind"])
        self.assertEqual(["Table 1: Revenue"], table_row["table_caption"])
        self.assertEqual(["USD millions"], table_row["table_footnote"])
        self.assertEqual("", table_row["img_path"])
        self.assertEqual([0.05, 0.1, 0.95, 0.6], table_row["middle_json"]["bbox"])
        self.assertEqual(list(table.source_observation_ids), table_row["source_observation_ids"])
        self.assertEqual(table.revision_id, table_row["canonical_revision_id"])
        self.assertEqual(
            ["obs_table_docling", "obs_table_mineru"],
            [item["observation_id"] for item in table_row["parser_provenance_list"]],
        )
        self.assertEqual(
            "obs_table_mineru", table_row["parser_provenance"]["selected_observation_id"]
        )
        self.assertIn(PROJECTION_NAME, table_row["transformations"])
        self.assertEqual(len(table.representations), len(table_row["representations"]))
        self.assertEqual(len(table.assets), len(table_row["assets"]))
        self.assertEqual(
            {item.representation_id for item in table.representations},
            set(table_row["canonical_representation_ids"]),
        )
        self.assertEqual(
            {item.asset_id for item in table.assets},
            set(table_row["canonical_asset_ids"]),
        )
        self.assertEqual(
            table_row["canonical_representation_ids"],
            table_row["source_evidence_refs"][0]["representation_ids"],
        )
        self.assertEqual(
            table_row["canonical_asset_ids"],
            table_row["source_evidence_refs"][0]["asset_ids"],
        )
        self.assertEqual(
            {
                table_html.representation_id,
                caption.representation_id,
                footnote.representation_id,
            },
            set(table_row["selected_representation_ids"]),
        )
        self.assertEqual(
            table_row["selected_representation_ids"],
            table_row["source_evidence_refs"][0][
                "selected_representation_ids"
            ],
        )
        self.assertEqual([], table_row["selected_asset_ids"])

        self.assertEqual("workflow", diagram_row["semantic_subtype"])
        self.assertEqual("", diagram_row["img_path"])
        self.assertEqual(
            ["Figure 2", "A three-step ingestion workflow", "Upload -> parse -> answer"],
            diagram_row["image_caption"],
        )
        self.assertEqual(["Arrows show data flow"], diagram_row["image_footnote"])
        self.assertEqual(
            {
                item.representation_id
                for item in diagram.representations
                if item.kind is not RepresentationKind.DIAGRAM_GRAPH
            },
            set(diagram_row["selected_representation_ids"]),
        )
        self.assertEqual([], diagram_row["selected_asset_ids"])
        self.assertEqual(
            "no_verified_local_image_selected",
            diagram_row["image_material_status"],
        )
        self.assertFalse(diagram_row["legacy_inspection_only"])
        self.assertIn(
            RepresentationKind.DIAGRAM_GRAPH.value,
            {item["kind"] for item in diagram_row["representations"]},
        )

        # Projected rows are controller-safe JSON rather than frozen DTO internals.
        json.dumps(projected, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def test_output_is_deterministic_across_input_order_and_alias(self):
        blocks = [
            canonical_block(
                self.document,
                "b",
                page=2,
                block_type=BlockType.PARAGRAPH,
                representations=(representation(RepresentationKind.TEXT, "B"),),
                reading_order=None,
            ),
            canonical_block(
                self.document,
                "a",
                page=2,
                block_type=BlockType.LIST_ITEM,
                representations=(
                    representation(RepresentationKind.TEXT, "Item"),
                    representation(RepresentationKind.MARKDOWN, "- Item"),
                ),
                reading_order=1,
            ),
        ]

        forward = project_canonical_blocks(self.document, blocks)
        reverse = project_canonical_blocks(self.document, list(reversed(blocks)))

        self.assertEqual(forward, reverse)
        self.assertEqual(forward, project_to_bookrag_pdf_list(self.document, blocks))
        self.assertEqual("- Item", forward[0]["text"])
        self.assertEqual("list_item", forward[0]["semantic_subtype"])
        self.assertEqual([1, 2], [row["pdf_id"] for row in forward])
        self.assertEqual(
            [
                {
                    "observation_id": blocks[1].source_observation_ids[0],
                    "provenance_available": False,
                }
            ],
            forward[0]["parser_provenance_list"],
        )

    def test_equal_reading_order_uses_stable_block_id_tie_break(self):
        blocks = [
            canonical_block(
                self.document,
                key,
                page=1,
                block_type=BlockType.PARAGRAPH,
                representations=(representation(RepresentationKind.TEXT, key),),
                reading_order=2,
            )
            for key in ("tie-z", "tie-a", "tie-m")
        ]

        rows = project_canonical_blocks(self.document, list(reversed(blocks)))

        self.assertEqual(
            sorted(block.block_id for block in blocks),
            [row["canonical_block_id"] for row in rows],
        )


class TableFallbackProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = document(pages=8)

    def test_markdown_csv_simple_grid_and_spanned_grid_convert_deterministically(self):
        cases = {
            "markdown": (
                representation(
                    RepresentationKind.MARKDOWN,
                    "| Name | Value |\n| :--- | ---: |\n| A | x\\|y |\n| B | <tag> |",
                ),
                ("<thead>", "x|y", "&lt;tag&gt;"),
            ),
            "csv": (
                representation(RepresentationKind.CSV, 'Name,"x,y"\nA,1'),
                ("<td>x,y</td>", "<td>A</td>"),
            ),
            "rows": (
                representation(
                    RepresentationKind.TABLE_GRID,
                    {"rows": [["A", 1], ["B", True]]},
                ),
                ("<td>1</td>", "<td>true</td>"),
            ),
            "grid": (
                representation(
                    RepresentationKind.TABLE_GRID,
                    {"grid": [["A", 1], ["B", 2]]},
                ),
                ("<td>A</td>", "<td>2</td>"),
            ),
            "spans": (
                representation(
                    RepresentationKind.TABLE_GRID,
                    {
                        "num_rows": 2,
                        "num_cols": 2,
                        "grid": [],
                        "table_cells": [
                            {
                                "text": "Header",
                                "start_row_offset_idx": 0,
                                "end_row_offset_idx": 1,
                                "start_col_offset_idx": 0,
                                "end_col_offset_idx": 2,
                                "column_header": True,
                            },
                            {
                                "text": "A",
                                "start_row_offset_idx": 1,
                                "end_row_offset_idx": 2,
                                "start_col_offset_idx": 0,
                                "end_col_offset_idx": 1,
                            },
                            {
                                "text": "1",
                                "start_row_offset_idx": 1,
                                "end_row_offset_idx": 2,
                                "start_col_offset_idx": 1,
                                "end_col_offset_idx": 2,
                            },
                        ],
                    },
                ),
                ('<th colspan="2">Header</th>', "<td>1</td>"),
            ),
        }

        for index, (name, (table_representation, expected)) in enumerate(cases.items()):
            with self.subTest(name=name):
                block = canonical_block(
                    self.document,
                    name,
                    page=index,
                    block_type=BlockType.TABLE,
                    representations=(table_representation,),
                )
                row = project_canonical_blocks(self.document, [block])[0]
                self.assertTrue(row["table_body"].startswith("<table>"))
                for fragment in expected:
                    self.assertIn(fragment, row["table_body"])
                if name == "markdown":
                    self.assertEqual(2, row["table_body"].count("<th>"))
                    self.assertEqual(2, row["table_body"].count("</th>"))

    def test_markdown_separator_width_and_backslash_content_are_loss_aware(self):
        invalid = canonical_block(
            self.document,
            "bad-markdown-separator",
            page=0,
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    RepresentationKind.MARKDOWN,
                    "| A | B |\n| --- |\n| 1 | 2 |",
                ),
            ),
        )
        with self.assertRaises(ProjectionTableError) as raised:
            project_canonical_blocks(self.document, [invalid])
        self.assertEqual(ProjectionErrorCode.INVALID_TABLE, raised.exception.code)

        valid = canonical_block(
            self.document,
            "markdown-backslashes",
            page=0,
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    RepresentationKind.MARKDOWN,
                    "| Path | Value |\n"
                    "| --- | --- |\n"
                    r"| C:\Users\rahul | left\|right |",
                ),
            ),
        )
        row = project_canonical_blocks(self.document, [valid])[0]
        self.assertIn(r"<td>C:\Users\rahul</td>", row["table_body"])
        self.assertIn("<td>left|right</td>", row["table_body"])

    def test_invalid_nonrectangular_grid_raises_typed_table_error(self):
        block = canonical_block(
            self.document,
            "bad-grid",
            page=0,
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    RepresentationKind.TABLE_GRID,
                    {"rows": [["A"], ["B", "C"]]},
                ),
            ),
        )

        with self.assertRaises(ProjectionTableError) as raised:
            project_canonical_blocks(self.document, [block])

        self.assertEqual(ProjectionErrorCode.INVALID_TABLE, raised.exception.code)
        self.assertEqual(block.block_id, raised.exception.block_id)

    def test_invalid_preferred_representation_falls_back_without_hiding_error(self):
        block = canonical_block(
            self.document,
            "fallback-after-invalid",
            page=0,
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    RepresentationKind.MARKDOWN,
                    "| A | B |\n| --- | --- |\n| 1 |",
                ),
                representation(RepresentationKind.CSV, "A,B\n1,2"),
            ),
        )

        row = project_canonical_blocks(self.document, [block])[0]

        self.assertEqual("csv", row["table_body_source_kind"])
        self.assertIn("<td>1</td>", row["table_body"])
        self.assertEqual(1, len(row["table_conversion_errors"]))
        self.assertIn("markdown:", row["table_conversion_errors"][0])


class AssetOnlyAndFailureProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = document()

    def test_unverified_local_asset_only_rows_require_named_legacy_inspection_policy(self):
        table = canonical_block(
            self.document,
            "image-table",
            page=0,
            block_type=BlockType.TABLE,
            assets=(AssetRef(AssetKind.REGION_CROP, "crops/table.png", "image/png"),),
        )
        formula = canonical_block(
            self.document,
            "image-formula",
            page=1,
            block_type=BlockType.EQUATION,
            assets=(AssetRef(AssetKind.REGION_CROP, "/tmp/formula.png", "image/png"),),
        )
        sketch = canonical_block(
            self.document,
            "sketch",
            page=2,
            block_type=BlockType.UNKNOWN,
            semantic_subtype="sketch",
            assets=(
                AssetRef(
                    AssetKind.EXTRACTED_IMAGE,
                    "file://LOCALHOST/tmp/sketch.webp",
                    "image/webp",
                ),
            ),
        )

        for block in (table, formula, sketch):
            with self.subTest(default_rejects=block.block_id):
                with self.assertRaises(ProjectionEvidenceLossError):
                    project_canonical_blocks(self.document, (block,))

        rows = project_to_bookrag_pdf_list(
            self.document,
            [sketch, formula, table],
            image_material_policy=(
                BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
            ),
        )

        self.assertEqual(["table", "equation", "image"], [row["type"] for row in rows])
        self.assertEqual("crops/table.png", rows[0]["img_path"])
        self.assertEqual("", rows[0]["table_body"])
        self.assertEqual("/tmp/formula.png", rows[1]["img_path"])
        self.assertEqual("", rows[1]["text"])
        self.assertEqual("/tmp/sketch.webp", rows[2]["img_path"])
        self.assertEqual("sketch", rows[2]["semantic_subtype"])
        self.assertTrue(all(row["legacy_inspection_only"] for row in rows))
        self.assertTrue(
            all(
                row["image_material_policy"]
                == BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS.value
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["image_material_status"]
                == "legacy_inspection_unverified_mutable_local_path"
                for row in rows
            )
        )
        self.assertTrue(
            all("asset_materialization_attestation" not in row for row in rows)
        )

    def test_visual_subtype_normalization_and_damaged_table_image_fallback(self):
        flow_chart = canonical_block(
            self.document,
            "flow-chart",
            page=0,
            block_type=BlockType.UNKNOWN,
            semantic_subtype="Flow Chart",
            representations=(
                representation(
                    RepresentationKind.VISUAL_DESCRIPTION,
                    "A two-step flow chart",
                ),
            ),
        )
        damaged_table = canonical_block(
            self.document,
            "damaged-table",
            page=1,
            block_type=BlockType.TABLE,
            representations=(
                representation(
                    RepresentationKind.TABLE_GRID,
                    {"rows": [["A"], ["B", "C"]]},
                ),
            ),
            assets=(
                AssetRef(AssetKind.REGION_CROP, "crops/damaged.png", "image/png"),
            ),
        )

        flow_row = project_canonical_blocks(self.document, [flow_chart])[0]
        with self.assertRaises(ProjectionTableError):
            project_canonical_blocks(self.document, [damaged_table])
        table_row = project_canonical_blocks(
            self.document,
            [damaged_table],
            image_material_policy=(
                BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
            ),
        )[0]

        self.assertEqual("image", flow_row["type"])
        self.assertEqual(["A two-step flow chart"], flow_row["image_caption"])
        self.assertEqual("table", table_row["type"])
        self.assertEqual("crops/damaged.png", table_row["img_path"])
        self.assertEqual("", table_row["table_body"])
        self.assertEqual(1, len(table_row["table_conversion_errors"]))
        self.assertTrue(table_row["legacy_inspection_only"])

    def test_remote_or_traversal_asset_only_evidence_fails_closed(self):
        for key, uri in (
            ("remote", "artifact://worker/image.png"),
            ("traversal", "../outside/image.png"),
            ("encoded-traversal", "%2E%2E/outside/image.png"),
            ("encoded-null", "file:///tmp/image%00.png"),
            ("encoded-control", "file:///tmp/image%0A.png"),
            ("unc", "//server/share/image.png"),
            ("encoded-unc", "%2F%2Fserver/share/image.png"),
            ("invalid-utf8", "file:///tmp/%FF.png"),
            ("invalid-file-uri", "file://[invalid/image.png"),
        ):
            with self.subTest(uri=uri):
                block = canonical_block(
                    self.document,
                    key,
                    page=0,
                    block_type=BlockType.FIGURE,
                    assets=(AssetRef(AssetKind.EXTRACTED_IMAGE, uri, "image/png"),),
                )
                with self.assertRaises(ProjectionEvidenceLossError) as raised:
                    project_canonical_blocks(
                        self.document,
                        [block],
                        image_material_policy=(
                            BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
                        ),
                    )
                self.assertEqual(
                    ProjectionErrorCode.UNREPRESENTABLE_VISUAL,
                    raised.exception.code,
                )

    def test_structured_only_text_fails_instead_of_fabricating_text(self):
        block = canonical_block(
            self.document,
            "structured-text",
            page=0,
            block_type=BlockType.PARAGRAPH,
            representations=(
                representation(RepresentationKind.STRUCTURED_DATA, {"answer": 42}),
            ),
        )

        with self.assertRaises(BookRAGProjectionError) as raised:
            project_canonical_blocks(self.document, [block])

        self.assertEqual(ProjectionErrorCode.UNREPRESENTABLE_TEXT, raised.exception.code)

    def test_document_mismatch_duplicate_block_and_incomplete_provenance_are_typed(self):
        block = canonical_block(
            self.document,
            "bound",
            page=0,
            block_type=BlockType.PARAGRAPH,
            representations=(representation(RepresentationKind.TEXT, "Bound"),),
        )
        other = document(source="8" * 64)
        with self.assertRaises(ProjectionBindingError) as mismatch:
            project_canonical_blocks(other, [block])
        self.assertEqual(ProjectionErrorCode.DOCUMENT_BINDING, mismatch.exception.code)

        with self.assertRaises(ProjectionBindingError) as duplicate:
            project_canonical_blocks(self.document, [block, block])
        self.assertEqual(ProjectionErrorCode.DUPLICATE_BLOCK, duplicate.exception.code)

        incomplete = canonical_block(
            self.document,
            "incomplete-provenance",
            page=0,
            block_type=BlockType.PARAGRAPH,
            representations=(representation(RepresentationKind.TEXT, "Text"),),
            source_ids=("obs_a", "obs_b"),
            metadata={"source_parsers": [source_parser("obs_a", "docling")]},
        )
        with self.assertRaises(ProjectionBindingError) as provenance:
            project_canonical_blocks(self.document, [incomplete])
        self.assertEqual(
            ProjectionErrorCode.INVALID_PROVENANCE, provenance.exception.code
        )

        invalid_selected = canonical_block(
            self.document,
            "invalid-selected-observation",
            page=0,
            block_type=BlockType.PARAGRAPH,
            representations=(representation(RepresentationKind.TEXT, "Text"),),
            metadata={"selected_observation_id": 42},
        )
        with self.assertRaises(ProjectionBindingError) as selected:
            project_canonical_blocks(self.document, [invalid_selected])
        self.assertEqual(
            ProjectionErrorCode.INVALID_PROVENANCE, selected.exception.code
        )

    def test_missing_parser_metadata_retains_sorted_observation_lineage(self):
        block = canonical_block(
            self.document,
            "fallback-lineage",
            page=0,
            block_type=BlockType.PARAGRAPH,
            representations=(representation(RepresentationKind.TEXT, "Lineage"),),
            source_ids=("obs_z", "obs_a"),
        )

        row = project_canonical_blocks(self.document, [block])[0]

        self.assertEqual("obs_a", row["observation_id"])
        self.assertEqual(
            [
                {"observation_id": "obs_a", "provenance_available": False},
                {"observation_id": "obs_z", "provenance_available": False},
            ],
            row["parser_provenance_list"],
        )

    def test_verified_cas_lease_supplies_core_img_path_without_changing_asset_identity(self):
        encoded_image = b"verified image bytes for BookRAG"
        with tempfile.TemporaryDirectory() as directory:
            store = LocalContentAddressedArtifactStore(Path(directory) / "cas")
            stored = store.publish(encoded_image, media_type="image/png")
            canonical_asset = stored.to_asset_ref(kind=AssetKind.REGION_CROP)
            visual = canonical_block(
                self.document,
                "leased-cas-image",
                page=0,
                block_type=BlockType.DIAGRAM,
                assets=(canonical_asset,),
            )
            with self.assertRaises(ProjectionEvidenceLossError):
                project_canonical_blocks(self.document, (visual,))
            lease = BookRAGAssetMaterializationLease(
                store.make_asset_resolver(), (canonical_asset,)
            )

            with lease:
                row = project_canonical_blocks(
                    self.document,
                    (visual,),
                    asset_materialization_lease=lease,
                )[0]
                self.assertEqual(os.fspath(stored.path), row["img_path"])
                self.assertEqual(encoded_image, Path(row["img_path"]).read_bytes())
                self.assertEqual(canonical_asset.asset_id, row["selected_asset_id"])
                self.assertEqual(
                    [canonical_asset.asset_id], row["selected_asset_ids"]
                )
                self.assertEqual(
                    [canonical_asset.asset_id], row["canonical_asset_ids"]
                )
                self.assertEqual(
                    canonical_asset.uri,
                    row["assets"][0]["uri"],
                )
                attestation = row["asset_materialization_attestation"]
                self.assertEqual(
                    BOOKRAG_ASSET_LEASE_SCHEMA_VERSION,
                    attestation["schema_version"],
                )
                self.assertEqual(canonical_asset.uri, attestation["canonical_asset_uri"])
                self.assertEqual(stored.sha256, attestation["sha256"])
                self.assertEqual(os.fspath(stored.path), attestation["materialized_path"])
                self.assertEqual(
                    BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY.value,
                    row["image_material_policy"],
                )
                self.assertEqual(
                    "verified_cas_lease", row["image_material_status"]
                )
                self.assertFalse(row["legacy_inspection_only"])
                with self.assertRaises(ProjectionBindingError):
                    project_canonical_blocks(
                        self.document,
                        (visual,),
                        asset_materialization_lease=lease,
                        image_material_policy=(
                            BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
                        ),
                    )

            self.assertFalse(lease.active)
            with self.assertRaises(BookRAGAssetLeaseError):
                project_canonical_blocks(
                    self.document,
                    (visual,),
                    asset_materialization_lease=lease,
                )

    def test_asset_lease_rejects_arbitrary_local_path_even_when_digest_is_declared(self):
        encoded_image = b"a local file is not canonical CAS material"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "untrusted.png"
            path.write_bytes(encoded_image)
            store = LocalContentAddressedArtifactStore(root / "cas")
            local_asset = AssetRef(
                kind=AssetKind.REGION_CROP,
                uri=os.fspath(path),
                media_type="image/png",
                sha256=hashlib.sha256(encoded_image).hexdigest(),
            )

            with self.assertRaisesRegex(BookRAGAssetLeaseError, "cas://sha256"):
                BookRAGAssetMaterializationLease(
                    store.make_asset_resolver(), (local_asset,)
                )

    def test_image_material_policy_requires_the_explicit_enum(self):
        for value in (
            "legacy_unverified",
            "legacy_inspection_allow_unverified_mutable_local_paths",
            True,
            None,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ProjectionBindingError) as raised:
                    project_canonical_blocks(
                        self.document,
                        (),
                        image_material_policy=value,  # type: ignore[arg-type]
                    )
                self.assertEqual(
                    ProjectionErrorCode.INVALID_INPUT, raised.exception.code
                )

    def test_asset_lease_reverification_fails_when_cas_bytes_change(self):
        encoded_image = b"immutable before extraction"
        with tempfile.TemporaryDirectory() as directory:
            store = LocalContentAddressedArtifactStore(Path(directory) / "cas")
            stored = store.publish(encoded_image, media_type="image/png")
            canonical_asset = stored.to_asset_ref(kind=AssetKind.EXTRACTED_IMAGE)
            lease = BookRAGAssetMaterializationLease(
                store.make_asset_resolver(), (canonical_asset,)
            )
            os.chmod(stored.path, 0o600)
            stored.path.write_bytes(b"tampered after the pre-use attestation")

            with self.assertRaises(AssetMaterializationError):
                lease.close()
            self.assertFalse(lease.active)


class ProjectionImportBoundaryTests(unittest.TestCase):
    def test_import_does_not_load_bookrag_vendor_model_or_storage_modules(self):
        script = r'''
import json
import sys
import docarag.bookrag_projection

forbidden = (
    "Core", "docling", "mineru", "paddle", "psycopg", "qdrant_client",
    "torch", "transformers"
)
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
print(json.dumps(loaded))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertEqual([], json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
