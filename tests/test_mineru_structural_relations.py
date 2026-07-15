from __future__ import annotations

import unittest

from docarag.adapters.mineru_compat import MinerUNormalizedAdapter
from docarag.domain import (
    BlockType,
    ParserModality,
    RelationType,
    RepresentationKind,
)
from tests.test_parser_contracts import request_fixture


class MinerUStructuralRelationTests(unittest.TestCase):
    def adapter(self, blocks):
        return MinerUNormalizedAdapter(
            lambda _request: blocks,
            parser_version="3.4.4-fixture",
            model_version="mineru-fixture",
            config={"backend": "pipeline", "table": True, "formula": True},
        )

    @staticmethod
    def by_native(result):
        return {
            observation.provenance.parser_native_id: observation
            for observation in result.observations
        }

    def test_explicit_ids_and_complete_native_order_become_directed_relations(self):
        blocks = [
            {
                "type": "title",
                "block_id": "section",
                "text": "Section",
                "page_idx": 0,
                "reading_order": 0,
                "child_ids": ["body"],
                "reference_id": "remote",
            },
            {
                "type": "text",
                "block_id": "body",
                "text": "Body",
                "page_idx": 0,
                "reading_order": 1,
                "parent_id": {"block_id": "section", "confidence": 0.91},
            },
            {
                "type": "image",
                "block_id": "figure",
                "image_caption": ["Figure"],
                "img_path": "/tmp/figure.png",
                "page_idx": 0,
                "reading_order": 2,
                "caption_id": "caption",
                "footnote_id": "note",
            },
            {
                "type": "text",
                "block_id": "caption",
                "text": "Figure caption",
                "page_idx": 0,
                "reading_order": 3,
                "caption_of_id": "figure",
            },
            {
                "type": "footnote",
                "block_id": "note",
                "text": "Figure note",
                "page_idx": 0,
                "reading_order": 4,
                "footnote_of_id": "figure",
            },
            {
                "type": "text",
                "block_id": "remote",
                "text": "Referenced on another page",
                "page_idx": 1,
            },
        ]
        result = self.adapter(blocks).parse(request_fixture())
        by_native = self.by_native(result)
        section = by_native["block_id:section"]
        body = by_native["block_id:body"]
        figure = by_native["block_id:figure"]
        caption = by_native["block_id:caption"]
        note = by_native["block_id:note"]
        remote = by_native["block_id:remote"]

        def edges(observation):
            return {
                (relation.relation_type, relation.target_observation_id)
                for relation in observation.relations
            }

        self.assertIn((RelationType.PARENT_OF, body.observation_id), edges(section))
        self.assertIn((RelationType.REFERENCES, remote.observation_id), edges(section))
        self.assertIn((RelationType.READING_NEXT, body.observation_id), edges(section))
        self.assertIn((RelationType.CHILD_OF, section.observation_id), edges(body))
        self.assertIn((RelationType.HAS_CAPTION, caption.observation_id), edges(figure))
        self.assertIn((RelationType.HAS_FOOTNOTE, note.observation_id), edges(figure))
        self.assertIn((RelationType.CAPTION_OF, figure.observation_id), edges(caption))
        self.assertIn((RelationType.FOOTNOTE_OF, figure.observation_id), edges(note))

        parent = next(
            relation
            for relation in body.relations
            if relation.relation_type is RelationType.CHILD_OF
        )
        self.assertEqual(0.91, parent.confidence)
        self.assertEqual(
            "mineru_explicit_native_reference",
            parent.metadata["assertion_origin"],
        )
        self.assertEqual("parent_id", parent.metadata["source_field"])
        reference_record = next(
            record
            for record in section.metadata["mineru_structural_ref_resolution"]
            if record["source_field"] == "reference_id"
        )
        self.assertEqual(
            "resolved_globally_unique_cross_page", reference_record["status"]
        )

    def test_unresolved_ambiguous_container_and_invalid_refs_are_retained_not_guessed(self):
        blocks = [
            {
                "type": "text",
                "block_id": "source",
                "text": "Source",
                "page_idx": 0,
                "reference_ids": [
                    "duplicate",
                    "missing",
                    {"group_id": 7},
                    {"not_an_id": "x"},
                ],
                "parent_id": 0,
            },
            {
                "type": "text",
                "block_id": "duplicate",
                "text": "First duplicate",
                "page_idx": 0,
            },
            {
                "type": "text",
                "block_id": "duplicate",
                "text": "Second duplicate",
                "page_idx": 0,
            },
        ]
        result = self.adapter(blocks).parse(request_fixture())
        source = self.by_native(result)["block_id:source"]
        self.assertFalse(source.relations)
        records = source.metadata["mineru_structural_ref_resolution"]
        statuses = {record["status"] for record in records}
        self.assertEqual(
            {"ambiguous_same_page", "unresolved", "container_only", "invalid_reference"},
            statuses,
        )
        invalid = next(record for record in records if record["status"] == "invalid_reference")
        self.assertEqual("missing_supported_id", invalid["invalid_reason"])
        unresolved = next(record for record in records if record["status"] == "unresolved")
        self.assertEqual("missing", unresolved["target_ref"])

    def test_partial_or_ambiguous_order_emits_no_reading_edges(self):
        partial = [
            {
                "type": "text",
                "block_id": "a",
                "text": "A",
                "page_idx": 0,
                "reading_order": 0,
            },
            {"type": "text", "block_id": "b", "text": "B", "page_idx": 0},
        ]
        result = self.adapter(partial).parse(request_fixture())
        self.assertTrue(all(not item.relations for item in result.observations))
        statuses = {
            record["status"]
            for item in result.observations
            for record in item.metadata["mineru_structural_ref_resolution"]
        }
        self.assertEqual(
            {"incomplete_page_reading_order", "missing_reading_order"}, statuses
        )

        duplicate = [dict(item, reading_order=1) for item in partial]
        duplicate_result = self.adapter(duplicate).parse(request_fixture())
        self.assertTrue(
            all(not item.relations for item in duplicate_result.observations)
        )
        self.assertEqual(
            {"ambiguous_page_reading_order"},
            {
                record["status"]
                for item in duplicate_result.observations
                for record in item.metadata["mineru_structural_ref_resolution"]
            },
        )

    def test_explicit_table_continuations_resolve_across_pages_with_bounded_types(self):
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
            {
                "type": "text",
                "block_id": "not-table",
                "text": "Not a table",
                "page_idx": 1,
                "next_table_id": "table-1",
            },
        ]
        result = self.adapter(blocks).parse(request_fixture())
        by_native = self.by_native(result)
        first = by_native["block_id:table-1"]
        second = by_native["block_id:table-2"]
        non_table = by_native["block_id:not-table"]
        self.assertIn(
            (RelationType.TABLE_CONTINUES, second.observation_id),
            {(item.relation_type, item.target_observation_id) for item in first.relations},
        )
        self.assertIn(
            (RelationType.TABLE_CONTINUED_FROM, first.observation_id),
            {(item.relation_type, item.target_observation_id) for item in second.relations},
        )
        self.assertFalse(non_table.relations)
        self.assertEqual(
            "invalid_source_type",
            non_table.metadata["mineru_structural_ref_resolution"][0]["status"],
        )

    def test_explicit_high_fidelity_modalities_are_preserved_without_derivation(self):
        blocks = [
            {
                "type": "image",
                "sub_type": "workflow",
                "block_id": "diagram",
                "ocr_text": "Upload -> Parse -> Retrieve",
                "diagram_graph": {"nodes": ["upload", "parse", "retrieve"]},
                "page_idx": 0,
            },
            {
                "type": "table",
                "block_id": "table",
                "table_body": "<table><tr><td>A</td></tr></table>",
                "table_grid": {"rows": [["A"]], "cells": [{"row": 0, "col": 0}]},
                "ocr_text": "A",
                "page_idx": 0,
            },
            {
                "type": "equation",
                "block_id": "equation",
                "text": "x^2",
                "mathml": "<math><msup><mi>x</mi><mn>2</mn></msup></math>",
                "page_idx": 1,
            },
            {
                "type": "chart",
                "block_id": "chart",
                "content": "Q1: 3",
                "chart_data": {"quarters": ["Q1"], "values": [3]},
                "page_idx": 1,
            },
        ]
        adapter = self.adapter(blocks)
        result = adapter.parse(request_fixture())
        by_native = self.by_native(result)
        diagram = by_native["block_id:diagram"]
        table = by_native["block_id:table"]
        equation = by_native["block_id:equation"]
        chart = by_native["block_id:chart"]

        self.assertEqual(BlockType.DIAGRAM, diagram.block_type)
        self.assertEqual("workflow", diagram.semantic_subtype)
        self.assertEqual(
            {RepresentationKind.OCR_TEXT, RepresentationKind.DIAGRAM_GRAPH},
            {item.kind for item in diagram.representations},
        )
        self.assertIn(
            RepresentationKind.TABLE_GRID,
            {item.kind for item in table.representations},
        )
        self.assertIn(
            RepresentationKind.MATHML,
            {item.kind for item in equation.representations},
        )
        self.assertIn(
            RepresentationKind.STRUCTURED_DATA,
            {item.kind for item in chart.representations},
        )
        for observation in result.observations:
            for representation in observation.representations:
                if representation.kind in {
                    RepresentationKind.OCR_TEXT,
                    RepresentationKind.MATHML,
                    RepresentationKind.TABLE_GRID,
                    RepresentationKind.DIAGRAM_GRAPH,
                    RepresentationKind.STRUCTURED_DATA,
                }:
                    self.assertEqual(
                        "mineru_explicit_normalized_field",
                        representation.metadata["assertion_origin"],
                    )
        manifest = adapter.manifest()
        self.assertEqual("1.1.0", manifest.adapter_version)
        self.assertIn(ParserModality.DIAGRAM, manifest.modalities)
        self.assertIn(BlockType.DIAGRAM, manifest.block_types)

    def test_geometry_alone_never_creates_structural_edges(self):
        blocks = [
            {
                "type": "text",
                "block_id": "left",
                "text": "Left column",
                "bbox": [0, 0, 400, 900],
                "page_idx": 0,
            },
            {
                "type": "text",
                "block_id": "right",
                "text": "Right column",
                "bbox": [500, 0, 900, 900],
                "page_idx": 0,
            },
        ]
        result = self.adapter(blocks).parse(request_fixture())
        self.assertTrue(all(not item.relations for item in result.observations))


if __name__ == "__main__":
    unittest.main()
