from __future__ import annotations

import copy
import unittest

from document_services.parser_adapters.paddleocr_vl import (
    ADAPTER_VERSION,
    TRANSLATION_NAME,
    PaddleOCRVLPageAdapter,
)
from document_services.domain import BlockType, ParseStatus, RelationType, RepresentationKind
from document_services.parsing.reconciliation import reconcile_observations
from tests.test_paddleocr_vl_adapter import (
    block,
    page_results_fixture,
    request_fixture,
)


class PaddleOCRVLStructuralRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = request_fixture()

    def parse(self, pages):
        return PaddleOCRVLPageAdapter(
            lambda _request: pages,
            parser_version="paddleocr-3.4.0",
            model_version="PaddleOCR-VL-1.6-0.9B",
            config={"pipeline_version": "v1.6", "backend": "complete-pipeline"},
        ).parse(self.request)

    @staticmethod
    def by_native(result):
        return {
            item.provenance.parser_native_id: item for item in result.observations
        }

    @staticmethod
    def edge_pairs(observation, *types):
        accepted = set(types)
        return {
            (edge.relation_type, edge.target_observation_id)
            for edge in observation.relations
            if not accepted or edge.relation_type in accepted
        }

    def test_contract_version_and_attested_native_order_are_directed(self) -> None:
        result = self.parse(page_results_fixture())
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        self.assertEqual("1.1.0", ADAPTER_VERSION)
        self.assertTrue(TRANSLATION_NAME.endswith(".v2"))
        by_native = self.by_native(result)

        title = by_native["page:0:block:1"]
        text = by_native["page:0:block:2"]
        table = by_native["page:0:block:3"]
        formula = by_native["page:0:block:4"]
        self.assertEqual(
            {(RelationType.READING_NEXT, text.observation_id)},
            self.edge_pairs(title, RelationType.READING_NEXT, RelationType.READING_PREVIOUS),
        )
        self.assertEqual(
            {
                (RelationType.READING_PREVIOUS, title.observation_id),
                (RelationType.READING_NEXT, table.observation_id),
            },
            self.edge_pairs(text, RelationType.READING_NEXT, RelationType.READING_PREVIOUS),
        )
        self.assertEqual(
            {
                (RelationType.READING_PREVIOUS, table.observation_id),
            },
            self.edge_pairs(formula, RelationType.READING_NEXT, RelationType.READING_PREVIOUS),
        )
        edge = next(
            item
            for item in title.relations
            if item.relation_type is RelationType.READING_NEXT
        )
        self.assertEqual(1.0, edge.confidence)
        self.assertEqual(
            "paddleocr_vl_complete_layout_order",
            edge.metadata["assertion_origin"],
        )
        self.assertEqual(
            {"namespace": "block_order", "page_index": 0, "value": 2.0},
            dict(edge.metadata["target_reference"]),
        )

        # Page 1 carries numeric orders in the fixture but does not attest that
        # layout detection ran.  Those values are retained, never promoted.
        diagram = by_native["page:1:block:7"]
        caption = by_native["page:1:block:8"]
        self.assertFalse(diagram.relations)
        self.assertFalse(caption.relations)
        self.assertIn(
            "layout_detection_not_attested",
            {
                item["status"]
                for item in diagram.metadata["paddle_structural_ref_resolution"]
            },
        )

    def test_duplicate_block_order_is_ambiguous_and_never_geometry_tiebroken(self) -> None:
        pages = page_results_fixture()
        pages[0]["parsing_res_list"][2]["block_order"] = 2
        result = self.parse(pages)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        by_native = self.by_native(result)

        for native_id in (
            "page:0:block:1",
            "page:0:block:2",
            "page:0:block:3",
            "page:0:block:4",
        ):
            self.assertFalse(
                self.edge_pairs(
                    by_native[native_id],
                    RelationType.READING_NEXT,
                    RelationType.READING_PREVIOUS,
                )
            )
        for native_id in ("page:0:block:2", "page:0:block:3"):
            records = by_native[native_id].metadata[
                "paddle_structural_ref_resolution"
            ]
            ambiguous = next(
                item
                for item in records
                if item["status"] == "ambiguous_duplicate_block_order"
            )
            self.assertEqual(
                ["page:0:block:2", "page:0:block:3"],
                list(ambiguous["candidate_parser_native_ids"]),
            )

    def test_explicit_id_descriptors_cover_structure_without_semantic_guessing(self) -> None:
        pages = page_results_fixture()
        footnote = block("footnote", "Source note", [30, 680, 600, 720], 9, None)
        pages[1]["parsing_res_list"].append(footnote)
        caption = pages[1]["parsing_res_list"][3]
        caption["parser_native_id"] = "caption-native"
        diagram = pages[1]["parsing_res_list"][2]
        diagram["docarag_structural_relations"] = [
            {
                "relation_type": "has_caption",
                "target_parser_native_id": "caption-native",
                "confidence": 0.98,
            },
            {"relation_type": "parent_of", "target_block_id": 6},
            {"relation_type": "has_footnote", "target_block_id": 9},
            {
                "relation_type": "references",
                "target_block_id": 3,
                "target_page_index": 0,
            },
        ]
        result = self.parse(pages)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        by_native = self.by_native(result)
        source = by_native["page:1:block:7"]
        expected = {
            (RelationType.HAS_CAPTION, by_native["page:1:caption-native"].observation_id),
            (RelationType.PARENT_OF, by_native["page:1:block:6"].observation_id),
            (RelationType.HAS_FOOTNOTE, by_native["page:1:block:9"].observation_id),
            (RelationType.REFERENCES, by_native["page:0:block:3"].observation_id),
        }
        self.assertEqual(expected, self.edge_pairs(source))
        caption_edge = next(
            item
            for item in source.relations
            if item.relation_type is RelationType.HAS_CAPTION
        )
        self.assertEqual(0.98, caption_edge.confidence)
        self.assertEqual(
            "paddleocr_vl_explicit_id_relation",
            caption_edge.metadata["assertion_origin"],
        )
        self.assertEqual(
            {
                "relation_type": "has_caption",
                "target_parser_native_id": "caption-native",
                "confidence": 0.98,
            },
            dict(caption_edge.metadata["descriptor"]),
        )

        # A shared group is native layout grouping, not evidence for caption,
        # parent, or semantic relations.
        pages_without_descriptors = copy.deepcopy(pages)
        del pages_without_descriptors[1]["parsing_res_list"][2][
            "docarag_structural_relations"
        ]
        pages_without_descriptors[1]["parsing_res_list"][2]["group_id"] = 123
        pages_without_descriptors[1]["parsing_res_list"][3]["group_id"] = 123
        no_guess = self.by_native(self.parse(pages_without_descriptors))[
            "page:1:block:7"
        ]
        self.assertFalse(no_guess.relations)

    def test_unresolved_self_invalid_and_ambiguous_refs_remain_auditable(self) -> None:
        pages = page_results_fixture()
        source = pages[0]["parsing_res_list"][0]
        source["docarag_structural_relations"] = [
            {"relation_type": "references", "target_block_id": 999},
            {"relation_type": "references", "target_block_id": 1},
            {"relation_type": "references", "target_global_block_id": 500},
            {"relation_type": "not-a-relation", "target_block_id": 2},
        ]
        pages[0]["parsing_res_list"][1].update(
            {"global_block_id": 500, "global_group_id": 500}
        )
        pages[0]["parsing_res_list"][2].update(
            {"global_block_id": 500, "global_group_id": 500}
        )
        result = self.parse(pages)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        title = self.by_native(result)["page:0:block:1"]
        self.assertFalse(
            self.edge_pairs(title, RelationType.REFERENCES)
        )
        records = [
            item
            for item in title.metadata["paddle_structural_ref_resolution"]
            if item["assertion_origin"] == "paddleocr_vl_explicit_id_relation"
        ]
        self.assertEqual(
            {
                "unresolved",
                "self_target_omitted",
                "ambiguous_native_id",
                "invalid_descriptor",
            },
            {item["status"] for item in records},
        )
        ambiguous = next(
            item for item in records if item["status"] == "ambiguous_native_id"
        )
        self.assertEqual(
            ["page:0:block:2", "page:0:block:3"],
            list(ambiguous["candidate_parser_native_ids"]),
        )
        unresolved = next(item for item in records if item["status"] == "unresolved")
        self.assertEqual(
            {"namespace": "block_id", "page_index": 0, "value": 999},
            dict(unresolved["target_reference"]),
        )

    def test_reference_to_failed_block_is_not_rewired_around_the_gap(self) -> None:
        pages = page_results_fixture()
        pages[0]["parsing_res_list"][0]["docarag_structural_relations"] = [
            {"relation_type": "references", "target_block_id": 2}
        ]
        pages[0]["parsing_res_list"][1]["block_bbox"] = [20, 70, 2000, 130]
        result = self.parse(pages)
        self.assertEqual(ParseStatus.PARTIAL, result.status)
        title = self.by_native(result)["page:0:block:1"]
        self.assertFalse(self.edge_pairs(title, RelationType.REFERENCES))
        records = title.metadata["paddle_structural_ref_resolution"]
        explicit = next(
            item
            for item in records
            if item["assertion_origin"] == "paddleocr_vl_explicit_id_relation"
        )
        order = next(
            item
            for item in records
            if item["assertion_origin"] == "paddleocr_vl_complete_layout_order"
        )
        self.assertEqual("target_not_translated", explicit["status"])
        self.assertEqual("target_not_translated", order["status"])
        self.assertFalse(
            self.edge_pairs(
                title, RelationType.READING_NEXT, RelationType.READING_PREVIOUS
            )
        )

    def test_restructured_global_table_group_proves_cross_page_continuation(self) -> None:
        pages = page_results_fixture()
        root_table = pages[0]["parsing_res_list"][2]
        root_table.update({"global_block_id": 100, "global_group_id": 100})
        continuation = block(
            "table",
            "",
            [20, 680, 450, 760],
            9,
            None,
            global_block_id=200,
            global_group_id=100,
        )
        pages[1]["parsing_res_list"].append(continuation)
        result = self.parse(pages)
        self.assertEqual(ParseStatus.SUCCEEDED, result.status)
        by_native = self.by_native(result)
        root = by_native["page:0:block:3"]
        continued = by_native["page:1:block:9"]
        self.assertEqual(BlockType.TABLE, continued.block_type)
        marker = next(
            item
            for item in continued.representations
            if item.kind is RepresentationKind.STRUCTURED_DATA
        )
        self.assertEqual(
            "paddleocr_vl_table_continuation", marker.content["kind"]
        )
        self.assertIn(
            (RelationType.TABLE_CONTINUED_FROM, root.observation_id),
            self.edge_pairs(continued),
        )
        self.assertIn(
            (RelationType.TABLE_CONTINUES, continued.observation_id),
            self.edge_pairs(root),
        )
        edge = next(
            item
            for item in continued.relations
            if item.relation_type is RelationType.TABLE_CONTINUED_FROM
        )
        self.assertEqual(
            "paddleocr_vl_restructured_table_group",
            edge.metadata["assertion_origin"],
        )
        self.assertEqual(
            {"namespace": "global_block_id", "value": 100},
            dict(edge.metadata["target_reference"]),
        )

    def test_native_assertion_reconciles_with_exact_source_lineage(self) -> None:
        parsed = self.parse(page_results_fixture())
        by_native = self.by_native(parsed)
        source = by_native["page:0:block:1"]
        target = by_native["page:0:block:2"]
        source_assertion = next(
            item
            for item in source.relations
            if item.relation_type is RelationType.READING_NEXT
        )
        reconciled = reconcile_observations(parsed.observations)
        owner = {
            observation_id: canonical
            for canonical in reconciled.canonical_blocks
            for observation_id in canonical.source_observation_ids
        }
        canonical_source = owner[source.observation_id]
        canonical_target = owner[target.observation_id]
        canonical_edge = next(
            item
            for item in canonical_source.relations
            if item.relation_type is RelationType.READING_NEXT
        )
        self.assertEqual(canonical_target.block_id, canonical_edge.target_block_id)
        self.assertEqual(
            (source.observation_id,), canonical_edge.source_observation_ids
        )
        self.assertEqual(
            (source_assertion.relation_id,), canonical_edge.source_relation_ids
        )


if __name__ == "__main__":
    unittest.main()
