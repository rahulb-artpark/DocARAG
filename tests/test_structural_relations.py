from __future__ import annotations

from dataclasses import replace
import unittest

from retrieval_engine.indexing.tree.nodes import create_node_by_type
from document_services.evidence.projections.bookrag import (
    ProjectionBindingError,
    project_canonical_blocks,
)
from document_services.domain import (
    BoundingBox,
    CanonicalBlock,
    ContractValidationError,
    ObservationRelation,
    QualityStatus,
    RelationType,
    stable_id,
)
from document_services.parsing.reconciliation import (
    ReconciliationIssueCode,
    ReconciliationResult,
    reconcile_observations,
)
from tests.test_docling_adapter import (
    adapter as docling_adapter,
    export_fixture as docling_export,
    request_fixture as docling_request,
)
from tests.test_parser_contracts import fake_observation, request_fixture


class StructuralRelationContractTests(unittest.TestCase):
    def test_relation_types_are_bounded_directed_and_have_explicit_opposites(self):
        self.assertEqual(RelationType.CAPTION_OF, RelationType.HAS_CAPTION.opposite)
        self.assertEqual(RelationType.HAS_CAPTION, RelationType.CAPTION_OF.opposite)
        self.assertEqual(
            RelationType.TABLE_CONTINUED_FROM,
            RelationType.TABLE_CONTINUES.opposite,
        )
        self.assertEqual(
            RelationType.TABLE_CONTINUES,
            RelationType.TABLE_CONTINUED_FROM.opposite,
        )
        target = stable_id("obs", "target")
        forward = ObservationRelation(RelationType.HAS_CAPTION, target)
        reverse_label = ObservationRelation(RelationType.CAPTION_OF, target)
        self.assertNotEqual(forward.relation_id, reverse_label.relation_id)

        with self.assertRaisesRegex(ContractValidationError, "relation_type"):
            ObservationRelation("near", target)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ContractValidationError, "canonical DocARAG ID"):
            ObservationRelation(RelationType.REFERENCES, "vendor-native-7")

    def test_observation_rejects_a_relation_to_itself(self):
        observation = fake_observation(request_fixture())
        relation = ObservationRelation(
            RelationType.REFERENCES, observation.observation_id
        )
        with self.assertRaisesRegex(ContractValidationError, "target itself"):
            replace(observation, relations=(relation,))

    def test_single_observation_shortcut_cannot_silently_drop_relations(self):
        observation = fake_observation(request_fixture())
        relation = ObservationRelation(
            RelationType.REFERENCES, stable_id("obs", "external-target")
        )
        observation = replace(observation, relations=(relation,))
        with self.assertRaisesRegex(
            ContractValidationError, "complete observation set"
        ):
            CanonicalBlock.from_observation(observation)

    def test_legacy_empty_relation_canonical_payload_keeps_its_hash(self):
        block = CanonicalBlock.from_observation(
            fake_observation(request_fixture())
        )
        payload = block.to_dict()
        payload.pop("relations")
        restored = CanonicalBlock.from_dict(payload)
        self.assertEqual(block, restored)
        self.assertEqual(block.content_sha256, restored.content_sha256)


class DoclingStructuralRelationTests(unittest.TestCase):
    def test_explicit_docling_refs_and_body_order_become_observation_relations(self):
        result = docling_adapter().parse(docling_request())
        by_ref = {
            item.provenance.parser_native_id: item for item in result.observations
        }
        picture = by_ref["#/pictures/0"]
        caption = by_ref["#/texts/6"]
        child_text = by_ref["#/texts/5"]
        table = by_ref["#/tables/0"]

        picture_edges = {
            (edge.relation_type, edge.target_observation_id)
            for edge in picture.relations
        }
        self.assertIn(
            (RelationType.HAS_CAPTION, caption.observation_id), picture_edges
        )
        self.assertIn(
            (RelationType.PARENT_OF, child_text.observation_id), picture_edges
        )
        self.assertIn(
            (RelationType.READING_PREVIOUS, table.observation_id), picture_edges
        )
        self.assertIn(
            (
                RelationType.CAPTION_OF,
                picture.observation_id,
            ),
            {
                (edge.relation_type, edge.target_observation_id)
                for edge in caption.relations
            },
        )
        self.assertTrue(
            all(
                edge.metadata["assertion_origin"]
                == "docling_explicit_reference"
                for edge in picture.relations
            )
        )
        self.assertIn(
            "container_only",
            {
                item["status"]
                for item in picture.metadata[
                    "docling_structural_ref_resolution"
                ]
            },
        )
        self.assertEqual(
            "#/body", picture.metadata["docling_structural_refs"]["parent"]
        )

    def test_references_can_cross_pages_only_when_the_target_is_globally_unique(self):
        export = docling_export()
        export["texts"][0]["references"] = [{"$ref": "#/pictures/0"}]
        result = docling_adapter(export).parse(docling_request())
        by_ref = {
            item.provenance.parser_native_id: item for item in result.observations
        }
        title = by_ref["#/texts/0"]
        picture = by_ref["#/pictures/0"]
        edge = next(
            relation
            for relation in title.relations
            if relation.relation_type == RelationType.REFERENCES
        )
        self.assertEqual(picture.observation_id, edge.target_observation_id)
        record = next(
            item
            for item in title.metadata["docling_structural_ref_resolution"]
            if item["source_field"] == "references"
        )
        self.assertEqual("resolved_globally_unique_cross_page", record["status"])

    def test_reconciliation_preserves_direction_and_exact_assertion_lineage(self):
        parsed = docling_adapter().parse(docling_request())
        reconciled = reconcile_observations(parsed.observations)
        owner = {
            source_id: block
            for block in reconciled.canonical_blocks
            for source_id in block.source_observation_ids
        }
        by_ref = {
            item.provenance.parser_native_id: item for item in parsed.observations
        }
        picture_observation = by_ref["#/pictures/0"]
        caption_observation = by_ref["#/texts/6"]
        picture = owner[picture_observation.observation_id]
        caption = owner[caption_observation.observation_id]

        forward = next(
            edge
            for edge in picture.relations
            if edge.relation_type == RelationType.HAS_CAPTION
            and edge.target_block_id == caption.block_id
        )
        reverse = next(
            edge
            for edge in caption.relations
            if edge.relation_type == RelationType.CAPTION_OF
            and edge.target_block_id == picture.block_id
        )
        source_assertion = next(
            edge
            for edge in picture_observation.relations
            if edge.relation_type == RelationType.HAS_CAPTION
            and edge.target_observation_id == caption_observation.observation_id
        )
        self.assertEqual(
            (picture_observation.observation_id,),
            forward.source_observation_ids,
        )
        self.assertEqual((source_assertion.relation_id,), forward.source_relation_ids)
        self.assertNotEqual(forward.relation_id, reverse.relation_id)
        self.assertEqual(
            reconciled, ReconciliationResult.from_json(reconciled.to_json())
        )

    def test_relations_survive_the_bookrag_tree_handoff(self):
        request = docling_request()
        parsed = docling_adapter().parse(request)
        reconciled = reconcile_observations(parsed.observations)
        by_ref = {
            item.provenance.parser_native_id: item for item in parsed.observations
        }
        owner = {
            source_id: block
            for block in reconciled.canonical_blocks
            for source_id in block.source_observation_ids
        }
        picture = owner[by_ref["#/pictures/0"].observation_id]
        rows = project_canonical_blocks(
            request.document, reconciled.canonical_blocks
        )
        row = next(
            item for item in rows if item["canonical_block_id"] == picture.block_id
        )

        self.assertEqual(
            [relation.to_dict() for relation in picture.relations],
            row["canonical_relations"],
        )
        self.assertEqual(
            row["canonical_relations"],
            row["source_evidence_refs"][0]["canonical_relations"],
        )
        node = create_node_by_type(row, isTitle=False)
        self.assertEqual(
            row["canonical_relations"], node.meta_info.canonical_relations
        )
        self.assertEqual(
            row["canonical_representation_ids"],
            node.meta_info.canonical_representation_ids,
        )
        self.assertEqual(
            row["canonical_asset_ids"], node.meta_info.canonical_asset_ids
        )
        with self.assertRaisesRegex(
            ProjectionBindingError, "outside the projected block set"
        ):
            project_canonical_blocks(request.document, (picture,))

    def test_unresolved_target_is_not_guessed_and_marks_block_for_review(self):
        source = fake_observation(request_fixture())
        unresolved = ObservationRelation(
            RelationType.REFERENCES,
            stable_id("obs", "not-in-this-reconciliation"),
            confidence=0.9,
        )
        source = replace(source, relations=(unresolved,))
        result = reconcile_observations((source,))

        self.assertEqual(QualityStatus.NEEDS_REVIEW, result.canonical_blocks[0].quality_status)
        self.assertFalse(result.canonical_blocks[0].relations)
        issue = next(
            item
            for item in result.issues
            if item.code == ReconciliationIssueCode.UNRESOLVED_RELATION_TARGET
        )
        self.assertEqual((source.observation_id,), issue.observation_ids)
        self.assertEqual(
            unresolved.target_observation_id,
            issue.metadata["target_observation_id"],
        )

    def test_matching_parser_assertions_fuse_without_losing_either_source(self):
        base = fake_observation(request_fixture())
        source_a = replace(
            base,
            provenance=replace(base.provenance, parser_native_id="source-a"),
        )
        source_b = replace(
            base,
            provenance=replace(
                base.provenance,
                parser_name="second-parser",
                parser_version="1.0",
                parser_native_id="source-b",
            ),
        )
        target_geometry = replace(
            base.geometry, bbox=BoundingBox(10, 120, 300, 160)
        )
        target_representation = replace(
            base.representations[0], content="Referenced target"
        )
        target_a = replace(
            base,
            provenance=replace(base.provenance, parser_native_id="target-a"),
            geometry=target_geometry,
            representations=(target_representation,),
        )
        target_b = replace(
            target_a,
            provenance=replace(
                base.provenance,
                parser_name="second-parser",
                parser_version="1.0",
                parser_native_id="target-b",
            ),
        )
        relation_a = ObservationRelation(
            RelationType.REFERENCES, target_a.observation_id, confidence=0.8
        )
        relation_b = ObservationRelation(
            RelationType.REFERENCES, target_b.observation_id, confidence=0.9
        )
        source_a = replace(source_a, relations=(relation_a,))
        source_b = replace(source_b, relations=(relation_b,))

        result = reconcile_observations((source_a, source_b, target_a, target_b))
        source_block = next(
            block
            for block in result.canonical_blocks
            if source_a.observation_id in block.source_observation_ids
        )
        target_block = next(
            block
            for block in result.canonical_blocks
            if target_a.observation_id in block.source_observation_ids
        )
        self.assertEqual(1, len(source_block.relations))
        edge = source_block.relations[0]
        self.assertEqual(target_block.block_id, edge.target_block_id)
        self.assertEqual(
            tuple(sorted((source_a.observation_id, source_b.observation_id))),
            edge.source_observation_ids,
        )
        self.assertEqual(
            tuple(sorted((relation_a.relation_id, relation_b.relation_id))),
            edge.source_relation_ids,
        )
        self.assertEqual(
            {
                (source_a.observation_id, relation_a.relation_id),
                (source_b.observation_id, relation_b.relation_id),
            },
            {
                (
                    item["source_observation_id"],
                    item["source_relation_id"],
                )
                for item in edge.metadata["source_assertions"]
            },
        )
        self.assertEqual(0.9, edge.confidence)


if __name__ == "__main__":
    unittest.main()
