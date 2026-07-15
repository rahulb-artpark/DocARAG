from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from docarag.domain.base import ContractValidationError
from docarag.domain.knowledge import (
    EntityMention,
    EvidenceAnchor,
    EvidenceGraphRelease,
    GTLink,
    RelationAssertion,
)


class KnowledgeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table_anchor = EvidenceAnchor(
            canonical_revision_id="block-revision-table",
            representation_id="repr-table-grid",
            table_cell=(2, 3),
        )
        self.formula_anchor = EvidenceAnchor(
            canonical_revision_id="block-revision-formula",
            representation_id="repr-latex",
            char_span=(11, 28),
        )
        self.diagram_anchor = EvidenceAnchor(
            canonical_revision_id="block-revision-diagram",
            asset_id="asset-diagram-crop",
            page_id="page-7",
            bbox=(0.1, 0.2, 0.8, 0.9),
            diagram_element_id="decision-diamond-2",
        )
        self.metric = EntityMention(
            label="Revenue",
            entity_type="metric",
            extraction_revision_id="table-extractor-v1",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
            extraction_confidence=0.91,
        )
        # Equal labels do not imply equal entities: type and evidence remain in identity.
        self.process = EntityMention(
            label="Revenue",
            entity_type="process_step",
            extraction_revision_id="diagram-extractor-v1",
            evidence_anchor_ids=(self.diagram_anchor.anchor_id,),
            extraction_confidence=0.72,
        )
        self.formula = EntityMention(
            label="gross_margin = revenue - cost",
            entity_type="formula",
            extraction_revision_id="formula-extractor-v1",
            evidence_anchor_ids=(self.formula_anchor.anchor_id,),
            extraction_confidence=0.83,
        )
        self.parallel_a = RelationAssertion(
            source_mention_id=self.metric.mention_id,
            target_mention_id=self.process.mention_id,
            relation_type="feeds",
            extraction_revision_id="diagram-extractor-v1",
            evidence_anchor_ids=(self.diagram_anchor.anchor_id,),
            extraction_confidence=0.81,
            assertion_key="extractor-output-17",
        )
        self.parallel_b = RelationAssertion(
            source_mention_id=self.metric.mention_id,
            target_mention_id=self.process.mention_id,
            relation_type="feeds",
            extraction_revision_id="diagram-extractor-v1",
            evidence_anchor_ids=(self.diagram_anchor.anchor_id,),
            extraction_confidence=0.79,
            assertion_key="extractor-output-18",
        )
        self.opposite = RelationAssertion(
            source_mention_id=self.process.mention_id,
            target_mention_id=self.metric.mention_id,
            relation_type="feeds",
            extraction_revision_id="table-extractor-v1",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
            extraction_confidence=0.66,
            assertion_key="extractor-output-19",
        )
        self.metric_link = GTLink(
            graph_object_id=self.metric.mention_id,
            tree_node_id="tree-node-table",
            role="defined_in",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
            link_confidence=0.95,
        )
        self.relation_link = GTLink(
            graph_object_id=self.parallel_a.relation_id,
            tree_node_id="tree-node-diagram",
            role="asserted_in",
            evidence_anchor_ids=(self.diagram_anchor.anchor_id,),
            link_confidence=0.87,
        )
        self.release = EvidenceGraphRelease(
            document_id="document-1",
            document_version_id="document-version-1",
            source_sha256="0" * 64,
            build_spec_sha256="a" * 64,
            canonical_artifact_sha256="b" * 64,
            tree_revision_id="tree-revision-1",
            tree_artifact_sha256="c" * 64,
            graph_revision_id="graph-revision-1",
            graph_artifact_sha256="d" * 64,
            mapping_revision_id="mapping-revision-1",
            mapping_artifact_sha256="e" * 64,
            canonical_revision_ids=(
                "block-revision-table",
                "block-revision-formula",
                "block-revision-diagram",
            ),
            representation_ids=("repr-table-grid", "repr-latex"),
            asset_ids=("asset-diagram-crop",),
            tree_node_ids=("tree-node-table", "tree-node-diagram"),
            anchors=(self.table_anchor, self.formula_anchor, self.diagram_anchor),
            mentions=(self.metric, self.process, self.formula),
            relations=(self.parallel_a, self.parallel_b, self.opposite),
            gt_links=(self.metric_link, self.relation_link),
            metadata={"extractor": {"name": "fixture", "revision": 1}},
        )

    def test_round_trip_determinism_and_immutability(self) -> None:
        restored = EvidenceGraphRelease.from_json(self.release.to_json())
        self.assertEqual(restored, self.release)
        self.assertEqual(restored.to_json(), self.release.to_json())

        reordered = EvidenceGraphRelease(
            document_id=self.release.document_id,
            document_version_id=self.release.document_version_id,
            source_sha256=self.release.source_sha256,
            build_spec_sha256=self.release.build_spec_sha256,
            canonical_artifact_sha256=self.release.canonical_artifact_sha256,
            tree_revision_id=self.release.tree_revision_id,
            tree_artifact_sha256=self.release.tree_artifact_sha256,
            graph_revision_id=self.release.graph_revision_id,
            graph_artifact_sha256=self.release.graph_artifact_sha256,
            mapping_revision_id=self.release.mapping_revision_id,
            mapping_artifact_sha256=self.release.mapping_artifact_sha256,
            canonical_revision_ids=tuple(reversed(self.release.canonical_revision_ids)),
            representation_ids=tuple(reversed(self.release.representation_ids)),
            asset_ids=self.release.asset_ids,
            tree_node_ids=tuple(reversed(self.release.tree_node_ids)),
            anchors=tuple(reversed(self.release.anchors)),
            mentions=tuple(reversed(self.release.mentions)),
            relations=tuple(reversed(self.release.relations)),
            gt_links=tuple(reversed(self.release.gt_links)),
            metadata={"extractor": {"revision": 1, "name": "fixture"}},
        )
        self.assertEqual(reordered.content_sha256, self.release.content_sha256)
        self.assertEqual(reordered.release_id, self.release.release_id)
        changed_tree_bytes = replace(
            self.release, tree_artifact_sha256="f" * 64
        )
        self.assertNotEqual(
            changed_tree_bytes.content_sha256, self.release.content_sha256
        )
        self.assertNotEqual(changed_tree_bytes.release_id, self.release.release_id)
        with self.assertRaises(FrozenInstanceError):
            self.metric.label = "changed"  # type: ignore[misc]

        tampered = self.release.to_dict()
        tampered["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractValidationError, "content_sha256"):
            EvidenceGraphRelease.from_dict(tampered)

    def test_parallel_and_opposite_relations_survive_and_confidences_stay_separate(self) -> None:
        relations = self.release.relations
        self.assertEqual(len(relations), 3)
        self.assertEqual(len({item.relation_id for item in relations}), 3)
        self.assertIn(self.parallel_a, relations)
        self.assertIn(self.parallel_b, relations)
        self.assertIn(self.opposite, relations)
        self.assertNotEqual(self.metric.mention_id, self.process.mention_id)
        self.assertEqual(self.metric.label, self.process.label)
        self.assertEqual(self.metric.extraction_confidence, 0.91)
        self.assertEqual(self.parallel_a.extraction_confidence, 0.81)
        self.assertEqual(self.metric_link.link_confidence, 0.95)

    def test_table_formula_and_diagram_anchor_shapes_round_trip(self) -> None:
        self.assertEqual(self.table_anchor.table_cell, (2, 3))
        self.assertEqual(self.formula_anchor.char_span, (11, 28))
        self.assertEqual(self.diagram_anchor.bbox, (0.1, 0.2, 0.8, 0.9))
        self.assertEqual(
            EvidenceAnchor.from_json(self.diagram_anchor.to_json()), self.diagram_anchor
        )
        with self.assertRaisesRegex(ContractValidationError, "bbox requires page_id"):
            EvidenceAnchor(
                canonical_revision_id="block-revision-diagram",
                asset_id="asset-diagram-crop",
                bbox=(0.1, 0.1, 0.2, 0.2),
            )

    def test_release_rejects_dangling_mentions_relations_and_gt_links(self) -> None:
        dangling_mention = EntityMention(
            label="orphan",
            entity_type="concept",
            extraction_revision_id="fixture-extractor-v1",
            evidence_anchor_ids=("anchor_missing",),
        )
        dangling_relation = RelationAssertion(
            source_mention_id="mention_missing",
            target_mention_id=self.metric.mention_id,
            relation_type="references",
            extraction_revision_id="fixture-extractor-v1",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
        )
        dangling_graph_link = GTLink(
            graph_object_id="mention_missing",
            tree_node_id="tree-node-table",
            role="defined_in",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
        )
        dangling_tree_link = GTLink(
            graph_object_id=self.metric.mention_id,
            tree_node_id="tree-node-missing",
            role="defined_in",
            evidence_anchor_ids=(self.table_anchor.anchor_id,),
        )
        cases = (
            ("mention", {"mentions": self.release.mentions + (dangling_mention,)}),
            ("relation", {"relations": self.release.relations + (dangling_relation,)}),
            ("graph_object_id", {"gt_links": self.release.gt_links + (dangling_graph_link,)}),
            ("tree_node_id", {"gt_links": self.release.gt_links + (dangling_tree_link,)}),
        )
        for expected, changes in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractValidationError, expected):
                    replace(self.release, **changes)

    def test_release_rejects_anchor_outside_declared_canonical_material(self) -> None:
        bad_anchor = EvidenceAnchor(
            canonical_revision_id="block-revision-missing",
            representation_id="repr-table-grid",
        )
        with self.assertRaisesRegex(ContractValidationError, "canonical_revision_id"):
            replace(self.release, anchors=self.release.anchors + (bad_anchor,))


if __name__ == "__main__":
    unittest.main()
