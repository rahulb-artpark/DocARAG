from __future__ import annotations

import unittest

from docarag.persistence.contracts import (
    AssetRecord,
    BlockRecord,
    DocumentEvidence,
    ElementEdgeRecord,
    ElementRecord,
    EntityRecord,
    MentionRecord,
    RepresentationRecord,
    StatementEvidenceRecord,
    StatementRecord,
    StatementRoleRecord,
    StructureRecord,
    TreeNodeRecord,
)

DIGEST = "a" * 64


def diagram() -> StructureRecord:
    return StructureRecord(
        kind="diagram",
        producer="model_derived",
        elements=(
            ElementRecord(local_key="n1", kind="diagram_element", content="Speaker"),
            ElementRecord(local_key="n2", kind="diagram_element", content="Amplifier"),
        ),
        edges=(
            ElementEdgeRecord(
                src_local_key="n2", dst_local_key="n1", kind="connects_to"
            ),
        ),
    )


def table() -> StructureRecord:
    return StructureRecord(
        kind="table",
        producer="parser_native",
        elements=(
            ElementRecord(
                local_key="r0c0",
                kind="table_cell",
                content="EdgeHD",
                role="header",
                row_no=0,
                col_no=0,
            ),
            ElementRecord(
                local_key="r1c0", kind="table_cell", content="800", row_no=1, col_no=0
            ),
        ),
    )


class StructureContractTests(unittest.TestCase):
    """A table, a formula and a diagram are the same record differently filled."""

    def test_a_table_and_a_diagram_share_one_shape(self):
        for structure in (table(), diagram()):
            self.assertEqual(2, len(structure.elements))
            self.assertTrue(all(item.local_key for item in structure.elements))

    def test_grid_position_is_carried_only_by_table_cells(self):
        cells = {item.local_key: item for item in table().elements}
        self.assertEqual((0, 0), (cells["r0c0"].row_no, cells["r0c0"].col_no))
        self.assertTrue(all(item.row_no is None for item in diagram().elements))

    def test_a_structure_without_elements_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one element"):
            StructureRecord(kind="diagram", producer="p", elements=())

    def test_an_edge_outside_its_structure_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside its structure"):
            StructureRecord(
                kind="diagram",
                producer="p",
                elements=(ElementRecord(local_key="n1", kind="diagram_element"),),
                edges=(
                    ElementEdgeRecord(
                        src_local_key="n1", dst_local_key="ghost", kind="connects_to"
                    ),
                ),
            )

    def test_an_element_cannot_connect_to_itself(self):
        with self.assertRaisesRegex(ValueError, "endpoints must differ"):
            ElementEdgeRecord(src_local_key="n1", dst_local_key="n1", kind="x")


class StatementContractTests(unittest.TestCase):
    def test_a_binary_edge_is_a_statement_with_two_roles(self):
        statement = StatementRecord(
            predicate="connects_to",
            roles=(
                StatementRoleRecord(entity_ref="e1", role="subject"),
                StatementRoleRecord(entity_ref="e2", role="object"),
            ),
        )

        self.assertEqual(2, len(statement.roles))
        self.assertTrue(statement.directed)

    def test_a_fact_may_carry_more_than_two_participants(self):
        statement = StatementRecord(
            predicate="supplied",
            roles=(
                StatementRoleRecord(entity_ref="supplier", role="agent"),
                StatementRoleRecord(entity_ref="part", role="object"),
                StatementRoleRecord(entity_ref="plant", role="recipient"),
            ),
            properties={"as_of": "2026-Q3"},
        )

        self.assertEqual(3, len(statement.roles))
        self.assertEqual("2026-Q3", statement.properties["as_of"])

    def test_a_statement_with_one_participant_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least two roles"):
            StatementRecord(
                predicate="p",
                roles=(StatementRoleRecord(entity_ref="e1", role="subject"),),
            )

    def test_evidence_must_locate_the_claim(self):
        with self.assertRaisesRegex(ValueError, "locate a node or a block"):
            StatementEvidenceRecord()


class DocumentEvidenceTests(unittest.TestCase):
    def evidence(self) -> DocumentEvidence:
        return DocumentEvidence(
            revision_id="rev-1",
            assets=(
                AssetRecord(
                    sha256=DIGEST,
                    media_type="image/jpeg",
                    byte_size=99,
                    cas_uri=f"cas://sha256/{DIGEST}",
                ),
            ),
            blocks=(
                BlockRecord(
                    uid="b1",
                    kind="figure",
                    block_no=0,
                    page_no=2,
                    representations=(
                        RepresentationRecord(kind="ocr_text", text="Speaker"),
                        RepresentationRecord(
                            kind="visual_description", text="A wiring diagram"
                        ),
                    ),
                    asset_roles=((DIGEST, "region_crop"),),
                    structures=(diagram(),),
                ),
            ),
            tree_nodes=(
                TreeNodeRecord(node_no=0, kind="root"),
                TreeNodeRecord(node_no=1, kind="figure", parent_no=0, block_uid="b1"),
            ),
            entities=(EntityRecord(ref="e1", kind="HARDWARE", canonical="speaker"),),
            mentions=(MentionRecord(entity_ref="e1", node_no=1, surface="Speaker"),),
        )

    def test_counts_report_every_layer_that_was_written(self):
        self.assertEqual(
            {
                "assets": 1,
                "blocks": 1,
                "representations": 2,
                "structures": 1,
                "elements": 2,
                "tree_nodes": 2,
                "entities": 1,
                "mentions": 1,
                "statements": 0,
            },
            self.evidence().counts(),
        )

    def test_local_names_must_be_unique_within_a_revision(self):
        # A projection addresses blocks, nodes and entities by local name, and
        # the writer resolves each to one surrogate; a duplicate would silently
        # bind two different things to the same row.
        base = self.evidence()
        for field, duplicate in (
            ("blocks", base.blocks + base.blocks),
            ("tree_nodes", base.tree_nodes + (base.tree_nodes[0],)),
            ("entities", base.entities + base.entities),
        ):
            with self.assertRaises(ValueError, msg=f"{field} allowed a duplicate"):
                DocumentEvidence(revision_id="rev-1", **{field: duplicate})

    def test_a_representation_must_carry_text_or_an_asset(self):
        with self.assertRaisesRegex(ValueError, "text or an asset"):
            RepresentationRecord(kind="text")

    def test_a_node_cannot_be_its_own_parent(self):
        with self.assertRaisesRegex(ValueError, "own parent"):
            TreeNodeRecord(node_no=3, kind="text", parent_no=3)


if __name__ == "__main__":
    unittest.main()
