from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docarag.evidence_projection import build_document_evidence
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


class ProjectionTests(unittest.TestCase):
    """A run's published artifacts become records addressed by their own names."""

    def _run(self, directory: Path) -> Path:
        run = directory / "run"
        run.mkdir()
        (run / "canonical_blocks.json").write_text(
            json.dumps(
                {
                    "canonical_blocks": [
                        {
                            "block_id": "blk-title",
                            "block_type": "title",
                            "page_index": 0,
                            "representations": [{"kind": "text", "content": "Manual"}],
                        },
                        {
                            "block_id": "blk-figure",
                            "block_type": "figure",
                            "page_index": 2,
                            "representations": [
                                {
                                    "kind": "ocr_text",
                                    "content": "Speaker",
                                    "confidence": 0.9,
                                    "metadata": {"producer": "mineru"},
                                }
                            ],
                            "assets": [
                                {
                                    "kind": "region_crop",
                                    "media_type": "image/jpeg",
                                    "sha256": DIGEST,
                                    "uri": f"cas://sha256/{DIGEST}",
                                }
                            ],
                            "semantic_substructures": [
                                {
                                    "kind": "diagram",
                                    "producer_kind": "model_derived",
                                    "availability": "complete",
                                    "source_field_paths": [f"cas://sha256/{DIGEST}"],
                                    "elements": [
                                        {
                                            "element_id": "selem_a",
                                            "local_key": "n1",
                                            "kind": "diagram_element",
                                            "content": "Speaker",
                                        },
                                        {
                                            "element_id": "selem_b",
                                            "local_key": "n2",
                                            "kind": "diagram_element",
                                            "content": "Amplifier",
                                        },
                                    ],
                                    "relations": [
                                        {
                                            "kind": "diagram_connects_to",
                                            "source_element_id": "selem_b",
                                            "target_element_id": "selem_a",
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "block_id": "blk-table",
                            "block_type": "table",
                            "page_index": 3,
                            "representations": [
                                {
                                    "kind": "table_grid",
                                    "content": [["Model", "Aperture"], ["800", "203mm"]],
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run / "tree.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"index_id": 0, "parent_id": None, "type": "root", "meta_info": {}},
                        {
                            "index_id": 1,
                            "parent_id": 0,
                            "type": "title",
                            "meta_info": {
                                "canonical_block_id": "blk-title",
                                "content": "Manual",
                            },
                            "summary": "The manual's title.",
                        },
                        {
                            "index_id": 2,
                            "parent_id": 1,
                            "type": "image",
                            "meta_info": {"canonical_block_id": "blk-figure"},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run / "graph_data.json").write_text(
            json.dumps(
                {
                    "graph": {
                        "nodes": [
                            {
                                "id": "Name: speaker\nType: HARDWARE",
                                "entity_name": "speaker",
                                "entity_type": "HARDWARE",
                                "description": "an output device",
                            },
                            {
                                "id": "Name: amplifier\nType: HARDWARE",
                                "entity_name": "amplifier",
                                "entity_type": "HARDWARE",
                            },
                        ],
                        "links": [
                            {
                                "src_entity_name": "amplifier",
                                "tgt_entity_name": "speaker",
                                "relation_name": "drives",
                                "weight": 8.0,
                                "source_ids": [2],
                            }
                        ],
                    },
                    "tree2kg": {"2": ["Name: speaker\nType: HARDWARE"]},
                }
            ),
            encoding="utf-8",
        )
        return run

    def project(self, directory: Path) -> DocumentEvidence:
        return build_document_evidence(self._run(directory), "rev-1")

    def test_every_layer_of_a_run_is_projected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            self.assertEqual(
                {
                    "assets": 1,
                    "blocks": 3,
                    "representations": 3,
                    "structures": 1,
                    "elements": 2,
                    "tree_nodes": 3,
                    "entities": 2,
                    "mentions": 1,
                    "statements": 1,
                },
                evidence.counts(),
            )

    def test_a_diagram_graph_survives_projection(self):
        # These were produced on every run and discarded before reaching the
        # store, because nothing carried them out of the canonical artifact.
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            figure = next(b for b in evidence.blocks if b.kind == "figure")
            structure = figure.structures[0]
            self.assertEqual("diagram", structure.kind)
            self.assertEqual("model_derived", structure.producer)
            self.assertEqual(
                {"Speaker", "Amplifier"},
                {item.content for item in structure.elements},
            )
            edge = structure.edges[0]
            self.assertEqual(("n2", "n1"), (edge.src_local_key, edge.dst_local_key))

    def test_a_tree_node_is_bound_to_the_block_it_was_built_from(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            bound = {n.node_no: n.block_uid for n in evidence.tree_nodes}
            self.assertEqual({0: None, 1: "blk-title", 2: "blk-figure"}, bound)
            self.assertEqual(
                1, sum(1 for n in evidence.tree_nodes if n.parent_no is None)
            )

    def test_siblings_are_numbered_in_publication_order(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            by_no = {n.node_no: n for n in evidence.tree_nodes}
            self.assertEqual(0, by_no[1].sibling_no)
            self.assertEqual(0, by_no[2].sibling_no)

    def test_a_structured_representation_is_flattened_for_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            table = next(b for b in evidence.blocks if b.kind == "table")
            self.assertEqual("Model | Aperture\n800 | 203mm", table.representations[0].text)

    def test_graph_edges_become_statements_with_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            statement = evidence.statements[0]
            self.assertEqual("drives", statement.predicate)
            self.assertEqual(
                ["subject", "object"], [role.role for role in statement.roles]
            )
            self.assertEqual((2,), tuple(e.node_no for e in statement.evidence))

    def test_an_entity_keeps_one_ref_however_it_is_referenced(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.project(Path(directory))

            refs = {entity.canonical: entity.ref for entity in evidence.entities}
            statement = evidence.statements[0]
            self.assertEqual(refs["amplifier"], statement.roles[0].entity_ref)
            self.assertEqual(refs["speaker"], evidence.mentions[0].entity_ref)

    def test_a_run_without_a_graph_still_projects_its_document(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
            (run / "graph_data.json").unlink()

            evidence = build_document_evidence(run, "rev-1")

            self.assertEqual(3, len(evidence.blocks))
            self.assertEqual((), evidence.entities)
            self.assertEqual((), evidence.statements)
