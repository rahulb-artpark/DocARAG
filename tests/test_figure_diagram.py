from __future__ import annotations

import unittest

from docarag.domain import (
    AssetKind,
    AssetRef,
    SemanticElementKind,
    SemanticProducerKind,
    SemanticRelationKind,
    SemanticStructureKind,
)
from docarag.figure_diagram import (
    DiagramEdge,
    DiagramGraph,
    DiagramNode,
    diagram_substructure,
)
from docarag.figure_diagram_vlm import DiagramSchema, VLMDiagramExtractor

OBSERVATIONS = ("obs_" + "a" * 32,)


def make_asset() -> AssetRef:
    return AssetRef(
        kind=AssetKind.REGION_CROP,
        uri="cas://sha256/" + "e" * 64,
        media_type="image/png",
        sha256="e" * 64,
    )


def flowchart(edges: tuple[DiagramEdge, ...]) -> DiagramGraph:
    return DiagramGraph(
        kind=SemanticStructureKind.FLOWCHART,
        nodes=(
            DiagramNode(key="n1", label="Receive invoice", role="start"),
            DiagramNode(key="n2", label="Amount matches PO?", role="decision"),
            DiagramNode(key="n3", label="Post to ledger", role="process"),
            DiagramNode(key="n4", label="Flag exception", role="process"),
        ),
        edges=edges,
    )


class DiagramSubstructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.asset = make_asset()

    def build(self, graph: DiagramGraph):
        return diagram_substructure(
            graph, asset=self.asset, source_observation_ids=OBSERVATIONS
        )

    def test_same_label_boxes_fold_into_one_element(self) -> None:
        # A boundary box and its inner label box are one thing drawn twice.
        graph = DiagramGraph(
            kind=SemanticStructureKind.FLOWCHART,
            nodes=(
                DiagramNode(key="n1", label="CPU", role="process"),
                DiagramNode(key="n2", label="CPU", role="process", inside="n1"),
                DiagramNode(key="n3", label="Control Unit", role="process", inside="n2"),
                DiagramNode(key="n4", label="Input", role="input"),
            ),
            edges=(DiagramEdge(source="n4", target="n1"),),
        )

        structure = self.build(graph)

        contents = [e.content for e in structure.elements]
        self.assertEqual(1, contents.count("CPU"))
        contains = [
            r for r in structure.relations
            if r.kind is SemanticRelationKind.DIAGRAM_CONTAINS
        ]
        by_id = {e.element_id: e.content for e in structure.elements}
        self.assertEqual(
            [("CPU", "Control Unit")],
            [(by_id[r.source_element_id], by_id[r.target_element_id]) for r in contains],
        )

    def test_a_node_drawn_inside_another_becomes_a_containment_relation(self) -> None:
        graph = DiagramGraph(
            kind=SemanticStructureKind.FLOWCHART,
            nodes=(
                DiagramNode(key="n1", label="Central Processing Unit", role="process"),
                DiagramNode(key="n2", label="Control Unit", role="process", inside="n1"),
                DiagramNode(key="n3", label="Input Device", role="input"),
            ),
            edges=(DiagramEdge(source="n3", target="n1"),),
        )

        structure = self.build(graph)

        kinds = {r.kind for r in structure.relations}
        self.assertIn(SemanticRelationKind.DIAGRAM_CONTAINS, kinds)
        contains = [
            r for r in structure.relations
            if r.kind is SemanticRelationKind.DIAGRAM_CONTAINS
        ][0]
        by_id = {e.element_id: e.content for e in structure.elements}
        self.assertEqual("Central Processing Unit", by_id[contains.source_element_id])
        self.assertEqual("Control Unit", by_id[contains.target_element_id])

    def test_containment_naming_an_unknown_or_self_parent_is_dropped(self) -> None:
        graph = DiagramGraph(
            kind=SemanticStructureKind.FLOWCHART,
            nodes=(
                DiagramNode(key="n1", label="CPU", role="process", inside="n1"),
                DiagramNode(key="n2", label="Ghost", role="process", inside="n9"),
                DiagramNode(key="n3", label="Input", role="input"),
            ),
            edges=(DiagramEdge(source="n3", target="n1"),),
        )

        structure = self.build(graph)

        kinds = {r.kind for r in structure.relations}
        self.assertNotIn(SemanticRelationKind.DIAGRAM_CONTAINS, kinds)

    def test_flowchart_branches_map_to_typed_relations(self) -> None:
        graph = flowchart(
            (
                DiagramEdge(source="n1", target="n2"),
                DiagramEdge(source="n2", target="n3", label="yes", branch="true"),
                DiagramEdge(source="n2", target="n4", label="no", branch="false"),
            )
        )

        structure = self.build(graph)

        self.assertEqual(SemanticStructureKind.FLOWCHART, structure.kind)
        self.assertEqual(
            SemanticProducerKind.MODEL_DERIVED, structure.producer_kind
        )
        kinds = sorted(relation.kind.value for relation in structure.relations)
        self.assertEqual(
            ["flow_branch_false", "flow_branch_true", "flow_next"], kinds
        )
        self.assertTrue(
            all(
                element.kind is SemanticElementKind.FLOWCHART_ELEMENT
                for element in structure.elements
            )
        )

    def test_every_node_and_edge_is_bound_to_the_crop_it_was_read_from(self) -> None:
        structure = self.build(
            flowchart((DiagramEdge(source="n1", target="n2"),))
        )

        selector_ids = {
            selector.selector_id
            for element in structure.elements
            for selector in element.source_selectors
        }
        for element in structure.elements:
            self.assertEqual(
                self.asset.asset_id, element.source_selectors[0].asset_id
            )
        for relation in structure.relations:
            self.assertTrue(set(relation.source_selector_ids) <= selector_ids)
        self.assertEqual(OBSERVATIONS, structure.source_observation_ids)
        self.assertTrue(structure.is_source_bound)

    def test_edge_label_is_kept_as_a_canonical_edge_type(self) -> None:
        structure = self.build(
            DiagramGraph(
                kind=SemanticStructureKind.DIAGRAM,
                nodes=(
                    DiagramNode(key="a", label="Odoo ERP"),
                    DiagramNode(key="b", label="Data map"),
                ),
                edges=(DiagramEdge(source="a", target="b", label="Feeds Into"),),
            )
        )

        relation = structure.relations[0]
        self.assertIs(SemanticRelationKind.DIAGRAM_CONNECTS_TO, relation.kind)
        self.assertEqual("Feeds Into", relation.label)
        self.assertEqual("feeds_into", relation.metadata["normalized_edge_type"])

    def test_edges_to_unknown_or_identical_nodes_are_discarded(self) -> None:
        # A model may name an endpoint it never declared, or loop a node onto
        # itself; the contract rejects both, so neither may reach it.
        structure = self.build(
            flowchart(
                (
                    DiagramEdge(source="n1", target="n1"),
                    DiagramEdge(source="n1", target="ghost"),
                    DiagramEdge(source="ghost", target="n2"),
                    DiagramEdge(source="n1", target="n2"),
                )
            )
        )

        self.assertEqual(1, len(structure.relations))

    def test_duplicate_edges_are_reported_once(self) -> None:
        structure = self.build(
            flowchart(
                (
                    DiagramEdge(source="n1", target="n2"),
                    DiagramEdge(source="n1", target="n2"),
                )
            )
        )

        self.assertEqual(1, len(structure.relations))

    def test_unlabelled_and_duplicate_nodes_are_discarded(self) -> None:
        structure = self.build(
            DiagramGraph(
                kind=SemanticStructureKind.DIAGRAM,
                nodes=(
                    DiagramNode(key="a", label="Kept"),
                    DiagramNode(key="a", label="Duplicate key"),
                    DiagramNode(key="b", label="   "),
                    DiagramNode(key="", label="No key"),
                ),
                edges=(),
            )
        )

        self.assertEqual(1, len(structure.elements))
        self.assertEqual("Kept", structure.elements[0].content)

    def test_a_graph_with_no_usable_node_yields_nothing(self) -> None:
        graph = DiagramGraph(
            kind=SemanticStructureKind.DIAGRAM,
            nodes=(DiagramNode(key="", label=""),),
            edges=(),
        )

        self.assertIsNone(self.build(graph))

    def test_a_non_graph_structure_kind_is_refused(self) -> None:
        graph = DiagramGraph(
            kind=SemanticStructureKind.TABLE,
            nodes=(DiagramNode(key="a", label="Cell"),),
            edges=(),
        )

        with self.assertRaisesRegex(ValueError, "unsupported diagram structure"):
            self.build(graph)


class _FakeVisionModel:
    def __init__(self, payload: DiagramSchema) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def generate_json(self, prompt_or_memory, images=None, schema=None):
        self.prompts.append(prompt_or_memory)
        return self.payload


class VLMDiagramExtractorTest(unittest.TestCase):
    def test_a_flowchart_is_returned_as_a_typed_graph(self) -> None:
        vlm = _FakeVisionModel(
            DiagramSchema(
                kind="flowchart",
                nodes=[
                    {"key": "n1", "label": "Start", "role": "start"},
                    {"key": "n2", "label": "Check", "role": "decision"},
                ],
                edges=[{"source": "n1", "target": "n2", "branch": "true"}],
            )
        )

        graph = VLMDiagramExtractor(vlm)(b"png-bytes", "Start Check")

        self.assertEqual(SemanticStructureKind.FLOWCHART, graph.kind)
        self.assertEqual(("Start", "Check"), tuple(n.label for n in graph.nodes))
        self.assertEqual("true", graph.edges[0].branch)
        self.assertIn("Start Check", vlm.prompts[0])

    def test_a_figure_with_no_topology_yields_nothing(self) -> None:
        vlm = _FakeVisionModel(DiagramSchema(kind="none"))

        self.assertIsNone(VLMDiagramExtractor(vlm)(b"png-bytes", ""))

    def test_a_declared_graph_without_nodes_yields_nothing(self) -> None:
        vlm = _FakeVisionModel(DiagramSchema(kind="diagram", nodes=[]))

        self.assertIsNone(VLMDiagramExtractor(vlm)(b"png-bytes", ""))

    def test_ungrounded_prompt_omits_the_ocr_section(self) -> None:
        vlm = _FakeVisionModel(DiagramSchema(kind="none"))

        VLMDiagramExtractor(vlm)(b"png-bytes", "")

        self.assertNotIn("authoritative", vlm.prompts[0])


if __name__ == "__main__":
    unittest.main()
