"""A figure's subject is in the heading above it, not in its own words.

The Wolter Surface figure's summary is accurate and says what the picture is:
"a 3D rendered, smooth, matte gray tapered cylinder that widens at one end".
Asking "show me the image of Wolter Surface, its parameters and the equation
that defines it" still returned twenty text references and no image. The figure
was outside the top 200 of every one of the four legs, while the first hit of
the ``text`` leg was its own sibling paragraph and the first hit of ``context``
was its own section title.

Measured against the live index (cut-offs 0.632 for ``text``, 0.570 for
``context``):

    0.5001  the ``text`` vector as indexed    (summary + pixel description)
    0.5680  the summary by itself
    0.8466  heading + section summary

The subject word is in the heading and nowhere else on the node. Blending the
heading into ``text`` would answer that question and break the other one: the
query describing the picture scored 0.8317 against the figure's own prose and
0.6205 -- below the cut-off -- once the heading was mixed in. So the section
becomes its own vector, in the ``context`` slot that titles and equations
already use and figures left empty, and each leg keeps the question it is good
at.

This also reaches a figure whose own summary is wrong, which no arrangement of
the node's own words can do. The Zernike Surface figure's summary argues the
picture is *not* a Zernike surface; its section scores 0.8323.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from retrieval_engine.indexes.tree import DocumentTree, NodeType, TreeNode
from retrieval_engine.indexing.vector import builder as vdb_index

HEADING = "2.3.2.2.93. Wolter Surface"
SECTION_SUMMARY = (
    "Defines the Wolter Surface as a radially symmetric shell object used in "
    "X-ray telescope optics."
)


def _tree(node_type, *, section_summary=SECTION_SUMMARY, parent_type=NodeType.TITLE, **meta):
    tree = DocumentTree(
        meta_dict={
            "file_name": "optics.pdf",
            "source_sha256": "c" * 64,
            "document_version_id": "document-version-9",
        },
        cfg=SimpleNamespace(save_path="/tmp"),
    )
    section = TreeNode({"pdf_id": 1, "page_idx": 0, "content": HEADING})
    section.type = parent_type
    section.summary = section_summary
    tree.add_node(section)
    tree.root_node.add_child(section)

    child = TreeNode({"pdf_id": 2, "page_idx": 0, **meta})
    child.type = node_type
    child.summary = meta.get("summary", "")
    tree.add_node(child)
    section.add_child(child)
    return tree, child


def _vectors(node):
    return vdb_index._node_vectors(node, node.type)[0]


class AFigureCarriesItsSection(unittest.TestCase):
    def test_the_heading_and_section_summary_become_the_context_vector(self):
        _, figure = _tree(
            NodeType.IMAGE,
            summary="A 3D rendered, smooth, matte gray tapered cylinder.",
        )

        self.assertEqual(f"{HEADING}\n{SECTION_SUMMARY}", _vectors(figure)["context"])

    def test_the_section_does_not_displace_the_figure_s_own_words(self):
        # The whole point of a separate slot: the query that describes the
        # picture still has an undiluted vector to match.
        _, figure = _tree(
            NodeType.IMAGE,
            summary="A 3D rendered, smooth, matte gray tapered cylinder.",
        )

        vectors = _vectors(figure)

        self.assertIn("A 3D rendered, smooth, matte gray tapered cylinder.", vectors["text"])
        self.assertNotIn(HEADING, vectors["text"])

    def test_a_figure_whose_own_summary_is_wrong_is_still_reachable(self):
        # Node 8297: the summary argues the picture is not what the section
        # says it is, so only the section can retrieve it.
        _, figure = _tree(
            NodeType.IMAGE,
            summary="This visual description is inconsistent with the title.",
        )

        self.assertIn("Wolter Surface", _vectors(figure)["context"])

    def test_charts_and_diagrams_get_it_too(self):
        for node_type in (NodeType.CHART, NodeType.DIAGRAM):
            with self.subTest(node_type=node_type):
                _, node = _tree(node_type, summary="A plotted curve.")

                self.assertIn(HEADING, _vectors(node)["context"])


class TheOtherTypesThatShowWithoutNaming(unittest.TestCase):
    def test_a_table_carries_its_section(self):
        _, table = _tree(NodeType.TABLE, summary="Parameters and their meanings.")

        self.assertIn(HEADING, _vectors(table)["context"])

    def test_a_listing_carries_its_section(self):
        _, listing = _tree(
            NodeType.CODE,
            content="<zerntype> <numterms> <datatype>",
            summary="The INT Zernike file format.",
        )

        self.assertIn(HEADING, _vectors(listing)["context"])

    def test_an_equation_keeps_its_neighbourhood_instead(self):
        # Measured separately and deliberately: an equation's context is the
        # sentences around it, and the section must not overwrite that.
        tree, equation = _tree(NodeType.EQUATION, content="r^2 = ...")
        sibling = TreeNode(
            {"pdf_id": 3, "page_idx": 0, "content": "where c is the curvature."}
        )
        sibling.type = NodeType.TEXT
        tree.add_node(sibling)
        equation.parent.add_child(sibling)

        context = _vectors(equation).get("context", "")

        self.assertNotIn(SECTION_SUMMARY, context)

    def test_a_paragraph_does_not_get_one(self):
        # A paragraph already is the section's prose. Giving 12,000 of them a
        # near-identical section vector would crowd out the leg they share.
        _, paragraph = _tree(NodeType.TEXT, content="The shell has a radius.")

        self.assertNotIn("context", _vectors(paragraph))


class TheSectionHasToBeThere(unittest.TestCase):
    def test_a_figure_under_something_other_than_a_heading_gets_no_context(self):
        _, figure = _tree(
            NodeType.IMAGE, parent_type=NodeType.TEXT, summary="A tapered cylinder."
        )

        self.assertNotIn("context", _vectors(figure))

    def test_a_heading_with_no_summary_still_contributes_its_words(self):
        _, figure = _tree(NodeType.IMAGE, section_summary="", summary="A cylinder.")

        self.assertEqual(HEADING, _vectors(figure)["context"])

    def test_a_figure_with_nothing_of_its_own_is_not_rescued_by_its_section(self):
        # Otherwise every contentless figure in a section becomes a point
        # near-identical to that section's own, in the same leg.
        _, figure = _tree(NodeType.IMAGE)

        self.assertEqual({}, _vectors(figure))

    def test_an_empty_section_yields_no_vector_rather_than_an_empty_one(self):
        # An empty string embeds to whatever the model returns for "", which is
        # one shared point every figure would land on.
        tree, figure = _tree(NodeType.IMAGE, summary="A cylinder.")
        figure.parent.meta_info.content = ""
        figure.parent.summary = ""

        self.assertNotIn("context", _vectors(figure))


class TheSchemaSaysItChanged(unittest.TestCase):
    """Readiness never recomputes the projection.

    It compares the configured vector *names*, which a change to what goes in
    a slot does not move. So a projection change that is not accompanied by a
    schema bump reaches no document: every existing index passes as ready.
    The version is not pinned here -- pinning it means editing this test on
    every bump, which is the opposite of a guard -- but the two constants have
    to move together, since one seals the attestation and the other the
    projection stream inside it.
    """

    def test_both_schema_constants_carry_the_same_version(self):
        index_version = vdb_index.TREE_VECTOR_INDEX_SCHEMA.rsplit(".", 1)[-1]
        projection_version = vdb_index.TREE_VECTOR_PROJECTION_SCHEMA.rsplit(".", 1)[-1]

        self.assertEqual(index_version, projection_version)

    def test_the_version_is_past_the_shape_without_section_context(self):
        version = int(vdb_index.TREE_VECTOR_INDEX_SCHEMA.rsplit(".v", 1)[-1])

        self.assertGreaterEqual(version, 8)


if __name__ == "__main__":
    unittest.main()
