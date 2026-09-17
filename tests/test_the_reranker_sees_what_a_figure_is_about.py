"""The reranker has to be told what a figure is, not just what it looks like.

``_node_rerank_text`` projected a node as content + caption + footnote. For a
figure that is the vision model's account of the pixels and nothing else: not
the summary that names the subject, not the heading above it. So the Wolter
Surface figure reached the reranker as "3D rendered image of a smooth, gray
cylindrical object against a gradient purple background".

Asked "show me the image of Wolter Surface, its parameters and the equation that
defines it", the reranker scored that **0.0000** -- the floor, not merely a low
rank -- and the answer said the document contained no image of the Wolter
Surface. Retrieval had already ranked the node first in two of its legs; the
final ten slots are chosen by this projection, and it rejected the figure
outright.

Measured against the ten nodes that actually won those slots:

    content + caption + footnote   0.0000
    + summary                      0.8516   below the weakest winner
    + heading                      0.9961   above seven of the ten
    + heading + summary            0.9922   above seven of the ten
                  weakest winner   0.9297

The same shape of fix as the index's section ``context`` vector, one stage
later, and for the same reason: these node types show something without naming
it, and the name is what a question says.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.rag.gbc.retrieval import (
    _SECTION_RERANK_NODE_TYPES,
    _SUMMARY_RERANK_NODE_TYPES,
    Retriever,
)

HEADING = "2.3.2.2.93. Wolter Surface"
DESCRIPTION = "3D rendered image of a smooth, gray cylindrical object."
SUMMARY = "The Wolter Surface is depicted as a 3D rendered tapered cylinder."


def _node(node_type, *, content=DESCRIPTION, summary=SUMMARY, parent=True,
          parent_type=NodeType.TITLE, heading=HEADING):
    parent_node = (
        SimpleNamespace(type=parent_type, meta_info=SimpleNamespace(content=heading))
        if parent
        else None
    )
    return SimpleNamespace(
        type=node_type,
        summary=summary,
        parent=parent_node,
        meta_info=SimpleNamespace(
            content=content, caption=None, footnote=None,
            table_body=None, semantic_substructures=None,
        ),
    )


class AFigureArrivesNamed(unittest.TestCase):
    def test_the_heading_leads_the_projection(self):
        text = Retriever._node_rerank_text(_node(NodeType.IMAGE))

        self.assertTrue(text.startswith(HEADING), text[:80])

    def test_the_summary_is_included(self):
        # It was never read at all: the projection took content, caption and
        # footnote, and a figure's summary is none of those.
        self.assertIn(SUMMARY, Retriever._node_rerank_text(_node(NodeType.IMAGE)))

    def test_the_pixel_description_is_still_there(self):
        # Added to, not replaced: a question about what the picture looks like
        # still needs it.
        self.assertIn(DESCRIPTION, Retriever._node_rerank_text(_node(NodeType.IMAGE)))

    def test_the_subject_word_reaches_the_reranker(self):
        text = Retriever._node_rerank_text(_node(NodeType.IMAGE))

        self.assertIn("Wolter", text)

    def test_every_kind_that_shows_without_naming_gets_it(self):
        for node_type in _SECTION_RERANK_NODE_TYPES:
            with self.subTest(node_type=node_type):
                self.assertIn(HEADING, Retriever._node_rerank_text(_node(node_type)))


class ProseIsLeftAlone(unittest.TestCase):
    def test_a_paragraph_gets_no_heading(self):
        # It already is the section's prose; prefixing every paragraph with its
        # heading would say the same words to the reranker N times.
        text = Retriever._node_rerank_text(_node(NodeType.TEXT, content="The shell."))

        self.assertNotIn(HEADING, text)

    def test_a_title_gets_no_heading_of_its_own(self):
        text = Retriever._node_rerank_text(
            _node(NodeType.TITLE, content=HEADING, summary="Defines the surface.")
        )

        self.assertEqual(HEADING, text.strip())

    def test_the_two_sets_agree(self):
        self.assertEqual(_SECTION_RERANK_NODE_TYPES, _SUMMARY_RERANK_NODE_TYPES)


class TheHeadingHasToBeThere(unittest.TestCase):
    def test_a_figure_with_no_parent_still_projects(self):
        text = Retriever._node_rerank_text(_node(NodeType.IMAGE, parent=False))

        self.assertIn(DESCRIPTION, text)
        self.assertNotIn(HEADING, text)

    def test_a_parent_that_is_not_a_heading_contributes_nothing(self):
        text = Retriever._node_rerank_text(
            _node(NodeType.IMAGE, parent_type=NodeType.TEXT)
        )

        self.assertNotIn(HEADING, text)

    def test_an_empty_heading_is_not_a_blank_line(self):
        text = Retriever._node_rerank_text(_node(NodeType.IMAGE, heading="   "))

        self.assertTrue(text.startswith(SUMMARY), text[:60])

    def test_a_figure_with_no_summary_still_projects(self):
        text = Retriever._node_rerank_text(_node(NodeType.IMAGE, summary=None))

        self.assertIn(HEADING, text)
        self.assertIn(DESCRIPTION, text)


if __name__ == "__main__":
    unittest.main()
