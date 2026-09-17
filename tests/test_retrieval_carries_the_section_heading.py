"""A figure, an equation and a listing arrive with the heading above them.

Each of these states what it shows without stating what it is about. Retrieval
scores every node on its own text, so the heading is never why one of them is
found and never arrives beside it.

Measured on a 1871-page optics manual: the sag equation defining the Zernike
surface ranks first for "the equation that defines the Zernike Surface", and
what reached the answer was the equation alone -- nothing said which surface it
defines. The section's own figure was described by a vision model that had been
shown the crop and not the heading, so it reached the index as "an abstract
teardrop-shaped form" with nothing tying it to Zernike at all.

The heading is one short node away and is the whole context, so retrieval
brings it along. 2956 of that document's 2957 figure, equation and listing
nodes sit directly under their section title.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.rag.gbc.agent import GBCRAG


class _View:
    """A store-backed node view: it names its parent by id."""

    def __init__(self, index_id, node_type, parent_id=None):
        self.index_id = index_id
        self.type = node_type
        self.parent_id = parent_id


class _Node:
    """An in-memory tree node: it holds its parent."""

    def __init__(self, index_id, node_type, parent=None):
        self.index_id = index_id
        self.type = node_type
        self.parent = parent


class _Tree:
    def __init__(self, nodes):
        self._nodes = {node.index_id: node for node in nodes}

    def get_node_by_index_id(self, node_id):
        return self._nodes.get(node_id)


def _expand(nodes, node_ids):
    rag = GBCRAG.__new__(GBCRAG)
    rag.gbc_index = SimpleNamespace(TreeIndex=_Tree(nodes))
    return rag._with_section_titles(list(node_ids))


class TheHeadingComesWithIt(unittest.TestCase):
    def test_a_figure_brings_its_section_title(self):
        nodes = [_View(10, "title"), _View(11, "image", parent_id=10)]
        self.assertEqual([11, 10], _expand(nodes, [11]))

    def test_an_equation_brings_its_section_title(self):
        # The Zernike case: the sag equation alone never says which surface.
        nodes = [_View(10, "title"), _View(11, "equation", parent_id=10)]
        self.assertEqual([11, 10], _expand(nodes, [11]))

    def test_a_code_listing_brings_its_section_title(self):
        nodes = [_View(10, "title"), _View(11, "code", parent_id=10)]
        self.assertEqual([11, 10], _expand(nodes, [11]))

    def test_chart_diagram_and_algorithm_are_covered_too(self):
        for kind in ("chart", "diagram", "algorithm"):
            with self.subTest(kind=kind):
                nodes = [_View(10, "title"), _View(11, kind, parent_id=10)]
                self.assertEqual([11, 10], _expand(nodes, [11]))

    def test_an_in_memory_node_holding_its_parent_works_the_same(self):
        title = _Node(10, NodeType.TITLE)
        figure = _Node(11, NodeType.IMAGE, parent=title)
        self.assertEqual([11, 10], _expand([title, figure], [11]))


class NothingElseIsDisturbed(unittest.TestCase):
    def test_a_paragraph_does_not_pull_its_heading(self):
        # Prose already states its own subject; adding the heading to every
        # retrieved sentence would crowd the evidence for no gain.
        nodes = [_View(10, "title"), _View(11, "text", parent_id=10)]
        self.assertEqual([11], _expand(nodes, [11]))

    def test_a_title_already_retrieved_is_not_added_twice(self):
        nodes = [_View(10, "title"), _View(11, "image", parent_id=10)]
        self.assertEqual([10, 11], _expand(nodes, [10, 11]))

    def test_two_figures_under_one_heading_add_it_once(self):
        nodes = [
            _View(10, "title"),
            _View(11, "image", parent_id=10),
            _View(12, "equation", parent_id=10),
        ]
        self.assertEqual([11, 12, 10], _expand(nodes, [11, 12]))

    def test_the_retrieved_order_is_preserved_and_titles_follow(self):
        nodes = [
            _View(10, "title"),
            _View(11, "image", parent_id=10),
            _View(20, "title"),
            _View(21, "equation", parent_id=20),
        ]
        self.assertEqual([21, 11, 20, 10], _expand(nodes, [21, 11]))

    def test_a_figure_whose_parent_is_not_a_heading_adds_nothing(self):
        nodes = [_View(10, "text"), _View(11, "image", parent_id=10)]
        self.assertEqual([11], _expand(nodes, [11]))

    def test_a_node_parented_to_the_root_has_no_heading_to_add(self):
        nodes = [_View(0, "root"), _View(11, "image", parent_id=0)]
        self.assertEqual([11], _expand(nodes, [11]))

    def test_an_orphan_and_an_unknown_id_are_survived(self):
        nodes = [_View(11, "image", parent_id=None)]
        self.assertEqual([11, 99], _expand(nodes, [11, 99]))

    def test_nothing_retrieved_stays_nothing(self):
        self.assertEqual([], _expand([_View(10, "title")], []))


if __name__ == "__main__":
    unittest.main()
