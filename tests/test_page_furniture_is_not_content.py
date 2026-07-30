"""Page furniture is traceable but never answerable.

A running header, a page number and a scanner watermark are printed on the page
and say nothing about the document.  They projected as ``text`` and so were
embedded, summarised and mined for entities as though they were prose -- 9 of
the 85 nodes on a five-page scan.  ``discarded`` was worse: it reached no
projection set at all, so the first one to arrive raised UNREPRESENTABLE_TEXT.
"""

import unittest

from Core.Index.Tree import FURNITURE_NODE_TYPES, NodeType
from Core.pipelines.doc_tree_builder import _TITLE_CAPABLE_TYPES
from Core.pipelines.kg_extractor import _TEXTUAL_KG_NODE_TYPES
from Core.pipelines.tree_node_builder import (
    _TEXT_CONTENT_TYPES,
    create_node_by_type,
)
from Core.pipelines.tree_node_summary import _SUMMARISABLE_NODE_TYPES
from Core.pipelines.vdb_index import _TEXT_NODE_TYPES, _VISUAL_NODE_TYPES
from docarag.adapters.mineru_compat import DECLARED_BLOCK_TYPES
from docarag.bookrag_projection import (
    _NODE_TYPE_BY_BLOCK,
    _TEXT_BLOCKS,
)
from docarag.domain.evidence import BlockType

_FURNITURE_BLOCKS = (
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.ASIDE,
    BlockType.DISCARDED,
)


class FurnitureSurvivesProjection(unittest.TestCase):
    def test_every_furniture_block_has_a_projection_path(self):
        for block_type in _FURNITURE_BLOCKS:
            with self.subTest(block_type=block_type):
                self.assertIn(block_type, _TEXT_BLOCKS)

    def test_furniture_keeps_its_own_node_type_rather_than_passing_as_text(self):
        for block_type in _FURNITURE_BLOCKS:
            with self.subTest(block_type=block_type):
                projected = _NODE_TYPE_BY_BLOCK.get(block_type)
                self.assertIsNotNone(projected, "would project as text")
                self.assertIn(NodeType(projected), FURNITURE_NODE_TYPES)

    def test_the_adapter_still_declares_them_so_provenance_is_kept(self):
        for block_type in _FURNITURE_BLOCKS:
            with self.subTest(block_type=block_type):
                self.assertIn(block_type, DECLARED_BLOCK_TYPES)


class FurnitureReachesTheTreeWithItsOwnType(unittest.TestCase):
    def test_a_footer_becomes_a_footer_node_not_a_text_node(self):
        node = create_node_by_type(
            {"type": "footer", "text": "Scanned with CamScanner", "page_idx": 1},
            False,
        )
        self.assertIs(node.type, NodeType.FOOTER)

    def test_a_discarded_block_no_longer_falls_through_to_text(self):
        node = create_node_by_type({"type": "discarded", "text": "x"}, False)
        self.assertIs(node.type, NodeType.DISCARDED)

    def test_every_furniture_type_is_mapped(self):
        mapped = {
            node_type
            for node_type in _TEXT_CONTENT_TYPES.values()
            if node_type in FURNITURE_NODE_TYPES
        }
        self.assertEqual(mapped, set(FURNITURE_NODE_TYPES))


class FurnitureIsNeverAnswerable(unittest.TestCase):
    def test_furniture_can_never_open_a_section(self):
        for name, node_type in _TEXT_CONTENT_TYPES.items():
            if node_type in FURNITURE_NODE_TYPES:
                with self.subTest(name=name):
                    self.assertNotIn(name, _TITLE_CAPABLE_TYPES)

    def test_prose_types_are_still_title_capable(self):
        self.assertIn("text", _TITLE_CAPABLE_TYPES)
        self.assertIn("code", _TITLE_CAPABLE_TYPES)

    def test_furniture_is_not_indexed(self):
        for node_type in FURNITURE_NODE_TYPES:
            with self.subTest(node_type=node_type):
                self.assertNotIn(node_type, _TEXT_NODE_TYPES)
                self.assertNotIn(node_type, _VISUAL_NODE_TYPES)

    def test_furniture_is_not_summarised(self):
        for node_type in FURNITURE_NODE_TYPES:
            with self.subTest(node_type=node_type):
                self.assertNotIn(node_type, _SUMMARISABLE_NODE_TYPES)

    def test_furniture_yields_no_entities(self):
        for node_type in FURNITURE_NODE_TYPES:
            with self.subTest(node_type=node_type):
                self.assertNotIn(node_type, _TEXTUAL_KG_NODE_TYPES)


class EveryProjectedNodeTypeCanBeHydrated(unittest.TestCase):
    """Hydration must accept every node type the projection can produce.

    These two lists are kept in step by hand, and were not: page furniture gained
    its own projected node type and hydration was never told about it. Nothing
    noticed, because the only path that hydrates a whole section -- a global
    query -- was refusing on its evidence budget before hydration ran. Correcting
    that budget made a section holding two footers and two asides fail with
    "canonical projection produced an unsupported node type".

    Indexing is the opposite case and stays that way: furniture is deliberately
    absent from vdb_index's own _TEXT_NODE_TYPES, because it must never be
    embedded. Hydration has to *validate* what indexing declines to *index*.
    """

    def _projected_types(self):
        from docarag.bookrag_projection import _VISUAL_NODE_TYPE_BY_BLOCK

        return (
            set(_NODE_TYPE_BY_BLOCK.values())
            | set(_VISUAL_NODE_TYPE_BY_BLOCK.values())
            | {"text", "image"}
        )

    def _hydratable_types(self):
        from docarag.evidence_hydration import (
            _TEXT_NODE_TYPES as HYDRATABLE_TEXT,
            _VISUAL_NODE_TYPES as HYDRATABLE_VISUAL,
        )

        # table and equation are matched by their own branches rather than a set.
        return HYDRATABLE_TEXT | HYDRATABLE_VISUAL | {"table", "equation"}

    def test_no_projected_type_is_unhydratable(self):
        missing = sorted(self._projected_types() - self._hydratable_types())
        self.assertEqual(
            missing,
            [],
            "these node types are projected but would fail hydration as "
            "'unsupported node type'",
        )

    def test_furniture_is_hydratable_even_though_it_is_not_indexed(self):
        from docarag.evidence_hydration import _TEXT_NODE_TYPES as HYDRATABLE_TEXT

        for node_type in FURNITURE_NODE_TYPES:
            with self.subTest(node_type=node_type):
                self.assertIn(node_type, HYDRATABLE_TEXT)
                self.assertNotIn(node_type, _TEXT_NODE_TYPES)
