"""A document's own title survives all the way into the tree as a title.

MinerU 3.x reports a document title separately from a section heading, but the
projection only derived ``text_level`` for ``BlockType.TITLE``. A ``DOC_TITLE``
block therefore reached the tree as ordinary body text with no level, and the
tree ended up with no title for the document at all -- the summariser, handed
the document's own name and nothing else, called it "nonsensical or corrupted".

1368 tests passed while that was true, because none of them followed a title
from a canonical block through to a node type. This one does.
"""

import unittest

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.indexing.tree.nodes import create_node_by_type
from document_services.evidence.projections.bookrag import project_canonical_blocks
from document_services.domain.evidence import BlockType, RepresentationKind

from tests.test_bookrag_projection import (
    canonical_block,
    document,
    representation,
)


class DocumentTitleReachesTree(unittest.TestCase):
    def setUp(self) -> None:
        self.document = document()

    def _project(self, block_type, text, metadata=None):
        block = canonical_block(
            self.document,
            "title-block",
            page=0,
            block_type=block_type,
            representations=(representation(RepresentationKind.TEXT, text),),
            reading_order=1.0,
            metadata=metadata,
        )
        (row,) = project_canonical_blocks(self.document, [block])
        return row

    def test_document_title_projects_with_a_level(self):
        row = self._project(BlockType.DOC_TITLE, "Human Eye and the Colourful World")
        self.assertEqual(1, row["text_level"])
        self.assertTrue(row["is_document_title"])

    def test_document_title_becomes_a_title_node(self):
        row = self._project(BlockType.DOC_TITLE, "Human Eye and the Colourful World")
        node = create_node_by_type(row, isTitle=True)
        self.assertEqual(NodeType.TITLE, node.type)
        self.assertEqual(1, node.meta_info.title_level)

    def test_document_title_outranks_a_section_heading(self):
        doc_row = self._project(BlockType.DOC_TITLE, "Human Eye")
        section_row = self._project(
            BlockType.TITLE, "Defects of Vision", metadata={"level": 2}
        )
        self.assertLess(
            doc_row["text_level"],
            section_row["text_level"],
            "a section heading must not outrank the document's own title",
        )

    def test_unlevelled_heading_still_declares_itself_a_heading(self):
        """MinerU labelling a heading is evidence; discarding it is not free."""

        row = self._project(BlockType.TITLE, "Scattering of Light")
        self.assertIn("text_level", row)
        node = create_node_by_type(row, isTitle=True)
        self.assertEqual(NodeType.TITLE, node.type)


if __name__ == "__main__":
    unittest.main()
