"""A heading the adapter recognises must reach the tree as a heading.

The adapter calls a block a title on any of three signals: the content-list
type, the middle-json type, or a text_level. The tree reads only text_level.
A heading MinerU declares in middle.json but does not level in content_list is
therefore a title in the canonical record and body text in the tree, and the
section it opens never forms.
"""

import unittest

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.indexing.tree.nodes import create_node_by_type
from document_services.parser_adapters.mineru_compat import _block_type
from document_services.evidence.projections.bookrag import project_canonical_blocks
from document_services.domain.evidence import BlockType, RepresentationKind

from tests.test_bookrag_projection import canonical_block, document, representation


class TitleSignalAgreement(unittest.TestCase):
    def test_middle_json_title_without_text_level_is_still_a_title(self):
        block_type, _ = _block_type({"type": "text"}, {"type": "title"})
        self.assertEqual(BlockType.TITLE, block_type)

    def test_content_list_text_level_alone_is_a_title(self):
        block_type, _ = _block_type({"type": "text", "text_level": 1}, {})
        self.assertEqual(BlockType.TITLE, block_type)

    def test_a_title_block_always_projects_a_level(self):
        """Without one the tree cannot tell it apart from a paragraph."""

        doc = document()
        for metadata in ({}, {"level": 3}):
            with self.subTest(metadata=metadata):
                block = canonical_block(
                    doc,
                    "heading",
                    page=0,
                    block_type=BlockType.TITLE,
                    representations=(
                        representation(RepresentationKind.TEXT, "Scattering of Light"),
                    ),
                    reading_order=1.0,
                    metadata=metadata,
                )
                (row,) = project_canonical_blocks(doc, [block])
                self.assertIsNotNone(
                    row.get("text_level"),
                    "a title projected without a level reaches the tree as prose",
                )
                node = create_node_by_type(row, isTitle=True)
                self.assertEqual(NodeType.TITLE, node.type)


if __name__ == "__main__":
    unittest.main()
