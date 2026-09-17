"""Headings the parser found build hierarchy even when the outline pass is silent.

A heading used to exist only if the LLM outline pass nominated it. MinerU's own
typing was consulted solely as a default *inside* an entry that pass had already
created, so a parser heading it did not renominate became body text.

On a five-page chapter MinerU 3.x reported 14 headings; the tree was built from
the 4 the outline pass happened to agree on, and one mid-document heading
("Defects of Vision") ended up parenting three quarters of the document --
including sections on dispersion, rainbows and atmospheric refraction.

So: parser headings are seeded directly, and a heading with no outline entry
nests under the nearest shallower heading rather than falling to the root.
"""

import unittest
from types import SimpleNamespace

from retrieval_engine.indexes.tree import DocumentTree, NodeType
from retrieval_engine.indexing.tree.builder import construct_tree_index


def _new_tree():
    return DocumentTree(
        meta_dict={"file_name": "source.pdf"},
        cfg=SimpleNamespace(save_path="/tmp"),
    )


def _block(pdf_id, *, text="body", level=None, block_type="text"):
    row = {
        "pdf_id": pdf_id,
        "type": block_type,
        "text": text,
        "page_idx": 0,
        "middle_json": {},
    }
    if level is not None:
        row["text_level"] = level
    return row


def _ids(nodes):
    return [node.meta_info.pdf_id for node in nodes]


class ParserHierarchySurvives(unittest.TestCase):
    def test_parser_headings_become_titles_without_an_outline_entry(self):
        blocks = [
            _block(1, text="Human Eye and the Colourful World", level=1),
            _block(2, text="intro prose"),
            _block(3, text="Defects of Vision", level=2),
            _block(4, text="myopia prose"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        self.assertEqual(NodeType.TITLE, tree.get_node_by_pdf_id(1).type)
        self.assertEqual(NodeType.TITLE, tree.get_node_by_pdf_id(3).type)
        self.assertEqual(NodeType.TEXT, tree.get_node_by_pdf_id(2).type)

    def test_deeper_heading_nests_under_the_shallower_one_above_it(self):
        blocks = [
            _block(1, text="Human Eye", level=1),
            _block(2, text="Defects of Vision", level=2),
            _block(3, text="myopia prose"),
            _block(4, text="Scattering of Light", level=2),
            _block(5, text="tyndall prose"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        doc_title = tree.get_node_by_pdf_id(1)
        # Both level-2 headings sit under the document title, and neither
        # adopts the other -- the failure this whole change is about.
        self.assertEqual([1], _ids(tree.root_node.children))
        self.assertEqual([2, 4], _ids(doc_title.children))
        self.assertEqual([3], _ids(tree.get_node_by_pdf_id(2).children))
        self.assertEqual([5], _ids(tree.get_node_by_pdf_id(4).children))

    def test_a_sibling_heading_does_not_adopt_the_next_section(self):
        """The 10P_CH11 failure, reduced."""

        blocks = [
            _block(1, text="Defects of Vision", level=2),
            _block(2, text="myopia prose"),
            _block(3, text="Atmospheric Refraction", level=2),
            _block(4, text="refraction prose"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        defects = tree.get_node_by_pdf_id(1)
        self.assertNotIn(3, _ids(defects.children))
        self.assertEqual([1, 3], _ids(tree.root_node.children))

    def test_a_code_block_can_open_a_section(self):
        """Title capability used to be restricted to type == "text" exactly."""

        blocks = [
            _block(1, text="listing 1", level=2, block_type="code"),
            _block(2, text="explanation"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        self.assertEqual(NodeType.TITLE, tree.get_node_by_pdf_id(1).type)
        self.assertEqual([2], _ids(tree.get_node_by_pdf_id(1).children))

    def test_every_block_is_still_attached_exactly_once(self):
        blocks = [
            _block(1, text="Doc", level=1),
            _block(2, text="A", level=2),
            _block(3, text="body"),
            _block(4, text="B", level=3),
            _block(5, text="body2"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        seen = []

        def walk(node):
            for child in node.children:
                seen.append(child.meta_info.pdf_id)
                walk(child)

        walk(tree.root_node)
        self.assertEqual([1, 2, 3, 4, 5], sorted(seen))
        self.assertEqual(len(seen), len(set(seen)))


if __name__ == "__main__":
    unittest.main()
