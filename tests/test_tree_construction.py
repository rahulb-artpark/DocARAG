from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from retrieval_engine.indexes.tree import DocumentTree, NodeType
from retrieval_engine.indexing.tree import builder as doc_tree_builder
from retrieval_engine.indexing.tree.builder import (
    TreeConstructionError,
    construct_tree_index,
)


def _block(pdf_id: int, *, block_type: str = "text", text: str | None = None):
    content = {
        "pdf_id": pdf_id,
        "type": block_type,
        "page_idx": (pdf_id - 1) // 3,
        "canonical_block_id": f"block-{pdf_id}",
        "canonical_revision_id": f"revision-{pdf_id}",
        "source_sha256": "a" * 64,
        "geometry": {"bbox": [0, pdf_id, 10, pdf_id + 1]},
    }
    if block_type in {"text", "equation"}:
        content["text"] = text if text is not None else f"content {pdf_id}"
    elif block_type == "table":
        content["table_body"] = f"| row {pdf_id} |"
    elif block_type == "image":
        content["image_caption"] = [f"image {pdf_id}"]
    return content


def _new_tree() -> DocumentTree:
    return DocumentTree(
        meta_dict={"file_name": "source.pdf"},
        cfg=SimpleNamespace(save_path="/tmp"),
    )


def _pdf_ids(nodes):
    return [node.meta_info.pdf_id for node in nodes]


class LosslessTreeConstructionTests(unittest.TestCase):
    def assert_every_source_node_attached_once(self, tree: DocumentTree) -> None:
        attached = Counter(
            child.meta_info.pdf_id
            for parent in tree.nodes
            for child in parent.children
        )
        expected = {node.meta_info.pdf_id for node in tree.get_nodes()}
        self.assertEqual(expected, set(attached))
        self.assertTrue(all(attached[pdf_id] == 1 for pdf_id in expected))
        self.assertTrue(all(node.parent is not None for node in tree.get_nodes()))

    def test_titleless_document_keeps_every_modality_under_root(self):
        blocks = [
            _block(1),
            _block(2, block_type="table"),
            _block(3, block_type="image"),
            _block(4, block_type="equation"),
        ]

        tree = construct_tree_index(_new_tree(), blocks, [])

        self.assertEqual([1, 2, 3, 4], _pdf_ids(tree.root_node.children))
        self.assertEqual(
            [NodeType.TEXT, NodeType.TABLE, NodeType.IMAGE, NodeType.EQUATION],
            [node.type for node in tree.root_node.children],
        )
        self.assert_every_source_node_attached_once(tree)

    def test_preamble_and_nested_sections_preserve_source_evidence(self):
        blocks = [
            _block(1, text="preamble"),
            _block(2, text="source title"),
            _block(3, text="section body"),
            _block(4, text="nested title"),
            _block(5, text="nested body"),
        ]
        # Deliberately shuffled, with stale content/evidence fields.  Only the
        # two hierarchy fields are allowed to influence the source nodes.
        outline = [
            {
                "pdf_id": 4,
                "text_level": 1,
                "parent_id": 2,
                "text": "tampered nested title",
                "canonical_block_id": "tampered-4",
            },
            {
                "pdf_id": 2,
                "text_level": 0,
                "parent_id": None,
                "text": "tampered title",
                "canonical_block_id": "tampered-2",
            },
        ]

        tree = construct_tree_index(_new_tree(), blocks, outline)
        title = tree.get_node_by_pdf_id(2)
        nested = tree.get_node_by_pdf_id(4)

        self.assertEqual([1, 2], _pdf_ids(tree.root_node.children))
        self.assertEqual([3, 4], _pdf_ids(title.children))
        self.assertEqual([5], _pdf_ids(nested.children))
        self.assertEqual("source title", title.meta_info.content)
        self.assertEqual("block-2", title.meta_info.canonical_block_id)
        self.assertEqual("block-4", nested.meta_info.canonical_block_id)
        self.assert_every_source_node_attached_once(tree)

    def test_invalid_title_parents_fall_back_to_root_without_losing_body(self):
        blocks = [_block(pdf_id) for pdf_id in range(1, 7)]
        outline = [
            {"pdf_id": 1, "text_level": 0, "parent_id": None},
            # Forward reference.
            {"pdf_id": 2, "text_level": 1, "parent_id": 3},
            # Self reference.
            {"pdf_id": 3, "text_level": 1, "parent_id": 3},
            # Missing parent.
            {"pdf_id": 5, "text_level": 2, "parent_id": None},
        ]

        tree = construct_tree_index(_new_tree(), blocks, outline)

        self.assertEqual([1, 2, 3, 5], _pdf_ids(tree.root_node.children))
        self.assertEqual([4], _pdf_ids(tree.get_node_by_pdf_id(3).children))
        self.assertEqual([6], _pdf_ids(tree.get_node_by_pdf_id(5).children))
        self.assert_every_source_node_attached_once(tree)

    def test_non_text_outline_entry_cannot_retype_source_evidence(self):
        blocks = [_block(1, block_type="image"), _block(2)]

        tree = construct_tree_index(
            _new_tree(),
            blocks,
            [{"pdf_id": 1, "text_level": 0, "parent_id": None}],
        )

        self.assertEqual(NodeType.IMAGE, tree.get_node_by_pdf_id(1).type)
        self.assertEqual([1, 2], _pdf_ids(tree.root_node.children))
        self.assert_every_source_node_attached_once(tree)

    def test_identity_corruption_fails_closed(self):
        cases = (
            (
                [_block(1), _block(1)],
                [],
                "duplicate pdf_id in pdf_list",
            ),
            (
                [_block(1), _block(3)],
                [],
                "contiguous and ordered",
            ),
            (
                [_block(1)],
                [
                    {"pdf_id": 1, "text_level": 0},
                    {"pdf_id": 1, "text_level": 0},
                ],
                "duplicate pdf_id in title_outline",
            ),
            (
                [_block(1)],
                [{"pdf_id": 2, "text_level": 0}],
                "references missing source",
            ),
            (
                [{**_block(1), "pdf_id": True}],
                [],
                "invalid pdf_id",
            ),
        )
        for blocks, outline, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TreeConstructionError, message):
                    construct_tree_index(_new_tree(), blocks, outline)

    def test_creates_one_node_per_block_in_linear_pass(self):
        blocks = [_block(pdf_id) for pdf_id in range(1, 1001)]
        outline = [
            {"pdf_id": pdf_id, "text_level": 0, "parent_id": None}
            for pdf_id in range(1, 1001, 10)
        ]

        with patch.object(
            doc_tree_builder,
            "create_node_by_type",
            wraps=doc_tree_builder.create_node_by_type,
        ) as create_node:
            tree = construct_tree_index(_new_tree(), blocks, outline)

        self.assertEqual(len(blocks), create_node.call_count)
        self.assertEqual(len(blocks) + 1, len(tree.nodes))
        self.assert_every_source_node_attached_once(tree)


if __name__ == "__main__":
    unittest.main()
