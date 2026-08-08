"""A node's readings share a point, and each keeps its own name.

The index used to emit one point per representation, so a figure's picture and
its prose were separate candidates in one undifferentiated ranked list. They now
share a point under named vectors: ``text`` is the node's own words, ``context``
the words around or under it, ``image`` its crop.

The property worth guarding is that slots go by **role, never by availability**.
An equation with no text of its own carries ``context`` alone rather than being
promoted into ``text`` -- the two are scored separately because blending them
measured worse, and a promotion would quietly undo that for exactly the nodes
where it matters.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines import vdb_index


def _tree(directory):
    tree = DocumentTree(
        meta_dict={
            "file_name": "optics.pdf",
            "source_sha256": "d" * 64,
            "document_version_id": "document-version-7",
        },
        cfg=SimpleNamespace(save_path=directory),
    )
    section = TreeNode({"pdf_id": 1, "page_idx": 0, "content": "2. Zernike Surface"})
    section.type = NodeType.TITLE
    tree.add_node(section)
    tree.root_node.add_child(section)
    return tree, section


def _child(tree, section, node_type, **meta):
    node = TreeNode({"pdf_id": 9, "page_idx": 0, **meta})
    node.type = node_type
    tree.add_node(node)
    section.add_child(node)
    return node


def _crop(directory):
    path = f"{directory}/f.jpg"
    Path(path).write_bytes(b"\xff\xd8\xff\xe0jpeg")
    return path


def _points(tree):
    return list(vdb_index.iter_tree_point_records(tree))


def _by_kind(tree, kind):
    return [p for p in _points(tree) if p.kind == kind]


class ANodeIsOnePoint(unittest.TestCase):
    def test_a_figure_carries_its_prose_and_its_crop_together(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            figure = _child(
                tree, section, NodeType.IMAGE,
                content="A shaded 3D form.", img_path=_crop(directory),
            )
            figure.summary = "The Zernike Surface illustration."

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "image")

            # ``context`` is the heading above it: a figure states what it
            # shows without stating what it is about. See
            # test_a_figure_is_reachable_through_its_section.
            self.assertEqual({"text", "context", "image"}, set(point.vectors))
            self.assertIn("Zernike Surface illustration", point.vectors["text"])
            self.assertEqual("2. Zernike Surface", point.vectors["context"])
            self.assertTrue(point.vectors["image"].endswith(".jpg"))

    def test_a_title_keeps_its_heading_and_its_section_summary_apart(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            section.summary = "Defines the Zernike surface by a sag equation."

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "title")

            self.assertEqual("2. Zernike Surface", point.vectors["text"])
            self.assertIn("sag equation", point.vectors["context"])

    def test_an_equation_keeps_its_own_text_and_its_neighbourhood_apart(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="Before the equation.")
            equation = _child(tree, section, NodeType.EQUATION, content="z = cr^2")
            equation.summary = "The sag equation."
            _child(tree, section, NodeType.TEXT, content="After the equation.")

            point = next(
                p for p in _points(tree) if p.metadata["node_type"] == "equation"
            )

            self.assertIn("z = cr^2", point.vectors["text"])
            self.assertIn("Before the equation.", point.vectors["context"])

    def test_a_paragraph_carries_one_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="Just prose.")

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "text")

            self.assertEqual({"text"}, set(point.vectors))


class SlotsGoByRoleNotByAvailability(unittest.TestCase):
    def test_an_equation_with_no_text_of_its_own_carries_only_context(self):
        # The subtle one. Promoting the neighbourhood into `text` would blend
        # the two readings the equation branch deliberately keeps apart.
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="Explaining prose.")
            _child(tree, section, NodeType.EQUATION, content="")

            point = next(
                p for p in _points(tree) if p.metadata["node_type"] == "equation"
            )

            self.assertEqual({"context"}, set(point.vectors))
            self.assertIn("Explaining prose.", point.vectors["context"])

    def test_a_title_with_no_summary_carries_only_its_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "title")

            self.assertEqual({"text"}, set(point.vectors))

    def test_a_figure_with_nothing_at_all_yields_no_point(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.IMAGE)

            self.assertEqual(
                [], [p for p in _points(tree) if p.metadata["node_type"] == "image"]
            )


class ASemanticElementKeepsItsOwnPoint(unittest.TestCase):
    def test_an_element_is_a_point_of_its_own_kind(self):
        # Not a node: a cell, symbol or flowchart box inside one, whose element
        # id and box are per-element payload a shared point could not carry.
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="prose")

            kinds = {p.kind for p in _points(tree)}

            self.assertIn("node", kinds)
            self.assertTrue(all(k in ("node", "semantic_element") for k in kinds))

    def test_no_point_carries_a_point_level_modality(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="prose")

            for point in _points(tree):
                self.assertNotIn("modality", point.metadata)
                self.assertNotIn("type", point.metadata)


class TheLexicalSourceIsEveryTextThePointHolds(unittest.TestCase):
    def test_it_joins_the_named_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            section.summary = "A section about sag equations."

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "title")

            self.assertIn("2. Zernike Surface", point.lexical_source)
            self.assertIn("sag equations", point.lexical_source)

    def test_a_crop_contributes_no_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.IMAGE,
                   content="A shaded form.", img_path=_crop(directory))

            point = next(p for p in _points(tree) if p.metadata["node_type"] == "image")

            self.assertNotIn(".jpg", point.lexical_source)
            self.assertIn("shaded form", point.lexical_source)


class TheFlattenedViewMatchesWhatTheIndexUsedToEmit(unittest.TestCase):
    def test_a_point_flattens_to_one_record_per_vector_in_slot_order(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="Before.")
            equation = _child(tree, section, NodeType.EQUATION, content="z = cr^2",
                              img_path=_crop(directory))
            equation.summary = "The sag equation."

            records = [
                r
                for r in vdb_index.iter_tree_index_records(tree)
                if r.metadata.get("node_type") == "equation"
            ]

            # self text, then neighbourhood, then crop -- the order the
            # per-branch dispatch used to yield them in.
            self.assertEqual(["text", "text", "image"], [r.modality for r in records])
            self.assertIn("z = cr^2", records[0].value)
            self.assertIn("Before.", records[1].value)
            self.assertTrue(records[2].value.endswith(".jpg"))

    def test_the_flattened_records_carry_a_modality_again(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, section = _tree(directory)
            _child(tree, section, NodeType.TEXT, content="prose")

            for record in vdb_index.iter_tree_index_records(tree):
                self.assertEqual(record.modality, record.metadata["modality"])


if __name__ == "__main__":
    unittest.main()
