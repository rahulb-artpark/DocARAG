"""A node has to be reachable by its subject, not only by its own surface.

Three branches of the tree index text that is not about anything a question
would say, and in each the words naming the subject sat in a field nothing
read.

A visual node's searchable text was ``caption + footnote``, and MinerU fills
neither: the vision model's account of the pixels lands in ``content``, the
section's own words in ``summary``. On a 1871-page optics manual that indexed
1866 of 2002 figures as the empty string. The failure that surfaced it: "show
me the image of Zernike Surface" returned three screenshots of Zemax
coefficient windows -- the embedder can read "Zernike" rendered inside a
screenshot -- while the Zernike Surface section's own figure, an abstract
shaded form with no text in it, was never a candidate.

A listing indexed as syntax, for the same reason a grid indexes as markup:
"<zerntype> <numterms> <datatype>" is the shape of a file format, not a
sentence about one. All 313 code nodes had a summary that said what the format
was; none of it was indexed.

A heading indexed as its own few words, so "2.3.2.2.94. Zernike Surface" was
reachable and "defines the Zernike surface ... characterized by a sag
equation" -- the section summary sitting on the same node -- was not.

Prose types are deliberately left alone: a paragraph already is the sentence a
question resembles.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval_engine.indexes.tree import DocumentTree, NodeType, TreeNode
from retrieval_engine.indexing.vector import builder as vdb_index


def _tree(directory, **figure_meta):
    tree = DocumentTree(
        meta_dict={
            "file_name": "optics.pdf",
            "source_sha256": "c" * 64,
            "document_version_id": "document-version-9",
        },
        cfg=SimpleNamespace(save_path=directory),
    )
    section = TreeNode({"pdf_id": 1, "page_idx": 0, "content": "2.3.2.2.94. Zernike Surface"})
    section.type = NodeType.TITLE
    tree.add_node(section)
    tree.root_node.add_child(section)
    figure = TreeNode({"pdf_id": 2, "page_idx": 0, **figure_meta})
    figure.type = NodeType.IMAGE
    tree.add_node(figure)
    section.add_child(figure)
    return tree, figure


def _texts(tree):
    return [
        record.value
        for record in vdb_index.iter_tree_index_records(tree)
        if record.modality == "text"
    ]


class AFigureIsFoundByItsSubject(unittest.TestCase):
    def test_the_summary_is_indexed_when_there_is_no_caption(self):
        # The Zernike Surface figure exactly: no caption, a pixel description
        # that never says "Zernike", and a summary that does.
        with tempfile.TemporaryDirectory() as directory:
            tree, figure = _tree(
                directory,
                content="Abstract 3D shape resembling a teardrop, in grayscale.",
            )
            figure.summary = (
                "This visual description is inconsistent with the title "
                '"Zernike Surface", which refers to a mathematical function.'
            )

            indexed = _texts(tree)
            self.assertTrue(
                any("Zernike Surface" in text for text in indexed),
                f"the figure's subject never reached the index: {indexed}",
            )

    def test_the_pixel_description_is_kept_beside_the_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, figure = _tree(directory, content="A teardrop form in grayscale.")
            figure.summary = "The Zernike Surface section's illustration."

            # By "teardrop", not by "Zernike": the section title is a node of
            # its own and is indexed too.
            text = next(t for t in _texts(tree) if "teardrop" in t)
            self.assertIn("teardrop form in grayscale", text)
            # The summary leads: it is the sentence a question resembles, and
            # the part worth keeping if the embedder's window truncates.
            self.assertTrue(text.startswith("The Zernike Surface section's"))

    def test_a_caption_is_still_indexed_when_the_producer_writes_one(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(
                directory,
                caption="Figure 4: the Zernike sag surface.",
                content="A shaded 3D form.",
            )

            text = next(t for t in _texts(tree) if "Figure 4" in t)
            self.assertIn("A shaded 3D form", text)

    def test_a_caption_already_inside_the_content_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(
                directory,
                caption="Figure 4: the Zernike sag surface.",
                content="Figure 4: the Zernike sag surface. A shaded 3D form.",
            )

            text = next(t for t in _texts(tree) if "Figure 4" in t)
            self.assertEqual(1, text.count("Figure 4"))

    def test_the_crop_keeps_a_visual_vector_not_a_prose_one(self):
        # The paired text is fused *into* the image vector, not stored beside
        # it. Feeding every figure its whole summary put 4 image vectors in a
        # top 200 -- best at rank 36 -- and a question that said "show me the
        # image" retrieved none. The prose belongs in the text record; the
        # crop stays a picture.
        with tempfile.TemporaryDirectory() as directory:
            image = f"{directory}/figure.jpg"
            Path(image).write_bytes(b"\xff\xd8\xff\xe0jpeg")
            tree, figure = _tree(directory, content="A teardrop form.", img_path=image)
            figure.summary = "The Zernike Surface illustration."

            records = list(vdb_index.iter_tree_index_records(tree))
            paired = [r.image_text for r in records if r.modality == "image"]
            texts = [r.value for r in records if r.modality == "text"]

            self.assertEqual([None], paired, "the crop must carry no text at all")
            self.assertTrue(
                any("Zernike Surface illustration" in t for t in texts),
                "the prose must still be indexed, as its own text record",
            )

    def test_even_a_caption_is_not_fused_into_the_crop(self):
        # A caption was the cautious middle ground while a figure had one
        # vector. It has two now, and the caption is already in the text one --
        # so the crop is left a picture, which only it can be matched on.
        with tempfile.TemporaryDirectory() as directory:
            image = f"{directory}/figure.jpg"
            Path(image).write_bytes(b"\xff\xd8\xff\xe0jpeg")
            tree, figure = _tree(
                directory,
                caption="Figure 4: the Zernike sag surface.",
                content="A long visual description that must not be fused in.",
                img_path=image,
            )
            figure.summary = "A long summary that must not be fused in either."

            records = list(vdb_index.iter_tree_index_records(tree))
            paired = [r.image_text for r in records if r.modality == "image"]
            texts = [r.value for r in records if r.modality == "text"]

            self.assertEqual([None], paired)
            self.assertTrue(
                any("Figure 4: the Zernike sag surface." in t for t in texts),
                "the caption is not lost; it is searched as the prose it is",
            )


class NothingIsIndexedAsTheEmptyString(unittest.TestCase):
    """Every captionless figure used to embed to whatever "" produces."""

    def test_a_figure_with_no_words_at_all_yields_no_text_record(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)

            self.assertEqual([], [t for t in _texts(tree) if not t.strip()])

    def test_a_figure_with_words_still_yields_exactly_one_text_record(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, figure = _tree(directory, content="A teardrop form.")
            figure.summary = "The Zernike Surface illustration."

            figure_texts = [t for t in _texts(tree) if "teardrop" in t]
            self.assertEqual(1, len(figure_texts), figure_texts)


class AListingIsFoundByWhatItDoes(unittest.TestCase):
    """Code and algorithms embed as syntax; the summary is their only prose."""

    def _listing(self, directory, node_type):
        tree, _ = _tree(directory)
        node = TreeNode(
            {
                "pdf_id": 3,
                "page_idx": 0,
                "content": "! comment (optional)\n<zerntype> <numterms> <datatype>",
            }
        )
        node.type = node_type
        node.summary = (
            "The INT Zernike file format consists of optional comment lines "
            "followed by the Zernike coefficient data."
        )
        tree.add_node(node)
        tree.root_node.children[0].add_child(node)
        return tree

    def test_a_code_listing_is_indexed_with_what_it_is_for(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = self._listing(directory, NodeType.CODE)

            text = next(t for t in _texts(tree) if "zerntype" in t)
            self.assertIn("INT Zernike file format", text)
            self.assertTrue(text.startswith("The INT Zernike file format"))

    def test_an_algorithm_is_indexed_the_same_way(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = self._listing(directory, NodeType.ALGORITHM)

            self.assertTrue(
                any("INT Zernike file format" in t for t in _texts(tree)),
                _texts(tree),
            )

    def test_prose_types_are_not_given_their_summary_as_well(self):
        # A paragraph already is the sentence a question resembles; restating
        # it would index the same claim twice.
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)
            para = TreeNode({"pdf_id": 4, "page_idx": 0, "content": "The sag is given by."})
            para.type = NodeType.TEXT
            para.summary = "A restatement that must not be indexed separately."
            tree.add_node(para)
            tree.root_node.children[0].add_child(para)

            self.assertNotIn("A restatement that must not be indexed separately.", _texts(tree))


class ASectionIsFoundByWhatItCovers(unittest.TestCase):
    def test_a_title_carries_its_section_summary_as_a_second_record(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)
            title = tree.root_node.children[0]
            title.summary = (
                "This section defines the Zernike surface, characterized by a "
                "sag equation combining base conic and aspheric terms."
            )

            indexed = _texts(tree)
            self.assertIn("2.3.2.2.94. Zernike Surface", indexed)
            self.assertTrue(
                any("sag equation" in t for t in indexed),
                f"the section's own summary never reached the index: {indexed}",
            )

    def test_the_heading_stays_a_record_of_its_own(self):
        # Blending would lengthen the vector someone typing the exact heading
        # is matching against, which is what the second record avoids.
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)
            tree.root_node.children[0].summary = "A long description of the section."

            self.assertIn("2.3.2.2.94. Zernike Surface", _texts(tree))

    def test_a_summary_equal_to_the_heading_is_not_indexed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, _ = _tree(directory)
            title = tree.root_node.children[0]
            title.summary = "2.3.2.2.94. Zernike Surface"

            headings = [t for t in _texts(tree) if t == "2.3.2.2.94. Zernike Surface"]
            self.assertEqual(1, len(headings), headings)


if __name__ == "__main__":
    unittest.main()
