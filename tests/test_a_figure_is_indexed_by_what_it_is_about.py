"""A figure has to be reachable by its subject, not only by its pixels.

A visual node's searchable text was ``caption + footnote``. MinerU fills
neither: the vision model's account of the pixels lands in ``content`` and the
section's own words land in ``summary``. On a 1871-page optics manual that
indexed 1866 of 2002 figures as the empty string, so every one of them was
reachable only through the image vector.

The failure that surfaced it: "show me the image of Zernike Surface" returned
three screenshots of Zemax coefficient windows -- the embedder can read
"Zernike" rendered inside a screenshot -- while the Zernike Surface section's
own figure, an abstract shaded form with no text in it, was never a candidate.
Its summary names the subject in its first line, and nothing was reading it.
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines import vdb_index


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

    def test_the_fused_image_text_carries_the_same_words(self):
        # Tables and equations already pass their full text as the image's
        # paired text; a figure passed the empty string.
        with tempfile.TemporaryDirectory() as directory:
            image = f"{directory}/figure.jpg"
            open(image, "wb").write(b"\xff\xd8\xff\xe0jpeg")
            tree, figure = _tree(directory, content="A teardrop form.", img_path=image)
            figure.summary = "The Zernike Surface illustration."

            paired = [
                record.image_text
                for record in vdb_index.iter_tree_index_records(tree)
                if record.modality == "image"
            ]
            self.assertTrue(paired and "Zernike Surface" in paired[0], paired)


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


if __name__ == "__main__":
    unittest.main()
