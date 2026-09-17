"""A crop reaches the vision model with the sentence naming its subject.

The answer-time VLM was handed a figure's crop beside only the vision model's
own account of the pixels. For a figure that shows something without naming it
that account is all surface, so shown test.pdf's Zernike surface the VLM
answered "The provided image is a 3D rendered icon or graphic, not a chart,
table, or diagram ... typical of digital icons or interface elements".

The final answer repeats its own visual sub-answer, so it said "The provided
documents do not contain an image of the Zernike Surface" while rendering that
very image one line below. A contradiction is worse than a miss: the reader can
see the figure and is told it does not exist.

The summary is the sentence that names the subject and is already on the
retrieved node -- the same text the tree view lets you edit. It leads; the pixel
description follows, so "what does this figure look like" is still answerable
from the same prompt.

This is the same defect as the text reranker's, one stage later: a figure judged
on pixels alone with the one thing that names it withheld.
"""

from __future__ import annotations

import unittest

from retrieval_engine.rag.gbc.answer import _visual_evidence_content

SUMMARY = "Zernike Surface. A 3D rendering of the Zernike surface."
CONTENT = (
    "This is a 3D rendered icon or graphic, not a chart, table, or diagram, "
    "typical of digital icons or interface elements."
)


class TheSummaryLeads(unittest.TestCase):
    def test_the_naming_sentence_comes_first(self):
        text = _visual_evidence_content({"summary": SUMMARY, "content": CONTENT})

        self.assertTrue(text.startswith(SUMMARY), text[:60])

    def test_the_pixel_description_is_still_there(self):
        # A question about what the figure looks like has to remain answerable.
        text = _visual_evidence_content({"summary": SUMMARY, "content": CONTENT})

        self.assertIn("interface elements", text)

    def test_a_figure_with_no_summary_still_sends_its_description(self):
        text = _visual_evidence_content({"content": CONTENT})

        self.assertEqual(CONTENT, text)

    def test_a_figure_with_no_description_still_sends_its_summary(self):
        text = _visual_evidence_content({"summary": SUMMARY, "content": ""})

        self.assertEqual(SUMMARY, text)

    def test_a_summary_equal_to_the_content_is_not_repeated(self):
        text = _visual_evidence_content({"summary": CONTENT, "content": CONTENT})

        self.assertEqual(CONTENT, text)

    def test_a_blank_summary_is_ignored(self):
        text = _visual_evidence_content({"summary": "   ", "content": CONTENT})

        self.assertEqual(CONTENT, text)

    def test_a_node_with_neither_yields_nothing(self):
        self.assertEqual("", _visual_evidence_content({"summary": None, "content": ""}))

    def test_a_non_string_summary_does_not_break_the_prompt(self):
        text = _visual_evidence_content({"summary": 42, "content": CONTENT})

        self.assertEqual(CONTENT, text)


class BothVisualPromptsUseIt(unittest.TestCase):
    """One figure per prompt and one figure with its text both go through it."""

    @staticmethod
    def _source():
        from pathlib import Path

        from retrieval_engine.rag.gbc import answer as gbc_answer

        return Path(gbc_answer.__file__).read_text(encoding="utf-8")

    def test_the_per_figure_prompt_uses_it(self):
        self.assertIn("_evidence_text(node, _visual_evidence_content(node))", self._source())

    def test_the_single_figure_prompt_uses_it(self):
        self.assertIn(
            "_visual_evidence_content(selected_image)", self._source()
        )

    def test_no_visual_prompt_reads_content_alone(self):
        # The bug was `_evidence_text(selected_image, node_content)`: a crop
        # described by its pixels and nothing else.
        source = self._source()

        self.assertNotIn("_evidence_text(selected_image, node_content)", source)


if __name__ == "__main__":
    unittest.main()
