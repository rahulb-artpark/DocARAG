"""Text recovered into a block must be readable whatever the block's type is.

Indexing a 93 MB scan failed after three hours with "normalized block 112 is
invalid: observation requires a representation or asset". The culprit was not an
empty block -- the pruner that exists for those handled them correctly. It was a
table whose body was never recognised: the recovery pass read the page's native
text layer and wrote it to the block's "text" key, but every table
representation is built from table_body, caption and footnote, so the recovered
text was invisible. The block then carried neither a representation nor an
asset, and one such block aborts the whole document.

Skipping the block instead is not available: binding requires a one-to-one map
from legacy blocks to observations, which is what attaches canonical evidence
back to the tree. The text has to become evidence rather than the block
disappearing.
"""

from __future__ import annotations

import unittest

from document_services.parser_adapters.mineru_compat import _representations
from document_services.domain.evidence import BlockType, RepresentationKind


def _reps(block, block_type):
    return _representations(block, block_type, None, "en")


class RecoveredTextBecomesEvidence(unittest.TestCase):
    def _recovered_table(self):
        # The real block, from the document that failed.
        return {
            "type": "table",
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
            "text": "It indicates that the file exists in both locations but has "
            "different contents (Files do not match) or is missing.",
            "_text_recovered_from": "pdf_native_text_layer",
        }

    def test_a_table_with_no_body_still_yields_a_representation(self):
        reps = _reps(self._recovered_table(), BlockType.TABLE)
        self.assertTrue(reps, "the block would abort the document with none")
        self.assertIn("Files do not match", reps[0].content)

    def test_it_is_plain_text_not_a_claimed_table_body(self):
        # Presenting page text as an HTML table body would be a lie about what
        # was recognised.
        rep = _reps(self._recovered_table(), BlockType.TABLE)[0]
        self.assertEqual(rep.kind, RepresentationKind.TEXT)
        self.assertEqual(rep.media_type, "text/plain")

    def test_an_equation_with_no_latex_is_covered_too(self):
        # Same shape: an asset-typed block whose own channels came back empty.
        block = {"type": "equation", "img_path": "", "text": "E = mc squared"}
        self.assertTrue(_reps(block, BlockType.EQUATION))

    def test_a_list_with_no_items_is_covered_too(self):
        block = {"type": "list", "text": "a recovered line"}
        self.assertTrue(_reps(block, BlockType.LIST_ITEM))


class TheFallbackDoesNotDisturbBlocksThatAlreadyWork(unittest.TestCase):
    """It fires only when nothing else produced anything."""

    def test_a_real_table_keeps_its_html_body(self):
        block = {
            "type": "table",
            "table_body": "<table><tr><td>real</td></tr></table>",
            "text": "stray page text",
        }
        reps = _reps(block, BlockType.TABLE)
        kinds = {rep.kind for rep in reps}
        self.assertIn(RepresentationKind.HTML, kinds)
        self.assertNotIn("stray page text", [rep.content for rep in reps])

    def test_a_table_with_only_a_caption_is_unchanged(self):
        block = {"type": "table", "table_caption": ["Table 1"], "text": "stray"}
        contents = [rep.content for rep in _reps(block, BlockType.TABLE)]
        self.assertIn("Table 1", contents)
        self.assertNotIn("stray", contents)

    def test_a_paragraph_is_represented_once_not_twice(self):
        block = {"type": "text", "text": "ordinary prose"}
        self.assertEqual(len(_reps(block, BlockType.PARAGRAPH)), 1)

    def test_a_block_with_nothing_at_all_still_yields_nothing(self):
        # The contract must still reject a block that genuinely says nothing;
        # this widens what counts as evidence, it does not invent any.
        self.assertEqual(_reps({"type": "table", "text": "   "}, BlockType.TABLE), ())


if __name__ == "__main__":
    unittest.main()
