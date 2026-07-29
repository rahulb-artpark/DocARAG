"""Every MinerU 3.x block type maps somewhere deliberate.

MinerU 3.4.4 declares 41 block types; the adapter previously recognised 16.
The rest fell through to ``BlockType.UNKNOWN`` silently -- including
``interline_equation``, which is what MinerU 3.x actually emits for a display
equation (7 of them on a single 5-page test document). Equations would have
stopped being equations with nothing in the logs to say so.

This pins the whole surface so a future MinerU upgrade that adds a type fails
here instead of quietly degrading a document.
"""

import unittest

from docarag.adapters.mineru_compat import _block_type
from docarag.domain.evidence import BlockType

# Mirrors mineru.utils.enum_class.BlockType at 3.4.4. Kept as a literal list so
# this test does not depend on MinerU being importable.
MINERU_35_BLOCK_TYPES = {
    "image": BlockType.FIGURE,
    "table": BlockType.TABLE,
    "chart": BlockType.CHART,
    "image_body": BlockType.FIGURE,
    "table_body": BlockType.TABLE,
    "chart_body": BlockType.CHART,
    "caption": BlockType.CAPTION,
    "image_caption": BlockType.CAPTION,
    "table_caption": BlockType.CAPTION,
    "chart_caption": BlockType.CAPTION,
    "algorithm_caption": BlockType.CAPTION,
    "footnote": BlockType.FOOTNOTE,
    "image_footnote": BlockType.FOOTNOTE,
    "table_footnote": BlockType.FOOTNOTE,
    "chart_footnote": BlockType.FOOTNOTE,
    "code_footnote": BlockType.FOOTNOTE,
    "text": BlockType.PARAGRAPH,
    "title": BlockType.TITLE,
    "interline_equation": BlockType.EQUATION,
    "equation": BlockType.EQUATION,
    "formula_number": BlockType.EQUATION,
    "list": BlockType.LIST_ITEM,
    "index": BlockType.LIST_ITEM,
    "discarded": BlockType.DISCARDED,
    "code": BlockType.CODE,
    "code_body": BlockType.CODE,
    "code_caption": BlockType.CAPTION,
    "algorithm": BlockType.ALGORITHM,
    "ref_text": BlockType.REFERENCE,
    "phonetic": BlockType.ASIDE,
    "header": BlockType.HEADER,
    "footer": BlockType.FOOTER,
    "page_number": BlockType.FOOTER,
    "aside_text": BlockType.ASIDE,
    "page_footnote": BlockType.FOOTNOTE,
    "abstract": BlockType.ABSTRACT,
    "doc_title": BlockType.DOC_TITLE,
    "paragraph_title": BlockType.TITLE,
    "vertical_text": BlockType.ASIDE,
    "header_image": BlockType.FIGURE,
    "footer_image": BlockType.FIGURE,
}


class MinerU35BlockTypeMapping(unittest.TestCase):
    def test_every_mineru_type_maps_as_declared(self):
        for raw, expected in sorted(MINERU_35_BLOCK_TYPES.items()):
            with self.subTest(mineru_type=raw):
                got, _ = _block_type({"type": raw}, {})
                self.assertEqual(
                    got,
                    expected,
                    f"MinerU type {raw!r} mapped to {got}, expected {expected}",
                )

    def test_no_mineru_type_falls_through_to_unknown(self):
        """UNKNOWN must mean 'the parser said unknown', not 'we forgot this one'."""

        leaked = [
            raw
            for raw in MINERU_35_BLOCK_TYPES
            if _block_type({"type": raw}, {})[0] is BlockType.UNKNOWN
        ]
        self.assertEqual(leaked, [], f"unmapped MinerU types: {leaked}")

    def test_display_equation_is_an_equation(self):
        """The specific regression that motivated this test."""

        got, _ = _block_type({"type": "interline_equation"}, {})
        self.assertEqual(got, BlockType.EQUATION)

    def test_document_title_is_distinct_from_section_heading(self):
        doc, _ = _block_type({"type": "doc_title"}, {})
        section, _ = _block_type({"type": "paragraph_title"}, {})
        self.assertEqual(doc, BlockType.DOC_TITLE)
        self.assertEqual(section, BlockType.TITLE)
        self.assertNotEqual(doc, section)

    def test_genuinely_unknown_type_still_reports_unknown(self):
        got, subtype = _block_type({"type": "some_future_type"}, {})
        self.assertEqual(got, BlockType.UNKNOWN)
        self.assertEqual(subtype, "some_future_type")


class AdapterDeclaresWhatItEmits(unittest.TestCase):
    """The manifest and the translation must not drift apart.

    The parser port validates every observation against the adapter's declared
    block_types and aborts the document on an undeclared one. So a type added
    to _block_type but not to the manifest does not degrade -- it fails the
    whole bind, which is how 'aside' surfaced only during a real parse.
    """

    def test_every_mapped_type_is_declared(self):
        from docarag.adapters.mineru_compat import DECLARED_BLOCK_TYPES

        emitted = {_block_type({"type": raw}, {})[0] for raw in MINERU_35_BLOCK_TYPES}
        undeclared = sorted(bt.value for bt in emitted - DECLARED_BLOCK_TYPES)
        self.assertEqual([], undeclared, f"mapped but not declared: {undeclared}")

    def test_every_text_type_can_produce_a_representation(self):
        """An observation with no representation and no asset is rejected outright."""

        from docarag.adapters.mineru_compat import _TEXT_REPRESENTED_BLOCKS
        from docarag.domain.evidence import BlockType as BT

        visual = {BT.FIGURE, BT.CHART, BT.DIAGRAM, BT.TABLE, BT.EQUATION}
        emitted = {_block_type({"type": raw}, {})[0] for raw in MINERU_35_BLOCK_TYPES}
        missing = sorted(
            bt.value for bt in emitted - visual - _TEXT_REPRESENTED_BLOCKS
        )
        self.assertEqual([], missing, f"no representation path: {missing}")


if __name__ == "__main__":
    unittest.main()
