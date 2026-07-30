"""Detection and alignment for re-transcribing scanned pages.

Fixtures are the real readings of 10P_CH11 page 2: MinerU's text on one side,
qwen3.6:35b's transcription of the same pixels on the other.
"""

import unittest

from docarag.page_transcription import (
    align_transcription,
    lexicon_validity,
    page_text_quality,
    transcription_lines,
    _LEXICON,
)

_MINERU = [
    "Presbyopia -x",
    "• Neaoint of th ey gradally sudues way lwith aging. D",
    "• Power of accomodation of eye devreases. Caures",
    "www.notesd",
    "• An eye sufrs from both myopia and Hypermetropia.",
    "↓ Correction",
    "Bifocal lens.",
]
_TRANSCRIPT = """Presbyopia
- Near point of the eye gradually reduces away with aging.
- Power of accommodation of eye decreases.
↓ Causes
- Gradual weakening of the ciliary muscles.
- Diminishing flexibility of the eye lens.
- An eye suffers from both myopia and Hypermetropia.
↓ Correction
Bifocal lens.
"""


class LinesAreNormalised(unittest.TestCase):
    def test_bullets_and_blanks_are_dropped(self):
        lines = transcription_lines("- one\n\n• two\n   \n* three\n")
        self.assertEqual(lines, ["one", "two", "three"])


class QualityDecidesWithoutAModel(unittest.TestCase):
    def test_a_page_with_a_text_layer_is_never_transcribed(self):
        verdict = page_text_quality(
            0, parsed_text="qqqq wwww eeee", has_text_layer=True,
            min_validity=0.9, min_words=5,
        )
        self.assertFalse(verdict.needs_transcription)
        self.assertIn("text layer", verdict.reason)

    def test_too_few_words_is_not_judged(self):
        verdict = page_text_quality(
            0, parsed_text="qqqq", has_text_layer=False,
            min_validity=0.9, min_words=20,
        )
        self.assertFalse(verdict.needs_transcription)
        self.assertIn("too few", verdict.reason)


@unittest.skipUnless(_LEXICON, "no system lexicon available")
class QualitySeparatesGoodScansFromBad(unittest.TestCase):
    def test_mineru_reading_of_a_scan_is_flagged(self):
        verdict = page_text_quality(
            2, parsed_text=" ".join(_MINERU), has_text_layer=False,
            min_validity=0.9, min_words=5,
        )
        self.assertTrue(verdict.needs_transcription, verdict.reason)

    def test_the_same_page_read_correctly_is_not_flagged(self):
        verdict = page_text_quality(
            2, parsed_text=_TRANSCRIPT, has_text_layer=False,
            min_validity=0.9, min_words=5,
        )
        self.assertFalse(verdict.needs_transcription, verdict.reason)

    def test_a_clean_typed_scan_is_not_flagged(self):
        typed = (
            "The refractive index of a medium determines how much light bends "
            "when it passes from one transparent substance into another."
        )
        verdict = page_text_quality(
            1, parsed_text=typed, has_text_layer=False,
            min_validity=0.9, min_words=5,
        )
        self.assertFalse(
            verdict.needs_transcription,
            "a good scan of printed text must not pay for the VLM",
        )

    def test_validity_rises_once_the_page_is_re_read(self):
        _, before = lexicon_validity(" ".join(_MINERU))
        _, after = lexicon_validity(_TRANSCRIPT)
        self.assertGreater(after, before + 0.2)


class AlignmentIsMonotonicAndSafe(unittest.TestCase):
    def setUp(self):
        self.lines = transcription_lines(_TRANSCRIPT)

    def test_a_misread_block_is_matched_to_its_correct_reading(self):
        matched = align_transcription(_MINERU, self.lines)
        self.assertEqual(
            matched[1], "Near point of the eye gradually reduces away with aging."
        )

    def test_a_watermark_the_transcription_ignored_stays_unmatched(self):
        matched = align_transcription(_MINERU, self.lines)
        self.assertIsNone(
            matched[_MINERU.index("www.notesd")],
            "an unmatched block must keep the parser's text, not borrow a line",
        )

    def test_lines_the_parser_missed_do_not_break_the_chain(self):
        # "Gradual weakening" and "Diminishing flexibility" exist only in the
        # transcription. The block after them must still find its own line.
        matched = align_transcription(_MINERU, self.lines)
        self.assertEqual(
            matched[4], "An eye suffers from both myopia and Hypermetropia."
        )

    def test_no_line_is_claimed_twice(self):
        matched = [m for m in align_transcription(_MINERU, self.lines) if m]
        self.assertEqual(len(matched), len(set(matched)))

    def test_repeated_headings_are_resolved_by_order_not_by_text(self):
        blocks = ["↓ Causes", "first body", "↓ Causes", "second body"]
        lines = ["↓ Causes", "first body", "↓ Causes", "second body"]
        self.assertEqual(align_transcription(blocks, lines), lines)

    def test_an_unrelated_transcription_matches_nothing(self):
        matched = align_transcription(
            ["Presbyopia and the ageing eye lens"],
            ["Atmospheric refraction over a hot road"],
        )
        self.assertEqual(matched, [None])

    def test_an_empty_block_is_never_matched(self):
        self.assertEqual(align_transcription(["", "   "], self.lines), [None, None])
