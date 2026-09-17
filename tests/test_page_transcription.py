"""Detection and alignment for re-transcribing scanned pages.

Fixtures are the real readings of 10P_CH11 page 2: MinerU's text on one side,
qwen3.6:35b's transcription of the same pixels on the other.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from document_services.enrichment.page_transcription import (
    align_transcription,
    lexicon_validity,
    merge_band_lines,
    page_text_quality,
    transcription_lines,
    _LEXICON,
    _alignment_units,
    _unit_max_spans,
    _write_transcript,
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


class AListIsMatchedItemByItem(unittest.TestCase):
    """MinerU keeps a whole list in one block; the page reads it as many lines."""

    def test_each_item_is_its_own_unit(self):
        block = {"list_items": ["first item", "second item", "  "]}
        self.assertEqual(_alignment_units(block), ("first item", "second item"))

    def test_an_ordinary_block_stays_one_unit(self):
        self.assertEqual(_alignment_units({"text": "a paragraph"}), ("a paragraph",))

    def test_an_empty_block_has_no_units(self):
        self.assertEqual(_alignment_units({"text": "   "}), ())

    def test_a_long_list_matches_where_one_joined_blob_cannot(self):
        # The real 10P_CH11 page 1 list, which spans far more transcribed lines
        # than any single span the aligner will try.
        items = [
            "lornea A Thin membrane covering the swrfau of eyball.",
            "Dus Adark nuscular diaphragm located just luhind the Corner",
            "o. Pupil A black opening lutween the aqueous humour",
            "Cilliary lontrols the foul length of the eye lens",
            "Crystalline lens A tvansparent double convex lens",
            "Retina A light sensitive screen at the buck of eye",
        ]
        lines = [
            "1. Cornea A Thin membrane covering the surface of eyeball.",
            "2. Iris A dark muscular diaphragm located just behind the Cornea.",
            "3. Pupil A black opening between the aqueous humour",
            "4. Ciliary Controls the focal length of the eye lens",
            "5. Crystalline lens A transparent double convex lens",
            "6. Retina A light sensitive screen at the back of eye",
        ]
        # Six items need six lines, past the span cap the aligner will try, so
        # the joined block either misses or wins on a partial overlap and
        # replaces the whole list with the few lines it matched.
        joined = align_transcription([" ".join(items)], lines, as_lines=True)
        self.assertNotEqual(
            list(joined[0] or ()), lines, "the joined list cannot claim every line"
        )
        per_item = align_transcription(list(items), lines, as_lines=True)
        self.assertEqual([m[0] for m in per_item], lines)

    def test_an_unmatched_item_does_not_cost_the_list_its_siblings(self):
        items = ["lornea A Thin membrane", "utterly unrelated marginalia", "o. Pupil"]
        lines = ["1. Cornea A Thin membrane", "3. Pupil"]
        matches = align_transcription(list(items), lines, as_lines=True)
        rebuilt = []
        for parser_text, claimed in zip(items, matches):
            rebuilt.extend(claimed or (parser_text,))
        self.assertEqual(
            rebuilt,
            ["1. Cornea A Thin membrane", "utterly unrelated marginalia", "3. Pupil"],
        )


class BandsAreMergedAtTheirSeam(unittest.TestCase):
    def test_a_reflowed_overlap_is_dropped_once(self):
        # The bands disagree on line breaks, which is why the seam is matched on
        # text and not line-for-line.
        upper = [
            "3. Pupil A black opening between the aqueous humour and the eye lens.",
            "Regulates and controls the amount of light entering",
        ]
        lower = [
            "3. Pupil",
            "A black opening between the aqueous humour and the eye lens.",
            "Regulates and controls the amount of light entering the eye.",
            "4. Ciliary Muscles",
        ]
        merged = merge_band_lines([upper, lower])
        self.assertEqual(merged, lower, "the lower band's fuller reading wins")

    def test_unrelated_bands_are_concatenated_untouched(self):
        upper = ["Presbyopia", "Near point of the eye reduces with aging."]
        lower = ["Tyndall Effect", "Colour of the sky is blue."]
        self.assertEqual(merge_band_lines([upper, lower]), upper + lower)

    def test_a_legitimately_repeated_line_survives(self):
        upper = ["Some prose about the eye lens.", "Scanned with CamScanner"]
        lower = ["A different paragraph entirely.", "Scanned with CamScanner"]
        self.assertEqual(merge_band_lines([upper, lower]), upper + lower)

    def test_one_band_is_returned_as_is(self):
        self.assertEqual(merge_band_lines([["only", "band"]]), ["only", "band"])

    def test_no_bands_is_empty(self):
        self.assertEqual(merge_band_lines([]), [])


class ABlockCannotAbsorbMoreLinesThanItHas(unittest.TestCase):
    def test_the_cap_comes_from_the_parser_line_count(self):
        one_line = {"middle_json": {"lines": [{"bbox": [54, 170, 249, 201]}]}}
        six_lines = {"middle_json": {"lines": [{} for _ in range(6)]}}
        self.assertEqual(_unit_max_spans(one_line, 1), [2])
        self.assertEqual(_unit_max_spans(six_lines, 1), [7])

    def test_a_list_item_gets_a_list_item_cap(self):
        block = {"list_items": ["a", "b"]}
        self.assertEqual(_unit_max_spans(block, 2), [3, 3])

    def test_no_line_geometry_falls_back_to_the_uniform_default(self):
        self.assertEqual(_unit_max_spans({"text": "x"}, 1), [4])

    def test_a_one_line_block_cannot_claim_a_paragraph(self):
        lines = [
            "Basic Events due to Scattering",
            "When a light ray passes through the air, the particles of dust",
            "and smoke present in air make the path of the light ray visible.",
            "Such a phenomenon is known as the Tyndall Effect.",
        ]
        claimed = align_transcription(
            ["Basic Events du to Scattering X"], lines, max_spans=[2], as_lines=True
        )[0]
        self.assertLessEqual(len(claimed), 2)
        self.assertIn("Basic Events", claimed[0])


class ThePageReadingIsPersisted(unittest.TestCase):
    """Without this file, a misread line and a misplaced line look the same."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _read(self, page=2):
        return json.loads((self.dir / f"page_{page:04d}.json").read_text())

    def test_the_model_reply_is_kept_verbatim_per_band(self):
        _write_transcript(
            self.dir, 2, {"page_index": 2},
            band_replies=["- upper band\n- second line", "- lower band"],
            merged_lines=["upper band", "second line", "lower band"],
        )
        bands = self._read()["bands"]
        self.assertEqual([b["band"] for b in bands], [0, 1])
        self.assertEqual(bands[0]["reply"], "- upper band\n- second line")
        self.assertEqual(bands[0]["lines"], ["upper band", "second line"])

    def test_each_unit_records_what_it_was_allowed_and_what_it_took(self):
        _write_transcript(
            self.dir, 2, {"page_index": 2},
            units=[
                {"parser_text": "Basic Events du to Scattering X",
                 "max_span": 2, "claimed": ["Basic Events due to Scattering"]},
                {"parser_text": "www.notesd", "max_span": 2, "claimed": []},
            ],
        )
        units = self._read()["units"]
        self.assertEqual(units[0]["max_span"], 2)
        self.assertEqual(units[0]["claimed"], ["Basic Events due to Scattering"])
        self.assertEqual(units[1]["claimed"], [], "an unmatched unit is recorded too")

    def test_a_failed_page_is_written_as_well(self):
        _write_transcript(self.dir, 5, {"page_index": 5, "failed": "TimeoutError"})
        written = self._read(5)
        self.assertEqual(written["failed"], "TimeoutError")
        self.assertEqual(written["bands"], [])

    def test_no_directory_means_no_file_and_no_error(self):
        _write_transcript(None, 2, {"page_index": 2})
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_an_unwritable_directory_never_fails_the_run(self):
        blocked = self.dir / "file-not-a-dir"
        blocked.write_text("occupied")
        _write_transcript(blocked, 2, {"page_index": 2})  # must not raise

    def test_the_file_is_replaced_not_appended_on_a_rerun(self):
        _write_transcript(self.dir, 2, {"page_index": 2, "blocks_replaced": 1})
        _write_transcript(self.dir, 2, {"page_index": 2, "blocks_replaced": 9})
        self.assertEqual(self._read()["blocks_replaced"], 9)
        self.assertEqual(
            [p.name for p in self.dir.iterdir()], ["page_0002.json"],
            "no temporary file may be left behind",
        )
