"""label_merger closes a real gap found in a user's document
(312_LG_E_L3.pdf): "Answer to Check Yourself" and its answer key
"1(A) 2(A) 3(A) 4(D) 5(A)" were parsed as two separate, unlinked tree
nodes -- the label carried no content, and the answer was unreachable from
it. is_likely_incomplete_paragraph's own Rule 1 (short text is assumed to
be a self-contained caption/title, never a candidate) is exactly why this
never got merged; label_merger targets that same excluded population from
the other side, with the actual merge decision left to the LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from retrieval_engine.parsing.pdf_refinement import (
    StitchingJudgmentsResponse,
    is_likely_content_label,
    label_merger,
    search_label_content_candidate,
)
from retrieval_engine.prompts.refinement import StitchingJudgment


class IsLikelyContentLabelTests(unittest.TestCase):
    def test_the_actual_reported_case(self):
        self.assertTrue(is_likely_content_label("Answer to Check Yourself"))

    def test_short_labels_in_other_wording_still_match(self):
        # No keyword list -- must generalize beyond the specific words in
        # the one reported case.
        for text in ("Key Takeaways", "Try It Yourself", "Definition", "Chinyanja Verb"):
            with self.subTest(text=text):
                self.assertTrue(is_likely_content_label(text))

    def test_a_complete_sentence_is_not_a_label_even_if_short(self):
        self.assertFalse(is_likely_content_label("The cat sat."))

    def test_long_text_is_not_a_label(self):
        long_text = "This is a long paragraph that goes on for quite a while " * 3
        self.assertFalse(is_likely_content_label(long_text))

    def test_a_bare_page_number_is_not_a_label(self):
        self.assertFalse(is_likely_content_label("42"))

    def test_a_bullet_or_punctuation_only_fragment_is_not_a_label(self):
        self.assertFalse(is_likely_content_label("---"))

    def test_empty_and_whitespace_text_is_not_a_label(self):
        self.assertFalse(is_likely_content_label(""))
        self.assertFalse(is_likely_content_label("   "))

    def test_never_overlaps_with_is_likely_incomplete_paragraphs_own_population(self):
        # The two passes must never both claim the same block: label_merger
        # targets exactly what Rule 1 excludes from continuation-merging.
        from retrieval_engine.parsing.pdf_refinement import is_likely_incomplete_paragraph

        cases = [
            "Answer to Check Yourself",
            "The cat sat.",
            "42",
            "A body continues to be in a state of rest or of uniform motion unless acted upon",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    is_likely_content_label(text)
                    and is_likely_incomplete_paragraph(text)
                )


def _block(pdf_id, page_idx, text, *, type_="text", text_level=-1):
    return {
        "pdf_id": pdf_id,
        "page_idx": page_idx,
        "type": type_,
        "text": text,
        "text_level": text_level,
        "middle_json": {"index": pdf_id},
    }


class SearchLabelContentCandidateTests(unittest.TestCase):
    def test_finds_the_very_next_block_regardless_of_case_or_leading_digit(self):
        pdf_list = [
            _block(1, 2, "Answer to Check Yourself"),
            _block(2, 2, "1(A) 2(A) 3(A) 4(D) 5(A)"),
        ]
        candidate = search_label_content_candidate(pdf_list[0], pdf_list, 0, 1)
        self.assertIs(pdf_list[1], candidate)

    def test_skips_tables_images_and_titles_to_find_the_next_real_text(self):
        pdf_list = [
            _block(1, 0, "Solution"),
            _block(2, 0, "", type_="table"),
            _block(3, 0, "Some Title", text_level=1),
            _block(4, 0, "The worked solution text."),
        ]
        candidate = search_label_content_candidate(pdf_list[0], pdf_list, 0, 3)
        self.assertIs(pdf_list[3], candidate)

    def test_respects_the_max_page_gap_bound(self):
        pdf_list = [
            _block(1, 0, "Note"),
            _block(2, 5, "Far-away content, outside the search window."),
        ]
        candidate = search_label_content_candidate(
            pdf_list[0], pdf_list, 0, 1, max_page_gap=2
        )
        self.assertIsNone(candidate)

    def test_returns_none_when_nothing_follows(self):
        pdf_list = [_block(1, 0, "Note")]
        self.assertIsNone(search_label_content_candidate(pdf_list[0], pdf_list, 0, 0))


class _StubLLM:
    """Records the prompt it was given and returns a pre-scripted response."""

    def __init__(self, judgments: list[StitchingJudgment]):
        self.config = SimpleNamespace(max_tokens=8000)
        self.prompts_seen: list[str] = []
        self._judgments = judgments

    def get_json_completion(self, *, prompt, schema):
        self.prompts_seen.append(prompt)
        return StitchingJudgmentsResponse(judgments=self._judgments)


class LabelMergerTests(unittest.TestCase):
    def test_merges_the_actual_reported_case_when_the_llm_says_to(self):
        pdf_list = [
            _block(1, 2, "STRETCH YOURSELF", text_level=1),
            _block(2, 2, "1. What is the principle...?"),
            _block(3, 2, "Answer to Check Yourself"),
            _block(4, 2, "1(A) 2(A) 3(A) 4(D) 5(A)"),
        ]
        llm = _StubLLM([
            StitchingJudgment(stitched_pdf_ids=[4], explanation="label introduces the answer key"),
        ])

        result = label_merger(pdf_list, llm)

        label_block = next(b for b in result if b["pdf_id"] == 3)
        answer_block = next(b for b in result if b["pdf_id"] == 4)
        self.assertEqual(
            "Answer to Check Yourself 1(A) 2(A) 3(A) 4(D) 5(A)", label_block["text"]
        )
        self.assertTrue(answer_block.get("invalid"))

    def test_does_not_merge_a_genuine_caption_the_llm_rejects(self):
        pdf_list = [
            _block(1, 0, "Figure 3: The pulley system"),
            _block(2, 0, "The Second Law of Motion states that force equals mass times acceleration."),
        ]
        llm = _StubLLM([
            StitchingJudgment(stitched_pdf_ids=[-1], explanation="unrelated new topic"),
        ])

        result = label_merger(pdf_list, llm)

        caption = next(b for b in result if b["pdf_id"] == 1)
        other = next(b for b in result if b["pdf_id"] == 2)
        self.assertEqual("Figure 3: The pulley system", caption["text"])
        self.assertFalse(other.get("invalid", False))

    def test_no_candidates_returns_the_list_unchanged_without_calling_the_llm(self):
        pdf_list = [_block(1, 0, "This is a complete, ordinary sentence.")]

        class _ExplodingLLM:
            def get_json_completion(self, *args, **kwargs):
                raise AssertionError("must not be called when there is nothing to judge")

        result = label_merger(pdf_list, _ExplodingLLM())
        self.assertEqual(pdf_list, result)


if __name__ == "__main__":
    unittest.main()
