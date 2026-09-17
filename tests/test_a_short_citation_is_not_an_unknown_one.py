"""A sub-answer's evidence is not unknown because it was named the short way.

Evidence for a sub-question is labelled ``Q1-E1``, and the answer prompt offers
the model both that form and the plain ``E1``. A model that takes the shorter
one is citing evidence it was genuinely given, under a name the binding map
does not hold -- so validation reported every marker as ``unknown_citations``
and the whole synthesis was thrown away before generation.

On one 1871-page manual the same question, unchanged, answered at 13:15,
failed at 14:51 and answered at 15:29 with no code change between them. The
only thing that varied was which of the two offered forms the model chose.

The markers are namespaced to the scope the evidence already carries before
they are judged, the same rewrite the final synthesis already performs.
"""

from __future__ import annotations

import unittest

from retrieval_engine.rag.gbc.answer import _evidence_scope_of, _scope_answer_markers


def _nodes(*labels):
    return [{"evidence_label": label} for label in labels]


class TheScopeIsReadFromTheEvidence(unittest.TestCase):
    def test_a_sub_question_evidence_set_reports_its_scope(self):
        self.assertEqual("Q1", _evidence_scope_of(_nodes("Q1-E1", "Q1-E2")))

    def test_a_second_sub_question_reports_its_own(self):
        self.assertEqual("Q2", _evidence_scope_of(_nodes("Q2-E1", "Q2-E7")))

    def test_a_simple_question_has_no_scope_to_apply(self):
        self.assertIsNone(_evidence_scope_of(_nodes("E1", "E2")))

    def test_a_mixed_set_normalizes_nothing(self):
        # Two scopes in one binding map: rewriting E1 would have to guess.
        self.assertIsNone(_evidence_scope_of(_nodes("Q1-E1", "Q2-E1")))

    def test_an_empty_or_unlabelled_set_is_left_alone(self):
        self.assertIsNone(_evidence_scope_of([]))
        self.assertIsNone(_evidence_scope_of([{}]))
        self.assertIsNone(_evidence_scope_of(_nodes("Q1-E1") + [{}]))

    def test_a_label_that_is_not_a_scope_is_refused(self):
        self.assertIsNone(_evidence_scope_of(_nodes("X1-E1")))


class TheShortFormIsNamespacedBeforeItIsJudged(unittest.TestCase):
    def test_a_short_marker_becomes_the_scoped_one(self):
        self.assertEqual(
            "The sag equation [Q1-E1] defines it [Q1-E2].",
            _scope_answer_markers("The sag equation [E1] defines it [E2].", "Q1"),
        )

    def test_an_already_scoped_marker_is_untouched(self):
        answer = "Zernike surface [Q1-E1] and its equation [Q1-E4]."
        self.assertEqual(answer, _scope_answer_markers(answer, "Q1"))

    def test_a_mixture_of_both_forms_ends_up_scoped(self):
        self.assertEqual(
            "one [Q1-E1] two [Q1-E2]",
            _scope_answer_markers("one [E1] two [Q1-E2]", "Q1"),
        )

    def test_applying_it_twice_changes_nothing_further(self):
        # The complex path scopes the sub-answer again on the way to synthesis.
        once = _scope_answer_markers("a [E1] b [E2]", "Q1")
        self.assertEqual(once, _scope_answer_markers(once, "Q1"))

    def test_prose_that_is_not_a_marker_is_left_alone(self):
        # Note this is the marker regex the final synthesis already uses: it
        # recognizes [E1] wherever it appears, so a bracketed number or a bare
        # scope is untouched but an [E1] embedded in prose is still a marker.
        for text in ("see figure [3]", "[Q1]", "no markers", "[E0]"):
            with self.subTest(text=text):
                self.assertEqual(text, _scope_answer_markers(text, "Q1"))


class TheGateStillRejectsWhatItShould(unittest.TestCase):
    """Namespacing must not turn a genuinely unknown citation into a valid one."""

    def test_a_label_beyond_the_evidence_is_still_unknown(self):
        from document_services.retrieval.answer_validation import validate_answer_citations

        scope = _evidence_scope_of(_nodes("Q1-E1", "Q1-E2"))
        answer = _scope_answer_markers("cites [E9]", scope)
        result = validate_answer_citations(
            answer,
            {"Q1-E1": "citation_" + "a" * 32, "Q1-E2": "citation_" + "b" * 32},
            required=True,
        )
        self.assertEqual("unknown_citations", result.status)
        self.assertIn("Q1-E9", result.unknown_labels)

    def test_the_short_form_now_resolves_to_its_evidence(self):
        from document_services.retrieval.answer_validation import validate_answer_citations

        scope = _evidence_scope_of(_nodes("Q1-E1", "Q1-E2"))
        answer = _scope_answer_markers("the sag equation [E1]", scope)
        result = validate_answer_citations(
            answer,
            {"Q1-E1": "citation_" + "a" * 32, "Q1-E2": "citation_" + "b" * 32},
            required=True,
        )
        self.assertEqual("verified", result.status)
        self.assertEqual(("Q1-E1",), result.cited_labels)


if __name__ == "__main__":
    unittest.main()
