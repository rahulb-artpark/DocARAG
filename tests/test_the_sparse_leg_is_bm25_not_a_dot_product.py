"""The lexical leg has to rank by relevance, not by length.

Qdrant's ``Modifier.IDF`` supplies the idf factor and nothing else. The
term-frequency saturation and the document-length normalization are the
client's to apply, baked into the values it stores. Sending raw counts makes
the score a plain tf-idf dot product, and a dot product grows without bound
with length.

Measured on a 1871-page optics manual, nodes running 3 to 10047 tokens against
a median of 32: the single longest node -- an operand reference table -- came
back first for every query, because it matches "surface" two hundred times
where the heading "2.3.2.2.93. Wolter Surface" matches once. The top twenty hits
averaged 5203 tokens against a corpus mean of 78. Scoring the same corpus with
BM25 returned that heading first and averaged 216.

Entities never showed this: their names and descriptions are all one length,
which is why ``lexical_terms`` is left exactly as it is and BM25 is a second
function beside it rather than an edit to it.
"""

from __future__ import annotations

import unittest

from Core.pipelines.entity_facets import (
    BM25_B,
    BM25_K1,
    bm25_query_terms,
    bm25_terms,
    lexical_terms,
)


def _score(document: str, query: str, *, average_length: float) -> float:
    """One document's BM25 score for a query, with every term equally rare."""

    weights = bm25_terms(document, average_length=average_length)
    return sum(
        weight
        for token_id, weight in weights.items()
        if token_id in bm25_query_terms(query)
    )


class LengthStopsBuyingScore(unittest.TestCase):
    def test_a_long_document_no_longer_outscores_a_short_exact_match(self):
        heading = "Wolter Surface"
        table = "Wolter " * 200 + "operand value config " * 300

        self.assertGreater(
            _score(heading, "Wolter Surface", average_length=100.0),
            _score(table, "Wolter Surface", average_length=100.0),
        )

    def test_raw_counts_are_what_got_this_backwards(self):
        # The v8 behaviour, kept as a test so the regression is legible: with
        # raw counts the padded table wins by two orders of magnitude.
        heading = lexical_terms("Wolter Surface")
        table = lexical_terms("Wolter " * 200 + "operand value config " * 300)
        query = set(lexical_terms("Wolter Surface"))

        heading_score = sum(v for k, v in heading.items() if k in query)
        table_score = sum(v for k, v in table.items() if k in query)

        self.assertGreater(table_score, heading_score * 50)

    def test_repeating_a_term_saturates(self):
        once = _score("surface", "surface", average_length=10.0)
        many = _score("surface " * 50, "surface", average_length=10.0)

        # More is worth something, but nowhere near fifty times something.
        self.assertGreater(many, once)
        self.assertLess(many, once * (BM25_K1 + 1.0))

    def test_a_term_can_never_be_worth_more_than_the_saturation_ceiling(self):
        for repeats in (1, 10, 1000):
            with self.subTest(repeats=repeats):
                score = _score("surface " * repeats, "surface", average_length=50.0)

                self.assertLess(score, BM25_K1 + 1.0)

    def test_a_below_average_document_is_favoured_over_an_above_average_one(self):
        short = _score("surface optics", "surface", average_length=100.0)
        long = _score("surface optics " + "filler " * 200, "surface", average_length=100.0)

        self.assertGreater(short, long)


class TheQuerySideIsNotWeighted(unittest.TestCase):
    def test_every_query_term_counts_once(self):
        self.assertEqual(
            {1.0}, set(bm25_query_terms("surface surface surface optics").values())
        )

    def test_a_repeated_question_word_does_not_outvote_the_corpus(self):
        plain = bm25_query_terms("show me the Wolter Surface")
        repetitive = bm25_query_terms("show show show me the Wolter Surface")

        self.assertEqual(plain, repetitive)

    def test_the_query_keeps_the_same_token_ids_as_the_documents(self):
        # A different id space would score every query against nothing.
        self.assertEqual(
            set(lexical_terms("Wolter Surface")), set(bm25_query_terms("Wolter Surface"))
        )


class TheNormalizerHasToBeReal(unittest.TestCase):
    def test_a_zero_average_is_refused_rather_than_dividing_by_it(self):
        with self.assertRaises(ValueError):
            bm25_terms("surface", average_length=0.0)

    def test_an_empty_text_has_no_terms(self):
        self.assertEqual({}, bm25_terms("", average_length=10.0))

    def test_b_and_k1_are_the_standard_pair(self):
        self.assertEqual(1.2, BM25_K1)
        self.assertEqual(0.75, BM25_B)


class TheEntityPathIsUntouched(unittest.TestCase):
    def test_lexical_terms_still_returns_raw_counts(self):
        # bookrag-entities-hybrid-v1 was built on these values and is out of
        # scope; changing them in place would silently re-weight that index.
        self.assertEqual([3.0], list(lexical_terms("surface surface surface").values()))


if __name__ == "__main__":
    unittest.main()
