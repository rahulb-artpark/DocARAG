"""One entity must read back as one row, whatever the store returns.

Several points resolving to one entity is what BookRAG 4.3.2 must never see:
near-identical adjacent scores read as a smooth run where a sharp drop belongs,
so ``case_a`` fires and the duplicate is created instead of merged.
"""

from __future__ import annotations

import unittest

from retrieval_engine.indexing.graph.entity_facets import (
    CANONICAL_FIELD,
    _CONTEXT_DESCRIPTION_WORDS,
    collapse_entity_hits,
    context_text,
    lexical_terms,
    lexical_text,
    _stable_token_id,
)


def _hit(distance: float, node_id: str, canonical: str | None = None) -> dict:
    metadata = {"node_id": node_id, "entity_name": node_id}
    if canonical is not None:
        metadata[CANONICAL_FIELD] = canonical
    return {"distance": distance, "metadata": metadata}


class ContextTextTests(unittest.TestCase):
    def test_the_description_is_appended_to_the_node_name(self):
        self.assertEqual(
            "Name: llm\nType: METHOD\nDescription: a transformer",
            context_text("Name: llm\nType: METHOD", "a transformer"),
        )

    def test_an_empty_description_leaves_the_name_alone(self):
        self.assertEqual("Name: llm", context_text("Name: llm", ""))
        self.assertEqual("Name: llm", context_text("Name: llm", "   "))

    def test_a_long_description_is_bounded(self):
        # Both producers must cut at the same place or one entity gets
        # different vectors from different code paths. The bound is asserted
        # against the constant so the two cannot drift apart.
        text = context_text("n", " ".join(str(i) for i in range(2000)))
        self.assertEqual(
            _CONTEXT_DESCRIPTION_WORDS,
            len(text.split("Description: ")[1].split()),
        )


class LexicalTermTests(unittest.TestCase):
    def test_term_frequencies_are_counted(self):
        terms = lexical_terms("ray ray stop")
        self.assertEqual(2, len(terms))
        self.assertEqual({1.0, 2.0}, set(terms.values()))

    def test_tokenization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(
            set(lexical_terms("Qwen3-Reranker-4B")),
            set(lexical_terms("qwen3 reranker 4b")),
        )

    def test_token_ids_are_stable_across_processes(self):
        # hash() is seeded per process, so a rebuilt index would disagree with
        # the one it replaced and every sparse lookup would silently miss.
        # Values are FNV-1a 32-bit masked to 31 bits, computed from the spec
        # rather than copied from this implementation.
        self.assertEqual(1396700634, _stable_token_id("entity"))
        self.assertEqual(895062104, _stable_token_id("llm"))
        self.assertEqual(1040124287, _stable_token_id("ray"))

    def test_token_ids_stay_non_negative(self):
        for token in ("a", "zzzz", "qwen3", "reranker", "4b", "été"):
            self.assertGreaterEqual(_stable_token_id(token), 0)

    def test_empty_text_yields_no_terms(self):
        self.assertEqual({}, lexical_terms(""))
        self.assertEqual({}, lexical_terms("!!! ???"))

    def test_lexical_text_joins_the_name_and_description(self):
        self.assertEqual("llm a model", lexical_text("llm", "a model"))
        self.assertEqual("llm ", lexical_text("llm", None))


class CollapseTests(unittest.TestCase):
    def test_two_points_of_one_entity_become_one_row(self):
        hits = [_hit(0.10, "a", "c"), _hit(0.11, "c", "c"), _hit(0.40, "d", "d")]

        collapsed = collapse_entity_hits(hits, limit=10)

        self.assertEqual(2, len(collapsed))
        self.assertEqual(["c", "d"], [h["metadata"][CANONICAL_FIELD] for h in collapsed])

    def test_the_best_score_survives(self):
        hits = [_hit(0.40, "c", "c"), _hit(0.10, "a", "c")]

        collapsed = collapse_entity_hits(hits, limit=10)

        self.assertEqual(1, len(collapsed))
        self.assertAlmostEqual(0.10, collapsed[0]["distance"])
        self.assertEqual("a", collapsed[0]["metadata"]["node_id"])

    def test_the_alias_map_overrides_a_stale_canonical_payload(self):
        # canonical_id is written after a merge commits, so a search racing
        # that update sees payloads that disagree; the alias map is truth.
        hits = [_hit(0.10, "a", "c"), _hit(0.20, "c", "c")]

        collapsed = collapse_entity_hits(
            hits, limit=10, resolve=lambda node_id: "e"
        )

        self.assertEqual(1, len(collapsed))
        self.assertAlmostEqual(0.10, collapsed[0]["distance"])

    def test_results_are_ordered_by_score(self):
        hits = [_hit(0.50, "d", "d"), _hit(0.10, "c", "c"), _hit(0.30, "e", "e")]

        collapsed = collapse_entity_hits(hits, limit=10)

        self.assertEqual([0.10, 0.30, 0.50], [h["distance"] for h in collapsed])

    def test_the_limit_counts_entities_not_points(self):
        hits = [_hit(0.10, "a", "c"), _hit(0.11, "c", "c"), _hit(0.12, "d", "d")]

        self.assertEqual(1, len(collapse_entity_hits(hits, limit=1)))
        self.assertEqual(2, len(collapse_entity_hits(hits, limit=2)))

    def test_node_id_stands_in_when_canonical_is_absent(self):
        hits = [_hit(0.10, "a"), _hit(0.20, "a"), _hit(0.30, "b")]

        collapsed = collapse_entity_hits(hits, limit=10)

        self.assertEqual(2, len(collapsed))

    def test_unusable_hits_are_dropped_rather_than_crashing(self):
        hits = [
            _hit(0.10, "a", "c"),
            {"distance": 0.2, "metadata": {}},
            {"distance": "near", "metadata": {"node_id": "d"}},
            {"metadata": {"node_id": "e"}},
            "not a hit",
        ]

        collapsed = collapse_entity_hits(hits, limit=10)

        self.assertEqual(1, len(collapsed))

    def test_an_empty_result_collapses_to_nothing(self):
        self.assertEqual([], collapse_entity_hits([], limit=10))

    def test_the_limit_must_be_positive(self):
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                collapse_entity_hits([], limit=bad)


if __name__ == "__main__":
    unittest.main()
