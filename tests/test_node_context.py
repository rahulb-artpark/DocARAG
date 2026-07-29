"""A leaf node's surroundings, as the summariser and the index both read them.

The window is not a fixed span. In a worked derivation the neighbours are
themselves equations joined by "Therefore," or "i.e.,", and the sentence naming
the subject can be several nodes back -- measured, 50 of 1066 equations in a
probability textbook had under six words of prose within one sibling.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from Core.pipelines.node_context import (
    MAX_STEPS,
    PROSE_WORDS,
    adjacent_context,
    context_contribution,
    is_prose,
    nearest_prose,
    node_content,
    parent_title,
    prose_word_count,
)


def node(content="", *, caption="", footnote=""):
    return SimpleNamespace(
        meta_info=SimpleNamespace(content=content, caption=caption, footnote=footnote),
        parent=None,
        children=[],
    )


def section(title, children):
    parent = node(title)
    parent.children = list(children)
    for child in parent.children:
        child.parent = parent
    return parent


class ContributionTests(unittest.TestCase):
    def test_a_figures_caption_is_part_of_what_it_contributes(self):
        figure = node("A 2D plot with a shaded band.", caption="Figure 1.5: Romeo and Juliet.")

        self.assertEqual(
            "Figure 1.5: Romeo and Juliet. A 2D plot with a shaded band.",
            context_contribution(figure),
        )

    def test_a_caption_already_inside_the_content_is_not_repeated(self):
        figure = node("Figure 1.5: Romeo and Juliet. A plot.", caption="Figure 1.5: Romeo and Juliet.")

        self.assertEqual("Figure 1.5: Romeo and Juliet. A plot.", context_contribution(figure))

    def test_a_node_without_caption_fields_still_contributes_its_text(self):
        self.assertEqual("plain sentence", context_contribution(node("plain sentence")))

    def test_missing_metadata_is_not_an_error(self):
        self.assertEqual("", node_content(SimpleNamespace()))
        self.assertEqual("", context_contribution(SimpleNamespace()))
        self.assertEqual("", parent_title(SimpleNamespace()))


class ProseTests(unittest.TestCase):
    def test_latex_is_not_prose(self):
        self.assertEqual(0, prose_word_count(r"$$\frac{1}{2} = 0.5$$"))
        self.assertEqual(0, prose_word_count(r"$\mathbf{P}(A \cap B)$"))

    def test_connectives_do_not_clear_the_bar(self):
        for connective in ("i.e.,", "Therefore,", "So, we get", "Now,"):
            with self.subTest(connective=connective):
                self.assertFalse(is_prose(node(connective)))

    def test_a_real_sentence_clears_the_bar(self):
        self.assertTrue(
            is_prose(node("Three cards are drawn from an ordinary 52-card deck."))
        )

    def test_the_bar_is_the_documented_number_of_words(self):
        self.assertEqual(PROSE_WORDS, prose_word_count("one two three four five six"))
        self.assertTrue(is_prose(node("one two three four five six")))
        self.assertFalse(is_prose(node("one two three four five")))


class WindowTests(unittest.TestCase):
    def _derivation(self):
        # A chain: the sentence naming the example is three siblings back.
        lead = node("Example 1.10. Three cards are drawn from an ordinary 52-card deck.")
        eq1 = node(r"$$A_i = \{the i-th card\}$$")
        connective = node("i.e.,")
        target = node(r"$$P(A_1 \cap A_2 \cap A_3)$$")
        trailing = node(r"$$P(A_1) = 39/52$$")
        return section("Multiplication Rule", [lead, eq1, connective, target, trailing]), target

    def test_an_adjacent_sentence_is_used_as_is(self):
        before = node("Let us use as sample space the unit square.")
        target = node(r"$$M = \{(x,y)\}$$")
        after = node("and is shaded in the figure, with area 7/16.")
        section("Probability Laws", [before, target, after])

        self.assertEqual(
            (context_contribution(before), context_contribution(after)),
            nearest_prose(target),
        )

    def test_the_walk_passes_symbol_only_siblings_to_reach_a_sentence(self):
        _tree, target = self._derivation()

        before, after = nearest_prose(target)

        self.assertIn("52-card deck", before)
        self.assertEqual("", after)

    def test_the_adjacent_window_does_not_walk(self):
        # adjacent_context is the literal neighbour, whatever it is -- the
        # index uses it where the summariser uses the walk.
        _tree, target = self._derivation()

        before, after = adjacent_context(target)

        self.assertEqual("i.e.,", before)
        self.assertIn("39/52", after)

    def test_the_walk_stops_rather_than_crossing_the_whole_section(self):
        far = node("A sentence far away from the equation in question.")
        filler = [node(r"$$x_%d = %d$$" % (i, i)) for i in range(MAX_STEPS + 2)]
        target = node(r"$$y = 1$$")
        section("Chain", [far, *filler, target])

        before, after = nearest_prose(target)

        self.assertEqual("", before)
        self.assertEqual("", after)

    def test_a_node_outside_any_parent_has_no_surroundings(self):
        orphan = node(r"$$x = 1$$")

        self.assertEqual(("", ""), nearest_prose(orphan))
        self.assertEqual(("", ""), adjacent_context(orphan))


if __name__ == "__main__":
    unittest.main()
