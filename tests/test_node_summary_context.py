"""What the summariser is told about a leaf, by node type.

A displayed equation states nothing in words. Summarised from its symbols
alone the model can only describe the mathematics -- "a diagonal band of width
1/4" for the set deciding whether Romeo and Juliet meet. Given its
surroundings it writes what the equation is for, which is what a question asks
about: measured on real nodes, +0.44 for that equation and +0.23 for one whose
example sentence sat three siblings back.

A figure already names its own subject in its caption, so it is given only the
heading it sits under. Text is self-contained and is left alone.
"""

from __future__ import annotations

import unittest

from retrieval_engine.indexes.tree import NodeType, TreeNode
from retrieval_engine.indexing.tree.summary import get_node_summary_prompt

MAX_TOKEN = 5000


def leaf(node_type, content, **meta):
    node = TreeNode({"content": content, **meta})
    node.type = node_type
    return node


def section(title, children):
    parent = leaf(NodeType.TITLE, title)
    for child in children:
        parent.add_child(child)
    return parent


class EquationPromptTests(unittest.TestCase):
    def _derivation(self):
        lead = leaf(NodeType.TEXT, "Example 1.10. Three cards are drawn from a 52-card deck.")
        connective = leaf(NodeType.TEXT, "i.e.,")
        target = leaf(NodeType.EQUATION, r"$$P(A_1 \cap A_2 \cap A_3)$$")
        section("Multiplication Rule", [lead, connective, target])
        return target

    def test_an_equation_is_given_its_section_and_its_sentences(self):
        target = self._derivation()

        prompt = get_node_summary_prompt(target, MAX_TOKEN)

        self.assertIn("Title: Multiplication Rule", prompt)
        self.assertIn("Preceding text: Example 1.10.", prompt)
        self.assertIn(r"$$P(A_1 \cap A_2 \cap A_3)$$", prompt)

    def test_the_connective_between_them_is_not_offered_as_context(self):
        # "i.e.," is the adjacent sibling; it says nothing about the subject.
        target = self._derivation()

        prompt = get_node_summary_prompt(target, MAX_TOKEN)

        self.assertNotIn("Preceding text: i.e.,", prompt)

    def test_an_equation_with_no_surroundings_still_yields_a_prompt(self):
        alone = leaf(NodeType.EQUATION, r"$$E = mc^2$$")

        prompt = get_node_summary_prompt(alone, MAX_TOKEN)

        self.assertIn(r"$$E = mc^2$$", prompt)
        self.assertNotIn("Title:", prompt)
        self.assertNotIn("Preceding text:", prompt)


class SoftContextTests(unittest.TestCase):
    def test_a_figure_is_placed_in_its_section_but_not_between_sentences(self):
        before = leaf(NodeType.TEXT, "A sentence with plenty of ordinary words in it.")
        figure = leaf(NodeType.IMAGE, "Figure 1.5: the event M.")
        after = leaf(NodeType.TEXT, "Another sentence with plenty of ordinary words.")
        section("Probability Laws", [before, figure, after])

        prompt = get_node_summary_prompt(figure, MAX_TOKEN)

        self.assertIn("Title: Probability Laws", prompt)
        self.assertIn("Figure 1.5: the event M.", prompt)
        self.assertNotIn("Preceding text:", prompt)
        self.assertNotIn("Following text:", prompt)

    def test_a_table_keeps_its_body_and_gains_only_the_heading(self):
        table = leaf(NodeType.TABLE, "Rainfall by month", table_body="|jan|feb|")
        section("Climate", [table])

        prompt = get_node_summary_prompt(table, MAX_TOKEN)

        self.assertIn("Title: Climate", prompt)
        self.assertIn("|jan|feb|", prompt)
        self.assertNotIn("Preceding text:", prompt)


class TextIsUntouchedTests(unittest.TestCase):
    def test_a_text_node_prompt_is_unchanged_by_its_surroundings(self):
        # Text is self-contained prose. Changing its prompt would invalidate
        # every text summary checkpoint for no measured gain.
        body = "The probability of the union of two events is at most their sum."
        standalone = leaf(NodeType.TEXT, body)
        embedded = leaf(NodeType.TEXT, body)
        section("Some Properties", [leaf(NodeType.TEXT, "A neighbouring sentence here."), embedded])

        self.assertEqual(
            get_node_summary_prompt(standalone, MAX_TOKEN),
            get_node_summary_prompt(embedded, MAX_TOKEN),
        )
        self.assertNotIn("Title:", get_node_summary_prompt(embedded, MAX_TOKEN))


if __name__ == "__main__":
    unittest.main()
