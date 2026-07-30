"""A listing reaches the tree as code, not as MinerU's rendering of code.

Fixtures are the real content-list entries from Basics-of-Algorithm.pdf, where
one find-the-maximum algorithm produced six blocks alternating between
sub_type "algorithm" and sub_type "code".
"""

import unittest

from docarag.adapters.mineru_compat import (
    content_block_text_parts,
    unwrap_presentation,
)

_ALGORITHM = (
    '<div class="mineru-algorithm" style="white-space: pre-wrap; '
    'font-family:monospace;">\nIf $N &lt; 1$\nThen print(\'Empty Vector\')\n'
    "Exit\n</div>"
)
_CODE = "```txt\nMAX=A[1] [We assume A[1] is the greatest]\nI=2\n```"


class TheWrapperIsRemoved(unittest.TestCase):
    def test_an_algorithm_loses_its_div_and_keeps_its_lines(self):
        self.assertEqual(
            unwrap_presentation(_ALGORITHM),
            "If $N < 1$\nThen print('Empty Vector')\nExit",
        )

    def test_a_code_block_loses_its_fence_and_keeps_its_lines(self):
        self.assertEqual(
            unwrap_presentation(_CODE),
            "MAX=A[1] [We assume A[1] is the greatest]\nI=2",
        )

    def test_html_escaping_is_undone_only_inside_a_div(self):
        # MinerU escaped the angle brackets to put them in an element. A fenced
        # block was never escaped, so an ampersand in it is the program's own.
        self.assertIn("$N < 1$", unwrap_presentation(_ALGORITHM))
        fenced = "```txt\nif a &amp;&amp; b:\n    pass\n```"
        self.assertIn("&amp;&amp;", unwrap_presentation(fenced))

    def test_a_language_tag_on_the_fence_is_not_kept(self):
        self.assertEqual(
            unwrap_presentation("```python\nif x < 1:\n    pass\n```"),
            "if x < 1:\n    pass",
        )

    def test_ordinary_text_is_untouched(self):
        prose = "The criteria require that an algorithm produces one output."
        self.assertEqual(unwrap_presentation(prose), prose)

    def test_an_unclosed_wrapper_is_left_alone(self):
        # Better to keep a stray marker than to guess where a listing ended.
        broken = '<div class="mineru-algorithm">\nIf $N < 1$'
        self.assertEqual(unwrap_presentation(broken), broken)

    def test_text_that_merely_mentions_a_fence_is_left_alone(self):
        mentions = "Use ``` to open a fenced block."
        self.assertEqual(unwrap_presentation(mentions), mentions)


class TheAdapterUnwrapsWhereverTheTextLives(unittest.TestCase):
    def test_code_body_is_unwrapped(self):
        block = {"type": "code", "sub_type": "algorithm", "code_body": _ALGORITHM}
        parts = content_block_text_parts(block)
        self.assertEqual(len(parts), 1)
        self.assertFalse(parts[0].startswith("<div"))
        self.assertIn("\n", parts[0], "a listing keeps its line breaks")

    def test_a_paragraph_still_reads_the_same(self):
        block = {"type": "text", "text": "An algorithm has zero or more inputs."}
        self.assertEqual(
            content_block_text_parts(block),
            ("An algorithm has zero or more inputs.",),
        )


if __name__ == "__main__":
    unittest.main()
