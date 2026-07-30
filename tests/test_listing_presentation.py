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


class TheCanonicalRepresentationIsUnwrapped(unittest.TestCase):
    """The representation is what downstream selects, so it must be clean.

    Stripping the wrapper in content_block_text_parts alone left the <div> in
    the canonical text and changed nothing in the tree -- the reader was patched
    and the producer was not.
    """

    def _text_representation(self, block):
        from docarag.adapters.mineru_compat import _representations

        from docarag.domain.evidence import BlockType

        return [
            r
            for r in _representations(
                block, BlockType.CODE, confidence=None, language="en"
            )
            if isinstance(getattr(r, "content", None), str)
        ]

    def test_an_algorithm_representation_has_no_div(self):
        block = {"type": "code", "sub_type": "algorithm", "code_body": _ALGORITHM}
        contents = [r.content for r in self._text_representation(block)]
        self.assertTrue(contents, "a code block must produce a representation")
        for content in contents:
            self.assertNotIn("<div", content)
            self.assertNotIn("&lt;", content)

    def test_a_code_representation_keeps_its_newlines(self):
        block = {"type": "code", "sub_type": "code", "code_body": _CODE}
        content = self._text_representation(block)[0].content
        self.assertNotIn("```", content)
        self.assertIn("\n", content)
