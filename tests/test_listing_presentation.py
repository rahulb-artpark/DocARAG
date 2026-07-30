"""A listing reaches the tree as code, not as MinerU's rendering of code.

Fixtures are the real content-list entries from Basics-of-Algorithm.pdf, where
one find-the-maximum algorithm produced six blocks alternating between
sub_type "algorithm" and sub_type "code".
"""

import unittest

from docarag.adapters.mineru_compat import (
    content_block_text_parts,
    fence_language,
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


class TheFenceLanguageIsKept(unittest.TestCase):
    """MinerU names the language on the fence, and the name is worth keeping.

    A named language identifies a program: on lecs106.pdf every ```python block
    is one of the book's Programs. The absence of a name means only that MinerU
    could not name the language -- it is deliberately not read as "this is not a
    program". On lecs106 the three untagged blocks are console transcripts, but
    on Basics-of-Algorithm.pdf all four untagged blocks are pseudocode, so
    retyping an untagged listing on the strength of the missing tag would be
    wrong on one of the only two documents that exercise it.
    """

    def test_a_named_language_is_returned(self):
        self.assertEqual(fence_language("```python\ndef f():\n    pass\n```"), "python")
        self.assertEqual(fence_language("```cpp\nint main() {}\n```"), "cpp")

    def test_a_placeholder_language_is_not_a_language(self):
        for fence in ("txt", "text", "plain", "plaintext", "none", ""):
            self.assertIsNone(
                fence_language(f"```{fence}\nHow many elements in your list? 4\n```"),
                f"'{fence}' names no language",
            )

    def test_case_and_padding_do_not_matter(self):
        self.assertEqual(fence_language("```  Python  \nx = 1\n```"), "python")

    def test_unfenced_text_has_no_language(self):
        self.assertIsNone(fence_language("def f(): pass"))
        self.assertIsNone(fence_language(None))

    def test_an_algorithm_div_has_no_language(self):
        self.assertIsNone(fence_language(_ALGORITHM))


class TheLanguageReachesTheProjection(unittest.TestCase):
    """The fence is stripped before projection, so the name has to be carried.

    It travels in the representation's metadata rather than in
    Representation.language, which is the natural language of the text.
    """

    def _representations(self, block):
        from docarag.adapters.mineru_compat import _representations
        from docarag.domain.evidence import BlockType

        return _representations(block, BlockType.CODE, confidence=None, language="en")

    def _listing_language(self, block):
        for representation in self._representations(block):
            name = representation.metadata.get("listing_language")
            if name is not None:
                return name
        return None

    def test_a_named_language_is_carried_on_the_representation(self):
        block = {"type": "code", "code_body": "```python\nx = 1\n```"}
        self.assertEqual(self._listing_language(block), "python")

    def test_the_natural_language_field_is_not_overloaded(self):
        block = {"type": "code", "code_body": "```python\nx = 1\n```"}
        for representation in self._representations(block):
            self.assertEqual(representation.language, "en")

    def test_an_untagged_listing_carries_no_language(self):
        block = {"type": "code", "code_body": _CODE}
        self.assertIsNone(self._listing_language(block))

    def test_an_algorithm_carries_no_language(self):
        block = {"type": "code", "sub_type": "algorithm", "code_body": _ALGORITHM}
        self.assertIsNone(self._listing_language(block))

    def test_a_merged_listing_carries_no_language(self):
        # listing_merger writes the rejoined listing into "text" and drops the
        # code_body, so there is no fence left to read a language off.
        block = {"type": "code", "sub_type": "algorithm", "text": "Step 1\nStep 2"}
        self.assertIsNone(self._listing_language(block))


if __name__ == "__main__":
    unittest.main()
