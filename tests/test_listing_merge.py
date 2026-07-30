"""A pseudocode listing split across blocks is rebuilt from the page layout.

Fixtures are the real geometry of the two documents that shred a listing two
different ways: lecs106.pdf page 6 splits Algorithm 6.2 on the lines carrying
maths, leaving one sub_type "algorithm" block between Step 4 and Step 5 with
everything else as single-line text; Basics-of-Algorithm.pdf page 2 splits on
indentation depth, alternating "algorithm" and "code" blocks with the step
labels as text between them.

The two stopping cases are what the rule has to get right, and each document
only exhibits one of them: lecs106's prose follows at the same left margin as
the listing (so only its wrapping separates it), and Basics' next section
heading is a single line (so only its level separates it).
"""

import unittest

from Core.pipelines.pdf_refiner import listing_merger


def _text(text, *, top, left=95, right=480, lines=1, level=None, page=6):
    block = {
        "type": "text",
        "text": text,
        "bbox": [left, top, right, top + 18],
        "page_idx": page,
        "middle_json": {"lines": [{}] * lines},
    }
    if level is not None:
        block["text_level"] = level
    return block


def _algorithm(body, *, top, left=202, right=582, height=131, page=6):
    """An anchor in the shape the pass actually receives.

    listing_merger runs after bind_canonical_evidence, which overwrites "type"
    with the canonical projection's node type and fills "text" from the selected
    representation. So an anchor arrives typed "algorithm" with its listing
    already unwrapped into "text" -- not typed "code" with the text still inside
    a wrapped code_body, which is MinerU's shape and is what the first version
    of this pass looked for. It found nothing on every real document.
    """

    return {
        "type": "algorithm",
        "sub_type": "algorithm",
        "text": body,
        "code_body": (
            '<div class="mineru-algorithm" style="white-space: pre-wrap; '
            f'font-family:monospace;">\n{body}\n</div>'
        ),
        "bbox": [left, top, right, top + height],
        "page_idx": page,
        "middle_json": {"blocks": []},
    }


def _code(body, *, top, left=199, right=520, height=43, page=2, language="txt"):
    return {
        "type": "code",
        "sub_type": "code",
        "text": body,
        "code_body": f"```{language}\n{body}\n```",
        "bbox": [left, top, right, top + height],
        "page_idx": page,
        "middle_json": {"blocks": []},
    }


def _survivors(pdf_list):
    return [block for block in pdf_list if not block.get("invalid")]


def _listing(pdf_list):
    found = [b for b in _survivors(pdf_list) if b.get("sub_type") == "algorithm"]
    assert len(found) == 1, f"expected one listing, got {len(found)}"
    return found[0]


class TheAnchorIsFoundInEitherShape(unittest.TestCase):
    """The pass must not depend on where it sits relative to the bind step.

    Typing the anchor "algorithm" is what bind_canonical_evidence does; typing
    it "code" with sub_type "algorithm" is MinerU's own content list. Looking
    for only the second made this pass a no-op in production while every unit
    test passed, so both are covered.
    """

    def _page(self, anchor):
        return [anchor, _text("Step 2: continue", top=135)]

    def test_the_post_bind_shape_is_an_anchor(self):
        page = self._page(_algorithm("Step 1", top=89, height=40))
        listing_merger(page)
        self.assertIn("Step 2: continue", _listing(page)["text"])

    def test_the_raw_mineru_shape_is_an_anchor(self):
        anchor = _algorithm("Step 1", top=89, height=40)
        anchor["type"] = "code"
        anchor.pop("text")
        page = self._page(anchor)
        listing_merger(page)
        self.assertIn("Step 2: continue", _listing(page)["text"])


class ASplitListingIsRejoined(unittest.TestCase):
    """lecs106.pdf page 6: split on the lines carrying maths."""

    def _page(self):
        return [
            _text("Algorithm 6.2: Binary Search", top=64, level=2),
            _text("BinarySearch(numList, key)", top=89),
            _text("Step 1: SET first = 0, last = n-1", top=113),
            _text("Step 2: Calculate mid = (first+last)//2", top=137),
            _text("Step 3: WHILE first <= last REPEAT Step 4", top=160),
            _text("Step 4: IF numList[mid] = key", top=184),
            _algorithm('PRINT "Element found at position",\n    STOP\nELSE', top=207),
            _text("Step 5: PRINT 'Search unsuccessful'", top=347),
            _text(
                "Example 6.2 Consider a sorted list comprising of 15 elements.",
                top=375,
                lines=3,
            ),
        ]

    def test_the_steps_are_rejoined_in_reading_order(self):
        page = self._page()
        listing_merger(page)
        self.assertEqual(
            _listing(page)["text"].splitlines(),
            [
                "BinarySearch(numList, key)",
                "Step 1: SET first = 0, last = n-1",
                "Step 2: Calculate mid = (first+last)//2",
                "Step 3: WHILE first <= last REPEAT Step 4",
                "Step 4: IF numList[mid] = key",
                'PRINT "Element found at position",',
                "    STOP",
                "ELSE",
                "Step 5: PRINT 'Search unsuccessful'",
            ],
        )

    def test_the_listings_own_indentation_survives(self):
        # A listing whose leading whitespace is lost is not a listing. This is
        # the same reason the frontend renders it in a <pre>.
        page = self._page()
        listing_merger(page)
        self.assertIn("\n    STOP\n", _listing(page)["text"])

    def test_wrapped_prose_at_the_same_left_margin_is_not_absorbed(self):
        # "Example 6.2 ..." starts at x=95 like every step, and the gap above it
        # is no larger than the gaps inside the listing. Only its wrapping over
        # three lines says it is a paragraph.
        page = self._page()
        listing_merger(page)
        self.assertNotIn("Example 6.2", _listing(page)["text"])
        self.assertIn(page[8], _survivors(page))

    def test_the_caption_stays_its_own_node(self):
        # The heading is the listing's title and a tree node in its own right,
        # so it ends the run upward rather than being swallowed by it.
        page = self._page()
        listing_merger(page)
        self.assertNotIn("Algorithm 6.2", _listing(page)["text"])
        self.assertIn(page[0], _survivors(page))

    def test_the_pieces_are_marked_invalid_exactly_once(self):
        page = self._page()
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 3)

    def test_the_bbox_covers_every_piece(self):
        page = self._page()
        listing_merger(page)
        self.assertEqual(_listing(page)["bbox"], [95.0, 89.0, 582.0, 365.0])

    def test_the_wrapped_code_body_does_not_linger(self):
        # Every reader prefers "text", so a leftover code_body would be a second
        # reading of the same block that nothing consumes.
        page = self._page()
        listing_merger(page)
        self.assertNotIn("code_body", _listing(page))

    def test_the_merge_is_recorded_as_derived(self):
        page = self._page()
        listing_merger(page)
        self.assertIn(
            "bookrag.listing_merge.v1",
            _listing(page).get("transformations") or [],
        )


class AlternatingPiecesCoalesceIntoOneListing(unittest.TestCase):
    """Basics-of-Algorithm.pdf page 2: split on indentation depth."""

    def _page(self):
        return [
            _algorithm("If $N < 1$\nExit", top=89, left=197, height=70, page=2),
            _text("2. [Initialize]", top=167, left=171, lines=1, page=2),
            _code("MAX=A[1]\nI=2", top=194, page=2),
            _text("3. [Examine all elements]", top=247, left=171, page=2),
            _algorithm("Do while $I <= N$", top=273, left=200, height=16, page=2),
            _text("3.1 [Change MAX if smaller]", top=299, left=200, page=2),
            _code("If MAX < A[I]\nThen MAX = A[I]", top=325, left=228, height=44, page=2),
            _text("3.2 [Prepare to examine next element]", top=378, left=200, page=2),
            _algorithm("$I = I + I$", top=404, left=230, height=16, page=2),
            _text("4. [Finished]", top=430, left=171, page=2),
            _code("Exit", top=457, left=200, height=15, page=2),
            _text("2.3 Format Convention of algorithm", top=491, left=111, level=2, page=2),
            _text("Hence from this example we can consider", top=532, left=109, lines=3, page=2),
        ]

    def test_all_three_anchors_become_one_listing(self):
        page = self._page()
        listing_merger(page)
        self.assertEqual(
            _listing(page)["text"].splitlines(),
            [
                "If $N < 1$",
                "Exit",
                "2. [Initialize]",
                "MAX=A[1]",
                "I=2",
                "3. [Examine all elements]",
                "Do while $I <= N$",
                "3.1 [Change MAX if smaller]",
                "If MAX < A[I]",
                "Then MAX = A[I]",
                "3.2 [Prepare to examine next element]",
                "$I = I + I$",
                "4. [Finished]",
                "Exit",
            ],
        )

    def test_a_single_line_heading_ends_the_run(self):
        # This heading is one line and sits below the listing at a small gap, so
        # the wrapping test cannot reject it -- its level is what does.
        page = self._page()
        listing_merger(page)
        self.assertNotIn("Format Convention", _listing(page)["text"])
        self.assertIn(page[11], _survivors(page))

    def test_the_absorbed_code_pieces_stop_being_their_own_nodes(self):
        # These are the four ```txt pieces that reached the tree as separate
        # code nodes when they are lines of one algorithm.
        page = self._page()
        listing_merger(page)
        self.assertEqual([b.get("sub_type") for b in _survivors(page)], ["algorithm", None, None])


class RealCodeIsNeverTouched(unittest.TestCase):
    """The anchor is MinerU's own label, so a program is never a candidate."""

    def test_a_python_listing_is_left_exactly_as_it_arrived(self):
        page = [
            _text("Program 6-2 Binary Search", top=64, level=2, page=8),
            _code(
                "def binarySearch(list, key):\n    first = 0\n    return -1",
                top=89,
                height=200,
                page=8,
                language="python",
            ),
            _code("How many elements? 4", top=300, height=60, page=8),
        ]
        before = [dict(block) for block in page]
        listing_merger(page)
        self.assertEqual(page, before)

    def test_two_adjacent_programs_are_not_joined(self):
        page = [
            _code("print('a')", top=89, height=40, page=8, language="python"),
            _code("print('b')", top=140, height=40, page=8, language="python"),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 2)


class TheRunStopsWhereTheListingDoes(unittest.TestCase):
    def test_a_neighbouring_column_is_not_absorbed(self):
        page = [
            _algorithm("Step 1", top=89, height=40),
            _text("A note beside the listing", top=95, left=700, right=880),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 2)

    def test_a_distant_block_is_not_absorbed(self):
        page = [
            _algorithm("Step 1", top=89, height=40),
            _text("Step 2 far below", top=400),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 2)

    def test_a_table_ends_the_run(self):
        page = [
            _algorithm("Step 1", top=89, height=40),
            {"type": "table", "table_body": "<table/>", "bbox": [95, 135, 500, 200], "page_idx": 6},
            _text("Step 2", top=210),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 3)

    def test_a_lone_anchor_is_left_alone(self):
        page = [
            _text("Some heading", top=64, level=2),
            _algorithm("If $N < 1$\nExit", top=200, height=40),
            _text("Wrapped prose follows", top=400, lines=4),
        ]
        before = dict(page[1])
        listing_merger(page)
        listing = _listing(page)
        # Nothing was rejoined, so nothing about the block may change -- in
        # particular it keeps its code_body and earns no merge provenance.
        self.assertEqual(listing, before)
        self.assertNotIn("transformations", listing)

    def test_a_block_without_a_bbox_does_not_stop_the_pass(self):
        page = [
            {"type": "text", "text": "no geometry", "page_idx": 6},
            _algorithm("Step 1", top=89, height=40),
            _text("Step 2", top=135),
        ]
        listing_merger(page)
        self.assertIn("Step 2", _listing(page)["text"])

    def test_pages_are_kept_apart(self):
        page = [
            _algorithm("Step 1", top=880, height=40, page=6),
            _text("A first line on the next page", top=64, page=7),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 2)


if __name__ == "__main__":
    unittest.main()
