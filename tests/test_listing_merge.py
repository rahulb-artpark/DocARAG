"""A listing split across blocks is rebuilt from the page layout.

Fixtures are the real geometry of three documents that each shred a listing a
different way:

  - lecs106.pdf page 6 splits Algorithm 6.2 on the lines carrying maths, leaving
    one sub_type "algorithm" block between Step 4 and Step 5 and everything else
    as single-line text.
  - Basics-of-Algorithm.pdf page 2 splits on indentation depth, alternating
    "algorithm" and "code" blocks with the step labels as text between them.
  - 016.0 SORTING.pdf page 11 splits a real Python program one line per block,
    with no algorithm label anywhere and no leading whitespace left in any piece.

The first two are grown outward from the label MinerU gave them; the third has no
label and is recovered from the fragments themselves. Each stopping case appears
in only one document, which is why all of them are needed here: lecs106's prose
follows at the same left margin as the listing, so only its wrapping separates
it; Basics' next section heading is a single line, so only its level separates
it; and lecs106's program and console transcript sit adjacent, so only the fact
that both are already whole keeps them apart.
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

    def test_two_adjacent_whole_programs_are_not_joined(self):
        # Both carry their own line breaks, so both are already complete and
        # neither is a fragment.
        page = [
            _code("print('a')\nprint('b')", top=89, height=90, page=8, language="python"),
            _code("print('c')\nprint('d')", top=200, height=90, page=8, language="python"),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 2)

    def test_two_adjacent_single_line_blocks_do_join(self):
        # Deliberate, and the opposite of what this file asserted before
        # 016.0 SORTING.pdf: two adjacent one-line code blocks are exactly what
        # one two-line program split in two looks like, and nothing distinguishes
        # them. Assuming otherwise is what left an insertion sort spread over
        # twelve nodes with no node holding the program.
        page = [
            _code("print('a')", top=89, height=40, page=8, language="python"),
            _code("print('b')", top=140, height=40, page=8, language="python"),
        ]
        listing_merger(page)
        self.assertEqual(len(_survivors(page)), 1)
        self.assertEqual(_survivors(page)[0]["text"], "print('a')\nprint('b')")


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


def _fragment(line, *, top, left, page=10, language=None, height=46):
    """One line of a program, cropped on its own.

    This is the shape 016.0 SORTING.pdf produces: a code block per line, with no
    leading whitespace left in the text at all -- the indentation survives only
    in the left edge.
    """

    block = {
        "type": "code",
        "sub_type": "code",
        "text": line,
        "code_body": f"```{language or 'txt'}\n{line}\n```",
        "bbox": [left, top, left + 300, top + height],
        "page_idx": page,
        "middle_json": {"blocks": []},
    }
    if language:
        block["code_language"] = language
    return block


# The real geometry of the insertion sort on page 11 of 016.0 SORTING.pdf, whose
# twelve lines reached the tree as twelve separate code nodes -- so no node held
# the program and the answer correctly said the document did not contain it.
_INSERTION_SORT = [
    ("aList = [21,8,15,27,2,80,1]", 238, 52, "javascript"),
    ('print("Original list is ", aList)', 296, 56, "lua"),
    ("n = len(aList)", 359, 56, "lua"),
    ("for i in range(1,n):", 418, 56, None),
    ("key = aList[i]", 475, 152, None),
    ("j=i-1", 535, 155, None),
    ("while j>=0 and aList[j]>key:", 592, 155, None),
    ("aList[j+1]=aList[j]", 651, 256, None),
    ("j=j-1", 712, 255, None),
    ("aList[j+1]=key", 770, 156, "python"),
    ('print("After pass", i,":",aList)', 825, 156, None),
    ('print("Sorted List is ",aList)', 883, 56, "lua"),
]


class AProgramIsRecoveredFromItsFragments(unittest.TestCase):
    """A split program has no algorithm label anywhere to grow a run from."""

    def _page(self):
        return [
            _text("Python Code - Insertion Sorting", top=131, left=56, page=10),
            *[_fragment(line, top=top, left=left, language=lang)
              for line, top, left, lang in _INSERTION_SORT],
        ]

    def test_every_line_ends_up_in_one_listing(self):
        page = self._page()
        listing_merger(page)
        survivors = _survivors(page)
        listings = [b for b in survivors if b["type"] == "code"]
        self.assertEqual(len(listings), 1)
        self.assertEqual(len(listings[0]["text"].splitlines()), 12)

    def test_the_indentation_is_rebuilt_from_the_left_edges(self):
        # A fragment is cropped on its own, so no leading whitespace reaches us.
        # Joining the pieces as-is produces Python that cannot run, which is the
        # whole reason the left edge has to be read.
        page = self._page()
        listing_merger(page)
        listing = [b for b in _survivors(page) if b["type"] == "code"][0]
        compile(listing["text"], "<merged>", "exec")  # raises if indentation is wrong
        lines = listing["text"].splitlines()
        self.assertEqual(lines[3], "for i in range(1,n):")
        self.assertEqual(lines[4], "    key = aList[i]")
        self.assertEqual(lines[7], "        aList[j+1]=aList[j]")
        self.assertEqual(lines[11], 'print("Sorted List is ",aList)')

    def test_a_language_the_fragments_disagree_on_is_dropped(self):
        # MinerU tagged these twelve lines of one program javascript, lua, txt
        # and python. Reporting any single one of those would be a wrong answer.
        page = self._page()
        listing_merger(page)
        listing = [b for b in _survivors(page) if b["type"] == "code"][0]
        self.assertIsNone(listing.get("code_language"))

    def test_a_language_every_fragment_agrees_on_is_kept(self):
        page = [
            _fragment("x = 1", top=100, left=56, language="python"),
            _fragment("y = 2", top=150, left=56, language="python"),
        ]
        listing_merger(page)
        self.assertEqual(_survivors(page)[0]["code_language"], "python")

    def test_the_caption_above_the_program_stays_its_own_node(self):
        page = self._page()
        listing_merger(page)
        self.assertIn(page[0], _survivors(page))


class AWholeListingIsNeverMerged(unittest.TestCase):
    """What keeps a program and the output printed under it apart.

    On lecs106.pdf a ```python program and the ```txt transcript of its console
    output sit adjacent on the page. Both arrive whole, so neither is a fragment
    and the run never starts -- verified against that document, where the pass
    leaves all nine listings exactly as they were.
    """

    def test_a_program_and_its_transcript_stay_separate(self):
        page = [
            _code("def binarySearch(list, key):\n    first = 0\n    return -1",
                  top=89, height=200, page=8, language="python"),
            _code("How many elements in your list? 4\nThe List contents are:",
                  top=300, height=60, page=8),
        ]
        before = [dict(b) for b in page]
        listing_merger(page)
        self.assertEqual(page, before)

    def test_one_lone_fragment_is_not_a_listing(self):
        page = [
            _text("A heading", top=60, level=2, page=10),
            _fragment("print('hi')", top=100, left=56),
            _text("Wrapped prose after it", top=200, lines=3, page=10),
        ]
        before = dict(page[1])
        listing_merger(page)
        self.assertEqual(page[1], before)

    def test_a_heading_between_fragments_splits_them(self):
        page = [
            _fragment("x = 1", top=100, left=56),
            _fragment("y = 2", top=150, left=56),
            _text("Second Program", top=210, left=56, level=2, page=10),
            _fragment("z = 3", top=260, left=56),
            _fragment("w = 4", top=310, left=56),
        ]
        listing_merger(page)
        listings = [b for b in _survivors(page) if b["type"] == "code"]
        self.assertEqual(len(listings), 2)
        for listing in listings:
            self.assertEqual(len(listing["text"].splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
