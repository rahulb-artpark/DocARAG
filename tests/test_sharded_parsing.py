"""Sharded parsing preserves page continuity across shard boundaries.

Every risk here was found by reading the real merge/search code, not
assumed -- see Core/pipelines/sharded_parsing.py's module docstring for the
exact evidence each test below is checking.
"""

from __future__ import annotations

import unittest

from Core.pipelines.sharded_parsing import (
    CARRY_PAGES,
    ParsedShard,
    offset_page_indices,
    parse_sharded,
    shard_ranges,
)


class ShardRangesTests(unittest.TestCase):
    def test_even_division(self):
        self.assertEqual([(0, 1), (2, 3)], list(shard_ranges(4, 2)))

    def test_final_shard_is_bounded_by_page_count_not_shard_size(self):
        self.assertEqual([(0, 1), (2, 3), (4, 4)], list(shard_ranges(5, 2)))

    def test_a_document_smaller_than_one_shard_is_one_shard(self):
        self.assertEqual([(0, 4)], list(shard_ranges(5, 50)))

    def test_rejects_nonpositive_inputs(self):
        with self.assertRaises(ValueError):
            list(shard_ranges(0, 10))
        with self.assertRaises(ValueError):
            list(shard_ranges(10, 0))


class OffsetPageIndicesTests(unittest.TestCase):
    def test_zero_offset_is_a_true_noop(self):
        middle = {"pdf_info": [{"page_idx": 0}]}
        content = [{"page_idx": 0, "text": "x"}]
        out_middle, out_content = offset_page_indices(middle, content, 0)
        self.assertIs(out_middle, middle)
        self.assertIs(out_content, content)

    def test_shifts_page_idx_in_both_middle_json_and_content_list(self):
        middle = {"pdf_info": [{"page_idx": 0}, {"page_idx": 1}]}
        content = [{"page_idx": 0}, {"page_idx": 1}]

        out_middle, out_content = offset_page_indices(middle, content, 50)

        self.assertEqual([50, 51], [p["page_idx"] for p in out_middle["pdf_info"]])
        self.assertEqual([50, 51], [c["page_idx"] for c in out_content])

    def test_original_inputs_are_not_mutated(self):
        middle = {"pdf_info": [{"page_idx": 0}]}
        content = [{"page_idx": 0}]

        offset_page_indices(middle, content, 10)

        self.assertEqual(0, middle["pdf_info"][0]["page_idx"])
        self.assertEqual(0, content[0]["page_idx"])

    def test_items_without_a_page_idx_pass_through_untouched(self):
        content = [{"text": "no page field"}]
        _, out_content = offset_page_indices({"pdf_info": []}, content, 5)
        self.assertEqual([{"text": "no page field"}], out_content)

    def test_rejects_negative_offset(self):
        with self.assertRaises(ValueError):
            offset_page_indices({"pdf_info": []}, [], -1)


def _page(idx: int, text: str, *, incomplete: bool = False) -> dict:
    return {"page_idx": idx, "type": "text", "text": text, "_incomplete": incomplete}


def _fake_parse_fn(calls: list[tuple[int, int]], pages_per_shard: dict):
    """A parse_fn double that records exactly which page range it was asked
    for, and returns only that range's synthetic content -- if the loop ever
    asked for the whole document, this would return everything at once and
    the "never sees more than a shard" tests below would fail."""

    def parse_fn(pdf_path, output_dir, *, start_page_id, end_page_id):
        calls.append((start_page_id, end_page_id))
        content = [
            item
            for page in range(start_page_id, end_page_id + 1)
            for item in pages_per_shard.get(page, [])
        ]
        # page_idx must come back shard-relative, exactly like real MinerU
        # does (confirmed live) -- tests the offset step, not a shortcut.
        rebased = [{**item, "page_idx": item["page_idx"] - start_page_id} for item in content]
        middle = {"pdf_info": [{"page_idx": p - start_page_id} for p in range(start_page_id, end_page_id + 1)]}
        return middle, rebased

    return parse_fn


def _merging_refine_fn(items, start_pdf_id):
    """Simulates pdf_info_refiner: assigns a globally-continuing pdf_id, and
    merges an item marked _incomplete into the item immediately following it
    -- the same "does this continue into the next block" shape as the real
    text_merger, without needing the real one (or an LLM) in a unit test."""

    merged = []
    i = 0
    while i < len(items):
        item = items[i]
        if item.get("_incomplete") and i + 1 < len(items):
            nxt = items[i + 1]
            merged.append({**item, "text": item["text"] + " " + nxt["text"], "_incomplete": False})
            i += 2
        else:
            merged.append(item)
            i += 1
    for offset, item in enumerate(merged):
        item["pdf_id"] = start_pdf_id + offset
    return merged


class ParseShardedTests(unittest.TestCase):
    def test_parse_fn_never_sees_more_than_one_shard_at_a_time(self):
        pages = {p: [_page(p, f"page{p}")] for p in range(6)}
        calls: list[tuple[int, int]] = []

        list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=6, shard_size=2,
                parse_fn=_fake_parse_fn(calls, pages),
            )
        )

        self.assertEqual([(0, 1), (2, 3), (4, 5)], calls)

    def test_document_relative_page_idx_is_correct_after_every_shard(self):
        pages = {p: [_page(p, f"page{p}")] for p in range(6)}
        calls: list[tuple[int, int]] = []

        shards = list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=6, shard_size=2,
                parse_fn=_fake_parse_fn(calls, pages),
            )
        )

        all_page_idx = [item["page_idx"] for shard in shards for item in shard.content_list]
        self.assertEqual(list(range(6)), sorted(all_page_idx))

    def test_a_paragraph_split_exactly_at_the_shard_boundary_still_merges(self):
        # Page 1 (last page of shard 0) ends mid-sentence; page 2 (first page
        # of shard 1) continues it. A naive per-shard refiner would emit two
        # disconnected fragments instead of one merged paragraph.
        pages = {
            0: [_page(0, "Intro.")],
            1: [_page(1, "The result was", incomplete=True)],
            2: [_page(2, "inconclusive.")],
            3: [_page(3, "Conclusion.")],
        }
        calls: list[tuple[int, int]] = []

        shards = list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=4, shard_size=2,
                parse_fn=_fake_parse_fn(calls, pages),
                refine_fn=_merging_refine_fn,
            )
        )

        all_text = [item["text"] for shard in shards for item in shard.content_list]
        self.assertIn("The result was inconclusive.", all_text)
        self.assertNotIn("The result was", all_text)  # not left unmerged
        self.assertNotIn("inconclusive.", all_text)  # not left unmerged

    def test_pdf_id_is_globally_unique_and_monotonic_across_shards(self):
        pages = {p: [_page(p, f"page{p}")] for p in range(6)}
        calls: list[tuple[int, int]] = []

        shards = list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=6, shard_size=2,
                parse_fn=_fake_parse_fn(calls, pages),
                refine_fn=_merging_refine_fn,
            )
        )

        all_ids = [item["pdf_id"] for shard in shards for item in shard.content_list]
        self.assertEqual(sorted(all_ids), all_ids)  # monotonic
        self.assertEqual(len(all_ids), len(set(all_ids)))  # unique

    def test_the_last_shard_flushes_everything_carried(self):
        pages = {p: [_page(p, f"page{p}")] for p in range(4)}
        calls: list[tuple[int, int]] = []

        shards = list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=4, shard_size=2,
                parse_fn=_fake_parse_fn(calls, pages),
            )
        )

        self.assertEqual(4, sum(len(s.content_list) for s in shards))

    def test_carry_pages_is_a_page_count_not_an_item_count(self):
        # Three items on the same last page must all carry together, not
        # just "the last item" -- a table's caption and body can be two
        # separate items on the same boundary page.
        pages = {
            0: [_page(0, "a"), _page(0, "b"), _page(0, "c")],
            1: [_page(1, "d")],
        }
        calls: list[tuple[int, int]] = []
        captured_combined = []

        def capturing_refine(items, start_pdf_id):
            captured_combined.append(len(items))
            return _merging_refine_fn(items, start_pdf_id)

        list(
            parse_sharded(
                "doc.pdf", "/out",
                page_count=2, shard_size=1,
                parse_fn=_fake_parse_fn(calls, pages),
                refine_fn=capturing_refine,
                carry_pages=2,
            )
        )

        # Second shard's refiner call must have seen all 3 carried items
        # from page 0 plus its own item from page 1.
        self.assertEqual(4, captured_combined[1])


if __name__ == "__main__":
    unittest.main()
