from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock

from docarag.text_recovery import (
    _covered_fraction,
    _is_empty_text_block,
    _normalize,
    recover_and_prune_empty_text_blocks,
)


class _FakeTextPage:
    def __init__(self, recorder):
        self.recorder = recorder

    def get_text_bounded(self, *, left, right, bottom, top):
        self.recorder.append((left, right, bottom, top))
        return "RECOVERED"


class _FakePage:
    def __init__(self, recorder):
        self.recorder = recorder

    def get_size(self):
        return (720.0, 540.0)

    def get_textpage(self):
        return _FakeTextPage(self.recorder)


class _FakeDocument:
    def __init__(self, recorder):
        self.recorder = recorder

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return _FakePage(self.recorder)

    def close(self):
        pass


class CoordinateSpaceTests(unittest.TestCase):
    """MinerU boxes blocks in page points against its own reported page size."""

    def test_a_block_is_read_from_the_region_its_points_describe(self):
        # A scale factor here silently reads the wrong part of the page and
        # grafts one region's words onto another block.
        recorder: list[tuple[float, float, float, float]] = []
        module = importlib.import_module("docarag.text_recovery")
        fake = types.SimpleNamespace(PdfDocument=lambda _path: _FakeDocument(recorder))
        blocks = [{"type": "text", "text": "", "page_idx": 0, "bbox": [128, 272, 493, 492]}]

        with mock.patch.dict(sys.modules, {"pypdfium2": fake}):
            result, recovered, _dropped = module.recover_and_prune_empty_text_blocks(
                "x.pdf", blocks
            )

        self.assertEqual(1, recovered)
        self.assertEqual("RECOVERED", result[0]["text"])
        left, right, bottom, top = recorder[0]
        self.assertEqual((128, 493), (left, right))
        # 540-point page, top-left origin: y0=272 -> 268 from the bottom.
        self.assertEqual((48, 268), (bottom, top))


class CoveredFractionTests(unittest.TestCase):
    def test_a_region_under_two_blocks_reports_their_combined_coverage(self):
        region = (128, 272, 493, 492)
        others = [(60, 160, 650, 310), (60, 320, 650, 470)]

        self.assertGreater(_covered_fraction(region, others), 0.5)

    def test_an_untouched_region_reports_nothing(self):
        self.assertEqual(0.0, _covered_fraction((0, 0, 10, 10), [(50, 50, 60, 60)]))

    def test_a_degenerate_region_never_divides_by_zero(self):
        self.assertEqual(0.0, _covered_fraction((5, 5, 5, 5), [(0, 0, 10, 10)]))


class TextRecoveryUnitTests(unittest.TestCase):
    def test_empty_text_block_detection(self):
        self.assertTrue(_is_empty_text_block({"type": "text", "text": ""}))
        self.assertTrue(_is_empty_text_block({"type": "text", "text": "   "}))
        self.assertFalse(_is_empty_text_block({"type": "text", "text": "hi"}))
        self.assertFalse(
            _is_empty_text_block({"type": "text", "text": "", "img_path": "x.jpg"})
        )

    def test_a_block_holding_evidence_in_any_channel_is_not_empty(self):
        self.assertFalse(
            _is_empty_text_block({"type": "image", "text": "", "img_path": "x.jpg"})
        )
        self.assertFalse(
            _is_empty_text_block(
                {"type": "table", "text": "", "table_body": "<table></table>"}
            )
        )
        self.assertFalse(
            _is_empty_text_block(
                {"type": "image", "text": "", "image_caption": ["Figure 1"]}
            )
        )
        self.assertFalse(
            _is_empty_text_block({"type": "equation", "text": "E = mc^2"})
        )

    def test_an_asset_type_that_produced_no_asset_is_empty(self):
        # A layout model can label a text region an equation and then recognize
        # no formula in it, leaving neither text nor crop. Assuming the type
        # implies an asset lets that block reach the contract and fail the index.
        self.assertTrue(_is_empty_text_block({"type": "equation", "text": ""}))
        self.assertTrue(_is_empty_text_block({"type": "image", "text": ""}))
        self.assertTrue(_is_empty_text_block({"type": "table", "text": ""}))

    def test_no_empty_blocks_returns_input_unchanged(self):
        blocks = [
            {"type": "text", "text": "a"},
            {"type": "table", "text": "", "table_body": "<table></table>"},
        ]
        result, recovered, dropped = recover_and_prune_empty_text_blocks("x.pdf", blocks)
        self.assertIs(result, blocks)
        self.assertEqual((0, 0), (recovered, dropped))

    def test_a_region_already_covered_by_kept_blocks_is_dropped_not_read(self):
        # Reading a spurious region back would return the same words clipped
        # mid-token, so it must be recognized as redundant before any read.
        blocks = [
            {
                "type": "text",
                "text": "BIT is a unit of information",
                "page_idx": 0,
                "bbox": [60, 160, 650, 310],
            },
            {
                "type": "text",
                "text": "BYTE is a sequence of 8 bits",
                "page_idx": 0,
                "bbox": [60, 320, 650, 470],
            },
            {"type": "equation", "text": "", "page_idx": 0, "bbox": [128, 272, 493, 492]},
        ]

        result, recovered, dropped = recover_and_prune_empty_text_blocks(
            "missing.pdf", blocks
        )

        self.assertEqual((0, 1), (recovered, dropped))
        self.assertEqual(2, len(result))
        self.assertTrue(all((b.get("text") or "").strip() for b in result))

    def test_an_uncovered_empty_region_is_not_treated_as_redundant(self):
        blocks = [
            {"type": "text", "text": "header", "page_idx": 0, "bbox": [0, 0, 100, 50]},
            {"type": "text", "text": "", "page_idx": 0, "bbox": [0, 400, 100, 450]},
        ]

        result, recovered, dropped = recover_and_prune_empty_text_blocks(
            "missing.pdf", blocks
        )

        # No native layer is available here, so it is pruned rather than
        # recovered; the point is that it was not dismissed as redundant.
        self.assertEqual((0, 1), (recovered, dropped))
        self.assertEqual(1, len(result))

    def test_empty_blocks_are_pruned_when_native_layer_unavailable(self):
        # An unreadable PDF path means no native recovery; empty layout
        # artifacts are still pruned so the document is not fail-closed, and no
        # non-empty content is touched.
        blocks = [
            {"type": "text", "text": "kept one", "page_idx": 0},
            {"type": "text", "text": "", "page_idx": 0, "bbox": [1, 2, 3, 4]},
            {"type": "text", "text": "kept two", "page_idx": 1},
        ]
        result, recovered, dropped = recover_and_prune_empty_text_blocks(
            "/nonexistent/does-not-exist.pdf", blocks
        )
        self.assertEqual(0, recovered)
        self.assertEqual(1, dropped)
        self.assertEqual(["kept one", "kept two"], [b["text"] for b in result])
        self.assertFalse(
            any(not (b.get("text") or "").strip() for b in result if b["type"] == "text")
        )

    def test_redundancy_normalization(self):
        kept = _normalize("The questions that matter most take the longest")
        fragment = _normalize("matter mos")
        self.assertIn(fragment, kept)


if __name__ == "__main__":
    unittest.main()
