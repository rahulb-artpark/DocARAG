from __future__ import annotations

import unittest

from docarag.text_recovery import (
    _is_empty_text_block,
    _normalize,
    recover_and_prune_empty_text_blocks,
)


class TextRecoveryUnitTests(unittest.TestCase):
    def test_empty_text_block_detection(self):
        self.assertTrue(_is_empty_text_block({"type": "text", "text": ""}))
        self.assertTrue(_is_empty_text_block({"type": "text", "text": "   "}))
        self.assertFalse(_is_empty_text_block({"type": "text", "text": "hi"}))
        self.assertFalse(_is_empty_text_block({"type": "image", "text": ""}))
        self.assertFalse(_is_empty_text_block({"type": "table", "text": ""}))
        self.assertFalse(
            _is_empty_text_block({"type": "text", "text": "", "img_path": "x.jpg"})
        )

    def test_no_empty_blocks_returns_input_unchanged(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "table", "text": ""}]
        result, recovered, dropped = recover_and_prune_empty_text_blocks("x.pdf", blocks)
        self.assertIs(result, blocks)
        self.assertEqual((0, 0), (recovered, dropped))

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
