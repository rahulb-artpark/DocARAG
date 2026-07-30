"""A MinerU 3.x list must reconcile against its own middle-json block.

MinerU 3.x moved a content-list list's text out of ``text`` and into
``list_items``, leaving no ``text`` key at all.  Read through ``text`` alone the
block has no semantic text, so it can never match the middle-json block it came
from; under the retain policy both views were then emitted and the same list
reached the tree twice, at two different coordinate scales.

The whole suite passed while that was true, because nothing followed one list
from the two parser views through to a single output block.
"""

import tempfile
import unittest

from Core.provider.mineru_reconciliation import reconcile_middle_content
from docarag.adapters.mineru_compat import content_block_text_parts

_ITEMS = (
    "• Cannot su nearby olyects distinctly",
    "• Can se distant oljeets dlearly.",
    "• Imageof nearly olyjeet is formed behind the retina.",
)


def _span_block(bbox, index, content):
    return {
        "bbox": bbox,
        "type": "text",
        "index": index,
        "lines": [{"bbox": bbox, "spans": [{"bbox": bbox, "type": "text",
                                            "content": content, "score": 0.9}]}],
    }


def _middle_json():
    return {
        "pdf_info": [{
            "page_idx": 0,
            "page_size": [595, 842],
            "para_blocks": [{
                "bbox": [45, 491, 279, 579],
                "type": "list",
                "index": 11,
                "blocks": [
                    _span_block([45, 491, 279, 515], 12, _ITEMS[0]),
                    _span_block([45, 514, 256, 537], 13, _ITEMS[1]),
                    _span_block([46, 535, 270, 579], 14, _ITEMS[2]),
                ],
            }],
            "preproc_blocks": [],
            "discarded_blocks": [],
        }]
    }


def _content_list():
    """The real MinerU 3.x shape: list_items, sub_type, and no ``text`` key."""

    return [{
        "type": "list",
        "sub_type": "text",
        "list_items": list(_ITEMS),
        "bbox": [522, 587, 942, 684],
        "page_idx": 0,
    }]


class ListTextIsFoundWhereMinerUPutsIt(unittest.TestCase):
    def test_list_items_are_read_as_the_blocks_text(self):
        parts = content_block_text_parts(_content_list()[0])
        self.assertEqual(parts, _ITEMS)

    def test_a_text_key_still_wins_when_present(self):
        parts = content_block_text_parts({"text": "plain", "list_items": list(_ITEMS)})
        self.assertEqual(parts, ("plain",))

    def test_code_body_is_read_for_a_code_block(self):
        parts = content_block_text_parts({"type": "code", "code_body": "print(1)"})
        self.assertEqual(parts, ("print(1)",))

    def test_a_block_with_no_text_bearing_key_yields_nothing(self):
        self.assertEqual(content_block_text_parts({"type": "image"}), ())

    def test_an_empty_list_items_does_not_shadow_a_later_key(self):
        parts = content_block_text_parts({"list_items": ["", "  "], "code_body": "x"})
        self.assertEqual(parts, ("x",))


class OneListReachesOutputOnce(unittest.TestCase):
    def _reconcile(self):
        with tempfile.TemporaryDirectory() as parse_dir:
            return reconcile_middle_content(
                _middle_json(), _content_list(), parse_dir, mismatch_policy="retain"
            )

    def test_the_two_views_pair_into_a_single_output_block(self):
        result = self._reconcile()
        summary = result.diagnostics["summary"]
        self.assertEqual(summary["middle_count"], 1)
        self.assertEqual(summary["content_count"], 1)
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["output_count"], 1, "the list was emitted twice")
        self.assertEqual(len(result.items), 1)

    def test_nothing_is_left_unmatched(self):
        diagnostics = self._reconcile().diagnostics
        self.assertEqual(diagnostics["unmatched_middle"], [])
        self.assertEqual(diagnostics["unmatched_content"], [])
        self.assertEqual(diagnostics["status"], "matched")

    def test_the_pairing_is_earned_by_text_not_by_position(self):
        methods = self._reconcile().diagnostics["summary"]["match_methods"]
        self.assertEqual(methods.get("exact_text"), 1, methods)
