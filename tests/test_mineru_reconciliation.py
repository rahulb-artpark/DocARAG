import json
import tempfile
import unittest
from pathlib import Path

from retrieval_engine.parsing.mineru_reconciliation import (
    MinerUReconciliationError,
    reconcile_middle_content,
)


def _block(text, *, block_type="text", bbox=None, **extra):
    block = {
        "type": block_type,
        "lines": [{"spans": [{"type": "text", "content": text}]}],
        **extra,
    }
    if bbox is not None:
        block["bbox"] = bbox
    return block


def _page(page_idx, *blocks):
    return {
        "page_idx": page_idx,
        "page_size": [100, 100],
        "para_blocks": list(blocks),
    }


def _middle(*pages):
    return {"pdf_info": list(pages)}


class MinerUReconciliationTests(unittest.TestCase):
    def test_internal_missing_content_block_does_not_shift_later_blocks(self):
        middle = _middle(
            _page(
                0,
                _block("alpha", block_id="a"),
                _block("missing in content", block_id="b"),
                _block("charlie", block_id="c"),
            )
        )
        content = [
            {"type": "text", "text": "alpha", "page_idx": 0, "block_id": "a"},
            {
                "type": "text",
                "text": "charlie",
                "page_idx": 0,
                "block_id": "c",
            },
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual([item["text"] for item in result.items], [
            "alpha",
            "missing in content",
            "charlie",
        ])
        self.assertEqual(result.items[1]["_reconciliation"]["status"], "unmatched_middle")
        self.assertEqual(result.items[2]["middle_json"]["block_id"], "c")
        self.assertEqual(result.diagnostics["summary"]["unmatched_middle_count"], 1)
        self.assertEqual(result.diagnostics["summary"]["matched_count"], 2)

    def test_internal_extra_content_uses_independent_matches_as_anchors(self):
        middle = _middle(
            _page(0, _block("alpha"), _block("charlie"))
        )
        content = [
            {"type": "text", "text": "alpha", "page_idx": 0},
            {"type": "text", "text": "bravo", "page_idx": 0},
            {"type": "text", "text": "charlie", "page_idx": 0},
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual([item["text"] for item in result.items], [
            "alpha",
            "bravo",
            "charlie",
        ])
        marker = result.items[1]["_reconciliation"]
        self.assertEqual(marker["status"], "unmatched_content")
        self.assertEqual(marker["placement"], "between_matched_anchors")
        self.assertEqual(result.items[2]["middle_json"]["lines"][0]["spans"][0]["content"], "charlie")

    def test_reordered_content_is_restored_to_middle_reading_order(self):
        middle = _middle(
            _page(0, _block("alpha"), _block("bravo"), _block("charlie"))
        )
        content = [
            {"type": "text", "text": "charlie", "page_idx": 0},
            {"type": "text", "text": "alpha", "page_idx": 0},
            {"type": "text", "text": "bravo", "page_idx": 0},
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(
            [item["text"] for item in result.items], ["alpha", "bravo", "charlie"]
        )
        self.assertEqual(result.diagnostics["status"], "matched")
        self.assertEqual(
            result.diagnostics["summary"]["match_methods"], {"exact_text": 3}
        )

    def test_duplicate_text_is_disambiguated_by_geometry_not_position(self):
        middle = _middle(
            _page(
                0,
                _block("duplicate", bbox=[0, 0, 20, 10]),
                _block("duplicate", bbox=[0, 40, 20, 50]),
            )
        )
        content = [
            {
                "type": "text",
                "text": "duplicate",
                "page_idx": 0,
                "bbox": [0, 40, 20, 50],
                "observation": "bottom",
            },
            {
                "type": "text",
                "text": "duplicate",
                "page_idx": 0,
                "bbox": [0, 0, 20, 10],
                "observation": "top",
            },
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(
            [item["observation"] for item in result.items], ["top", "bottom"]
        )
        self.assertEqual(
            result.diagnostics["summary"]["match_methods"], {"exact_geometry": 2}
        )

    def test_identical_text_on_different_pages_never_crosses_page_boundary(self):
        middle = _middle(
            _page(0, _block("same", block_id="page-zero")),
            _page(1, _block("same", block_id="page-one")),
        )
        content = [
            {"type": "text", "text": "same", "page_idx": 1, "tag": "one"},
            {"type": "text", "text": "same", "page_idx": 0, "tag": "zero"},
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual([item["tag"] for item in result.items], ["zero", "one"])
        self.assertEqual([item["page_idx"] for item in result.items], [0, 1])
        self.assertEqual(result.items[0]["middle_json"]["block_id"], "page-zero")
        self.assertEqual(result.items[1]["middle_json"]["block_id"], "page-one")

    def test_unequal_duplicate_groups_are_retained_without_positional_shift(self):
        middle = _middle(
            _page(0, _block("before"), _block("same"), _block("same"), _block("after"))
        )
        content = [
            {"type": "text", "text": "before", "page_idx": 0},
            {"type": "text", "text": "same", "page_idx": 0},
            {"type": "text", "text": "after", "page_idx": 0},
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(result.diagnostics["summary"]["matched_count"], 2)
        self.assertEqual(result.diagnostics["summary"]["unmatched_middle_count"], 2)
        self.assertEqual(result.diagnostics["summary"]["unmatched_content_count"], 1)
        self.assertEqual(result.items[-1]["text"], "after")
        self.assertEqual(result.items[-1]["middle_json"]["lines"][0]["spans"][0]["content"], "after")

    def test_duplicate_labels_pair_ordinally_only_between_stable_anchors(self):
        middle = _middle(
            _page(
                0,
                _block("before"),
                _block("SIGNED"),
                _block("SIGNED"),
                _block("after"),
            )
        )
        content = [
            {"type": "text", "text": "before", "page_idx": 0},
            {"type": "text", "text": "SIGNED", "page_idx": 0, "tag": "first"},
            {"type": "text", "text": "SIGNED", "page_idx": 0, "tag": "second"},
            {"type": "text", "text": "after", "page_idx": 0},
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(result.diagnostics["status"], "matched")
        self.assertEqual([result.items[1]["tag"], result.items[2]["tag"]], [
            "first",
            "second",
        ])
        self.assertEqual(
            result.diagnostics["summary"]["match_methods"]["anchored_duplicate_text"],
            2,
        )

    def test_renderer_markup_and_soft_word_breaks_are_format_insensitive(self):
        middle = _middle(
            _page(
                0,
                _block("Impulse = F.Δt"),
                _block("On- prem value is ~20% at $2M"),
            )
        )
        content = [
            {"type": "text", "text": "Impulse $=$ F.Δt", "page_idx": 0},
            {
                "type": "text",
                "text": "Onprem value is \\~20% at \\$2M",
                "page_idx": 0,
            },
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(result.diagnostics["status"], "matched")
        self.assertEqual(
            result.diagnostics["summary"]["match_methods"],
            {"format_insensitive_text": 2},
        )

    def test_small_ocr_variation_requires_unique_mutual_similarity(self):
        middle = _middle(
            _page(0, _block("The reconciliation accuracy is 92.4 percent today."))
        )
        content = [
            {
                "type": "text",
                "text": "The reconciliation accuracy is 92.4 pereent today.",
                "page_idx": 0,
            }
        ]

        result = reconcile_middle_content(middle, content, ".")

        self.assertEqual(result.diagnostics["status"], "matched")
        self.assertEqual(
            result.diagnostics["summary"]["match_methods"],
            {"normalized_similarity": 1},
        )

    def test_strict_mismatch_raises_structured_diagnostics(self):
        middle = _middle(_page(0, _block("layout observation")))
        content = [{"type": "text", "text": "different content", "page_idx": 0}]

        with self.assertRaises(MinerUReconciliationError) as raised:
            reconcile_middle_content(
                middle, content, ".", mismatch_policy="error"
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["status"], "failed_mismatch")
        self.assertEqual(diagnostics["summary"]["unmatched_middle_count"], 1)
        self.assertEqual(diagnostics["summary"]["unmatched_content_count"], 1)
        self.assertIn("layout observation", diagnostics["unmatched_middle"][0]["text_preview"])
        json.dumps(diagnostics)

    def test_image_asset_identity_and_path_are_preserved(self):
        middle = _middle(
            _page(
                0,
                {
                    "type": "image",
                    "blocks": [
                        {
                            "lines": [
                                {"spans": [{"type": "image", "image_path": "crop.png"}]}
                            ]
                        }
                    ],
                },
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory, "images")
            image_dir.mkdir()
            Path(image_dir, "crop.png").touch()
            content = [
                {
                    "type": "image",
                    "page_idx": 0,
                    "img_path": "images/crop.png",
                    "image_caption": [],
                    "image_footnote": [],
                }
            ]

            result = reconcile_middle_content(middle, content, directory)

            self.assertEqual(
                result.diagnostics["summary"]["match_methods"], {"asset_identity": 1}
            )
            self.assertEqual(result.items[0]["img_path"], str(Path(image_dir, "crop.png")))


if __name__ == "__main__":
    unittest.main()
