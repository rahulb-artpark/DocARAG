"""NUL-byte stripping in evidence_projection's artifact read boundary.

Real incident (test.pdf, 1870 pages): MinerU's text extraction emits a
literal NUL byte as its fallback for a handful of glyphs a source PDF's
font has no usable Unicode mapping for (Greek letters and a degree sign, in
Zernike-polynomial notation, in this document). PostgreSQL's text and jsonb
types both reject an embedded NUL outright -- psycopg raises before the
value even reaches the server -- so persistence failed after a 13+ hour
tree build, on the very last step, having never actually reached the
database. _strip_nul_bytes runs once on every artifact evidence_projection
reads, so no downstream field access has to remember the constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docarag.evidence_projection import _strip_nul_bytes, build_document_evidence

NUL = chr(0)


class StripNulBytesTests(unittest.TestCase):
    def test_removes_nul_from_a_plain_string(self):
        self.assertEqual("rho", _strip_nul_bytes("r" + NUL + "ho"))

    def test_string_without_nul_is_returned_unchanged(self):
        text = "no null bytes here"
        self.assertIs(text, _strip_nul_bytes(text))

    def test_strips_recursively_through_nested_dicts_and_lists(self):
        payload = {
            "a": "has" + NUL + "nul",
            "b": [{"c": "also" + NUL + "here"}, "clean", ["deep" + NUL + "er"]],
            "d": {"e": {"f": NUL + NUL + NUL}},
        }
        cleaned = _strip_nul_bytes(payload)
        self.assertEqual(
            {
                "a": "hasnul",
                "b": [{"c": "alsohere"}, "clean", ["deeper"]],
                "d": {"e": {"f": ""}},
            },
            cleaned,
        )

    def test_dict_keys_are_left_untouched(self):
        # Keys are fixed field names from the artifact schema, never derived
        # from PDF content -- only values need sanitizing.
        cleaned = _strip_nul_bytes({"block_type": "para" + NUL + "graph"})
        self.assertEqual({"block_type": "paragraph"}, cleaned)

    def test_non_string_scalars_pass_through_unchanged(self):
        self.assertEqual(3, _strip_nul_bytes(3))
        self.assertEqual(1.5, _strip_nul_bytes(1.5))
        self.assertIsNone(_strip_nul_bytes(None))
        self.assertEqual(True, _strip_nul_bytes(True))


class BuildDocumentEvidenceStripsNulTests(unittest.TestCase):
    def test_a_nul_in_block_text_never_reaches_the_evidence_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            canonical = {
                "document": {"document_id": "doc-1"},
                "canonical_blocks": [
                    {
                        "block_id": "block-1",
                        "block_type": "paragraph",
                        "representations": [
                            {
                                "kind": "text",
                                "content": (
                                    "the normalized radial coordinate is "
                                    + NUL
                                    + " and the angular one is "
                                    + NUL
                                ),
                            }
                        ],
                    }
                ],
            }
            (run / "canonical_blocks.json").write_text(
                json.dumps(canonical), encoding="utf-8"
            )
            (run / "tree_structure.json").write_text("[]", encoding="utf-8")

            evidence = build_document_evidence(
                run, "revision-1", parts=frozenset({"tree"})
            )

            self.assertEqual(1, len(evidence.blocks))
            representation = evidence.blocks[0].representations[0]
            self.assertNotIn(NUL, representation.text)
            self.assertEqual(
                "the normalized radial coordinate is  and the angular one is ",
                representation.text,
            )
            # The raw per-block snapshot kept for canonical/debug purposes
            # must be clean too -- it round-trips through json.dumps into a
            # jsonb column downstream, and jsonb rejects an escaped NUL
            # sequence just as surely as psycopg rejects a raw NUL byte.
            self.assertNotIn(
                NUL,
                json.dumps(evidence.blocks[0].canonical),
            )


if __name__ == "__main__":
    unittest.main()
