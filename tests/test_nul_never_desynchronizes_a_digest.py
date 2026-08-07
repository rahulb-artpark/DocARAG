"""A digest must describe the text the store actually holds.

MinerU emits a literal NUL as its fallback for a glyph a source PDF's font has
no usable Unicode mapping for -- Greek letters and a degree sign, in the
Zernike-polynomial notation of a 1871-page optics manual. PostgreSQL's text and
jsonb types both reject an embedded NUL, so something has to remove it before
the document can be stored.

It used to be removed on the way *into* Postgres, after the contract had
already hashed the text containing it. Six blocks were then stored carrying a
content_sha256 for text that no longer existed, and because every read goes
through CanonicalBlock.from_dict, the first question asked of that document
failed with "Representation content_sha256 mismatch" -- a whole index lost to
a character that was never storable in the first place.

These tests hold the removal in front of the hash.
"""

from __future__ import annotations

import unittest

from docarag.adapters.mineru_compat import _joined_text
from docarag.domain.base import content_sha256, freeze_json
from docarag.domain.evidence import Representation, RepresentationKind

NUL = "\x00"
ZERNIKE = f"in lens units, {NUL} is the normalized radial ray coordinate."


def _representation(content: str) -> Representation:
    return Representation(
        kind=RepresentationKind.TEXT,
        content=content,
        media_type="text/plain",
        language="en",
        confidence=1.0,
    )


class ADigestDescribesWhatIsStored(unittest.TestCase):
    def test_a_representation_round_trips_through_its_own_serialization(self):
        # The exact failure: to_dict() then from_dict() must not raise.
        original = _representation(ZERNIKE)
        restored = Representation.from_dict(original.to_dict())
        self.assertEqual(original.content_sha256, restored.content_sha256)
        self.assertEqual(original.representation_id, restored.representation_id)

    def test_the_stored_content_carries_no_nul(self):
        stored = _representation(ZERNIKE)
        self.assertNotIn(NUL, stored.content)
        self.assertEqual("in lens units,  is the normalized radial ray coordinate.", stored.content)

    def test_the_digest_is_taken_over_the_stripped_text(self):
        with_nul = _representation(ZERNIKE)
        without_nul = _representation(ZERNIKE.replace(NUL, ""))
        self.assertEqual(without_nul.content_sha256, with_nul.content_sha256)
        self.assertEqual(without_nul.representation_id, with_nul.representation_id)

    def test_a_later_strip_can_no_longer_change_the_content(self):
        # What evidence_projection._strip_nul_bytes does to a parsed artifact.
        payload = _representation(ZERNIKE).to_dict()
        stripped = {
            key: value.replace(NUL, "") if isinstance(value, str) else value
            for key, value in payload.items()
        }
        self.assertEqual(payload, stripped)


class NulIsGoneBeforeAnythingHashes(unittest.TestCase):
    def test_freeze_json_drops_it_from_a_bare_string(self):
        self.assertEqual("ab", freeze_json(f"a{NUL}b"))

    def test_freeze_json_drops_it_from_nested_metadata(self):
        frozen = freeze_json({"note": [f"x{NUL}y", {"deep": f"{NUL}z"}]})
        self.assertEqual("xy", frozen["note"][0])
        self.assertEqual("z", frozen["note"][1]["deep"])

    def test_a_hash_is_blind_to_a_nul(self):
        self.assertEqual(content_sha256({"t": f"a{NUL}b"}), content_sha256({"t": "ab"}))

    def test_ordinary_text_is_returned_unchanged(self):
        for value in ("plain", "ρ is the coordinate", "5°", "", "a\nb"):
            with self.subTest(value=value):
                self.assertEqual(value, freeze_json(value))

    def test_freezing_still_rejects_what_it_always_rejected(self):
        from docarag.domain.base import ContractValidationError

        with self.assertRaises(ContractValidationError):
            freeze_json({1: "non-string key"})
        with self.assertRaises(ContractValidationError):
            freeze_json(object())

    def test_booleans_are_not_mistaken_for_text(self):
        self.assertIs(True, freeze_json(True))
        self.assertIs(None, freeze_json(None))
        self.assertEqual(7, freeze_json(7))


class AGlyphOnlyBlockDoesNotFailTheDocument(unittest.TestCase):
    """Stripping must not turn a block's text into an empty representation."""

    def test_text_that_is_only_an_unmappable_glyph_counts_as_no_text(self):
        self.assertIsNone(_joined_text(NUL))
        self.assertIsNone(_joined_text(f" {NUL} "))

    def test_a_list_drops_only_the_glyph_only_items(self):
        self.assertEqual("real", _joined_text([NUL, "real"]))

    def test_text_around_a_glyph_is_still_text(self):
        self.assertEqual(ZERNIKE, _joined_text(ZERNIKE))


if __name__ == "__main__":
    unittest.main()
