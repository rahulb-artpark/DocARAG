"""Correcting what a node says its source shows is recorded, not substituted.

A node's summary is a gloss and nothing seals it, so an edit simply replaces it.
Its *content* is different: it is the observation the parser and vision model
produced, sealed into the canonical block's digest and re-projected on every
answer. An answer citing it claims it was not altered.

So a correction does not overwrite it. The node keeps its source references and
gains a transformation naming what happened -- ``docarag.human_edit.v1``, the
same shape ``bookrag.figure_merge.v1`` already uses -- and a digest over the new
text *and* that lineage. The answer then uses the corrected words while the
record still says which canonical block they were derived from, and who from.

Live on test.pdf node 8297, whose vision-model description called a Zernike
surface "a 3D rendered icon or graphic, not a chart, table, or diagram":
the correction sealed as ``derived:39622e0d…``, kept
``block_8369a6a9…``/``blockrev_bccd6e78…`` as its source, kept its crop, and
still hydrates into an answer.
"""

from __future__ import annotations

import unittest

from docarag.derived_identity import (
    derived_evidence_id,
    derived_identity_payload,
    derived_output_identity,
)
from docarag.edit import (
    HUMAN_EDIT_TRANSFORMATION,
    _DIRECT_EVIDENCE_FIELDS,
    _derived_evidence_for_edit,
)

BLOCK = "block_8369a6a9d3fcc9264d90a938846358ed"
REVISION = "blockrev_bccd6e78aa96eb998c21219d4aac3c63"
OBSERVATION = "obs_92c73885ff2ecce3d34d280cccf40cda"
OLD = "A 3D rendered icon or graphic, not a chart, table, or diagram."
NEW = "A 3D rendering of a Zernike surface: a saddle-like sheet with a central aperture."


def _direct_evidence(**extra):
    return {
        "type": "image",
        "content": OLD,
        "page": 577,
        "page_id": "page_083d5c7f59356d750be61da04ae94f5a",
        "document_version_id": "docv_8ca205f69f77f094f8e897ed50e51a24",
        "source_sha256": "4" * 64,
        "canonical_block_id": BLOCK,
        "canonical_revision_id": REVISION,
        "observation_id": OBSERVATION,
        "source_observation_ids": [OBSERVATION],
        "img_path": "/runs/vlm/images/figure.jpg",
        "source_evidence_refs": [
            {
                "canonical_block_id": BLOCK,
                "canonical_revision_id": REVISION,
                "observation_id": OBSERVATION,
                "source_observation_ids": [OBSERVATION],
                "document_version_id": "docv_8ca205f69f77f094f8e897ed50e51a24",
                "page_id": "page_083d5c7f59356d750be61da04ae94f5a",
                "page_idx": 577,
            }
        ],
        **extra,
    }


class _StubType:
    """The projected type of the primary block, without opening a store."""

    def __init__(self, value="image"):
        self.value = value

    def __enter__(self):
        import docarag.edit as edit

        self.original = edit._primary_block_type
        edit._primary_block_type = lambda _run, _evidence: self.value
        return self

    def __exit__(self, *_exc):
        import docarag.edit as edit

        edit._primary_block_type = self.original
        return False


def _edited(evidence=None, text=NEW, node_type="image"):
    with _StubType(node_type):
        return _derived_evidence_for_edit("/run", evidence or _direct_evidence(), text)


class TheCorrectionIsRecordedAsADerivation(unittest.TestCase):
    def test_the_new_text_is_what_the_node_carries(self):
        self.assertEqual(NEW, _edited()["content"])

    def test_it_names_the_transformation(self):
        self.assertEqual([HUMAN_EDIT_TRANSFORMATION], _edited()["transformations"])

    def test_the_source_block_is_still_cited(self):
        updated = _edited()

        self.assertEqual([BLOCK], updated["source_canonical_block_ids"])
        self.assertEqual([REVISION], updated["source_canonical_revision_ids"])
        self.assertEqual(1, len(updated["source_evidence_refs"]))

    def test_it_no_longer_claims_to_be_a_verbatim_reading(self):
        # Keeping canonical_block_id would assert this text came straight out of
        # the document, which is exactly what is no longer true.
        updated = _edited()

        for field in _DIRECT_EVIDENCE_FIELDS:
            self.assertNotIn(field, updated)

    def test_the_digest_covers_the_new_text(self):
        first = _edited(text="one wording")["derived_evidence_id"]
        second = _edited(text="another wording")["derived_evidence_id"]

        self.assertNotEqual(first, second)

    def test_the_digest_covers_the_lineage_too(self):
        other = _direct_evidence()
        other["source_evidence_refs"][0]["canonical_revision_id"] = "blockrev_" + "9" * 32

        self.assertNotEqual(
            _edited()["derived_evidence_id"],
            _edited(other)["derived_evidence_id"],
        )

    def test_the_digest_is_what_the_hydrator_will_recompute(self):
        # The hydrator rebuilds this from the node and the primary block's
        # projected type; if the two ever disagree the node is refused.
        updated = _edited()
        expected = derived_evidence_id(
            derived_identity_payload(
                output=derived_output_identity(content_type="image", text=NEW),
                source_revision_ids=[REVISION],
                source_document_version_ids=["docv_8ca205f69f77f094f8e897ed50e51a24"],
                source_block_ids=[BLOCK],
                source_observation_ids=[OBSERVATION],
                source_references=updated["source_evidence_refs"],
                transformations=[HUMAN_EDIT_TRANSFORMATION],
            )
        )

        self.assertEqual(expected, updated["derived_evidence_id"])

    def test_the_crop_survives_the_correction(self):
        self.assertEqual("/runs/vlm/images/figure.jpg", _edited()["img_path"])

    def test_editing_twice_keeps_one_transformation(self):
        once = _edited()
        twice = _edited(once, text="a third wording")

        self.assertEqual([HUMAN_EDIT_TRANSFORMATION], twice["transformations"])
        self.assertEqual([BLOCK], twice["source_canonical_block_ids"])

    def test_an_earlier_transformation_is_not_dropped(self):
        merged = _direct_evidence(transformations=["bookrag.figure_merge.v1"])

        self.assertEqual(
            ["bookrag.figure_merge.v1", HUMAN_EDIT_TRANSFORMATION],
            _edited(merged)["transformations"],
        )

    def test_a_node_with_no_sources_is_refused(self):
        naked = _direct_evidence()
        naked["source_evidence_refs"] = []

        with self.assertRaises(SystemExit):
            _edited(naked)


class TheEndpointSaysWhatItDid(unittest.TestCase):
    @staticmethod
    def _source():
        from pathlib import Path

        return Path("Scripts/web.py").read_text(encoding="utf-8")

    def test_it_reports_the_evidence_became_derived(self):
        source = self._source()

        self.assertIn('"evidence": "derived"', source)
        self.assertIn(HUMAN_EDIT_TRANSFORMATION, source)

    def test_it_releases_the_index_lock_first(self):
        # Same deadlock the summary edit hit: the server's own agent holds a
        # shared lease on the lock the reconcile needs exclusively.
        source = self._source()
        content_handler = source.split("def edit_content(self):", 1)[1].split("def edit_entity", 1)[0]

        self.assertIn("release_index_locks()", content_handler)
        self.assertLess(
            content_handler.index("release_index_locks()"),
            content_handler.index("edit_node_content("),
        )


if __name__ == "__main__":
    unittest.main()
