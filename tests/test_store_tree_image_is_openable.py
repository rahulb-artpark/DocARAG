"""A node's image must reach a provider as something it can open.

Every chart and diagram in a document lost its graph entities to this. The
graph and vector stages read nodes through StoreTreeSource, whose meta took
``evidence["img_path"]`` in preference to the node's own ``meta.img_path``.
Evidence carries an image as its canonical identity -- ``cas://sha256/<digest>``
-- while the materialized local path lives on the node. So a chart arrived at
KG extraction holding a URI where a filename belonged, and the VLM client
posted the URI itself as though it were image bytes; Ollama answered "illegal
base64 data at input byte 3", the 3 being the colon in "cas:".

Plain figures were unaffected because their evidence carried no URI, which is
why the same images extracted perfectly while they were typed "image" and
stopped the moment they were classified as chart or diagram.
"""

from __future__ import annotations

import unittest

from docarag.vdb_tree_source import _openable_image


class IdentityNeverDisplacesAPath(unittest.TestCase):
    def test_a_cas_uri_does_not_shadow_the_materialized_path(self):
        self.assertEqual(
            "/runs/rev/vlm/images/figure.jpg",
            _openable_image(
                "cas://sha256/" + "e4" * 32, "/runs/rev/vlm/images/figure.jpg"
            ),
        )

    def test_no_scheme_survives_as_an_image_reference(self):
        for uri in (
            "cas://sha256/" + "ab" * 32,
            "file:///runs/rev/figure.jpg",
            "https://example.invalid/figure.jpg",
            "s3://bucket/figure.jpg",
        ):
            with self.subTest(uri=uri):
                self.assertEqual("", _openable_image(uri, ""))

    def test_evidence_still_wins_when_it_is_a_real_path(self):
        # The store remains the authority when it has something openable.
        self.assertEqual(
            "/from/evidence.jpg",
            _openable_image("/from/evidence.jpg", "/from/meta.jpg"),
        )

    def test_the_node_path_is_used_when_evidence_has_none(self):
        self.assertEqual("/from/meta.jpg", _openable_image(None, "/from/meta.jpg"))
        self.assertEqual("/from/meta.jpg", _openable_image("", "/from/meta.jpg"))

    def test_a_node_with_no_image_at_all_is_empty_not_none(self):
        # Callers test truthiness and pass this straight to os.path.exists.
        self.assertEqual("", _openable_image(None, None))
        self.assertEqual("", _openable_image(None, ""))

    def test_a_non_string_is_never_returned(self):
        self.assertEqual("/real.jpg", _openable_image(object(), "/real.jpg"))


if __name__ == "__main__":
    unittest.main()
