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

Refusing the URI was necessary but not sufficient. ``crop_materialization``
publishes every visual crop into the run's content-addressed store and rewrites
the block to carry the CAS URI, so for a materialized chart the URI is often the
*only* image reference the tree holds -- on a real 267-node document, six such
nodes had nothing else. Refusing it left those figures with no image at all,
which fails just as quietly. The URI has to be resolved to the file the store
published.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from document_services.persistence.vector_tree_source import asset_cas_root, openable_image


class IdentityNeverDisplacesAPath(unittest.TestCase):
    def test_a_cas_uri_does_not_shadow_the_materialized_path(self):
        self.assertEqual(
            "/runs/rev/vlm/images/figure.jpg",
            openable_image(
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
                self.assertEqual("", openable_image(uri, ""))

    def test_evidence_still_wins_when_it_is_a_real_path(self):
        # The store remains the authority when it has something openable.
        self.assertEqual(
            "/from/evidence.jpg",
            openable_image("/from/evidence.jpg", "/from/meta.jpg"),
        )

    def test_the_node_path_is_used_when_evidence_has_none(self):
        self.assertEqual("/from/meta.jpg", openable_image(None, "/from/meta.jpg"))
        self.assertEqual("/from/meta.jpg", openable_image("", "/from/meta.jpg"))

    def test_a_node_with_no_image_at_all_is_empty_not_none(self):
        # Callers test truthiness and pass this straight to os.path.exists.
        self.assertEqual("", openable_image(None, None))
        self.assertEqual("", openable_image(None, ""))

    def test_a_non_string_is_never_returned(self):
        self.assertEqual("/real.jpg", openable_image(object(), "/real.jpg"))


class AMaterializedCropIsResolvedNotRefused(unittest.TestCase):
    """The six nodes whose only reference was a URI must still get an image."""

    def _run_with_crop(self, directory, digest):
        published = Path(directory) / "asset_cas" / digest[:2]
        published.mkdir(parents=True)
        (published / digest).write_bytes(b"\xff\xd8\xff\xe0jpeg")
        return published / digest

    def test_a_cas_uri_resolves_to_the_published_file(self):
        digest = "e4" + "c" * 62
        with tempfile.TemporaryDirectory() as directory:
            expected = self._run_with_crop(directory, digest)
            root = asset_cas_root(directory)
            self.assertEqual(
                str(expected),
                openable_image(f"cas://sha256/{digest}", None, cas_root=root),
            )

    def test_a_local_path_still_wins_over_the_store(self):
        digest = "e4" + "c" * 62
        with tempfile.TemporaryDirectory() as directory:
            self._run_with_crop(directory, digest)
            root = asset_cas_root(directory)
            self.assertEqual(
                "/from/meta.jpg",
                openable_image(
                    f"cas://sha256/{digest}", "/from/meta.jpg", cas_root=root
                ),
            )

    def test_a_uri_the_store_does_not_hold_is_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_with_crop(directory, "e4" + "c" * 62)
            root = asset_cas_root(directory)
            self.assertEqual(
                "",
                openable_image("cas://sha256/" + "d" * 64, None, cas_root=root),
            )

    def test_only_cas_sha256_is_resolved(self):
        digest = "e4" + "c" * 62
        with tempfile.TemporaryDirectory() as directory:
            self._run_with_crop(directory, digest)
            root = asset_cas_root(directory)
            for uri in (
                f"https://example.invalid/{digest}",
                f"cas://md5/{digest}",
                f"file:///{digest}",
            ):
                with self.subTest(uri=uri):
                    self.assertEqual("", openable_image(uri, None, cas_root=root))

    def test_a_digest_that_is_not_a_digest_is_refused(self):
        # The digest becomes a path segment, so anything else is a traversal.
        with tempfile.TemporaryDirectory() as directory:
            root = asset_cas_root(directory) or Path(directory)
            for digest in ("../../etc/passwd", "e4", "g" * 64, ""):
                with self.subTest(digest=digest):
                    self.assertEqual(
                        "", openable_image(f"cas://sha256/{digest}", None, cas_root=root)
                    )

    def test_a_run_without_a_store_resolves_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(asset_cas_root(directory))
            self.assertEqual(
                "", openable_image("cas://sha256/" + "e" * 64, None, cas_root=None)
            )


if __name__ == "__main__":
    unittest.main()
