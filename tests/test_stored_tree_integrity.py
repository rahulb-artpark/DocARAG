"""The integrity gate must check the tree that is read, not the one beside it.

A run keeps ``tree.pkl`` and ``tree.json`` as build artifacts while the graph
and vector stages read the store. Comparing the two artifacts to each other
therefore proved nothing about the tree that runs, and on a real document it
proved nothing loudly: every chart and diagram reached the graph stage holding
``cas://sha256/<digest>`` in the field a filename belongs in, the VLM was
handed the URI as though it were image bytes, and all of them finished with no
entities at all. Both artifacts held a clean local path throughout, so the old
gate passed.

These tests hold the gate on the store.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_services.evidence import tree_integrity
from document_services.persistence.document_store import DocumentStore, DocumentStoreError
from document_services.evidence.tree_integrity import TreeIntegrityError, verify_stored_tree
from document_services.persistence.tree_store import NodeMeta, TreeNodeView


def _view(index_id, node_type, parent_id, *, img_path=None, evidence_img=None):
    return TreeNodeView(
        index_id=index_id,
        type=node_type,
        depth=0 if parent_id is None else 1,
        meta_info=NodeMeta(img_path=img_path),
        evidence={"img_path": evidence_img} if evidence_img is not None else {},
        parent_id=parent_id,
    )


class _Tree:
    def __init__(self, views):
        self._views = views

    def nodes(self, node_ids=None):
        return list(self._views)


class _Store:
    def __init__(self, views):
        self.tree = _Tree(views)
        self.closed = False

    def close(self):
        self.closed = True


def _verify(views, run_dir):
    store = _Store(views)
    with patch.object(DocumentStore, "open", classmethod(lambda cls, _run: store)):
        verify_stored_tree(Path(run_dir))
    return store


class TheGateReadsTheStore(unittest.TestCase):
    def test_a_sound_stored_tree_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "figure.jpg"
            image.write_bytes(b"jpeg")
            store = _verify(
                [
                    _view(0, "root", None),
                    _view(1, "text", 0),
                    _view(2, "chart", 0, img_path=str(image)),
                ],
                directory,
            )
        self.assertTrue(store.closed, "the gate must not leak its connection")

    def test_an_unpersisted_tree_is_not_silently_accepted(self):
        # The tree stage succeeding does not mean its output reached the store.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TreeIntegrityError, "no tree"):
                _verify([], directory)

    def test_a_store_that_cannot_be_opened_is_an_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:

            def refuse(cls, _run):
                raise DocumentStoreError("run-manifest.json is missing")

            with patch.object(DocumentStore, "open", classmethod(refuse)):
                with self.assertRaisesRegex(TreeIntegrityError, "cannot be opened"):
                    verify_stored_tree(Path(directory))

    def test_broken_topology_in_the_rows_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TreeIntegrityError):
                _verify(
                    [_view(0, "root", None), _view(2, "text", 0)],
                    directory,
                )


class AVisualNodeMustCarryAnImageAStageCanOpen(unittest.TestCase):
    """The check the old artifact-pair gate could not perform."""

    def test_an_identifier_where_a_filename_belongs_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TreeIntegrityError) as raised:
                _verify(
                    [
                        _view(0, "root", None),
                        _view(
                            1,
                            "chart",
                            0,
                            evidence_img="cas://sha256/" + "a" * 64,
                        ),
                    ],
                    directory,
                )
        message = str(raised.exception)
        self.assertIn("node 1", message)
        self.assertIn("cas://", message)

    def test_a_figure_the_parser_never_cropped_is_reported_not_blocked(self):
        # A real 500-page book has one captioned figure MinerU produced no crop
        # for. Failing its whole index over that trades a silent loss for a
        # blocked run, which is worse; the loss is stated instead.
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stderr", new_callable=io.StringIO) as captured:
                _verify(
                    [_view(0, "root", None), _view(1, "diagram", 0)],
                    directory,
                )
        note = captured.getvalue()
        self.assertIn("node 1", note)
        self.assertIn("caption", note)

    def test_a_broken_reference_still_fails_beside_an_uncropped_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(TreeIntegrityError) as raised:
                    _verify(
                        [
                            _view(0, "root", None),
                            _view(1, "diagram", 0),
                            _view(2, "chart", 0, evidence_img="cas://sha256/" + "f" * 64),
                        ],
                        directory,
                    )
        self.assertIn("node 2", str(raised.exception))
        self.assertNotIn("node 1", str(raised.exception))

    def test_an_image_the_run_no_longer_has_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TreeIntegrityError, "missing image"):
                _verify(
                    [
                        _view(0, "root", None),
                        _view(1, "image", 0, img_path=str(Path(directory) / "gone.jpg")),
                    ],
                    directory,
                )

    def test_a_run_relative_image_is_resolved_against_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "auto" / "images"
            crop.mkdir(parents=True)
            (crop / "panel.jpg").write_bytes(b"jpeg")
            _verify(
                [
                    _view(0, "root", None),
                    _view(1, "image", 0, img_path="auto/images/panel.jpg"),
                ],
                directory,
            )

    def test_a_local_path_beside_the_identifier_is_accepted(self):
        # Evidence names the canonical identity and meta names the crop; the
        # node is fine, and the gate must not read the URI as the image.
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "panel.jpg"
            image.write_bytes(b"jpeg")
            _verify(
                [
                    _view(0, "root", None),
                    _view(
                        1,
                        "chart",
                        0,
                        img_path=str(image),
                        evidence_img="cas://sha256/" + "b" * 64,
                    ),
                ],
                directory,
            )

    def test_text_and_table_nodes_are_not_required_to_have_images(self):
        with tempfile.TemporaryDirectory() as directory:
            _verify(
                [
                    _view(0, "root", None),
                    _view(1, "text", 0),
                    _view(2, "table", 0),
                    _view(3, "equation", 0),
                ],
                directory,
            )

    def test_every_broken_node_is_named_not_just_the_first(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TreeIntegrityError) as raised:
                _verify(
                    [
                        _view(0, "root", None),
                        _view(1, "chart", 0, evidence_img="cas://sha256/" + "a" * 64),
                        _view(2, "diagram", 0, img_path="/gone/panel.jpg"),
                    ],
                    directory,
                )
        message = str(raised.exception)
        self.assertIn("node 1", message)
        self.assertIn("node 2", message)


class TheCommandLineSelectsWhatIsChecked(unittest.TestCase):
    def test_store_only_never_touches_the_artifacts(self):
        # An empty run directory has no tree.pkl; --check store must still work,
        # because the artifacts are not what it verifies.
        with tempfile.TemporaryDirectory() as directory:
            store = _Store([_view(0, "root", None), _view(1, "text", 0)])
            with patch.object(DocumentStore, "open", classmethod(lambda cls, _r: store)):
                exit_code = tree_integrity._main(["--run", directory, "--check", "store"])
        self.assertEqual(0, exit_code)

    def test_a_failing_store_check_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _Store([])
            with patch.object(DocumentStore, "open", classmethod(lambda cls, _r: store)):
                exit_code = tree_integrity._main(["--run", directory, "--check", "store"])
        self.assertEqual(1, exit_code)

    def test_artifacts_remain_the_default_so_the_early_gate_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            # No artifacts and no manifest: the default path must still be the
            # artifact check, which fails here rather than reaching the store.
            with patch.object(
                DocumentStore, "open", classmethod(lambda cls, _r: _Store([]))
            ):
                exit_code = tree_integrity._main(["--run", directory])
        self.assertEqual(1, exit_code)


if __name__ == "__main__":
    unittest.main()
