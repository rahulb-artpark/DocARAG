"""An image reference must not be left for the client to guess about.

Six chart and diagram figures failed KG extraction inside the indexing run with
"illegal base64 data at input byte 3", returned by the server, identically on
all 24 attempts across six different images -- a constant payload rather than
any node's own bytes -- and never reproducible outside the run.

ollama-python decides what a ``str`` image is by trying ``Path(value).exists()``
inside a bare ``except Exception: pass``, then falling through to
``b64decode(value)``. That decode runs without ``validate=True``, so it discards
every character outside the base64 alphabet instead of refusing, and a value
that survives is posted to the server as though it were the image itself.

Handing the client a ``Path`` takes the first branch of that serializer with no
guess in between, and a file that cannot be read says so here, naming itself,
rather than becoming a 400 about base64.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePosixPath

from retrieval_engine.providers.vision_model import _as_image_payloads, _root_cause


class AnImageFileBecomesAPath(unittest.TestCase):
    def _jpeg(self, directory, name="crop.jpg"):
        path = Path(directory) / name
        path.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
        return path

    def test_an_existing_image_path_is_handed_over_as_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._jpeg(directory)
            self.assertEqual([path], _as_image_payloads([str(path)]))

    def test_every_image_suffix_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.png", "b.jpg", "c.jpeg", "d.webp"):
                with self.subTest(name=name):
                    path = self._jpeg(directory, name)
                    self.assertIsInstance(_as_image_payloads([str(path)])[0], Path)

    def test_an_uppercase_suffix_is_resolved_too(self):
        # ollama's own check is case-sensitive, so ".JPG" falls through its path
        # branch entirely and is decoded as base64.
        with tempfile.TemporaryDirectory() as directory:
            path = self._jpeg(directory, "shout.JPG")
            self.assertIsInstance(_as_image_payloads([str(path)])[0], Path)

    def test_a_missing_image_names_itself(self):
        with self.assertRaises(FileNotFoundError) as caught:
            _as_image_payloads(["/no/such/figure.jpg"])
        self.assertIn("/no/such/figure.jpg", str(caught.exception))

    def test_a_directory_is_not_an_image(self):
        with tempfile.TemporaryDirectory() as directory:
            impostor = Path(directory) / "looks-like.jpg"
            impostor.mkdir()
            with self.assertRaises(FileNotFoundError):
                _as_image_payloads([str(impostor)])


class OtherPayloadsArePassedThroughUntouched(unittest.TestCase):
    def test_bytes_are_left_alone(self):
        self.assertEqual([b"\xff\xd8"], _as_image_payloads([b"\xff\xd8"]))

    def test_an_existing_path_object_is_left_alone(self):
        given = PurePosixPath("/tmp/whatever.jpg")
        self.assertEqual([given], _as_image_payloads([given]))

    def test_a_base64_string_is_left_alone(self):
        # A caller that already encoded its own image must keep working.
        encoded = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcG"
        self.assertEqual([encoded], _as_image_payloads([encoded]))

    def test_no_images_is_an_empty_list(self):
        self.assertEqual([], _as_image_payloads(None))
        self.assertEqual([], _as_image_payloads([]))


class TheFailureSaysWhatBroke(unittest.TestCase):
    def test_the_root_cause_is_reached_past_a_wrapper(self):
        try:
            try:
                raise ValueError("illegal base64 data at input byte 3")
            except ValueError as error:
                raise RuntimeError("Could not get a JSON response") from error
        except RuntimeError as outer:
            self.assertEqual("illegal base64 data at input byte 3", _root_cause(outer))

    def test_an_unwrapped_error_is_its_own_cause(self):
        self.assertEqual("plain", _root_cause(ValueError("plain")))

    def test_a_message_less_error_still_names_its_type(self):
        self.assertEqual("KeyboardInterrupt", _root_cause(KeyboardInterrupt()))

    def test_a_cycle_cannot_hang_the_walk(self):
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first
        self.assertIn(_root_cause(first), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
