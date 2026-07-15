from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from Core.utils.summary_checkpoint import (
    SummaryCheckpointStore,
    summary_cache_identity,
)


class SummaryCheckpointTests(unittest.TestCase):
    def test_identity_binds_prompt_model_and_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "crop.png"
            image.write_bytes(b"first pixels")
            moved_copy = Path(directory) / "same-crop.png"
            moved_copy.write_bytes(b"first pixels")
            first = summary_cache_identity(
                kind="visual-node",
                prompt="summarize",
                model={"backend": "fake", "model": "v1"},
                image_paths=[image],
            )
            same_bytes = summary_cache_identity(
                kind="visual-node",
                prompt="summarize",
                model={"backend": "fake", "model": "v1"},
                image_paths=[moved_copy],
            )
            image.write_bytes(b"other pixels")
            second = summary_cache_identity(
                kind="visual-node",
                prompt="summarize",
                model={"backend": "fake", "model": "v1"},
                image_paths=[image],
            )
            different_model = summary_cache_identity(
                kind="visual-node",
                prompt="summarize",
                model={"backend": "fake", "model": "v2"},
                image_paths=[image],
            )

            self.assertEqual(first[0], same_bytes[0])
            self.assertNotEqual(first[0], second[0])
            self.assertNotEqual(second[0], different_model[0])

    def test_checkpoint_is_idempotent_first_writer_wins_and_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "summaries.sqlite3"
            key, prompt_hash, images = summary_cache_identity(
                kind="text-node",
                prompt="one prompt",
                model={"backend": "fake", "model": "v1"},
            )
            with SummaryCheckpointStore(database) as store:
                first = store.put(
                    cache_key=key,
                    kind="text-node",
                    prompt_sha256=prompt_hash,
                    model={"backend": "fake", "model": "v1"},
                    image_identities=images,
                    summary="first valid result",
                )
                second = store.put(
                    cache_key=key,
                    kind="text-node",
                    prompt_sha256=prompt_hash,
                    model={"backend": "fake", "model": "v1"},
                    image_identities=images,
                    summary="different retry result",
                )
                self.assertEqual("first valid result", first)
                self.assertEqual(first, second)
                self.assertEqual(1, store.count())

            self.assertEqual(0o600, database.stat().st_mode & 0o777)
            with SummaryCheckpointStore(database) as reopened:
                self.assertEqual("first valid result", reopened.get(key))

    def test_empty_or_post_close_writes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "summaries.sqlite3"
            key, prompt_hash, images = summary_cache_identity(
                kind="text-node",
                prompt="one prompt",
                model={"backend": "fake", "model": "v1"},
            )
            store = SummaryCheckpointStore(database)
            with self.assertRaisesRegex(ValueError, "empty summary"):
                store.put(
                    cache_key=key,
                    kind="text-node",
                    prompt_sha256=prompt_hash,
                    model={"backend": "fake", "model": "v1"},
                    image_identities=images,
                    summary="  ",
                )
            store.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                store.get(key)


if __name__ == "__main__":
    unittest.main()
