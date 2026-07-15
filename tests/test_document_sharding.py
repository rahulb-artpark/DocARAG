from __future__ import annotations

from pathlib import Path
import unittest

from docarag.sharding import document_shard_number


class DocumentShardingTests(unittest.TestCase):
    def test_assignment_is_deterministic_bounded_and_one_indexed(self):
        assignments = [
            document_shard_number("document-17", "/data/report.pdf", 7)
            for _ in range(10)
        ]

        self.assertEqual([assignments[0]] * 10, assignments)
        self.assertGreaterEqual(assignments[0], 1)
        self.assertLessEqual(assignments[0], 7)
        self.assertEqual(
            document_shard_number("document-17", Path("/data/report.pdf"), 7),
            assignments[0],
        )

    def test_assignment_does_not_depend_on_corpus_order_or_size(self):
        documents = [
            (f"document-{index}", f"/data/{index}.pdf") for index in range(100)
        ]
        first = {
            key: document_shard_number(*key, 8)
            for key in documents
        }
        reordered_and_extended = [
            ("new-before", "/data/new-before.pdf"),
            *reversed(documents),
            ("new-after", "/data/new-after.pdf"),
        ]
        second = {
            key: document_shard_number(*key, 8)
            for key in reordered_and_extended
        }

        self.assertEqual(first, {key: second[key] for key in documents})
        self.assertGreater(len(set(first.values())), 1)

    def test_length_prefix_keeps_component_boundaries_unambiguous(self):
        self.assertNotEqual(
            document_shard_number("ab", "c", 2**31 - 1),
            document_shard_number("a", "bc", 2**31 - 1),
        )

    def test_invalid_shard_count_or_identity_fails_closed(self):
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(nsplit=invalid), self.assertRaisesRegex(
                ValueError, "nsplit"
            ):
                document_shard_number("document", "/data/report.pdf", invalid)

        for doc_uuid, doc_path in ((None, "/data/report.pdf"), ("document", None), ("", "x"), ("x", "")):
            with self.subTest(doc_uuid=doc_uuid, doc_path=doc_path), self.assertRaisesRegex(
                ValueError, "identity"
            ):
                document_shard_number(doc_uuid, doc_path, 2)

    def test_single_shard_always_selects_worker_one(self):
        for index in range(20):
            self.assertEqual(
                1,
                document_shard_number(index, f"/data/{index}.pdf", 1),
            )


if __name__ == "__main__":
    unittest.main()
