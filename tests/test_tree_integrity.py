from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

from docarag.tree_integrity import TreeIntegrityError, verify_tree_pair


_IDENTITY = {
    "file_name": "original.pdf",
    "source_uri": "file:///original.pdf",
    "source_sha256": "a" * 64,
    "snapshot_path": None,  # filled in per-test with the real snapshot path
}


class _TupleBboxTree:
    """Picklable stand-in whose node metadata carries MinerU-style tuple bboxes."""

    def __init__(self, snapshot_path: str) -> None:
        self.meta_info = {**_IDENTITY, "snapshot_path": snapshot_path}

    def to_json_summary(self) -> dict:
        return {
            "nodes": [
                {
                    "index_id": 0,
                    "parent_id": None,
                    "type": "root",
                    "meta_info": {},
                    "summary": "",
                },
                {
                    "index_id": 1,
                    "parent_id": 0,
                    "type": "title",
                    # bbox as a tuple, exactly as MinerU emits it.
                    "meta_info": {"pdf_para_block": {"bbox": (57, 49, 554, 93)}},
                    "summary": "A title",
                },
            ],
            "meta_info": self.meta_info,
        }


class RealBookRAGTreeIntegrityTests(unittest.TestCase):
    def test_real_document_tree_pickle_json_and_source_identity(self):
        try:
            from Core.Index.Tree import DocumentTree
        except ImportError as exc:
            self.skipTest(f"BookRAG runtime dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            source = root / "source.pdf"
            source.write_bytes(b"%PDF-1.4\nsource\n")
            source_uri = str(root / "original.pdf")
            source_sha = "a" * 64
            metadata = {
                "file_name": "original.pdf",
                "source_uri": source_uri,
                "source_sha256": source_sha,
                "snapshot_path": str(source),
            }
            tree = DocumentTree(
                meta_dict=metadata,
                cfg=SimpleNamespace(save_path=str(run)),
            )
            tree.save_to_file()

            verify_tree_pair(
                run,
                expected_source_sha256=source_sha,
                expected_source_uri=source_uri,
                expected_snapshot_path=source,
                expected_source_name="original.pdf",
            )

            payload = json.loads((run / "tree.json").read_text(encoding="utf-8"))
            payload["meta_info"]["source_sha256"] = "b" * 64
            (run / "tree.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                TreeIntegrityError, "different trees|source identity"
            ):
                verify_tree_pair(
                    run,
                    expected_source_sha256=source_sha,
                    expected_source_uri=source_uri,
                    expected_snapshot_path=source,
                    expected_source_name="original.pdf",
                )


class TupleListRoundTripTests(unittest.TestCase):
    def _write_pair(self, run: Path, snapshot: Path) -> None:
        tree = _TupleBboxTree(str(snapshot))
        (run / "tree.pkl").write_bytes(pickle.dumps(tree))
        # tree.json is JSON: the pickle's tuple bboxes serialize to arrays here.
        (run / "tree.json").write_text(
            json.dumps(tree.to_json_summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_tuple_bbox_in_pickle_matches_json_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            snapshot = Path(temporary) / "snap.pdf"
            snapshot.write_bytes(b"%PDF-1.4\n")
            self._write_pair(run, snapshot)
            # tree.pkl (tuples) and tree.json (arrays) describe one tree, so the
            # integrity check must accept the benign representation difference.
            verify_tree_pair(
                run,
                expected_source_sha256=_IDENTITY["source_sha256"],
                expected_source_uri=_IDENTITY["source_uri"],
                expected_snapshot_path=snapshot,
                expected_source_name=_IDENTITY["file_name"],
            )

    def test_real_content_divergence_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            snapshot = Path(temporary) / "snap.pdf"
            snapshot.write_bytes(b"%PDF-1.4\n")
            self._write_pair(run, snapshot)
            payload = json.loads((run / "tree.json").read_text(encoding="utf-8"))
            payload["nodes"][1]["summary"] = "tampered summary"
            (run / "tree.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TreeIntegrityError, "different trees"):
                verify_tree_pair(
                    run,
                    expected_source_sha256=_IDENTITY["source_sha256"],
                    expected_source_uri=_IDENTITY["source_uri"],
                    expected_snapshot_path=snapshot,
                    expected_source_name=_IDENTITY["file_name"],
                )


if __name__ == "__main__":
    unittest.main()
