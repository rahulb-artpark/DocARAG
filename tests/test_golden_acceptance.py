from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from document_services.evaluation.golden import (
    GOLDEN_MANIFEST_SCHEMA_VERSION,
    GOLDEN_PREDICTION_SCHEMA_VERSION,
    evaluate_golden_run,
)


class GoldenAcceptanceTests(unittest.TestCase):
    def test_perfect_multimodal_prediction_passes_all_declared_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pdf"
            source.write_bytes(b"%PDF-golden-fixture")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            page = {
                "page_no": 1,
                "text": "Revenue was 42 USD.",
                "reading_order": ["title", "table", "formula", "figure"],
                "blocks": [
                    {"type": "title", "bbox": [0, 0, 100, 10]},
                    {"type": "table", "bbox": [0, 20, 100, 60]},
                ],
                "table_cells": [
                    {"table_id": "t1", "row": 0, "column": 0, "text": "Revenue"},
                    {"table_id": "t1", "row": 0, "column": 1, "text": "42"},
                ],
                "formulas": [{"value": "x = 2"}],
                "captions": [{"figure_id": "f1", "text": "System flow"}],
                "provenance_required": True,
            }
            manifest = {
                "schema_version": GOLDEN_MANIFEST_SCHEMA_VERSION,
                "protocol_revision": "test-v1",
                "required_metrics": [
                    "q_text", "q_order", "q_block", "q_table", "q_formula",
                    "q_caption", "provenance_validity", "citation_precision",
                    "citation_recall", "answer_f1", "hydrated_structured_evidence_rate",
                    "seconds_per_page", "gpu_peak_mib",
                ],
                "thresholds": {"q_text": 1, "q_table": 1, "gpu_peak_mib": 42000},
                "documents": [{
                    "document_id": "fixture",
                    "source_path": str(source),
                    "source_sha256": digest,
                    "pages": [page],
                    "questions": [{
                        "question_id": "q1", "answer": "42 USD", "citations": ["fixture:p1:t1"],
                        "requires_structured_evidence": True,
                    }],
                }],
            }
            predictions = {
                "schema_version": GOLDEN_PREDICTION_SCHEMA_VERSION,
                "systems": [{
                    "system_id": "router",
                    "system_release": "router@1.0.0",
                    "documents": [{
                        "document_id": "fixture", "pages": [{**page, "provenance_valid": True}],
                        "answers": [{
                            "question_id": "q1", "answer": "42 USD", "citations": ["fixture:p1:t1"],
                            "structured_asset_hydrated": True,
                        }],
                    }],
                    "runtime": {"seconds_per_page": 1.2, "gpu_peak_mib": 1000},
                    "crash_recovery": {
                        "resumed_from_checkpoint": True,
                        "no_committed_data_loss": True,
                        "no_false_success": True,
                    },
                }],
            }
            manifest_path = root / "manifest.json"
            prediction_path = root / "prediction.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            report = evaluate_golden_run(manifest_path, prediction_path)
        self.assertTrue(report["passed"])
        self.assertEqual(1.0, report["systems"][0]["metrics"]["q_table"])

    def test_missing_document_and_crash_receipts_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.write_bytes(b"pdf")
            manifest = {
                "schema_version": GOLDEN_MANIFEST_SCHEMA_VERSION,
                "protocol_revision": "test-v1",
                "required_metrics": ["q_text"],
                "documents": [{
                    "document_id": "missing",
                    "source_path": str(source),
                    "source_sha256": hashlib.sha256(b"pdf").hexdigest(),
                    "pages": [{"page_no": 1, "text": "required"}],
                }],
            }
            predictions = {
                "schema_version": GOLDEN_PREDICTION_SCHEMA_VERSION,
                "systems": [{"system_id": "empty", "documents": [], "runtime": {}}],
            }
            m = root / "m.json"; p = root / "p.json"
            m.write_text(json.dumps(manifest), encoding="utf-8")
            p.write_text(json.dumps(predictions), encoding="utf-8")
            report = evaluate_golden_run(m, p)
        self.assertFalse(report["passed"])
        system = report["systems"][0]
        self.assertEqual(["missing"], system["missing_documents"])
        self.assertFalse(system["crash_recovery_passed"])


if __name__ == "__main__":
    unittest.main()
