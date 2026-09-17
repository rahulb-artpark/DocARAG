"""Deterministic golden-document evaluation contracts."""

from document_services.evaluation.golden import (
    GOLDEN_MANIFEST_SCHEMA_VERSION,
    GOLDEN_PREDICTION_SCHEMA_VERSION,
    evaluate_golden_run,
    load_golden_manifest,
)

__all__ = [
    "GOLDEN_MANIFEST_SCHEMA_VERSION",
    "GOLDEN_PREDICTION_SCHEMA_VERSION",
    "evaluate_golden_run",
    "load_golden_manifest",
]
