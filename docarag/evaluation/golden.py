"""Executable golden-document parser and RAG acceptance metrics.

The evaluator consumes versioned, source-hash-bound annotations and a common
prediction envelope.  It performs no parsing and invokes no model, so the same
frozen annotations can compare Docling, MinerU, PaddleOCR-VL, or the router.
Missing applicable annotations and missing predictions remain visible; they are
never removed from a release denominator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from docarag.domain.base import canonical_json_bytes


GOLDEN_MANIFEST_SCHEMA_VERSION = "docarag.golden-manifest.v1"
GOLDEN_PREDICTION_SCHEMA_VERSION = "docarag.golden-predictions.v1"
GOLDEN_REPORT_SCHEMA_VERSION = "docarag.golden-report.v1"
MAX_GOLDEN_FILE_BYTES = 64 * 1024 * 1024
MAX_PAGE_TEXT_CHARACTERS = 250_000
MAX_EDIT_MATRIX_CELLS = 20_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GoldenEvaluationError(ValueError):
    pass


def _load_json(path: Path, *, maximum: int = MAX_GOLDEN_FILE_BYTES) -> dict[str, Any]:
    try:
        before = path.stat()
        if before.st_size > maximum:
            raise GoldenEvaluationError(f"golden file exceeds {maximum} bytes: {path}")
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise GoldenEvaluationError(f"golden file is unreadable: {path}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise GoldenEvaluationError(f"golden file changed while reading: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoldenEvaluationError(f"golden file is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GoldenEvaluationError("golden file root must be an object")
    return value


def load_golden_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    manifest = _load_json(source)
    if manifest.get("schema_version") != GOLDEN_MANIFEST_SCHEMA_VERSION:
        raise GoldenEvaluationError("unsupported golden manifest schema")
    if not isinstance(manifest.get("protocol_revision"), str) or not manifest["protocol_revision"]:
        raise GoldenEvaluationError("golden manifest requires protocol_revision")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise GoldenEvaluationError("golden manifest requires documents")
    ids: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise GoldenEvaluationError("golden document entry must be an object")
        document_id = document.get("document_id")
        digest = document.get("source_sha256")
        if not isinstance(document_id, str) or not document_id or document_id in ids:
            raise GoldenEvaluationError("golden document IDs must be non-empty and unique")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise GoldenEvaluationError("golden document requires source_sha256")
        ids.add(document_id)
        pages = document.get("pages", [])
        if not isinstance(pages, list):
            raise GoldenEvaluationError("golden document pages must be an array")
        page_numbers: set[int] = set()
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("page_no"), int):
                raise GoldenEvaluationError("golden page requires integer page_no")
            if page["page_no"] < 1 or page["page_no"] in page_numbers:
                raise GoldenEvaluationError("golden page numbers must be positive and unique")
            page_numbers.add(page["page_no"])
            _validate_page_annotation(page)
    return manifest


def evaluate_golden_run(
    manifest_path: str | Path,
    prediction_path: str | Path,
    *,
    verify_sources: bool = True,
) -> dict[str, object]:
    manifest = load_golden_manifest(manifest_path)
    predictions = _load_json(Path(prediction_path).expanduser().resolve())
    if predictions.get("schema_version") != GOLDEN_PREDICTION_SCHEMA_VERSION:
        raise GoldenEvaluationError("unsupported golden prediction schema")
    systems = predictions.get("systems")
    if not isinstance(systems, list) or not systems:
        raise GoldenEvaluationError("prediction envelope requires at least one system")
    system_ids: set[str] = set()
    results: list[dict[str, object]] = []
    source_checks = [_source_check(item, verify=verify_sources) for item in manifest["documents"]]
    for system in systems:
        if not isinstance(system, dict):
            raise GoldenEvaluationError("prediction system must be an object")
        system_id = system.get("system_id")
        if not isinstance(system_id, str) or not system_id or system_id in system_ids:
            raise GoldenEvaluationError("prediction system IDs must be non-empty and unique")
        system_ids.add(system_id)
        results.append(_evaluate_system(manifest, system))
    report: dict[str, object] = {
        "schema_version": GOLDEN_REPORT_SCHEMA_VERSION,
        "protocol_revision": manifest["protocol_revision"],
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "prediction_sha256": hashlib.sha256(canonical_json_bytes(predictions)).hexdigest(),
        "source_checks": source_checks,
        "systems": results,
    }
    report["qualification_ready"] = all(item["verified"] for item in source_checks) and all(
        bool(item["qualification_ready"]) for item in results
    )
    report["passed"] = bool(report["qualification_ready"]) and all(
        bool(item["passed"]) for item in results
    )
    return report


def _validate_page_annotation(page: Mapping[str, object]) -> None:
    text = page.get("text")
    if text is not None and (
        not isinstance(text, str) or len(text) > MAX_PAGE_TEXT_CHARACTERS
    ):
        raise GoldenEvaluationError("annotated page text is invalid or too large")
    for key in ("reading_order", "blocks", "table_cells", "formulas", "captions"):
        value = page.get(key)
        if value is not None and not isinstance(value, list):
            raise GoldenEvaluationError(f"annotated page {key} must be an array")


def _source_check(document: Mapping[str, object], *, verify: bool) -> dict[str, object]:
    path_value = document.get("source_path")
    result: dict[str, object] = {
        "document_id": document["document_id"],
        "expected_sha256": document["source_sha256"],
        "verified": False,
    }
    if not verify:
        result.update({"verified": True, "status": "verification_disabled"})
        return result
    if not isinstance(path_value, str) or not path_value:
        result["status"] = "source_path_missing"
        return result
    path = Path(path_value).expanduser()
    try:
        digest = _hash_stable_file(path)
    except OSError:
        result["status"] = "source_unavailable"
        return result
    result["observed_sha256"] = digest
    result["verified"] = digest == document["source_sha256"]
    result["status"] = "verified" if result["verified"] else "digest_mismatch"
    return result


def _hash_stable_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OSError("source changed during hashing")
    return digest.hexdigest()


def _evaluate_system(
    manifest: Mapping[str, object], system: Mapping[str, object]
) -> dict[str, object]:
    documents = system.get("documents")
    if not isinstance(documents, list):
        raise GoldenEvaluationError("prediction system documents must be an array")
    by_id = {
        item.get("document_id"): item
        for item in documents
        if isinstance(item, dict) and isinstance(item.get("document_id"), str)
    }
    samples: dict[str, list[float]] = {
        "q_text": [],
        "cer": [],
        "q_order": [],
        "q_block": [],
        "q_table": [],
        "q_formula": [],
        "q_caption": [],
        "provenance_validity": [],
        "citation_precision": [],
        "citation_recall": [],
        "answer_f1": [],
        "hydrated_structured_evidence_rate": [],
    }
    missing_documents: list[str] = []
    failed_pages = 0
    annotated_pages = 0
    for truth_document in manifest["documents"]:
        prediction_document = by_id.get(truth_document["document_id"])
        if not isinstance(prediction_document, Mapping):
            missing_documents.append(str(truth_document["document_id"]))
            for page in truth_document.get("pages", []):
                annotated_pages += 1
                failed_pages += 1
                _append_failed_applicable(samples, page)
            continue
        predicted_pages = {
            item.get("page_no"): item
            for item in prediction_document.get("pages", [])
            if isinstance(item, dict) and isinstance(item.get("page_no"), int)
        }
        for truth_page in truth_document.get("pages", []):
            annotated_pages += 1
            predicted_page = predicted_pages.get(truth_page["page_no"])
            if not isinstance(predicted_page, Mapping) or predicted_page.get("failed") is True:
                failed_pages += 1
                _append_failed_applicable(samples, truth_page)
                continue
            _score_page(samples, truth_page, predicted_page)
        _score_answers(samples, truth_document.get("questions", []), prediction_document.get("answers", []))

    metrics = {
        name: (sum(values) / len(values) if values else None)
        for name, values in samples.items()
    }
    runtime = system.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise GoldenEvaluationError("prediction runtime must be an object")
    metrics.update(
        {
            "seconds_per_page": _optional_nonnegative(runtime.get("seconds_per_page")),
            "gpu_peak_mib": _optional_nonnegative(runtime.get("gpu_peak_mib")),
            "cpu_peak_rss_mib": _optional_nonnegative(runtime.get("cpu_peak_rss_mib")),
            "page_failure_rate": failed_pages / annotated_pages if annotated_pages else None,
        }
    )
    thresholds = manifest.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise GoldenEvaluationError("manifest thresholds must be an object")
    required = tuple(manifest.get("required_metrics", ()))
    if any(not isinstance(item, str) for item in required):
        raise GoldenEvaluationError("required_metrics must contain strings")
    missing_metrics = [name for name in required if metrics.get(name) is None]
    threshold_failures: list[str] = []
    for name, threshold in thresholds.items():
        if name not in metrics or metrics[name] is None:
            continue
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise GoldenEvaluationError("metric thresholds must be numeric")
        if name in {"cer", "seconds_per_page", "gpu_peak_mib", "cpu_peak_rss_mib", "page_failure_rate"}:
            if float(metrics[name]) > float(threshold):
                threshold_failures.append(str(name))
        elif float(metrics[name]) < float(threshold):
            threshold_failures.append(str(name))
    crash = system.get("crash_recovery")
    crash_ready = isinstance(crash, Mapping) and all(
        crash.get(key) is True
        for key in ("resumed_from_checkpoint", "no_committed_data_loss", "no_false_success")
    )
    qualification_ready = not missing_documents and not missing_metrics and annotated_pages > 0
    return {
        "system_id": system["system_id"],
        "system_release": system.get("system_release"),
        "metrics": metrics,
        "applicable_counts": {name: len(values) for name, values in samples.items()},
        "annotated_pages": annotated_pages,
        "missing_documents": missing_documents,
        "missing_required_metrics": missing_metrics,
        "crash_recovery_passed": crash_ready,
        "qualification_ready": qualification_ready,
        "threshold_failures": threshold_failures,
        "passed": qualification_ready and crash_ready and not threshold_failures,
    }


def _append_failed_applicable(samples: dict[str, list[float]], page: Mapping[str, object]) -> None:
    mapping = {
        "text": ("q_text", "cer"),
        "reading_order": ("q_order",),
        "blocks": ("q_block",),
        "table_cells": ("q_table",),
        "formulas": ("q_formula",),
        "captions": ("q_caption",),
    }
    for annotation, metrics in mapping.items():
        if annotation in page:
            for metric in metrics:
                samples[metric].append(1.0 if metric == "cer" else 0.0)
    if "provenance_required" in page:
        samples["provenance_validity"].append(0.0)


def _score_page(
    samples: dict[str, list[float]],
    truth: Mapping[str, object],
    predicted: Mapping[str, object],
) -> None:
    if "text" in truth:
        actual = str(predicted.get("text", ""))
        expected = str(truth["text"])
        wer = _error_rate(_tokens(expected), _tokens(actual))
        cer = _error_rate(list(expected), list(actual))
        samples["q_text"].append(max(0.0, 1.0 - wer))
        samples["cer"].append(cer)
    if "reading_order" in truth:
        samples["q_order"].append(
            _reading_order_score(truth["reading_order"], predicted.get("reading_order", []))
        )
    if "blocks" in truth:
        samples["q_block"].append(_block_f1(truth["blocks"], predicted.get("blocks", [])))
    if "table_cells" in truth:
        samples["q_table"].append(
            _set_f1(_table_cell_set(truth["table_cells"]), _table_cell_set(predicted.get("table_cells", [])))
        )
    if "formulas" in truth:
        samples["q_formula"].append(
            _set_f1(_formula_set(truth["formulas"]), _formula_set(predicted.get("formulas", [])))
        )
    if "captions" in truth:
        samples["q_caption"].append(
            _set_f1(_caption_set(truth["captions"]), _caption_set(predicted.get("captions", [])))
        )
    if truth.get("provenance_required") is True:
        samples["provenance_validity"].append(
            1.0 if predicted.get("provenance_valid") is True else 0.0
        )


def _score_answers(samples: dict[str, list[float]], truth: object, predicted: object) -> None:
    if not isinstance(truth, Sequence) or isinstance(truth, (str, bytes)):
        return
    predicted_values = (
        predicted
        if isinstance(predicted, Sequence) and not isinstance(predicted, (str, bytes))
        else ()
    )
    predicted_by_id = {
        item.get("question_id"): item
        for item in predicted_values
        if isinstance(item, Mapping)
    }
    for question in truth:
        if not isinstance(question, Mapping):
            continue
        answer = predicted_by_id.get(question.get("question_id"), {})
        expected_answer = str(question.get("answer", ""))
        actual_answer = str(answer.get("answer", "")) if isinstance(answer, Mapping) else ""
        samples["answer_f1"].append(_token_f1(expected_answer, actual_answer))
        expected_citations = set(map(str, question.get("citations", [])))
        actual_citations = set(map(str, answer.get("citations", []))) if isinstance(answer, Mapping) else set()
        samples["citation_precision"].append(
            len(expected_citations & actual_citations) / len(actual_citations)
            if actual_citations
            else (1.0 if not expected_citations else 0.0)
        )
        samples["citation_recall"].append(
            len(expected_citations & actual_citations) / len(expected_citations)
            if expected_citations
            else 1.0
        )
        if question.get("requires_structured_evidence") is True:
            samples["hydrated_structured_evidence_rate"].append(
                1.0 if isinstance(answer, Mapping) and answer.get("structured_asset_hydrated") is True else 0.0
            )


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", value.casefold(), flags=re.UNICODE)


def _error_rate(expected: Sequence[object], actual: Sequence[object]) -> float:
    if not expected:
        return 0.0 if not actual else 1.0
    expected_length = len(expected)
    if len(expected) * len(actual) > MAX_EDIT_MATRIX_CELLS:
        raise GoldenEvaluationError("edit-distance input exceeds the frozen complexity bound")
    if len(expected) < len(actual):
        expected, actual = actual, expected
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1] / expected_length


def _reading_order_score(expected: object, actual: object) -> float:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return 0.0
    expected_ids = [str(value) for value in expected]
    if len(expected_ids) < 2:
        return 1.0 if expected_ids == [str(value) for value in actual] else 0.0
    positions = {str(value): index for index, value in enumerate(actual)}
    comparable = [positions[value] for value in expected_ids if value in positions]
    inversions = _inversion_count(comparable)
    correct = len(comparable) * (len(comparable) - 1) // 2 - inversions
    total = len(expected_ids) * (len(expected_ids) - 1) // 2
    return correct / total


def _inversion_count(values: Sequence[int]) -> int:
    def merge_count(items: list[int]) -> tuple[list[int], int]:
        if len(items) < 2:
            return items, 0
        middle = len(items) // 2
        left, left_count = merge_count(items[:middle])
        right, right_count = merge_count(items[middle:])
        merged: list[int] = []
        i = j = extra = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1; extra += len(left) - i
        return merged + left[i:] + right[j:], left_count + right_count + extra
    return merge_count(list(values))[1]


def _block_f1(expected: object, actual: object, *, iou_threshold: float = 0.5) -> float:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return 0.0
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(expected):
        if not isinstance(left, Mapping):
            continue
        for right_index, right in enumerate(actual):
            if not isinstance(right, Mapping) or left.get("type") != right.get("type"):
                continue
            score = _bbox_iou(left.get("bbox"), right.get("bbox"))
            if score >= iou_threshold:
                candidates.append((score, left_index, right_index))
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    for _, left_index, right_index in sorted(candidates, reverse=True):
        if left_index not in matched_left and right_index not in matched_right:
            matched_left.add(left_index); matched_right.add(right_index)
    return _f1_counts(len(matched_left), len(actual) - len(matched_right), len(expected) - len(matched_left))


def _bbox_iou(left: object, right: object) -> float:
    if not (
        isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 4
        and all(isinstance(value, (int, float)) for value in left + right)
    ):
        return 0.0
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    union = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0) + max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0) - intersection
    return intersection / union if union > 0 else 0.0


def _table_cell_set(value: object) -> set[tuple[object, ...]]:
    if not isinstance(value, list):
        return set()
    return {
        (
            str(item.get("table_id", "")), int(item.get("row", -1)), int(item.get("column", -1)),
            int(item.get("row_span", 1)), int(item.get("column_span", 1)), _normalize_text(str(item.get("text", ""))),
        )
        for item in value if isinstance(item, Mapping)
    }


def _formula_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {_normalize_formula(str(item.get("value", item))) if isinstance(item, Mapping) else _normalize_formula(str(item)) for item in value}


def _caption_set(value: object) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        return set()
    return {
        (str(item.get("figure_id", "")), _normalize_text(str(item.get("text", ""))))
        for item in value if isinstance(item, Mapping)
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_formula(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("\\left", "").replace("\\right", "")


def _set_f1(expected: set[object], actual: set[object]) -> float:
    return _f1_counts(len(expected & actual), len(actual - expected), len(expected - actual))


def _f1_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def _token_f1(expected: str, actual: str) -> float:
    left = _tokens(expected); right = _tokens(actual)
    from collections import Counter
    overlap = sum((Counter(left) & Counter(right)).values())
    return _f1_counts(overlap, len(right) - overlap, len(left) - overlap)


def _optional_nonnegative(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise GoldenEvaluationError("runtime metric must be finite and non-negative")
    return float(value)


__all__ = [
    "GOLDEN_MANIFEST_SCHEMA_VERSION",
    "GOLDEN_PREDICTION_SCHEMA_VERSION",
    "GOLDEN_REPORT_SCHEMA_VERSION",
    "GoldenEvaluationError",
    "evaluate_golden_run",
    "load_golden_manifest",
]
