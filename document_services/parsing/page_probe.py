"""Streaming, source-bound PDF page feature probing.

This application service deliberately understands neither OCR nor parser
vendors.  It verifies the local source bytes, opens a lightweight injected PDF
backend, and converts one page snapshot at a time into the routing contract's
``PageFeatures``.  Signals that cannot be observed cheaply remain ``None``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import hmac
import math
from os import PathLike, stat_result
from pathlib import Path
import re
from typing import Protocol
import unicodedata
from urllib.parse import unquote, urlparse

from document_services.domain.base import ContractValidationError, JsonObject
from document_services.domain.evidence import DocumentRef, PageRef
from document_services.ports.page_probe import PageProbe, PageRange
from document_services.parsing.routing import PageFeatures


Rectangle = tuple[float, float, float, float]
_WORD = re.compile(r"\S+")
_KNOWN_PDF_MIME_TYPE = "application/pdf"
_UNKNOWN_SIGNALS = (
    "ocr_native_disagreement_ratio",
    "connected_component_density",
    "skew_degrees",
    "candidate_table_regions",
    "candidate_equation_regions",
    "candidate_chart_regions",
    "candidate_diagram_regions",
    "column_count",
    "is_slide_like",
    "is_photographed_page",
)


class PageProbeError(RuntimeError):
    """Base class for source, selection, and backend probe failures."""


class UnsupportedProbeSourceError(PageProbeError):
    """The source is not a local PDF that this probe can bind safely."""


class SourceBindingError(PageProbeError):
    """Local bytes or page metadata differ from the authoritative source."""


class PageSelectionError(PageProbeError):
    """The requested page set is duplicate, malformed, or out of range."""


class PageProbeBackendError(PageProbeError):
    """A probe backend failed or violated its snapshot contract."""


class PageProbeDependencyError(PageProbeBackendError):
    """An optional PDF probe dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class PdfPageSnapshot:
    """One backend-neutral, unrotated PDF page snapshot.

    Geometry uses top-left page-point coordinates relative to the visible
    crop box.  The snapshot contains native text and object metadata only; it
    must never contain rendered pixels, embedded image bytes, or OCR output.
    """

    page_index: int
    width: float
    height: float
    rotation_degrees: int
    native_text: str
    text_boxes: tuple[Rectangle, ...] = ()
    image_boxes: tuple[Rectangle, ...] = ()
    font_count: int | None = None
    vector_path_count: int | None = None
    metadata: JsonObject = field(default_factory=dict)


class PdfProbeSession(Protocol):
    """Random-access lightweight PDF session owned by a backend."""

    @property
    def page_count(self) -> int:
        ...

    def inspect_page(self, page_index: int) -> PdfPageSnapshot:
        ...

    def close(self) -> None:
        ...


class PdfProbeBackend(Protocol):
    """Infrastructure port used by :class:`SourceBoundPdfPageProbe`."""

    @property
    def backend_name(self) -> str:
        ...

    @property
    def backend_version(self) -> str:
        ...

    def open(self, source_path: Path) -> PdfProbeSession:
        ...


@dataclass(frozen=True, slots=True)
class PdfProbePolicy:
    """Bounded, deterministic policy for cheap source probing."""

    sha256_chunk_bytes: int = 1024 * 1024
    infer_scanned_pages: bool = True
    scanned_max_native_characters: int = 20
    scanned_min_raster_coverage_ratio: float = 0.70

    def __post_init__(self) -> None:
        if (
            isinstance(self.sha256_chunk_bytes, bool)
            or not isinstance(self.sha256_chunk_bytes, int)
            or self.sha256_chunk_bytes < 4096
        ):
            raise ContractValidationError(
                "sha256_chunk_bytes must be an integer of at least 4096"
            )
        if not isinstance(self.infer_scanned_pages, bool):
            raise ContractValidationError("infer_scanned_pages must be a boolean")
        if (
            isinstance(self.scanned_max_native_characters, bool)
            or not isinstance(self.scanned_max_native_characters, int)
            or self.scanned_max_native_characters < 0
        ):
            raise ContractValidationError(
                "scanned_max_native_characters must be a non-negative integer"
            )
        ratio = self.scanned_min_raster_coverage_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0.0 <= float(ratio) <= 1.0
        ):
            raise ContractValidationError(
                "scanned_min_raster_coverage_ratio must be between 0 and 1"
            )
        object.__setattr__(
            self, "scanned_min_raster_coverage_ratio", float(ratio)
        )


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: stat_result) -> _SourceFingerprint:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


class SourceBoundPdfPageProbe(PageProbe[PageFeatures]):
    """Verify source identity and lazily produce cheap PDF routing features.

    Complexity is ``O(file_bytes)`` once for SHA-256, then ``O(R log R)`` time
    and ``O(R)`` memory per page for exact clipped rectangle-union coverage,
    where ``R`` is that page's text/image region count.  Only the current page
    snapshot is retained.  No renderer, OCR engine, parser, or VLM is invoked.
    """

    def __init__(
        self,
        backend: PdfProbeBackend,
        *,
        policy: PdfProbePolicy | None = None,
    ) -> None:
        name = getattr(backend, "backend_name", None)
        version = getattr(backend, "backend_version", None)
        if not isinstance(name, str) or not name.strip():
            raise ContractValidationError(
                "PDF probe backend_name must be a non-empty string"
            )
        if not isinstance(version, str) or not version.strip():
            raise ContractValidationError(
                "PDF probe backend_version must be a non-empty string"
            )
        if not callable(getattr(backend, "open", None)):
            raise ContractValidationError("PDF probe backend must define open()")
        self._backend = backend
        self._policy = policy or PdfProbePolicy()

    def probe(
        self,
        document: DocumentRef,
        *,
        source_path: str | PathLike[str] | None = None,
        page_indices: Sequence[int] | None = None,
        page_ranges: Sequence[PageRange] | None = None,
    ) -> Iterator[PageFeatures]:
        """Return a streaming iterator after verifying authoritative bytes.

        Page indices and ranges are copied immediately so caller mutation
        cannot change an in-flight probe.  Source SHA-256 verification also
        occurs before a backend is opened.
        """

        if not isinstance(document, DocumentRef):
            raise ContractValidationError("document must be DocumentRef")
        if document.mime_type != _KNOWN_PDF_MIME_TYPE:
            raise UnsupportedProbeSourceError(
                "PDF page probe requires mime_type 'application/pdf'"
            )
        if page_indices is not None and page_ranges is not None:
            raise PageSelectionError(
                "page_indices and page_ranges are mutually exclusive"
            )
        copied_indices = None if page_indices is None else tuple(page_indices)
        copied_ranges = None if page_ranges is None else tuple(page_ranges)
        path = _resolve_local_source(document, source_path)
        fingerprint = _verify_source_bytes(path, document, self._policy)
        return self._iterate_pages(
            document=document,
            source_path=path,
            fingerprint=fingerprint,
            page_indices=copied_indices,
            page_ranges=copied_ranges,
        )

    def _iterate_pages(
        self,
        *,
        document: DocumentRef,
        source_path: Path,
        fingerprint: _SourceFingerprint,
        page_indices: tuple[object, ...] | None,
        page_ranges: tuple[object, ...] | None,
    ) -> Iterator[PageFeatures]:
        _assert_source_unchanged(source_path, fingerprint)
        try:
            session = self._backend.open(source_path)
        except PageProbeError:
            raise
        except Exception as exc:
            raise PageProbeBackendError(
                f"{self._backend.backend_name} could not open the PDF: {exc}"
            ) from exc

        primary_error: BaseException | None = None
        try:
            try:
                observed_page_count = session.page_count
            except PageProbeError:
                raise
            except Exception as exc:
                raise PageProbeBackendError(
                    f"{self._backend.backend_name} could not report page_count: {exc}"
                ) from exc
            page_count = _validate_page_count(observed_page_count)
            _assert_source_unchanged(source_path, fingerprint)
            if document.page_count is not None and document.page_count != page_count:
                raise SourceBindingError(
                    "DocumentRef.page_count does not match the verified PDF: "
                    f"expected {document.page_count}, observed {page_count}"
                )
            selected = _select_page_indices(
                page_count=page_count,
                page_indices=page_indices,
                page_ranges=page_ranges,
            )
            for page_index in selected:
                _assert_source_unchanged(source_path, fingerprint)
                try:
                    snapshot = session.inspect_page(page_index)
                except PageProbeError:
                    raise
                except Exception as exc:
                    raise PageProbeBackendError(
                        f"{self._backend.backend_name} failed on page "
                        f"{page_index}: {exc}"
                    ) from exc
                yield self._to_features(
                    document=document,
                    snapshot=snapshot,
                    expected_page_index=page_index,
                    source_size=fingerprint.size,
                )
            _assert_source_unchanged(source_path, fingerprint)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                session.close()
            except Exception as exc:
                if primary_error is None:
                    raise PageProbeBackendError(
                        f"{self._backend.backend_name} could not close the PDF: {exc}"
                    ) from exc

    def _to_features(
        self,
        *,
        document: DocumentRef,
        snapshot: PdfPageSnapshot,
        expected_page_index: int,
        source_size: int,
    ) -> PageFeatures:
        if not isinstance(snapshot, PdfPageSnapshot):
            raise PageProbeBackendError(
                "PDF backend inspect_page() must return PdfPageSnapshot"
            )
        _validate_snapshot(snapshot, expected_page_index)
        text_boxes, invalid_text_boxes = _validated_rectangles(snapshot.text_boxes)
        image_boxes, invalid_image_boxes = _validated_rectangles(snapshot.image_boxes)
        page_area = float(snapshot.width) * float(snapshot.height)
        text_coverage = (
            None
            if invalid_text_boxes
            else _rectangle_union_area(
                text_boxes, width=float(snapshot.width), height=float(snapshot.height)
            )
            / page_area
        )
        raster_coverage = (
            None
            if invalid_image_boxes
            else _rectangle_union_area(
                image_boxes,
                width=float(snapshot.width),
                height=float(snapshot.height),
            )
            / page_area
        )
        text = snapshot.native_text
        character_count = len(text)
        word_count = len(_WORD.findall(text))
        replacement_ratio = _character_ratio(text, lambda char: char == "\ufffd")
        control_ratio = _character_ratio(
            text,
            lambda char: unicodedata.category(char) == "Cc"
            and char not in "\t\n\r\f",
        )
        duplicated_ratio = _duplicated_line_ratio(text)
        vector_density = (
            None
            if snapshot.vector_path_count is None
            else float(snapshot.vector_path_count) * 10_000.0 / page_area
        )
        scanned_page = None
        if (
            self._policy.infer_scanned_pages
            and raster_coverage is not None
            and character_count <= self._policy.scanned_max_native_characters
            and raster_coverage
            >= self._policy.scanned_min_raster_coverage_ratio
        ):
            scanned_page = True

        native_stats: dict[str, object] = {
            "native_character_count": character_count,
            "native_word_count": word_count,
            "text_coverage_ratio": text_coverage,
            "raster_image_coverage_ratio": raster_coverage,
            "replacement_character_ratio": replacement_ratio,
            "control_character_ratio": control_ratio,
            "duplicated_text_ratio": duplicated_ratio,
            "font_count": snapshot.font_count,
            "vector_path_density": vector_density,
            "candidate_figure_regions": len(snapshot.image_boxes),
            "is_scanned_page": scanned_page,
        }
        page = PageRef(
            document_version_id=document.document_version_id,
            page_index=expected_page_index,
            width=float(snapshot.width),
            height=float(snapshot.height),
            rotation_degrees=snapshot.rotation_degrees,
            native_stats=native_stats,
        )
        metadata: dict[str, object] = {
            "probe_source": self._backend.backend_name,
            "probe_backend_version": self._backend.backend_version,
            "probe_contract": "source_bound_pdf_page_probe.v1",
            "source_sha256_verified": True,
            "source_size_bytes": source_size,
            "geometry": {
                "coordinate_space": "unrotated_cropbox_page_points",
                "coordinate_origin": "top_left",
            },
            "coverage_algorithm": "exact_clipped_rectangle_union.v1",
            "text_block_count": len(snapshot.text_boxes),
            "raster_image_instance_count": len(snapshot.image_boxes),
            "invalid_text_box_count": invalid_text_boxes,
            "invalid_image_box_count": invalid_image_boxes,
            "vector_path_density_unit": "paths_per_10000_square_points",
            "scanned_page_heuristic": {
                "positive_only": True,
                "max_native_characters": self._policy.scanned_max_native_characters,
                "min_raster_coverage_ratio": self._policy.scanned_min_raster_coverage_ratio,
            },
            "signals_not_computed": _UNKNOWN_SIGNALS,
            "unknown_signals": tuple(
                sorted(
                    field_name
                    for field_name in PageFeatures.NATIVE_STAT_FIELDS
                    if native_stats.get(field_name) is None
                )
            ),
            "candidate_figure_semantics": (
                "displayed_raster_image_instances_not_semantic_figures"
            ),
            "backend_page_metadata": snapshot.metadata,
        }
        try:
            return PageFeatures.from_page_ref(
                page,
                document.mime_type,
                metadata=metadata,
            )
        except ContractValidationError as exc:
            raise PageProbeBackendError(
                "PDF backend snapshot contains invalid contract metadata"
            ) from exc


def _resolve_local_source(
    document: DocumentRef,
    source_path: str | PathLike[str] | None,
) -> Path:
    if source_path is not None:
        try:
            candidate = Path(source_path)
        except (TypeError, ValueError) as exc:
            raise UnsupportedProbeSourceError(
                "source_path must identify a local filesystem path"
            ) from exc
    else:
        parsed = urlparse(document.source_uri)
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise UnsupportedProbeSourceError(
                    "file source_uri must refer to the local host"
                )
            candidate = Path(unquote(parsed.path))
        elif parsed.scheme:
            raise UnsupportedProbeSourceError(
                "PDF page probe supports local paths and file:// URIs only"
            )
        else:
            candidate = Path(document.source_uri)
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsupportedProbeSourceError(
            f"local PDF source is unavailable: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise UnsupportedProbeSourceError(
            f"local PDF source is not a regular file: {resolved}"
        )
    return resolved


def _verify_source_bytes(
    source_path: Path,
    document: DocumentRef,
    policy: PdfProbePolicy,
) -> _SourceFingerprint:
    try:
        before = _SourceFingerprint.from_stat(source_path.stat())
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            while chunk := stream.read(policy.sha256_chunk_bytes):
                digest.update(chunk)
        after = _SourceFingerprint.from_stat(source_path.stat())
    except OSError as exc:
        raise UnsupportedProbeSourceError(
            f"could not read local PDF source: {source_path}"
        ) from exc
    if before != after:
        raise SourceBindingError("local PDF source changed during SHA-256 verification")
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, document.content_sha256):
        raise SourceBindingError(
            "local PDF SHA-256 differs from DocumentRef.content_sha256"
        )
    return after


def _assert_source_unchanged(
    source_path: Path,
    expected: _SourceFingerprint,
) -> None:
    try:
        actual = _SourceFingerprint.from_stat(source_path.stat())
    except OSError as exc:
        raise SourceBindingError("verified local PDF source became unavailable") from exc
    if actual != expected:
        raise SourceBindingError("verified local PDF source changed during probing")


def _validate_page_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PageProbeBackendError(
            "PDF backend page_count must be a non-negative integer"
        )
    return value


def _select_page_indices(
    *,
    page_count: int,
    page_indices: tuple[object, ...] | None,
    page_ranges: tuple[object, ...] | None,
) -> Iterable[int]:
    if page_indices is None and page_ranges is None:
        return range(page_count)
    if page_indices is not None:
        validated: list[int] = []
        seen: set[int] = set()
        for value in page_indices:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PageSelectionError(
                    "page_indices must contain non-negative integers"
                )
            if value in seen:
                raise PageSelectionError(
                    f"page selection contains duplicate page index {value}"
                )
            if value >= page_count:
                raise PageSelectionError(
                    f"page index {value} is outside PDF page_count {page_count}"
                )
            seen.add(value)
            validated.append(value)
        return tuple(sorted(validated))

    assert page_ranges is not None
    validated_ranges: list[PageRange] = []
    for value in page_ranges:
        if not isinstance(value, PageRange):
            raise PageSelectionError("page_ranges must contain PageRange values")
        if value.stop_index > page_count:
            raise PageSelectionError(
                "page range "
                f"[{value.start_index}, {value.stop_index}) exceeds PDF "
                f"page_count {page_count}"
            )
        validated_ranges.append(value)
    validated_ranges.sort(key=lambda item: (item.start_index, item.stop_index))
    previous: PageRange | None = None
    for current in validated_ranges:
        if previous is not None and current.start_index < previous.stop_index:
            raise PageSelectionError(
                "page_ranges overlap and would select a duplicate page: "
                f"[{previous.start_index}, {previous.stop_index}) and "
                f"[{current.start_index}, {current.stop_index})"
            )
        previous = current
    return (
        page_index
        for page_range in validated_ranges
        for page_index in range(page_range.start_index, page_range.stop_index)
    )


def _validate_snapshot(snapshot: PdfPageSnapshot, expected_page_index: int) -> None:
    if snapshot.page_index != expected_page_index:
        raise SourceBindingError(
            "PDF backend snapshot is bound to page "
            f"{snapshot.page_index}, expected {expected_page_index}"
        )
    for value, field_name in (
        (snapshot.width, "width"),
        (snapshot.height, "height"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise PageProbeBackendError(
                f"PDF page snapshot {field_name} must be finite and positive"
            )
    if snapshot.rotation_degrees not in {0, 90, 180, 270}:
        raise PageProbeBackendError(
            "PDF page snapshot rotation_degrees must be 0, 90, 180, or 270"
        )
    if not isinstance(snapshot.native_text, str):
        raise PageProbeBackendError("PDF page snapshot native_text must be a string")
    for value, field_name in (
        (snapshot.font_count, "font_count"),
        (snapshot.vector_path_count, "vector_path_count"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise PageProbeBackendError(
                f"PDF page snapshot {field_name} must be a non-negative integer or null"
            )
    if not isinstance(snapshot.metadata, Mapping):
        raise PageProbeBackendError("PDF page snapshot metadata must be a mapping")


def _validated_rectangles(
    values: object,
) -> tuple[tuple[Rectangle, ...], int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise PageProbeBackendError("PDF page snapshot boxes must be a sequence")
    valid: list[Rectangle] = []
    invalid_count = 0
    for value in values:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != 4
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            invalid_count += 1
            continue
        x0, y0, x1, y1 = (float(item) for item in value)
        if x1 < x0 or y1 < y0:
            invalid_count += 1
            continue
        valid.append((x0, y0, x1, y1))
    return tuple(valid), invalid_count


def _rectangle_union_area(
    rectangles: Sequence[Rectangle],
    *,
    width: float,
    height: float,
) -> float:
    """Return exact clipped axis-aligned union area in ``O(R log R)`` time."""

    clipped: list[Rectangle] = []
    for x0, y0, x1, y1 in rectangles:
        left = min(width, max(0.0, x0))
        top = min(height, max(0.0, y0))
        right = min(width, max(0.0, x1))
        bottom = min(height, max(0.0, y1))
        if right > left and bottom > top:
            clipped.append((left, top, right, bottom))
    if not clipped:
        return 0.0

    y_values = sorted({value for rect in clipped for value in (rect[1], rect[3])})
    y_index = {value: index for index, value in enumerate(y_values)}
    segment_count = len(y_values) - 1
    cover = [0] * (4 * segment_count + 8)
    covered_length = [0.0] * (4 * segment_count + 8)

    def update(
        node: int,
        left: int,
        right: int,
        query_left: int,
        query_right: int,
        delta: int,
    ) -> None:
        if query_left <= left and right <= query_right:
            cover[node] += delta
        else:
            middle = (left + right) // 2
            if query_left < middle:
                update(node * 2, left, middle, query_left, query_right, delta)
            if middle < query_right:
                update(
                    node * 2 + 1,
                    middle,
                    right,
                    query_left,
                    query_right,
                    delta,
                )
        if cover[node] > 0:
            covered_length[node] = y_values[right] - y_values[left]
        elif right - left == 1:
            covered_length[node] = 0.0
        else:
            covered_length[node] = (
                covered_length[node * 2] + covered_length[node * 2 + 1]
            )

    events: list[tuple[float, int, int, int]] = []
    for x0, y0, x1, y1 in clipped:
        start = y_index[y0]
        stop = y_index[y1]
        events.append((x0, 1, start, stop))
        events.append((x1, -1, start, stop))
    events.sort(key=lambda item: item[0])

    area = 0.0
    previous_x = events[0][0]
    index = 0
    while index < len(events):
        x = events[index][0]
        area += (x - previous_x) * covered_length[1]
        while index < len(events) and events[index][0] == x:
            _, delta, start, stop = events[index]
            update(1, 0, segment_count, start, stop, delta)
            index += 1
        previous_x = x
    return min(width * height, max(0.0, area))


def _character_ratio(
    text: str,
    predicate: Callable[[str], bool],
) -> float | None:
    if not text:
        return None
    return sum(1 for char in text if predicate(char)) / len(text)


def _duplicated_line_ratio(text: str) -> float | None:
    normalized = [
        " ".join(line.split()).casefold()
        for line in text.splitlines()
        if line.strip()
    ]
    if not normalized:
        return None
    counts = Counter(normalized)
    total_characters = sum(len(line) for line in normalized)
    if total_characters == 0:
        return None
    duplicate_characters = sum(
        len(line) * (count - 1) for line, count in counts.items() if count > 1
    )
    return duplicate_characters / total_characters


__all__ = [
    "PageProbeBackendError",
    "PageProbeDependencyError",
    "PageProbeError",
    "PageSelectionError",
    "PdfPageSnapshot",
    "PdfProbeBackend",
    "PdfProbePolicy",
    "PdfProbeSession",
    "SourceBindingError",
    "SourceBoundPdfPageProbe",
    "UnsupportedProbeSourceError",
]
