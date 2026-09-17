"""Dependency-free PaddleOCR-VL page-result adapter.

The controller must not import PaddleOCR, PaddlePaddle, vLLM, or Torch.  A
composition root therefore injects a loader which returns JSON-compatible page
results.  The accepted core shape is the one written by PaddleOCR-VL v1.6's
``save_to_json`` method::

    {
      "input_path": "/staged/document.pdf",
      "page_index": 0,
      "page_count": 12,
      "width": 1224,
      "height": 1584,
      "model_settings": {...},
      "layout_det_res": {"boxes": [...]},
      "parsing_res_list": [{
        "block_label": "table",
        "block_content": "<table>...</table>",
        "block_bbox": [x0, y0, x1, y1],
        "block_polygon_points": [[x, y], ...],
        "block_id": 7,
        "block_order": 3,
        "group_id": 7
      }]
    }

An out-of-process runner may enrich that JSON with ``source_sha256``,
``document_version_id``, ``source_uri``, ``raw_artifact_uri``, explicit block
``assets``, a page ``saved_assets`` map, or ``asset_base_uri``.  Supplying
``asset_base_uri`` declares that Paddle's standard ``imgs/img_in_*`` crops were
saved beneath it; this module only constructs references and never touches the
filesystem.  A runner which retains native parent/caption/footnote/reference
IDs may publish strict ``docarag_structural_relations`` descriptors; those IDs
are resolved without text or geometry fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
import mimetypes
import os
from time import perf_counter
from urllib.parse import unquote, urlparse

from document_services.domain.base import (
    ContractValidationError,
    content_sha256,
    freeze_json,
    json_ready,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
)
from document_services.domain.evidence import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    CoordinateSpace,
    ObservationRelation,
    ParserBlockObservation,
    ParserProvenance,
    Point,
    Representation,
    RepresentationKind,
    RelationType,
    SourceGeometry,
)
from document_services.domain.parsing import (
    AcceleratorKind,
    ErrorClassification,
    IssueSeverity,
    ParseErrorCode,
    ParseIssue,
    ParseRequest,
    ParseResult,
    ParseStatus,
    ParserArtifactReceipt,
    ParserCapabilityManifest,
    ParserExecutionError,
    ParserModality,
)
from document_services.parser_adapters.semantic_normalization import (
    explicit_diagram_substructure,
    explicit_table_grid_substructure,
    mathml_formula_substructure,
    preflight_explicit_json,
    preflight_mathml,
)


PaddlePageLoader = Callable[[ParseRequest], Iterable[Mapping[str, object]]]

ADAPTER_NAME = "paddleocr-vl-page"
ADAPTER_VERSION = "1.1.0"
PARSER_NAME = "paddleocr-vl"
TRANSLATION_NAME = "paddleocr_vl_page_result_to_observation.v2"

# PaddleOCR-VL's official page JSON exposes parser-native ``block_id`` and
# ``block_order`` values, plus ``global_block_id``/``global_group_id`` after
# ``restructure_pages``.  A source-bound runner may additionally publish the
# strict descriptor array below when it can retain other native relations that
# are not part of Paddle's public page-result JSON.  Descriptors are ID based;
# text, geometry, group proximity, and label adjacency are never relation
# evidence.
_STRUCTURAL_RELATIONS_FIELD = "docarag_structural_relations"
_STRUCTURAL_RELATION_FIELDS = {
    "relation_type",
    "target_block_id",
    "target_parser_native_id",
    "target_global_block_id",
    "target_page_index",
    "confidence",
}

_TEXT_LABELS = {
    "text",
    "abstract",
    "content",
    "reference_content",
    "aside_text",
}
_TITLE_LABELS = {"doc_title", "paragraph_title", "section_title", "title"}
_LIST_LABELS = {"list", "list_item"}
_TABLE_LABELS = {"table"}
_FORMULA_LABELS = {"display_formula", "formula", "equation", "inline_formula"}
_FIGURE_LABELS = {
    "image",
    "figure",
    "seal",
    "header_image",
    "footer_image",
}
_CHART_LABELS = {"chart", "plot"}
_DIAGRAM_LABELS = {"diagram", "flowchart", "workflow", "sketch"}
_CAPTION_LABELS = {
    "caption",
    "figure_caption",
    "image_caption",
    "table_caption",
}
_VISUAL_LABELS = _FIGURE_LABELS | _CHART_LABELS | _DIAGRAM_LABELS


def _sequence(value: object, field: str) -> Sequence[object]:
    return require_sequence(value, field)


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ContractValidationError(f"{field}[{index}] must be a string")
            if item.strip():
                result.append(item.strip())
        return "\n".join(result) or None
    raise ContractValidationError(f"{field} must be a string or string array")


def _positive_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ContractValidationError(f"{field} must be finite and positive")
    return float(value)


def _page_index(value: object, field: str = "page_index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field} must be a non-negative integer")
    return value


def _probability(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ContractValidationError(f"{field} must be between 0 and 1")
    return float(value)


def _reading_order(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractValidationError("block_order must be a finite number or null")
    return float(value)


def _normalized_source(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme in {"", "file"}:
        path = unquote(parsed.path) if parsed.scheme == "file" else value
        return "file", os.path.normcase(os.path.abspath(os.path.normpath(path)))
    return "uri", value


def _validate_source_binding(request: ParseRequest, page: Mapping[str, object]) -> None:
    input_path = require_non_empty(page.get("input_path"), "page.input_path")
    matched = _normalized_source(input_path) == _normalized_source(
        request.document.source_uri
    )

    declared_version = page.get("document_version_id")
    if declared_version is not None:
        if declared_version != request.document.document_version_id:
            raise ContractValidationError(
                "page document_version_id differs from the parse request"
            )
        matched = True

    declared_hash = page.get("source_sha256")
    if declared_hash is not None:
        require_sha256(declared_hash, "page.source_sha256")
        if declared_hash != request.document.content_sha256:
            raise ContractValidationError("page source_sha256 differs from source bytes")
        matched = True

    declared_uri = page.get("source_uri")
    if declared_uri is not None:
        require_non_empty(declared_uri, "page.source_uri")
        if _normalized_source(declared_uri) != _normalized_source(
            request.document.source_uri
        ):
            raise ContractValidationError("page source_uri differs from the parse request")
        matched = True

    if not matched:
        raise ContractValidationError(
            "page input_path is not bound to the requested source; inject "
            "source_sha256 or source_uri when parsing a staged copy"
        )


def _bbox(value: object, field: str, *, width: float, height: float) -> BoundingBox:
    bbox = BoundingBox.from_sequence(_sequence(value, field))
    if bbox.x1 > width or bbox.y1 > height:
        raise ContractValidationError(f"{field} lies outside page dimensions")
    return bbox


def _polygon(value: object, field: str, *, width: float, height: float) -> tuple[Point, ...]:
    if value is None:
        return ()
    values = _sequence(value, field)
    points: list[Point] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            point = Point.from_dict(item)
        else:
            coordinates = _sequence(item, f"{field}[{index}]")
            if len(coordinates) != 2:
                raise ContractValidationError(
                    f"{field}[{index}] must contain exactly x and y"
                )
            point = Point(coordinates[0], coordinates[1])  # type: ignore[arg-type]
        if point.x < 0 or point.y < 0 or point.x > width or point.y > height:
            raise ContractValidationError(f"{field}[{index}] lies outside page dimensions")
        points.append(point)
    if points and len(points) < 3:
        raise ContractValidationError(f"{field} must contain at least three points")
    return tuple(points)


def _geometry(
    block: Mapping[str, object], *, width: float, height: float
) -> SourceGeometry:
    bbox_value = block.get("block_bbox", block.get("bbox"))
    polygon_value = block.get("block_polygon_points", block.get("polygon_points"))
    if bbox_value is None and polygon_value is None:
        raise ContractValidationError("block requires block_bbox or block_polygon_points")
    return SourceGeometry(
        coordinate_space=CoordinateSpace.PAGE_PIXELS,
        bbox=(
            _bbox(bbox_value, "block_bbox", width=width, height=height)
            if bbox_value is not None
            else None
        ),
        polygon=_polygon(
            polygon_value, "block_polygon_points", width=width, height=height
        ),
        page_width=width,
        page_height=height,
    )


def _block_type(label: str) -> tuple[BlockType, str | None]:
    if label in _TITLE_LABELS:
        return BlockType.TITLE, label
    if label in _TEXT_LABELS:
        return BlockType.PARAGRAPH, None if label == "text" else label
    if label in _LIST_LABELS:
        return BlockType.LIST_ITEM, label
    if label in _TABLE_LABELS:
        return BlockType.TABLE, None
    if label in _FORMULA_LABELS:
        return BlockType.EQUATION, label
    if label in _FIGURE_LABELS:
        return BlockType.FIGURE, label
    if label in _CHART_LABELS:
        return BlockType.CHART, label
    if label in _DIAGRAM_LABELS:
        return BlockType.DIAGRAM, label
    if label in _CAPTION_LABELS:
        return BlockType.CAPTION, label
    if label == "footnote":
        return BlockType.FOOTNOTE, None
    if label == "header":
        return BlockType.HEADER, None
    if label == "footer":
        return BlockType.FOOTER, None
    if label == "code":
        return BlockType.CODE, None
    return BlockType.UNKNOWN, label


def _layout_confidences(
    page: Mapping[str, object], *, width: float, height: float
) -> tuple[dict[tuple[str, tuple[float, ...]], tuple[float, ...]], list[str]]:
    layout = page.get("layout_det_res")
    if layout is None:
        return {}, []
    layout_mapping = require_mapping(layout, "layout_det_res")
    boxes = _sequence(layout_mapping.get("boxes", ()), "layout_det_res.boxes")
    values: dict[tuple[str, tuple[float, ...]], list[float]] = {}
    errors: list[str] = []
    for index, raw_box in enumerate(boxes):
        try:
            box = require_mapping(raw_box, f"layout_det_res.boxes[{index}]")
            label = require_non_empty(
                box.get("label"), f"layout_det_res.boxes[{index}].label"
            ).strip().lower()
            coordinate = _bbox(
                box.get("coordinate"),
                f"layout_det_res.boxes[{index}].coordinate",
                width=width,
                height=height,
            )
            score = _probability(
                box.get("score"), f"layout_det_res.boxes[{index}].score"
            )
            key = (label, tuple(coordinate.to_list()))
            values.setdefault(key, []).append(score)
        except (ContractValidationError, TypeError, ValueError) as exc:
            errors.append(f"layout box {index} is invalid: {exc}")
    return {key: tuple(sorted(items)) for key, items in values.items()}, errors


def _confidence(
    block: Mapping[str, object],
    *,
    label: str,
    geometry: SourceGeometry,
    layout_scores: Mapping[tuple[str, tuple[float, ...]], tuple[float, ...]],
) -> float | None:
    for key in ("confidence", "block_confidence", "score"):
        if key in block and block[key] is not None:
            return _probability(block[key], key)
    if geometry.bbox is None:
        return None
    candidates = layout_scores.get((label, tuple(geometry.bbox.to_list())), ())
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise ContractValidationError(
            "layout confidence is ambiguous for duplicate label/geometry"
        )
    return candidates[0]


def _add_representation(
    destination: list[Representation],
    *,
    kind: RepresentationKind,
    value: object,
    media_type: str,
    confidence: float | None,
    language: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    text = _optional_text(value, kind.value)
    if text is not None:
        destination.append(
            Representation(
                kind=kind,
                content=text,
                media_type=media_type,
                confidence=confidence,
                language=language,
                metadata={} if metadata is None else metadata,
            )
        )


def _use_ocr_for_image_block(settings: Mapping[str, object]) -> bool | None:
    raw_options = settings.get("docarag_inference_options")
    if raw_options is None:
        return None
    options = require_mapping(
        raw_options, "model_settings.docarag_inference_options"
    )
    value = options.get("useOcrForImageBlock")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ContractValidationError(
            "model_settings.docarag_inference_options.useOcrForImageBlock "
            "must be a boolean"
        )
    return value


def _representations(
    block: Mapping[str, object],
    *,
    block_type: BlockType,
    label: str,
    confidence: float | None,
    language: str | None,
    use_ocr_for_image_block: bool | None,
) -> tuple[Representation, ...]:
    result: list[Representation] = []
    content = block.get("block_content")

    if block_type == BlockType.TABLE:
        _add_representation(
            result,
            kind=RepresentationKind.HTML,
            value=block.get("table_html"),
            media_type="text/html",
            confidence=confidence,
        )
        _add_representation(
            result,
            kind=RepresentationKind.MARKDOWN,
            value=block.get("table_markdown"),
            media_type="text/markdown",
            confidence=confidence,
        )
        _add_representation(
            result,
            kind=RepresentationKind.CSV,
            value=block.get("table_csv"),
            media_type="text/csv",
            confidence=confidence,
        )
        if block.get("table_grid") is not None:
            preflight_explicit_json(block["table_grid"], field="table_grid")
            result.append(
                Representation(
                    RepresentationKind.TABLE_GRID,
                    block["table_grid"],
                    "application/json",
                    confidence=confidence,
                )
            )
        content_text = _optional_text(content, "block_content")
        if content_text is not None:
            content_format = block.get("content_format")
            if content_format is not None and content_format not in {
                "html",
                "markdown",
                "csv",
                "text",
            }:
                raise ContractValidationError("unsupported table content_format")
            if content_format == "html" or (
                content_format is None and content_text.lstrip().lower().startswith("<table")
            ):
                kind, media_type = RepresentationKind.HTML, "text/html"
            elif content_format == "markdown":
                kind, media_type = RepresentationKind.MARKDOWN, "text/markdown"
            elif content_format == "csv":
                kind, media_type = RepresentationKind.CSV, "text/csv"
            else:
                kind, media_type = RepresentationKind.OCR_TEXT, "text/plain"
            _add_representation(
                result,
                kind=kind,
                value=content_text,
                media_type=media_type,
                confidence=confidence,
                language=language if kind == RepresentationKind.OCR_TEXT else None,
            )
        _add_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=block.get("caption", block.get("table_caption")),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=block.get("footnote", block.get("table_footnote")),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_representation(
            result,
            kind=RepresentationKind.OCR_TEXT,
            value=block.get("ocr_text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        # Paddle's restructure_pages(merge_tables=True) deliberately clears
        # the continuation block's content after merging it into the retained
        # root table.  Preserve its explicit global-ID assertion as structured
        # evidence so the continuation region is not discarded as empty.
        if not result:
            raw_global_id = block.get("global_block_id")
            raw_group_id = block.get("global_group_id")
            if raw_global_id is not None and raw_group_id is not None:
                global_id = _native_reference_scalar(
                    raw_global_id, "global_block_id"
                )
                group_id = _native_reference_scalar(
                    raw_group_id, "global_group_id"
                )
                if _native_reference_key(global_id) != _native_reference_key(
                    group_id
                ):
                    result.append(
                        Representation(
                            RepresentationKind.STRUCTURED_DATA,
                            {
                                "kind": "paddleocr_vl_table_continuation",
                                "global_block_id": global_id,
                                "global_group_id": group_id,
                            },
                            "application/json",
                            confidence=confidence,
                            metadata={
                                "assertion_origin": (
                                    "paddleocr_vl_restructured_table_group"
                                ),
                                "source_fields": [
                                    "global_block_id",
                                    "global_group_id",
                                ],
                            },
                        )
                    )
    elif block_type == BlockType.EQUATION:
        explicit_latex = block.get("latex")
        if explicit_latex is not None:
            _add_representation(
                result,
                kind=RepresentationKind.LATEX,
                value=explicit_latex,
                media_type="application/x-latex",
                confidence=confidence,
                metadata={"source_field": "latex", "format": "latex"},
            )
        content_text = _optional_text(content, "block_content")
        if content_text is not None:
            raw_format = block.get("content_format", block.get("math_format"))
            explicit_format = (
                raw_format.strip().casefold()
                if isinstance(raw_format, str) and raw_format.strip()
                else None
            )
            is_latex = explicit_format in {"latex", "tex"}
            _add_representation(
                result,
                kind=(
                    RepresentationKind.LATEX
                    if is_latex
                    else RepresentationKind.OCR_TEXT
                ),
                value=content_text,
                media_type=("application/x-latex" if is_latex else "text/plain"),
                confidence=confidence,
                language=None if is_latex else language,
                metadata={
                    "source_field": "block_content",
                    "format": explicit_format or "unknown",
                    "assertion_origin": "paddleocr_vl_formula_block_content",
                },
            )
        if block.get("mathml") is not None:
            preflight_mathml(block["mathml"], field="mathml")
        _add_representation(
            result,
            kind=RepresentationKind.MATHML,
            value=block.get("mathml"),
            media_type="application/mathml+xml",
            confidence=confidence,
        )
        _add_representation(
            result,
            kind=RepresentationKind.OCR_TEXT,
            value=block.get("ocr_text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    elif block_type in {BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM}:
        _add_representation(
            result,
            kind=RepresentationKind.VISUAL_DESCRIPTION,
            value=block.get("visual_description"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
            metadata={
                "source_field": "visual_description",
                "assertion_origin": "paddleocr_vl_explicit_visual_description",
            },
        )
        if content is not None:
            content_is_ocr = use_ocr_for_image_block is True
            _add_representation(
                result,
                kind=(
                    RepresentationKind.OCR_TEXT
                    if content_is_ocr
                    else RepresentationKind.TEXT
                ),
                value=content,
                media_type="text/plain",
                confidence=confidence,
                language=language,
                metadata={
                    "source_field": "block_content",
                    "format": "ocr_text" if content_is_ocr else "unknown",
                    "assertion_origin": (
                        "model_settings.docarag_inference_options."
                        "useOcrForImageBlock"
                        if content_is_ocr
                        else "paddleocr_vl_ambiguous_visual_block_content"
                    ),
                },
            )
        _add_representation(
            result,
            kind=RepresentationKind.OCR_TEXT,
            value=block.get("ocr_text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=block.get("caption", block.get("image_caption")),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        if block.get("diagram_graph") is not None:
            preflight_explicit_json(block["diagram_graph"], field="diagram_graph")
            result.append(
                Representation(
                    RepresentationKind.DIAGRAM_GRAPH,
                    block["diagram_graph"],
                    "application/json",
                    confidence=confidence,
                )
            )
        if block.get("chart_data") is not None:
            preflight_explicit_json(block["chart_data"], field="chart_data")
            result.append(
                Representation(
                    RepresentationKind.STRUCTURED_DATA,
                    block["chart_data"],
                    "application/json",
                    confidence=confidence,
                )
            )
    elif block_type == BlockType.CAPTION:
        _add_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=content,
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    elif block_type == BlockType.FOOTNOTE:
        _add_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=content,
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    else:
        _add_representation(
            result,
            kind=RepresentationKind.TEXT,
            value=block.get("native_text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_representation(
            result,
            kind=RepresentationKind.OCR_TEXT,
            value=block.get("ocr_text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        content_source = block.get("content_source", "ocr_text")
        if content_source in {"native", "native_text"}:
            content_kind = RepresentationKind.TEXT
        elif content_source in {"ocr", "ocr_text", None}:
            content_kind = RepresentationKind.OCR_TEXT
        else:
            raise ContractValidationError(
                "content_source must be native_text or ocr_text"
            )
        _add_representation(
            result,
            kind=content_kind,
            value=content,
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )

    unique: dict[str, Representation] = {}
    for representation in result:
        unique.setdefault(representation.representation_id, representation)
    return tuple(unique.values())


def _join_asset_uri(base_uri: str | None, value: str) -> str:
    require_non_empty(value, "asset uri")
    if urlparse(value).scheme or os.path.isabs(value) or base_uri is None:
        return value
    return f"{base_uri.rstrip('/')}/{value.lstrip('/')}"


def _asset_from_descriptor(
    value: object, *, base_uri: str | None, field: str
) -> AssetRef:
    if isinstance(value, str):
        descriptor: Mapping[str, object] = {"uri": value}
    else:
        descriptor = require_mapping(value, field)
        reject_unknown_fields(
            descriptor,
            {"uri", "path", "sha256", "media_type", "kind", "metadata"},
            dto="PaddleOCRVLAsset",
        )
    uri_value = descriptor.get("uri", descriptor.get("path"))
    uri = _join_asset_uri(base_uri, require_non_empty(uri_value, f"{field}.uri"))
    sha256 = descriptor.get("sha256")
    if sha256 is not None:
        require_sha256(sha256, f"{field}.sha256")
    media_type = descriptor.get("media_type")
    if media_type is None:
        media_type = mimetypes.guess_type(uri)[0]
    elif not isinstance(media_type, str) or not media_type.strip():
        raise ContractValidationError(f"{field}.media_type must be a non-empty string")
    kind_value = descriptor.get("kind", AssetKind.REGION_CROP.value)
    try:
        kind = AssetKind(kind_value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field}.kind is unsupported") from exc
    metadata = require_mapping(descriptor.get("metadata", {}), f"{field}.metadata")
    return AssetRef(
        kind=kind,
        uri=uri,
        media_type=media_type,  # type: ignore[arg-type]
        sha256=sha256,  # type: ignore[arg-type]
        metadata=metadata,
    )


def _filename_coordinate(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _assets(
    page: Mapping[str, object],
    block: Mapping[str, object],
    *,
    label: str,
    native_id: str,
    geometry: SourceGeometry,
) -> tuple[AssetRef, ...]:
    base_uri = _optional_text(page.get("asset_base_uri"), "asset_base_uri")
    descriptors: list[object] = []

    block_assets = block.get("assets")
    if block_assets is not None:
        if isinstance(block_assets, Mapping) or isinstance(block_assets, str):
            descriptors.append(block_assets)
        else:
            descriptors.extend(_sequence(block_assets, "block.assets"))

    direct_uri = None
    for key in ("asset_uri", "asset_path", "image_path", "img_path"):
        if block.get(key) is not None:
            candidate = require_non_empty(block[key], f"block.{key}")
            if direct_uri is not None and candidate != direct_uri:
                raise ContractValidationError("block declares conflicting asset paths")
            direct_uri = candidate
    if direct_uri is not None:
        direct_descriptor: dict[str, object] = {"uri": direct_uri}
        if block.get("asset_sha256") is not None:
            direct_descriptor["sha256"] = block["asset_sha256"]
        descriptors.append(direct_descriptor)

    saved_assets_value = page.get("saved_assets")
    if saved_assets_value is not None:
        saved_assets = require_mapping(saved_assets_value, "page.saved_assets")
        block_id = block.get("block_id")
        keys = (native_id, f"block:{block_id}", str(block_id))
        for key in keys:
            if key in saved_assets:
                saved = saved_assets[key]
                if isinstance(saved, Sequence) and not isinstance(
                    saved, (str, bytes, bytearray, Mapping)
                ):
                    descriptors.extend(saved)
                else:
                    descriptors.append(saved)
                break

    if not descriptors and base_uri is not None and label in _VISUAL_LABELS:
        if geometry.bbox is None:
            raise ContractValidationError(
                "standard Paddle saved-asset naming requires a block bbox"
            )
        coordinates = "_".join(
            _filename_coordinate(value) for value in geometry.bbox.to_list()
        )
        descriptors.append(f"imgs/img_in_{label}_box_{coordinates}.jpg")

    assets = [
        _asset_from_descriptor(value, base_uri=base_uri, field=f"assets[{index}]")
        for index, value in enumerate(descriptors)
    ]
    unique: dict[str, AssetRef] = {}
    for asset in assets:
        unique.setdefault(asset.asset_id, asset)
    return tuple(unique.values())


def _native_id(page_index: int, block: Mapping[str, object]) -> str:
    native = block.get("parser_native_id")
    if native is not None:
        return f"page:{page_index}:{require_non_empty(native, 'parser_native_id')}"
    block_id = block.get("block_id")
    if block_id is not None:
        if isinstance(block_id, bool) or not isinstance(block_id, (str, int)):
            raise ContractValidationError("block_id must be a string or integer")
        if isinstance(block_id, str) and not block_id.strip():
            raise ContractValidationError("block_id must not be empty")
        return f"page:{page_index}:block:{block_id}"
    signature = {
        "page_index": page_index,
        "label": block.get("block_label"),
        "bbox": block.get("block_bbox", block.get("bbox")),
        "polygon": block.get("block_polygon_points", block.get("polygon_points")),
        "order": block.get("block_order"),
        "group": block.get("group_id"),
        "content": block.get("block_content"),
    }
    return f"page:{page_index}:content:{content_sha256(signature)[:32]}"


@dataclass(frozen=True, slots=True)
class _PaddleBlockContext:
    """Raw-to-observation binding retained for structural resolution."""

    page_index: int
    block_index: int
    block: Mapping[str, object]
    native_id: str
    observation_id: str | None = None


def _native_reference_scalar(value: object, field: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ContractValidationError(f"{field} must be a string or integer native ID")
    if isinstance(value, str) and not value.strip():
        raise ContractValidationError(f"{field} must not be empty")
    return value


def _native_reference_key(value: str | int) -> tuple[str, str | int]:
    # JSON integers and strings are different native-ID namespaces.  In
    # particular, block 7 must never be guessed to mean block "7".
    return ("integer", value) if isinstance(value, int) else ("string", value)


def _attach_structural_relations(
    observations: Sequence[ParserBlockObservation],
    contexts: Sequence[_PaddleBlockContext],
    *,
    layout_order_attested_by_page: Mapping[int, bool],
) -> tuple[ParserBlockObservation, ...]:
    """Attach only native, ID-resolved Paddle structural assertions.

    The public PaddleOCR-VL page contract proves reading order through a
    non-null ``block_order`` when layout detection ran.  Restructured results
    additionally prove cross-page table grouping with global IDs.  Other
    relation types are accepted only through the strict, runner-supplied
    ``docarag_structural_relations`` descriptor array.  Every attempted
    reference is audited on the source observation; unresolved, ambiguous,
    malformed, self, and type-mismatched targets never become an edge.
    """

    observation_by_id = {item.observation_id: item for item in observations}
    assertions: dict[str, list[ObservationRelation]] = {
        item.observation_id: list(item.relations) for item in observations
    }
    resolution_records: dict[str, list[dict[str, object]]] = {
        item.observation_id: [] for item in observations
    }

    by_page_block_id: dict[
        tuple[int, tuple[str, str | int]], list[_PaddleBlockContext]
    ] = {}
    by_page_parser_native_id: dict[
        tuple[int, tuple[str, str | int]], list[_PaddleBlockContext]
    ] = {}
    by_global_block_id: dict[
        tuple[str, str | int], list[_PaddleBlockContext]
    ] = {}
    by_page: dict[int, list[_PaddleBlockContext]] = {}

    for context in contexts:
        by_page.setdefault(context.page_index, []).append(context)
        raw_block_id = context.block.get("block_id")
        if raw_block_id is not None:
            try:
                block_id = _native_reference_scalar(raw_block_id, "block_id")
            except ContractValidationError:
                pass
            else:
                by_page_block_id.setdefault(
                    (context.page_index, _native_reference_key(block_id)), []
                ).append(context)
        raw_parser_id = context.block.get("parser_native_id")
        if raw_parser_id is not None:
            try:
                parser_id = _native_reference_scalar(
                    raw_parser_id, "parser_native_id"
                )
            except ContractValidationError:
                pass
            else:
                by_page_parser_native_id.setdefault(
                    (context.page_index, _native_reference_key(parser_id)), []
                ).append(context)
        raw_global_id = context.block.get("global_block_id")
        if raw_global_id is not None:
            try:
                global_id = _native_reference_scalar(
                    raw_global_id, "global_block_id"
                )
            except ContractValidationError:
                pass
            else:
                by_global_block_id.setdefault(
                    _native_reference_key(global_id), []
                ).append(context)

    def record(
        source: _PaddleBlockContext,
        *,
        relation_type: RelationType | str,
        source_field: str,
        assertion_origin: str,
        status: str,
        target_reference: Mapping[str, object] | None = None,
        target: _PaddleBlockContext | None = None,
        details: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        if source.observation_id not in resolution_records:
            return None
        value: dict[str, object] = {
            "assertion_origin": assertion_origin,
            "relation_type": (
                relation_type.value
                if isinstance(relation_type, RelationType)
                else relation_type
            ),
            "source_field": source_field,
            "source_parser_native_id": source.native_id,
            "status": status,
        }
        if target_reference is not None:
            value["target_reference"] = dict(target_reference)
        if target is not None:
            value["target_page_index"] = target.page_index
            value["target_parser_native_id"] = target.native_id
            if target.observation_id is not None:
                value["target_observation_id"] = target.observation_id
        if details:
            value.update(details)
        resolution_records[source.observation_id].append(value)
        return value

    def add(
        source: _PaddleBlockContext,
        target: _PaddleBlockContext | None,
        relation_type: RelationType,
        source_field: str,
        assertion_origin: str,
        *,
        target_reference: Mapping[str, object],
        confidence: float | None,
        unresolved_status: str = "unresolved",
        details: Mapping[str, object] | None = None,
    ) -> None:
        status = unresolved_status
        if target is not None:
            if target.observation_id is None:
                status = "target_not_translated"
            elif target.observation_id == source.observation_id:
                status = "self_target_omitted"
            else:
                status = "resolved"
        audit = record(
            source,
            relation_type=relation_type,
            source_field=source_field,
            assertion_origin=assertion_origin,
            status=status,
            target_reference=target_reference,
            target=target,
            details=details,
        )
        if (
            audit is None
            or status != "resolved"
            or source.observation_id is None
            or target is None
            or target.observation_id is None
        ):
            return
        relation_metadata = {
            key: value
            for key, value in audit.items()
            if key
            not in {
                "status",
                "target_observation_id",
            }
        }
        assertions[source.observation_id].append(
            ObservationRelation(
                relation_type=relation_type,
                target_observation_id=target.observation_id,
                confidence=confidence,
                metadata=relation_metadata,
            )
        )

    def resolve_candidates(
        candidates: Sequence[_PaddleBlockContext],
    ) -> tuple[_PaddleBlockContext | None, str]:
        if not candidates:
            return None, "unresolved"
        if len(candidates) > 1:
            return None, "ambiguous_native_id"
        return candidates[0], "resolved"

    # A runner can retain explicit relations which Paddle's public JSON does
    # not expose.  The ID namespace and page are always declared; there is no
    # text matching or coordinate fallback.
    for source in contexts:
        raw_descriptors = source.block.get(_STRUCTURAL_RELATIONS_FIELD)
        if raw_descriptors is None or source.observation_id is None:
            continue
        source_field = _STRUCTURAL_RELATIONS_FIELD
        if not isinstance(raw_descriptors, Sequence) or isinstance(
            raw_descriptors, (str, bytes, bytearray)
        ):
            record(
                source,
                relation_type="invalid",
                source_field=source_field,
                assertion_origin="paddleocr_vl_explicit_id_relation",
                status="invalid_descriptor_array",
                details={"descriptor_value": json_ready(freeze_json(raw_descriptors))},
            )
            continue
        for descriptor_index, raw_descriptor in enumerate(raw_descriptors):
            descriptor_field = f"{source_field}[{descriptor_index}]"
            lineage_value = json_ready(freeze_json(raw_descriptor))
            try:
                descriptor = require_mapping(raw_descriptor, descriptor_field)
                reject_unknown_fields(
                    descriptor,
                    _STRUCTURAL_RELATION_FIELDS,
                    dto="PaddleOCRVLStructuralRelation",
                )
                try:
                    relation_type = RelationType(descriptor.get("relation_type"))
                except (TypeError, ValueError) as exc:
                    raise ContractValidationError(
                        f"{descriptor_field}.relation_type is unsupported"
                    ) from exc
                target_fields = tuple(
                    field_name
                    for field_name in (
                        "target_block_id",
                        "target_parser_native_id",
                        "target_global_block_id",
                    )
                    if descriptor.get(field_name) is not None
                )
                if len(target_fields) != 1:
                    raise ContractValidationError(
                        f"{descriptor_field} requires exactly one target ID field"
                    )
                target_field = target_fields[0]
                raw_target = _native_reference_scalar(
                    descriptor[target_field], f"{descriptor_field}.{target_field}"
                )
                target_page = source.page_index
                if descriptor.get("target_page_index") is not None:
                    if target_field == "target_global_block_id":
                        raise ContractValidationError(
                            f"{descriptor_field}.target_page_index cannot qualify a global ID"
                        )
                    target_page = _page_index(
                        descriptor["target_page_index"],
                        f"{descriptor_field}.target_page_index",
                    )
                raw_confidence = descriptor.get("confidence")
                confidence = (
                    None
                    if raw_confidence is None
                    else _probability(raw_confidence, f"{descriptor_field}.confidence")
                )
                key = _native_reference_key(raw_target)
                if target_field == "target_block_id":
                    candidates = by_page_block_id.get((target_page, key), ())
                elif target_field == "target_parser_native_id":
                    candidates = by_page_parser_native_id.get((target_page, key), ())
                else:
                    candidates = by_global_block_id.get(key, ())
                target, status = resolve_candidates(candidates)
                candidate_native_ids = sorted(
                    item.native_id for item in candidates
                )
                target_reference: dict[str, object] = {
                    "namespace": target_field.removeprefix("target_"),
                    "value": raw_target,
                }
                if target_field != "target_global_block_id":
                    target_reference["page_index"] = target_page
                add(
                    source,
                    target,
                    relation_type,
                    f"{descriptor_field}.{target_field}",
                    "paddleocr_vl_explicit_id_relation",
                    target_reference=target_reference,
                    confidence=confidence,
                    unresolved_status=status,
                    details={
                        "descriptor_index": descriptor_index,
                        "descriptor": lineage_value,
                        **(
                            {"candidate_parser_native_ids": candidate_native_ids}
                            if len(candidate_native_ids) > 1
                            else {}
                        ),
                    },
                )
            except (ContractValidationError, TypeError, ValueError) as exc:
                record(
                    source,
                    relation_type=(
                        str(raw_descriptor.get("relation_type", "invalid"))
                        if isinstance(raw_descriptor, Mapping)
                        else "invalid"
                    ),
                    source_field=descriptor_field,
                    assertion_origin="paddleocr_vl_explicit_id_relation",
                    status="invalid_descriptor",
                    details={
                        "descriptor": lineage_value,
                        "validation_error": str(exc),
                    },
                )

    # A page-level reading chain is valid only when every accepted observation
    # has one unique, explicit native order and layout detection is attested.
    # A partial order is not a license to link ordered islands across omitted
    # blocks, and duplicates are never broken by IDs, list position, or
    # geometry.
    for page_index, page_contexts in by_page.items():
        accepted_contexts = [
            context
            for context in page_contexts
            if context.observation_id in observation_by_id
        ]
        if not accepted_contexts:
            continue
        if layout_order_attested_by_page.get(page_index) is not True:
            for source in accepted_contexts:
                record(
                    source,
                    relation_type="reading_order",
                    source_field="block_order",
                    assertion_origin="paddleocr_vl_complete_layout_order",
                    status="layout_detection_not_attested",
                    details={"native_order": source.block.get("block_order")},
                )
            continue
        ordered_groups: dict[float, list[_PaddleBlockContext]] = {}
        unordered: list[_PaddleBlockContext] = []
        # Retain untranslated native blocks as gaps in an otherwise accepted
        # order chain.  They cannot become endpoints, and ordered neighbors are
        # not rewired around lost evidence.
        for context in page_contexts:
            try:
                native_order = _reading_order(context.block.get("block_order"))
            except ContractValidationError:
                native_order = None
            if native_order is not None:
                ordered_groups.setdefault(native_order, []).append(context)
            else:
                unordered.append(context)
        if unordered:
            missing_native_ids = sorted(item.native_id for item in unordered)
            for source in accepted_contexts:
                record(
                    source,
                    relation_type="reading_order",
                    source_field="block_order",
                    assertion_origin="paddleocr_vl_complete_layout_order",
                    status="incomplete_page_reading_order",
                    details={
                        "missing_parser_native_ids": missing_native_ids,
                        "accepted_observation_count": len(accepted_contexts),
                    },
                )
            continue
        grouped_order = sorted(ordered_groups.items(), key=lambda item: item[0])
        duplicates = {
            native_order: values
            for native_order, values in grouped_order
            if len(values) > 1
        }
        if duplicates:
            duplicate_details = {
                str(native_order): sorted(item.native_id for item in values)
                for native_order, values in duplicates.items()
            }
            for source in accepted_contexts:
                source_order = _reading_order(source.block.get("block_order"))
                source_candidates = duplicates.get(source_order, ())
                record(
                    source,
                    relation_type="reading_order",
                    source_field="block_order",
                    assertion_origin="paddleocr_vl_complete_layout_order",
                    status="ambiguous_duplicate_block_order",
                    details={
                        "duplicate_orders": duplicate_details,
                        **(
                            {
                                "native_order": source_order,
                                "candidate_parser_native_ids": sorted(
                                    item.native_id for item in source_candidates
                                ),
                            }
                            if source_candidates
                            else {}
                        ),
                    },
                )
            continue
        for (left_order, left_values), (right_order, right_values) in zip(
            grouped_order, grouped_order[1:], strict=False
        ):
            left = left_values[0]
            right = right_values[0]
            add(
                left,
                right,
                RelationType.READING_NEXT,
                "block_order",
                "paddleocr_vl_complete_layout_order",
                target_reference={
                    "namespace": "block_order",
                    "page_index": page_index,
                    "value": right_order,
                },
                confidence=1.0,
                details={
                    "source_native_order": left_order,
                    "target_native_order": right_order,
                },
            )
            add(
                right,
                left,
                RelationType.READING_PREVIOUS,
                "block_order",
                "paddleocr_vl_complete_layout_order",
                target_reference={
                    "namespace": "block_order",
                    "page_index": page_index,
                    "value": left_order,
                },
                confidence=1.0,
                details={
                    "source_native_order": right_order,
                    "target_native_order": left_order,
                },
            )

    # ``restructure_pages(merge_tables=True)`` assigns every block a global
    # ID and changes only a merged continuation table's global_group_id to the
    # retained table's global_block_id.  That is native continuation evidence;
    # ordinary group_id/global_group_id equality is deliberately not mapped to
    # parentage or semantic similarity.
    for source in contexts:
        if source.observation_id is None:
            continue
        source_observation = observation_by_id[source.observation_id]
        raw_global_id = source.block.get("global_block_id")
        raw_group_id = source.block.get("global_group_id")
        if raw_global_id is None and raw_group_id is None:
            continue
        try:
            global_id = _native_reference_scalar(
                raw_global_id, "global_block_id"
            )
            group_id = _native_reference_scalar(
                raw_group_id, "global_group_id"
            )
        except ContractValidationError as exc:
            record(
                source,
                relation_type="table_continuation",
                source_field="global_group_id",
                assertion_origin="paddleocr_vl_restructured_table_group",
                status="invalid_global_group_reference",
                details={"validation_error": str(exc)},
            )
            continue
        if _native_reference_key(global_id) == _native_reference_key(group_id):
            continue
        target_candidates = by_global_block_id.get(
            _native_reference_key(group_id), ()
        )
        target, status = resolve_candidates(target_candidates)
        candidate_details = (
            {
                "candidate_parser_native_ids": sorted(
                    item.native_id for item in target_candidates
                )
            }
            if len(target_candidates) > 1
            else {}
        )
        if source_observation.block_type is not BlockType.TABLE:
            record(
                source,
                relation_type="table_continuation",
                source_field="global_group_id",
                assertion_origin="paddleocr_vl_restructured_table_group",
                status="non_table_group_not_mapped",
                target_reference={
                    "namespace": "global_block_id",
                    "value": group_id,
                },
                target=target,
                details=candidate_details,
            )
            continue
        if (
            target is not None
            and target.observation_id is not None
            and observation_by_id[target.observation_id].block_type is not BlockType.TABLE
        ):
            record(
                source,
                relation_type=RelationType.TABLE_CONTINUED_FROM,
                source_field="global_group_id",
                assertion_origin="paddleocr_vl_restructured_table_group",
                status="target_type_mismatch",
                target_reference={
                    "namespace": "global_block_id",
                    "value": group_id,
                },
                target=target,
                details={"source_global_block_id": global_id},
            )
            continue
        else:
            target_for_edge = target
        reference = {"namespace": "global_block_id", "value": group_id}
        add(
            source,
            target_for_edge,
            RelationType.TABLE_CONTINUED_FROM,
            "global_group_id",
            "paddleocr_vl_restructured_table_group",
            target_reference=reference,
            confidence=1.0,
            unresolved_status=status,
            details={"source_global_block_id": global_id, **candidate_details},
        )
        if (
            target_for_edge is not None
            and target_for_edge.observation_id is not None
            and source.observation_id is not None
        ):
            add(
                target_for_edge,
                source,
                RelationType.TABLE_CONTINUES,
                "global_group_id",
                "paddleocr_vl_restructured_table_group_inverse",
                target_reference={
                    "namespace": "global_block_id",
                    "value": global_id,
                },
                confidence=1.0,
                details={"continuation_global_group_id": group_id},
            )

    enriched: list[ParserBlockObservation] = []
    for observation in observations:
        relations = tuple(
            sorted(
                assertions[observation.observation_id],
                key=lambda item: (
                    item.relation_type.value,
                    item.target_observation_id,
                    item.relation_id,
                ),
            )
        )
        metadata = dict(observation.metadata)
        records = resolution_records[observation.observation_id]
        if records:
            metadata["paddle_structural_ref_resolution"] = sorted(
                records,
                key=lambda item: (
                    str(item.get("source_field", "")),
                    str(item.get("relation_type", "")),
                    str(item.get("target_reference", "")),
                    str(item.get("status", "")),
                ),
            )
        enriched.append(
            replace(observation, relations=relations, metadata=metadata)
        )
    return tuple(enriched)


def _translate_block(
    request: ParseRequest,
    page_payload: Mapping[str, object],
    block: Mapping[str, object],
    *,
    page_index: int,
    width: float,
    height: float,
    native_id: str,
    layout_scores: Mapping[tuple[str, tuple[float, ...]], tuple[float, ...]],
    parser_version: str,
    model_version: str,
    config_sha256: str,
    default_raw_artifact_uri: str | None,
) -> ParserBlockObservation:
    page = request.page_by_index[page_index]
    label = require_non_empty(block.get("block_label"), "block_label").strip().lower()
    geometry = _geometry(block, width=width, height=height)
    confidence = _confidence(
        block, label=label, geometry=geometry, layout_scores=layout_scores
    )
    block_type, subtype = _block_type(label)
    language = request.options.languages[0] if len(request.options.languages) == 1 else None
    settings = require_mapping(page_payload.get("model_settings", {}), "model_settings")
    representations = _representations(
        block,
        block_type=block_type,
        label=label,
        confidence=confidence,
        language=language,
        use_ocr_for_image_block=_use_ocr_for_image_block(settings),
    )
    semantic_substructures = []
    representation_by_kind = {
        representation.kind: representation for representation in representations
    }
    if block.get("table_grid") is not None:
        table_representation = representation_by_kind.get(
            RepresentationKind.TABLE_GRID
        )
        if table_representation is None:
            raise ContractValidationError(
                "table_grid requires a TABLE_GRID representation"
            )
        semantic_substructures.append(
            explicit_table_grid_substructure(
                block["table_grid"],
                table_representation,
                source_field="table_grid",
                parser_name=PARSER_NAME,
                confidence=confidence,
            )
        )
    if block.get("diagram_graph") is not None:
        diagram_representation = representation_by_kind.get(
            RepresentationKind.DIAGRAM_GRAPH
        )
        if diagram_representation is None:
            raise ContractValidationError(
                "diagram_graph requires a DIAGRAM_GRAPH representation"
            )
        normalized_subtype = (subtype or label).casefold().replace("-", "_")
        semantic_substructures.append(
            explicit_diagram_substructure(
                block["diagram_graph"],
                diagram_representation,
                source_field="diagram_graph",
                parser_name=PARSER_NAME,
                confidence=confidence,
                flowchart=normalized_subtype
                in {"flow", "flowchart", "flow_chart", "workflow"},
            )
        )
    if block.get("mathml") is not None:
        mathml_representation = representation_by_kind.get(RepresentationKind.MATHML)
        if mathml_representation is None:
            raise ContractValidationError("mathml requires a MATHML representation")
        semantic_substructures.append(
            mathml_formula_substructure(
                mathml_representation.content,  # type: ignore[arg-type]
                mathml_representation,
                source_field="mathml",
                parser_name=PARSER_NAME,
                confidence=confidence,
            )
        )
    assets = _assets(
        page_payload, block, label=label, native_id=native_id, geometry=geometry
    )
    raw_artifact_uri = _optional_text(
        block.get(
            "raw_artifact_uri",
            page_payload.get("raw_artifact_uri", default_raw_artifact_uri),
        ),
        "raw_artifact_uri",
    )
    group_id = block.get("group_id")
    if group_id is not None and (
        isinstance(group_id, bool) or not isinstance(group_id, (str, int, float))
    ):
        raise ContractValidationError("group_id must be a JSON scalar")
    metadata: dict[str, object] = {
        "native_label": label,
        "normalized_block_sha256": content_sha256(block),
        "model_settings_sha256": content_sha256(settings),
        "translation": TRANSLATION_NAME,
    }
    if group_id is not None:
        metadata["group_id"] = group_id
    for field_name in ("global_block_id", "global_group_id"):
        if block.get(field_name) is not None:
            metadata[field_name] = _native_reference_scalar(
                block[field_name], field_name
            )
    return ParserBlockObservation(
        document_version_id=request.document.document_version_id,
        page_id=page.page_id,
        page_index=page_index,
        provenance=ParserProvenance(
            parser_name=PARSER_NAME,
            parser_version=parser_version,
            model_version=model_version,
            config_sha256=config_sha256,
            parser_native_id=native_id,
            raw_artifact_uri=raw_artifact_uri,
            transformations=(TRANSLATION_NAME,),
        ),
        block_type=block_type,
        semantic_subtype=subtype,
        geometry=geometry,
        reading_order=_reading_order(block.get("block_order")),
        representations=representations,
        semantic_substructures=tuple(semantic_substructures),
        assets=assets,
        confidence=confidence,
        metadata=metadata,
    )


def _issue(
    message: str,
    *,
    page_index: int | None = None,
    native_id: str | None = None,
    details: Mapping[str, object] | None = None,
    code: ParseErrorCode = ParseErrorCode.INVALID_NORMALIZED_BLOCK,
    classification: ErrorClassification = ErrorClassification.PERMANENT,
) -> ParseIssue:
    return ParseIssue(
        code=code,
        classification=classification,
        severity=IssueSeverity.ERROR,
        message=message,
        page_index=page_index,
        parser_native_id=native_id,
        details=details or {"translation": TRANSLATION_NAME},
    )


def translate_page_results(
    request: ParseRequest,
    page_results: Iterable[Mapping[str, object]] | Mapping[str, object],
    *,
    parser_version: str,
    model_version: str,
    config_sha256: str,
    raw_artifact_uri: str | None = None,
) -> tuple[tuple[ParserBlockObservation, ...], tuple[ParseIssue, ...]]:
    """Translate injected v1.6 page results into stable parser observations.

    Page and block corruption is isolated into typed issues.  Iteration/runtime
    failures are retryable and preserve already translated observations, which
    lets the caller report a truthful partial result.
    """

    require_non_empty(parser_version, "parser_version")
    require_non_empty(model_version, "model_version")
    require_sha256(config_sha256, "config_sha256")
    if raw_artifact_uri is not None:
        require_non_empty(raw_artifact_uri, "raw_artifact_uri")

    source: Iterable[Mapping[str, object]]
    source = (page_results,) if isinstance(page_results, Mapping) else page_results
    if isinstance(source, (str, bytes, bytearray)):
        raise ContractValidationError("page_results must be an iterable of JSON objects")
    try:
        iterator = iter(source)
    except TypeError as exc:
        raise ContractValidationError("page_results must be iterable") from exc

    observations: list[ParserBlockObservation] = []
    contexts: list[_PaddleBlockContext] = []
    issues: list[ParseIssue] = []
    seen_pages: set[int] = set()
    seen_native_ids: set[str] = set()
    layout_order_attested_by_page: dict[int, bool] = {}
    result_index = 0
    while True:
        try:
            raw_page = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            issues.append(
                _issue(
                    "PaddleOCR-VL page-result stream failed",
                    code=ParseErrorCode.PARSER_FAILURE,
                    classification=ErrorClassification.RETRYABLE,
                    details={
                        "exception_type": type(exc).__name__,
                        "page_result_index": result_index,
                        "translation": TRANSLATION_NAME,
                    },
                )
            )
            break

        current_index = result_index
        result_index += 1
        page_index: int | None = None
        try:
            page_payload = require_mapping(raw_page, f"page_results[{current_index}]")
            page_index = _page_index(page_payload.get("page_index"))
            if page_index in seen_pages:
                raise ContractValidationError(f"duplicate page result for page {page_index}")
            seen_pages.add(page_index)
            if page_index not in request.page_by_index:
                raise ContractValidationError(
                    f"page_index {page_index} is outside requested pages"
                )
            _validate_source_binding(request, page_payload)
            page_count = page_payload.get("page_count")
            if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
                raise ContractValidationError("page_count must be a positive integer")
            if (
                request.document.page_count is not None
                and page_count != request.document.page_count
            ):
                raise ContractValidationError(
                    "page_count differs from the requested document"
                )
            width = _positive_number(page_payload.get("width"), "page.width")
            height = _positive_number(page_payload.get("height"), "page.height")
            blocks = _sequence(
                page_payload.get("parsing_res_list"), "parsing_res_list"
            )
            settings = require_mapping(
                page_payload.get("model_settings", {}), "model_settings"
            )
            layout_order_attested_by_page[page_index] = (
                settings.get("use_layout_detection") is True
            )
            layout_scores, layout_errors = _layout_confidences(
                page_payload, width=width, height=height
            )
            for message in layout_errors:
                issues.append(
                    _issue(
                        message,
                        page_index=page_index,
                        details={
                            "page_result_index": current_index,
                            "translation": TRANSLATION_NAME,
                        },
                    )
                )
        except (ContractValidationError, TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    f"PaddleOCR-VL page result {current_index} is invalid: {exc}",
                    page_index=page_index,
                    details={
                        "page_result_index": current_index,
                        "translation": TRANSLATION_NAME,
                    },
                )
            )
            continue

        for block_index, raw_block in enumerate(blocks):
            native_id: str | None = None
            context: _PaddleBlockContext | None = None
            try:
                block = require_mapping(
                    raw_block, f"parsing_res_list[{block_index}]"
                )
                native_id = _native_id(page_index, block)
                context = _PaddleBlockContext(
                    page_index=page_index,
                    block_index=block_index,
                    block=block,
                    native_id=native_id,
                )
                if native_id in seen_native_ids:
                    raise ContractValidationError(
                        f"duplicate parser-native ID {native_id!r}"
                    )
                seen_native_ids.add(native_id)
                observation = _translate_block(
                    request,
                    page_payload,
                    block,
                    page_index=page_index,
                    width=width,
                    height=height,
                    native_id=native_id,
                    layout_scores=layout_scores,
                    parser_version=parser_version,
                    model_version=model_version,
                    config_sha256=config_sha256,
                    default_raw_artifact_uri=raw_artifact_uri,
                )
                observations.append(observation)
                context = replace(
                    context, observation_id=observation.observation_id
                )
            except (ContractValidationError, TypeError, ValueError) as exc:
                issues.append(
                    _issue(
                        f"PaddleOCR-VL block {block_index} on page {page_index} is invalid: {exc}",
                        page_index=page_index,
                        native_id=native_id,
                        details={
                            "page_result_index": current_index,
                            "block_index": block_index,
                            "translation": TRANSLATION_NAME,
                        },
                    )
                )
            finally:
                if context is not None:
                    contexts.append(context)

    for missing_page in sorted(set(request.page_by_index) - seen_pages):
        issues.append(
            _issue(
                f"PaddleOCR-VL returned no result for requested page {missing_page}",
                page_index=missing_page,
                details={"missing_page": missing_page, "translation": TRANSLATION_NAME},
            )
        )

    observations = list(
        _attach_structural_relations(
            observations,
            contexts,
            layout_order_attested_by_page=layout_order_attested_by_page,
        )
    )

    def observation_key(item: ParserBlockObservation) -> tuple[object, ...]:
        bbox = item.geometry.bbox if item.geometry is not None else None
        return (
            item.page_index,
            float("inf") if item.reading_order is None else item.reading_order,
            float("inf") if bbox is None else bbox.y0,
            float("inf") if bbox is None else bbox.x0,
            item.observation_id,
        )

    observations.sort(key=observation_key)
    issues.sort(key=lambda item: item.issue_id)
    return tuple(observations), tuple(issues)


class PaddleOCRVLPageAdapter:
    """Bridge from injected PaddleOCR-VL page JSON to ``ParserAdapter``."""

    def __init__(
        self,
        page_loader: PaddlePageLoader,
        *,
        parser_version: str,
        model_version: str,
        config: Mapping[str, object],
        raw_artifact_uri: str | None = None,
    ) -> None:
        if not callable(page_loader):
            raise TypeError("page_loader must be callable")
        require_non_empty(parser_version, "parser_version")
        require_non_empty(model_version, "model_version")
        if raw_artifact_uri is not None:
            require_non_empty(raw_artifact_uri, "raw_artifact_uri")
        self._loader = page_loader
        self._parser_version = parser_version
        self._model_version = model_version
        self._config_sha256 = content_sha256(config)
        self._raw_artifact_uri = raw_artifact_uri
        self._manifest = ParserCapabilityManifest(
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            parser_name=PARSER_NAME,
            parser_version=parser_version,
            model_version=model_version,
            supported_mime_types=("application/pdf",),
            modalities=(
                ParserModality.NATIVE_TEXT,
                ParserModality.OCR_TEXT,
                ParserModality.LAYOUT,
                ParserModality.TABLE,
                ParserModality.FORMULA,
                ParserModality.IMAGE,
                ParserModality.CHART,
                ParserModality.DIAGRAM,
            ),
            block_types=tuple(BlockType),
            accelerator=AcceleratorKind.REMOTE,
            supports_page_selection=True,
            supports_region_selection=False,
            supports_batch=True,
            deterministic=False,
            metadata={
                "adapter_contract_version": ADAPTER_VERSION,
                "input_shape": "paddleocr_vl_v1.6_page_result.v1",
                "translation": TRANSLATION_NAME,
                "vendor_imports": False,
                "asset_resolution": "injected_only",
                "structural_relations": (
                    "explicit_id_descriptors+attested_layout_order+"
                    "restructured_table_groups"
                ),
                "geometry_relation_inference": False,
            },
        )

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def parse(self, request: ParseRequest) -> ParseResult:
        started = perf_counter()
        artifacts: tuple[ParserArtifactReceipt, ...] = ()
        try:
            loaded = self._loader(request)
            raw_pages = getattr(loaded, "pages", loaded)
            raw_artifacts = getattr(loaded, "artifact_receipts", ())
            # Preserve lazy injected streams: translate_page_results isolates a
            # mid-stream transport failure and retains preceding page evidence.
            pages = raw_pages
            candidate_artifacts = tuple(raw_artifacts)
            if any(
                not isinstance(item, ParserArtifactReceipt)
                for item in candidate_artifacts
            ):
                raise ContractValidationError(
                    "Paddle page loader artifact_receipts must contain "
                    "ParserArtifactReceipt DTOs"
                )
            if any(
                item.document_version_id != request.document.document_version_id
                or item.source_sha256 != request.document.content_sha256
                for item in candidate_artifacts
            ):
                raise ContractValidationError(
                    "Paddle page loader artifact_receipts belong to another source"
                )
            artifacts = candidate_artifacts
        except ParserExecutionError as exc:
            return ParseResult(
                request_id=request.request_id,
                adapter_name=ADAPTER_NAME,
                adapter_version=ADAPTER_VERSION,
                status=ParseStatus.FAILED,
                issues=(exc.issue,),
                artifacts=exc.artifacts,
                duration_ms=(perf_counter() - started) * 1000,
                metrics={"valid_observations": 0, "invalid_results": 1},
            )
        except Exception as exc:
            issue = _issue(
                "PaddleOCR-VL page loader failed",
                code=ParseErrorCode.PARSER_FAILURE,
                classification=ErrorClassification.RETRYABLE,
                details={"exception_type": type(exc).__name__},
            )
            return ParseResult(
                request_id=request.request_id,
                adapter_name=ADAPTER_NAME,
                adapter_version=ADAPTER_VERSION,
                status=ParseStatus.FAILED,
                issues=(issue,),
                artifacts=artifacts,
                duration_ms=(perf_counter() - started) * 1000,
                metrics={"valid_observations": 0, "invalid_results": 1},
            )

        try:
            observations, issues = translate_page_results(
                request,
                pages,
                parser_version=self._parser_version,
                model_version=self._model_version,
                config_sha256=self._config_sha256,
                raw_artifact_uri=self._raw_artifact_uri,
            )
        except Exception as exc:
            issue = _issue(
                "PaddleOCR-VL normalized output could not be consumed",
                code=ParseErrorCode.CONTRACT_VIOLATION,
                classification=ErrorClassification.PERMANENT,
                details={"exception_type": type(exc).__name__},
            )
            return ParseResult(
                request_id=request.request_id,
                adapter_name=ADAPTER_NAME,
                adapter_version=ADAPTER_VERSION,
                status=ParseStatus.FAILED,
                issues=(issue,),
                artifacts=artifacts,
                duration_ms=(perf_counter() - started) * 1000,
                metrics={"valid_observations": 0, "invalid_results": 1},
            )

        errors = sum(issue.severity == IssueSeverity.ERROR for issue in issues)
        if errors and observations:
            status = ParseStatus.PARTIAL
        elif errors:
            status = ParseStatus.FAILED
        else:
            status = ParseStatus.SUCCEEDED
        return ParseResult(
            request_id=request.request_id,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            status=status,
            observations=observations if status != ParseStatus.FAILED else (),
            issues=issues,
            artifacts=artifacts,
            duration_ms=(perf_counter() - started) * 1000,
            metrics={
                "valid_observations": len(observations),
                "invalid_results": errors,
            },
        )


__all__ = [
    "ADAPTER_NAME",
    "ADAPTER_VERSION",
    "PARSER_NAME",
    "PaddleOCRVLPageAdapter",
    "PaddlePageLoader",
    "TRANSLATION_NAME",
    "translate_page_results",
]
