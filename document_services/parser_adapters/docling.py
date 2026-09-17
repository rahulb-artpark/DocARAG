"""Dependency-free translator for Docling's serialized document export.

The adapter deliberately consumes the JSON-compatible mapping returned by
``DoclingDocument.export_to_dict()`` (or read from ``export_to_json()``).  It
does not import Docling.  A composition root running in the isolated Docling
environment injects a loader which performs conversion and returns that
mapping; an RPC/file loader can provide the same boundary.

Docling uses one-based ``page_no`` values and records reading order through
JSON references in ``body`` and ``groups``.  This module validates those
semantics explicitly and translates them into DocARAG's zero-based, source-
bound parser observations without flattening tables, formulae, or images into
plain text.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
from html import escape as escape_html
import math
import mimetypes
from time import perf_counter
from urllib.parse import unquote_to_bytes, urldefrag

from document_services.domain.base import (
    ContractValidationError,
    content_sha256,
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
    CoordinateOrigin,
    CoordinateSpace,
    ObservationRelation,
    ParserBlockObservation,
    ParserProvenance,
    Point,
    Representation,
    RepresentationKind,
    RelationType,
    SourceGeometry,
    MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE,
    MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE,
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
    ParserCapabilityManifest,
    ParserExecutionError,
    ParserModality,
)
from document_services.parser_adapters.semantic_normalization import (
    docling_table_substructure,
    preflight_explicit_json,
)


DoclingExportLoader = Callable[[ParseRequest], Mapping[str, object]]

ADAPTER_NAME = "docling-export"
ADAPTER_VERSION = "1.1.0"
PARSER_NAME = "docling"
TRANSLATION_NAME = "docling_document_export_to_observation.v3"
DEFAULT_MAX_EMBEDDED_ASSET_BYTES = 16 * 1024 * 1024

_COLLECTIONS = ("texts", "tables", "pictures")
_CHART_CLASSES = frozenset(
    {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "box_plot"}
)
_DIAGRAM_CLASSES = frozenset(
    {
        "flow_chart",
        "engineering_drawing",
        "chemistry_structure",
        "geographical_map",
        "topographical_map",
    }
)


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _finite_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractValidationError(f"{field} must be a finite number")
    return float(value)


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{field} must be a positive integer")
    return value


def _probability(value: object, field: str) -> float:
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{field} must be between 0 and 1")
    return result


def _reference(value: object, field: str) -> str:
    if isinstance(value, str):
        return require_non_empty(value, field)
    mapping = require_mapping(value, field)
    return require_non_empty(mapping.get("$ref"), f"{field}.$ref")


def _parent_ref(item: Mapping[str, object]) -> str | None:
    parent = item.get("parent")
    return _reference(parent, "item.parent") if parent is not None else None


def _reading_order(export: Mapping[str, object]) -> dict[str, float]:
    """Flatten body/group references while rejecting cycles deterministically."""

    groups: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(
        require_sequence(export.get("groups", ()), "docling.groups")
    ):
        group = require_mapping(value, f"docling.groups[{index}]")
        self_ref = require_non_empty(
            group.get("self_ref"), f"docling.groups[{index}].self_ref"
        )
        if self_ref in groups:
            raise ContractValidationError(f"duplicate Docling group self_ref: {self_ref}")
        groups[self_ref] = group

    body = require_mapping(export.get("body", {}), "docling.body")
    order: dict[str, float] = {}
    active: set[str] = set()
    cursor = 0

    def walk_reference(reference: str) -> None:
        nonlocal cursor
        group = groups.get(reference)
        if group is None:
            if reference not in order:
                order[reference] = float(cursor)
                cursor += 1
            return
        if reference in active:
            raise ContractValidationError(
                f"cycle in Docling reading-order groups at {reference}"
            )
        active.add(reference)
        children = require_sequence(
            group.get("children", ()), f"Docling group {reference}.children"
        )
        for child_index, child in enumerate(children):
            walk_reference(
                _reference(child, f"Docling group {reference}.children[{child_index}]")
            )
        active.remove(reference)

    for index, child in enumerate(
        require_sequence(body.get("children", ()), "docling.body.children")
    ):
        walk_reference(_reference(child, f"docling.body.children[{index}]"))
    return order


def _text_index(export: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(
        require_sequence(export.get("texts", ()), "docling.texts")
    ):
        if not isinstance(value, Mapping):
            continue
        self_ref = value.get("self_ref")
        if isinstance(self_ref, str) and self_ref.strip() and self_ref not in result:
            result[self_ref] = value
    return result


def _resolved_texts(
    values: object,
    *,
    text_by_ref: Mapping[str, Mapping[str, object]],
    field: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    texts: list[str] = []
    refs: list[str] = []
    for index, value in enumerate(require_sequence(values, field)):
        if isinstance(value, str):
            direct = _text(value)
            if direct is not None:
                texts.append(direct)
            continue
        reference = _reference(value, f"{field}[{index}]")
        target = text_by_ref.get(reference)
        if target is None:
            raise ContractValidationError(
                f"{field}[{index}] references missing text item {reference}"
            )
        resolved = _text(target.get("text")) or _text(target.get("orig"))
        if resolved is None:
            raise ContractValidationError(
                f"{field}[{index}] references text item without content"
            )
        texts.append(resolved)
        refs.append(reference)
    return tuple(texts), tuple(refs)


def _item_text(item: Mapping[str, object], provenance: Mapping[str, object]) -> str | None:
    value = _text(item.get("text")) or _text(item.get("orig"))
    if value is None:
        return None
    charspan = provenance.get("charspan")
    if charspan is None:
        return value
    values = require_sequence(charspan, "provenance.charspan")
    if len(values) != 2:
        raise ContractValidationError("provenance.charspan must contain start and end")
    start = _non_negative_integer(values[0], "provenance.charspan[0]")
    end = _non_negative_integer(values[1], "provenance.charspan[1]")
    if end < start:
        raise ContractValidationError("provenance.charspan end precedes its start")
    # Docling char spans refer to the parser's original token/string stream.
    # Formula enrichment and text normalization can change the length of
    # ``text`` relative to that stream, so using the offsets to slice the final
    # value truncates normalized formula content and can reject correct text.
    # Preserve the offsets as source metadata and retain the full value.
    return value


def _page_index(provenance: Mapping[str, object]) -> int:
    page_no = provenance.get("page_no")
    normalized_index = provenance.get("page_index")
    if page_no is None and normalized_index is None:
        raise ContractValidationError("Docling provenance requires page_no")
    if page_no is not None:
        one_based = _positive_integer(page_no, "provenance.page_no")
        result = one_based - 1
        if normalized_index is not None and normalized_index != result:
            raise ContractValidationError(
                "provenance page_no and page_index identify different pages"
            )
        return result
    return _non_negative_integer(normalized_index, "provenance.page_index")


def _docling_coordinate_origin(value: object, field: str) -> CoordinateOrigin:
    raw = require_non_empty(value, field)
    normalized = raw.strip().replace("-", "_").upper()
    if normalized in {"TOPLEFT", "TOP_LEFT"}:
        return CoordinateOrigin.TOP_LEFT
    if normalized in {"BOTTOMLEFT", "BOTTOM_LEFT"}:
        return CoordinateOrigin.BOTTOM_LEFT
    raise ContractValidationError(
        f"{field} has unsupported coordinate origin {raw!r}; "
        "expected TOPLEFT or BOTTOMLEFT"
    )


def _geometry(
    provenance: Mapping[str, object], request: ParseRequest, page_index: int
) -> tuple[SourceGeometry | None, str | None]:
    bbox_value = provenance.get("bbox")
    polygon_value = provenance.get("polygon", ())
    bbox: BoundingBox | None = None
    polygon: tuple[Point, ...] = ()
    raw_coord_origin: str | None = None
    coordinate_origin: CoordinateOrigin | None = None
    if bbox_value is not None:
        bbox_mapping = require_mapping(bbox_value, "provenance.bbox")
        raw_coord_origin = _text(bbox_mapping.get("coord_origin")) or _text(
            provenance.get("coord_origin")
        )
        coordinate_origin = _docling_coordinate_origin(
            raw_coord_origin, "provenance.bbox.coord_origin"
        )
        if all(key in bbox_mapping for key in ("l", "t", "r", "b")):
            left = _finite_number(bbox_mapping.get("l"), "provenance.bbox.l")
            top = _finite_number(bbox_mapping.get("t"), "provenance.bbox.t")
            right = _finite_number(bbox_mapping.get("r"), "provenance.bbox.r")
            bottom = _finite_number(bbox_mapping.get("b"), "provenance.bbox.b")
            raw_x = (left, right)
            raw_y = (top, bottom)
        elif all(key in bbox_mapping for key in ("x0", "y0", "x1", "y1")):
            raw_x = (
                _finite_number(bbox_mapping.get("x0"), "provenance.bbox.x0"),
                _finite_number(bbox_mapping.get("x1"), "provenance.bbox.x1"),
            )
            raw_y = (
                _finite_number(bbox_mapping.get("y0"), "provenance.bbox.y0"),
                _finite_number(bbox_mapping.get("y1"), "provenance.bbox.y1"),
            )
        else:
            raise ContractValidationError(
                "provenance.bbox requires l/t/r/b or x0/y0/x1/y1"
            )
        page = request.page_by_index[page_index]
        if coordinate_origin == CoordinateOrigin.BOTTOM_LEFT:
            if page.height is None:
                raise ContractValidationError(
                    "BOTTOMLEFT geometry requires authoritative request page height"
                )
            canonical_y = tuple(float(page.height) - value for value in raw_y)
        else:
            canonical_y = raw_y
        bbox = BoundingBox(
            min(raw_x), min(canonical_y), max(raw_x), max(canonical_y)
        )
    if polygon_value:
        polygon_origin_value = provenance.get("coord_origin")
        if coordinate_origin is None:
            raw_coord_origin = _text(polygon_origin_value)
            coordinate_origin = _docling_coordinate_origin(
                raw_coord_origin, "provenance.coord_origin"
            )
        elif polygon_origin_value is not None:
            polygon_origin = _docling_coordinate_origin(
                polygon_origin_value, "provenance.coord_origin"
            )
            if polygon_origin != coordinate_origin:
                raise ContractValidationError(
                    "bbox and polygon coordinate origins do not match"
                )
        page = request.page_by_index[page_index]
        points: list[Point] = []
        for index, value in enumerate(
            require_sequence(polygon_value, "provenance.polygon")
        ):
            if isinstance(value, Mapping):
                point = Point.from_dict(value)
            else:
                coordinates = require_sequence(
                    value, f"provenance.polygon[{index}]"
                )
                if len(coordinates) != 2:
                    raise ContractValidationError(
                        f"provenance.polygon[{index}] must contain x and y"
                    )
                point = Point(coordinates[0], coordinates[1])  # type: ignore[arg-type]
            if coordinate_origin == CoordinateOrigin.BOTTOM_LEFT:
                if page.height is None:
                    raise ContractValidationError(
                        "BOTTOMLEFT geometry requires authoritative request page height"
                    )
                point = Point(point.x, float(page.height) - float(point.y))
            points.append(point)
        polygon = tuple(points)
    if bbox is None and not polygon:
        return None, None
    page = request.page_by_index[page_index]
    return (
        SourceGeometry(
            coordinate_space=CoordinateSpace.PAGE_POINTS,
            bbox=bbox,
            polygon=polygon,
            page_width=page.width,
            page_height=page.height,
            coordinate_origin=CoordinateOrigin.TOP_LEFT,
            rotation_degrees=page.rotation_degrees,
        ),
        raw_coord_origin,
    )


def _predictions(item: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    candidates: object = ()
    annotations = require_sequence(item.get("annotations", ()), "item.annotations")
    for index, value in enumerate(annotations):
        annotation = require_mapping(value, f"item.annotations[{index}]")
        if annotation.get("kind") == "classification":
            candidates = annotation.get("predicted_classes", ())
            break
    if not candidates:
        meta = item.get("meta")
        if meta is not None:
            meta_mapping = require_mapping(meta, "item.meta")
            classification = meta_mapping.get("classification")
            if classification is not None:
                candidates = require_mapping(
                    classification, "item.meta.classification"
                ).get("predictions", ())

    result: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for index, value in enumerate(
        require_sequence(candidates, "item.classification.predictions")
    ):
        prediction = require_mapping(
            value, f"item.classification.predictions[{index}]"
        )
        class_name = require_non_empty(
            prediction.get("class_name"),
            f"item.classification.predictions[{index}].class_name",
        ).strip().lower()
        confidence = _probability(
            prediction.get("confidence"),
            f"item.classification.predictions[{index}].confidence",
        )
        if class_name not in seen:
            result.append({"class_name": class_name, "confidence": confidence})
            seen.add(class_name)
    result.sort(key=lambda item: (-float(item["confidence"]), str(item["class_name"])))
    return tuple(result)


def _block_type(
    collection: str,
    item: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
) -> tuple[BlockType, str | None]:
    label = str(item.get("label", "unspecified")).strip().lower()
    if collection == "tables" or label == "table":
        return BlockType.TABLE, None
    if collection == "pictures" or label in {"picture", "figure"}:
        subtype = (
            str(predictions[0]["class_name"]) if predictions else "picture"
        )
        if subtype in _CHART_CLASSES:
            return BlockType.CHART, subtype
        if subtype in _DIAGRAM_CLASSES:
            return BlockType.DIAGRAM, subtype
        return BlockType.FIGURE, subtype
    label_types = {
        "title": BlockType.TITLE,
        "section_header": BlockType.TITLE,
        "text": BlockType.PARAGRAPH,
        "paragraph": BlockType.PARAGRAPH,
        "list_item": BlockType.LIST_ITEM,
        "formula": BlockType.EQUATION,
        "equation": BlockType.EQUATION,
        "code": BlockType.CODE,
        "caption": BlockType.CAPTION,
        "footnote": BlockType.FOOTNOTE,
        "page_header": BlockType.HEADER,
        "header": BlockType.HEADER,
        "page_footer": BlockType.FOOTER,
        "footer": BlockType.FOOTER,
    }
    block_type = label_types.get(label, BlockType.UNKNOWN)
    subtype = None if block_type != BlockType.UNKNOWN else label
    return block_type, subtype


def _image_description(item: Mapping[str, object]) -> tuple[str | None, str | None]:
    annotations = require_sequence(item.get("annotations", ()), "item.annotations")
    for index, value in enumerate(annotations):
        annotation = require_mapping(value, f"item.annotations[{index}]")
        if annotation.get("kind") == "description":
            description = _text(annotation.get("text")) or _text(
                annotation.get("description")
            )
            if description is not None:
                return description, _text(annotation.get("provenance"))
    meta = item.get("meta")
    if meta is not None:
        description = require_mapping(meta, "item.meta").get("description")
        if description is not None:
            description_mapping = require_mapping(description, "item.meta.description")
            return (
                _text(description_mapping.get("text")),
                _text(description_mapping.get("created_by")),
            )
    return _text(item.get("description")), None


def _validate_embedded_asset_limit(value: object) -> int:
    return _positive_integer(value, "max_embedded_asset_bytes")


def _base64_decoded_size(payload: str) -> int:
    encoded_size = len(payload)
    if encoded_size % 4:
        # Invalid padding will be reported by the decoder, but this upper
        # bound still prevents an oversized malformed payload from reaching it.
        return ((encoded_size + 3) // 4) * 3
    padding = 2 if payload.endswith("==") else 1 if payload.endswith("=") else 0
    return (encoded_size // 4) * 3 - padding


def _percent_decoded_size(payload: str, maximum: int) -> int:
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    decoded_size = 0
    index = 0
    while index < len(payload):
        if (
            payload[index] == "%"
            and index + 2 < len(payload)
            and payload[index + 1] in hexadecimal
            and payload[index + 2] in hexadecimal
        ):
            decoded_size += 1
            index += 3
        else:
            decoded_size += len(payload[index].encode("utf-8"))
            index += 1
        if decoded_size > maximum:
            return decoded_size
    return decoded_size


def _data_uri_sha256(
    uri: str, *, max_embedded_asset_bytes: int
) -> tuple[str | None, str | None, int | None]:
    if uri[:5].lower() != "data:":
        return None, None, None
    try:
        header, payload = uri[5:].split(",", 1)
    except ValueError as exc:
        raise ContractValidationError("image data URI is missing its payload") from exc
    parts = header.split(";")
    media_type = parts[0] or None
    is_base64 = any(part.lower() == "base64" for part in parts[1:])
    decoded_size = (
        _base64_decoded_size(payload)
        if is_base64
        else _percent_decoded_size(payload, max_embedded_asset_bytes)
    )
    if decoded_size > max_embedded_asset_bytes:
        raise ContractValidationError(
            "embedded image data URI exceeds max_embedded_asset_bytes "
            f"({decoded_size} > {max_embedded_asset_bytes}); externalize the asset "
            "and provide its URI and SHA-256"
        )
    try:
        if is_base64:
            decoded = base64.b64decode(payload, validate=True)
        else:
            decoded = unquote_to_bytes(payload)
    except (ValueError, binascii.Error) as exc:
        raise ContractValidationError("image data URI payload is invalid") from exc
    if len(decoded) > max_embedded_asset_bytes:
        # Defensive check in case a future decoder behaves differently from
        # the size estimator.  Decoding remains bounded by the pre-check.
        raise ContractValidationError(
            "embedded image data URI exceeds max_embedded_asset_bytes; "
            "externalize the asset"
        )
    return hashlib.sha256(decoded).hexdigest(), media_type, len(decoded)


def _asset_from_image(
    value: Mapping[str, object],
    *,
    kind: AssetKind,
    field: str,
    max_embedded_asset_bytes: int,
) -> AssetRef:
    uri = require_non_empty(value.get("uri"), f"{field}.uri")
    embedded_hash, embedded_media_type, embedded_bytes = _data_uri_sha256(
        uri, max_embedded_asset_bytes=max_embedded_asset_bytes
    )
    media_type = _text(value.get("mimetype")) or _text(value.get("media_type"))
    if media_type is None:
        media_type = embedded_media_type or mimetypes.guess_type(uri)[0]
    declared_hash = value.get("sha256") or value.get("image_sha256")
    if declared_hash is not None:
        require_sha256(declared_hash, f"{field}.sha256")
        if embedded_hash is not None and declared_hash != embedded_hash:
            raise ContractValidationError(f"{field}.sha256 does not match data URI bytes")
    metadata: dict[str, object] = {"source": "docling_export"}
    if embedded_bytes is not None:
        metadata["embedded_bytes"] = embedded_bytes
    if value.get("dpi") is not None:
        metadata["dpi"] = _finite_number(value.get("dpi"), f"{field}.dpi")
    if value.get("size") is not None:
        size = require_mapping(value.get("size"), f"{field}.size")
        metadata["size"] = {
            "width": _finite_number(size.get("width"), f"{field}.size.width"),
            "height": _finite_number(size.get("height"), f"{field}.size.height"),
        }
    return AssetRef(
        kind=kind,
        uri=uri,
        media_type=media_type,
        sha256=declared_hash or embedded_hash,  # type: ignore[arg-type]
        metadata=metadata,
    )


def _assets(
    item: Mapping[str, object],
    block_type: BlockType,
    *,
    include_images: bool,
    max_embedded_asset_bytes: int,
) -> tuple[AssetRef, ...]:
    if not include_images:
        return ()
    kind = (
        AssetKind.EXTRACTED_IMAGE
        if block_type in {BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM}
        else AssetKind.REGION_CROP
    )
    result: list[AssetRef] = []
    image = item.get("image")
    if image is not None:
        image_mapping = require_mapping(image, "item.image")
        if image_mapping.get("uri") is not None:
            result.append(
                _asset_from_image(
                    image_mapping,
                    kind=kind,
                    field="item.image",
                    max_embedded_asset_bytes=max_embedded_asset_bytes,
                )
            )
    elif item.get("image_uri") is not None:
        result.append(
            _asset_from_image(
                {
                    "uri": item.get("image_uri"),
                    "mimetype": item.get("image_mimetype"),
                    "sha256": item.get("image_sha256"),
                },
                kind=kind,
                field="item",
                max_embedded_asset_bytes=max_embedded_asset_bytes,
            )
        )
    for index, value in enumerate(
        require_sequence(item.get("assets", ()), "item.assets")
    ):
        asset = require_mapping(value, f"item.assets[{index}]")
        result.append(
            _asset_from_image(
                asset,
                kind=kind,
                field=f"item.assets[{index}]",
                max_embedded_asset_bytes=max_embedded_asset_bytes,
            )
        )
    return tuple(result)


def _table_matrix(
    data: Mapping[str, object],
) -> tuple[list[list[str]], list[list[bool]], list[Mapping[str, object]]]:
    preflight_explicit_json(data, field="table.data")
    rows_value = data.get("num_rows")
    cols_value = data.get("num_cols")
    grid_value = data.get("grid", ())
    grid = require_sequence(grid_value, "table.data.grid")
    if rows_value is None:
        rows = len(grid)
    else:
        rows = _positive_integer(rows_value, "table.data.num_rows")
    if cols_value is None:
        cols = max(
            (
                len(require_sequence(row, f"table.data.grid[{index}]"))
                for index, row in enumerate(grid)
            ),
            default=0,
        )
    else:
        cols = _positive_integer(cols_value, "table.data.num_cols")
    if rows < 1 or cols < 1:
        raise ContractValidationError("table data requires at least one row and column")
    if rows > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE // cols:
        raise ContractValidationError(
            "table dimensions exceed the bounded table-position limit"
        )

    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    headers = [[False for _ in range(cols)] for _ in range(rows)]
    raw_cells = require_sequence(data.get("table_cells", ()), "table.data.table_cells")
    if len(raw_cells) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
        raise ContractValidationError("table.data.table_cells exceeds the cell limit")
    cells: list[Mapping[str, object]] = []
    if raw_cells:
        for index, value in enumerate(raw_cells):
            cell = require_mapping(value, f"table.data.table_cells[{index}]")
            start_row = _non_negative_integer(
                cell.get("start_row_offset_idx"),
                f"table.data.table_cells[{index}].start_row_offset_idx",
            )
            end_row = _positive_integer(
                cell.get("end_row_offset_idx"),
                f"table.data.table_cells[{index}].end_row_offset_idx",
            )
            start_col = _non_negative_integer(
                cell.get("start_col_offset_idx"),
                f"table.data.table_cells[{index}].start_col_offset_idx",
            )
            end_col = _positive_integer(
                cell.get("end_col_offset_idx"),
                f"table.data.table_cells[{index}].end_col_offset_idx",
            )
            if end_row <= start_row or end_col <= start_col or end_row > rows or end_col > cols:
                raise ContractValidationError(
                    f"table.data.table_cells[{index}] span falls outside the table"
                )
            cell_text = cell.get("text", "")
            if not isinstance(cell_text, str):
                raise ContractValidationError(
                    f"table.data.table_cells[{index}].text must be a string"
                )
            matrix[start_row][start_col] = cell_text
            headers[start_row][start_col] = bool(
                cell.get("column_header", False) or cell.get("row_header", False)
            )
            cells.append(cell)
    else:
        for row_index, row_value in enumerate(grid):
            if row_index >= rows:
                raise ContractValidationError("table.data.grid has more rows than num_rows")
            row = require_sequence(row_value, f"table.data.grid[{row_index}]")
            if len(row) > cols:
                raise ContractValidationError(
                    f"table.data.grid[{row_index}] has more columns than num_cols"
                )
            for col_index, value in enumerate(row):
                if isinstance(value, str):
                    matrix[row_index][col_index] = value
                    continue
                cell = require_mapping(
                    value, f"table.data.grid[{row_index}][{col_index}]"
                )
                cell_text = cell.get("text", "")
                if not isinstance(cell_text, str):
                    raise ContractValidationError(
                        f"table.data.grid[{row_index}][{col_index}].text must be a string"
                    )
                matrix[row_index][col_index] = cell_text
                headers[row_index][col_index] = bool(
                    cell.get("column_header", False) or cell.get("row_header", False)
                )
    return matrix, headers, cells


def _table_html(
    matrix: Sequence[Sequence[str]],
    headers: Sequence[Sequence[bool]],
    cells: Sequence[Mapping[str, object]],
) -> str:
    starts: dict[tuple[int, int], Mapping[str, object]] = {}
    covered: set[tuple[int, int]] = set()
    for cell in cells:
        start_row = int(cell["start_row_offset_idx"])
        end_row = int(cell["end_row_offset_idx"])
        start_col = int(cell["start_col_offset_idx"])
        end_col = int(cell["end_col_offset_idx"])
        key = (start_row, start_col)
        if key in starts or key in covered:
            raise ContractValidationError("table cells contain overlapping spans")
        starts[key] = cell
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if (row, col) != key:
                    if (row, col) in starts or (row, col) in covered:
                        raise ContractValidationError("table cells contain overlapping spans")
                    covered.add((row, col))

    output = ["<table><tbody>"]
    for row_index, row in enumerate(matrix):
        output.append("<tr>")
        for col_index, value in enumerate(row):
            if (row_index, col_index) in covered:
                continue
            cell = starts.get((row_index, col_index))
            is_header = headers[row_index][col_index]
            attributes = ""
            if cell is not None:
                rowspan = int(cell["end_row_offset_idx"]) - int(
                    cell["start_row_offset_idx"]
                )
                colspan = int(cell["end_col_offset_idx"]) - int(
                    cell["start_col_offset_idx"]
                )
                if rowspan > 1:
                    attributes += f' rowspan="{rowspan}"'
                if colspan > 1:
                    attributes += f' colspan="{colspan}"'
            tag = "th" if is_header else "td"
            output.append(f"<{tag}{attributes}>{escape_html(value)}</{tag}>")
        output.append("</tr>")
    output.append("</tbody></table>")
    return "".join(output)


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _table_markdown(matrix: Sequence[Sequence[str]]) -> str:
    rows = ["| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in matrix]
    rows.insert(1, "| " + " | ".join("---" for _ in matrix[0]) + " |")
    return "\n".join(rows)


def _append_resolved_representations(
    result: list[Representation],
    item: Mapping[str, object],
    *,
    text_by_ref: Mapping[str, Mapping[str, object]],
    confidence: float | None,
    language: str | None,
) -> None:
    for field, kind in (
        ("captions", RepresentationKind.CAPTION),
        ("footnotes", RepresentationKind.FOOTNOTE),
    ):
        texts, refs = _resolved_texts(
            item.get(field, ()), text_by_ref=text_by_ref, field=f"item.{field}"
        )
        if texts:
            result.append(
                Representation(
                    kind=kind,
                    content="\n".join(texts),
                    media_type="text/plain",
                    language=language,
                    confidence=confidence,
                    metadata={"source_refs": refs},
                )
            )


def _representations(
    item: Mapping[str, object],
    provenance: Mapping[str, object],
    block_type: BlockType,
    *,
    text_by_ref: Mapping[str, Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    confidence: float | None,
    language: str | None,
) -> tuple[Representation, ...]:
    result: list[Representation] = []
    value = _item_text(item, provenance)
    parent_ref = _parent_ref(item)
    if block_type == BlockType.EQUATION:
        latex = _text(item.get("latex"))
        if latex is not None:
            result.append(
                Representation(
                    RepresentationKind.LATEX,
                    latex,
                    "application/x-latex",
                    confidence=confidence,
                    metadata={"source_field": "latex", "format": "latex"},
                )
            )
        elif value is not None:
            result.append(
                Representation(
                    RepresentationKind.TEXT,
                    value,
                    "text/plain",
                    confidence=confidence,
                    metadata={
                        "source_field": "text" if item.get("text") else "orig",
                        "format": "unknown",
                    },
                )
            )
    elif block_type == BlockType.TABLE:
        data = require_mapping(item.get("data", {}), "table.data")
        direct_html = _text(item.get("html"))
        direct_markdown = _text(item.get("markdown"))
        if data:
            matrix, headers, cells = _table_matrix(data)
            result.append(
                Representation(
                    RepresentationKind.TABLE_GRID,
                    data,
                    "application/json",
                    confidence=confidence,
                )
            )
            direct_html = direct_html or _table_html(matrix, headers, cells)
            direct_markdown = direct_markdown or _table_markdown(matrix)
        if direct_html is not None:
            result.append(
                Representation(
                    RepresentationKind.HTML,
                    direct_html,
                    "text/html",
                    language=language,
                    confidence=confidence,
                )
            )
        if direct_markdown is not None:
            result.append(
                Representation(
                    RepresentationKind.MARKDOWN,
                    direct_markdown,
                    "text/markdown",
                    language=language,
                    confidence=confidence,
                )
            )
    elif block_type in {BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM}:
        description, described_by = _image_description(item)
        if description is not None:
            result.append(
                Representation(
                    RepresentationKind.VISUAL_DESCRIPTION,
                    description,
                    "text/plain",
                    language=language,
                    confidence=confidence,
                    metadata={"created_by": described_by} if described_by else {},
                )
            )
        child_texts, child_refs = _resolved_texts(
            item.get("children", ()),
            text_by_ref=text_by_ref,
            field="item.children",
        )
        if child_texts:
            result.append(
                Representation(
                    RepresentationKind.OCR_TEXT,
                    "\n".join(child_texts),
                    "text/plain",
                    language=language,
                    confidence=confidence,
                    metadata={"source_refs": child_refs},
                )
            )
        if predictions:
            result.append(
                Representation(
                    RepresentationKind.STRUCTURED_DATA,
                    {"classification_predictions": list(predictions)},
                    "application/json",
                    confidence=confidence,
                )
            )
    elif value is not None:
        if block_type == BlockType.CAPTION:
            representation_kind = RepresentationKind.CAPTION
        elif block_type == BlockType.FOOTNOTE:
            representation_kind = RepresentationKind.FOOTNOTE
        elif parent_ref is not None and parent_ref.startswith("#/pictures/"):
            representation_kind = RepresentationKind.OCR_TEXT
        else:
            representation_kind = RepresentationKind.TEXT
        original = _text(item.get("orig"))
        metadata: dict[str, object] = {"source_field": "text"}
        if original is not None and original != value:
            metadata["original_text"] = original
        result.append(
            Representation(
                representation_kind,
                value,
                "text/plain",
                language=language,
                confidence=confidence,
                metadata=metadata,
            )
        )
        if block_type == BlockType.TITLE:
            level = item.get("level", 1)
            if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 6:
                raise ContractValidationError("title level must be an integer from 1 to 6")
            result.append(
                Representation(
                    RepresentationKind.MARKDOWN,
                    f"{'#' * level} {value}",
                    "text/markdown",
                    language=language,
                    confidence=confidence,
                )
            )
        elif block_type == BlockType.LIST_ITEM:
            marker = _text(item.get("marker")) or "-"
            result.append(
                Representation(
                    RepresentationKind.MARKDOWN,
                    f"{marker} {value}",
                    "text/markdown",
                    language=language,
                    confidence=confidence,
                )
            )

    if block_type in {
        BlockType.TABLE,
        BlockType.FIGURE,
        BlockType.CHART,
        BlockType.DIAGRAM,
    }:
        _append_resolved_representations(
            result,
            item,
            text_by_ref=text_by_ref,
            confidence=confidence,
            language=language,
        )
    return tuple(result)


def _confidence(
    item: Mapping[str, object],
    provenance: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
) -> float | None:
    if item.get("confidence") is not None:
        return _probability(item.get("confidence"), "item.confidence")
    if provenance.get("confidence") is not None:
        return _probability(provenance.get("confidence"), "provenance.confidence")
    if predictions:
        return float(predictions[0]["confidence"])
    return None


def _raw_item_uri(base_uri: str | None, item: Mapping[str, object], self_ref: str) -> str | None:
    direct = _text(item.get("raw_artifact_uri"))
    if direct is not None:
        return direct
    if base_uri is None:
        return None
    root, _fragment = urldefrag(base_uri)
    return f"{root}{self_ref}" if self_ref.startswith("#") else root


def _native_id(
    self_ref: str,
    provenance: Mapping[str, object],
    provenance_count: int,
) -> str:
    if provenance_count == 1:
        return self_ref
    signature = {
        "page_no": provenance.get("page_no"),
        "page_index": provenance.get("page_index"),
        "bbox": provenance.get("bbox"),
        "polygon": provenance.get("polygon"),
        "charspan": provenance.get("charspan"),
    }
    return f"{self_ref}@prov:{content_sha256(signature)[:16]}"


def _translate_item(
    request: ParseRequest,
    export: Mapping[str, object],
    item: Mapping[str, object],
    *,
    collection: str,
    fallback_order: float,
    order_by_ref: Mapping[str, float],
    text_by_ref: Mapping[str, Mapping[str, object]],
    parser_version: str,
    model_version: str,
    config_sha256: str,
    raw_artifact_uri: str | None,
    max_embedded_asset_bytes: int,
) -> tuple[ParserBlockObservation, ...]:
    self_ref = require_non_empty(item.get("self_ref"), f"{collection}.self_ref")
    expected_prefix = f"#/{collection}/"
    if not self_ref.startswith(expected_prefix):
        raise ContractValidationError(
            f"{collection}.self_ref must start with {expected_prefix!r}"
        )
    predictions = _predictions(item) if collection == "pictures" else ()
    block_type, semantic_subtype = _block_type(collection, item, predictions)
    if block_type == BlockType.TABLE and not request.options.parse_tables:
        return ()
    if block_type == BlockType.EQUATION and not request.options.parse_formulas:
        return ()
    if collection == "pictures" and not request.options.parse_images:
        return ()

    provenance_values = require_sequence(item.get("prov"), f"{self_ref}.prov")
    if not provenance_values:
        raise ContractValidationError(f"{self_ref}.prov must not be empty")
    language = request.options.languages[0] if len(request.options.languages) == 1 else None
    observations: list[ParserBlockObservation] = []
    seen_native_ids: set[str] = set()
    for provenance_index, value in enumerate(provenance_values):
        provenance = require_mapping(value, f"{self_ref}.prov[{provenance_index}]")
        page_index = _page_index(provenance)
        if (
            request.document.page_count is not None
            and page_index >= request.document.page_count
        ):
            raise ContractValidationError(
                f"{self_ref} references page {page_index + 1} beyond document page_count"
            )
        page = request.page_by_index.get(page_index)
        if page is None:
            # A full-document Docling export may be filtered to a routed page
            # request.  Non-requested, otherwise valid pages are not failures.
            continue
        native_id = _native_id(self_ref, provenance, len(provenance_values))
        if native_id in seen_native_ids:
            raise ContractValidationError(
                f"{self_ref} contains duplicate provenance entries"
            )
        seen_native_ids.add(native_id)
        confidence = _confidence(item, provenance, predictions)
        representations = _representations(
            item,
            provenance,
            block_type,
            text_by_ref=text_by_ref,
            predictions=predictions,
            confidence=confidence,
            language=language,
        )
        semantic_substructures = ()
        if block_type is BlockType.TABLE:
            data = require_mapping(item.get("data", {}), "table.data")
            if data.get("table_cells") is not None:
                table_representation = next(
                    (
                        representation
                        for representation in representations
                        if representation.kind is RepresentationKind.TABLE_GRID
                    ),
                    None,
                )
                if table_representation is None:
                    raise ContractValidationError(
                        "Docling table_cells require a TABLE_GRID representation"
                    )
                semantic_substructures = (
                    docling_table_substructure(
                        data,
                        table_representation,
                        confidence=confidence,
                    ),
                )
        assets = _assets(
            item,
            block_type,
            include_images=request.options.parse_images,
            max_embedded_asset_bytes=max_embedded_asset_bytes,
        )
        geometry, coord_origin = _geometry(provenance, request, page_index)
        parent_ref = _parent_ref(item)
        metadata: dict[str, object] = {
            "translation": TRANSLATION_NAME,
            "docling_schema_name": export["schema_name"],
            "docling_document_version": export["version"],
            "docling_collection": collection,
            "docling_self_ref": self_ref,
            "docling_label": str(item.get("label", "unspecified")),
            "normalized_item_sha256": content_sha256(item),
        }
        if coord_origin is not None:
            metadata["source_coordinate_origin"] = coord_origin
        if parent_ref is not None:
            metadata["parent_ref"] = parent_ref
        structural_refs: dict[str, object] = {}
        if parent_ref is not None:
            structural_refs["parent"] = parent_ref
        for field_name in ("children", "captions", "footnotes", "references"):
            references = _relation_refs(item.get(field_name, ()))
            if references:
                structural_refs[field_name] = list(references)
        if structural_refs:
            metadata["docling_structural_refs"] = structural_refs
        if item.get("content_layer") is not None:
            metadata["content_layer"] = item.get("content_layer")
        if item.get("level") is not None:
            metadata["level"] = item.get("level")
        if provenance.get("charspan") is not None:
            metadata["charspan"] = provenance.get("charspan")
        observations.append(
            ParserBlockObservation(
                document_version_id=request.document.document_version_id,
                page_id=page.page_id,
                page_index=page_index,
                provenance=ParserProvenance(
                    parser_name=PARSER_NAME,
                    parser_version=parser_version,
                    model_version=model_version,
                    config_sha256=config_sha256,
                    parser_native_id=native_id,
                    raw_artifact_uri=_raw_item_uri(
                        raw_artifact_uri, item, self_ref
                    ),
                    transformations=(TRANSLATION_NAME,),
                ),
                block_type=block_type,
                semantic_subtype=semantic_subtype,
                geometry=geometry,
                reading_order=order_by_ref.get(self_ref, fallback_order),
                representations=representations,
                semantic_substructures=semantic_substructures,
                assets=assets,
                confidence=confidence,
                metadata=metadata,
            )
        )
    return tuple(observations)


def _validate_export_binding(
    request: ParseRequest, export: Mapping[str, object]
) -> None:
    schema_name = require_non_empty(export.get("schema_name"), "docling.schema_name")
    if schema_name != "DoclingDocument":
        raise ContractValidationError(
            f"unsupported Docling schema_name {schema_name!r}; expected 'DoclingDocument'"
        )
    require_non_empty(export.get("version"), "docling.version")
    declared_sha256 = export.get("source_sha256")
    if declared_sha256 is not None:
        require_sha256(declared_sha256, "docling.source_sha256")
        if declared_sha256 != request.document.content_sha256:
            raise ContractValidationError(
                "Docling export source_sha256 differs from the parse request"
            )
    origin = export.get("origin")
    if origin is not None:
        origin_mapping = require_mapping(origin, "docling.origin")
        origin_mime = _text(origin_mapping.get("mimetype"))
        if origin_mime is not None and origin_mime != request.document.mime_type:
            raise ContractValidationError(
                "Docling export origin MIME type differs from the parse request"
            )
    pages = export.get("pages")
    if pages is not None:
        require_mapping(pages, "docling.pages")
    for collection in _COLLECTIONS:
        require_sequence(export.get(collection, ()), f"docling.{collection}")


def _relation_refs(value: object) -> tuple[str, ...]:
    """Return only explicit Docling JSON references; direct text is not an edge."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    result: list[str] = []
    for entry in value:
        try:
            reference = _reference(entry, "docling structural relation")
        except ContractValidationError:
            continue
        if reference.startswith("#/"):
            result.append(reference)
    return tuple(result)


def _attach_structural_relations(
    export: Mapping[str, object],
    observations: Sequence[ParserBlockObservation],
    *,
    order_by_ref: Mapping[str, float],
) -> tuple[ParserBlockObservation, ...]:
    """Resolve explicit Docling references to immutable observation IDs.

    A reference is accepted only when it resolves to exactly one translated
    observation on the source page.  Body/group references and ambiguous
    multi-provenance targets remain available in raw metadata but do not
    become guessed edges.
    """

    items_by_ref: dict[str, Mapping[str, object]] = {}
    for collection in _COLLECTIONS:
        for value in require_sequence(export.get(collection, ()), collection):
            if not isinstance(value, Mapping):
                continue
            self_ref = value.get("self_ref")
            if isinstance(self_ref, str) and self_ref.strip():
                items_by_ref.setdefault(self_ref, value)

    by_ref_page: dict[tuple[str, int], list[ParserBlockObservation]] = {}
    by_ref: dict[str, list[ParserBlockObservation]] = {}
    for observation in observations:
        self_ref = observation.metadata.get("docling_self_ref")
        if not isinstance(self_ref, str):
            continue
        by_ref_page.setdefault((self_ref, observation.page_index), []).append(
            observation
        )
        by_ref.setdefault(self_ref, []).append(observation)

    def resolve(
        reference: str,
        page_index: int,
        *,
        allow_cross_page: bool,
    ) -> tuple[ParserBlockObservation | None, str]:
        candidates = by_ref_page.get((reference, page_index), ())
        if len(candidates) == 1:
            return candidates[0], "resolved_same_page"
        if len(candidates) > 1:
            return None, "ambiguous_same_page"
        if allow_cross_page:
            global_candidates = by_ref.get(reference, ())
            if len(global_candidates) == 1:
                return global_candidates[0], "resolved_globally_unique_cross_page"
            if len(global_candidates) > 1:
                return None, "ambiguous_cross_page"
        if reference == "#/body" or reference.startswith("#/groups/"):
            return None, "container_only"
        return None, "unresolved"

    assertions: dict[str, list[ObservationRelation]] = {
        item.observation_id: [] for item in observations
    }
    resolution_records: dict[str, list[dict[str, object]]] = {
        item.observation_id: [] for item in observations
    }

    def add(
        source: ParserBlockObservation,
        target_ref: str,
        relation_type: RelationType,
        source_field: str,
        *,
        allow_cross_page: bool = False,
    ) -> None:
        target, status = resolve(
            target_ref,
            source.page_index,
            allow_cross_page=allow_cross_page,
        )
        record: dict[str, object] = {
            "relation_type": relation_type.value,
            "source_field": source_field,
            "target_ref": target_ref,
            "status": status,
        }
        if target is not None:
            record["target_observation_id"] = target.observation_id
        resolution_records[source.observation_id].append(record)
        if target is None:
            return
        if target.observation_id == source.observation_id:
            record["status"] = "self_target_omitted"
            return
        assertions[source.observation_id].append(
            ObservationRelation(
                relation_type=relation_type,
                target_observation_id=target.observation_id,
                confidence=1.0,
                metadata={
                    "assertion_origin": "docling_explicit_reference",
                    "source_field": source_field,
                    "source_ref": source.metadata["docling_self_ref"],
                    "target_ref": target_ref,
                },
            )
        )

    for source in observations:
        self_ref = source.metadata.get("docling_self_ref")
        item = items_by_ref.get(self_ref) if isinstance(self_ref, str) else None
        if item is None:
            continue
        parent_ref = _parent_ref(item)
        if parent_ref is not None:
            relation_type = (
                RelationType.CAPTION_OF
                if source.block_type == BlockType.CAPTION
                else RelationType.FOOTNOTE_OF
                if source.block_type == BlockType.FOOTNOTE
                else RelationType.CHILD_OF
            )
            add(source, parent_ref, relation_type, "parent")
        for field_name, relation_type in (
            ("children", RelationType.PARENT_OF),
            ("captions", RelationType.HAS_CAPTION),
            ("footnotes", RelationType.HAS_FOOTNOTE),
            ("references", RelationType.REFERENCES),
        ):
            for target_ref in _relation_refs(item.get(field_name, ())):
                add(
                    source,
                    target_ref,
                    relation_type,
                    field_name,
                    allow_cross_page=field_name == "references",
                )

    # Docling's body/group traversal is explicit parser-native order.  Create
    # only adjacent, same-page edges among uniquely resolvable body references.
    by_page: dict[int, list[tuple[float, str, ParserBlockObservation]]] = {}
    for (self_ref, page_index), candidates in by_ref_page.items():
        if self_ref not in order_by_ref or len(candidates) != 1:
            continue
        by_page.setdefault(page_index, []).append(
            (order_by_ref[self_ref], self_ref, candidates[0])
        )
    for values in by_page.values():
        ordered = sorted(values, key=lambda item: (item[0], item[1]))
        for left, right in zip(ordered, ordered[1:], strict=False):
            add(left[2], right[1], RelationType.READING_NEXT, "body.children")
            add(
                right[2],
                left[1],
                RelationType.READING_PREVIOUS,
                "body.children",
            )

    enriched: list[ParserBlockObservation] = []
    for observation in observations:
        relations = tuple(
            sorted(
                (*observation.relations, *assertions[observation.observation_id]),
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
            metadata["docling_structural_ref_resolution"] = sorted(
                records,
                key=lambda item: (
                    str(item["source_field"]),
                    str(item["relation_type"]),
                    str(item["target_ref"]),
                    str(item["status"]),
                ),
            )
        enriched.append(
            replace(observation, relations=relations, metadata=metadata)
        )
    return tuple(enriched)


def translate_docling_export(
    request: ParseRequest,
    export: Mapping[str, object],
    *,
    parser_version: str,
    model_version: str,
    config_sha256: str,
    raw_artifact_uri: str | None = None,
    max_embedded_asset_bytes: int = DEFAULT_MAX_EMBEDDED_ASSET_BYTES,
) -> tuple[tuple[ParserBlockObservation, ...], tuple[ParseIssue, ...]]:
    """Translate a JSON-compatible ``DoclingDocument`` export.

    Root/schema/source-binding failures raise ``ContractValidationError``.
    Invalid individual items become permanent typed issues so valid evidence
    survives in a ``PARTIAL`` result.
    """

    if not isinstance(request, ParseRequest):
        raise TypeError("request must be ParseRequest")
    export = require_mapping(export, "docling export")
    require_non_empty(parser_version, "parser_version")
    require_non_empty(model_version, "model_version")
    require_sha256(config_sha256, "config_sha256")
    max_embedded_asset_bytes = _validate_embedded_asset_limit(
        max_embedded_asset_bytes
    )
    if raw_artifact_uri is not None:
        require_non_empty(raw_artifact_uri, "raw_artifact_uri")
    _validate_export_binding(request, export)
    order_by_ref = _reading_order(export)
    text_by_ref = _text_index(export)

    observations: list[ParserBlockObservation] = []
    issues: list[ParseIssue] = []
    seen_refs: set[str] = set()
    fallback = float(len(order_by_ref))
    for collection_rank, collection in enumerate(_COLLECTIONS):
        values = require_sequence(export.get(collection, ()), f"docling.{collection}")
        for index, value in enumerate(values):
            native_id: str | None = None
            page_index: int | None = None
            try:
                item = require_mapping(value, f"docling.{collection}[{index}]")
                self_ref = require_non_empty(
                    item.get("self_ref"), f"docling.{collection}[{index}].self_ref"
                )
                native_id = self_ref
                if self_ref in seen_refs:
                    raise ContractValidationError(
                        f"duplicate Docling item self_ref: {self_ref}"
                    )
                seen_refs.add(self_ref)
                provenance = require_sequence(
                    item.get("prov"), f"docling.{collection}[{index}].prov"
                )
                if provenance and isinstance(provenance[0], Mapping):
                    try:
                        page_index = _page_index(provenance[0])
                    except ContractValidationError:
                        page_index = None
                observations.extend(
                    _translate_item(
                        request,
                        export,
                        item,
                        collection=collection,
                        fallback_order=fallback + collection_rank * 1_000_000 + index,
                        order_by_ref=order_by_ref,
                        text_by_ref=text_by_ref,
                        parser_version=parser_version,
                        model_version=model_version,
                        config_sha256=config_sha256,
                        raw_artifact_uri=raw_artifact_uri,
                        max_embedded_asset_bytes=max_embedded_asset_bytes,
                    )
                )
            except (ContractValidationError, TypeError, ValueError) as exc:
                issues.append(
                    ParseIssue(
                        code=ParseErrorCode.INVALID_NORMALIZED_BLOCK,
                        classification=ErrorClassification.PERMANENT,
                        severity=IssueSeverity.ERROR,
                        message=f"Docling {collection} item {index} is invalid: {exc}",
                        page_index=page_index,
                        parser_native_id=native_id,
                        details={
                            "collection": collection,
                            "item_index": index,
                            "translation": TRANSLATION_NAME,
                        },
                    )
                )
    observations = list(
        _attach_structural_relations(
            export, observations, order_by_ref=order_by_ref
        )
    )
    observations.sort(
        key=lambda item: (
            item.page_index,
            float("inf") if item.reading_order is None else item.reading_order,
            item.observation_id,
        )
    )
    issues.sort(key=lambda item: item.issue_id)
    return tuple(observations), tuple(issues)


class DoclingExportAdapter:
    """Injected Docling export bridge implementing the parser port."""

    def __init__(
        self,
        export_loader: DoclingExportLoader | Mapping[str, object],
        *,
        parser_version: str,
        model_version: str,
        config: Mapping[str, object],
        raw_artifact_uri: str | None = None,
        max_embedded_asset_bytes: int = DEFAULT_MAX_EMBEDDED_ASSET_BYTES,
    ) -> None:
        if not callable(export_loader) and not isinstance(export_loader, Mapping):
            raise TypeError("export_loader must be callable or a JSON object")
        require_non_empty(parser_version, "parser_version")
        require_non_empty(model_version, "model_version")
        if raw_artifact_uri is not None:
            require_non_empty(raw_artifact_uri, "raw_artifact_uri")
        max_embedded_asset_bytes = _validate_embedded_asset_limit(
            max_embedded_asset_bytes
        )
        self._loader = export_loader
        self._parser_version = parser_version
        self._model_version = model_version
        self._max_embedded_asset_bytes = max_embedded_asset_bytes
        self._config_sha256 = content_sha256(
            {
                "docling": config,
                "adapter": {
                    "translation": TRANSLATION_NAME,
                    "max_embedded_asset_bytes": max_embedded_asset_bytes,
                },
            }
        )
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
                ParserModality.CODE,
            ),
            block_types=(
                BlockType.TITLE,
                BlockType.PARAGRAPH,
                BlockType.LIST_ITEM,
                BlockType.TABLE,
                BlockType.EQUATION,
                BlockType.FIGURE,
                BlockType.CHART,
                BlockType.DIAGRAM,
                BlockType.CODE,
                BlockType.CAPTION,
                BlockType.FOOTNOTE,
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.UNKNOWN,
            ),
            accelerator=AcceleratorKind.ANY,
            supports_page_selection=True,
            supports_region_selection=False,
            supports_batch=True,
            deterministic=False,
            metadata={
                "adapter_contract_version": ADAPTER_VERSION,
                "input_shape": "DoclingDocument.export_to_dict.v1",
                "translation": TRANSLATION_NAME,
                "vendor_imports": False,
                "page_numbering": "docling_one_based_to_contract_zero_based",
                "selection_enforcement": "translator_filter",
                "max_embedded_asset_bytes": max_embedded_asset_bytes,
                "embedded_asset_policy": (
                    "externalize data-URI assets larger than the configured byte limit "
                    "and provide URI plus SHA-256"
                ),
            },
        )

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    def manifest(self) -> ParserCapabilityManifest:
        return self._manifest

    def _load(self, request: ParseRequest) -> Mapping[str, object]:
        if callable(self._loader):
            return self._loader(request)
        return self._loader

    def _failed_result(
        self,
        request: ParseRequest,
        started: float,
        issue: ParseIssue,
    ) -> ParseResult:
        return ParseResult(
            request_id=request.request_id,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            status=ParseStatus.FAILED,
            issues=(issue,),
            duration_ms=(perf_counter() - started) * 1000,
            metrics={"valid_observations": 0, "invalid_items": 0},
        )

    def parse(self, request: ParseRequest) -> ParseResult:
        started = perf_counter()
        try:
            export = self._load(request)
        except ParserExecutionError as exc:
            return self._failed_result(request, started, exc.issue)
        except Exception as exc:
            return self._failed_result(
                request,
                started,
                ParseIssue(
                    code=ParseErrorCode.PARSER_FAILURE,
                    classification=ErrorClassification.RETRYABLE,
                    severity=IssueSeverity.ERROR,
                    message="Docling export loader failed",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        try:
            observations, issues = translate_docling_export(
                request,
                export,
                parser_version=self._parser_version,
                model_version=self._model_version,
                config_sha256=self._config_sha256,
                raw_artifact_uri=self._raw_artifact_uri,
                max_embedded_asset_bytes=self._max_embedded_asset_bytes,
            )
        except (ContractValidationError, TypeError, ValueError) as exc:
            return self._failed_result(
                request,
                started,
                ParseIssue(
                    code=ParseErrorCode.CONTRACT_VIOLATION,
                    classification=ErrorClassification.PERMANENT,
                    severity=IssueSeverity.ERROR,
                    message=f"Docling export violates the adapter contract: {exc}",
                    details={
                        "exception_type": type(exc).__name__,
                        "translation": TRANSLATION_NAME,
                    },
                ),
            )

        if issues and observations:
            status = ParseStatus.PARTIAL
        elif issues:
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
            duration_ms=(perf_counter() - started) * 1000,
            metrics={
                "valid_observations": len(observations),
                "invalid_items": len(issues),
            },
        )
