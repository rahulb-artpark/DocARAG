"""Adapter for BookRAG's current MinerU-normalized block list.

This module intentionally does not import MinerU.  A composition root injects
the existing function (or a file/RPC loader) that produces the normalized
``list[dict]`` shape.  This keeps the controller and contract tests usable in a
clean environment while providing one precise hook for the legacy pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from html import unescape
import math
import mimetypes
import re
from time import perf_counter

from docarag.domain.base import (
    ContractValidationError,
    content_sha256,
    require_mapping,
    require_non_empty,
    require_sha256,
)
from docarag.domain.evidence import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    CoordinateSpace,
    ObservationRelation,
    ParserBlockObservation,
    ParserProvenance,
    Point,
    RelationType,
    Representation,
    RepresentationKind,
    SourceGeometry,
)
from docarag.adapters.semantic_normalization import (
    explicit_diagram_substructure,
    explicit_table_grid_substructure,
    mathml_formula_substructure,
    preflight_explicit_json,
    preflight_mathml,
)
from docarag.domain.parsing import (
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


NormalizedBlockLoader = Callable[[ParseRequest], object]

ADAPTER_NAME = "mineru-normalized"
ADAPTER_VERSION = "1.1.0"
PARSER_NAME = "mineru"
TRANSLATION_NAME = "bookrag_mineru_normalized_to_observation.v2"

_DIAGRAM_SUBTYPES = frozenset(
    {"diagram", "flowchart", "flow_chart", "workflow", "sketch"}
)


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _joined_text(value: object) -> str | None:
    """The block's text, or None when it carries none worth representing.

    A NUL is MinerU's fallback for a glyph it could not map, and the contract
    drops it before hashing (see ``freeze_json``). Emptiness is therefore
    judged on the text that will actually be stored: a block whose only
    "text" is an unmappable glyph has none, and saying so here degrades that
    one block instead of failing the bind on an empty representation.
    """

    if isinstance(value, str):
        return value if value.replace("\x00", "").strip() else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.replace("\x00", "").strip()
        ]
        return " ".join(items) if items else None
    return None


def _coordinate_quad(value: object) -> tuple[float, float, float, float] | None:
    """Four finite numbers, or nothing."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 4:
        return None
    try:
        quad = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if any(item != item or item in (float("inf"), float("-inf")) for item in quad):
        return None
    return quad  # type: ignore[return-value]


def _descendant_bboxes(node: Mapping[str, object]):
    for child in node.get("blocks") or ():
        if not isinstance(child, Mapping):
            continue
        quad = _coordinate_quad(child.get("bbox"))
        if quad is not None:
            yield quad
        yield from _descendant_bboxes(child)


def _bbox_from_children(middle: Mapping[str, object]) -> list[float] | None:
    """The extent of a group MinerU never computed one for.

    A figure group with a caption but no image region -- MinerU emits
    ``{"type": "image", "bbox": [], "blocks": [{"type": "image_caption", ...}]}``
    -- has no bbox of its own because there is no image sub-block to take one
    from.  Its children have real geometry, and a group's extent is the union
    of what it contains, so the caption is placed on the page rather than
    discarded.  One such block failing a translation that must stay one-to-one
    with its input took a whole book down with it.
    """

    quads = list(_descendant_bboxes(middle))
    if not quads:
        return None
    return [
        min(q[0] for q in quads),
        min(q[1] for q in quads),
        max(q[2] for q in quads),
        max(q[3] for q in quads),
    ]


def _middle_json(block: Mapping[str, object]) -> Mapping[str, object]:
    value = block.get("middle_json", {})
    middle = dict(require_mapping(value, "middle_json"))
    # MinerU 3.x's supported content_list.json surface exposes the 0..1000
    # bbox at the top level.  Preserve its declared coordinate scale instead
    # of dropping geometry when no legacy reconciliation payload is present.
    if "bbox" not in middle and block.get("bbox") is not None:
        middle["bbox"] = block.get("bbox")
        middle["coordinate_space"] = CoordinateSpace.PARSER_PAGE.value
        middle["page_width"] = 1000
        middle["page_height"] = 1000
    if _coordinate_quad(middle.get("bbox")) is None:
        derived = _bbox_from_children(middle)
        if derived is not None:
            middle["bbox"] = derived
    return middle


def _block_type(block: Mapping[str, object], middle: Mapping[str, object]) -> tuple[BlockType, str | None]:
    raw = str(block.get("type", "unknown")).strip().lower()
    middle_type = str(middle.get("type", "")).strip().lower()
    subtype = _non_empty_text(
        block.get("semantic_subtype", block.get("sub_type"))
    )
    subtype_label = (
        subtype.strip().casefold().replace("-", "_").replace(" ", "_")
        if subtype is not None
        else None
    )
    if raw in {"text", "title", "paragraph", "doc_title", "paragraph_title"}:
        text_level = block.get("text_level")
        has_title_level = (
            isinstance(text_level, int)
            and not isinstance(text_level, bool)
            and text_level >= 0
        ) or (isinstance(text_level, str) and text_level.isdigit())
        # MinerU 3.x reports the document's own title separately from section
        # headings.  Keeping them apart is the whole point: the outline pass no
        # longer has to infer which of several equal-looking titles names the
        # document, which is what let a mid-document heading adopt most of a
        # document's blocks.
        if raw == "doc_title" or middle_type == "doc_title":
            return BlockType.DOC_TITLE, subtype
        if (
            raw in {"title", "paragraph_title"}
            or middle_type in {"title", "paragraph_title"}
            or has_title_level
        ):
            return BlockType.TITLE, subtype
        if middle_type in {"list", "list_item"}:
            return BlockType.LIST_ITEM, subtype
        if middle_type in {"header", "page_header"}:
            return BlockType.HEADER, subtype
        if middle_type in {"footer", "page_footer"}:
            return BlockType.FOOTER, subtype
        return BlockType.PARAGRAPH, subtype
    if raw in _DIAGRAM_SUBTYPES or subtype_label in _DIAGRAM_SUBTYPES:
        return BlockType.DIAGRAM, subtype_label or raw
    if raw in {"image", "image_body", "header_image", "footer_image"}:
        return BlockType.FIGURE, subtype or ("figure" if raw == "image" else raw)
    if raw in {"chart", "chart_body"}:
        return BlockType.CHART, subtype or "chart"
    if raw in {"table", "table_body"}:
        return BlockType.TABLE, subtype
    # interline_equation is what MinerU 3.x actually emits for a display
    # equation; matching only "equation" sent every one of them to UNKNOWN.
    # inline_equation is deliberately absent: it is a span inside a sentence,
    # never a block, and is transcribed in place by transcribe_inline_equations.
    if raw in {"equation", "interline_equation"}:
        return BlockType.EQUATION, subtype or (
            raw if raw != "equation" else None
        )
    # "(3.1)" beside an equation is its label, not mathematics. As an equation
    # node it would carry no formula and earn its own index records.
    if raw == "formula_number":
        return BlockType.CAPTION, subtype or raw
    if raw in {"code", "code_body"}:
        return BlockType.CODE, subtype
    if raw == "algorithm":
        return BlockType.ALGORITHM, subtype
    if raw == "abstract":
        return BlockType.ABSTRACT, subtype
    if raw == "ref_text":
        return BlockType.REFERENCE, subtype
    if raw in {"aside_text", "vertical_text", "phonetic"}:
        return BlockType.ASIDE, subtype or raw
    if raw == "discarded":
        return BlockType.DISCARDED, subtype or raw
    if raw in {"list", "index"}:
        return BlockType.LIST_ITEM, subtype or raw
    if raw in {
        "caption",
        "image_caption",
        "table_caption",
        "chart_caption",
        "code_caption",
        "algorithm_caption",
    }:
        return BlockType.CAPTION, subtype or raw
    if raw in {"header", "page_header"}:
        return BlockType.HEADER, subtype
    if raw in {"footer", "page_footer", "page_number"}:
        return BlockType.FOOTER, subtype or raw
    if raw in {
        "page_footnote",
        "footnote",
        "image_footnote",
        "table_footnote",
        "chart_footnote",
        "code_footnote",
    }:
        return BlockType.FOOTNOTE, subtype or (
            raw if raw != "footnote" else None
        )
    if subtype is not None:
        return BlockType.UNKNOWN, subtype
    return BlockType.UNKNOWN, None if raw == "unknown" else raw


def _geometry(middle: Mapping[str, object]) -> SourceGeometry | None:
    bbox_value = middle.get("bbox")
    polygon_value = middle.get("polygon", ())
    bbox: BoundingBox | None = None
    polygon: tuple[Point, ...] = ()
    if bbox_value is not None:
        if not isinstance(bbox_value, Sequence) or isinstance(
            bbox_value, (str, bytes, bytearray)
        ):
            raise ContractValidationError("middle_json.bbox must be an array")
        # An empty bbox is MinerU declining to place the block, not a
        # malformed one. Absent geometry is already a supported outcome
        # below; refusing it here instead failed the whole document, which
        # cannot be the right answer to one unplaced caption.
        if len(bbox_value) > 0:
            bbox = BoundingBox.from_sequence(bbox_value)
    if polygon_value:
        if not isinstance(polygon_value, Sequence) or isinstance(
            polygon_value, (str, bytes, bytearray)
        ):
            raise ContractValidationError("middle_json.polygon must be an array")
        points: list[Point] = []
        for index, value in enumerate(polygon_value):
            if isinstance(value, Mapping):
                points.append(Point.from_dict(value))
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                if len(value) != 2:
                    raise ContractValidationError(
                        f"middle_json.polygon[{index}] must contain x and y"
                    )
                points.append(Point(value[0], value[1]))  # type: ignore[arg-type]
            else:
                raise ContractValidationError(
                    f"middle_json.polygon[{index}] must be a point"
                )
        polygon = tuple(points)
    if bbox is None and not polygon:
        return None
    coordinate_value = middle.get("coordinate_space", CoordinateSpace.PARSER_PAGE.value)
    try:
        coordinate_space = CoordinateSpace(coordinate_value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"unsupported middle_json.coordinate_space: {coordinate_value!r}"
        ) from exc
    width = middle.get("page_width")
    height = middle.get("page_height")
    return SourceGeometry(
        coordinate_space=coordinate_space,
        bbox=bbox,
        polygon=polygon,
        page_width=width,  # type: ignore[arg-type]
        page_height=height,  # type: ignore[arg-type]
    )


def _reading_order(block: Mapping[str, object], middle: Mapping[str, object]) -> float | None:
    value = block.get("reading_order", middle.get("index"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractValidationError("reading order must be a finite number")
    return float(value)


def _scores(value: object) -> list[float]:
    scores: list[float] = []
    if isinstance(value, Mapping):
        score = value.get("score")
        if (
            not isinstance(score, bool)
            and isinstance(score, (int, float))
            and math.isfinite(float(score))
            and 0 <= float(score) <= 1
        ):
            scores.append(float(score))
        for child in value.values():
            scores.extend(_scores(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            scores.extend(_scores(child))
    return scores


def _confidence(block: Mapping[str, object], middle: Mapping[str, object]) -> float | None:
    explicit = block.get("confidence")
    if explicit is not None:
        if (
            isinstance(explicit, bool)
            or not isinstance(explicit, (int, float))
            or not math.isfinite(float(explicit))
            or not 0 <= float(explicit) <= 1
        ):
            raise ContractValidationError("confidence must be between 0 and 1")
        return float(explicit)
    scores = _scores(middle)
    return sum(scores) / len(scores) if scores else None


def _add_text_representation(
    destination: list[Representation],
    *,
    kind: RepresentationKind,
    value: object,
    media_type: str,
    confidence: float | None,
    language: str | None,
    source_field: str | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> None:
    text = _joined_text(value)
    if text is not None:
        metadata: dict[str, object] = {}
        if source_field is not None:
            metadata["assertion_origin"] = "mineru_explicit_normalized_field"
            metadata["source_field"] = source_field
        if extra_metadata:
            metadata.update(extra_metadata)
        destination.append(
            Representation(
                kind=kind,
                content=text,
                media_type=media_type,
                confidence=confidence,
                language=language,
                metadata=metadata,
            )
        )


# MinerU 3.x keeps a list's text in ``list_items`` and a code block's in
# ``code_body``; a block of either kind carries no ``text`` key at all. Anything
# reading text out of a raw MinerU content-list block goes through the accessor
# below, so this layout is known here and nowhere else. Re-deriving it per
# consumer is what let reconciliation read every list as empty while recovery,
# which had been taught the same fact separately, read them correctly.
_TEXT_BEARING_KEYS = ("text", "list_items", "code_body")

# MinerU wraps a listing in how it wants the listing displayed: an algorithm
# arrives inside a styled <div>, with its angle brackets HTML-escaped, and a code
# block inside a fenced Markdown span. Both wrappers reached the tree verbatim,
# so what got embedded, summarised and rendered for
# "If N < 1 / Then print('Empty Vector') / Exit" was
#   <div class="mineru-algorithm" style="white-space: pre-wrap; ...">
#   If $N &lt; 1$ ... </div>
# Stripping it here rather than at each reader keeps one owner for MinerU's
# layout, the same reason this module owns which key the text lives under. The
# listing's own line breaks are what is left behind, because a listing without
# them is not a listing.
_ALGORITHM_WRAPPER = re.compile(
    r"^\s*<div\b[^>]*>(?P<body>.*?)</div>\s*$", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE = re.compile(
    r"^\s*```(?P<language>[^\n`]*)\n(?P<body>.*?)\n?```\s*$", re.DOTALL
)
# MinerU names the language on the fence, and the name is worth keeping: on
# lecs106.pdf the ```python blocks are the programs and the ```txt blocks are
# their console transcripts -- "How many elements in your list? 4 ..." was
# stored as code because the tag was the only thing separating them and it was
# being discarded. It is also what a frontend needs to highlight a listing.
_NO_LANGUAGE = frozenset({"", "txt", "text", "plain", "plaintext", "none"})


def fence_language(text: object) -> str | None:
    """The language MinerU named on a listing's fence, if it named one."""

    if not isinstance(text, str):
        return None
    match = _CODE_FENCE.match(text)
    if match is None:
        return None
    language = match.group("language").strip().casefold()
    return None if language in _NO_LANGUAGE else language


def unwrap_presentation(text: str) -> str:
    """A listing's own text, without the markup MinerU wrapped it in."""

    for pattern in (_ALGORITHM_WRAPPER, _CODE_FENCE):
        match = pattern.match(text)
        if match:
            # Only a wrapper is removed, never nested markup: the body of a
            # <div> is unescaped because MinerU escaped it to put it there, but
            # a fenced block was never escaped and must not be touched.
            body = match.group("body")
            if pattern is _ALGORITHM_WRAPPER:
                body = unescape(body)
            return body.strip("\n")
    return text


def content_block_text_parts(block: Mapping[str, object]) -> tuple[str, ...]:
    """The text pieces a raw MinerU content-list block carries, in reading order.

    One piece per list item, so callers can join them the way their own
    representation needs without knowing which key they came from.

    A listing is returned unwrapped -- see ``unwrap_presentation``.
    """

    for key in _TEXT_BEARING_KEYS:
        value = block.get(key)
        if isinstance(value, str):
            if value.strip():
                return (unwrap_presentation(value),)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            parts = tuple(
                item for item in value if isinstance(item, str) and item.strip()
            )
            if parts:
                return parts
    return ()


def _list_markdown(value: object) -> str | None:
    """Preserve MinerU list-item boundaries in a textual representation."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return _joined_text(value)
    items: list[str] = []
    for raw in value:
        text = _joined_text(raw)
        if not text:
            continue
        # MinerU may retain a visual bullet in the OCR text. The Markdown
        # marker carries that structure, so keeping both would render two.
        text = re.sub(r"^\s*(?:[•●▪◦‣⁃*-]|\d+[.)])\s*", "", text).strip()
        if text:
            items.append(f"- {text}")
    return "\n".join(items) or None


def _add_structured_representation(
    destination: list[Representation],
    *,
    kind: RepresentationKind,
    value: object,
    confidence: float | None,
    source_field: str,
) -> None:
    """Preserve a parser-declared JSON representation without deriving it."""

    if value is None:
        return
    preflight_explicit_json(value, field=source_field)
    destination.append(
        Representation(
            kind=kind,
            content=value,  # type: ignore[arg-type]
            media_type="application/json",
            confidence=confidence,
            metadata={
                "assertion_origin": "mineru_explicit_normalized_field",
                "source_field": source_field,
            },
        )
    )


# Every canonical type _block_type can return. Declared once so the manifest
# and the translation cannot drift apart: the parser port validates each
# observation against the manifest, and a type this adapter maps but does not
# declare aborts the document rather than degrading it.
DECLARED_BLOCK_TYPES = frozenset(
    {
        BlockType.DOC_TITLE,
        BlockType.TITLE,
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.EQUATION,
        BlockType.FIGURE,
        BlockType.CHART,
        BlockType.DIAGRAM,
        BlockType.CODE,
        BlockType.ALGORITHM,
        BlockType.ABSTRACT,
        BlockType.REFERENCE,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE,
        BlockType.DISCARDED,
        BlockType.UNKNOWN,
    }
)

_TEXT_REPRESENTED_BLOCKS = frozenset(
    {
        BlockType.DOC_TITLE,
        BlockType.TITLE,
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
        BlockType.ABSTRACT,
        BlockType.REFERENCE,
        BlockType.ALGORITHM,
        BlockType.ASIDE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.FOOTNOTE,
        BlockType.CAPTION,
        BlockType.CODE,
        BlockType.DISCARDED,
        BlockType.UNKNOWN,
    }
)


def _representations(
    block: Mapping[str, object],
    block_type: BlockType,
    confidence: float | None,
    language: str | None,
) -> tuple[Representation, ...]:
    result: list[Representation] = []
    # Every block type whose evidence is its own text. A type missing here
    # produces an observation with no representation and no asset, which the
    # contract rejects outright -- so the document fails to bind rather than
    # degrading. MinerU 3.x emits aside_text, page_number and doc_title on
    # ordinary pages, and all three landed here.
    if block_type in _TEXT_REPRESENTED_BLOCKS:
        is_list = block_type == BlockType.LIST_ITEM and block.get("text") is None
        wraps_a_listing = (
            block_type == BlockType.CODE
            and block.get("text") is None
            and isinstance(block.get("code_body"), str)
        )
        # The fence tag is the parser's own answer to "what language is this".
        # It is recorded rather than acted on here: an untagged listing is a
        # listing whose language MinerU could not name, which is not the same
        # claim as "this is not a program".
        #
        # Only a listing carrying its own line breaks is taken to have a
        # language. On one line MinerU has too little to go on, and it shows:
        # the twelve single-line fragments of one Python insertion sort in
        # 016.0 SORTING.pdf were tagged javascript, lua, txt and python, so the
        # panel would have labelled `print("Original list is ", aList)` as Lua.
        # No tag is better than a wrong one.
        listing_language = None
        if wraps_a_listing and "\n" in unwrap_presentation(block["code_body"]).strip():
            listing_language = fence_language(block["code_body"])
        _add_text_representation(
            result,
            kind=(
                RepresentationKind.MARKDOWN
                if is_list
                else RepresentationKind.TEXT
            ),
            value=(
                # The representation is what every downstream reader selects,
                # so the wrapper has to come off here and not only in
                # content_block_text_parts -- stripping it there alone left the
                # <div> in the canonical text and changed nothing in the tree.
                unwrap_presentation(block["code_body"])
                if wraps_a_listing
                else _list_markdown(block.get("list_items"))
                if is_list
                else block.get("text")
            ),
            media_type="text/markdown" if is_list else "text/plain",
            confidence=confidence,
            language=language,
            source_field="list_items" if is_list else None,
            extra_metadata=(
                {"listing_language": listing_language} if listing_language else None
            ),
        )
    if block_type == BlockType.TABLE:
        _add_text_representation(
            result,
            kind=RepresentationKind.HTML,
            value=block.get("table_body"),
            media_type="text/html",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=block.get("table_caption"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=block.get("table_footnote"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    elif block_type in {BlockType.FIGURE, BlockType.DIAGRAM}:
        _add_text_representation(
            result,
            kind=RepresentationKind.VISUAL_DESCRIPTION,
            value=block.get("image_content", block.get("content")),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=block.get("image_caption"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=block.get("image_footnote"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    elif block_type == BlockType.CHART:
        _add_text_representation(
            result,
            kind=RepresentationKind.MARKDOWN,
            value=block.get("content", block.get("text")),
            media_type="text/markdown",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.CAPTION,
            value=block.get("chart_caption"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=block.get("chart_footnote"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
        _add_text_representation(
            result,
            kind=RepresentationKind.FOOTNOTE,
            value=block.get("image_footnote"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    elif block_type == BlockType.EQUATION:
        raw_text = _joined_text(block.get("text"))
        if raw_text is not None:
            raw_format = block.get("text_format")
            explicit_format = (
                raw_format.strip().casefold()
                if isinstance(raw_format, str) and raw_format.strip()
                else None
            )
            is_latex = explicit_format in {"latex", "tex"}
            result.append(
                Representation(
                    kind=(
                        RepresentationKind.LATEX
                        if is_latex
                        else RepresentationKind.TEXT
                    ),
                    content=raw_text,
                    media_type=("application/x-latex" if is_latex else "text/plain"),
                    confidence=confidence,
                    metadata={
                        "source_field": "text",
                        "format": explicit_format or "unknown",
                        "assertion_origin": "mineru_normalized_formula_text",
                    },
                )
            )

    # These fields are optional extensions to MinerU's normalized stream.  They
    # are copied only when the worker explicitly provides them; this bridge
    # never reconstructs a grid, OCR transcript, graph, MathML tree, or chart
    # series from HTML, geometry, or prose.
    if block.get("mathml") is not None:
        preflight_mathml(block["mathml"], field="mathml")
    _add_text_representation(
        result,
        kind=RepresentationKind.OCR_TEXT,
        value=block.get("ocr_text"),
        media_type="text/plain",
        confidence=confidence,
        language=language,
        source_field="ocr_text",
    )
    _add_text_representation(
        result,
        kind=RepresentationKind.MATHML,
        value=block.get("mathml"),
        media_type="application/mathml+xml",
        confidence=confidence,
        language=None,
        source_field="mathml",
    )
    _add_structured_representation(
        result,
        kind=RepresentationKind.TABLE_GRID,
        value=block.get("table_grid"),
        confidence=confidence,
        source_field="table_grid",
    )
    _add_structured_representation(
        result,
        kind=RepresentationKind.DIAGRAM_GRAPH,
        value=block.get("diagram_graph"),
        confidence=confidence,
        source_field="diagram_graph",
    )
    _add_structured_representation(
        result,
        kind=RepresentationKind.STRUCTURED_DATA,
        value=block.get("chart_data"),
        confidence=confidence,
        source_field="chart_data",
    )
    if not result:
        # Last resort: the block's own text, whatever its type says its evidence
        # channels are. This exists for text the native-PDF recovery pass puts
        # back into a block whose type does not read that channel -- a table
        # whose body was never recognised gets its page text written to "text",
        # and every table representation is built from table_body, caption and
        # footnote, so the recovered text was invisible. The block then carried
        # neither a representation nor an asset, and one such block aborts the
        # whole document: a 93 MB scan died on it after three hours.
        #
        # Only when nothing else produced anything, so no existing block changes
        # shape, and text that was recovered is represented instead of discarded.
        _add_text_representation(
            result,
            kind=RepresentationKind.TEXT,
            value=block.get("text"),
            media_type="text/plain",
            confidence=confidence,
            language=language,
        )
    return tuple(result)


def _assets(block: Mapping[str, object]) -> tuple[AssetRef, ...]:
    path = _non_empty_text(block.get("img_path"))
    path_semantics = "legacy_bookrag_normalized"
    if path is None:
        middle = _middle_json(block)

        def find_image_path(value: object) -> str | None:
            if isinstance(value, Mapping):
                candidate = _non_empty_text(value.get("image_path"))
                if candidate is not None:
                    return candidate
                for child in value.values():
                    candidate = find_image_path(child)
                    if candidate is not None:
                        return candidate
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for child in value:
                    candidate = find_image_path(child)
                    if candidate is not None:
                        return candidate
            return None

        path = find_image_path(middle)
        path_semantics = "mineru_middle_json_relative"
    if path is None:
        return ()
    media_type = _non_empty_text(block.get("img_media_type"))
    if media_type is None:
        media_type, _encoding = mimetypes.guess_type(path)
    asset_hash = block.get("img_sha256")
    if asset_hash is not None:
        require_sha256(asset_hash, "img_sha256")
    return (
        AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=path,
            media_type=media_type,
            sha256=asset_hash,  # type: ignore[arg-type]
            metadata={"path_semantics": path_semantics},
        ),
    )


def _native_base(block: Mapping[str, object], middle: Mapping[str, object]) -> str:
    for key in ("parser_native_id", "block_id"):
        value = block.get(key)
        if value is not None and str(value).strip() and str(value).strip() != "-1":
            return f"{key}:{value}"
    for key in ("id", "block_id"):
        value = middle.get(key)
        if value is not None and str(value).strip():
            return f"middle_{key}:{value}"
    signature = {
        "page_idx": block.get("page_idx"),
        "type": block.get("type"),
        "bbox": middle.get("bbox"),
        "reading_order": block.get("reading_order", middle.get("index")),
        "text": block.get("text"),
        "table_body": block.get("table_body"),
    }
    return f"compat:{content_sha256(signature)[:32]}"


_GENERIC_ID_FIELDS = frozenset(
    {
        "parser_native_id",
        "block_id",
        "id",
        "uid",
        "uuid",
        "content_id",
        "element_id",
        "para_id",
        "paragraph_id",
    }
)
_POSITION_ID_FIELDS = frozenset({"block_idx", "para_idx", "index"})
_REFERENCE_ID_FIELDS = (
    "target_id",
    "target_block_id",
    "parser_native_id",
    "block_id",
    "id",
    "uid",
    "uuid",
    "content_id",
    "element_id",
    "para_id",
    "paragraph_id",
    "block_idx",
    "para_idx",
    "index",
    "$ref",
    "ref",
)

# The names below are deliberately finite.  Free-form keys, shared group IDs,
# geometric proximity, and text similarity are not structural assertions.
_STRUCTURAL_FIELDS: dict[str, tuple[RelationType, bool]] = {
    "parent_id": (RelationType.CHILD_OF, False),
    "parent_block_id": (RelationType.CHILD_OF, False),
    "parent_ref": (RelationType.CHILD_OF, False),
    "child_id": (RelationType.PARENT_OF, False),
    "child_ids": (RelationType.PARENT_OF, False),
    "children_ids": (RelationType.PARENT_OF, False),
    "child_ref": (RelationType.PARENT_OF, False),
    "child_refs": (RelationType.PARENT_OF, False),
    "children_refs": (RelationType.PARENT_OF, False),
    "caption_id": (RelationType.HAS_CAPTION, False),
    "caption_ids": (RelationType.HAS_CAPTION, False),
    "caption_block_id": (RelationType.HAS_CAPTION, False),
    "caption_block_ids": (RelationType.HAS_CAPTION, False),
    "caption_ref": (RelationType.HAS_CAPTION, False),
    "caption_refs": (RelationType.HAS_CAPTION, False),
    "caption_of_id": (RelationType.CAPTION_OF, False),
    "caption_parent_id": (RelationType.CAPTION_OF, False),
    "caption_of_ref": (RelationType.CAPTION_OF, False),
    "footnote_id": (RelationType.HAS_FOOTNOTE, False),
    "footnote_ids": (RelationType.HAS_FOOTNOTE, False),
    "footnote_block_id": (RelationType.HAS_FOOTNOTE, False),
    "footnote_block_ids": (RelationType.HAS_FOOTNOTE, False),
    "footnote_ref": (RelationType.HAS_FOOTNOTE, False),
    "footnote_refs": (RelationType.HAS_FOOTNOTE, False),
    "footnote_of_id": (RelationType.FOOTNOTE_OF, False),
    "footnote_parent_id": (RelationType.FOOTNOTE_OF, False),
    "footnote_of_ref": (RelationType.FOOTNOTE_OF, False),
    "reference_id": (RelationType.REFERENCES, True),
    "reference_ids": (RelationType.REFERENCES, True),
    "reference_ref": (RelationType.REFERENCES, True),
    "reference_refs": (RelationType.REFERENCES, True),
    "referenced_block_id": (RelationType.REFERENCES, True),
    "referenced_block_ids": (RelationType.REFERENCES, True),
    "referenced_by_id": (RelationType.REFERENCED_BY, True),
    "referenced_by_ids": (RelationType.REFERENCED_BY, True),
    "reading_next_id": (RelationType.READING_NEXT, False),
    "next_block_id": (RelationType.READING_NEXT, False),
    "next_id": (RelationType.READING_NEXT, False),
    "reading_previous_id": (RelationType.READING_PREVIOUS, False),
    "previous_block_id": (RelationType.READING_PREVIOUS, False),
    "previous_id": (RelationType.READING_PREVIOUS, False),
    "prev_id": (RelationType.READING_PREVIOUS, False),
    "next_table_id": (RelationType.TABLE_CONTINUES, True),
    "continued_by_id": (RelationType.TABLE_CONTINUES, True),
    "continues_to_id": (RelationType.TABLE_CONTINUES, True),
    "table_continuation_id": (RelationType.TABLE_CONTINUES, True),
    "previous_table_id": (RelationType.TABLE_CONTINUED_FROM, True),
    "continues_from_id": (RelationType.TABLE_CONTINUED_FROM, True),
    "continuation_of_id": (RelationType.TABLE_CONTINUED_FROM, True),
}
_ROOT_REFERENCE_FIELDS = frozenset(
    {"parent_id", "parent_block_id", "parent_ref"}
)


def _id_scalar(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(int(value)) if value.is_integer() else repr(value)
    return None


def _qualified_id_tokens(field: str, value: object) -> tuple[str, ...]:
    scalar = _id_scalar(value)
    if scalar is None:
        return ()
    result = {f"{field}:{scalar}"}
    if field in _GENERIC_ID_FIELDS:
        result.add(f"id:{scalar}")
    if field == "parser_native_id":
        result.add(f"native:{scalar}")
        result.add(f"native:parser_native_id:{scalar}")
    return tuple(sorted(result))


def _observation_identity_tokens(
    block: Mapping[str, object],
    middle: Mapping[str, object],
    observation: ParserBlockObservation,
) -> tuple[str, ...]:
    result = {
        f"native:{observation.provenance.parser_native_id}",
        f"id:{observation.provenance.parser_native_id}",
    }
    for value in (block, middle):
        for field in (*sorted(_GENERIC_ID_FIELDS), *sorted(_POSITION_ID_FIELDS)):
            result.update(_qualified_id_tokens(field, value.get(field)))
    return tuple(sorted(result))


def _walk_nested_mappings(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_nested_mappings(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from _walk_nested_mappings(child)


def _container_identity_tokens(
    block: Mapping[str, object], middle: Mapping[str, object]
) -> tuple[str, ...]:
    """Index non-observation/group IDs only to explain refused references."""

    result: set[str] = set()
    for value in (block, middle):
        group = _id_scalar(value.get("group_id"))
        if group is not None:
            result.add(f"group_id:{group}")
            result.add(f"id:{group}")
        for key, child in value.items():
            if value is block and key == "middle_json":
                continue
            for nested in _walk_nested_mappings(child):
                for field in (
                    *sorted(_GENERIC_ID_FIELDS),
                    *sorted(_POSITION_ID_FIELDS),
                    "group_id",
                ):
                    result.update(_qualified_id_tokens(field, nested.get(field)))
                    if field == "group_id":
                        scalar = _id_scalar(nested.get(field))
                        if scalar is not None:
                            result.add(f"id:{scalar}")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class _NativeReference:
    raw: object
    tokens: tuple[str, ...]
    confidence: float | None = None
    container_only: bool = False
    invalid_reason: str | None = None


def _native_reference(value: object, *, source_field: str) -> _NativeReference:
    if source_field in _ROOT_REFERENCE_FIELDS and _id_scalar(value) in {"-1", "0"}:
        return _NativeReference(value, (), container_only=True)

    confidence: float | None = None
    if isinstance(value, Mapping):
        raw_confidence = value.get("confidence")
        if raw_confidence is not None:
            if (
                isinstance(raw_confidence, bool)
                or not isinstance(raw_confidence, (int, float))
                or not math.isfinite(float(raw_confidence))
                or not 0 <= float(raw_confidence) <= 1
            ):
                return _NativeReference(
                    value, (), invalid_reason="confidence_must_be_probability"
                )
            confidence = float(raw_confidence)
        group = _id_scalar(value.get("group_id"))
        if group is not None:
            return _NativeReference(
                value,
                (f"group_id:{group}", f"id:{group}"),
                confidence=confidence,
                container_only=True,
            )
        if source_field in _ROOT_REFERENCE_FIELDS:
            for field in _REFERENCE_ID_FIELDS:
                if _id_scalar(value.get(field)) in {"-1", "0"}:
                    return _NativeReference(
                        value, (), confidence=confidence, container_only=True
                    )
        tokens: set[str] = set()
        for field in _REFERENCE_ID_FIELDS:
            raw_id = value.get(field)
            if raw_id is None:
                continue
            if field in {"$ref", "ref", "target_id", "target_block_id"}:
                scalar = _id_scalar(raw_id)
                if scalar is not None:
                    tokens.update((f"id:{scalar}", f"native:{scalar}"))
                    if field == "target_block_id":
                        tokens.add(f"block_id:{scalar}")
            else:
                tokens.update(_qualified_id_tokens(field, raw_id))
        if not tokens:
            return _NativeReference(
                value, (), confidence=confidence, invalid_reason="missing_supported_id"
            )
        return _NativeReference(value, tuple(sorted(tokens)), confidence=confidence)

    scalar = _id_scalar(value)
    if scalar is None:
        return _NativeReference(value, (), invalid_reason="id_must_be_scalar_or_object")
    return _NativeReference(
        value,
        (f"id:{scalar}", f"native:{scalar}"),
        confidence=confidence,
    )


def _reference_values(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(value)
    return (value,)


def _reading_order_source(
    block: Mapping[str, object], middle: Mapping[str, object]
) -> tuple[float, str] | None:
    if block.get("reading_order") is not None:
        return float(block["reading_order"]), "reading_order"  # validated earlier
    if middle.get("index") is not None:
        return float(middle["index"]), "middle_json.index"  # validated earlier
    return None


def _translate_one(
    request: ParseRequest,
    block: Mapping[str, object],
    *,
    native_id: str,
    parser_version: str,
    model_version: str,
    config_sha256: str,
    raw_artifact_uri: str | None,
) -> ParserBlockObservation:
    page_index = block.get("page_idx")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
        raise ContractValidationError("page_idx must be a non-negative integer")
    page = request.page_by_index.get(page_index)
    if page is None:
        raise ContractValidationError(f"page_idx {page_index} is outside requested pages")
    middle = _middle_json(block)
    block_type, semantic_subtype = _block_type(block, middle)
    confidence = _confidence(block, middle)
    language = request.options.languages[0] if len(request.options.languages) == 1 else None
    representations = _representations(block, block_type, confidence, language)
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
        normalized_subtype = (semantic_subtype or "").casefold().replace("-", "_")
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
    assets = _assets(block)
    block_raw_artifact = _non_empty_text(block.get("raw_artifact_uri")) or raw_artifact_uri
    metadata: dict[str, object] = {
        "legacy_type": str(block.get("type", "unknown")),
        "normalized_block_sha256": content_sha256(block),
        "translation": TRANSLATION_NAME,
    }
    if block.get("text_level") is not None:
        metadata["text_level"] = block.get("text_level")
    if block.get("pdf_id") is not None:
        metadata["legacy_pdf_id"] = block.get("pdf_id")
    service_metadata = block.get("_mineru_service")
    if service_metadata is not None:
        metadata["mineru_service"] = dict(
            require_mapping(service_metadata, "_mineru_service")
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
            raw_artifact_uri=block_raw_artifact,
            transformations=(TRANSLATION_NAME,),
        ),
        block_type=block_type,
        semantic_subtype=semantic_subtype,
        geometry=_geometry(middle),
        reading_order=_reading_order(block, middle),
        representations=representations,
        semantic_substructures=tuple(semantic_substructures),
        assets=assets,
        confidence=confidence,
        metadata=metadata,
    )


def _attach_structural_relations(
    translated: Sequence[
        tuple[Mapping[str, object], Mapping[str, object], ParserBlockObservation]
    ],
) -> tuple[ParserBlockObservation, ...]:
    """Resolve only explicit MinerU-normalized structure, fail closed.

    Ordinary relations resolve on the source page.  Only parser-declared
    cross-page references and table continuations may fall back to a globally
    unique target.  Numeric reading order produces adjacent links only when
    every accepted observation on a page has one unique explicit order value.
    """

    by_token: dict[str, set[str]] = {}
    by_observation_id = {item[2].observation_id: item[2] for item in translated}
    container_tokens: set[str] = set()
    for block, middle, observation in translated:
        for token in _observation_identity_tokens(block, middle, observation):
            by_token.setdefault(token, set()).add(observation.observation_id)
        container_tokens.update(_container_identity_tokens(block, middle))

    assertions: dict[str, list[ObservationRelation]] = {
        observation_id: [] for observation_id in by_observation_id
    }
    records: dict[str, list[dict[str, object]]] = {
        observation_id: [] for observation_id in by_observation_id
    }

    def resolve(
        source: ParserBlockObservation,
        reference: _NativeReference,
        *,
        allow_cross_page: bool,
    ) -> tuple[ParserBlockObservation | None, str]:
        if reference.invalid_reason is not None:
            return None, "invalid_reference"
        if reference.container_only:
            return None, "container_only"
        candidate_ids: set[str] = set()
        for token in reference.tokens:
            candidate_ids.update(by_token.get(token, ()))
        same_page = {
            candidate_id
            for candidate_id in candidate_ids
            if by_observation_id[candidate_id].page_index == source.page_index
        }
        if len(same_page) == 1:
            return by_observation_id[next(iter(same_page))], "resolved_same_page"
        if len(same_page) > 1:
            return None, "ambiguous_same_page"
        if allow_cross_page:
            if len(candidate_ids) == 1:
                return (
                    by_observation_id[next(iter(candidate_ids))],
                    "resolved_globally_unique_cross_page",
                )
            if len(candidate_ids) > 1:
                return None, "ambiguous_cross_page"
        elif candidate_ids:
            return None, "cross_page_not_allowed"
        if any(token in container_tokens for token in reference.tokens):
            return None, "container_only"
        return None, "unresolved"

    def add_explicit_reference(
        source: ParserBlockObservation,
        *,
        reference: _NativeReference,
        relation_type: RelationType,
        source_location: str,
        source_field: str,
        source_value_index: int,
        allow_cross_page: bool,
    ) -> None:
        target, status = resolve(
            source, reference, allow_cross_page=allow_cross_page
        )
        record: dict[str, object] = {
            "relation_type": relation_type.value,
            "source_location": source_location,
            "source_field": source_field,
            "source_value_index": source_value_index,
            "target_ref": reference.raw,  # type: ignore[dict-item]
            "target_tokens": list(reference.tokens),
            "status": status,
        }
        if reference.invalid_reason is not None:
            record["invalid_reason"] = reference.invalid_reason
        if target is not None:
            record["target_observation_id"] = target.observation_id
        records[source.observation_id].append(record)
        if target is None:
            return
        if target.observation_id == source.observation_id:
            record["status"] = "self_target_omitted"
            return
        assertions[source.observation_id].append(
            ObservationRelation(
                relation_type=relation_type,
                target_observation_id=target.observation_id,
                confidence=(
                    1.0 if reference.confidence is None else reference.confidence
                ),
                metadata={
                    "assertion_origin": "mineru_explicit_native_reference",
                    "source_native_id": source.provenance.parser_native_id,
                    "source_location": source_location,
                    "source_field": source_field,
                    "source_value_index": source_value_index,
                    "target_ref": reference.raw,
                    "resolution_status": status,
                },
            )
        )

    for block, middle, source in translated:
        for source_location, value in (("block", block), ("middle_json", middle)):
            for source_field, (relation_type, allow_cross_page) in (
                _STRUCTURAL_FIELDS.items()
            ):
                if source_field not in value:
                    continue
                for source_value_index, raw_reference in enumerate(
                    _reference_values(value[source_field])
                ):
                    if relation_type in {
                        RelationType.TABLE_CONTINUES,
                        RelationType.TABLE_CONTINUED_FROM,
                    } and source.block_type is not BlockType.TABLE:
                        records[source.observation_id].append(
                            {
                                "relation_type": relation_type.value,
                                "source_location": source_location,
                                "source_field": source_field,
                                "source_value_index": source_value_index,
                                "target_ref": raw_reference,
                                "target_tokens": [],
                                "status": "invalid_source_type",
                                "required_source_type": BlockType.TABLE.value,
                            }
                        )
                        continue
                    add_explicit_reference(
                        source,
                        reference=_native_reference(
                            raw_reference, source_field=source_field
                        ),
                        relation_type=relation_type,
                        source_location=source_location,
                        source_field=source_field,
                        source_value_index=source_value_index,
                        allow_cross_page=allow_cross_page,
                    )

    # A parser-native numeric order is a total order only when every translated
    # block on that page declares it and no positions collide.  Anything less
    # is retained as review metadata and never bridged by spatial guessing.
    by_page: dict[
        int,
        list[
            tuple[
                Mapping[str, object],
                Mapping[str, object],
                ParserBlockObservation,
            ]
        ],
    ] = {}
    for entry in translated:
        by_page.setdefault(entry[2].page_index, []).append(entry)
    for page_entries in by_page.values():
        ordered: list[tuple[float, str, ParserBlockObservation]] = []
        missing: list[ParserBlockObservation] = []
        for block, middle, observation in page_entries:
            source = _reading_order_source(block, middle)
            if source is None:
                missing.append(observation)
            else:
                ordered.append((source[0], source[1], observation))
        if not ordered:
            continue
        if missing:
            for _order, source_field, observation in ordered:
                records[observation.observation_id].append(
                    {
                        "relation_type": RelationType.READING_NEXT.value,
                        "source_location": (
                            "middle_json"
                            if source_field.startswith("middle_json.")
                            else "block"
                        ),
                        "source_field": source_field,
                        "target_ref": None,
                        "target_tokens": [],
                        "status": "incomplete_page_reading_order",
                        "missing_observation_count": len(missing),
                    }
                )
            for observation in missing:
                records[observation.observation_id].append(
                    {
                        "relation_type": RelationType.READING_NEXT.value,
                        "source_location": "block",
                        "source_field": "reading_order",
                        "target_ref": None,
                        "target_tokens": [],
                        "status": "missing_reading_order",
                    }
                )
            continue
        counts: dict[float, int] = {}
        for order, _source_field, _observation in ordered:
            counts[order] = counts.get(order, 0) + 1
        duplicates = sorted(order for order, count in counts.items() if count > 1)
        if duplicates:
            for _order, source_field, observation in ordered:
                records[observation.observation_id].append(
                    {
                        "relation_type": RelationType.READING_NEXT.value,
                        "source_location": (
                            "middle_json"
                            if source_field.startswith("middle_json.")
                            else "block"
                        ),
                        "source_field": source_field,
                        "target_ref": None,
                        "target_tokens": [],
                        "status": "ambiguous_page_reading_order",
                        "duplicate_orders": duplicates,
                    }
                )
            continue
        ordered.sort(key=lambda item: (item[0], item[2].observation_id))
        for left, right in zip(ordered, ordered[1:], strict=False):
            for source_item, target_item, relation_type in (
                (left, right, RelationType.READING_NEXT),
                (right, left, RelationType.READING_PREVIOUS),
            ):
                source_order, source_field, source_observation = source_item
                target_order, target_field, target_observation = target_item
                metadata = {
                    "assertion_origin": "mineru_explicit_reading_order",
                    "source_native_id": source_observation.provenance.parser_native_id,
                    "source_field": source_field,
                    "source_order": source_order,
                    "target_native_id": target_observation.provenance.parser_native_id,
                    "target_field": target_field,
                    "target_order": target_order,
                }
                assertions[source_observation.observation_id].append(
                    ObservationRelation(
                        relation_type=relation_type,
                        target_observation_id=target_observation.observation_id,
                        confidence=1.0,
                        metadata=metadata,
                    )
                )
                records[source_observation.observation_id].append(
                    {
                        "relation_type": relation_type.value,
                        "source_location": (
                            "middle_json"
                            if source_field.startswith("middle_json.")
                            else "block"
                        ),
                        "source_field": source_field,
                        "target_ref": {
                            "parser_native_id": target_observation.provenance.parser_native_id,
                            "reading_order": target_order,
                        },
                        "target_tokens": [],
                        "status": "resolved_adjacent_reading_order",
                        "target_observation_id": target_observation.observation_id,
                    }
                )

    enriched: list[ParserBlockObservation] = []
    for _block, _middle, observation in translated:
        unique_relations = {
            relation.relation_id: relation
            for relation in (*observation.relations, *assertions[observation.observation_id])
        }
        relations = tuple(
            sorted(
                unique_relations.values(),
                key=lambda item: (
                    item.relation_type.value,
                    item.target_observation_id,
                    item.relation_id,
                ),
            )
        )
        metadata = dict(observation.metadata)
        observation_records = records[observation.observation_id]
        if observation_records:
            metadata["mineru_structural_ref_resolution"] = sorted(
                observation_records,
                key=lambda item: (
                    str(item.get("source_location", "")),
                    str(item.get("source_field", "")),
                    str(item.get("relation_type", "")),
                    content_sha256(item.get("target_ref")),
                    str(item.get("status", "")),
                ),
            )
        enriched.append(
            replace(observation, relations=relations, metadata=metadata)
        )
    return tuple(enriched)


def translate_normalized_blocks(
    request: ParseRequest,
    blocks: Iterable[Mapping[str, object]],
    *,
    parser_version: str,
    model_version: str,
    config_sha256: str,
    raw_artifact_uri: str | None = None,
) -> tuple[tuple[ParserBlockObservation, ...], tuple[ParseIssue, ...]]:
    """Translate legacy normalized blocks without importing parser packages."""

    require_non_empty(parser_version, "parser_version")
    require_non_empty(model_version, "model_version")
    require_sha256(config_sha256, "config_sha256")
    if raw_artifact_uri is not None:
        require_non_empty(raw_artifact_uri, "raw_artifact_uri")

    observations: list[ParserBlockObservation] = []
    translated: list[
        tuple[Mapping[str, object], Mapping[str, object], ParserBlockObservation]
    ] = []
    issues: list[ParseIssue] = []
    duplicate_counts: dict[str, int] = {}
    for index, value in enumerate(blocks):
        native_id: str | None = None
        page_index: int | None = None
        try:
            block = require_mapping(value, f"blocks[{index}]")
            middle = _middle_json(block)
            base_id = _native_base(block, middle)
            duplicate_counts[base_id] = duplicate_counts.get(base_id, 0) + 1
            native_id = (
                base_id
                if duplicate_counts[base_id] == 1
                else f"{base_id}:duplicate-{duplicate_counts[base_id]}"
            )
            raw_page = block.get("page_idx")
            page_index = raw_page if isinstance(raw_page, int) and not isinstance(raw_page, bool) else None
            observation = _translate_one(
                request,
                block,
                native_id=native_id,
                parser_version=parser_version,
                model_version=model_version,
                config_sha256=config_sha256,
                raw_artifact_uri=raw_artifact_uri,
            )
            observations.append(observation)
            translated.append((block, middle, observation))
        except (ContractValidationError, TypeError, ValueError) as exc:
            issues.append(
                ParseIssue(
                    code=ParseErrorCode.INVALID_NORMALIZED_BLOCK,
                    classification=ErrorClassification.PERMANENT,
                    severity=IssueSeverity.ERROR,
                    message=f"normalized block {index} is invalid: {exc}",
                    page_index=page_index,
                    parser_native_id=native_id,
                    details={"block_index": index, "translation": TRANSLATION_NAME},
                )
            )
    observations = list(_attach_structural_relations(translated))
    observations.sort(
        key=lambda item: (
            item.page_index,
            float("inf") if item.reading_order is None else item.reading_order,
            item.observation_id,
        )
    )
    issues.sort(key=lambda item: item.issue_id)
    return tuple(observations), tuple(issues)


class MinerUNormalizedAdapter:
    """Injected bridge from the current BookRAG normalized list to the port."""

    def __init__(
        self,
        normalized_block_loader: NormalizedBlockLoader,
        *,
        parser_version: str,
        model_version: str,
        config: Mapping[str, object],
        raw_artifact_uri: str | None = None,
    ) -> None:
        if not callable(normalized_block_loader):
            raise TypeError("normalized_block_loader must be callable")
        require_non_empty(parser_version, "parser_version")
        require_non_empty(model_version, "model_version")
        if raw_artifact_uri is not None:
            require_non_empty(raw_artifact_uri, "raw_artifact_uri")
        self._loader = normalized_block_loader
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
            # What this adapter promises to emit. The port rejects anything
            # absent here, so this must track _block_type's mapping -- a type
            # translated but not declared fails the whole document at bind
            # time, which is how aside_text and doc_title first showed up.
            block_types=tuple(DECLARED_BLOCK_TYPES),
            accelerator=AcceleratorKind.ANY,
            supports_page_selection=True,
            supports_region_selection=False,
            supports_batch=False,
            deterministic=False,
            metadata={
                "adapter_contract_version": ADAPTER_VERSION,
                "input_shape": "bookrag_mineru_normalized_blocks.v1",
                "translation": TRANSLATION_NAME,
                "vendor_imports": False,
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
        input_block_count = 0
        try:
            loaded = self._loader(request)
            raw_blocks = getattr(loaded, "blocks", loaded)
            raw_artifacts = getattr(loaded, "artifact_receipts", ())
            blocks = tuple(raw_blocks)
            candidate_artifacts = tuple(raw_artifacts)
            if any(
                not isinstance(item, ParserArtifactReceipt)
                for item in candidate_artifacts
            ):
                raise ContractValidationError(
                    "normalized loader artifact_receipts must contain "
                    "ParserArtifactReceipt DTOs"
                )
            if any(
                item.document_version_id != request.document.document_version_id
                or item.source_sha256 != request.document.content_sha256
                for item in candidate_artifacts
            ):
                raise ContractValidationError(
                    "normalized loader artifact_receipts belong to another source"
                )
            artifacts = candidate_artifacts
            input_block_count = len(blocks)
            observations, issues = translate_normalized_blocks(
                request,
                blocks,
                parser_version=self._parser_version,
                model_version=self._model_version,
                config_sha256=self._config_sha256,
                raw_artifact_uri=self._raw_artifact_uri,
            )
        except ParserExecutionError as exc:
            return ParseResult(
                request_id=request.request_id,
                adapter_name=ADAPTER_NAME,
                adapter_version=ADAPTER_VERSION,
                status=ParseStatus.FAILED,
                issues=(exc.issue,),
                artifacts=exc.artifacts,
                duration_ms=(perf_counter() - started) * 1000,
                metrics={"input_blocks": 0, "valid_observations": 0, "invalid_blocks": 0},
            )
        except Exception as exc:
            # This is the adapter boundary: vendor/runtime exceptions become a
            # retry-classified result, never an empty string or fake block.
            issue = ParseIssue(
                code=ParseErrorCode.PARSER_FAILURE,
                classification=ErrorClassification.RETRYABLE,
                severity=IssueSeverity.ERROR,
                message="normalized block loader failed",
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
                metrics={"input_blocks": 0, "valid_observations": 0, "invalid_blocks": 0},
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
                "input_blocks": input_block_count,
                "valid_observations": len(observations),
                "invalid_blocks": len(issues),
            },
            artifacts=artifacts,
        )
