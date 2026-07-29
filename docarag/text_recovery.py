"""Repair empty parser blocks with the PDF native text layer (multi-tool).

A single parser (MinerU vlm or pipeline) can detect a region but return empty
text: sometimes real text it missed, sometimes a redundant/false-positive
region. Rather than fail-close the whole document, this cross-checks the PDF's
native text layer by bounding box, recovers text that is genuinely missing, and
drops only blocks that are empty or already covered by another block - so no
unique content is lost and the document can still be indexed.
"""

from __future__ import annotations

import re
from pathlib import Path

_TEXT_LIKE = {"text", "title", "list", "list_item", "header", "footer", "footnote", ""}
_ASSET_TYPES = {"image", "table", "equation", "interline_equation"}
_CONTENT_KEYS = (
    "text",
    # A MinerU 3.x list block puts its items in list_items and carries no
    # "text" at all. Judged by "text" alone every list looks empty, so
    # recovery rewrote it from the page's native layer and it reached the tree
    # as a paragraph -- the list type survived parsing, reconciliation and
    # projection, and was lost here.
    "list_items",
    "code_body",
    "img_path",
    "table_body",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_footnote",
)
_MIN_RECOVERED_ALNUM = 3
_REDUNDANT_OVERLAP = 0.5


def _block_bbox(block: dict) -> tuple[float, float, float, float] | None:
    middle = block.get("middle_json")
    bbox = middle.get("bbox") if isinstance(middle, dict) else None
    if bbox is None:
        bbox = block.get("bbox")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
    ):
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return None


def _has_content(block: dict) -> bool:
    for key in _CONTENT_KEYS:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and any(
            isinstance(item, str) and item.strip() for item in value
        ):
            return True
    return False


def _is_empty_text_block(block: dict) -> bool:
    """True when a block carries no evidence in any channel its type can use.

    A layout model can label a text region an equation and then recognize no
    formula in it, leaving a block with neither text nor crop.  Such a block is
    as empty as an empty paragraph and its region is just as recoverable, so
    emptiness is decided by what the block actually holds rather than by type.
    """

    kind = str(block.get("type", "")).strip().lower()
    if kind not in _TEXT_LIKE and kind not in _ASSET_TYPES:
        return False
    return not _has_content(block)


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _covered_fraction(
    region: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
) -> float:
    """Return how much of ``region`` already lies under content-bearing blocks."""

    x0, y0, x1, y1 = region
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0
    covered = 0.0
    for ox0, oy0, ox1, oy1 in others:
        width = min(x1, ox1) - max(x0, ox0)
        height = min(y1, oy1) - max(y0, oy0)
        if width > 0 and height > 0:
            covered += width * height
    return min(covered / area, 1.0)


def recover_and_prune_empty_text_blocks(
    pdf_path: str | Path,
    pdf_list: list[dict],
) -> tuple[list[dict], int, int]:
    """Return (blocks, recovered_count, dropped_count) with empties repaired or pruned."""

    empties = [
        index
        for index, block in enumerate(pdf_list)
        if isinstance(block, dict) and _is_empty_text_block(block)
    ]
    if not empties:
        return pdf_list, 0, 0

    kept_by_page: dict[int, str] = {}
    kept_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for block in pdf_list:
        if not isinstance(block, dict) or _is_empty_text_block(block):
            continue
        page = block.get("page_idx")
        if not isinstance(page, int) or isinstance(page, bool):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            kept_by_page[page] = kept_by_page.get(page, "") + _normalize(text)
        box = _block_bbox(block)
        if box is not None and _has_content(block):
            kept_boxes.setdefault(page, []).append(box)

    try:
        import pypdfium2 as pdfium
    except ImportError:
        pdfium = None

    document = None
    text_pages: dict[int, object] = {}
    sizes: dict[int, tuple[float, float]] = {}
    if pdfium is not None:
        try:
            document = pdfium.PdfDocument(str(Path(pdf_path).expanduser()))
        except Exception:  # noqa: BLE001 - unreadable PDF means no native recovery
            document = None

    def native_text(block: dict) -> str:
        if document is None:
            return ""
        page = block.get("page_idx")
        bbox = _block_bbox(block)
        if not isinstance(page, int) or isinstance(page, bool) or bbox is None:
            return ""
        if page < 0 or page >= len(document):
            return ""
        if page not in text_pages:
            handle = document[page]
            sizes[page] = handle.get_size()
            text_pages[page] = handle.get_textpage()
        _width, height = sizes[page]
        # MinerU boxes its blocks in page points against the same page size
        # this document reports, with a top-left origin. pypdfium2 ranges are
        # bottom-left, so only the vertical axis is converted.
        left, y0, right, y1 = bbox
        top = height - y0
        bottom = height - y1
        try:
            return (
                text_pages[page]
                .get_text_bounded(
                    left=min(left, right),
                    bottom=min(bottom, top),
                    right=max(left, right),
                    top=max(bottom, top),
                )
                or ""
            ).strip()
        except Exception:  # noqa: BLE001 - extraction failures mean "no native text"
            return ""

    result: list[dict] = []
    recovered = 0
    dropped = 0
    try:
        for index, block in enumerate(pdf_list):
            if index not in empties:
                result.append(block)
                continue
            page = block.get("page_idx")
            box = _block_bbox(block)
            # A layout model can drop a spurious region on top of blocks that
            # already carry the text. Reading that region back yields the same
            # words clipped mid-token, so the region is redundant, not lost.
            if (
                box is not None
                and isinstance(page, int)
                and _covered_fraction(box, kept_boxes.get(page, []))
                >= _REDUNDANT_OVERLAP
            ):
                dropped += 1
                continue
            text = native_text(block)
            normalized = _normalize(text)
            covered = (
                isinstance(page, int) and normalized and normalized in kept_by_page.get(page, "")
            )
            if len(normalized) >= _MIN_RECOVERED_ALNUM and not covered:
                repaired = dict(block)
                repaired["text"] = text
                repaired["_text_recovered_from"] = "pdf_native_text_layer"
                result.append(repaired)
                if isinstance(page, int):
                    kept_by_page[page] = kept_by_page.get(page, "") + normalized
                recovered += 1
            else:
                dropped += 1
    finally:
        if document is not None:
            document.close()
    return result, recovered, dropped


__all__ = ["recover_and_prune_empty_text_blocks"]
