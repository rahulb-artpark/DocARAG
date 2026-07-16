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
_HAS_ASSET_TYPES = {"image", "table", "equation", "interline_equation"}
_MIN_RECOVERED_ALNUM = 3


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


def _is_empty_text_block(block: dict) -> bool:
    if str(block.get("type", "")).strip().lower() in _HAS_ASSET_TYPES:
        return False
    if block.get("img_path"):
        return False
    if str(block.get("type", "")).strip().lower() not in _TEXT_LIKE:
        return False
    text = block.get("text")
    return not (isinstance(text, str) and text.strip())


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def recover_and_prune_empty_text_blocks(
    pdf_path: str | Path,
    pdf_list: list[dict],
    *,
    coordinate_scale: float = 1000.0,
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
    for block in pdf_list:
        if not isinstance(block, dict) or _is_empty_text_block(block):
            continue
        page = block.get("page_idx")
        text = block.get("text")
        if isinstance(page, int) and isinstance(text, str) and text.strip():
            kept_by_page[page] = kept_by_page.get(page, "") + _normalize(text)

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
        width, height = sizes[page]
        x0, y0, x1, y1 = bbox
        left = x0 / coordinate_scale * width
        right = x1 / coordinate_scale * width
        top = height - (y0 / coordinate_scale * height)
        bottom = height - (y1 / coordinate_scale * height)
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
            text = native_text(block)
            normalized = _normalize(text)
            page = block.get("page_idx")
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
