"""Parse a PDF as bounded page shards instead of loading it whole.

``Core.provider.extract_pdf_info.parse_doc`` accepts a page range and, when
given one, only renders, OCRs and analyzes those pages -- confirmed live
against the installed MinerU library, not assumed (see the design study this
implements). Peak memory becomes a function of shard size, not document size.

Three things a naive per-shard loop gets wrong, each verified against the
actual refiner code rather than assumed, and each handled here:

1. Each shard is parsed as its own small PDF, so MinerU numbers its pages
   from 0. ``page_idx`` in every shard's output is offset by that shard's
   starting page before use, or every shard after the first collides with
   page 0 (confirmed live: requesting pages 2-3 still returns page_idx
   [0, 1]).

2. ``Core.utils.utils.enumerate_pdf_list`` always starts its ``pdf_id``
   counter at 1. Called per shard in isolation, shard 2's ids would
   collide with shard 1's -- and pdf_id is what the outline extractor's
   parent/child stack and canonical block binding key off of.
   ``pdf_info_refiner`` now accepts ``start_pdf_id`` so the counter
   continues across shards instead of resetting.

3. ``pdf_info_refiner`` merges paragraphs and tables that continue across
   a page break. A shard boundary is a page break too. The merge search
   itself is bounded, not global -- ``search_continuation_candidates``
   stops after ``max_page_gap`` (2) pages forward, and
   ``search_previous_table`` stops after 1 page backward -- so carrying
   the last ``CARRY_PAGES`` (2) pages of each shard into the next shard's
   refiner pass is provably sufficient, not a guess: nothing the merger
   would ever have looked at falls outside that window.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import logging
from typing import Any

log = logging.getLogger(__name__)

ParseFn = Callable[..., tuple[dict, list]]
RefineFn = Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]

# >= the largest page-distance either merger's search can reach (2 forward,
# 1 backward -- see module docstring). A larger value only ever carries a
# few more small heading/paragraph items forward; it can never lose data.
CARRY_PAGES = 2


def pdf_page_count(pdf_path: str) -> int:
    """Return a PDF's page count without rendering or OCRing any page.

    Opens the file's structure only (the same pypdfium2 library MinerU already
    depends on) -- this is what the shard loop needs to plan ranges before any
    expensive work starts.
    """

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_path)
    try:
        return len(document)
    finally:
        document.close()


def shard_ranges(page_count: int, shard_size: int) -> Iterator[tuple[int, int]]:
    """Yield (start_page_id, end_page_id), both inclusive, 0-indexed."""

    if page_count < 1:
        raise ValueError("page_count must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    for start in range(0, page_count, shard_size):
        yield start, min(start + shard_size, page_count) - 1


def offset_page_indices(
    middle_json: dict[str, Any], content_list: list[dict[str, Any]], offset: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a shard's output with page_idx shifted from shard-relative to
    document-relative. Every other field (bbox, text, spans) is a within-page
    or content value and needs no adjustment."""

    if offset < 0:
        raise ValueError("offset must not be negative")
    if offset == 0:
        return middle_json, content_list

    shifted_pages = []
    for page in middle_json.get("pdf_info", []):
        page = dict(page)
        if isinstance(page.get("page_idx"), int):
            page["page_idx"] += offset
        shifted_pages.append(page)
    shifted_middle = {**middle_json, "pdf_info": shifted_pages}

    shifted_content = []
    for item in content_list:
        item = dict(item)
        if isinstance(item.get("page_idx"), int):
            item["page_idx"] += offset
        shifted_content.append(item)

    return shifted_middle, shifted_content


def _split_carry(
    items: list[dict[str, Any]], carry_pages: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (safe to emit now, held for the next shard's merge pass).

    Held-back items are every item on one of the last ``carry_pages`` pages
    seen in this batch -- a page count, not an item count, because that is
    what the merge search windows are actually bounded by.
    """

    page_indices = [
        item["page_idx"] for item in items if isinstance(item.get("page_idx"), int)
    ]
    if not page_indices:
        return items, []
    boundary = max(page_indices) - carry_pages + 1
    emit = [item for item in items if item.get("page_idx", boundary) < boundary]
    carry = [item for item in items if item.get("page_idx", boundary) >= boundary]
    return emit, carry


@dataclass(frozen=True, slots=True)
class ParsedShard:
    """One shard's contribution, ready to bind and persist independently."""

    content_list: list[dict[str, Any]]
    shard_index: int
    start_page_id: int
    end_page_id: int
    next_pdf_id: int


def parse_sharded(
    pdf_path: str,
    output_dir: str,
    *,
    page_count: int,
    shard_size: int,
    parse_fn: ParseFn,
    refine_fn: RefineFn | None = None,
    carry_pages: int = CARRY_PAGES,
    **parse_kwargs: Any,
) -> Iterator[ParsedShard]:
    """Yield one document's content as page shards, each document-relative,
    each independently mergeable, and each safe to persist and discard
    before the next shard is parsed.

    ``refine_fn(items, start_pdf_id)`` is any refiner that needs local
    continuity across a page break (``pdf_info_refiner``, in production
    use). It runs on ``carry + this shard's content``; everything on the
    last ``carry_pages`` pages of the result is held back into the next
    shard's refiner pass rather than emitted now, so a paragraph or table
    genuinely split across the shard boundary still merges correctly --
    the carried set is a few pages of items, not the document.
    """

    if refine_fn is None:
        refine_fn = lambda items, start_pdf_id: items  # noqa: E731

    carry: list[dict[str, Any]] = []
    next_pdf_id = 1
    ranges = list(shard_ranges(page_count, shard_size))
    for shard_index, (start_page_id, end_page_id) in enumerate(ranges):
        middle_json, content_list = parse_fn(
            pdf_path,
            output_dir,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **parse_kwargs,
        )
        _, content_list = offset_page_indices(middle_json, content_list, start_page_id)
        del middle_json  # not needed past offsetting; do not hold it across shards

        combined = carry + content_list
        is_last_shard = shard_index == len(ranges) - 1
        refined = refine_fn(combined, next_pdf_id)

        assigned_ids = [
            item["pdf_id"] for item in refined if isinstance(item.get("pdf_id"), int)
        ]
        next_pdf_id = (max(assigned_ids) + 1) if assigned_ids else next_pdf_id

        if is_last_shard:
            emit, carry = refined, []
        else:
            emit, carry = _split_carry(refined, carry_pages)

        log.info(
            "shard %d/%d (pages %d-%d): %d items ready, %d carried forward",
            shard_index + 1, len(ranges), start_page_id, end_page_id,
            len(emit), len(carry),
        )
        yield ParsedShard(
            content_list=emit,
            shard_index=shard_index,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            next_pdf_id=next_pdf_id,
        )


__all__ = [
    "CARRY_PAGES",
    "ParsedShard",
    "offset_page_indices",
    "parse_sharded",
    "pdf_page_count",
    "shard_ranges",
]
