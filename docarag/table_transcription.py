"""Re-read table crops with a vision model.

The table recognizer fails the same way the formula recognizer does, and for
the same reason: notation it cannot segment.  On a trigonometric ratios table
it returned ``1|2`` for a half, ``9|~ -9`` for root three over two, ``V2`` for
root two, and empty cells where one over root two belonged -- a table whose
every interesting value is a fraction or a radical came back with none of them
readable.  ``transcribe_formulas`` exists because that recognizer "drops lines
it cannot segment"; tables had no equivalent second chance.

Every table region already carries a verified crop, so a vision model reads the
same pixels independently.  Unlike the formula pass, which only ever adds a
channel, a table's transcription has to claim the ``html`` primary slot to
reach ``_table_body`` -- so it is admitted only when its shape matches what the
recognizer found.  A model that returns a differently shaped table has read a
different table, or invented one, and a plausible invention replacing a merely
damaged reading is worse than the damage.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, replace
from html.parser import HTMLParser
import logging
import re
from typing import Protocol

from docarag.asset_material import AssetMaterialResolver, AssetMaterializationError
from docarag.domain.evidence import (
    AssetKind,
    BlockType,
    CanonicalBlock,
    Representation,
    RepresentationKind,
)

log = logging.getLogger(__name__)

MAX_TABLE_CHARS = 20_000

PROMPT = """Transcribe this table image to HTML.

Reproduce every row and every cell, including the header row, in reading
order. Preserve fractions and mathematical notation as LaTeX inside the cell,
for example <td>$\\frac{1}{2}$</td> and <td>$\\sqrt{3}/2$</td>. Leave a cell
empty only when it is empty in the image.

Output only the HTML table, starting with <table> and ending with </table>,
with no commentary and no code fences."""


class VisionModel(Protocol):
    def generate(
        self, prompt_or_memory: str, images: list[str] | None = None
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TableTranscriptionStats:
    table_blocks: int = 0
    transcribed_blocks: int = 0
    skipped_blocks: int = 0
    failed_blocks: int = 0
    shape_mismatch_blocks: int = 0


class VLMTableTranscriber:
    """Turn a table crop into HTML, or nothing when the output is unusable."""

    def __init__(self, vlm: VisionModel) -> None:
        self._vlm = vlm

    def __call__(self, data: bytes) -> str | None:
        image = base64.b64encode(data).decode("ascii")
        raw = self._vlm.generate(PROMPT, images=[image])
        if not isinstance(raw, str):
            return None
        return clean_table_html(raw)


_FENCE = re.compile(r"^```[a-zA-Z]*\n?|```$")
_TABLE_SPAN = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.S)


def clean_table_html(raw: str) -> str | None:
    """Reduce a model reply to one bare HTML table, or None."""

    text = _FENCE.sub("", (raw or "").strip()).strip()
    match = _TABLE_SPAN.search(text)
    if match is None:
        return None
    table = match.group(0).strip()
    if len(table) > MAX_TABLE_CHARS:
        return None
    return table if table_shape(table) else None


class _ShapeReader(HTMLParser):
    """Row and column counts, which is all the shape gate needs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[int] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.rows.append(0)
        elif tag in ("td", "th") and self.rows:
            self.rows[-1] += 1


def table_shape(html: str) -> tuple[int, ...] | None:
    """Cells per row, or None when this is not a usable table."""

    if not isinstance(html, str) or "<table" not in html.lower():
        return None
    reader = _ShapeReader()
    try:
        reader.feed(html)
        reader.close()
    except (AssertionError, ValueError):
        return None
    rows = tuple(reader.rows)
    if not rows or not any(rows):
        return None
    return rows


def transcribe_table_blocks(
    blocks: Sequence[CanonicalBlock],
    *,
    resolver: AssetMaterialResolver,
    transcriber: VLMTableTranscriber,
) -> tuple[tuple[CanonicalBlock, ...], TableTranscriptionStats]:
    """Return blocks whose tables were re-read, where the reading agrees."""

    tables = transcribed = skipped = failed = mismatched = 0
    out: list[CanonicalBlock] = []
    for block in blocks:
        if block.block_type is not BlockType.TABLE:
            out.append(block)
            continue
        tables += 1
        asset = _verified_crop(block)
        original = _primary_html(block)
        if asset is None or _already_transcribed(block):
            skipped += 1
            out.append(block)
            continue
        try:
            material = resolver.resolve(asset, include_bytes=True)
            if not material.data:
                raise AssetMaterializationError(
                    f"verified asset {asset.asset_id} yielded no bytes"
                )
            html = transcriber(material.data)
        except (AssetMaterializationError, OSError, ValueError, RuntimeError):
            log.warning(
                "table transcription failed for block %s", block.block_id, exc_info=True
            )
            failed += 1
            out.append(block)
            continue
        if not html:
            skipped += 1
            out.append(block)
            continue
        if original is not None:
            before = table_shape(str(original.content))
            after = table_shape(html)
            if before is not None and before != after:
                # A different shape is a different table, not a better reading
                # of this one. Cell contents may be wrong; the grid the
                # recognizer found is the part it gets right.
                log.info(
                    "table transcription for block %s rejected: shape %s != %s",
                    block.block_id,
                    after,
                    before,
                )
                mismatched += 1
                out.append(block)
                continue
            if str(original.content).strip() == html.strip():
                skipped += 1
                out.append(block)
                continue
        corrected = Representation(
            kind=RepresentationKind.HTML,
            content=html,
            media_type="text/html",
            confidence=min(1.0, (getattr(original, "confidence", None) or 0.0) + 0.01),
            metadata={
                "producer": "vlm.table_transcription",
                "asset_sha256": asset.sha256,
                **(
                    {"source_representation_id": original.representation_id}
                    if original is not None
                    else {}
                ),
            },
        )
        # _table_body reads the html representation designated primary, so a
        # correction that is merely higher-confidence would never be seen. The
        # recognizer's own reading stays in place beside it.
        primary = dict(block.metadata.get("primary_representations") or {})
        primary[RepresentationKind.HTML.value] = corrected.representation_id
        out.append(
            replace(
                block,
                representations=(*block.representations, corrected),
                metadata={**block.metadata, "primary_representations": primary},
            )
        )
        transcribed += 1
    return tuple(out), TableTranscriptionStats(
        table_blocks=tables,
        transcribed_blocks=transcribed,
        skipped_blocks=skipped,
        failed_blocks=failed,
        shape_mismatch_blocks=mismatched,
    )


def _verified_crop(block: CanonicalBlock):
    for asset in block.assets:
        if (
            asset.kind is AssetKind.REGION_CROP
            and isinstance(asset.uri, str)
            and asset.uri.startswith("cas://")
        ):
            return asset
    return None


def _primary_html(block: CanonicalBlock) -> Representation | None:
    values = block.metadata.get("primary_representations")
    primary_id = None
    if isinstance(values, dict):
        candidate = values.get(RepresentationKind.HTML.value)
        primary_id = candidate if isinstance(candidate, str) and candidate else None
    html = [
        item for item in block.representations if item.kind is RepresentationKind.HTML
    ]
    if not html:
        return None
    for item in html:
        if item.representation_id == primary_id:
            return item
    return html[0]


def _already_transcribed(block: CanonicalBlock) -> bool:
    return any(
        (item.metadata or {}).get("producer") == "vlm.table_transcription"
        for item in block.representations
    )


__all__ = [
    "TableTranscriptionStats",
    "VLMTableTranscriber",
    "clean_table_html",
    "table_shape",
    "transcribe_table_blocks",
]
