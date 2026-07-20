"""Re-read equation crops with a vision model.

The dedicated formula recognizer drops rows and misreads symbols it cannot
segment -- a two-line derivation can come back with an empty middle row.  Every
equation region already has a verified crop, so a vision model reads the same
pixels independently and its transcription is attached as a second channel;
the recognizer's output is kept untouched beside it.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, replace
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

# Block types where an embedded inline formula misread by MinerU's own
# recognizer is plausible and splicing corrected text back in makes sense.
# Excludes EQUATION (transcribe_equation_blocks already owns those blocks
# whole) and shapes where "$...$" inside the content isn't LaTeX at all
# (a table cell, a figure/chart/diagram region, a code block that may use
# "$" for a shell prompt).
_INLINE_ELIGIBLE_BLOCK_TYPES = frozenset(
    {
        BlockType.TITLE,
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.UNKNOWN,
    }
)

_INLINE_LATEX_SPAN = re.compile(r"\$([^$]+)\$")
# A relational/arithmetic operator is what distinguishes an actual equation
# ("p = mv") from other "$...$"-wrapped notation MinerU also emits inline,
# like unit expressions ("$\mathrm{kg}\mathrm{m}\mathrm{s}{-}^{1}$") that
# have no operator and are not what a single equation crop corresponds to.
_EQUATION_OPERATOR = re.compile(r"=|\\approx|\\leq|\\geq|\\neq|\\propto")


@dataclass(frozen=True, slots=True)
class InlineFormulaCorrectionStats:
    candidate_blocks: int = 0
    corrected_blocks: int = 0
    skipped_blocks: int = 0
    failed_blocks: int = 0

log = logging.getLogger(__name__)

MAX_LATEX_CHARS = 4_000

PROMPT = """Transcribe this equation image to LaTeX.

Write every line of the equation, top to bottom. Output only the LaTeX,
with no commentary, no code fences and no $$ delimiters."""


class VisionModel(Protocol):
    def generate(
        self, prompt_or_memory: str, images: list[str] | None = None
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class FormulaTranscriptionStats:
    equation_blocks: int = 0
    transcribed_blocks: int = 0
    skipped_blocks: int = 0
    failed_blocks: int = 0


class VLMFormulaTranscriber:
    """Turn an equation crop into LaTeX, or nothing when the output is unusable."""

    def __init__(self, vlm: VisionModel) -> None:
        self._vlm = vlm

    def __call__(self, data: bytes) -> str | None:
        image = base64.b64encode(data).decode("ascii")
        raw = self._vlm.generate(PROMPT, images=[image])
        if not isinstance(raw, str):
            return None
        return clean_latex(raw)


_FENCE = re.compile(r"^```[a-zA-Z]*\n?|```$")
_PROSE = re.compile(r"\b(the|this|equation|image|shows|represents)\b", re.IGNORECASE)


def clean_latex(raw: str) -> str | None:
    """Reduce a model reply to bare LaTeX, or None when it is not a formula."""

    text = _FENCE.sub("", raw.strip()).strip()
    text = text.removeprefix("$$").removesuffix("$$").strip()
    if not text or len(text) > MAX_LATEX_CHARS:
        return None
    # A transcription contains notation; a refusal or an explanation contains
    # sentences.  Prose words with no math markup mean the model did not
    # transcribe.
    if _PROSE.search(text) and "\\" not in text and "=" not in text:
        return None
    return text


def transcribe_equation_blocks(
    blocks: Sequence[CanonicalBlock],
    *,
    resolver: AssetMaterialResolver,
    transcriber: VLMFormulaTranscriber,
) -> tuple[tuple[CanonicalBlock, ...], FormulaTranscriptionStats]:
    """Return blocks whose equations carry a vision transcription channel."""

    equations = transcribed = skipped = failed = 0
    out: list[CanonicalBlock] = []
    for block in blocks:
        if block.block_type is not BlockType.EQUATION:
            out.append(block)
            continue
        equations += 1
        asset = _verified_crop(block)
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
            latex = transcriber(material.data)
        except (AssetMaterializationError, OSError, ValueError, RuntimeError):
            log.warning(
                "formula transcription failed for block %s",
                block.block_id,
                exc_info=True,
            )
            failed += 1
            out.append(block)
            continue
        if not latex:
            skipped += 1
            out.append(block)
            continue
        out.append(
            replace(
                block,
                representations=(
                    *block.representations,
                    Representation(
                        kind=RepresentationKind.VLM_LATEX,
                        content=f"$$\n{latex}\n$$",
                        media_type="application/x-latex",
                        metadata={
                            "producer": "vlm.formula_transcription",
                            "asset_sha256": asset.sha256,
                        },
                    ),
                ),
            )
        )
        transcribed += 1
    return tuple(out), FormulaTranscriptionStats(
        equation_blocks=equations,
        transcribed_blocks=transcribed,
        skipped_blocks=skipped,
        failed_blocks=failed,
    )


def _find_unambiguous_equation_span(text: str) -> re.Match | None:
    """Find the one "$...$" span in text that a single equation crop could
    unambiguously correspond to.

    A block can carry more than one inline "$...$" span (an equation next to
    an unrelated unit expression, in the reported case) while only ever
    having one region crop -- there is no per-span geometry left once text
    has been merged into one string, so which span the crop belongs to is
    genuinely ambiguous unless exactly one candidate looks like an equation
    (contains a relational/arithmetic operator). Returns None rather than
    guessing when that isn't true: zero matches, or more than one operator-
    bearing span, both mean this function cannot tell which span is which.
    """

    matches = list(_INLINE_LATEX_SPAN.finditer(text))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    operator_matches = [m for m in matches if _EQUATION_OPERATOR.search(m.group(1))]
    if len(operator_matches) == 1:
        return operator_matches[0]
    return None


def has_inline_formula_candidates(blocks: Sequence[CanonicalBlock]) -> bool:
    """Cheap pre-check for whether paying for a VLM instance is worth it.

    A substring check, not the full span/operator logic transcribe_inline_
    equations itself applies -- this only needs to rule out the common case
    (no "$" anywhere in an eligible block) without doing real work twice.
    """

    for block in blocks:
        if block.block_type not in _INLINE_ELIGIBLE_BLOCK_TYPES:
            continue
        for rep in block.representations:
            if isinstance(rep.content, str) and "$" in rep.content:
                return True
    return False


def transcribe_inline_equations(
    blocks: Sequence[CanonicalBlock],
    *,
    resolver: AssetMaterialResolver,
    transcriber: VLMFormulaTranscriber,
) -> tuple[tuple[CanonicalBlock, ...], InlineFormulaCorrectionStats]:
    """Re-verify an equation MinerU embedded inline in a non-equation block's
    own text, rather than as its own EQUATION-type block -- confirmed live
    (312_LG_E_L3.pdf): MinerU crops individual equation spans regardless of
    the parent block's own type, so the crop this needs already exists even
    though transcribe_equation_blocks never looks at this block (its
    block_type isn't EQUATION).

    Unlike transcribe_equation_blocks, which is free to prefer a whole new
    representation over the original for a block that *is* the equation,
    here the formula is only part of a larger paragraph -- so a corrected
    reading is spliced into a full copy of the block's own text at the
    exact span the original occupied, added as a new, higher-confidence
    representation of the *same* kind next to the original (never
    overwritten in place), so the existing representation-selection
    mechanism in bookrag_projection.py picks it up automatically, for
    whichever representation kind this block actually uses for display.
    """

    candidates = corrected = skipped = failed = 0
    out: list[CanonicalBlock] = []
    for block in blocks:
        if block.block_type not in _INLINE_ELIGIBLE_BLOCK_TYPES:
            out.append(block)
            continue
        asset = _verified_crop(block)
        if asset is None:
            out.append(block)
            continue

        # Every representation is a candidate source text independently --
        # a block could carry more than one channel (OCR text alongside a
        # markdown rendering, say), and each needs its own span search since
        # they are not guaranteed to contain the same substring verbatim.
        # Stats are tallied once per block, not once per representation.
        has_candidate = block_failed = block_corrected = False
        for source in block.representations:
            if not isinstance(source.content, str):
                continue
            span = _find_unambiguous_equation_span(source.content)
            if span is None:
                continue
            has_candidate = True
            try:
                material = resolver.resolve(asset, include_bytes=True)
                if not material.data:
                    raise AssetMaterializationError(
                        f"verified asset {asset.asset_id} yielded no bytes"
                    )
                latex = transcriber(material.data)
            except (AssetMaterializationError, OSError, ValueError, RuntimeError):
                log.warning(
                    "inline formula transcription failed for block %s",
                    block.block_id,
                    exc_info=True,
                )
                block_failed = True
                continue
            original_span_text = span.group(1).strip()
            if not latex or latex.strip() == original_span_text:
                continue
            corrected_content = (
                source.content[: span.start()]
                + f"${latex}$"
                + source.content[span.end():]
            )
            block = replace(
                block,
                representations=(
                    *block.representations,
                    Representation(
                        kind=source.kind,
                        content=corrected_content,
                        media_type=source.media_type,
                        language=source.language,
                        # Strictly above the original so the existing
                        # confidence-ordered selection in
                        # bookrag_projection.py prefers this corrected
                        # version without needing to know this correction
                        # mechanism exists at all.
                        confidence=min(1.0, (source.confidence or 0.0) + 0.01),
                        metadata={
                            "producer": "vlm.inline_formula_correction",
                            "asset_sha256": asset.sha256,
                            "source_representation_id": source.representation_id,
                            "original_span": span.group(0),
                            "corrected_span": f"${latex}$",
                        },
                    ),
                ),
            )
            block_corrected = True
        if has_candidate:
            candidates += 1
            if block_corrected:
                corrected += 1
            elif block_failed:
                failed += 1
            else:
                skipped += 1
        out.append(block)
    return tuple(out), InlineFormulaCorrectionStats(
        candidate_blocks=candidates,
        corrected_blocks=corrected,
        skipped_blocks=skipped,
        failed_blocks=failed,
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


def _already_transcribed(block: CanonicalBlock) -> bool:
    return any(
        item.kind is RepresentationKind.VLM_LATEX for item in block.representations
    )


__all__ = [
    "FormulaTranscriptionStats",
    "InlineFormulaCorrectionStats",
    "VLMFormulaTranscriber",
    "clean_latex",
    "has_inline_formula_candidates",
    "transcribe_equation_blocks",
    "transcribe_inline_equations",
]
