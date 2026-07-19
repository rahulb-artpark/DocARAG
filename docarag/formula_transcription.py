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
    "VLMFormulaTranscriber",
    "clean_latex",
    "transcribe_equation_blocks",
]
