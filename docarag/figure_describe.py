"""Describe a figure with a vision model grounded on recovered OCR text.

A vision model asked to read a rendered table will silently reword its labels
and round its numbers.  The OCR pass already recovered those strings exactly,
so they are supplied as authoritative context and the model is left to do what
OCR cannot: say what the figure is and what it shows.
"""

from __future__ import annotations

import base64
import logging
from typing import Protocol

log = logging.getLogger(__name__)

MAX_GROUNDING_CHARS = 6_000

PROMPT_HEADER = """You are describing one figure extracted from a document page.

Describe the figure in 2 to 5 sentences:
- Say what kind of visual it is: table, bar/line/pie chart, flow diagram, \
screenshot, photo, logo, or icon.
- Say what it depicts and what its main point is.
- For a chart, name the chart type, the axes or categories, the series, and \
the notable values.
- For a table, say what it tabulates and name its columns.
- For a diagram or flowchart, name the nodes and say how they connect, in order.

Describe only what is visible. Do not speculate about intent or context that \
is not shown."""

GROUNDED_SUFFIX = """

Text already recovered from this figure by OCR is authoritative. Reuse these \
exact labels and numbers; do not invent or round values:

{ocr_text}"""

UNGROUNDED_SUFFIX = """

OCR recovered no text from this figure, so describe it visually."""


class VisionModel(Protocol):
    def generate(
        self, prompt_or_memory: str, images: list[str] | None = None
    ) -> str: ...


class GroundedFigureDescriber:
    """Turn a figure crop plus its OCR text into a factual description."""

    def __init__(self, vlm: VisionModel) -> None:
        self._vlm = vlm

    def __call__(self, data: bytes, ocr_text: str) -> str | None:
        if ocr_text.strip():
            prompt = PROMPT_HEADER + GROUNDED_SUFFIX.format(
                ocr_text=ocr_text[:MAX_GROUNDING_CHARS]
            )
        else:
            prompt = PROMPT_HEADER + UNGROUNDED_SUFFIX
        image = base64.b64encode(data).decode("ascii")
        description = self._vlm.generate(prompt, images=[image])
        if not isinstance(description, str) or not description.strip():
            return None
        return description.strip()


__all__ = ["GroundedFigureDescriber", "VisionModel"]
