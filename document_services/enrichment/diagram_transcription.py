"""Read a diagram's node/edge topology out of its raster with a vision model.

OCR recovers the labels inside a diagram but not the arrows between them, and a
prose description flattens the topology into a sentence.  The model is asked
for the graph itself under a fixed schema, grounded on the recovered labels so
it names the nodes the diagram actually shows.
"""

from __future__ import annotations

import base64
import logging
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

from document_services.domain.evidence import SemanticStructureKind
from document_services.enrichment.diagram_structure import (
    MAX_DIAGRAM_EDGES,
    MAX_DIAGRAM_NODES,
    MAX_LABEL_CHARS,
    DiagramEdge,
    DiagramGraph,
    DiagramNode,
)

log = logging.getLogger(__name__)

MAX_GROUNDING_CHARS = 6_000

PROMPT = """Return this figure as a graph.

kind: "flowchart" if boxes are joined by arrows in an order, "diagram" if \
labelled parts are joined by lines, "none" otherwise.
nodes: one per labelled box. key is a short id like "n1". label is its text, \
copied. role is its shape: start, end, process, decision, input, output, data. \
inside is the key of the bigger box this box is drawn inside, if any.
edges: one per arrow, source key to target key. label is the arrow's text. \
branch is "true" or "false" for a decision's two paths.

Report only what is drawn."""

GROUNDED_SUFFIX = """

Use these exact labels, read from the figure by OCR:

{ocr_text}"""


class _Node(BaseModel):
    key: str = Field(max_length=MAX_LABEL_CHARS)
    label: str = Field(max_length=MAX_LABEL_CHARS)
    role: Optional[str] = Field(default=None, max_length=MAX_LABEL_CHARS)
    inside: Optional[str] = Field(default=None, max_length=MAX_LABEL_CHARS)


class _Edge(BaseModel):
    source: str = Field(max_length=MAX_LABEL_CHARS)
    target: str = Field(max_length=MAX_LABEL_CHARS)
    label: Optional[str] = Field(default=None, max_length=MAX_LABEL_CHARS)
    branch: Optional[Literal["true", "false"]] = None


class DiagramSchema(BaseModel):
    kind: Literal["flowchart", "diagram", "none"]
    nodes: list[_Node] = Field(default_factory=list, max_length=MAX_DIAGRAM_NODES)
    edges: list[_Edge] = Field(default_factory=list, max_length=MAX_DIAGRAM_EDGES)


_STRUCTURE_KIND = {
    "diagram": SemanticStructureKind.DIAGRAM,
    "flowchart": SemanticStructureKind.FLOWCHART,
}


class VisionJsonModel(Protocol):
    def generate_json(
        self,
        prompt_or_memory: str,
        images: list[str] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> object: ...


class VLMDiagramExtractor:
    """Extract a figure's topology, or None when it has none to report."""

    def __init__(self, vlm: VisionJsonModel) -> None:
        self._vlm = vlm

    def __call__(self, data: bytes, ocr_text: str) -> DiagramGraph | None:
        prompt = PROMPT
        if ocr_text.strip():
            prompt += GROUNDED_SUFFIX.format(ocr_text=ocr_text[:MAX_GROUNDING_CHARS])
        image = base64.b64encode(data).decode("ascii")
        result = self._vlm.generate_json(prompt, images=[image], schema=DiagramSchema)
        if not isinstance(result, DiagramSchema):
            result = DiagramSchema.model_validate(result)
        kind = _STRUCTURE_KIND.get(result.kind)
        if kind is None or not result.nodes:
            return None
        return DiagramGraph(
            kind=kind,
            nodes=tuple(
                DiagramNode(
                    key=item.key,
                    label=item.label,
                    role=item.role,
                    inside=item.inside,
                )
                for item in result.nodes
            ),
            edges=tuple(
                DiagramEdge(
                    source=item.source,
                    target=item.target,
                    label=item.label,
                    branch=item.branch,
                )
                for item in result.edges
            ),
        )


__all__ = ["DiagramSchema", "VLMDiagramExtractor", "VisionJsonModel"]
