"""Deterministic identity helpers for transformed BookRAG evidence.

The legacy refinement pipeline and the dependency-light evidence hydrator must
compute the same identity without importing each other or any model runtime.
Keeping that algorithm here prevents a derived tree node from retaining a
plausible-looking ``derived:<digest>`` after its output or lineage is changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import hashlib
import json


DERIVED_EVIDENCE_PREFIX = "derived:"


def _node_type(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    return value if isinstance(value, str) and value else "unknown"


def _joined_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return " ".join(item for item in value if isinstance(item, str))
    return ""


def channel_text(value: object) -> str:
    """One BookRAG text channel read as the single string a node carries.

    Caption and footnote arrive as lists of entries; a node holds one string.
    """

    return " ".join(value) if isinstance(value, list) else ""


def caption_footnote_text(caption: object, footnote: object) -> str:
    """The content a caption/footnote-bearing node carries: both channels.

    A visual or table block has no ``text`` channel -- its evidence is the
    caption and footnote -- so this is the whole of what such a node retains.
    Three places must agree on it exactly: the tree builder writes it, the
    refiner seals a derived node's identity over it, and the hydrator
    recomputes both from the canonical projection. It lives here because a
    separator that differs between any two of them is silent until a figure
    that has both a caption and a footnote reaches an answer, and then that
    figure is refused.
    """

    parts = (channel_text(caption), channel_text(footnote))
    return " ".join(part for part in parts if part)


def derived_output_identity(
    *,
    content_type: object,
    text: object = "",
    table_body: object = "",
    table_caption: object = "",
    table_footnote: object = "",
) -> dict[str, str]:
    """Return the stable semantic output retained by a BookRAG tree node."""

    normalized_type = _node_type(content_type)
    if normalized_type == "table":
        return {
            "type": normalized_type,
            "table_body": table_body if isinstance(table_body, str) else "",
            "table_caption": _joined_text(table_caption),
            "table_footnote": _joined_text(table_footnote),
        }
    return {
        "type": normalized_type,
        "text": text if isinstance(text, str) else "",
    }


def derived_identity_payload(
    *,
    output: Mapping[str, object],
    source_revision_ids: Sequence[str],
    source_document_version_ids: Sequence[str],
    source_block_ids: Sequence[str],
    source_observation_ids: Sequence[str],
    source_references: Sequence[Mapping[str, object]],
    transformations: Sequence[str],
) -> dict[str, object]:
    """Build the version-one identity payload in its canonical field order."""

    return {
        "output": dict(output),
        "source_revision_ids": list(source_revision_ids),
        "source_document_version_ids": list(source_document_version_ids),
        "source_block_ids": list(source_block_ids),
        "source_observation_ids": list(source_observation_ids),
        "source_references": [dict(reference) for reference in source_references],
        "transformations": list(transformations),
    }


def derived_evidence_id(payload: Mapping[str, object]) -> str:
    """Hash a JSON-compatible derived identity payload deterministically."""

    digest = hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"{DERIVED_EVIDENCE_PREFIX}{digest}"
