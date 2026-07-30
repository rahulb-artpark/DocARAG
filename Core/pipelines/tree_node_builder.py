import logging
from typing import Optional

from Core.Index.Tree import NodeType, TreeNode

log = logging.getLogger(__name__)


def _evidence_meta(pdf_content: dict) -> dict:
    """Fields shared by every legacy node type for source-level traceability."""

    return {
        "document_version_id": pdf_content.get("document_version_id"),
        "source_sha256": pdf_content.get("source_sha256"),
        "page_id": pdf_content.get("page_id"),
        "canonical_block_id": pdf_content.get("canonical_block_id"),
        "canonical_revision_id": pdf_content.get("canonical_revision_id"),
        "observation_id": pdf_content.get("observation_id"),
        "derived_evidence_id": pdf_content.get("derived_evidence_id"),
        "source_canonical_block_ids": pdf_content.get(
            "source_canonical_block_ids"
        ),
        "source_canonical_revision_ids": pdf_content.get(
            "source_canonical_revision_ids"
        ),
        "source_observation_ids": pdf_content.get("source_observation_ids"),
        "source_evidence_refs": pdf_content.get("source_evidence_refs"),
        "canonical_representation_ids": pdf_content.get(
            "canonical_representation_ids"
        ),
        "canonical_asset_ids": pdf_content.get("canonical_asset_ids"),
        "selected_representation_ids": pdf_content.get(
            "selected_representation_ids"
        ),
        "selected_asset_ids": pdf_content.get("selected_asset_ids"),
        "canonical_relations": pdf_content.get("canonical_relations"),
        "semantic_substructures": pdf_content.get("semantic_substructures"),
        "transformations": pdf_content.get("transformations"),
        "source_geometry": pdf_content.get(
            "geometry", pdf_content.get("source_geometry")
        ),
        "semantic_subtype": pdf_content.get("semantic_subtype"),
        "parser_provenance": pdf_content.get("parser_provenance"),
        "quality_status": pdf_content.get("quality_status"),
        "reconciliation_method": pdf_content.get("reconciliation_method"),
    }


# Projected type string -> node type, for blocks whose evidence is text and
# for those whose evidence is a crop. Mirrors bookrag_projection's
# _NODE_TYPE_BY_BLOCK / _VISUAL_NODE_TYPE_BY_BLOCK. An unrecognised type still
# falls back to TEXT below, so a future parser type degrades rather than
# raising mid-ingestion.
_TEXT_CONTENT_TYPES = {
    "text": NodeType.TEXT,
    "list": NodeType.LIST,
    "code": NodeType.CODE,
    "algorithm": NodeType.ALGORITHM,
    "abstract": NodeType.ABSTRACT,
    "reference": NodeType.REFERENCE,
    "header": NodeType.HEADER,
    "footer": NodeType.FOOTER,
    "aside": NodeType.ASIDE,
    "discarded": NodeType.DISCARDED,
}
_VISUAL_CONTENT_TYPES = {
    "image": NodeType.IMAGE,
    "chart": NodeType.CHART,
    "diagram": NodeType.DIAGRAM,
}


def create_node_by_type(pdf_content: Optional[dict], isTitle: bool) -> TreeNode:
    if pdf_content is None:
        raise ValueError("pdf_content must be a normalized block")
    evidence_meta = _evidence_meta(pdf_content)
    content_type = pdf_content.get("type", "unknown")
    if content_type in _TEXT_CONTENT_TYPES:
        node_meta = {
            "content": pdf_content.get("text", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            **evidence_meta,
        }
        if isTitle:
            level = pdf_content.get("text_level", -1)
            if isinstance(level, str):
                try:
                    level = int(level)
                    node_meta["title_level"] = level
                except ValueError:
                    level = -1
                    isTitle = False
            else:
                node_meta["title_level"] = level

        node = TreeNode(node_meta)
        # A title promotion wins over the block's own type: the outline pass
        # has decided this block opens a section. Otherwise the node keeps the
        # type the projection gave it, so a code listing stays code.
        node.type = (
            NodeType.TITLE if isTitle else _TEXT_CONTENT_TYPES[content_type]
        )
        node.outline_node = isTitle
    elif content_type in _VISUAL_CONTENT_TYPES:
        caption = pdf_content.get("image_caption", [])
        caption_str = " ".join(caption) if isinstance(caption, list) else ""
        footnote = pdf_content.get("image_footnote", [])
        footnote_str = " ".join(footnote) if isinstance(footnote, list) else ""
        node_meta = {
            "img_path": pdf_content.get("img_path", ""),
            "caption": caption_str,
            "footnote": footnote_str,
            "content": " ".join(
                part for part in (caption_str, footnote_str) if part
            ),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            **evidence_meta,
        }
        node = TreeNode(node_meta)
        node.type = _VISUAL_CONTENT_TYPES[content_type]
    elif content_type == "table":
        caption = pdf_content.get("table_caption", [])
        caption_str = " ".join(caption) if isinstance(caption, list) else ""
        footnote = pdf_content.get("table_footnote", [])
        footnote_str = " ".join(footnote) if isinstance(footnote, list) else ""

        node_meta = {
            "img_path": pdf_content.get("img_path", ""),
            "caption": caption_str,
            "footnote": footnote_str,
            "content": " ".join(
                part for part in (caption_str, footnote_str) if part
            ),
            "table_body": pdf_content.get("table_body", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            **evidence_meta,
        }
        node = TreeNode(node_meta)
        node.type = NodeType.TABLE
    elif content_type == "equation":
        node_meta = {
            "content": pdf_content.get("text", ""),
            # MinerU emits a crop for interline equations.  Preserve it just as
            # we do for images and tables so downstream retrieval can use both
            # the parsed LaTeX and the original pixels.
            "img_path": pdf_content.get("img_path", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            "text_format": pdf_content.get("text_format", ""),
            **evidence_meta,
        }
        node = TreeNode(node_meta)
        node.type = NodeType.EQUATION
    else:
        log.warning(f"Unknown content type: {content_type}. Defaulting to text.")
        node_meta = {
            "content": pdf_content.get("text", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            **evidence_meta,
        }
        node = TreeNode(node_meta)
        node.type = NodeType.TEXT

    return node
