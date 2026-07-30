from pydantic import BaseModel, Field, field_validator
import pickle
import logging
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import Optional, Dict, Literal, Any, List, Set, Union
import os

log = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Enum for node types in the document tree.

    A node keeps its own type rather than collapsing into TEXT because type
    decides how the node is summarised, what records it contributes to the
    index, and how an answer renders it.  A code listing summarised as prose,
    or a chart indexed as a paragraph, loses the thing that made it findable.
    """

    ROOT = "root"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    TITLE = "title"
    LIST = "list"
    CODE = "code"
    CHART = "chart"
    DIAGRAM = "diagram"
    ALGORITHM = "algorithm"
    ABSTRACT = "abstract"
    REFERENCE = "reference"
    # Page furniture. A running header, a page number and a scanner watermark
    # are printed on the page but say nothing about the document, so they carry
    # their own types rather than passing as prose: the canonical record keeps
    # them for provenance while the tree stops answering questions out of them.
    HEADER = "header"
    FOOTER = "footer"
    ASIDE = "aside"
    DISCARDED = "discarded"
    UNKNOWN = "unknown"


# Nodes that exist for traceability only -- never embedded, never summarised,
# never a section heading, never quoted in an answer.
FURNITURE_NODE_TYPES = frozenset(
    {NodeType.HEADER, NodeType.FOOTER, NodeType.ASIDE, NodeType.DISCARDED}
)


class MetaInfo(BaseModel):
    # document info
    file_name: str | None = Field(description="the name of the file", default=None)
    file_path: str | None = Field(description="the path of the file", default=None)
    source_uri: str | None = Field(
        description="the original source URI/path supplied by the user", default=None
    )
    source_sha256: str | None = Field(
        description="SHA-256 of the immutable source bytes", default=None
    )
    snapshot_path: str | None = Field(
        description="the immutable local snapshot parsed by this run", default=None
    )
    document_version_id: str | None = Field(
        description="stable identity of the exact source-byte revision", default=None
    )
    canonical_artifact_path: str | None = Field(
        description="path to the source-bound canonical evidence artifact",
        default=None,
    )
    canonical_artifact_sha256: str | None = Field(
        description="SHA-256 of the exact canonical evidence artifact bytes",
        default=None,
    )
    canonical_artifact_size_bytes: int | None = Field(
        description="size of the exact canonical evidence artifact",
        default=None,
    )
    routed_audit_artifact_path: str | None = Field(
        description="path to the replayable routed parsing audit artifact",
        default=None,
    )
    routed_audit_artifact_sha256: str | None = Field(
        description="SHA-256 of the exact routed parsing audit bytes",
        default=None,
    )
    routed_audit_artifact_size_bytes: int | None = Field(
        description="size of the exact routed parsing audit artifact",
        default=None,
    )

    # page info
    page_idx: int | None = Field(description="the page index", default=None)
    page_path: str | None = Field(
        description="the path of the page image", default=None
    )

    # item info from PDF extractor
    pdf_id: int | None = Field(
        description="the unique identifier of the item in the PDF", default=None
    )
    pdf_para_block: dict | None = Field(
        description="the paragraph block information from the PDF extractor",
        default=None,
    )
    canonical_block_id: str | None = Field(
        description="stable canonical evidence block identity", default=None
    )
    canonical_revision_id: str | None = Field(
        description="immutable revision of the canonical evidence block", default=None
    )
    observation_id: str | None = Field(
        description="source parser observation used by this node", default=None
    )
    derived_evidence_id: str | None = Field(
        description="stable identity of content derived after parser refinement",
        default=None,
    )
    source_canonical_block_ids: list[str] | None = Field(
        description="ordered canonical blocks supporting derived content",
        default=None,
    )
    source_canonical_revision_ids: list[str] | None = Field(
        description="ordered immutable canonical revisions supporting derived content",
        default=None,
    )
    page_id: str | None = Field(
        description="stable page identity within the document revision", default=None
    )
    source_geometry: dict | None = Field(
        description="page coordinate space, bounding box, and polygon", default=None
    )
    semantic_subtype: str | None = Field(
        description="parser-provided semantic specialization", default=None
    )
    parser_provenance: dict | None = Field(
        description="parser/model/config identity for the observation", default=None
    )
    quality_status: str | None = Field(
        description="canonical reconciliation quality status", default=None
    )
    reconciliation_method: str | None = Field(
        description="method used to create the canonical evidence", default=None
    )
    source_observation_ids: list[str] | None = Field(
        description="all parser observations supporting the canonical block",
        default=None,
    )
    source_evidence_refs: list[dict] | None = Field(
        description=(
            "ordered source block, observation, page, and geometry references for "
            "derived content"
        ),
        default=None,
    )
    canonical_representation_ids: list[str] | None = Field(
        description="immutable canonical representation identities available to this node",
        default=None,
    )
    canonical_asset_ids: list[str] | None = Field(
        description="immutable canonical asset identities available to this node",
        default=None,
    )
    selected_representation_ids: list[str] | None = Field(
        description=(
            "canonical representations actually selected for the BookRAG node view"
        ),
        default=None,
    )
    selected_asset_ids: list[str] | None = Field(
        description="canonical assets actually selected for the BookRAG node view",
        default=None,
    )
    canonical_relations: list[dict] | None = Field(
        description=(
            "directed canonical structural relations with exact parser-assertion lineage"
        ),
        default=None,
    )
    semantic_substructures: list[dict] | None = Field(
        description=(
            "bounded canonical table, formula, diagram, flowchart, or layout "
            "subgraphs with exact parser-observation lineage"
        ),
        default=None,
    )
    transformations: list[str] | None = Field(
        description="ordered transformations applied after canonical parsing",
        default=None,
    )

    # image and table info
    img_path: str | None = Field(
        description="the path of the image or table", default=None
    )
    image_width: int | None = Field(
        description="the width of the image or table", default=0
    )
    image_height: int | None = Field(
        description="the height of the image or table", default=0
    )
    caption: str | None = Field(
        description="the caption of the image or table", default=None
    )
    footnote: str | None = Field(
        description="the footnote of the image or table", default=None
    )

    # table info
    table_body: str | None = Field(
        description="the body content of the table", default=None
    )

    # text info, TreeNodes of Any type have the content
    content: str | None = Field(description="the content of the text", default=None)

    # title info
    title_level: int | None = Field(
        description="the level of the title, 0 is the root", default=-1
    )

    # listing info
    code_language: str | None = Field(
        description="the language MinerU named on a listing's fence, if any",
        default=None,
    )

    @field_validator("semantic_substructures", mode="before")
    @classmethod
    def _validate_semantic_substructures(cls, value):
        """Keep only bounded, canonical, source-bound JSON graph DTOs.

        This validation is intentionally lazy so historical BookRAG-only trees do
        not import the canonical contract.  New evidence cannot smuggle an
        arbitrary or unbounded nested object through generic node metadata.
        """

        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError("semantic_substructures must be an array")

        from docarag.domain import (
            MAX_SEMANTIC_BLOCK_CANONICAL_BYTES,
            MAX_SEMANTIC_ELEMENTS_PER_BLOCK,
            MAX_SEMANTIC_RELATIONS_PER_BLOCK,
            MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
            SemanticSubstructure,
        )
        from docarag.domain.base import canonical_json_bytes

        if len(value) > MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK:
            raise ValueError("semantic_substructures limit exceeded")

        normalized: list[dict] = []
        element_count = 0
        relation_count = 0
        for position, raw_structure in enumerate(value):
            if not isinstance(raw_structure, Mapping):
                raise ValueError(
                    f"semantic_substructures[{position}] must be an object"
                )
            try:
                structure = SemanticSubstructure.from_dict(raw_structure)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"semantic_substructures[{position}] is not a valid "
                    "canonical semantic graph"
                ) from exc
            if not structure.is_source_bound:
                raise ValueError(
                    f"semantic_substructures[{position}] lacks source lineage"
                )
            element_count += len(structure.elements)
            relation_count += len(structure.relations)
            if element_count > MAX_SEMANTIC_ELEMENTS_PER_BLOCK:
                raise ValueError("semantic_substructures element limit exceeded")
            if relation_count > MAX_SEMANTIC_RELATIONS_PER_BLOCK:
                raise ValueError("semantic_substructures relation limit exceeded")
            normalized.append(structure.to_dict())

        if len(canonical_json_bytes(normalized)) > MAX_SEMANTIC_BLOCK_CANONICAL_BYTES:
            raise ValueError("semantic_substructures byte limit exceeded")
        return normalized


class TreeNode:
    def __init__(self, meta_dict: dict = None):
        self.children: List["TreeNode"] = []
        self.parent: "TreeNode" = None
        self.type: NodeType = None
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.depth = 0
        self.index_id: int = -1  # Unique identifier for the node, should be set later
        self.outline_node: bool = False  # Indicates if the node is a outline node
        self.summary: str = ""  # Summary of the node content

    def add_child(self, child_node: "TreeNode"):
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)

    def get_meta_info(self):
        return self.meta_info

    def get_outline_entries(self) -> list:
        """
        returns a list of tuples containing the outline entries.

        Each tuple contains (depth, title, id) for the node.
        """
        entries = []
        if not self.outline_node:
            return entries

        title = getattr(self.meta_info, "content", "Untitled")
        entries.append((self.depth, title, self.index_id))

        for child in self.children:
            entries.extend(child.get_outline_entries())

        return entries


class DocumentTree:
    def __init__(self, meta_dict: dict = None, cfg: Optional[Dict[str, Any]] = None):
        self.nodes: list[TreeNode] = []
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.root_node: Optional[TreeNode] = None
        self.init_root_node(meta_dict) if meta_dict else None
        self.save_dir = cfg.save_path
        self.pdf_id_to_index_id: Dict[int, int] = {}  # Maps pdf_id to index_id
        self.max_depth = -1

    def init_root_node(self, meta_dict: dict):
        self.root_node = TreeNode(meta_dict)
        self.root_node.index_id = 0
        self.root_node.depth = 0
        self.root_node.type = "root"
        self.root_node.meta_info.pdf_id = 0  # Root node has pdf_id 0
        self.nodes.append(self.root_node)

    def get_nodes(self, hasRoot: bool = False) -> list[TreeNode]:
        if hasRoot:
            return self.nodes
        else:
            return self.nodes[1:] if len(self.nodes) > 1 else []

    def get_outline(self):
        if self.root_node is None:
            return ""

        outline_entries = []
        for child in self.root_node.children:
            outline_entries.extend(child.get_outline_entries())

        lines = [f"{level}\t{title}\t{id_}" for level, title, id_ in outline_entries]
        return "\n".join(lines)

    def add_node(self, node: TreeNode):
        node.index_id = len(self.nodes)
        self.nodes.append(node)
        pdf_id = node.meta_info.pdf_id
        if pdf_id is not None:
            # Map the pdf_id to the index_id of the node
            self.pdf_id_to_index_id[pdf_id] = node.index_id

    def get_node_by_index_id(self, node_id: int) -> TreeNode:
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def get_nodes_by_ids(self, id_list: list[int]) -> list[TreeNode]:
        return [self.nodes[i] for i in id_list if 0 <= i < len(self.nodes)]

    def get_node_by_pdf_id(self, pdf_id: int) -> TreeNode:
        """
        Returns the first node with the given pdf_id.
        If no node is found, returns None.
        """
        node_idx = self.pdf_id_to_index_id.get(pdf_id, None)
        if node_idx is not None:
            # If the pdf_id is mapped to an index_id, return the node directly
            return self.nodes[node_idx]

        # If the pdf_id is not mapped, search through the nodes
        if len(self.nodes) > pdf_id and self.nodes[pdf_id].meta_info.pdf_id == pdf_id:
            # If the pdf_id matches the index_id, return the node directly
            # This is a special case where the pdf_id is used as the index_id
            return self.nodes[pdf_id]

        for node in self.nodes:
            if node.meta_info.pdf_id == pdf_id:
                return node
        return None

    def get_max_depth(self) -> int:
        if self.max_depth != -1:
            return self.max_depth
        if not self.root_node:
            return 0
        self.max_depth = 0
        for node in self.nodes:
            if node.depth > self.max_depth:
                self.max_depth = node.depth
        return self.max_depth

    def get_path_from_root(self, node_id: int) -> List[TreeNode]:
        """
        Returns the path from the root node to the node with the given index_id.
        If the node does not exist, returns an empty list.
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return []

        path: List[TreeNode] = []
        visited_ids: Set[int] = set()

        # 循环直到找到根节点或遇到无效节点
        while node:
            # 终止条件1: 通用的根节点判断 (没有父节点)
            # 原始逻辑是不包含根节点，所以我们在添加节点前先判断
            if node.parent is None:
                break

            # 终止条件2: 原始的根节点ID为0的判断
            if node.index_id == 0:
                break

            # 终止条件3: 检测到任何循环 (当前ID已在访问集合中)
            if node.index_id in visited_ids:
                break

            path.append(node)
            visited_ids.add(node.index_id)
            node = node.parent

        return path[::-1]

    def get_sibling_nodes(self, node_id: int) -> List[TreeNode]:
        """
        Returns a list of sibling nodes for the node with the given index_id.
        If the node does not exist or has no siblings, returns an empty list.
        """
        node = self.get_node_by_index_id(node_id)
        if not node or not node.parent:
            return []

        siblings = [
            sibling
            for sibling in node.parent.children
            if sibling.index_id != node.index_id
        ]
        return siblings

    def get_subtree_nodes(self, node_ids: Union[List[int], int]) -> List[TreeNode]:
        """
        Returns a unique list of TreeNode objects that are part of the subtrees 
        rooted at the given node_ids.
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        unique_nodes = {}
        visited_ids = set()

        for node_id in node_ids:
            # The helper function will populate the unique_nodes dictionary
            self._get_subtree_recursive(node_id, unique_nodes, visited_ids)
        
        return list(unique_nodes.values())

    def _get_subtree_recursive(self, node_id: int, unique_nodes: Dict[int, TreeNode], visited_ids: set):
        """
        Helper function to recursively traverse the tree and collect unique nodes.
        """
        if node_id in visited_ids:
            return

        node = self.get_node_by_index_id(node_id)
        if not node:
            return

        # Mark as visited and add to our collection
        visited_ids.add(node_id)
        unique_nodes[node_id] = node

        # Recurse for children
        for child in node.children:
            self._get_subtree_recursive(child.index_id, unique_nodes, visited_ids)


    def get_ancestor_at_depth(self, node_id: int, depth: int) -> Optional[TreeNode]:
        """
        Returns the ancestor of the node with the given index_id at the specified depth.
        If the node does not exist or the depth is invalid, returns None.
        """
        node = self.get_node_by_index_id(node_id)
        if not node or depth < 0:
            return None

        while node and node.depth > depth:
            node = node.parent
        return node if node and node.depth == depth else None

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """
        Returns a list of nodes that are at the specified depth.
        If no nodes are found at that depth, returns an empty list.
        """
        if depth < 0:
            return []

        nodes_at_depth = [node for node in self.nodes if node.depth == depth]
        return nodes_at_depth

    def get_nodes_data(
        self, node_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries containing the data of the nodes.
        If node_ids is provided, only those nodes will be included.
        """
        if node_ids is None:
            return []

        nodes_data = []
        for node_id in node_ids:
            node = self.get_node_by_index_id(node_id)
            content = node.meta_info.content
            page_idx = node.meta_info.page_idx
            node_data = {
                "index_id": node.index_id,
                "type": node.type,
                "content": content,
                "page": page_idx,
                # "summary": node.summary,
            }
            # Keep the historical retrieval payload unchanged for old trees,
            # while exposing complete evidence bindings for canonical nodes.
            trace_fields = {
                "page_id": node.meta_info.page_id,
                "document_version_id": node.meta_info.document_version_id,
                "canonical_block_id": node.meta_info.canonical_block_id,
                "canonical_revision_id": node.meta_info.canonical_revision_id,
                "observation_id": node.meta_info.observation_id,
                "derived_evidence_id": node.meta_info.derived_evidence_id,
                "source_canonical_block_ids": (
                    node.meta_info.source_canonical_block_ids
                ),
                "source_canonical_revision_ids": (
                    node.meta_info.source_canonical_revision_ids
                ),
                "source_observation_ids": node.meta_info.source_observation_ids,
                "source_evidence_refs": node.meta_info.source_evidence_refs,
                "canonical_representation_ids": (
                    node.meta_info.canonical_representation_ids
                ),
                "canonical_asset_ids": node.meta_info.canonical_asset_ids,
                "selected_representation_ids": (
                    node.meta_info.selected_representation_ids
                ),
                "selected_asset_ids": node.meta_info.selected_asset_ids,
                "canonical_relations": node.meta_info.canonical_relations,
                # ``getattr`` keeps retrieval compatible with MetaInfo objects
                # unpickled from releases predating this optional field.
                "semantic_substructures": getattr(
                    node.meta_info, "semantic_substructures", None
                ),
                "transformations": node.meta_info.transformations,
                "geometry": node.meta_info.source_geometry,
                "semantic_subtype": node.meta_info.semantic_subtype,
                "parser_provenance": node.meta_info.parser_provenance,
                "quality_status": node.meta_info.quality_status,
                "reconciliation_method": node.meta_info.reconciliation_method,
                "source_uri": node.meta_info.source_uri or self.meta_info.source_uri,
                "source_sha256": (
                    node.meta_info.source_sha256 or self.meta_info.source_sha256
                ),
                "snapshot_path": (
                    node.meta_info.snapshot_path or self.meta_info.snapshot_path
                ),
            }
            node_data.update(
                {key: value for key, value in trace_fields.items() if value is not None}
            )
            # Image-backed evidence is not limited to IMAGE nodes. MinerU also
            # emits crops for tables and interline equations; retrieval must
            # retain those paths after the tree serialization boundary.
            if node.type in (NodeType.IMAGE, NodeType.TABLE, NodeType.EQUATION):
                node_data["img_path"] = node.meta_info.img_path
            if node.type is NodeType.TABLE:
                node_data["table_body"] = node.meta_info.table_body
                node_data["caption"] = node.meta_info.caption
                node_data["footnote"] = node.meta_info.footnote

            nodes_data.append(node_data)
        return nodes_data

    def get_filtered_nodes(self, node_type: Union[str, NodeType]) -> List[TreeNode]:
        """
        Input a node type (str or NodeType), str from ["text", "image", "table", "equation", "title", "root"]
        Returns a list of nodes of the specified type.
        """
        if isinstance(node_type, str):
            node_type = NodeType(node_type)
        else:
            node_type = node_type
        sel_nodes = [node for node in self.nodes if node.type == node_type]

        return sel_nodes

    def to_json_summary(self):
        """
        Dump the document tree index to a JSON-like summary format.
        This includes a list of nodes with their index_id, parent_id, type,
        meta_info, and summary.
        This json is used for visualization and debugging purposes.
        """
        node_summaries = []
        for node in self.nodes:
            node_summaries.append(
                {
                    "index_id": node.index_id,
                    "parent_id": node.parent.index_id if node.parent else None,
                    "type": (node.type.value if isinstance(node.type, NodeType) else str(node.type)) if node.type else None,
                    "meta_info": (
                        node.meta_info.model_dump() if node.meta_info else None
                    ),
                    "summary": node.summary,
                }
            )
        return {"nodes": node_summaries, "meta_info": self.meta_info.model_dump()}

    def get_one_depth_summary(self, node_id: int) -> str:
        """
        Returns a one-depth summary of the node with the given index_id.
        If the node does not exist, returns an empty string.
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return ""

        cur_summary = str(node.index_id) + ": " + (node.summary or "")
        if not node.children:
            # If the node has no children, return its own summary
            return cur_summary

        # Collect summaries from all children
        summaries = [cur_summary]
        summaries.append("Children summaries:")
        for child in node.children:
            child_summary = str(child.index_id) + ": " + (child.summary or "")
            if child.summary:
                summaries.append(child_summary)

        # Join summaries with a newline
        return "\n".join(summaries)

    def save_to_file(self):
        save_file_path = DocumentTree.get_save_path(self.save_dir)
        with open(save_file_path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Document tree index saved to {save_file_path}")

        import json

        summary = self.to_json_summary()

        # Evidence persistence reads this one. It is a plain data dump of the
        # in-memory tree at the moment save_to_file runs -- never hand-edit
        # it; a correction belongs in the code that built the tree, then a
        # rebuild, so this file and the pickle it was written alongside never
        # disagree with each other or with what gets projected into Postgres.
        structure_path = os.path.join(self.save_dir, "tree_structure.json")
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info(f"Document tree structure saved to {structure_path}")

        # tree.json is a debug/visualization dump only. Nothing in the
        # pipeline reads it back; it is not evidence and never authoritative.
        json_path = os.path.join(self.save_dir, "tree.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info(f"Document tree summary (debug/visualization only) saved to {json_path}")

    @staticmethod
    def get_save_path(input_dir: str) -> str:
        return os.path.join(input_dir, "tree.pkl")

    @staticmethod
    def load_from_file(filepath: str) -> "DocumentTree":
        with open(filepath, "rb") as f:
            return pickle.load(f)
