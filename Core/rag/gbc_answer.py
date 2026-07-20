from typing import List, Optional, Dict, Any, Tuple, Union, Literal, Mapping
import json
import re

from Core.provider.llm import LLM
from Core.provider.vlm import VLM

from Core.Index.Tree import TreeNode, NodeType
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.prompts.gbc_prompt import (
    ITER_GENERATION_SYS_PROMPT,
    ITER_GENERATION_USER_PROMPT,
    ITER_GENERATION_GRAPH,
    VLM_GENERATION_USER_PROMPT,
    VLM_ONE_GENERATION_USER_PROMPT,
    SYNTHESIS_SYS_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from Core.utils.utils import num_tokens, TextProcessor
from Core.utils.table_utils import table2text
from Core.rag.gbc_plan import SubQuestion
from docarag.answer_validation import validate_answer_citations
import logging

log = logging.getLogger(__name__)


CITATION_INSTRUCTION = (
    "Cite factual statements with the supplied evidence labels, for example "
    "[E1], or with the scoped labels supplied for a complex query, for example "
    "[Q1-E1]. Use only labels present in the evidence and preserve each label "
    "exactly. If the evidence is insufficient, say so instead of inventing a "
    "citation."
)
CITATION_VALIDATION_FAILURE_ANSWER = (
    "I could not produce an answer with valid references to the retrieved "
    "source evidence, so no grounded answer is returned."
)
CITED_GENERATION_SYS_PROMPT = (
    f"{ITER_GENERATION_SYS_PROMPT.rstrip()}\n\n{CITATION_INSTRUCTION}"
)
CITED_SYNTHESIS_SYS_PROMPT = (
    f"{SYNTHESIS_SYS_PROMPT.rstrip()}\n\n{CITATION_INSTRUCTION}"
)
_EVIDENCE_LABEL = re.compile(r"(?:Q[1-9][0-9]*-)?E[1-9][0-9]*")
_EVIDENCE_MARKER = re.compile(r"\[((?:Q[1-9][0-9]*-)?E[1-9][0-9]*)\]")
_EVIDENCE_SCOPE = re.compile(r"Q[1-9][0-9]*")
_CANONICAL_CITATION_ID = re.compile(r"citation_[0-9a-f]{32}")


class EvidencePromptMemory(Memory):
    """A model prompt plus provenance captured before rendering document text."""

    evidence_labels: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()


def _evidence_identity(node: Dict[str, Any], position: int) -> str:
    """Return the strongest immutable evidence identity available."""

    for field_name in (
        "derived_evidence_id",
        "canonical_revision_id",
        "canonical_block_id",
    ):
        value = node.get(field_name)
        if isinstance(value, str) and value:
            return value
    document_version = node.get("document_version_id")
    node_id = node.get("index_id", node.get("node_id"))
    if document_version is not None and node_id is not None:
        return f"legacy:{document_version}:node:{node_id}"
    return f"legacy:retrieval-position:{position}"


def _citation_index(evidence_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return each evidence label once, in first-seen order, for a UI to link.

    ``evidence_nodes`` already carries the labels an answer's inline [E1]
    citations refer to (set in ``_prepare_evidence``/``answer_global_question``);
    this is the same list, deduplicated, describing what each label points to.
    """

    seen: set[str] = set()
    citations: List[Dict[str, Any]] = []
    for node in evidence_nodes:
        evidence_id = node.get("evidence_id")
        label = node.get("evidence_label")
        if not label or evidence_id in seen:
            continue
        seen.add(evidence_id)
        content = node.get("content")
        snippet = content.strip() if isinstance(content, str) else ""
        node_type = node.get("type", "text")
        if isinstance(node_type, NodeType):
            node_type = node_type.value
        citations.append(
            {
                "label": label,
                "index_id": node.get("index_id", node.get("node_id")),
                "type": str(node_type),
                "page": node.get("page"),
                "snippet": snippet[:400],
                "has_image": bool(node.get("img_path")),
            }
        )
    return citations


_CANONICAL_TEXT_KINDS = (
    "text",
    "ocr_text",
    "visual_description",
    "caption",
    "footnote",
    "markdown",
)


def _canonical_representation_text(node: Dict[str, Any]) -> str:
    """Text carried by the node's canonical evidence.

    The legacy tree leaves some records without content (figures/diagrams have
    only a crop). Their canonical block may still carry OCR, a visual
    description, or a caption; read it rather than answering from nothing.
    """

    sources = node.get("canonical_sources")
    if not isinstance(sources, list):
        return ""
    parts: list[str] = []
    for block in sources:
        if not isinstance(block, dict):
            continue
        for representation in block.get("representations") or []:
            if not isinstance(representation, dict):
                continue
            if representation.get("kind") not in _CANONICAL_TEXT_KINDS:
                continue
            content = representation.get("content")
            if isinstance(content, str) and content.strip():
                text = content.strip()
                if text not in parts:
                    parts.append(text)
    return "\n".join(parts)


def _evidence_text(node: Dict[str, Any], content: object) -> str:
    """Render a compact, source-bound evidence header for model prompts."""

    node_type = node.get("type", "text")
    if isinstance(node_type, NodeType):
        node_type = node_type.value
    lines = [
        f"Evidence: [{node.get('evidence_label', 'E?')}]",
        f"Evidence ID: {node.get('evidence_id', 'unknown')}",
        f"Type: {node_type} in Page: {node.get('page', -1)}",
    ]
    for label, field_name in (
        ("Document version", "document_version_id"),
        ("Source SHA-256", "source_sha256"),
        ("Page ID", "page_id"),
    ):
        value = node.get(field_name)
        if value not in (None, ""):
            lines.append(f"{label}: {value}")
    citation = node.get("citation")
    if isinstance(citation, Mapping):
        citation_id = citation.get("citation_id")
        if citation_id:
            lines.append(f"Citation ID: {citation_id}")
        revisions = citation.get("canonical_revision_ids")
        if revisions:
            lines.append(
                "Canonical revisions: "
                + json.dumps(revisions, sort_keys=True, separators=(",", ":"))
            )
        locations = citation.get("locations")
        if isinstance(locations, list):
            compact_locations = [
                {
                    "page_index": location.get("page_index"),
                    "page_id": location.get("page_id"),
                    "geometry": location.get("geometry"),
                }
                for location in locations
                if isinstance(location, Mapping)
            ]
            lines.append(
                "Source locations: "
                + json.dumps(
                    compact_locations, sort_keys=True, separators=(",", ":")
                )
            )
    geometry = node.get("source_geometry") or node.get("geometry")
    if geometry:
        lines.append(
            "Geometry: "
            + json.dumps(geometry, sort_keys=True, separators=(",", ":"))
        )
    lines.append(f"Content: {content}")
    return "\n".join(lines) + "\n"


def _prompt_evidence_labels(memory: Memory) -> list[str]:
    """Read trusted prompt metadata without scraping untrusted source text."""

    labels = getattr(memory, "evidence_labels", ())
    return list(labels) if isinstance(labels, (tuple, list)) else []


def _prompt_citation_ids(memory: Memory) -> list[str]:
    """Read trusted canonical IDs without scraping untrusted source text."""

    citation_ids = getattr(memory, "citation_ids", ())
    return list(citation_ids) if isinstance(citation_ids, (tuple, list)) else []


def _visual_asset_is_verified(node: Mapping[str, Any]) -> bool:
    """Fail closed when canonical hydration has not verified asset bytes.

    Legacy non-canonical runs have no material-status field and retain their
    compatibility behavior. Canonical hydration explicitly emits
    ``reference_only`` until a bounded artifact-store resolver hashes the bytes.
    """

    citation = node.get("citation")
    if not isinstance(citation, Mapping):
        return True
    return citation.get("asset_material_status") == "verified"


def _evidence_bindings(
    evidence_nodes: List[Dict[str, Any]],
) -> dict[str, str | None]:
    """Build an unambiguous label-to-canonical-citation map."""

    bindings: dict[str, str | None] = {}
    for node in evidence_nodes:
        label = node.get("evidence_label")
        if not isinstance(label, str) or not label:
            continue
        citation = node.get("citation")
        citation_id = node.get("citation_id")
        if not citation_id and isinstance(citation, Mapping):
            citation_id = citation.get("citation_id")
        normalized = citation_id if isinstance(citation_id, str) else None
        if label in bindings and bindings[label] != normalized:
            raise ValueError(f"evidence label {label} has conflicting citations")
        bindings[label] = normalized
    return bindings


def _node_citation_id(node: Mapping[str, Any]) -> str | None:
    citation_id = node.get("citation_id")
    citation = node.get("citation")
    if not citation_id and isinstance(citation, Mapping):
        citation_id = citation.get("citation_id")
    if (
        isinstance(citation_id, str)
        and _CANONICAL_CITATION_ID.fullmatch(citation_id) is not None
    ):
        return citation_id
    return None


def _evidence_prompt_metadata(
    nodes: List[Dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive trusted trace metadata from node fields before prompt rendering."""

    labels: list[str] = []
    citation_ids: list[str] = []
    seen_labels: set[str] = set()
    seen_citation_ids: set[str] = set()
    for node in nodes:
        label = node.get("evidence_label")
        if isinstance(label, str) and _EVIDENCE_LABEL.fullmatch(label) is not None:
            bracketed = f"[{label}]"
            if bracketed not in seen_labels:
                seen_labels.add(bracketed)
                labels.append(bracketed)
        citation_id = _node_citation_id(node)
        if citation_id is not None and citation_id not in seen_citation_ids:
            seen_citation_ids.add(citation_id)
            citation_ids.append(citation_id)
    labels.sort(key=_evidence_label_sort_key)
    citation_ids.sort()
    return tuple(labels), tuple(citation_ids)


def _evidence_prompt_memory(
    user_prompt: str,
    nodes: List[Dict[str, Any]],
) -> EvidencePromptMemory:
    labels, citation_ids = _evidence_prompt_metadata(nodes)
    memory = EvidencePromptMemory(
        evidence_labels=labels,
        citation_ids=citation_ids,
    )
    memory.add(Message(role="system", content=CITED_GENERATION_SYS_PROMPT))
    memory.add(Message(role="user", content=user_prompt))
    return memory


def _evidence_label_sort_key(bracketed_label: str) -> tuple[int, int]:
    """Sort local and scoped evidence labels deterministically."""

    label = bracketed_label[1:-1]
    scoped = re.fullmatch(r"Q([1-9][0-9]*)-E([1-9][0-9]*)", label)
    if scoped:
        return int(scoped.group(1)), int(scoped.group(2))
    local = re.fullmatch(r"E([1-9][0-9]*)", label)
    if local:
        return 0, int(local.group(1))
    return (2**31 - 1, 2**31 - 1)


def _normalize_evidence_scope(scope: str | None) -> str | None:
    if scope is None:
        return None
    if not isinstance(scope, str) or _EVIDENCE_SCOPE.fullmatch(scope) is None:
        raise ValueError("evidence_scope must match Q<positive integer>")
    return scope


def _scope_answer_markers(answer: str, scope: str) -> str:
    """Namespace legacy local markers while preserving already-scoped ones."""

    return _EVIDENCE_MARKER.sub(
        lambda match: (
            f"[{scope}-{match.group(1)}]"
            if match.group(1).startswith("E")
            else match.group(0)
        ),
        answer,
    )


def _raw_complex_bindings(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the explicit binding map or its serialized validation trace."""

    explicit = result.get("evidence_bindings")
    if isinstance(explicit, Mapping):
        return explicit
    validation = result.get("answer_validation")
    if not isinstance(validation, Mapping):
        return {}
    allowed = validation.get("allowed_bindings")
    if not isinstance(allowed, list):
        return {}
    bindings: dict[str, Any] = {}
    for item in allowed:
        if not isinstance(item, Mapping):
            continue
        label = item.get("evidence_label")
        if isinstance(label, str):
            bindings[label] = item.get("citation_id")
    return bindings


def _scope_complex_bindings(
    result: Mapping[str, Any], scope: str
) -> tuple[dict[str, str | None], list[str]]:
    """Namespace one sub-question's bindings and report unsafe inputs."""

    scoped: dict[str, str | None] = {}
    errors: list[str] = []
    for raw_label, raw_citation_id in _raw_complex_bindings(result).items():
        if not isinstance(raw_label, str):
            errors.append(f"{scope}: evidence binding label is not a string")
            continue
        label = (
            raw_label[1:-1]
            if raw_label.startswith("[") and raw_label.endswith("]")
            else raw_label
        )
        if re.fullmatch(r"E[1-9][0-9]*", label):
            label = f"{scope}-{label}"
        elif not label.startswith(f"{scope}-") or _EVIDENCE_LABEL.fullmatch(label) is None:
            errors.append(
                f"{scope}: evidence label {raw_label!r} belongs to a different or invalid scope"
            )
            continue
        citation_id = raw_citation_id if isinstance(raw_citation_id, str) else None
        if label in scoped and scoped[label] != citation_id:
            errors.append(f"{scope}: evidence label {label!r} has conflicting citations")
            continue
        scoped[label] = citation_id
    return scoped, errors


class AnswerAgent:
    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        variant: str = None,
        evidence_hydrator: object | None = None,
    ):
        self.llm = llm
        self.vlm = vlm
        self.variant = variant
        self.evidence_hydrator = evidence_hydrator

    def _prepare_evidence(
        self,
        retrieved_nodes: List[Dict],
        *,
        evidence_scope: str | None = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separates retrieved nodes into text-based and image-based categories.
        Tables are processed to be included in both.
        """
        evidence_scope = _normalize_evidence_scope(evidence_scope)
        image_nodes, text_nodes = [], []
        label_by_identity: dict[str, str] = {}
        for position, original_node in enumerate(retrieved_nodes, start=1):
            node = original_node.copy()
            evidence_id = _evidence_identity(node, position)
            evidence_label = label_by_identity.setdefault(
                evidence_id,
                (
                    f"{evidence_scope}-E{len(label_by_identity) + 1}"
                    if evidence_scope
                    else f"E{len(label_by_identity) + 1}"
                ),
            )
            node["evidence_id"] = evidence_id
            node["evidence_label"] = evidence_label
            if not str(node.get("content") or "").strip():
                canonical_text = _canonical_representation_text(node)
                if canonical_text:
                    node["content"] = canonical_text
            node_type = node.get("type", "text")
            if isinstance(node.get("page"), int):
                node["page"] += 1  # make page start from 1
            if node_type == NodeType.IMAGE:
                if _visual_asset_is_verified(node):
                    image_nodes.append(node)
                else:
                    # Canonical OCR/caption/description text remains usable;
                    # only the unverified pixel material is withheld.
                    if node.get("content"):
                        text_nodes.append(node)
                    log.warning(
                        "Withholding reference-only visual asset for evidence %s",
                        evidence_id,
                    )
            elif node_type in (NodeType.TABLE, NodeType.EQUATION):
                if node_type == NodeType.TABLE:
                    node["content"] = table2text(node)
                text_nodes.append(node)
                if node.get("img_path") and _visual_asset_is_verified(node):
                    image_nodes.append(node.copy())
            else:
                text_nodes.append(node)
        return text_nodes, image_nodes

    def _build_prompts(
        self,
        query: str,
        text_nodes: List[Dict],
        image_nodes: List[Dict],
        graph_str: str,
    ) -> Tuple[List[Memory], List[Dict[str, Any]]]:
        """
        Builds chunked and formatted prompts for both LLM and VLM.
        """
        # 1. Build VLM prompts for image-based evidence
        image_prompts = []
        for node in image_nodes:
            evidence_labels, citation_ids = _evidence_prompt_metadata([node])
            img_path = node["img_path"]
            node_content = node.get("content", "")
            content = _evidence_text(node, node_content)
            vlm_prompt = (
                f"{CITED_GENERATION_SYS_PROMPT.strip()}\n\n"
                f"{VLM_GENERATION_USER_PROMPT.format(question=query, content=content).strip()}"
            )
            if img_path:
                image_prompts.append(
                    {
                        "prompt": vlm_prompt,
                        "image_url": img_path,
                        "evidence_id": node["evidence_id"],
                        "evidence_label": node["evidence_label"],
                        "citation_id": _node_citation_id(node),
                        "evidence_labels": list(evidence_labels),
                        "citation_ids": list(citation_ids),
                    }
                )

        # 2. Build chunked LLM prompts for text-based evidence
        text_prompts = []

        # Build the graph part of the prompt only if graph_str is not empty
        graph_prompt_part = ""
        if graph_str:
            graph_prompt_part = ITER_GENERATION_GRAPH.format(
                knowledge_graph_subgraph=graph_str
            )

        # Calculate token budget for retrieved content
        base_prompt_tokens = num_tokens(
            ITER_GENERATION_USER_PROMPT.format(
                user_question=query, retrieved_content=""
            )
            + graph_prompt_part
        )
        system_prompt_tokens = num_tokens(CITED_GENERATION_SYS_PROMPT)
        content_limit = (
            self.llm.config.max_tokens - system_prompt_tokens - base_prompt_tokens - 400
        )  # 400 as buffer
        content_limit = max(256, content_limit)

        processed_nodes = []
        for node in text_nodes:
            node_content = node.get("content", "")
            node_text = _evidence_text(node, node_content)
            full_node_tokens = num_tokens(node_text)

            if full_node_tokens > content_limit:
                header_tokens = num_tokens(_evidence_text(node, ""))
                sub_content_chunks = TextProcessor.split_text_into_chunks(
                    text=node_content,
                    max_length=max(content_limit - header_tokens, 1),
                )
                for chunk_index, chunk_content in enumerate(
                    sub_content_chunks, start=1
                ):
                    processed_node = dict(node)
                    processed_node["content"] = chunk_content
                    processed_node["evidence_chunk"] = chunk_index
                    processed_nodes.append(processed_node)
            else:
                processed_nodes.append(node)

        # Chunking logic
        current_chunk_str = ""
        current_chunk_tokens = 0
        current_chunk_nodes: List[Dict[str, Any]] = []
        separator = "\n\n---\n\n"

        for node in processed_nodes:
            node_content = node.get("content", "")
            node_text = _evidence_text(node, node_content)
            node_tokens = num_tokens(node_text)

            if current_chunk_str and (
                current_chunk_tokens + node_tokens > content_limit
            ):

                user_prompt = (
                    ITER_GENERATION_USER_PROMPT.format(
                        user_question=query, retrieved_content=current_chunk_str
                    )
                    + graph_prompt_part
                )
                text_prompts.append(
                    _evidence_prompt_memory(user_prompt, current_chunk_nodes)
                )

                # 用当前节点开始一个新块
                current_chunk_str = node_text
                current_chunk_tokens = node_tokens
                current_chunk_nodes = [node]
            else:
                # 将节点内容添加到当前块
                if not current_chunk_str:
                    current_chunk_str = node_text
                    current_chunk_tokens = node_tokens
                    current_chunk_nodes = [node]
                else:
                    current_chunk_str += separator + node_text
                    current_chunk_tokens += node_tokens
                    current_chunk_nodes.append(node)

        if current_chunk_str:
            user_prompt = (
                ITER_GENERATION_USER_PROMPT.format(
                    user_question=query, retrieved_content=current_chunk_str
                )
                + graph_prompt_part
            )
            text_prompts.append(
                _evidence_prompt_memory(user_prompt, current_chunk_nodes)
            )

        return text_prompts, image_prompts

    def _build_single_prompt(
        self,
        query: str,
        text_nodes: List[Dict],
        image_nodes: List[Dict],
        graph_str: str,
    ) -> Tuple[List[Memory], List[Dict[str, Any]]]:
        """
        Builds exactly one prompt:
        - If image evidence exists, return one VLM prompt in image_prompts.
        - Otherwise, return one LLM prompt in text_prompts.
        """
        separator = "\n\n---\n\n"
        separator_tokens = num_tokens(separator)
        graph_prompt_part = ""

        def _collect_retrieved_content(
            nodes: List[Dict], content_limit: int, include_graph_block: bool = False
        ) -> Tuple[str, List[Dict[str, Any]]]:
            if content_limit <= 0:
                return "", []

            segments: List[Tuple[str, Optional[Dict[str, Any]]]] = []

            if include_graph_block and graph_prompt_part:
                graph_block = graph_prompt_part.strip()
                if num_tokens(graph_block) > content_limit:
                    graph_chunks = TextProcessor.split_text_into_chunks(
                        text=graph_block, max_length=max(content_limit, 1)
                    )
                    segments.extend(
                        (chunk, None) for chunk in graph_chunks if chunk
                    )
                else:
                    segments.append((graph_block, None))

            for node in nodes:
                node_content = node.get("content", "")
                prefix = _evidence_text(node, "")
                suffix = "\n"
                node_text = _evidence_text(node, node_content)

                if num_tokens(node_text) <= content_limit:
                    segments.append((node_text, node))
                    continue

                # Keep the node header and split only content to fit the per-segment budget.
                content_budget = max(content_limit - num_tokens(prefix + suffix), 1)
                chunks = TextProcessor.split_text_into_chunks(
                    text=node_content, max_length=content_budget
                )
                for chunk in chunks:
                    chunk_text = _evidence_text(node, chunk)
                    if num_tokens(chunk_text) <= content_limit:
                        segments.append((chunk_text, node))

            merged = ""
            merged_tokens = 0
            included_nodes: List[Dict[str, Any]] = []
            for segment, source_node in segments:
                seg_tokens = num_tokens(segment)
                extra_tokens = seg_tokens if not merged else (separator_tokens + seg_tokens)
                if merged_tokens + extra_tokens > content_limit:
                    break
                if not merged:
                    merged = segment
                else:
                    merged += separator + segment
                merged_tokens += extra_tokens
                if source_node is not None:
                    included_nodes.append(source_node)

            return merged, included_nodes

        # Prefer one VLM call when there is image evidence.
        valid_image_nodes = [node for node in image_nodes if node.get("img_path")]
        if valid_image_nodes:
            selected_image = valid_image_nodes[0]
            img_path = selected_image.get("img_path")
            node_content = selected_image.get("content", "")
            image_meta = _evidence_text(selected_image, node_content)
            max_tokens = getattr(
                self.vlm.config, "max_tokens", self.llm.config.max_tokens
            )

            # Ensure image metadata itself does not overflow the VLM context window.
            image_meta_budget = (
                max_tokens
                - num_tokens(CITED_GENERATION_SYS_PROMPT)
                - num_tokens(
                    VLM_ONE_GENERATION_USER_PROMPT.format(
                        question=query, content="", retrieved_content=""
                    )
                )
                - 400
            )
            if image_meta_budget <= 0:
                image_meta = ""
            elif num_tokens(image_meta) > image_meta_budget:
                image_meta_chunks = TextProcessor.split_text_into_chunks(
                    text=image_meta, max_length=image_meta_budget
                )
                image_meta = image_meta_chunks[0] if image_meta_chunks else ""

            base_prompt_tokens = num_tokens(
                VLM_ONE_GENERATION_USER_PROMPT.format(
                    question=query, content=image_meta, retrieved_content=""
                )
            )
            system_prompt_tokens = num_tokens(CITED_GENERATION_SYS_PROMPT)
            content_limit = max_tokens - system_prompt_tokens - base_prompt_tokens - 400

            if graph_str:
                graph_prompt_part = ITER_GENERATION_GRAPH.format(
                    knowledge_graph_subgraph=graph_str
                )
            retrieved_content, included_text_nodes = _collect_retrieved_content(
                nodes=text_nodes,
                content_limit=content_limit,
                include_graph_block=True,
            )
            evidence_labels, citation_ids = _evidence_prompt_metadata(
                [selected_image, *included_text_nodes]
            )
            vlm_prompt = (
                f"{CITED_GENERATION_SYS_PROMPT.strip()}\n\n"
                f"{VLM_ONE_GENERATION_USER_PROMPT.format(question=query, content=image_meta, retrieved_content=retrieved_content).strip()}"
            )
            return [], [
                {
                    "prompt": vlm_prompt,
                    "image_url": img_path,
                    "evidence_id": selected_image["evidence_id"],
                    "evidence_label": selected_image["evidence_label"],
                    "citation_id": _node_citation_id(selected_image),
                    "evidence_labels": list(evidence_labels),
                    "citation_ids": list(citation_ids),
                }
            ]

        # Fallback to one LLM prompt (no image or invalid image paths).
        if graph_str:
            graph_overhead_tokens = num_tokens(
                ITER_GENERATION_USER_PROMPT.format(
                    user_question=query, retrieved_content=""
                )
            ) + num_tokens(ITER_GENERATION_GRAPH.format(knowledge_graph_subgraph=""))
            graph_budget = (
                self.llm.config.max_tokens
                - num_tokens(CITED_GENERATION_SYS_PROMPT)
                - graph_overhead_tokens
                - 400
            )
            if graph_budget <= 0:
                graph_prompt_part = ""
            elif num_tokens(graph_str) > graph_budget:
                graph_chunks = TextProcessor.split_text_into_chunks(
                    text=graph_str, max_length=graph_budget
                )
                if graph_chunks:
                    graph_prompt_part = ITER_GENERATION_GRAPH.format(
                        knowledge_graph_subgraph=graph_chunks[0]
                    )
            else:
                graph_prompt_part = ITER_GENERATION_GRAPH.format(
                    knowledge_graph_subgraph=graph_str
                )

        base_prompt_tokens = num_tokens(
            ITER_GENERATION_USER_PROMPT.format(
                user_question=query, retrieved_content=""
            )
            + graph_prompt_part
        )
        system_prompt_tokens = num_tokens(CITED_GENERATION_SYS_PROMPT)
        content_limit = (
            self.llm.config.max_tokens - system_prompt_tokens - base_prompt_tokens - 400
        )
        retrieved_content, included_text_nodes = _collect_retrieved_content(
            nodes=text_nodes,
            content_limit=content_limit,
            include_graph_block=False,
        )
        user_prompt = (
            ITER_GENERATION_USER_PROMPT.format(
                user_question=query, retrieved_content=retrieved_content
            )
            + graph_prompt_part
        )

        return [_evidence_prompt_memory(user_prompt, included_text_nodes)], []

    def _synthesize_from_chunks(
        self,
        query: str,
        text_prompts: List[Memory],
        image_prompts: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes generation on chunked prompts and synthesizes a final answer.
        """
        partial_answers = []

        # Generate answers from text prompts
        for i, memory in enumerate(text_prompts):
            labels = _prompt_evidence_labels(memory)
            citation_ids = _prompt_citation_ids(memory)
            source = (
                "Text evidence " + ", ".join(labels)
                if labels
                else f"Text Chunk {i + 1}"
            )
            try:
                answer = self.llm.get_completion(memory)
                partial_answers.append(
                    {
                        "source": source,
                        "evidence_labels": labels,
                        "citation_ids": citation_ids,
                        "content": answer,
                    }
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": source,
                        "evidence_labels": labels,
                        "citation_ids": citation_ids,
                        "content": f"[Error generating from text: {e}]",
                    }
                )

        # Generate answers from image prompts
        for i, item in enumerate(image_prompts):
            label = f"[{item['evidence_label']}]"
            labels = item.get("evidence_labels")
            if not isinstance(labels, list):
                labels = [label]
            citation_ids = item.get("citation_ids")
            if not isinstance(citation_ids, list):
                citation_id = item.get("citation_id")
                citation_ids = [citation_id] if citation_id else []
            source = (
                "Visual evidence " + ", ".join(labels)
                if labels
                else f"Visual Chunk {i + 1}"
            )
            try:
                answer = self.vlm.generate(item["prompt"], images=[item["image_url"]])
                partial_answers.append(
                    {
                        "source": source,
                        "evidence_labels": labels,
                        "citation_ids": citation_ids,
                        "content": answer,
                    }
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": source,
                        "evidence_labels": labels,
                        "citation_ids": citation_ids,
                        "content": f"[Error generating from image: {e}]",
                    }
                )

        # Synthesize the final answer
        if not partial_answers:
            final_answer = (
                "Based on the provided information, I couldn't find an answer."
            )
            return final_answer, partial_answers

        if len(partial_answers) == 1:
            final_answer = partial_answers[0]["content"]
            return final_answer, partial_answers

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("Synthesizing the final answer from partial results...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=CITED_SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"Error during final synthesis step: {e}")
            error_header = (
                "I was able to analyze the provided information in parts, but "
                "encountered an error while trying to synthesize the final answer. "
                f"Here are the partial analyses I found:\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return final_answer, partial_answers

    def _finalize_answer_citations(
        self,
        answer: str,
        partial_answers: List[Dict[str, Any]],
        evidence_nodes: List[Dict[str, Any]],
        *,
        require_citations: bool,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate that generated references belong to supplied evidence.

        This is a structural source-reference gate, not a semantic entailment
        claim. The latter remains a separate post-generation verifier.
        """

        validation = validate_answer_citations(
            answer,
            _evidence_bindings(evidence_nodes),
            required=require_citations,
        )
        validation_payload = validation.to_dict()
        if partial_answers:
            for partial in partial_answers:
                partial["answer_citation_validation"] = validation_payload
        else:
            partial_answers.append(
                {
                    "source": "Answer citation validation",
                    "content": "",
                    "answer_citation_validation": validation_payload,
                }
            )
        citations = _citation_index(evidence_nodes)
        if require_citations and not validation.valid:
            log.error(
                "Rejecting answer with invalid source references: %s",
                validation.status,
            )
            return CITATION_VALIDATION_FAILURE_ANSWER, partial_answers, citations
        return answer, partial_answers, citations

    def answer_simple_question(
        self,
        query: str,
        retrieved_nodes: List[Dict],
        entities: List[Dict] = None,
        relations: List[Dict] = None,
        max_graph_tokens: int = 2000,
        max_relations: int = 32,
        *,
        require_citations: bool = False,
        evidence_scope: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Orchestrates answering a single, simple question by preparing evidence,
        building prompts, and synthesizing results.
        """
        if self.evidence_hydrator is not None:
            hydrate = getattr(self.evidence_hydrator, "hydrate", None)
            if not callable(hydrate):
                raise TypeError("evidence_hydrator must provide a callable hydrate method")
            retrieved_nodes = hydrate(retrieved_nodes)

        # 1. Prepare evidence: Separate nodes and handle optional entities
        graph_lines = []
        graph_tokens = 0

        def add_graph_line(line: str) -> bool:
            nonlocal graph_tokens
            line_tokens = num_tokens(line + "\n")
            if graph_tokens + line_tokens > max_graph_tokens:
                return False
            graph_lines.append(line)
            graph_tokens += line_tokens
            return True

        if entities:
            add_graph_line(f"There are {len(entities)} relevant entities:")
            for ent in entities:
                description = str(ent.get("description") or "")[:600]
                line = (
                    f"- Name: {ent.get('entity_name', '')}, "
                    f"Type: {ent.get('entity_type', '')}"
                )
                if description:
                    line += f", Description: {description}"
                if not add_graph_line(line):
                    break
        if relations:
            add_graph_line(
                f"Showing up to {max_relations} of {len(relations)} evidence-backed relations:"
            )
            for relation in relations[:max_relations]:
                source = relation.get("src_entity_name") or relation.get("source_node")
                target = relation.get("tgt_entity_name") or relation.get("target_node")
                name = relation.get("relation_name") or "related to"
                description = str(relation.get("description") or "")[:600]
                line = f"- {source} --{name}--> {target}"
                if description:
                    line += f": {description}"
                if not add_graph_line(line):
                    break

        graph_str = "\n".join(graph_lines)

        text_nodes, image_nodes = self._prepare_evidence(
            retrieved_nodes,
            evidence_scope=evidence_scope,
        )

        # 2. Build chunked prompts for LLM and VLM
        if self.variant == "wo_map":
            # If the variant is "without map", we can only build one prompt
            text_prompts, image_prompts = self._build_single_prompt(
                query, text_nodes, image_nodes, graph_str
            )
        else:
            text_prompts, image_prompts = self._build_prompts(
                query, text_nodes, image_nodes, graph_str
            )

        # 3. Execute prompts and synthesize the final answer
        final_answer, partial_answers = self._synthesize_from_chunks(
            query, text_prompts, image_prompts
        )

        return self._finalize_answer_citations(
            final_answer,
            partial_answers,
            [*text_nodes, *image_nodes],
            require_citations=require_citations,
        )

    def answer_complex_question(
        self,
        original_query: str,
        sub_question_plan: List[SubQuestion],
        sub_question_results: List[Dict[str, Any]],
        *,
        require_citations: bool = False,
    ) -> str:
        """Compatibility wrapper returning only the synthesized answer string."""

        final_answer, _ = self.answer_complex_question_with_validation(
            original_query=original_query,
            sub_question_plan=sub_question_plan,
            sub_question_results=sub_question_results,
            require_citations=require_citations,
        )
        return final_answer

    def answer_complex_question_with_validation(
        self,
        original_query: str,
        sub_question_plan: List[SubQuestion],
        sub_question_results: List[Dict[str, Any]],
        *,
        require_citations: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Synthesize sub-answers and structurally validate final references.

        Each retrieval result receives a stable query-local namespace (``Q1``,
        ``Q2``, ...). Bare labels in compatibility inputs are rewritten into
        that namespace before synthesis. The validation proves only that final
        markers resolve to canonical citation IDs supplied by the sub-steps;
        it does not prove that cited evidence entails the answer's claims.
        """

        if not sub_question_results:
            answer = (
                "I could not find the necessary information to answer the "
                "complex question."
            )
            validation = validate_answer_citations(
                answer,
                {},
                required=require_citations,
            ).to_dict()
            validation.update(
                {
                    "validation_scope": "complex_final_synthesis",
                    "semantic_entailment_verified": False,
                    "subquestion_count": 0,
                }
            )
            if require_citations:
                return CITATION_VALIDATION_FAILURE_ANSWER, validation
            return answer, validation

        findings: list[str] = []
        combined_bindings: dict[str, str | None] = {}
        binding_errors: list[str] = []
        for position, result in enumerate(sub_question_results, start=1):
            scope = f"Q{position}"
            if not isinstance(result, Mapping):
                binding_errors.append(f"{scope}: sub-question result is not a mapping")
                continue

            scoped_bindings, errors = _scope_complex_bindings(result, scope)
            binding_errors.extend(errors)
            if require_citations and not scoped_bindings:
                binding_errors.append(
                    f"{scope}: no canonical evidence bindings were supplied"
                )
            for label, citation_id in scoped_bindings.items():
                if label in combined_bindings and combined_bindings[label] != citation_id:
                    binding_errors.append(
                        f"{scope}: evidence label {label!r} has conflicting citations"
                    )
                    continue
                combined_bindings[label] = citation_id

            upstream_validation = result.get("answer_validation")
            if (
                require_citations
                and isinstance(upstream_validation, Mapping)
                and upstream_validation.get("valid") is not True
            ):
                binding_errors.append(
                    f"{scope}: sub-question citation validation failed "
                    f"({upstream_validation.get('status', 'unknown')})"
                )

            raw_answer = result.get("answer", "")
            scoped_answer = _scope_answer_markers(str(raw_answer), scope)
            question = str(result.get("question", ""))
            binding_lines = [
                (
                    f"- [{label}] -> {citation_id}"
                    if citation_id
                    else f"- [{label}] -> UNBOUND (do not cite)"
                )
                for label, citation_id in sorted(
                    scoped_bindings.items(),
                    key=lambda item: _evidence_label_sort_key(f"[{item[0]}]"),
                )
            ]
            findings.append(
                "\n".join(
                    [
                        f"--- Finding {scope} for {question!r} ---",
                        "Allowed canonical evidence bindings:",
                        *(binding_lines or ["- none supplied"]),
                        "Sub-question answer:",
                        scoped_answer,
                    ]
                )
            )

        intermediate_findings = "\n\n".join(findings)
        if require_citations and binding_errors:
            validation = validate_answer_citations(
                "",
                combined_bindings,
                required=True,
            ).to_dict()
            validation.update(
                {
                    "status": "invalid_evidence_bindings",
                    "valid": False,
                    "message": (
                        "One or more sub-question evidence bindings or "
                        "citation-validation results were invalid."
                    ),
                    "validation_scope": "complex_final_synthesis",
                    "semantic_entailment_verified": False,
                    "subquestion_count": len(sub_question_results),
                    "binding_errors": binding_errors,
                }
            )
            log.error(
                "Rejecting complex synthesis before generation because its "
                "source bindings are invalid"
            )
            return CITATION_VALIDATION_FAILURE_ANSWER, validation

        synthesis_step = next(
            (sq for sq in sub_question_plan if sq.type == "synthesis"), None
        )
        final_task = (
            synthesis_step.question
            if synthesis_step
            else (
                "Provide a comprehensive final answer to the original complex "
                "question."
            )
        )
        synthesis_prompt = f"""
You are an expert AI assistant synthesizing findings for a complex question.
Use ONLY the gathered findings. {CITATION_INSTRUCTION}

For this final synthesis, every evidence label is scoped to its retrieval
sub-question (for example [Q1-E1] and [Q2-E1]). Never remove or change the Q
scope, and never cite a bare label such as [E1]. Citation IDs are immutable
trace identities; cite their associated bracketed evidence labels in the final
answer.

--- ORIGINAL COMPLEX QUESTION ---
{original_query}

--- GATHERED FINDINGS ---
{intermediate_findings}

--- FINAL TASK ---
{final_task}

Final Answer:
"""

        generation_failed = False
        try:
            final_answer = self.llm.get_completion(synthesis_prompt)
            if not isinstance(final_answer, str):
                final_answer = str(final_answer)
        except Exception as error:
            generation_failed = True
            log.error("Error during complex synthesis step: %s", error)
            final_answer = (
                "I encountered an error while trying to synthesize the final "
                "answer from the gathered information. Here are the intermediate "
                f"findings I was able to collect:\n\n{intermediate_findings}"
            )

        validation_result = validate_answer_citations(
            final_answer,
            combined_bindings,
            required=require_citations,
        )
        validation = validation_result.to_dict()
        validation.update(
            {
                "validation_scope": "complex_final_synthesis",
                "semantic_entailment_verified": False,
                "subquestion_count": len(sub_question_results),
            }
        )
        if binding_errors:
            validation["binding_errors"] = binding_errors
        if generation_failed:
            validation.update(
                {
                    "status": "generation_failed",
                    "valid": False,
                    "message": (
                        "Complex synthesis failed before a final answer could "
                        "be structurally accepted."
                    ),
                }
            )

        if require_citations and (
            generation_failed or not validation_result.valid
        ):
            log.error(
                "Rejecting complex answer with invalid source references: %s",
                validation["status"],
            )
            return CITATION_VALIDATION_FAILURE_ANSWER, validation
        return final_answer, validation

    def _get_partial_analysis_for_node(
        self, original_query: str, operation: str, node: Dict
    ) -> str:
        """
        Generates a partial analysis for a single node in the context of a global query,
        respecting the model's token limits.
        """
        # Determine the instruction based on the operation
        operation_instruction = {
            "LIST": "Extract and list the key information from the evidence below that is relevant to the original question.",
            "SUMMARIZE": "Summarize the key points from the evidence below as they relate to the original question.",
            "ANALYZE": "Analyze the evidence below and state its contribution to answering the original question.",
        }.get(operation, "Analyze the following piece of evidence.")

        base_prompt_template = f"""
You are working on a larger query and your current task is to analyze a single piece of retrieved evidence. Your analysis will be combined with others later to form a final answer.

--- ORIGINAL GLOBAL QUERY ---
{original_query}

--- CURRENT TASK ---
{operation_instruction}

--- EVIDENCE TO ANALYZE ---
{{node_context}}

Your concise analysis of this single piece of evidence:
"""
        # Treat retrieval output as authoritative provenance.  The page stored
        # on BookRAG nodes is zero-based and must never be rewritten merely to
        # produce a one-based display label for a prompt.
        prompt_node = node.copy()
        node_type = prompt_node.get("type", "text")
        # Determine which model to use and its context limit
        img_path = prompt_node.get("img_path", "")
        raw_content = prompt_node.get("content", "")
        content = "" if raw_content is None else str(raw_content)
        page = prompt_node.get("page")
        if isinstance(page, int):
            page += 1
            prompt_node["page"] = page
        elif page in (None, ""):
            page = -1
            prompt_node["page"] = page

        use_vlm = bool(
            node_type == NodeType.IMAGE
            and img_path
            and _visual_asset_is_verified(prompt_node)
        )
        if node_type == NodeType.IMAGE and img_path and not use_vlm:
            log.warning(
                "Withholding reference-only visual asset for global evidence %s",
                prompt_node.get("evidence_id", "unknown"),
            )
        model_max_tokens = (
            self.vlm.config.max_tokens if use_vlm else self.llm.config.max_tokens
        )

        # Calculate the available token budget for the node's content
        prompt_overhead = num_tokens(base_prompt_template.format(node_context=""))
        content_budget = max(
            1,
            model_max_tokens - prompt_overhead - 400,
        )  # 400 as a safety buffer

        # Prepare the node content, truncating if necessary
        if num_tokens(content) > content_budget:
            # Using your TextProcessor logic for consistency
            content = TextProcessor.split_text_into_chunks(
                text=content, max_length=content_budget
            )[0]

        node_context = _evidence_text(prompt_node, content)

        # Final prompt construction
        prompt = base_prompt_template.format(node_context=node_context)

        # Call the appropriate model
        if use_vlm:
            # VLM is now used ONLY for IMAGE nodes
            return self.vlm.generate(prompt, images=[img_path])
        else:
            # LLM is used for TEXT and TABLE nodes
            return self.llm.get_completion(prompt)

    def answer_global_question(
        self,
        original_query: str,
        operation: Literal["LIST", "SUMMARIZE", "ANALYZE"],
        filtered_nodes: List[Dict],
        *,
        require_citations: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Answers a global question by analyzing each filtered node individually
        and then synthesizing the results.
        """
        if not filtered_nodes:
            return (
                "I found no items matching the specified filters to answer the question.",
                [],
                [],
            )

        if self.evidence_hydrator is not None:
            hydrate = getattr(self.evidence_hydrator, "hydrate", None)
            if not callable(hydrate):
                raise TypeError("evidence_hydrator must provide a callable hydrate method")
            filtered_nodes = hydrate(filtered_nodes)

        # 1. Map Step: Get partial analysis for each node individually
        partial_answers = []
        evidence_nodes = []
        label_by_identity: dict[str, str] = {}
        for i, original_node in enumerate(filtered_nodes):
            node = original_node.copy()
            evidence_id = _evidence_identity(node, i + 1)
            evidence_label = label_by_identity.setdefault(
                evidence_id,
                f"E{len(label_by_identity) + 1}",
            )
            node["evidence_id"] = evidence_id
            node["evidence_label"] = evidence_label
            evidence_nodes.append(node)
            node_type = node.get("type", "text")
            display_node_type = (
                node_type.value if isinstance(node_type, NodeType) else node_type
            )
            page = node.get("page")
            display_page = page + 1 if isinstance(page, int) else page
            if display_page in (None, ""):
                display_page = -1
            source_str = (
                f"Evidence [{evidence_label}] "
                f"(Type: {display_node_type}, Page: {display_page})"
            )
            citation = node.get("citation")
            citation_id = (
                citation.get("citation_id") or node.get("citation_id")
                if isinstance(citation, Mapping)
                else node.get("citation_id")
            )
            try:
                analysis = self._get_partial_analysis_for_node(
                    original_query, operation, node
                )
                partial_answers.append(
                    {
                        "source": source_str,
                        "evidence_labels": [f"[{evidence_label}]"],
                        "citation_ids": [citation_id] if citation_id else [],
                        "content": analysis,
                    }
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": source_str,
                        "evidence_labels": [f"[{evidence_label}]"],
                        "citation_ids": [citation_id] if citation_id else [],
                        "content": f"[Error analyzing node: {e}]",
                    }
                )

        # 2. Reduce Step: Synthesize the partial analyses into a final answer
        if not partial_answers:
            return "No analysis could be generated from the filtered items.", []

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("Synthesizing the final answer from partial results...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=original_query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=CITED_SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"Error during final synthesis step: {e}")
            error_header = (
                "I was able to analyze the provided information in parts, but "
                "encountered an error while trying to synthesize the final answer. "
                f"Here are the partial analyses I found:\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return self._finalize_answer_citations(
            final_answer,
            partial_answers,
            evidence_nodes,
            require_citations=require_citations,
        )
