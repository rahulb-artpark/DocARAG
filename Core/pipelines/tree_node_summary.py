from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional

from Core.Index.Tree import TreeNode, NodeType, DocumentTree
from Core.pipelines.node_context import nearest_prose, parent_title
from Core.prompts.summary_prompt import NODE_SUMMARY_PROMPT, SEC_SUMMARY_PROMPT
from Core.provider.llm import CompletionRequest, CompletionResult, LLM
from Core.provider.vlm import GenerationRequest, GenerationResult, VLM
from Core.utils.summary_checkpoint import (
    SummaryCheckpointStore,
    summary_cache_identity,
)
from Core.utils.utils import num_tokens, TextProcessor
import os
import logging

log = logging.getLogger(__name__)


class TreeSummaryGenerationError(RuntimeError):
    """A node summary could not be generated safely."""

    def __init__(self, node_id: int, provider: str, error: Exception):
        self.node_id = node_id
        self.provider = provider
        self.error = error
        super().__init__(
            f"{provider} summary generation failed for node {node_id}: {error}"
        )


@dataclass(frozen=True, slots=True)
class _PendingSummary:
    node: TreeNode
    cache_key: str
    prompt_sha256: str
    image_identities: tuple[dict[str, str], ...]
    kind: str


def _summary_model_identity(config: object) -> dict[str, object]:
    """Return output-affecting, secret-free provider identity for checkpoints."""

    identity: dict[str, object] = {}
    for field in (
        "backend",
        "model_name",
        "api_base",
        "max_tokens",
        "temperature",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(config, field, None)
        if value is not None:
            identity[field] = value
    if not identity:
        raise ValueError("summary provider has no cacheable model identity")
    return identity


#: Bounds one neighbour so a long sibling cannot crowd out the node itself.
_NEIGHBOUR_CHARS = 600


def _with_surroundings(tree_node: TreeNode, body: str, *, neighbours: bool) -> str:
    """Place a leaf in its section, and optionally between its sentences.

    ``NODE_SUMMARY_PROMPT`` already instructs the model to use a title when one
    is given; until now none ever was for these types.
    """

    before, after = nearest_prose(tree_node) if neighbours else ("", "")
    title = parent_title(tree_node)
    parts = [
        f"Title: {title}" if title else "",
        f"Preceding text: {before[:_NEIGHBOUR_CHARS]}" if before else "",
        body,
        f"Following text: {after[:_NEIGHBOUR_CHARS]}" if after else "",
    ]
    return "\n".join(part for part in parts if part)


# A node whose type is absent here gets no summary at all -- and a node with
# no summary is close to unreachable, because the summary is what the tree
# index embeds. Every type the projection can produce must appear.
_SELF_DESCRIPTION = {
    NodeType.IMAGE: "This is an image.",
    NodeType.TABLE: "This is a table.",
    NodeType.LIST: (
        "This is a list. Say what the items have in common and what they "
        "enumerate, not each item in turn."
    ),
    NodeType.CHART: "This is a chart. Report the quantities it plots.",
    NodeType.DIAGRAM: "This is a diagram. Report what it shows and how its parts relate.",
    NodeType.CODE: "This is a code listing. Say what the code does, not how it is written.",
    NodeType.ALGORITHM: "This is an algorithm. Say what it computes and by what strategy.",
}
# Types summarised from a caption plus their own body, with soft context: they
# already state their own subject, so they need only the heading above them.
_CAPTIONED_NODE_TYPES = frozenset(
    {NodeType.IMAGE, NodeType.TABLE, NodeType.CHART, NodeType.DIAGRAM}
)
# Types summarised as prose, exactly as body text is.
_PROSE_NODE_TYPES = frozenset(
    {NodeType.TEXT, NodeType.ABSTRACT, NodeType.REFERENCE}
)
_SUMMARISABLE_NODE_TYPES = (
    _CAPTIONED_NODE_TYPES
    | _PROSE_NODE_TYPES
    | {NodeType.EQUATION, NodeType.CODE, NodeType.ALGORITHM, NodeType.LIST}
)


def get_node_summary_prompt(tree_node: TreeNode, max_token: int) -> str:
    node_type = tree_node.type
    if node_type not in _SUMMARISABLE_NODE_TYPES:
        log.warning(f"Node type {node_type} is not supported for summary generation.")
        return ""

    if node_type in _CAPTIONED_NODE_TYPES:
        content = _SELF_DESCRIPTION[node_type]
        content += "Here is the caption: "
        content += tree_node.meta_info.content or ""
        content += (
            f"\n{tree_node.meta_info.table_body}" if node_type == NodeType.TABLE else ""
        )
        # Soft context: a figure already states its own subject in its caption,
        # so it needs only the heading it sits under to place that subject.
        content = _with_surroundings(tree_node, content, neighbours=False)
    elif node_type == NodeType.EQUATION:
        # A displayed equation states nothing in words. Summarised from its
        # symbols alone the model can only describe the mathematics -- "a
        # diagonal band of width 1/4" for the set that decides whether Romeo
        # and Juliet meet. Given the surroundings it writes what the equation
        # is *for*, which is what a question asks about.
        content = _with_surroundings(
            tree_node, tree_node.meta_info.content or "", neighbours=True
        )
    elif node_type is NodeType.LIST:
        # A list's items are fragments that share a subject stated once, in
        # the stem above them: "1. Cornea" carries the eye, "Acts as a primary
        # lens" does not. Read without that stem the items summarise into
        # unattached predicates, so a list gets its surroundings like an
        # equation rather than a caption.
        content = _with_surroundings(
            tree_node,
            f"{_SELF_DESCRIPTION[node_type]}\n{tree_node.meta_info.content or ''}",
            neighbours=True,
        )
    elif node_type in (NodeType.CODE, NodeType.ALGORITHM):
        # A listing names its own identifiers but rarely its purpose, and the
        # sentence that says what it is for is the prose introducing it -- the
        # same problem a displayed equation has, so it gets the same treatment.
        content = _with_surroundings(
            tree_node,
            f"{_SELF_DESCRIPTION[node_type]}\n{tree_node.meta_info.content or ''}",
            neighbours=True,
        )
    else:
        content = tree_node.meta_info.content or ""

    # 2. 【新增】检查并截断内容
    base_prompt_tokens = num_tokens(NODE_SUMMARY_PROMPT.format(node_text=""))
    available_tokens = max_token - base_prompt_tokens
    if num_tokens(content) > available_tokens:
        log.warning(
            f"Content length ({num_tokens(content)} tokens) exceeds max_token ({max_token}). Truncating."
        )
        # 调用静态方法进行切分
        chunks = TextProcessor.split_text_into_chunks(text=content, max_length=max_token)
        # 只取第一个分片
        content = chunks[0] if chunks else ""

    prompt = NODE_SUMMARY_PROMPT.format(node_text=content)
    return prompt


def generate_node_summary(
    tree_node: TreeNode, llm: LLM, use_VLM: bool = False, vlm: Optional[VLM] = None
) -> str:
    """Generate a summary for a single tree node.
    This function uses the LLM to generate a summary based on the node's content.
    If the node is an image or table, it will use the VLM if provided.
    """
    node_type = tree_node.type
    prompt = get_node_summary_prompt(tree_node, max_token=llm.config.max_tokens)

    if use_VLM and vlm is not None and node_type in _CAPTIONED_NODE_TYPES:
        # Use VLM for image or table nodes
        image_path = tree_node.meta_info.img_path
        if not os.path.exists(image_path):
            log.warning(
                f"Image path {image_path} does not exist for node {tree_node.index_id}."
            )
            return ""
        summary = vlm.generate(prompt_or_memory=prompt, images=[image_path])
    else:
        summary = llm.get_completion(prompt=prompt, json_response=False)

    if not summary:
        log.warning(f"Failed to generate summary for node {tree_node.index_id}.")
        return ""
    log.info(f"Generated summary for node {tree_node.index_id}: {summary}")
    return summary.strip()


def get_sec_summary_prompt(sec_node: TreeNode, max_token: int) -> str:
    """Get the prompt for generating a section summary.
    This function formats the section text and its immediate children's summaries into a prompt.
    """
    base_prompt_tokens = num_tokens(
        SEC_SUMMARY_PROMPT.format(section_text="", content_summary="")
    )
    available_tokens = max_token - base_prompt_tokens

    if available_tokens <= 0:
        log.warning(
            f"max_token ({max_token}) is too small to even fit the prompt template."
        )
        return ""

    # Get initial content
    section_text = sec_node.meta_info.content or ""

    def get_children_text(node: TreeNode) -> str:
        """Recursively get text from all immediate children nodes."""
        if not node.children:
            return ""
        children_text = []
        child_prefix = {
            NodeType.TEXT: "Text: ",
            NodeType.IMAGE: "Image: ",
            NodeType.TABLE: "Table: ",
            NodeType.EQUATION: "Equation: ",
        }
        for child in node.children:
            child_text = child.summary or child.meta_info.content or ""
            child_type = child.type
            if child_text:
                children_text.append(f"{child_prefix.get(child_type, '')}{child_text}")
        return "\n".join(children_text)

    children_text = get_children_text(sec_node)

    # Truncation Logic
    section_tokens = num_tokens(section_text)

    if section_tokens >= available_tokens:
        # If section_text alone is too long, truncate it and use no children_text
        log.warning(
            f"Section text ({section_tokens} tokens) exceeds available tokens ({available_tokens}). Truncating section text and omitting children summaries."
        )
        chunks = TextProcessor.split_text_into_chunks(
            text=section_text, max_length=available_tokens
        )
        section_text = chunks[0] if chunks else ""
        children_text = ""
    else:
        # If section_text fits, see how much of children_text can be included
        remaining_tokens = available_tokens - section_tokens
        children_tokens = num_tokens(children_text)
        if children_tokens > remaining_tokens:
            log.warning(
                f"Children summaries ({children_tokens} tokens) exceed remaining tokens ({remaining_tokens}). Truncating children summaries."
            )
            chunks = TextProcessor.split_text_into_chunks(
                text=children_text, max_length=remaining_tokens
            )
            children_text = chunks[0] if chunks else ""

    return SEC_SUMMARY_PROMPT.format(
        section_text=section_text, content_summary=children_text
    )


def generate_section_summary(sec_node: TreeNode, llm: LLM) -> str:
    """Generate a summary for a section node.
    This function uses the LLM to generate a summary based on the section's content and its children.
    It includes text from the section itself and summaries of its immediate children.
    """
    prompt = get_sec_summary_prompt(sec_node, llm.config.max_tokens)
    summary = llm.get_completion(prompt=prompt, json_response=False)
    return summary


def generate_tree_node_summary(
    tree_index: DocumentTree,
    llm: LLM,
    use_VLM: bool = False,
    vlm: Optional[VLM] = None,
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> DocumentTree:
    """Generate summaries bottom-up with bounded model request streams.

    Levels remain sequential because a section prompt consumes its children's
    summaries. Requests within one level execute concurrently, but only the
    provider's configured ``max_in_flight`` prompts and futures exist at once.
    Every successful item is committed to a content/model/image-keyed SQLite
    checkpoint before it becomes authoritative in the tree. Any item failure
    aborts the stage; exception text is never assigned to document data.
    """
    log.info("Generating summaries for tree nodes...")

    if tree_index.root_node is None:
        raise ValueError("Cannot summarize a tree without a root node")
    if use_VLM and vlm is None:
        raise ValueError("use_VLM=True requires a VLM instance")

    # Iterative traversal avoids Python recursion limits on deeply nested input.
    level_dict: dict[int, list[TreeNode]] = {}
    stack = [(tree_index.root_node, 0)]
    while stack:
        node, level = stack.pop()
        level_dict.setdefault(level, []).append(node)
        stack.extend((child, level + 1) for child in reversed(node.children))

    log.info(f"Processing tree with {len(level_dict)} levels")

    if checkpoint_path is None:
        if not tree_index.save_dir:
            raise ValueError("tree save directory is required for summary checkpoints")
        checkpoint_path = Path(tree_index.save_dir) / "summary-checkpoints.sqlite3"
    checkpoint_store = SummaryCheckpointStore(checkpoint_path)
    llm_identity = _summary_model_identity(llm.config)
    vlm_identity = _summary_model_identity(vlm.config) if vlm is not None else None

    def prompt_for(node: TreeNode) -> tuple[str, str]:
        if node.children or node.type == NodeType.TITLE:
            prompt = get_sec_summary_prompt(node, llm.config.max_tokens)
            kind = "section-llm"
        else:
            prompt = get_node_summary_prompt(node, llm.config.max_tokens)
            kind = "node-llm"
        if not prompt:
            raise TreeSummaryGenerationError(
                node.index_id,
                "prompt",
                ValueError("summary prompt is empty or node type is unsupported"),
            )
        return prompt, kind

    def uses_vlm(node: TreeNode) -> bool:
        return bool(
            use_VLM
            and not node.children
            and node.type in _CAPTIONED_NODE_TYPES
        )

    def pending_summary(
        node: TreeNode,
        *,
        prompt: str,
        kind: str,
        model: Mapping[str, object],
        images: tuple[str, ...] = (),
    ) -> _PendingSummary | None:
        cache_key, prompt_sha256, image_identities = summary_cache_identity(
            kind=kind,
            prompt=prompt,
            model=model,
            image_paths=images,
        )
        cached = checkpoint_store.get(cache_key)
        if cached is not None:
            node.summary = cached
            log.info("Reused checkpointed summary for node %s", node.index_id)
            return None
        return _PendingSummary(
            node=node,
            cache_key=cache_key,
            prompt_sha256=prompt_sha256,
            image_identities=image_identities,
            kind=kind,
        )

    def llm_requests(
        nodes: list[TreeNode],
    ) -> Iterator[CompletionRequest[_PendingSummary]]:
        for node in nodes:
            if node is tree_index.root_node or uses_vlm(node):
                continue
            prompt, kind = prompt_for(node)
            pending = pending_summary(
                node,
                prompt=prompt,
                kind=kind,
                model=llm_identity,
            )
            if pending is not None:
                yield CompletionRequest(key=pending, prompt=prompt)

    def vlm_requests(
        nodes: list[TreeNode],
    ) -> Iterator[GenerationRequest[_PendingSummary]]:
        for node in nodes:
            if node is tree_index.root_node or not uses_vlm(node):
                continue
            image_path = node.meta_info.img_path
            if not image_path or not os.path.isfile(image_path):
                error = FileNotFoundError(
                    f"image artifact does not exist: {image_path!r}"
                )
                raise TreeSummaryGenerationError(node.index_id, "VLM", error) from error
            prompt, _kind = prompt_for(node)
            assert vlm_identity is not None
            pending = pending_summary(
                node,
                prompt=prompt,
                kind="node-vlm",
                model=vlm_identity,
                images=(image_path,),
            )
            if pending is not None:
                yield GenerationRequest(
                    key=pending,
                    prompt_or_memory=prompt,
                    images=[image_path],
                )

    def apply_llm_result(result: CompletionResult[_PendingSummary]) -> None:
        pending = result.key
        if not result.ok:
            assert result.error is not None
            raise TreeSummaryGenerationError(
                pending.node.index_id, "LLM", result.error
            ) from result.error
        summary = result.unwrap().strip()
        if not summary:
            error = ValueError("model returned an empty summary")
            raise TreeSummaryGenerationError(
                pending.node.index_id, "LLM", error
            ) from error
        pending.node.summary = checkpoint_store.put(
            cache_key=pending.cache_key,
            kind=pending.kind,
            prompt_sha256=pending.prompt_sha256,
            model=llm_identity,
            image_identities=pending.image_identities,
            summary=summary,
        )
        log.info(
            "Generated LLM summary for node %s (%d characters)",
            pending.node.index_id,
            len(summary),
        )

    def apply_vlm_result(result: GenerationResult[_PendingSummary]) -> None:
        pending = result.key
        if not result.ok:
            assert result.error is not None
            raise TreeSummaryGenerationError(
                pending.node.index_id, "VLM", result.error
            ) from result.error
        summary = result.unwrap().strip()
        if not summary:
            error = ValueError("model returned an empty summary")
            raise TreeSummaryGenerationError(
                pending.node.index_id, "VLM", error
            ) from error
        assert vlm_identity is not None
        pending.node.summary = checkpoint_store.put(
            cache_key=pending.cache_key,
            kind=pending.kind,
            prompt_sha256=pending.prompt_sha256,
            model=vlm_identity,
            image_identities=pending.image_identities,
            summary=summary,
        )
        log.info(
            "Generated VLM summary for node %s (%d characters)",
            pending.node.index_id,
            len(summary),
        )

    try:
        for level in sorted(level_dict.keys(), reverse=True):
            nodes = level_dict[level]
            log.info("Processing level %s with %s nodes", level, len(nodes))

            for result in llm.iter_completion_results(
                llm_requests(nodes), json_response=False
            ):
                apply_llm_result(result)

            if use_VLM:
                assert vlm is not None
                for result in vlm.iter_generation_results(vlm_requests(nodes)):
                    apply_vlm_result(result)
    finally:
        checkpoint_store.close()

    log.info("All node summaries generated successfully.")
    return tree_index


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    tmp_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    llm = LLM(llm_config=cfg.llm)

    tree_index = generate_tree_node_summary(tree_index=tree_index, llm=llm)
    one_step_index_1 = tree_index.get_one_depth_summary(1)
    print(f"Node ID: 1, Summary: \n")
    print(one_step_index_1)
