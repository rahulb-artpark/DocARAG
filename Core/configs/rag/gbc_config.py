from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import AliasChoices, Field, model_validator

from Core.configs.rerank_config import RerankerConfig
from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.gbc_limits import (
    ABSOLUTE_MAX_FILTERS,
    ABSOLUTE_MAX_FILTER_VALUE_CHARS,
    ABSOLUTE_MAX_QUERY_CHARS,
    ABSOLUTE_MAX_SUBQUESTION_CHARS,
    ABSOLUTE_MAX_SUBQUESTIONS,
)


class GBCRAGConfig(BaseRAGStrategyConfig):
    """
    Configuration class for the GBC RAG (Graph-Based Contextual Retrieval Augmented Generation).
    This class defines the parameters required for initializing the GBC RAG agent.
    """

    strategy: Literal["gbc"] = "gbc"
    variant: Literal[
        "standard", "wo_plan", "wo_selector", "wo_graph", "wo_text", "wo_er", "wo_map"
    ] = Field(
        default="standard",
        description="The variant of the GBC RAG strategy to use. Options are 'standard', 'wo_plan', 'wo_selector', 'wo_graph', 'wo_text', 'wo_er', and 'wo_map'.",
    )
    topk: int = Field(
        default=10,
        ge=1,
        description="The number of top results to return from the graph-based retrieval.",
    )
    topk_relative_floor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "After reranking, drop candidates scoring below this fraction of "
            "the top match rather than padding topk with low-relevance nodes. "
            "topk stays the ceiling; this only ever shrinks the selection."
        ),
    )
    topk_min: int = Field(
        default=3,
        ge=1,
        description=(
            "Always keep at least this many top-ranked candidates even if "
            "topk_relative_floor would cut deeper, so a query with uniformly "
            "modest scores still gets an answer."
        ),
    )
    candidate_pool_size: int = Field(
        default=512,
        ge=1,
        le=4096,
        description=(
            "Maximum number of tree nodes sent to the text reranker for one "
            "sub-query. The bound controls reranker memory and latency."
        ),
    )
    global_evidence_limit: int = Field(
        default=512,
        ge=1,
        le=4096,
        description=(
            "Maximum evidence nodes that a non-COUNT global query may map in "
            "one synchronous request. This bounds per-node work -- hydration, "
            "store reads -- rather than the answer model's context, which "
            "global_evidence_char_budget bounds. Larger scopes fail explicitly "
            "instead of producing a silently sampled or unbounded answer."
        ),
    )
    global_evidence_char_budget: int = Field(
        default=48_000,
        ge=1_000,
        le=4_000_000,
        description=(
            "Ceiling on the characters of evidence text a non-COUNT global "
            "query may send in one synchronous request, at roughly four "
            "characters per token. This is only a ceiling: the effective budget "
            "is derived from the answer model's own num_ctx minus the tokens "
            "reserved for the answer, and the lower of the two wins. It applies "
            "on its own only when the backend declares no context window."
        ),
    )
    enable_global_classification: bool = Field(
        default=False,
        description=(
            "Allow the query planner to classify queries as global. When false, "
            "classification is restricted to simple and complex."
        ),
    )
    planner_max_query_chars: int = Field(
        default=4096,
        ge=1,
        le=ABSOLUTE_MAX_QUERY_CHARS,
        description=(
            "Maximum characters accepted in one BookRAG question. The limit "
            "is checked before any planner, vector-store, or answer-model call."
        ),
    )
    planner_max_subquestions: int = Field(
        default=16,
        ge=1,
        le=ABSOLUTE_MAX_SUBQUESTIONS,
        description=(
            "Maximum model-produced sub-questions in a complex query plan. "
            "Oversized plans are rejected rather than partially executed."
        ),
    )
    complex_synthesis_input_tokens: int = Field(
        default=12000,
        ge=1024,
        le=128000,
        description="Input token ceiling for complex synthesis and compaction; further limited by num_ctx minus output tokens and safety margin.",
    )
    planner_max_subquestion_chars: int = Field(
        default=1024,
        ge=1,
        le=ABSOLUTE_MAX_SUBQUESTION_CHARS,
        description="Maximum characters in each model-produced sub-question.",
    )
    planner_max_filters: int = Field(
        default=8,
        ge=1,
        le=ABSOLUTE_MAX_FILTERS,
        description=(
            "Maximum model-produced structural filters in a global plan."
        ),
    )
    planner_max_filter_value_chars: int = Field(
        default=256,
        ge=1,
        le=ABSOLUTE_MAX_FILTER_VALUE_CHARS,
        description="Maximum characters in one model-produced filter value.",
    )
    query_entity_limit: int = Field(
        default=16,
        ge=1,
        le=32,
        description=(
            "Maximum entities accepted from query entity extraction. Each "
            "remaining entity can cause one bounded entity-vector lookup."
        ),
    )
    query_entity_name_chars: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum characters accepted in one extracted entity name.",
    )
    query_entity_type_chars: int = Field(
        default=64,
        ge=1,
        le=128,
        description="Maximum characters accepted in one extracted entity type.",
    )
    query_entity_candidate_limit: int = Field(
        default=5,
        ge=1,
        le=32,
        description=(
            "Maximum entity-vector candidates supplied to the extraction "
            "model for one query."
        ),
    )
    section_selection_candidate_limit: int = Field(
        default=48,
        ge=1,
        le=256,
        description=(
            "Maximum linked plus supplementary section descriptors placed in "
            "one selection prompt. Linked sections are never silently dropped."
        ),
    )
    section_selection_supplementary_limit: int = Field(
        default=8,
        ge=1,
        le=32,
        description=(
            "Maximum supplementary section IDs accepted from the selection model."
        ),
    )
    section_selection_title_chars: int = Field(
        default=256,
        ge=16,
        le=2048,
        description="Maximum source-title characters represented per prompt field.",
    )
    section_selection_path_depth: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum structural title path depth represented per section.",
    )
    section_selection_prompt_chars: int = Field(
        default=65536,
        ge=4096,
        le=262144,
        description=(
            "Hard character ceiling for one rendered section-selection prompt."
        ),
    )
    sim_threshold_e: float = Field(
        default=0.3,
        validation_alias=AliasChoices("sim_threshold_e", "sim_threshold"),
        description="The similarity threshold for filtering retrieved results.",
    )
    select_depth: int = Field(
        default=2,
        description="The tree depth of section for LLM selection.",
    )
    x_percentile: float = Field(
        default=0.85,
        description="The percentile for selecting the top x% of edge similarity, used in Graph Augmentation.",
    )
    alpha: float = Field(
        default=0.5,
        description="PPR parameter.",
    )
    topk_ent: int = Field(
        default=5,
        description="The number of top entities to retrieve from the graph.",
    )
    max_documents_per_query: int = Field(
        default=4,
        ge=1,
        description=(
            "Corpus-scoped queries only: caps how many documents one query "
            "fans out to, independent of the entity-level topk_ent budget."
        ),
    )
    cross_doc_edge_discount_factor: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplier applied to a cross-docs edge's weight at graph-"
            "construction time, keyed on link_origin='cross_document_inferred'. "
            "Intra-document edge weights are never discounted."
        ),
    )

    max_retry: int = Field(
        default=3,
        description="The maximum number of retries for the LLM to generate a valid response.",
    )
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig,
    )
    mm_reranker_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
    )

    @model_validator(mode="after")
    def validate_retrieval_bounds(self):
        if self.candidate_pool_size < self.topk:
            raise ValueError("candidate_pool_size must be greater than or equal to topk")
        if self.query_entity_candidate_limit > self.query_entity_limit:
            raise ValueError(
                "query_entity_candidate_limit must be less than or equal to "
                "query_entity_limit"
            )
        if (
            self.section_selection_supplementary_limit
            > self.section_selection_candidate_limit
        ):
            raise ValueError(
                "section_selection_supplementary_limit must be less than or "
                "equal to section_selection_candidate_limit"
            )
        return self
