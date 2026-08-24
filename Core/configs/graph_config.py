from dataclasses import dataclass, field
import math
import os
from typing import Optional

from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.rerank_config import RerankerConfig


def graph_artifact_variant(config: "GraphConfig") -> str | None:
    """Return the graph filename variant used by both writers and readers."""

    if config.refine_type == "basic":
        return "basic"

    if config.reranker_config.model_name.lower().startswith("bge"):
        model = config.reranker_config.model_name.replace("/", "_").replace(" ", "_")
        return f"{config.g}_{model}"

    is_default_g = abs(config.g - 0.6) < 1e-9
    embedding_model = config.embedding_config.model_name
    if is_default_g and embedding_model.lower().startswith("qwen"):
        return None

    model = embedding_model.replace("/", "_").replace(" ", "_")
    return f"{config.g}_{model}"


def entity_attestation_directory(save_dir: str, config: "GraphConfig") -> str:
    """Return the run-local Qdrant attestation path for a graph profile."""

    if config.refine_type == "basic":
        directory = "entity_attestation_basic"
    else:
        embedding_model = config.embedding_config.model_name.replace("/", "_")
        reranker_model = config.reranker_config.model_name.replace("/", "_")
        directory = (
            f"entity_attestation_{config.g}_{embedding_model}_{reranker_model}"
        )
    return os.path.join(save_dir, directory)


@dataclass
class GraphConfig:
    # KG extraction
    extractor_type: str = "llm"  # Options: "llm", "local"
    local_model_name: str = "en_core_web_sm"
    local_device: str = "cpu"  # Options: "cpu", "cuda", "auto"
    image_description_force: bool = False
    max_gleaning: int = 0
    extraction_max_workers: int = 4
    extraction_max_in_flight: Optional[int] = None
    extraction_batch_size: int = 32
    refinement_max_workers: int = 8
    refinement_max_in_flight: Optional[int] = None

    # KG refinement
    refine_type: str = "advanced"  # Options: "basic", "advanced"
    g: float = 0.6  # For advanced refinement, the gradient similarity threshold
    # The hybrid schema stores named dense vectors plus a sparse one, which
    # Qdrant cannot convert an unnamed collection into. The name carries the
    # schema so a pre-hybrid collection is never opened as if it were current,
    # and so migrating does not mean dropping other documents' entities out of
    # a collection they share.
    entity_collection_name: str = "bookrag-entities-hybrid-v1"

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)

    # Cross-document linking (grouping nodes + cross-docs relations).
    # Candidate pairs land in [cross_doc_relatedness_floor, cross_doc_merge_threshold):
    # above the upper bound is kg_refiner's own same-entity territory, below the
    # floor is unrelated. Neither bound is coupled to kg_refiner's own threshold --
    # it never runs cross-document, so there is no existing value to inherit.
    cross_doc_merge_threshold: float = 0.80
    cross_doc_relatedness_floor: float = 0.55
    # Off by default: linking must be opted into, either per-ingest (auto) or
    # by explicitly selecting documents to link (direct/manual) -- it must
    # never be an unconditional side effect of ingestion.
    cross_doc_link_auto_trigger: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.entity_collection_name, str)
            or not self.entity_collection_name.strip()
            or self.entity_collection_name != self.entity_collection_name.strip()
        ):
            raise ValueError("entity_collection_name must be a non-empty trimmed string")
        if self.extractor_type not in {"llm", "local"}:
            raise ValueError("extractor_type must be 'llm' or 'local'")
        if self.local_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("local_device must be 'cpu', 'cuda', or 'auto'")
        if self.refine_type not in {"basic", "advanced"}:
            raise ValueError("refine_type must be 'basic' or 'advanced'")
        if isinstance(self.g, bool):
            raise ValueError("g must be a finite number in the interval (0, 1]")
        try:
            valid_g = math.isfinite(self.g) and 0 < self.g <= 1
        except TypeError as error:
            raise ValueError(
                "g must be a finite number in the interval (0, 1]"
            ) from error
        if not valid_g:
            raise ValueError("g must be a finite number in the interval (0, 1]")
        if self.extraction_max_workers < 1:
            raise ValueError("extraction_max_workers must be at least 1")
        if self.extraction_max_in_flight is None:
            self.extraction_max_in_flight = self.extraction_max_workers
        if self.extraction_max_in_flight < 1:
            raise ValueError("extraction_max_in_flight must be at least 1")
        if self.extraction_batch_size < 1:
            raise ValueError("extraction_batch_size must be at least 1")
        if self.refinement_max_workers < 1:
            raise ValueError("refinement_max_workers must be at least 1")
        if self.refinement_max_in_flight is None:
            self.refinement_max_in_flight = self.refinement_max_workers
        if self.refinement_max_in_flight < 1:
            raise ValueError("refinement_max_in_flight must be at least 1")
        for bound_name in ("cross_doc_merge_threshold", "cross_doc_relatedness_floor"):
            bound = getattr(self, bound_name)
            if isinstance(bound, bool) or not (0.0 <= bound <= 1.0):
                raise ValueError(f"{bound_name} must be a number in [0, 1]")
        if self.cross_doc_relatedness_floor >= self.cross_doc_merge_threshold:
            raise ValueError(
                "cross_doc_relatedness_floor must be strictly less than "
                "cross_doc_merge_threshold"
            )
