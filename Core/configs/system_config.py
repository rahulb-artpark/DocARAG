import yaml
from collections.abc import Mapping
from Core.configs.mineru_config import MinerU
from Core.configs.figure_config import FigureConfig
from Core.configs.llm_config import LLMConfig
from Core.configs.tree_config import TreeConfig
from Core.configs.graph_config import GraphConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag_config import RAGConfig
from Core.configs.vdb_config import VDBConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pathlib import Path
from typing import Optional


class SourceIdentity(BaseModel):
    """Immutable input identity separated from the parser's snapshot path."""

    model_config = ConfigDict(extra="forbid")

    display_name: str
    original_uri: str
    sha256: Optional[str] = None
    snapshot_path: Optional[str] = None
    document_id: Optional[str] = None
    document_version_id: Optional[str] = None
    index_revision_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_document_identity(self):
        supplied = (self.document_id, self.document_version_id)
        if any(value is not None for value in supplied):
            if any(value is None for value in supplied) or self.sha256 is None:
                raise ValueError(
                    "source document_id and document_version_id require each other "
                    "and source.sha256"
                )
            from docarag.domain.evidence import DocumentRef

            expected = DocumentRef.identified(
                content_sha256=self.sha256,
                mime_type="application/pdf",
                source_uri=self.original_uri,
                display_name=self.display_name,
            )
            if (
                self.document_id != expected.document_id
                or self.document_version_id != expected.document_version_id
            ):
                raise ValueError(
                    "source document IDs do not match its URI and content SHA-256"
                )
        for field_name in (
            "document_id",
            "document_version_id",
            "index_revision_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (not value or value != value.strip()):
                raise ValueError(f"source.{field_name} must be a trimmed identifier")
        return self


class SystemConfig(BaseModel):
    """
    Top-level application configuration model.
    Pydantic will automatically handle nested validation and instantiation.
    """

    model_config = ConfigDict(extra="forbid")

    # LLM Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    mineru: MinerU = Field(default_factory=MinerU)
    figures: FigureConfig = Field(default_factory=FigureConfig)

    # Index Configurations
    tree: TreeConfig = Field(default_factory=TreeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vdb: VDBConfig = Field(default_factory=VDBConfig)

    # Other Index selection
    index_type: Optional[str] = "gbc"  # Options: "gbc", "tree", "vanilla", "bm25", "raptor", "pdf_vanilla"

    rag_force_reprocess: Optional[bool] = False

    # RAG Configurations
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Paths
    pdf_path: Optional[str] = None
    save_path: Optional[str] = None
    source: Optional[SourceIdentity] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_shape(cls, value):
        """Normalize the one supported legacy shape before strict validation.

        Historical hand-written configs stored the selected RAG strategy directly
        under ``rag``.  Run snapshots use the canonical ``rag.strategy_config``
        envelope.  Accept both without ever wrapping the canonical form twice.
        """

        if not isinstance(value, Mapping):
            return value

        normalized = dict(value)
        if "index" in normalized:
            raise ValueError(
                "Unsupported root field 'index': index.chunk_size and "
                "index.overlap have no runtime consumer. Remove this block; "
                "select the index implementation with 'index_type'."
            )

        rag = normalized.get("rag")
        if not isinstance(rag, Mapping):
            return normalized

        has_envelope = "strategy_config" in rag
        has_legacy_strategy = "strategy" in rag
        if has_envelope and has_legacy_strategy:
            raise ValueError(
                "Ambiguous 'rag' configuration: use either legacy "
                "'rag.strategy' fields or canonical 'rag.strategy_config', not both."
            )
        if not has_envelope and has_legacy_strategy:
            normalized["rag"] = {"strategy_config": dict(rag)}
        return normalized


def load_system_config(path: str | Path = "config/default.yaml") -> SystemConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError(f"System configuration must be a YAML mapping: {path}")

    return SystemConfig.model_validate(raw_config)
