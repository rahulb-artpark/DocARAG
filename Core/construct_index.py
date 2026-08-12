import os
import logging
import time
from pathlib import Path
import pandas as pd

from Core.configs.vdb_config import VDBConfig

log = logging.getLogger(__name__)

from Core.Index.GBCIndex import GBC
from Core.Index.Graph import Graph
from Core.Index.Tree import DocumentTree
from Core.configs.graph_config import graph_artifact_variant
from Core.configs.system_config import SystemConfig
from Core.pipelines.doc_tree_builder import (
    CanonicalIngestionProvider,
    build_tree_from_pdf,
)
from Core.pipelines.kg_builder import build_knowledge_graph
from Core.pipelines.vdb_index import (
    build_other_vdb_index,
    build_vdb_index,
    compute_mm_embedding,
    compute_mm_embedding_question,
)
from Core.provider.TokenTracker import (
    TokenTracker,
    TokenUsageScope,
    current_token_scope,
)
from Core.utils.file_utils import save_indexing_stats
from Core.utils.artifact_paths import (
    resolve_run_artifact_path,
)
from Core.utils.vector_index_manifest import (
    VECTOR_INDEX_ATTESTATION_NAME,
    read_vector_index_attestation,
)
from docarag.persistence.stage_runtime import durable_stage


def _tree_source_sha256(tree) -> str | None:
    return getattr(getattr(tree, "meta_info", None), "source_sha256", None)


def _vdb_collection(cfg) -> str:
    return str(getattr(getattr(cfg, "vdb", None), "collection_name", "unspecified"))


def _entity_collection(cfg) -> str:
    return str(
        getattr(
            getattr(cfg, "graph", None),
            "entity_collection_name",
            "unspecified",
        )
    )


def _graph_variant(cfg) -> str:
    graph_cfg = getattr(cfg, "graph", None)
    return graph_artifact_variant(graph_cfg) if graph_cfg is not None else "basic"


def _persist_graph_evidence(run_directory: str) -> None:
    """Land this revision's graph rows before anything reads them back.

    Called in-process rather than shelled out, because the entity stage that
    follows is in this same process and the orchestrator's own persist step
    runs only after it exits.  Idempotent: persisting an already-persisted
    part rewrites the same rows.
    """

    from docarag.evidence_persist import main as persist_evidence

    code = persist_evidence(["--run", str(run_directory), "--part", "graph"])
    if code != 0:
        raise RuntimeError(f"graph evidence persistence failed (exit {code})")


def construct_GBC_index(
    cfg: SystemConfig,
    tree_only: bool = False,
    graph_only: bool = False,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
    token_scope: TokenUsageScope | None = None,
):
    """Construct one GBC index in an isolated run-level usage ledger."""

    usage = token_scope or current_token_scope() or TokenUsageScope(name="gbc-index")
    with usage:
        usage.begin_stage_sequence()
        return _construct_GBC_index(
            cfg,
            tree_only=tree_only,
            graph_only=graph_only,
            parsing_provider=parsing_provider,
        )


def _construct_GBC_index(
    cfg: SystemConfig,
    tree_only: bool = False,
    graph_only: bool = False,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
):
    """
    Construct the GBC index from the document tree and knowledge graph.

    :param cfg: Configuration object containing settings for the index construction.
    :return: A tuple containing the DocumentTree and Graph objects.
    """
    log.info("Starting GBC index construction...")

    token_tracker = TokenTracker.get_instance()
    # This dictionary will hold all stats for the CURRENT run
    current_run_stats = {}

    # --- Tree: built by the tree stage, read from the store by every other ---
    tree_start_time = time.time()
    if tree_only:
        tree_index = build_tree_from_pdf(cfg, parsing_provider=parsing_provider)
    else:
        from docarag.vdb_tree_source import StoreTreeSource

        tree_index = StoreTreeSource.open(cfg.save_path)
    tree_duration = time.time() - tree_start_time
    log.info(f"Document tree ready in {tree_duration:.2f} seconds.")
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    if tree_only:
        log.info("Only build tree index. Finished.")
        # Add final token usage to our stats dictionary
        current_run_stats["token_stage_history"] = token_tracker.stage_history

        # Save all collected stats and exit
        save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)
        return

    # --- Measure Knowledge Graph Building ---
    kg_start_time = time.time()
    source_sha256 = _tree_source_sha256(tree_index)
    gbc_index = None
    try:
        with durable_stage(
            "graph",
            payload={"variant": _graph_variant(cfg)},
            source_sha256=source_sha256,
        ) as graph_stage:
            graph_index = build_knowledge_graph(tree_index, cfg, stage=graph_stage)
            # A queryable BookRAG graph is the graph artifact *and* its entity
            # lookup index. Persist the graph first, then publish its lookup in
            # a separately leased stage below.
            gbc_index = GBC(config=cfg, graph_index=graph_index, TreeIndex=tree_index)
            # The immutable tree was already published by the summary stage.
            # Re-pickling it here could change bytes across processes while an
            # earlier PostgreSQL receipt still points at the same URI.
            if graph_stage.replayed:
                graph_stage.checkpoint("artifact.validated")
            else:
                graph_index.save_graph()
            if graph_stage.enabled:
                graph_stage.record_artifact(
                    os.path.join(
                        cfg.save_path,
                        Graph._get_filename(_graph_variant(cfg)),
                    ),
                    artifact_role="bookrag-knowledge-graph",
                    artifact_schema="bookrag.knowledge-graph.node-link.v1",
                    media_type="application/json",
                    source_sha256=source_sha256,
                    metadata={
                        "edge_count": graph_index.kg.number_of_edges(),
                        "node_count": graph_index.kg.number_of_nodes(),
                    },
                )

        kg_duration = time.time() - kg_start_time
        log.info(f"Knowledge graph constructed and saved in {kg_duration:.2f} seconds.")

        # The entity index is derived from the store: its records are the
        # revision's entity rows and its attestation binds to the graph seal
        # written when those rows land. So the rows have to land first. The
        # orchestrator persists a stage's evidence after the stage's process
        # exits, which is one process too late for the entity stage inside it.
        _persist_graph_evidence(cfg.save_path)

        with durable_stage(
            "entity",
            payload={"collection": _entity_collection(cfg)},
            source_sha256=source_sha256,
        ) as entity_stage:
            if entity_stage.replayed:
                ready_lease = gbc_index._acquire_ready_entity_attestation()
                ready_lease.close()
                entity_stage.checkpoint("attestation.validated")
            else:
                gbc_index.rebuild_vdb()
            if entity_stage.enabled:
                entity_stage.record_artifact(
                    os.path.join(
                        gbc_index.entity_attestation_path,
                        VECTOR_INDEX_ATTESTATION_NAME,
                    ),
                    artifact_role="qdrant-entity-index-attestation",
                    artifact_schema="docarag.qdrant-index-attestation.v1",
                    media_type="application/json",
                    source_sha256=source_sha256,
                    metadata={"collection": _entity_collection(cfg)},
                )
    finally:
        if gbc_index is not None:
            gbc_index.close()

    current_run_stats["build_kg_time"] = round(kg_duration, 2)

    if graph_only:
        log.info("Graph and entity index construction finished.")
        current_run_stats["token_stage_history"] = token_tracker.stage_history
        save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)
        return

    # --- Finalize and Save All Stats for the Full Run ---
    log.info("Full GBC index construction finished. Saving final stats...")
    current_run_stats["token_stage_history"] = token_tracker.stage_history

    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    return

def rebuild_graph_vdb(cfg: SystemConfig):
    gbc_index = GBC.load_gbc_index(cfg)
    try:
        source_sha256 = _tree_source_sha256(gbc_index.TreeIndex)
        with durable_stage(
            "entity",
            payload={"collection": _entity_collection(cfg)},
            source_sha256=source_sha256,
        ) as entity_stage:
            if entity_stage.replayed:
                ready_lease = gbc_index._acquire_ready_entity_attestation()
                ready_lease.close()
                entity_stage.checkpoint("attestation.validated")
            else:
                gbc_index.rebuild_vdb()
            if entity_stage.enabled:
                entity_stage.record_artifact(
                    os.path.join(
                        gbc_index.entity_attestation_path,
                        VECTOR_INDEX_ATTESTATION_NAME,
                    ),
                    artifact_role="qdrant-entity-index-attestation",
                    artifact_schema="docarag.qdrant-index-attestation.v1",
                    media_type="application/json",
                    source_sha256=source_sha256,
                    metadata={"collection": _entity_collection(cfg)},
                )
    finally:
        gbc_index.close()
    log.info("Rebuilt graph VDB successfully.")


def construct_vdb(
    cfg: SystemConfig,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
    token_scope: TokenUsageScope | None = None,
):
    """Construct one retrieval index in an isolated run-level usage ledger."""

    usage = (
        token_scope
        or current_token_scope()
        or TokenUsageScope(name="vector-index")
    )
    with usage:
        usage.begin_stage_sequence()
        return _construct_vdb(cfg, parsing_provider=parsing_provider)


def _construct_vdb(
    cfg: SystemConfig,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
):
    token_tracker = TokenTracker.get_instance()
    log.info("Starting vector database construction...")

    vdb_cfg: VDBConfig = cfg.vdb
    attestation_path = resolve_run_artifact_path(
        cfg.save_path,
        vdb_cfg.manifest_dir_name,
        label="vdb.manifest_dir_name",
    )
    vdb_cfg.manifest_dir_name = str(attestation_path)

    if cfg.index_type in ["vanilla", "bm25", "raptor"]:
        log.info(f"Index type is {cfg.index_type}. Start building other vdb index...")
        source = getattr(cfg, "source", None)
        source_sha256 = getattr(source, "sha256", None)
        with durable_stage(
            "vector",
            payload={
                "collection": vdb_cfg.collection_name,
                "index_type": cfg.index_type,
            },
            source_sha256=source_sha256,
        ) as vector_stage:
            build_other_vdb_index(cfg)
            if cfg.index_type == "bm25":
                artifact_path = Path(vdb_cfg.manifest_dir_name) / "bm25_index.pkl"
                artifact_role = "bookrag-bm25-index"
                artifact_schema = "bookrag.bm25-index.v1"
                media_type = "application/x-python-pickle"
            else:
                artifact_path = (
                    Path(vdb_cfg.manifest_dir_name) / VECTOR_INDEX_ATTESTATION_NAME
                )
                artifact_role = "qdrant-chunk-index-attestation"
                artifact_schema = "docarag.qdrant-index-attestation.v1"
                media_type = "application/json"
            if vector_stage.enabled:
                vector_stage.record_artifact(
                    artifact_path,
                    artifact_role=artifact_role,
                    artifact_schema=artifact_schema,
                    media_type=media_type,
                    source_sha256=source_sha256,
                    metadata={"collection": vdb_cfg.collection_name},
                )
        return

    current_run_stats = {}

    tree_start_time = time.time()
    from docarag.vdb_tree_source import StoreTreeSource

    tree_index = StoreTreeSource.open(cfg.save_path)
    tree_duration = time.time() - tree_start_time
    log.info(
        "Tree read from the store in %.2f seconds (%d nodes).",
        tree_duration,
        len(tree_index.nodes),
    )
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    current_run_stats["token_stage_history"] = token_tracker.stage_history

    # Save all collected stats and exit
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    log.info("Qdrant collection selected: %s", vdb_cfg.collection_name)

    os.makedirs(os.path.dirname(vdb_cfg.manifest_dir_name), exist_ok=True)

    vbd_start_time = time.time()
    with durable_stage(
        "vector",
        payload={
            "collection": vdb_cfg.collection_name,
            "index_type": cfg.index_type,
            # A rebuild against different sealed content is a different work
            # item; receipts stay append-only and history is kept.
            "store_seal": getattr(tree_index, "store_seal", None),
        },
        source_sha256=_tree_source_sha256(tree_index),
    ) as vector_stage:
        rebuilt = build_vdb_index(
            tree_index,
            vdb_cfg,
            reuse_if_ready=not vdb_cfg.force_rebuild,
        )
        if vector_stage.enabled:
            retrieval_manifest_path = (
                Path(vdb_cfg.manifest_dir_name) / VECTOR_INDEX_ATTESTATION_NAME
            )
            vector_stage.record_artifact(
                retrieval_manifest_path,
                artifact_role="qdrant-tree-index-attestation",
                artifact_schema="docarag.qdrant-index-attestation.v1",
                media_type="application/json",
                source_sha256=_tree_source_sha256(tree_index),
                metadata={"collection": vdb_cfg.collection_name},
            )
            attestation = read_vector_index_attestation(vdb_cfg.manifest_dir_name)
            point_count = None if attestation is None else attestation.get("point_count")
            if (
                isinstance(point_count, bool)
                or not isinstance(point_count, int)
                or point_count < 0
            ):
                raise ValueError(
                    "ready Qdrant attestation has no valid point_count for "
                    "PostgreSQL index publication"
                )
            vector_stage.publish_index_revision(
                configuration_sha256=os.environ.get(
                    "DOCARAG_SYSTEM_CONFIG_SHA256"
                ),
                canonical_artifact_sha256=getattr(
                    getattr(tree_index, "meta_info", None),
                    "canonical_artifact_sha256",
                    None,
                ),
                tree_artifact_path=DocumentTree.get_save_path(cfg.save_path),
                graph_artifact_path=os.path.join(
                    cfg.save_path,
                    Graph._get_filename(_graph_variant(cfg)),
                ),
                retrieval_manifest_path=retrieval_manifest_path,
                qdrant_collection=vdb_cfg.collection_name,
                qdrant_point_count=point_count,
                metadata={
                    "entity_collection": _entity_collection(cfg),
                    "retrieval_identity_sha256": attestation.get(
                        "identity_sha256"
                    ),
                },
            )
    vdb_duration = time.time() - vbd_start_time
    if rebuilt:
        log.info(f"Vector database constructed in {vdb_duration:.2f} seconds.")
    else:
        log.info(f"Ready vector database reused in {vdb_duration:.2f} seconds.")

    current_run_stats["build_vdb_time"] = round(vdb_duration, 2)

    # Save all collected stats and exit
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)


def compute_mm_reranker(
    cfg: SystemConfig,
    group: pd.DataFrame,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
):

    tree_index = build_tree_from_pdf(cfg, parsing_provider=parsing_provider)

    compute_mm_embedding(cfg, tree_index)
    
    compute_mm_embedding_question(cfg, group)


if __name__ == "__main__":
    print("test")

    # parser = argparse.ArgumentParser(description="Extract text content from PDF files.")
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
    #     help="Path to the configuration file.",
    # )

    # args = parser.parse_args()

    # cfg = load_system_config(args.config_path)

    # if not os.path.exists(cfg.save_path):
    #     os.makedirs(cfg.save_path)
    #     log.info(f"Created directory: {cfg.save_path}")
    # else:
    #     log.info(f"Directory already exists: {cfg.save_path}")

    # construct_vdb(cfg)

    # gbc_index = construct_GBC_index(cfg)
    # log.info("GBC index construction completed successfully.")
