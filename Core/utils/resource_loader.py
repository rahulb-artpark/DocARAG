from copy import copy
import os
from pathlib import Path
from typing import Dict, Any
from Core.configs.system_config import SystemConfig
import logging
from Core.utils.artifact_paths import resolve_run_artifact_path


log = logging.getLogger(__name__)


def _close_partial_resources(*resources: tuple[str, object]) -> None:
    """Best-effort cleanup for dependency factories that did not return."""

    seen: set[int] = set()
    for label, resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.exception("Failed to close partial RAG dependency %s", label)


def _runtime_scope(
    cfg: SystemConfig,
    vdb_cfg,
    *,
    index_kind: str,
) -> dict[str, str]:
    """Bind every remote vector access to one immutable document version."""

    from Core.pipelines.vdb_index import (
        VectorIndexBindingError,
        _is_lower_sha256,
        _validated_index_revision_id,
    )

    source = cfg.source
    if source is None or not _is_lower_sha256(source.sha256):
        raise VectorIndexBindingError(
            "Qdrant retrieval requires cfg.source.sha256"
        )
    if not cfg.save_path:
        raise VectorIndexBindingError("Qdrant retrieval requires cfg.save_path")
    document_version_id = (
        getattr(source, "document_version_id", None)
        or getattr(vdb_cfg, "document_version_id", None)
        or Path(cfg.save_path).resolve().name
    )
    if (
        not isinstance(document_version_id, str)
        or not document_version_id
        or document_version_id != document_version_id.strip()
    ):
        raise VectorIndexBindingError(
            "Qdrant retrieval requires the stable source document_version_id"
        )
    configured_document_version = getattr(vdb_cfg, "document_version_id", None)
    if (
        configured_document_version is not None
        and configured_document_version != document_version_id
    ):
        raise VectorIndexBindingError(
            "Qdrant retrieval document version differs from the configured scope"
        )
    return {
        "tenant_id": vdb_cfg.tenant_id,
        "corpus_id": vdb_cfg.corpus_id,
        "document_version_id": document_version_id,
        "index_revision_id": _validated_index_revision_id(
            vdb_cfg, run_directory=cfg.save_path
        ),
        "source_sha256": source.sha256,
        "index_kind": index_kind,
    }


def _vector_store_kwargs(vdb_cfg, scope_filter: dict[str, str]) -> dict[str, object]:
    return {
        "qdrant_url": vdb_cfg.qdrant_url,
        "collection_name": vdb_cfg.collection_name,
        "api_key_env": vdb_cfg.api_key_env,
        "timeout_seconds": vdb_cfg.timeout_seconds,
        "scope_filter": scope_filter,
    }


def _acquire_ready_scoped_qdrant(
    vdb_cfg,
    scope_filter: dict[str, str],
) -> Any:
    """Lease a local attestation only after exact live Qdrant validation."""

    from Core.pipelines.vdb_index import VectorIndexBindingError, _sha256_json
    from Core.utils.artifact_paths import qdrant_collection_state, qdrant_endpoint_sha256
    from Core.utils.vector_index_manifest import (
        VectorIndexAttestationReadLease,
        read_vector_index_attestation,
        vector_index_attestation_matches,
    )

    directory = vdb_cfg.manifest_dir_name
    lease = VectorIndexAttestationReadLease(directory)
    try:
        attestation = read_vector_index_attestation(directory)
        if not isinstance(attestation, dict):
            raise VectorIndexBindingError("Qdrant content attestation is missing")
        identity = {
            key: value
            for key, value in attestation.items()
            if key not in {"identity_sha256", "state"}
        }
        if (
            attestation.get("state") != "content_attested"
            or attestation.get("identity_sha256") != _sha256_json(identity)
            or attestation.get("scope_filter") != scope_filter
            or attestation.get("collection_name") != vdb_cfg.collection_name
            or attestation.get("qdrant_endpoint_sha256")
            != qdrant_endpoint_sha256(vdb_cfg.qdrant_url)
        ):
            raise VectorIndexBindingError(
                "Qdrant attestation differs from the requested source, scope, or endpoint"
            )
        remote = qdrant_collection_state(
            qdrant_url=vdb_cfg.qdrant_url,
            collection_name=vdb_cfg.collection_name,
            scope_filter=scope_filter,
            api_key_env=vdb_cfg.api_key_env,
            timeout_seconds=vdb_cfg.timeout_seconds,
        )
        if not vector_index_attestation_matches(
            directory,
            attestation,
            observed_point_count=(remote.point_count if remote.exists else None),
        ):
            raise VectorIndexBindingError(
                "Qdrant live point count differs from the local content attestation"
            )
        return lease
    except BaseException:
        lease.close()
        raise


def _acquire_ready_bookrag_tree_vdb(
    cfg: SystemConfig,
    vdb_cfg,
    attestation_path: str,
) -> Any:
    """Return a shared lease only after source-bound TreeVDB attestation."""

    from Core.Index.Tree import DocumentTree
    from Core.pipelines.vdb_index import (
        VectorIndexBindingError,
        bookrag_tree_artifact_sha256,
        tree_vector_scope,
        tree_vector_index_attestation_is_ready,
    )
    from Core.utils.vector_index_manifest import VectorIndexAttestationReadLease

    source = cfg.source
    if source is None or not source.sha256:
        raise VectorIndexBindingError(
            "BookRAG tree retrieval requires cfg.source.sha256 to validate TreeVDB"
        )
    lease = VectorIndexAttestationReadLease(attestation_path)
    try:
        tree_path = DocumentTree.get_save_path(cfg.save_path)
        digest_before = bookrag_tree_artifact_sha256(tree_path)
        tree = DocumentTree.load_from_file(tree_path)
        digest_after = bookrag_tree_artifact_sha256(tree_path)
        if digest_before != digest_after:
            raise VectorIndexBindingError(
                "BookRAG tree artifact changed while it was being loaded"
            )
        if tree.meta_info.source_sha256 != source.sha256:
            raise VectorIndexBindingError(
                "TreeVDB tree source SHA-256 differs from cfg.source"
            )
        if tree.meta_info.source_uri != source.original_uri:
            raise VectorIndexBindingError(
                "TreeVDB tree source URI differs from cfg.source"
            )
        if tree_vector_scope(tree, vdb_cfg) != _runtime_scope(
            cfg, vdb_cfg, index_kind="tree"
        ):
            raise VectorIndexBindingError(
                "TreeVDB scope differs from the selected document revision"
            )

        runtime_vdb_cfg = copy(vdb_cfg)
        runtime_vdb_cfg.manifest_dir_name = attestation_path
        if not tree_vector_index_attestation_is_ready(
            tree,
            runtime_vdb_cfg,
            tree_artifact_sha256=digest_after,
        ):
            raise VectorIndexBindingError(
                "TreeVDB is missing, incomplete, or belongs to another "
                "tree/source/embedding configuration; rebuild the vdb stage"
            )
        return lease
    except BaseException:
        lease.close()
        raise


def _gme_dtype(embed_cfg) -> dict:
    """Pass an explicit load dtype only when the configuration states one."""

    dtype = getattr(embed_cfg, "torch_dtype", None)
    return {"torch_dtype": dtype} if dtype else {}


def _open_ready_bookrag_tree_vdb(cfg: SystemConfig, vdb_cfg) -> Any:
    """Open the exact attested TreeVDB used by the BookRAG tree stage.

    The tree-vector builder selects its embedding provider from
    ``mm_embedding``.  Query startup deliberately makes the same choice and
    transfers the shared attestation lease to the returned store.
    """

    from Core.configs.embedding_config import EmbeddingConfig
    from Core.pipelines.vdb_index import _is_lower_sha256
    from Core.provider.vdb import VectorStore
    from Core.utils.vector_index_manifest import read_vector_index_attestation

    attestation_path = str(
        resolve_run_artifact_path(
            cfg.save_path,
            vdb_cfg.manifest_dir_name,
            label="vdb.manifest_dir_name",
        )
    )
    read_lease = _acquire_ready_bookrag_tree_vdb(
        cfg,
        vdb_cfg,
        attestation_path,
    )
    scope_filter = _runtime_scope(cfg, vdb_cfg, index_kind="tree")

    embed_model = None
    vdb = None
    try:
        embed_cfg: EmbeddingConfig = vdb_cfg.embedding_config
        if vdb_cfg.mm_embedding:
            from Core.provider.embedding import GmeEmbeddingProvider

            embed_model = GmeEmbeddingProvider(
                model_name=embed_cfg.model_name,
                device=embed_cfg.device,
                **_gme_dtype(embed_cfg),
            )
        else:
            from Core.provider.embedding import TextEmbeddingProvider

            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=getattr(embed_cfg, "api_key", None),
                num_ctx=getattr(embed_cfg, "num_ctx", None),
            )

        vdb = VectorStore(
            embedding_model=embed_model,
            **_vector_store_kwargs(vdb_cfg, scope_filter),
        )
        attestation = read_vector_index_attestation(attestation_path)
        tree_sha256 = (
            attestation.get("tree_sha256")
            if isinstance(attestation, dict)
            else None
        )
        if not _is_lower_sha256(tree_sha256):
            raise RuntimeError("TreeVDB attestation is missing its tree revision")
        vdb.tree_revision_id = f"tree-{tree_sha256}"
        vdb.attach_read_lease(read_lease)
        read_lease = None
        return vdb
    except BaseException:
        try:
            _close_partial_resources(
                ("partial BookRAG tree vector store", vdb),
                ("BookRAG tree embedder", embed_model),
            )
        finally:
            if read_lease is not None:
                read_lease.close()
        raise


def prepare_rag_dependencies(cfg: SystemConfig) -> Dict[str, Any]:
    """
    根据配置加载并准备RAG agent所需的依赖项。
    这是一个调度函数，它知道哪种策略需要哪种资源。
    """

    rag_config = cfg.rag.strategy_config
    strategy_name = rag_config.strategy
    log.info(f"Preparing dependencies for RAG strategy: '{strategy_name}'")

    dependencies = {}

    if strategy_name == "traverse":
        from Core.Index.Tree import DocumentTree

        # 加载 TraverseAgent 需要的 tree_index
        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        dependencies["tree_index"] = tree_index

    elif strategy_name == "gbc":
        from Core.Index.GBCIndex import GBC

        gbc_index = None
        try:
            gbc_index = GBC.load_gbc_index(cfg)
            tree_vdb = _open_ready_bookrag_tree_vdb(cfg, cfg.vdb)
        except BaseException:
            _close_partial_resources(("partial GBC index", gbc_index))
            raise
        log.info(
            "Successfully loaded GBC index and attested TreeVDB from %s",
            cfg.save_path,
        )
        dependencies["gbc_index"] = gbc_index
        dependencies["tree_vdb"] = tree_vdb
        asset_cas_root = os.environ.get("DOCARAG_ASSET_CAS_ROOT")
        if asset_cas_root is None:
            run_cas = Path(cfg.save_path).expanduser().resolve() / "asset_cas"
            if run_cas.is_dir():
                asset_cas_root = str(run_cas)
        if asset_cas_root is not None:
            if (
                not asset_cas_root
                or asset_cas_root != asset_cas_root.strip()
                or not Path(asset_cas_root).expanduser().is_absolute()
            ):
                _close_partial_resources(
                    ("GBC TreeVDB embedder", getattr(tree_vdb, "embedding_model", None)),
                    ("GBC TreeVDB", tree_vdb),
                    ("GBC index", gbc_index),
                )
                raise ValueError(
                    "DOCARAG_ASSET_CAS_ROOT must be an absolute trimmed path"
                )
            dependencies["asset_cas_root"] = asset_cas_root
    elif strategy_name == "graph":
        from Core.Index.GBCIndex import GBC

        gbc_index = GBC.load_gbc_index(cfg)
        log.info(f"Successfully loaded GBC index from {cfg.save_path}")
        dependencies["gbc_index"] = gbc_index

    elif strategy_name == "vanilla":
        from Core.configs.vdb_config import VDBConfig
        retrieval_method = rag_config.retrieval_method
        
        vdb_cfg: VDBConfig = rag_config.vdb_config
        if retrieval_method == "bm25":
            from Core.utils.bm25 import BM25
            lexical_store_path = str(
                resolve_run_artifact_path(
                    cfg.save_path,
                    vdb_cfg.manifest_dir_name,
                    label="rag.vdb_config.manifest_dir_name",
                )
            )
            bm25_path = os.path.join(lexical_store_path, "bm25_index.pkl")
            bm25 = BM25.load(bm25_path)
            log.info(f"Successfully loaded BM25 index from {bm25_path}")
            dependencies["bm25"] = bm25
        else:
            from Core.configs.embedding_config import EmbeddingConfig
            from Core.provider.vdb import VectorStore
            from Core.provider.embedding import TextEmbeddingProvider

            embed_cfg: EmbeddingConfig = rag_config.vdb_config.embedding_config

            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
                api_key=getattr(embed_cfg, "api_key", None),
                num_ctx=getattr(embed_cfg, "num_ctx", None),
            )
            
            scope_filter = _runtime_scope(
                cfg, vdb_cfg, index_kind=str(cfg.index_type)
            )
            read_lease = _acquire_ready_scoped_qdrant(vdb_cfg, scope_filter)
            vdb = None
            try:
                vdb = VectorStore(
                    embedding_model=embed_model,
                    **_vector_store_kwargs(vdb_cfg, scope_filter),
                )
                vdb.attach_read_lease(read_lease)
                read_lease = None
            except BaseException:
                _close_partial_resources(
                    ("partial vanilla vector store", vdb),
                    ("vanilla embedder", embed_model),
                )
                if read_lease is not None:
                    read_lease.close()
                raise
            log.info("Successfully loaded scoped Qdrant collection %s", vdb_cfg.collection_name)
            dependencies["vector_store"] = vdb

    elif strategy_name == "gbcvanilla":
        from Core.configs.embedding_config import EmbeddingConfig
        from Core.configs.vdb_config import VDBConfig
        from Core.provider.vdb import VectorStore
        from Core.provider.embedding import TextEmbeddingProvider

        # This strategy consumes a source-scoped chunk collection plus the
        # separately source-scoped entity collection.
        embed_cfg: EmbeddingConfig = rag_config.tree_vdb_config.embedding_config
        
        tree_vdb_cfg: VDBConfig = rag_config.tree_vdb_config
        graph_vdb_cfg: VDBConfig = rag_config.graph_vdb_config
        tree_scope = _runtime_scope(
            cfg, tree_vdb_cfg, index_kind="pdf_vanilla"
        )
        graph_scope = _runtime_scope(cfg, graph_vdb_cfg, index_kind="entity")
        tree_lease = _acquire_ready_scoped_qdrant(tree_vdb_cfg, tree_scope)
        graph_lease = None
        try:
            graph_lease = _acquire_ready_scoped_qdrant(graph_vdb_cfg, graph_scope)
        except BaseException:
            tree_lease.close()
            raise

        embed_model = TextEmbeddingProvider(
            model_name=embed_cfg.model_name,
            backend=embed_cfg.backend,
            device=embed_cfg.device,
            max_length=embed_cfg.max_length,
            api_base=embed_cfg.api_base,
            api_key=getattr(embed_cfg, "api_key", None),
            num_ctx=getattr(embed_cfg, "num_ctx", None),
        )
        
        tree_vdb = None
        graph_vdb = None
        try:
            tree_vdb = VectorStore(
                embedding_model=embed_model,
                **_vector_store_kwargs(tree_vdb_cfg, tree_scope),
            )
            tree_vdb.attach_read_lease(tree_lease)
            tree_lease = None

            graph_vdb = VectorStore(
                embedding_model=embed_model,
                **_vector_store_kwargs(graph_vdb_cfg, graph_scope),
            )
            graph_vdb.attach_read_lease(graph_lease)
            graph_lease = None
        except BaseException:
            try:
                _close_partial_resources(
                    ("partial graph VDB", graph_vdb),
                    ("partial tree VDB", tree_vdb),
                    ("shared GBC-vanilla embedder", embed_model),
                )
            finally:
                if tree_lease is not None:
                    tree_lease.close()
                if graph_lease is not None:
                    graph_lease.close()
            raise
        log.info("Successfully loaded scoped chunk and entity Qdrant collections")
        dependencies["tree_vdb"] = tree_vdb
        dependencies["graph_vdb"] = graph_vdb

    elif strategy_name == "mmr":
        from Core.configs.embedding_config import EmbeddingConfig
        from Core.provider.vdb import VectorStore

        from Core.configs.vdb_config import VDBConfig

        vdb_cfg: VDBConfig = rag_config.vdb_config
        vdb_attestation_path = str(
            resolve_run_artifact_path(
                cfg.save_path,
                vdb_cfg.manifest_dir_name,
                label="rag.vdb_config.manifest_dir_name",
            )
        )
        read_lease = _acquire_ready_bookrag_tree_vdb(
            cfg,
            vdb_cfg,
            vdb_attestation_path,
        )
        scope_filter = _runtime_scope(cfg, vdb_cfg, index_kind="tree")

        embed_model = None
        vdb = None
        try:
            embed_cfg: EmbeddingConfig = rag_config.vdb_config.embedding_config
            # The builder selects its provider from ``mm_embedding``. Query
            # startup must use the same effective setting; the historical
            # embedding ``type`` field is redundant and can otherwise open a
            # text embedder against a GME collection (or vice versa).
            if vdb_cfg.mm_embedding:
                from Core.provider.embedding import GmeEmbeddingProvider

                embed_model = GmeEmbeddingProvider(
                    model_name=embed_cfg.model_name,
                    device=embed_cfg.device,
                    **_gme_dtype(embed_cfg),
                )
            else:
                from Core.provider.embedding import TextEmbeddingProvider

                embed_model = TextEmbeddingProvider(
                    model_name=embed_cfg.model_name,
                    backend=embed_cfg.backend,
                    device=embed_cfg.device,
                    max_length=embed_cfg.max_length,
                    api_base=embed_cfg.api_base,
                    api_key=getattr(embed_cfg, "api_key", None),
                    num_ctx=getattr(embed_cfg, "num_ctx", None),
                )

            vdb = VectorStore(
                embedding_model=embed_model,
                **_vector_store_kwargs(vdb_cfg, scope_filter),
            )
            vdb.attach_read_lease(read_lease)
            read_lease = None
        except BaseException:
            try:
                _close_partial_resources(
                    ("partial MMR vector store", vdb),
                    ("partial MMR embedder", embed_model),
                )
            finally:
                if read_lease is not None:
                    read_lease.close()
            raise
        log.info("Successfully loaded scoped Qdrant collection %s", vdb_cfg.collection_name)
        dependencies["vector_store"] = vdb
    else:
        raise ValueError(f"Unknown or unsupported RAG strategy: '{strategy_name}'")

    return dependencies
