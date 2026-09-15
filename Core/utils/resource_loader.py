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


def _vector_store_kwargs(
    vdb_cfg,
    scope_filter: dict[str, str],
    *,
    vector_schema: "VectorSchema | None" = None,
) -> dict[str, object]:
    """Open kwargs for one store. The schema must match the collection.

    Only the tree collection carries named vectors; the chunk, vanilla and
    raptor collections keep the single unnamed one, and the entity collection
    has its own shape and its own opener. Passing the wrong one fails at the
    first call rather than reading a vector name that is not there.
    """

    kwargs: dict[str, object] = {
        "qdrant_url": vdb_cfg.qdrant_url,
        "collection_name": vdb_cfg.collection_name,
        "api_key_env": vdb_cfg.api_key_env,
        "timeout_seconds": vdb_cfg.timeout_seconds,
        "scope_filter": scope_filter,
    }
    if vector_schema is not None:
        kwargs["vector_schema"] = vector_schema
    return kwargs


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
    """Return a shared lease only after store-bound TreeVDB attestation.

    The attestation binds the vectors to the sealed store rows, so readiness
    is one identity comparison plus one seal read -- no serialized tree is
    opened on the query path.
    """

    from types import SimpleNamespace

    from Core.pipelines.vdb_index import (
        VectorIndexBindingError,
        tree_vector_index_attestation_is_ready,
    )
    from Core.utils.vector_index_manifest import VectorIndexAttestationReadLease

    source = cfg.source
    if source is None or not source.sha256:
        raise VectorIndexBindingError(
            "BookRAG tree retrieval requires cfg.source.sha256 to validate TreeVDB"
        )
    expected = _runtime_scope(cfg, vdb_cfg, index_kind="tree")
    identity = SimpleNamespace(
        meta_info=SimpleNamespace(
            source_sha256=source.sha256,
            document_version_id=expected["document_version_id"],
        ),
        nodes=(),
        save_dir=cfg.save_path,
    )
    lease = VectorIndexAttestationReadLease(attestation_path)
    try:
        runtime_vdb_cfg = copy(vdb_cfg)
        runtime_vdb_cfg.manifest_dir_name = attestation_path
        if not tree_vector_index_attestation_is_ready(identity, runtime_vdb_cfg):
            raise VectorIndexBindingError(
                "TreeVDB attestation does not match the sealed store revision; "
                "rebuild the vdb stage"
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
    from Core.provider.vdb import VectorSchema, VectorStore
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
            from Core.provider.embedding import MultimodalEmbeddingProvider

            embed_model = MultimodalEmbeddingProvider(
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
            **_vector_store_kwargs(
                vdb_cfg, scope_filter, vector_schema=VectorSchema.TREE_NODE
            ),
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
    """Load and prepare the RAG agent's dependencies for one strategy."""

    rag_config = cfg.rag.strategy_config
    strategy_name = rag_config.strategy
    log.info(f"Preparing dependencies for RAG strategy: '{strategy_name}'")

    dependencies = {}

    if strategy_name == "traverse":
        # TraverseAgent walks the same tree the other strategies answer from:
        # the stored rows, not the run's pickle.
        from docarag.vdb_tree_source import StoreTreeSource

        tree_index = StoreTreeSource.open(cfg.save_path)
        log.info("Successfully loaded tree index from the store at %s", cfg.save_path)
        dependencies["tree_index"] = tree_index

    elif strategy_name == "gbc":
        from Core.rag.stored_index import StoredIndex

        gbc_index = None
        try:
            # The tree and graph are read from the store a slice at a time, so
            # a query's memory follows the question rather than the document.
            gbc_index = StoredIndex.open(cfg.save_path, config=cfg)
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
        from Core.rag.stored_index import StoredIndex

        gbc_index = StoredIndex.open(cfg.save_path, config=cfg)
        log.info(f"Successfully opened the stored index at {cfg.save_path}")
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
            from Core.provider.vdb import VectorSchema, VectorStore
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
        from Core.provider.vdb import VectorSchema, VectorStore
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
        from Core.provider.vdb import VectorSchema, VectorStore

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
                from Core.provider.embedding import MultimodalEmbeddingProvider

                embed_model = MultimodalEmbeddingProvider(
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
                **_vector_store_kwargs(
                    vdb_cfg, scope_filter, vector_schema=VectorSchema.TREE_NODE
                ),
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


class _CompositeReadLease:
    """Several attestation leases held as one, released together."""

    __slots__ = ("_leases",)

    def __init__(self, leases: list) -> None:
        self._leases = list(leases)

    def close(self) -> None:
        for lease in self._leases:
            if lease is not None:
                try:
                    lease.close()
                except Exception:
                    log.exception("Failed to close a composite read lease member")


class CorpusIndexBundle:
    """Every document a corpus-scoped GBCRAG/GraphRAG query can draw from.

    One shared ``tree_vdb``/``entity_vdb`` (scoped across every
    ``document_version_id`` via MatchAny), plus one ``StoredIndex`` per
    document for its own tree/graph. ``GBCRAG.from_corpus_bundle`` accepts
    this in place of a single document's ``gbc_index``/``tree_vdb``.
    """

    __slots__ = (
        "tenant_id", "corpus_id", "documents", "tree_vdb", "entity_vdb",
        "tree_embedder", "embedder", "document_titles", "_closed",
    )

    def __init__(
        self,
        *,
        tenant_id: str,
        corpus_id: str,
        documents: Dict[str, Any],
        tree_vdb: Any,
        entity_vdb: Any,
        tree_embedder: Any,
        embedder: Any,
        document_titles: Dict[str, str],
    ) -> None:
        if not documents:
            raise ValueError("a corpus bundle needs at least one document")
        self.tenant_id = tenant_id
        self.corpus_id = corpus_id
        self.documents = documents
        self.tree_vdb = tree_vdb
        self.entity_vdb = entity_vdb
        # tree_embedder feeds tree_vdb (config.vdb.embedding_config); embedder
        # is the entity/graph-side one (config.graph.embedding_config) --
        # StoredIndex._open_entity_search keeps these two separate for the
        # same reason (they are independently configured models, frequently
        # of different dimensionality), so a corpus bundle must too.
        self.tree_embedder = tree_embedder
        self.embedder = embedder
        self.document_titles = document_titles
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stored_index in self.documents.values():
            try:
                stored_index.close()
            except Exception:
                log.exception("Failed to close a corpus bundle's document index")
        for resource in (self.tree_vdb, self.entity_vdb, self.tree_embedder, self.embedder):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    log.exception("Failed to close a corpus bundle's shared resource")

    def __enter__(self) -> "CorpusIndexBundle":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def prepare_corpus_rag_dependencies(
    document_configs: "list[SystemConfig]",
) -> Dict[str, Any]:
    """Multi-document counterpart to ``prepare_rag_dependencies``.

    Only ``gbc``/``graph`` strategies have the entity/graph layer a corpus
    bundle needs; the other four strategies stay single-document. Each entry
    in ``document_configs`` is a fully-resolved per-document ``SystemConfig``
    (``save_path``/``source`` set, the shape a caller already builds for one
    document via ``prepare_rag_dependencies``) -- all sharing one
    tenant/corpus/tree collection.
    """

    if not document_configs:
        raise ValueError("prepare_corpus_rag_dependencies requires at least one document")

    strategy_name = document_configs[0].rag.strategy_config.strategy
    if strategy_name not in ("gbc", "graph"):
        raise ValueError(
            f"corpus-scoped retrieval supports only 'gbc'/'graph', got {strategy_name!r}"
        )

    base_cfg = document_configs[0]
    base_vdb_cfg = base_cfg.vdb

    from Core.rag.stored_index import StoredIndex

    documents: Dict[str, Any] = {}
    scopes: list = []
    read_leases: list = []
    try:
        for cfg in document_configs:
            if cfg.rag.strategy_config.strategy != strategy_name:
                raise ValueError("every document in a corpus bundle must use the same strategy")
            if (
                cfg.vdb.tenant_id != base_vdb_cfg.tenant_id
                or cfg.vdb.corpus_id != base_vdb_cfg.corpus_id
            ):
                raise ValueError("every document in a corpus bundle must share one tenant/corpus")
            if cfg.vdb.collection_name != base_vdb_cfg.collection_name:
                raise ValueError("every document in a corpus bundle must share one tree collection")

            stored_index = StoredIndex.open(cfg.save_path, config=None)
            scope = _runtime_scope(cfg, cfg.vdb, index_kind="tree")
            attestation_path = str(
                resolve_run_artifact_path(
                    cfg.save_path, cfg.vdb.manifest_dir_name, label="vdb.manifest_dir_name"
                )
            )
            read_lease = _acquire_ready_bookrag_tree_vdb(cfg, cfg.vdb, attestation_path)
            document_version_id = scope["document_version_id"]
            if document_version_id in documents:
                stored_index.close()
                read_lease.close()
                raise ValueError(f"duplicate document_version_id in corpus bundle: {document_version_id}")
            documents[document_version_id] = stored_index
            scopes.append(scope)
            read_leases.append(read_lease)

        document_version_ids = [s["document_version_id"] for s in scopes]
        merged_tree_scope = {
            "tenant_id": scopes[0]["tenant_id"],
            "corpus_id": scopes[0]["corpus_id"],
            "document_version_id": document_version_ids,
            "index_kind": "tree",
        }

        from Core.configs.embedding_config import EmbeddingConfig
        from Core.pipelines.vdb_index import corpus_entity_vector_scope
        from Core.provider.embedding import TextEmbeddingProvider
        from Core.provider.vdb import VectorSchema, VectorStore

        # Two independently configured embedders, matching the single-document
        # path (StoredIndex._open_entity_search): tree_vdb uses vdb.embedding_
        # config, entity_vdb uses graph.embedding_config -- these are commonly
        # different models of different dimensionality (e.g. a Qwen3-VL tree
        # embedder alongside an Ollama qwen3-embedding entity embedder), and
        # entity_vdb's collection is created at whichever dimension its own
        # config's embedder produces.
        tree_embed_cfg: EmbeddingConfig = base_vdb_cfg.embedding_config
        tree_embedder = TextEmbeddingProvider(
            model_name=tree_embed_cfg.model_name,
            backend=tree_embed_cfg.backend,
            device=tree_embed_cfg.device,
            max_length=tree_embed_cfg.max_length,
            api_base=tree_embed_cfg.api_base,
            api_key=getattr(tree_embed_cfg, "api_key", None),
            num_ctx=getattr(tree_embed_cfg, "num_ctx", None),
        )
        entity_embed_cfg: EmbeddingConfig = base_cfg.graph.embedding_config
        entity_embedder = TextEmbeddingProvider(
            model_name=entity_embed_cfg.model_name,
            backend=entity_embed_cfg.backend,
            device=entity_embed_cfg.device,
            max_length=entity_embed_cfg.max_length,
            api_base=entity_embed_cfg.api_base,
            api_key=getattr(entity_embed_cfg, "api_key", None),
            num_ctx=getattr(entity_embed_cfg, "num_ctx", None),
        )
        tree_vdb = None
        entity_vdb = None
        try:
            tree_vdb = VectorStore(
                embedding_model=tree_embedder,
                **_vector_store_kwargs(
                    base_vdb_cfg, merged_tree_scope, vector_schema=VectorSchema.TREE_NODE
                ),
            )
            # tree_revision_id is per-document identity elsewhere in the
            # single-document path; a corpus digest is only ever used for
            # trace/logging here, never compared against a per-document seal.
            tree_vdb.tree_revision_id = "corpus-" + "-".join(sorted(document_version_ids))
            tree_vdb.attach_read_lease(_CompositeReadLease(read_leases))
            read_leases = []

            entity_vdb = VectorStore(
                qdrant_url=base_vdb_cfg.qdrant_url,
                embedding_model=entity_embedder,
                collection_name=base_cfg.graph.entity_collection_name,
                api_key_env=base_vdb_cfg.api_key_env,
                timeout_seconds=base_vdb_cfg.timeout_seconds,
                scope_filter=corpus_entity_vector_scope(base_vdb_cfg, document_version_ids),
                hybrid_entity_schema=True,
            )
        except BaseException:
            _close_partial_resources(
                ("corpus tree_vdb", tree_vdb),
                ("corpus entity_vdb", entity_vdb),
                ("corpus tree embedder", tree_embedder),
                ("corpus entity embedder", entity_embedder),
            )
            raise

        document_titles = {
            doc_id: stored_index.TreeIndex.meta_info.file_name
            for doc_id, stored_index in documents.items()
        }
        bundle = CorpusIndexBundle(
            tenant_id=scopes[0]["tenant_id"],
            corpus_id=scopes[0]["corpus_id"],
            documents=documents,
            tree_vdb=tree_vdb,
            entity_vdb=entity_vdb,
            tree_embedder=tree_embedder,
            embedder=entity_embedder,
            document_titles=document_titles,
        )
        log.info(
            "Successfully loaded a %d-document corpus bundle for strategy '%s'",
            len(documents), strategy_name,
        )
        return {"corpus_bundle": bundle}
    except BaseException:
        _close_partial_resources(*[(f"document {k}", v) for k, v in documents.items()])
        for lease in read_leases:
            if lease is not None:
                lease.close()
        raise
