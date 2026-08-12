from Core.Index.Tree import *
from Core.configs.system_config import SystemConfig
from Core.configs.graph_config import entity_attestation_directory, graph_artifact_variant
from Core.Index.Graph import Graph
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore
from Core.pipelines.vdb_index import entity_vector_scope
from Core.pipelines.entity_facets import (
    CANONICAL_FIELD,
    context_text,
    lexical_terms,
    lexical_text,
)
from Core.utils.artifact_paths import qdrant_collection_state, qdrant_endpoint_sha256
from Core.utils.vector_index_manifest import (
    VectorIndexAttestationReadLease,
    invalidate_vector_index_attestation,
    publish_vector_index_attestation,
    read_vector_index_attestation,
    vector_index_attestation_matches,
    vector_index_lock,
)

import hashlib
import json
from pathlib import Path


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class GBC:
    """
    A class representing the index combining graph and tree structures.
    This class allows for the creation and management of a tree index, which can be used
    to organize and retrieve information in multimodal applications.
    """

    def __init__(
        self,
        config: SystemConfig,
        graph_index: Optional[Graph] = None,
        TreeIndex: Optional[DocumentTree] = None,
        *,
        require_entity_ready: bool = False,
    ):
        """
        Initializes the TreeIndex with an optional index.

        :param index: Optional initial index for the tree.
        """
        self.save_dir = config.save_path
        self.config = config
        self.TreeIndex: DocumentTree = TreeIndex
        self.GraphIndex: Graph = graph_index

        if self.TreeIndex is None:
            raise ValueError("Entity Qdrant scope requires a source-bound BookRAG tree")
        self.entity_scope = entity_vector_scope(
            self.TreeIndex.meta_info, config.vdb, run_directory=self.TreeIndex.save_dir
        )
        self.entity_attestation_path = entity_attestation_directory(
            self.save_dir, config.graph
        )
        self._entity_read_lease = None
        self.embedder = None
        self.entity_vdb = None
        try:
            if require_entity_ready:
                self._entity_read_lease = self._acquire_ready_entity_attestation()
            self.embedder = TextEmbeddingProvider(
                model_name=config.graph.embedding_config.model_name,
                backend=config.graph.embedding_config.backend,
                max_length=config.graph.embedding_config.max_length,
                device=config.graph.embedding_config.device,
                api_base=config.graph.embedding_config.api_base,
                api_key=getattr(config.graph.embedding_config, "api_key", None),
                num_ctx=getattr(config.graph.embedding_config, "num_ctx", None),
            )
            self.entity_vdb = VectorStore(
                qdrant_url=config.vdb.qdrant_url,
                embedding_model=self.embedder,
                collection_name=config.graph.entity_collection_name,
                api_key_env=config.vdb.api_key_env,
                timeout_seconds=config.vdb.timeout_seconds,
                scope_filter=self.entity_scope,
                # Same collection the refiner writes, so the same schema.
                hybrid_entity_schema=True,
            )
            if self._entity_read_lease is not None:
                self.entity_vdb.attach_read_lease(self._entity_read_lease)
                self._entity_read_lease = None
        except BaseException:
            self.close()
            raise
        log.info(
            "Entity Qdrant collection loaded with an immutable document scope"
        )

    def _entity_records(self):
        """The entity points this index publishes, in the refiner's shape.

        A rebuilt index is read by the same fused, grouped query the refiner
        uses, so it has to carry the same three vectors. ``canonical_id`` is the
        node's own id here: merges are already resolved by the time the graph is
        saved, so every surviving entity is its own canonical.

        Read from the store rather than from ``graph_data.json``. The artifact
        was the only place a revision's own entity descriptions lived, which
        made a build artifact a source of truth for what queries search; the
        store now records them per revision and answers the same question
        without a file in the loop.
        """

        from docarag.document_store import DocumentStore
        from Core.rag.stored_index import entity_node_name

        store = DocumentStore.open(self.save_dir)
        try:
            # ``iter_graph_entities``, not ``graph_entities``: the latter caps
            # itself at MAX_GRAPH_ENTITIES for display, and an index built from
            # a cap can never reach the entities beyond it. One revision here
            # holds 36,572 against a cap of 8,192.
            rows = list(store.tree.iter_graph_entities())
        finally:
            store.close()
        for row in rows:
            node_name = entity_node_name(row["canonical"], row["entity_kind"])
            yield node_name, {
                "entity_name": row["canonical"],
                "entity_type": row["entity_kind"],
                "description": row["description"] or "",
                "node_id": node_name,
                CANONICAL_FIELD: node_name,
                **self.entity_scope,
            }

    def _graph_seal(self) -> str:
        """The store digest these entity vectors attest to."""

        from docarag.document_store import scope_for_run
        from docarag.persistence.postgres import PostgresLedger
        from docarag.settings import load_settings

        scope, revision_id = scope_for_run(Path(self.save_dir))
        seal = PostgresLedger(load_settings().resolve_database_url()).graph_seal(
            scope, revision_id
        )
        if not seal:
            raise ValueError(
                "entity attestation requires a sealed graph; re-run the graph stage"
            )
        return seal

    def _expected_entity_attestation(self) -> dict[str, object]:
        records = list(self._entity_records())
        if not records:
            raise ValueError("refusing to attest an empty entity index")
        embedding = self.config.graph.embedding_config
        embedding_identity = {
            "model_name": embedding.model_name,
            "backend": embedding.backend,
            "max_length": embedding.max_length,
            "api_base_sha256": (
                hashlib.sha256(embedding.api_base.encode("utf-8")).hexdigest()
                if embedding.api_base
                else None
            ),
        }
        projection = [
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
            for text, metadata in records
        ]
        identity = {
            # v4: the entity records come from the store and the binding is the
            # graph seal rather than the graph file's digest, so every v3
            # attestation is refused and rebuilt rather than trusted against a
            # source it no longer describes.
            "schema": "docarag.bookrag-entity-qdrant-index.v4",
            "source_sha256": self.entity_scope["source_sha256"],
            "document_version_id": self.entity_scope["document_version_id"],
            "index_revision_id": self.entity_scope["index_revision_id"],
            "graph_sha256": self._graph_seal(),
            "projection_sha256": _json_sha256(projection),
            "point_count": len(records),
            "collection_name": self.config.graph.entity_collection_name,
            "scope_filter": self.entity_scope,
            "qdrant_endpoint_sha256": qdrant_endpoint_sha256(
                self.config.vdb.qdrant_url
            ),
            "embedding_identity_sha256": _json_sha256(embedding_identity),
        }
        return {
            **identity,
            "identity_sha256": _json_sha256(identity),
            "state": "content_attested",
        }

    def _acquire_ready_entity_attestation(self):
        lease = VectorIndexAttestationReadLease(self.entity_attestation_path)
        try:
            expected = self._expected_entity_attestation()
            if read_vector_index_attestation(self.entity_attestation_path) != expected:
                raise ValueError("entity Qdrant content attestation is missing or stale")
            remote = qdrant_collection_state(
                qdrant_url=self.config.vdb.qdrant_url,
                collection_name=self.config.graph.entity_collection_name,
                scope_filter=self.entity_scope,
                api_key_env=self.config.vdb.api_key_env,
                timeout_seconds=self.config.vdb.timeout_seconds,
            )
            if not vector_index_attestation_matches(
                self.entity_attestation_path,
                expected,
                observed_point_count=(remote.point_count if remote.exists else None),
            ):
                raise ValueError(
                    "entity Qdrant live point count differs from its attestation"
                )
            return lease
        except BaseException:
            lease.close()
            raise

    def save_gbc_index(self):
        """
        Saves the GBC index to the specified path.

        :param save_path: The path where the index will be saved.
        """
        if self.TreeIndex:
            self.TreeIndex.save_to_file()
        if self.GraphIndex:
            self.GraphIndex.save_graph()

        log.info(f"GBC index saved")

    def rebuild_vdb(self, *, batch_size: int = 256):
        """
        Rebuilds the vector database for entities from the store's graph rows.
        """
        if not getattr(self, "save_dir", None):
            raise ValueError("Entity rebuild requires the run directory.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        expected = self._expected_entity_attestation()
        # Refuse before destroying anything. The old order invalidated the
        # attestation and reset the scope, then discovered at write time that
        # the collection could not hold named vectors -- so a revision on a
        # pre-hybrid collection lost the index it had and gained nothing. A
        # rebuild that cannot write must leave what is there alone.
        self.entity_vdb.require_writable()
        with vector_index_lock(self.entity_attestation_path, exclusive=True):
            invalidate_vector_index_attestation(self.entity_attestation_path)
            try:
                self.entity_vdb.reset()
                records = []
                total = 0
                for text, metadata in self._entity_records():
                    records.append(self._entity_point_record(text, metadata))
                    if len(records) >= batch_size:
                        self.entity_vdb.add_entity_points(
                            records, batch_size=batch_size
                        )
                        total += len(records)
                        records = []
                if records:
                    self.entity_vdb.add_entity_points(records, batch_size=batch_size)
                    total += len(records)
                observed = self.entity_vdb.count()
                if observed != total or total != expected["point_count"]:
                    raise ValueError(
                        f"entity Qdrant scope contains {observed} points; "
                        f"expected {expected['point_count']}"
                    )
                publish_vector_index_attestation(
                    self.entity_attestation_path, expected
                )
            except BaseException:
                invalidate_vector_index_attestation(self.entity_attestation_path)
                raise
        log.info("Rebuilt and attested entity Qdrant scope with %d entries.", total)

    def _entity_point_record(self, text, metadata) -> dict:
        """One entity's point in the shape the writer and the refiner share."""

        description = metadata.get("description", "") or ""
        return {
            "name_text": text,
            "context_text": context_text(text, description),
            "lexical_terms": lexical_terms(lexical_text(text, description)),
            "metadata": metadata,
        }

    def reindex_entities(self, node_names) -> int:
        """Re-embed the named entities in place, and re-attest the scope.

        An entity's point ID is derived from its name and metadata, not its
        description, so a rewritten description keeps the ID and an upsert
        replaces the point. Nothing is deleted and nothing is reset: the
        collection is shared between documents, and a reset would take this
        document's whole scope to change one entity.

        The attestation is recomputed over every entity because its projection
        digest covers all of them -- reading is cheap, embedding is what this
        avoids.
        """

        wanted = {str(name) for name in node_names}
        if not wanted:
            raise ValueError("reindex_entities requires at least one entity")
        expected = self._expected_entity_attestation()
        self.entity_vdb.require_writable()
        with vector_index_lock(self.entity_attestation_path, exclusive=True):
            invalidate_vector_index_attestation(self.entity_attestation_path)
            try:
                records = [
                    self._entity_point_record(text, metadata)
                    for text, metadata in self._entity_records()
                    if text in wanted
                ]
                missing = wanted.difference(
                    record["name_text"] for record in records
                )
                if missing:
                    raise ValueError(
                        "this revision does not mention: "
                        + ", ".join(sorted(missing)[:3])
                    )
                self.entity_vdb.add_entity_points(records)
                observed = self.entity_vdb.count()
                if observed != expected["point_count"]:
                    raise ValueError(
                        f"entity Qdrant scope contains {observed} points; "
                        f"expected {expected['point_count']}"
                    )
                publish_vector_index_attestation(
                    self.entity_attestation_path, expected
                )
            except BaseException:
                invalidate_vector_index_attestation(self.entity_attestation_path)
                raise
        log.info("Re-embedded %d entity point(s) in place.", len(records))
        return len(records)

    def close(self) -> None:
        """Release entity-store and embedding resources idempotently."""

        entity_vdb = getattr(self, "entity_vdb", None)
        if entity_vdb is not None:
            entity_vdb.close()
            self.entity_vdb = None
        read_lease = getattr(self, "_entity_read_lease", None)
        if read_lease is not None:
            read_lease.close()
            self._entity_read_lease = None
        embedder = getattr(self, "embedder", None)
        if embedder is not None:
            embedder.close()
            self.embedder = None

    @classmethod
    def load_gbc_index(cls, config: SystemConfig):
        """
        Loads the GBC index from the specified path.

        :param config: The configuration object containing the save path.
        :return: An instance of GBC with the loaded index.
        """
        # The tree comes from the store, exactly as the graph and vector stages
        # read it. A pickle beside the run is a build artifact that is allowed
        # to disagree with the rows, and reading it here meant answering out of
        # a tree no stage had used.
        from docarag.vdb_tree_source import StoreTreeSource

        tree_index = StoreTreeSource.open(config.save_path)

        variant = graph_artifact_variant(config.graph)

        graph_index = Graph.load_from_dir(config.save_path, variant=variant)
        GBC = cls(
            config=config,
            graph_index=graph_index,
            TreeIndex=tree_index,
            require_entity_ready=True,
        )
        log.info(f"GBC index loaded from {config.save_path}")
        return GBC
