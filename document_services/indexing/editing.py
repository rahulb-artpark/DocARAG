"""Edit stored content directly, and re-derive exactly what the edit reached.

An edit is a store write, a reseal, and a reconcile. The write and the reseal
are the easy part; the reconcile is what keeps the document answerable, because
a resealed revision whose vectors still describe the old text fails closed at
query time with "TreeVDB attestation does not match the sealed store revision".

Only the points whose text actually changed are re-embedded. Which points those
are is not worked out here: a tree point's ID is content-addressed over every
vector's source, so re-projecting the tree and diffing IDs against Qdrant finds
them, including the ones an edit reached indirectly -- a section heading's
summary is quoted into the ``context`` vector of every figure and table beneath
it, and those points re-ID themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_services.persistence.document_store import scope_for_run
from document_services.persistence.postgres import PostgresLedger
from document_services.runtime.settings import load_settings


def _reconcile_run_vectors(run: Path) -> str:
    """Re-derive the run's tree vectors in place and republish its manifest."""

    # Imported here: this module is the thin edit surface, and the vector stack
    # pulls in model providers that a caller reading a summary should not pay
    # for.
    import yaml

    from retrieval_engine.configuration.system import SystemConfig
    from retrieval_engine.indexing.vector.builder import (
        expected_tree_vector_index_attestation,
        reconcile_tree_vector_index,
    )
    from retrieval_engine.indexes.vector_manifest import (
        publish_vector_index_attestation,
        read_vector_index_attestation,
        vector_index_lock,
    )
    from document_services.indexing.pipeline import republish_run_artifacts
    from document_services.persistence.vector_tree_source import StoreTreeSource

    from retrieval_engine.utils.artifact_paths import resolve_run_artifact_path

    config = SystemConfig(**yaml.safe_load((run / "run_config.yaml").read_text()))
    vdb_cfg = config.vdb
    # The config names the attestation directory relative to the run, and the
    # stage resolves it before use. Reading it raw finds nothing and reports a
    # missing index rather than the wrong path.
    directory = resolve_run_artifact_path(
        config.save_path, vdb_cfg.manifest_dir_name, label="vdb.manifest_dir_name"
    )
    vdb_cfg.manifest_dir_name = str(directory)
    with vector_index_lock(directory, exclusive=True):
        previous = read_vector_index_attestation(directory)
        if not isinstance(previous, dict) or not previous.get("lexical_average_length"):
            raise SystemExit(
                "error: no attested vector index to reconcile against; "
                "run `docarag stage vdb --run <run>` first"
            )
        frozen = float(previous["lexical_average_length"])
        tree = StoreTreeSource.open(str(run))
        outcome = reconcile_tree_vector_index(
            tree, vdb_cfg, frozen_average_length=frozen
        )
        expected = expected_tree_vector_index_attestation(
            tree, vdb_cfg, frozen_average_length=frozen
        )
        publish_vector_index_attestation(directory, expected)
    published = republish_run_artifacts(run)
    if published != 0:
        raise SystemExit(f"error: artifact republication failed ({published})")
    return outcome.describe() + (
        f"; BM25 length normalizer held at {outcome.average_length:.3f} "
        f"(corpus {outcome.corpus_average_length:.3f}, drift {outcome.drift:.1%})"
    )


def edit_node_summary(run: Path, node_no: int, text: str) -> None:
    scope, revision_id = scope_for_run(run)
    ledger = PostgresLedger(load_settings().resolve_database_url())
    revision_key = ledger.revision_key(scope, revision_id)
    with ledger._connection() as connection:
        blob_key = ledger._intern_text(connection, text)
        updated = connection.execute(
            "UPDATE docarag.tree_node SET summary_blob_key = %s "
            "WHERE revision_key = %s AND node_no = %s",
            (blob_key, revision_key, node_no),
        ).rowcount
        if updated != 1:
            raise SystemExit(f"error: revision has no tree node {node_no}")
        ledger._seal_revision(connection, revision_key)
        # Stale until the reconcile below says otherwise. If this process dies
        # in between, the store says the vectors describe the old text, which
        # is exactly what happened.
        connection.execute(
            "INSERT INTO docarag.stage_state (revision_key, stage, status) "
            "VALUES (%s, 'vdb', 'stale') "
            "ON CONFLICT (revision_key, stage) DO UPDATE SET status = 'stale'",
            (revision_key,),
        )
        seal = str(
            connection.execute(
                "SELECT tree_sha256 FROM docarag.revision_seal "
                "WHERE revision_key = %s",
                (revision_key,),
            ).fetchone()["tree_sha256"]
        )
    print(f"node {node_no} summary updated; revision resealed {seal[:16]}…")

    report = _reconcile_run_vectors(run)
    with ledger._connection() as connection:
        connection.execute(
            "INSERT INTO docarag.stage_state (revision_key, stage, status) "
            "VALUES (%s, 'vdb', 'succeeded') "
            "ON CONFLICT (revision_key, stage) DO UPDATE SET status = 'succeeded'",
            (revision_key,),
        )
    print(report)


#: The transformation a human correction records. Named like the refiner's own
#: (``bookrag.figure_merge.v1``) because it is the same kind of claim: this text
#: was produced from those canonical sources by that named process.
HUMAN_EDIT_TRANSFORMATION = "docarag.human_edit.v1"

#: Set on a node whose content came straight from one canonical block. A
#: corrected node no longer has that identity -- it has a lineage instead.
_DIRECT_EVIDENCE_FIELDS = (
    "canonical_block_id",
    "canonical_revision_id",
    "observation_id",
)


def _ordered_unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in result:
            result.append(value)
    return result


def _primary_block_type(run: Path, evidence: dict) -> str:
    """The node type the first source block projects to.

    The hydrator recomputes a derived digest with this rather than the node's
    own type, because the outline pass may have promoted the node to a title
    after the refiner sealed it. Reading it from the same place keeps the two
    from disagreeing.
    """

    from document_services.evidence.projections.bookrag import project_canonical_blocks
    from document_services.persistence.document_store import DocumentStore

    reference = (evidence.get("source_evidence_refs") or [{}])[0]
    store = DocumentStore.open(str(run))
    try:
        catalog = store.evidence_catalog()
        block = catalog.validate_pair(
            reference.get("canonical_block_id"),
            reference.get("canonical_revision_id"),
        )
        projected = project_canonical_blocks(
            catalog.document, (block,), validate_relation_targets=False
        )[0]
    finally:
        store.close()
    return str(projected.get("type") or "unknown")


def _derived_evidence_for_edit(run: Path, evidence: dict, text: str) -> dict:
    """The node's evidence after a human correction, sealed as derived.

    Canonical evidence says "this is what we read out of the document". A
    correction is not that, however much more accurate it is, so the node keeps
    its sources and gains a transformation naming what happened to them. The
    digest covers the new text *and* that lineage, so a later change to either
    is still detected.
    """

    from document_services.evidence.identity import (
        derived_evidence_id,
        derived_identity_payload,
        derived_output_identity,
    )

    references = evidence.get("source_evidence_refs") or []
    if not references:
        raise SystemExit(
            "error: this node has no canonical source references to derive from"
        )
    updated = dict(evidence)
    updated["content"] = text
    block_ids = _ordered_unique(
        reference.get("canonical_block_id") for reference in references
    )
    revision_ids = _ordered_unique(
        reference.get("canonical_revision_id") for reference in references
    )
    observation_ids = _ordered_unique(
        [
            *(evidence.get("source_observation_ids") or []),
            *(
                value
                for reference in references
                for value in [
                    *(reference.get("source_observation_ids") or []),
                    reference.get("observation_id"),
                ]
            ),
        ]
    )
    transformations = _ordered_unique(
        [*(evidence.get("transformations") or []), HUMAN_EDIT_TRANSFORMATION]
    )
    document_ids = _ordered_unique(
        [
            evidence.get("document_version_id"),
            *(reference.get("document_version_id") for reference in references),
        ]
    )
    digest = derived_evidence_id(
        derived_identity_payload(
            # Every field read exactly as the hydrator reads it back.
            output=derived_output_identity(
                content_type=_primary_block_type(run, evidence),
                text=text,
                table_body=updated.get("table_body", ""),
                table_caption=updated.get("caption", ""),
                table_footnote=updated.get("footnote", ""),
            ),
            source_revision_ids=revision_ids,
            source_document_version_ids=document_ids,
            source_block_ids=block_ids,
            source_observation_ids=observation_ids,
            source_references=references,
            transformations=transformations,
        )
    )
    for field in _DIRECT_EVIDENCE_FIELDS:
        updated.pop(field, None)
    updated["source_canonical_block_ids"] = block_ids
    updated["source_canonical_revision_ids"] = revision_ids
    updated["source_observation_ids"] = observation_ids
    updated["transformations"] = transformations
    updated["derived_evidence_id"] = digest
    return updated


def _hydrates(run: Path, node_no: int) -> str | None:
    """Return why the node would be refused in an answer, or None if it is fine."""

    from document_services.persistence.document_store import DocumentStore
    from document_services.evidence.hydration import (
        CanonicalEvidenceHydrator,
        EvidenceHydrationError,
    )
    from retrieval_engine.rag.stored_indexes import StoredTreeIndex, StoredTreeMeta

    store = DocumentStore.open(str(run))
    try:
        tree = StoredTreeIndex(store.tree, StoredTreeMeta.from_store(store))
        meta = tree.meta_info
        hydrator = CanonicalEvidenceHydrator(
            store.evidence_catalog(),
            node_resolver=lambda ids: tree.get_nodes_data(list(ids)),
            expected_document_version_id=meta.document_version_id,
            expected_source_sha256=meta.source_sha256,
            expected_source_uri=meta.source_uri,
        )
        try:
            hydrator.hydrate(tree.get_nodes_data([node_no]))
        except EvidenceHydrationError as exc:
            return str(exc)
        except Exception as exc:  # a projection error is a refusal too
            return f"{type(exc).__name__}: {exc}"
    finally:
        store.close()
    return None


def edit_node_content(run: Path, node_no: int, text: str) -> None:
    """Correct what a node says its source shows, recorded as a human edit.

    The summary is a gloss and nothing seals it. Content is the observation the
    parser and vision model produced, and an answer citing it claims it was not
    altered -- so a correction cannot simply overwrite it. It becomes derived
    evidence instead: same sources, one more transformation, a digest over the
    new text and that lineage. The answer then uses your words and the record
    says who wrote them.

    Verified after writing by hydrating the node exactly as an answer would,
    and reverted if that refuses -- a node left unciteable is worse than an
    uncorrected description.
    """

    scope, revision_id = scope_for_run(run)
    ledger = PostgresLedger(load_settings().resolve_database_url())
    revision_key = ledger.revision_key(scope, revision_id)
    with ledger._connection() as connection:
        row = connection.execute(
            "SELECT attrs FROM docarag.tree_node "
            "WHERE revision_key = %s AND node_no = %s",
            (revision_key, node_no),
        ).fetchone()
        if row is None:
            raise SystemExit(f"error: revision has no tree node {node_no}")
        attrs = dict(row["attrs"] or {})
        previous = dict(attrs.get("evidence") or {})
        if not previous:
            raise SystemExit(f"error: tree node {node_no} carries no evidence")
        if previous.get("content", "") == text:
            raise SystemExit("error: the content is already exactly that")
        attrs["evidence"] = _derived_evidence_for_edit(run, previous, text)
        connection.execute(
            "UPDATE docarag.tree_node SET attrs = %s::jsonb "
            "WHERE revision_key = %s AND node_no = %s",
            (json.dumps(attrs, ensure_ascii=False), revision_key, node_no),
        )
        ledger._seal_revision(connection, revision_key)
        connection.execute(
            "INSERT INTO docarag.stage_state (revision_key, stage, status) "
            "VALUES (%s, 'vdb', 'stale') "
            "ON CONFLICT (revision_key, stage) DO UPDATE SET status = 'stale'",
            (revision_key,),
        )

    refusal = _hydrates(run, node_no)
    if refusal is not None:
        with ledger._connection() as connection:
            restored = dict(attrs)
            restored["evidence"] = previous
            connection.execute(
                "UPDATE docarag.tree_node SET attrs = %s::jsonb "
                "WHERE revision_key = %s AND node_no = %s",
                (json.dumps(restored, ensure_ascii=False), revision_key, node_no),
            )
            ledger._seal_revision(connection, revision_key)
        raise SystemExit(
            f"error: the edited node would be refused in an answer ({refusal}); "
            "the previous content has been restored"
        )

    print(
        f"node {node_no} content updated as derived evidence "
        f"{attrs['evidence']['derived_evidence_id'][:24]}…"
    )
    print(_reconcile_run_vectors(run))


def _reindex_run_entities(run: Path, node_names: list[str]) -> str:
    """Re-embed the named entities for one revision and republish."""

    import yaml

    from retrieval_engine.configuration.knowledge_graph import graph_artifact_variant
    from retrieval_engine.configuration.system import SystemConfig
    from retrieval_engine.indexes.gbc import GBC
    from retrieval_engine.indexes.graph import Graph
    from document_services.indexing.pipeline import republish_run_artifacts
    from document_services.persistence.vector_tree_source import StoreTreeSource

    config = SystemConfig(**yaml.safe_load((run / "run_config.yaml").read_text()))
    tree = StoreTreeSource.open(str(run))
    graph = Graph.load_from_dir(
        config.save_path, variant=graph_artifact_variant(config.graph)
    )
    index = GBC(
        config=config, graph_index=graph, TreeIndex=tree, require_entity_ready=False
    )
    try:
        written = index.reindex_entities(node_names)
    finally:
        index.close()
    published = republish_run_artifacts(run)
    if published != 0:
        raise SystemExit(f"error: artifact republication failed ({published})")
    return f"{written} entity point(s) re-embedded in place"


def edit_entity_description(run: Path, canonical: str, kind: str, text: str) -> None:
    """Rewrite what one revision says an entity is.

    Scoped to this revision on purpose. ``docarag.entity`` is corpus-wide, so
    writing the description there would change every document that mentions the
    entity -- 730 of one paper's 862 entities are mentioned by another revision.
    The per-revision row keeps the edit where it was made.
    """

    from retrieval_engine.rag.stored_indexes import entity_node_name

    scope, revision_id = scope_for_run(run)
    ledger = PostgresLedger(load_settings().resolve_database_url())
    revision_key = ledger.revision_key(scope, revision_id)
    with ledger._connection() as connection:
        row = connection.execute(
            "SELECT e.entity_key FROM docarag.entity e "
            "WHERE e.tenant_id = %s AND e.corpus_id = %s "
            "AND e.entity_kind = %s AND e.canonical = %s",
            (scope.tenant_id, scope.corpus_id, kind, canonical),
        ).fetchone()
        if row is None:
            raise SystemExit(f"error: this corpus has no {kind} entity {canonical!r}")
        entity_key = int(row["entity_key"])
        mentioned = connection.execute(
            "SELECT 1 FROM docarag.entity_mention "
            "WHERE revision_key = %s AND entity_key = %s "
            "UNION SELECT 1 FROM docarag.statement_role "
            "WHERE revision_key = %s AND entity_key = %s LIMIT 1",
            (revision_key, entity_key, revision_key, entity_key),
        ).fetchone()
        if mentioned is None:
            raise SystemExit(
                f"error: this revision does not mention {canonical!r}"
            )
        blob_key = ledger._intern_text(connection, text)
        connection.execute(
            "INSERT INTO docarag.entity_description "
            "(revision_key, entity_key, description_blob_key) VALUES (%s, %s, %s) "
            "ON CONFLICT (revision_key, entity_key) DO UPDATE SET "
            "description_blob_key = EXCLUDED.description_blob_key",
            (revision_key, entity_key, blob_key),
        )
        ledger._seal_graph(connection, revision_key)
        seal = str(
            connection.execute(
                "SELECT graph_sha256 FROM docarag.revision_seal "
                "WHERE revision_key = %s",
                (revision_key,),
            ).fetchone()["graph_sha256"]
        )
    print(f"{canonical!r} description updated; graph resealed {seal[:16]}…")
    print(_reindex_run_entities(run, [entity_node_name(canonical, kind)]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edit stored content in place")
    sub = parser.add_subparsers(dest="what", required=True)
    node = sub.add_parser("summary", help="Replace one tree node's summary")
    node.add_argument("--run", type=Path, required=True)
    node.add_argument("--node", type=int, required=True)
    node.add_argument("--text", required=True)
    entity = sub.add_parser(
        "entity", help="Replace one entity's description for this revision"
    )
    entity.add_argument("--run", type=Path, required=True)
    entity.add_argument("--canonical", required=True)
    entity.add_argument("--kind", required=True)
    entity.add_argument("--text", required=True)
    content = sub.add_parser(
        "content",
        help="Correct one node's content, recorded as a human-edit derivation",
    )
    content.add_argument("--run", type=Path, required=True)
    content.add_argument("--node", type=int, required=True)
    content.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    if args.what == "entity":
        edit_entity_description(args.run, args.canonical, args.kind, args.text)
    elif args.what == "content":
        edit_node_content(args.run, args.node, args.text)
    else:
        edit_node_summary(args.run, args.node, args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
