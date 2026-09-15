"""Read cross-document state -- groupings and cross-docs relations -- across a
corpus, not one revision.

Mirrors ``TreeStore``'s shape: a connection factory plus an identity, nothing
eager, one indexed query per method. Scoped by ``(tenant_id, corpus_id)``
instead of one ``revision_key``, since a grouping or a cross-docs relation
belongs to the corpus, not to any single document's revision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docarag.tree_store import GraphEdge, MAX_SUBGRAPH_EDGES, _Borrowed

GROUPING_MEMBER_PREDICATE = "grouping_member"

# Mirrors Core.rag.stored_index.entity_node_name/split_node_name -- duplicated
# rather than imported so this module (used at ingest time) never pulls in
# the query-time RAG package and its VLM/reranker dependencies.


def entity_node_name(entity_name: str, entity_type: str) -> str:
    return f"Name: {entity_name}\nType: {entity_type}"


def split_node_name(node_name: str) -> tuple[str, str | None]:
    text = str(node_name or "")
    if text.startswith("Name: ") and "\nType: " in text:
        name, _, kind = text[len("Name: ") :].partition("\nType: ")
        return name.strip(), kind.strip() or None
    return text.strip(), None


@dataclass(frozen=True, slots=True)
class EntityView:
    entity_key: int
    canonical: str
    entity_kind: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class GroupingMembership:
    grouping_key: int
    label: str
    description: str
    centroid: tuple[float, ...]
    member_count: int


@dataclass(frozen=True, slots=True)
class LinkStats:
    """Cross-doc link counts for one revision's own entities."""

    nodes_connected: int
    grouping_hubs: int
    cross_edges: int


class CorpusStore:
    """Bounded, corpus-scoped reads over entities, groupings and cross-docs edges."""

    __slots__ = ("_connect", "_tenant_id", "_corpus_id")

    def __init__(self, connect: Callable[[], Any], tenant_id: str, corpus_id: str) -> None:
        if not callable(connect):
            raise TypeError("connect must be callable")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise ValueError("corpus_id must be a non-empty string")
        self._connect = connect
        self._tenant_id = tenant_id
        self._corpus_id = corpus_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def corpus_id(self) -> str:
        return self._corpus_id

    # -- candidate generation ------------------------------------------------

    def entities_for_revision(self, revision_key: int) -> list[EntityView]:
        """This revision's own entities, with its own description if it wrote one."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT DISTINCT ent.entity_key, ent.canonical, ent.entity_kind, "
                "COALESCE(rev_desc.body, corpus_desc.body, '') AS description "
                "FROM docarag.entity_mention mention "
                "JOIN docarag.entity ent ON ent.entity_key = mention.entity_key "
                "LEFT JOIN docarag.entity_description ed "
                "  ON ed.revision_key = mention.revision_key AND ed.entity_key = ent.entity_key "
                "LEFT JOIN docarag.text_blob rev_desc ON rev_desc.blob_key = ed.description_blob_key "
                "LEFT JOIN docarag.text_blob corpus_desc ON corpus_desc.blob_key = ent.description_blob_key "
                "WHERE mention.revision_key = %s AND ent.tenant_id = %s AND ent.corpus_id = %s",
                (revision_key, self._tenant_id, self._corpus_id),
            ).fetchall()
        return [_entity_view(row) for row in rows]

    def entity_key_for_canonical(self, canonical: str, entity_kind: str) -> int | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT entity_key FROM docarag.entity "
                "WHERE tenant_id = %s AND corpus_id = %s AND entity_kind = %s AND canonical = %s",
                (self._tenant_id, self._corpus_id, entity_kind, canonical),
            ).fetchone()
        return int(row["entity_key"]) if row else None

    def document_version_id_for_revision(self, revision_key: int) -> str | None:
        """The document_version_id one revision belongs to -- the same value
        entity points are written to Qdrant under (entity_vector_scope), so a
        caller can turn a revision_key into a vector-search filter."""

        with self._session() as connection:
            row = connection.execute(
                "SELECT document_version_id FROM docarag.index_revisions "
                "WHERE revision_key = %s AND tenant_id = %s AND corpus_id = %s",
                (revision_key, self._tenant_id, self._corpus_id),
            ).fetchone()
        return str(row["document_version_id"]) if row else None

    def document_versions_for_entity(self, entity_key: int) -> frozenset[str]:
        """Every document_version_id mentioning this entity (not revision_key)."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT DISTINCT rev.document_version_id "
                "FROM docarag.entity_mention mention "
                "JOIN docarag.index_revisions rev ON rev.revision_key = mention.revision_key "
                "WHERE mention.entity_key = %s AND rev.tenant_id = %s AND rev.corpus_id = %s",
                (entity_key, self._tenant_id, self._corpus_id),
            ).fetchall()
        return frozenset(str(row["document_version_id"]) for row in rows)

    def revisions_for_entity(self, entity_key: int) -> frozenset[int]:
        """Every revision that mentions this entity."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT DISTINCT revision_key FROM docarag.entity_mention WHERE entity_key = %s",
                (entity_key,),
            ).fetchall()
        return frozenset(int(row["revision_key"]) for row in rows)

    def entity_text(self, entity_key: int) -> tuple[str, str]:
        """(canonical, description) for entity_key."""

        with self._session() as connection:
            row = connection.execute(
                "SELECT ent.canonical, COALESCE(blob.body, '') AS description "
                "FROM docarag.entity ent "
                "LEFT JOIN docarag.text_blob blob ON blob.blob_key = ent.description_blob_key "
                "WHERE ent.entity_key = %s AND ent.tenant_id = %s AND ent.corpus_id = %s",
                (entity_key, self._tenant_id, self._corpus_id),
            ).fetchone()
        if row is None:
            return "", ""
        return str(row["canonical"]), str(row["description"] or "")

    # -- grouping state -------------------------------------------------------

    def groupings_for_entity(self, entity_key: int) -> list[GroupingMembership]:
        """Every grouping this entity currently belongs to."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT g.entity_key AS grouping_key, g.attrs, "
                "COALESCE(blob.body, '') AS description "
                "FROM docarag.cross_doc_statement_role member_role "
                "JOIN docarag.cross_doc_statement stmt "
                "  ON stmt.tenant_id = member_role.tenant_id AND stmt.corpus_id = member_role.corpus_id "
                " AND stmt.stmt_key = member_role.stmt_key "
                "JOIN docarag.cross_doc_statement_role grouping_role "
                "  ON grouping_role.tenant_id = member_role.tenant_id "
                " AND grouping_role.corpus_id = member_role.corpus_id "
                " AND grouping_role.stmt_key = member_role.stmt_key "
                " AND grouping_role.role = 'grouping' "
                "JOIN docarag.entity g ON g.entity_key = grouping_role.entity_key "
                "LEFT JOIN docarag.text_blob blob ON blob.blob_key = g.description_blob_key "
                "WHERE member_role.tenant_id = %s AND member_role.corpus_id = %s "
                "  AND member_role.role = 'member' AND member_role.entity_key = %s "
                "  AND stmt.predicate = %s",
                (self._tenant_id, self._corpus_id, entity_key, GROUPING_MEMBER_PREDICATE),
            ).fetchall()
        return [_grouping_membership(row) for row in rows]

    def all_groupings(self) -> list[GroupingMembership]:
        """Every grouping in this corpus, for a create-vs-reuse centroid check."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT g.entity_key AS grouping_key, g.attrs, "
                "COALESCE(blob.body, '') AS description "
                "FROM docarag.entity g "
                "LEFT JOIN docarag.text_blob blob ON blob.blob_key = g.description_blob_key "
                "WHERE g.tenant_id = %s AND g.corpus_id = %s AND g.entity_kind = 'grouping'",
                (self._tenant_id, self._corpus_id),
            ).fetchall()
        return [_grouping_membership(row) for row in rows]

    def members_for_grouping(self, grouping_key: int) -> list[EntityView]:
        """Every entity currently attached to this grouping."""

        with self._session() as connection:
            rows = connection.execute(
                "SELECT m.entity_key, m.canonical, m.entity_kind, "
                "COALESCE(blob.body, '') AS description "
                "FROM docarag.cross_doc_statement_role grouping_role "
                "JOIN docarag.cross_doc_statement stmt "
                "  ON stmt.tenant_id = grouping_role.tenant_id AND stmt.corpus_id = grouping_role.corpus_id "
                " AND stmt.stmt_key = grouping_role.stmt_key "
                "JOIN docarag.cross_doc_statement_role member_role "
                "  ON member_role.tenant_id = grouping_role.tenant_id "
                " AND member_role.corpus_id = grouping_role.corpus_id "
                " AND member_role.stmt_key = grouping_role.stmt_key "
                " AND member_role.role = 'member' "
                "JOIN docarag.entity m ON m.entity_key = member_role.entity_key "
                "LEFT JOIN docarag.text_blob blob ON blob.blob_key = m.description_blob_key "
                "WHERE grouping_role.tenant_id = %s AND grouping_role.corpus_id = %s "
                "  AND grouping_role.role = 'grouping' AND grouping_role.entity_key = %s "
                "  AND stmt.predicate = %s",
                (self._tenant_id, self._corpus_id, grouping_key, GROUPING_MEMBER_PREDICATE),
            ).fetchall()
        return [_entity_view(row) for row in rows]

    def grouping_membership_stmt_key(self, grouping_key: int, member_key: int) -> int | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT grouping_role.stmt_key "
                "FROM docarag.cross_doc_statement_role grouping_role "
                "JOIN docarag.cross_doc_statement_role member_role "
                "  ON member_role.tenant_id = grouping_role.tenant_id "
                " AND member_role.corpus_id = grouping_role.corpus_id "
                " AND member_role.stmt_key = grouping_role.stmt_key "
                "JOIN docarag.cross_doc_statement stmt "
                "  ON stmt.tenant_id = grouping_role.tenant_id AND stmt.corpus_id = grouping_role.corpus_id "
                " AND stmt.stmt_key = grouping_role.stmt_key "
                "WHERE grouping_role.tenant_id = %s AND grouping_role.corpus_id = %s "
                "  AND grouping_role.role = 'grouping' AND grouping_role.entity_key = %s "
                "  AND member_role.role = 'member' AND member_role.entity_key = %s "
                "  AND stmt.predicate = %s",
                (
                    self._tenant_id, self._corpus_id, grouping_key, member_key,
                    GROUPING_MEMBER_PREDICATE,
                ),
            ).fetchone()
        return int(row["stmt_key"]) if row else None

    # -- relation state -------------------------------------------------------

    def cross_doc_relation_stmt_key(
        self, subject_key: int, object_key: int, predicate: str
    ) -> int | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT subj.stmt_key "
                "FROM docarag.cross_doc_statement_role subj "
                "JOIN docarag.cross_doc_statement_role obj "
                "  ON obj.tenant_id = subj.tenant_id AND obj.corpus_id = subj.corpus_id "
                " AND obj.stmt_key = subj.stmt_key "
                "JOIN docarag.cross_doc_statement stmt "
                "  ON stmt.tenant_id = subj.tenant_id AND stmt.corpus_id = subj.corpus_id "
                " AND stmt.stmt_key = subj.stmt_key "
                "WHERE subj.tenant_id = %s AND subj.corpus_id = %s "
                "  AND subj.role = 'subject' AND subj.entity_key = %s "
                "  AND obj.role = 'object' AND obj.entity_key = %s "
                "  AND stmt.predicate = %s",
                (self._tenant_id, self._corpus_id, subject_key, object_key, predicate),
            ).fetchone()
        return int(row["stmt_key"]) if row else None

    def inferred_relations_for_entities(
        self, entity_keys: Iterable[int], *, max_edges: int = MAX_SUBGRAPH_EDGES,
        document_version_ids: Iterable[str] | None = None,
    ) -> list[GraphEdge]:
        """Cross-docs relations with both endpoints in ``entity_keys``.

        When a document allowlist is supplied, both recorded source documents
        must belong to it. Shared entity identity alone is not provenance.
        Relations without recorded source documents are excluded in this mode.
        """

        wanted = list({int(key) for key in entity_keys if not isinstance(key, bool)})
        if not wanted:
            return []
        scoped_documents = sorted(set(document_version_ids)) if document_version_ids is not None else None
        if scoped_documents == []:
            return []
        scope_sql = ""
        scope_params = ()
        if scoped_documents is not None:
            for role in ("subject", "object"):
                scope_sql += (
                    " AND EXISTS (SELECT 1 FROM docarag.cross_doc_statement_property provenance"
                    " WHERE provenance.tenant_id = stmt.tenant_id"
                    " AND provenance.corpus_id = stmt.corpus_id"
                    " AND provenance.stmt_key = stmt.stmt_key"
                    " AND provenance.name = %s AND provenance.text_val = ANY(%s))"
                )
                scope_params += (f"source_document_version_id_{role}", scoped_documents)
        with self._session() as connection:
            rows = connection.execute(
                "SELECT subj_ent.canonical AS src, subj_ent.entity_kind AS src_kind, "
                "obj_ent.canonical AS dst, obj_ent.entity_kind AS dst_kind, "
                "stmt.predicate, stmt.weight, COALESCE(blob.body, '') AS description, "
                "origin.text_val AS link_origin "
                "FROM docarag.cross_doc_statement stmt "
                "JOIN docarag.cross_doc_statement_role subj_role "
                "  ON subj_role.tenant_id = stmt.tenant_id AND subj_role.corpus_id = stmt.corpus_id "
                " AND subj_role.stmt_key = stmt.stmt_key AND subj_role.role = 'subject' "
                "JOIN docarag.cross_doc_statement_role obj_role "
                "  ON obj_role.tenant_id = stmt.tenant_id AND obj_role.corpus_id = stmt.corpus_id "
                " AND obj_role.stmt_key = stmt.stmt_key AND obj_role.role = 'object' "
                "JOIN docarag.entity subj_ent ON subj_ent.entity_key = subj_role.entity_key "
                "JOIN docarag.entity obj_ent ON obj_ent.entity_key = obj_role.entity_key "
                "LEFT JOIN docarag.text_blob blob ON blob.blob_key = stmt.description_blob_key "
                "LEFT JOIN docarag.cross_doc_statement_property origin "
                "  ON origin.tenant_id = stmt.tenant_id AND origin.corpus_id = stmt.corpus_id "
                " AND origin.stmt_key = stmt.stmt_key AND origin.name = 'link_origin' "
                "WHERE stmt.tenant_id = %s AND stmt.corpus_id = %s "
                "  AND stmt.predicate != %s "
                "  AND subj_role.entity_key = ANY(%s) AND obj_role.entity_key = ANY(%s) "
                + scope_sql + " LIMIT %s",
                (
                    self._tenant_id, self._corpus_id, GROUPING_MEMBER_PREDICATE,
                    wanted, wanted, *scope_params, max_edges,
                ),
            ).fetchall()
        return [
            GraphEdge(
                source=str(row["src"]),
                target=str(row["dst"]),
                predicate=str(row["predicate"]),
                weight=row["weight"],
                description=str(row["description"] or ""),
                link_origin=str(row["link_origin"]) if row["link_origin"] else None,
                source_kind=str(row["src_kind"] or ""),
                target_kind=str(row["dst_kind"] or ""),
            )
            for row in rows
        ]

    def linked_revisions_for_revision(self, revision_key: int) -> frozenset[int]:
        """Other revisions this one is directly, provably cross-linked to.

        Reads each relation's recorded provenance -- which document actually
        anchored each side when the relation was written (see
        docarag/cross_doc_linker.py's CrossDocCandidate and
        PostgresLedger.write_cross_doc_relation's source_document_version_id_*
        properties) -- instead of walking raw entity-key reachability. Two
        documents can end up sharing an entity_key through entity resolution
        (a separate, per-document dedup process, unrelated to cross-linking)
        at any time, before or after a relation involving that entity is
        written; without provenance, that alone made every OTHER relation the
        shared entity took part in look like it involved this revision, even
        for documents the cross-linking that wrote it never compared against.

        Returns nothing for a relation written before provenance tracking
        existed (no property recorded) -- deliberately no fallback to the
        old, imprecise method, which is exactly what produced the wrong
        answer this replaces.
        """

        my_docv = self.document_version_id_for_revision(revision_key)
        if not my_docv:
            return frozenset()

        with self._session() as connection:
            rows = connection.execute(
                "SELECT subj.text_val AS subject_docv, obj.text_val AS object_docv "
                "FROM docarag.cross_doc_statement stmt "
                "JOIN docarag.cross_doc_statement_property subj "
                "  ON subj.tenant_id = stmt.tenant_id AND subj.corpus_id = stmt.corpus_id "
                " AND subj.stmt_key = stmt.stmt_key "
                " AND subj.name = 'source_document_version_id_subject' "
                "JOIN docarag.cross_doc_statement_property obj "
                "  ON obj.tenant_id = stmt.tenant_id AND obj.corpus_id = stmt.corpus_id "
                " AND obj.stmt_key = stmt.stmt_key "
                " AND obj.name = 'source_document_version_id_object' "
                "WHERE stmt.tenant_id = %s AND stmt.corpus_id = %s "
                "  AND (subj.text_val = %s OR obj.text_val = %s)",
                (self._tenant_id, self._corpus_id, my_docv, my_docv),
            ).fetchall()

        other_docvs: set[str] = set()
        for row in rows:
            other = row["object_docv"] if row["subject_docv"] == my_docv else row["subject_docv"]
            if other and other != my_docv:
                other_docvs.add(str(other))
        if not other_docvs:
            return frozenset()

        with self._session() as connection:
            neighbor_rows = connection.execute(
                "SELECT DISTINCT ON (document_version_id) document_version_id, revision_key "
                "FROM docarag.index_revisions "
                "WHERE tenant_id = %s AND corpus_id = %s AND document_version_id = ANY(%s) "
                "ORDER BY document_version_id, revision_key DESC",
                (self._tenant_id, self._corpus_id, list(other_docvs)),
            ).fetchall()
        return frozenset(int(row["revision_key"]) for row in neighbor_rows)

    def link_stats_for_revision(self, revision_key: int) -> LinkStats:
        """How this revision's own entities currently sit in the cross-docs
        mesh: how many of them participate in any cross-doc statement, how
        many grouping hubs they attach to, and how many named cross-document
        relations touch them."""

        with self._session() as connection:
            row = connection.execute(
                "WITH doc_entities AS ( "
                "  SELECT DISTINCT entity_key FROM docarag.entity_mention "
                "  WHERE revision_key = %s "
                "), roles AS ( "
                "  SELECT r.stmt_key, r.entity_key, stmt.predicate "
                "  FROM docarag.cross_doc_statement_role r "
                "  JOIN docarag.cross_doc_statement stmt "
                "    ON stmt.tenant_id = r.tenant_id AND stmt.corpus_id = r.corpus_id "
                "   AND stmt.stmt_key = r.stmt_key "
                "  WHERE r.tenant_id = %s AND r.corpus_id = %s "
                "    AND r.role IN ('member', 'subject', 'object') "
                "    AND r.entity_key IN (SELECT entity_key FROM doc_entities) "
                ") "
                "SELECT "
                "  (SELECT count(DISTINCT entity_key) FROM roles) AS nodes_connected, "
                "  (SELECT count(DISTINCT stmt_key) FROM roles WHERE predicate = %s) AS grouping_hubs, "
                "  (SELECT count(DISTINCT stmt_key) FROM roles WHERE predicate != %s) AS cross_edges",
                (
                    revision_key, self._tenant_id, self._corpus_id,
                    GROUPING_MEMBER_PREDICATE, GROUPING_MEMBER_PREDICATE,
                ),
            ).fetchone()
        return LinkStats(
            nodes_connected=int(row["nodes_connected"] or 0),
            grouping_hubs=int(row["grouping_hubs"] or 0),
            cross_edges=int(row["cross_edges"] or 0),
        )

    # -- plumbing -------------------------------------------------------------

    def _session(self):
        return _Borrowed(self._connect())


def _entity_view(row: Mapping[str, Any]) -> EntityView:
    return EntityView(
        entity_key=int(row["entity_key"]),
        canonical=str(row["canonical"]),
        entity_kind=str(row["entity_kind"]),
        description=str(row["description"] or ""),
    )


def _grouping_membership(row: Mapping[str, Any]) -> GroupingMembership:
    attrs = row["attrs"] or {}
    centroid = tuple(float(v) for v in attrs.get("centroid", ()))
    return GroupingMembership(
        grouping_key=int(row["grouping_key"]),
        label=str(attrs.get("label", "")),
        description=str(row["description"] or ""),
        centroid=centroid,
        member_count=int(attrs.get("member_count", 0)),
    )
