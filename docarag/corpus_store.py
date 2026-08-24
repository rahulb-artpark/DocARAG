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
    ) -> list[GraphEdge]:
        """Cross-docs relations with both endpoints in ``entity_keys``.

        Not subtree-scoped -- a cross-docs edge has an endpoint outside any
        one document's subtree by definition.
        """

        wanted = list({int(key) for key in entity_keys if not isinstance(key, bool)})
        if not wanted:
            return []
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
                "LIMIT %s",
                (
                    self._tenant_id, self._corpus_id, GROUPING_MEMBER_PREDICATE,
                    wanted, wanted, max_edges,
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
