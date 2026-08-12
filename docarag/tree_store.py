"""Read one revision's tree and graph from the store, a slice at a time.

Retrieval works on a slice of a document: the nodes mentioning a query's
entities, the subtree beneath a chosen section, and the graph those nodes
induce.  Every read here is bounded by that slice, so a query's memory follows
the question rather than the size of the corpus.

Nothing is loaded eagerly.  The accessor holds a revision key and a connection
factory; each method is one indexed query.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

MAX_NODES_PER_READ = 4096
MAX_SUBGRAPH_EDGES = 8192
MAX_GRAPH_ENTITIES = 8192


@dataclass(frozen=True, slots=True)
class NodeMeta:
    """The fields a retrieved node carries about the block it was built from."""

    content: str = ""
    page_idx: int | None = None
    img_path: str | None = None
    caption: str = ""
    footnote: str = ""
    table_body: str = ""
    block_uid: str | None = None
    block_revision_uid: str | None = None
    page_uid: str | None = None
    geometry: Mapping[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "page_idx": self.page_idx,
            "img_path": self.img_path,
            "caption": self.caption,
            "footnote": self.footnote,
            "table_body": self.table_body,
            "block_uid": self.block_uid,
            "block_revision_uid": self.block_revision_uid,
            "page_uid": self.page_uid,
            "geometry": self.geometry,
        }


@dataclass(frozen=True, slots=True)
class TreeNodeView:
    """One node as retrieval sees it, without loading the document around it."""

    index_id: int
    type: str
    depth: int
    meta_info: NodeMeta
    evidence: Mapping[str, Any] | None = None
    parent_id: int | None = None
    summary: str = ""
    outline_node: bool = False
    children: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    predicate: str
    weight: float | None = None
    description: str = ""


class TreeStore:
    """Bounded reads over one sealed revision."""

    __slots__ = ("_connect", "_revision_key")

    def __init__(self, connect: Callable[[], Any], revision_key: int) -> None:
        if not callable(connect):
            raise TypeError("connect must be callable")
        if isinstance(revision_key, bool) or not isinstance(revision_key, int):
            raise TypeError("revision_key must be an integer")
        self._connect = connect
        self._revision_key = revision_key

    @property
    def revision_key(self) -> int:
        return self._revision_key

    # -- nodes --------------------------------------------------------------

    def nodes(self, node_ids: Sequence[int] | None = None) -> list[TreeNodeView]:
        """Return whole nodes by number, or every node when none is named."""

        if node_ids is None:
            return self._query(_NODES_SQL, (self._revision_key,))
        wanted = _bounded_ids(node_ids)
        if not wanted:
            return []
        return self._query(_NODES_BY_NO_SQL, (self._revision_key, wanted))

    def node(self, node_id: int) -> TreeNodeView | None:
        found = self.nodes([node_id])
        return found[0] if found else None

    def root(self) -> TreeNodeView | None:
        found = self._query(_ROOT_SQL, (self._revision_key,))
        return found[0] if found else None

    def children(self, node_id: int) -> list[TreeNodeView]:
        return self._query(_CHILDREN_SQL, (self._revision_key, node_id))

    def siblings(self, node_id: int) -> list[TreeNodeView]:
        return self._query(_SIBLINGS_SQL, (self._revision_key, node_id, node_id))

    def path_from_root(self, node_id: int) -> list[TreeNodeView]:
        """Return the ancestors of a node, root first, in one query."""

        return self._query(_ANCESTORS_SQL, (self._revision_key, node_id))

    def subtree(self, node_ids: Sequence[int] | int) -> list[TreeNodeView]:
        """Return every node beneath the given sections, including them."""

        wanted = _bounded_ids([node_ids] if isinstance(node_ids, int) else node_ids)
        if not wanted:
            return []
        return self._query(_SUBTREE_SQL, (self._revision_key, wanted))

    def nodes_at_depth(self, depth: int) -> list[TreeNodeView]:
        return self._query(_AT_DEPTH_SQL, (self._revision_key, depth))

    def nodes_of_kind(self, node_kind: str) -> list[TreeNodeView]:
        return self._query(_BY_KIND_SQL, (self._revision_key, str(node_kind)))

    def ancestor_at_depth(self, node_id: int, depth: int) -> TreeNodeView | None:
        for node in self.path_from_root(node_id):
            if node.depth == depth:
                return node
        return None

    def max_depth(self) -> int:
        row = self._one(
            "SELECT coalesce(max(depth), 0) AS depth FROM docarag.tree_node "
            "WHERE revision_key = %s",
            (self._revision_key,),
        )
        return int(row["depth"]) if row else 0

    def retrieval_texts(self, node_ids: Sequence[int]) -> dict[int, str]:
        """Return one string per node for reranking: the hottest read there is."""

        wanted = _bounded_ids(node_ids)
        if not wanted:
            return {}
        with self._session() as connection:
            rows = connection.execute(
                "SELECT node.node_no, feed.retrieval_text "
                "FROM docarag.node_retrieval feed "
                "JOIN docarag.tree_node node "
                "  ON node.revision_key = feed.revision_key "
                " AND node.node_key = feed.node_key "
                "WHERE feed.revision_key = %s AND node.node_no = ANY(%s)",
                (self._revision_key, wanted),
            ).fetchall()
        return {int(row["node_no"]): str(row["retrieval_text"]) for row in rows}

    # -- graph --------------------------------------------------------------

    def nodes_for_entity(self, canonical: str, entity_kind: str | None = None) -> list[int]:
        """Return the nodes mentioning an entity: retrieval's way into the tree."""

        with self._session() as connection:
            rows = connection.execute(
                _NODES_FOR_ENTITY_SQL,
                (self._revision_key, str(canonical), entity_kind, entity_kind),
            ).fetchall()
        return [int(row["node_no"]) for row in rows]

    def induced_subgraph(self, node_ids: Sequence[int]) -> list[GraphEdge]:
        """Return the statements the given nodes support, as binary edges."""

        wanted = _bounded_ids(node_ids)
        if not wanted:
            return []
        with self._session() as connection:
            rows = connection.execute(
                _SUBGRAPH_SQL, (self._revision_key, wanted, MAX_SUBGRAPH_EDGES)
            ).fetchall()
        return [
            GraphEdge(
                source=str(row["src"]),
                target=str(row["dst"]),
                predicate=str(row["predicate"]),
                weight=row["weight"],
                description=str(row["description"] or ""),
            )
            for row in rows
        ]

    def structures_for_nodes(
        self, node_ids: Sequence[int]
    ) -> dict[int, list[dict[str, Any]]]:
        """Return each node's extracted structures: elements and directed edges."""

        wanted = _bounded_ids(node_ids)
        if not wanted:
            return {}
        with self._session() as connection:
            element_rows = connection.execute(
                _STRUCTURE_ELEMENTS_SQL, (self._revision_key, wanted)
            ).fetchall()
            edge_rows = connection.execute(
                _STRUCTURE_EDGES_SQL, (self._revision_key, wanted)
            ).fetchall()

        structures: dict[int, dict[str, Any]] = {}
        order: dict[int, list[int]] = {}
        for row in element_rows:
            key = int(row["structure_key"])
            entry = structures.get(key)
            if entry is None:
                entry = structures[key] = {
                    "kind": str(row["structure_kind"]),
                    "elements": [],
                    "edges": [],
                }
                order.setdefault(int(row["node_no"]), []).append(key)
            entry["elements"].append(
                {
                    "content": str(row["content"] or ""),
                    "role": str(row["role"] or ""),
                    "kind": str(row["element_kind"]),
                }
            )
        for row in edge_rows:
            entry = structures.get(int(row["structure_key"]))
            if entry is not None:
                entry["edges"].append(
                    {
                        "source": str(row["src_content"] or ""),
                        "target": str(row["dst_content"] or ""),
                        "kind": str(row["edge_kind"]),
                        "label": str(row["label"] or ""),
                    }
                )
        return {
            node_no: [structures[key] for key in keys]
            for node_no, keys in order.items()
        }

    def graph_entities(self, limit: int = MAX_GRAPH_ENTITIES) -> list[dict[str, Any]]:
        """Return every entity this revision mentions, with its evidence nodes."""

        with self._session() as connection:
            rows = connection.execute(
                _GRAPH_ENTITIES_SQL,
                (
                    self._revision_key,  # mentions
                    self._revision_key,  # statement roles
                    self._revision_key,  # this revision's own descriptions
                    self._revision_key,  # evidence nodes
                    limit,
                ),
            ).fetchall()
        return [
            {
                "canonical": str(row["canonical"]),
                "entity_kind": str(row["entity_kind"]),
                "description": str(row["description"] or ""),
                "node_ids": sorted(int(n) for n in (row["node_nos"] or ())),
            }
            for row in rows
        ]

    def iter_graph_entities(self, page_size: int = 2048) -> Iterator[dict[str, Any]]:
        """Every entity this revision mentions, in identity order, unbounded.

        ``graph_entities`` caps itself for display; an index built from a cap
        answers nothing about the entities beyond it. Paged by (canonical,
        kind) rather than OFFSET so a page costs the same at 36,000 entities as
        at the first.
        """

        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
            raise ValueError("page_size must be at least 1")
        cursor = ("", "")
        while True:
            with self._session() as connection:
                rows = connection.execute(
                    _GRAPH_ENTITIES_PAGE_SQL,
                    (
                        self._revision_key,
                        self._revision_key,
                        self._revision_key,
                        cursor[0],
                        cursor[1],
                        page_size,
                    ),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield {
                    "canonical": str(row["canonical"]),
                    "entity_kind": str(row["entity_kind"]),
                    "description": str(row["description"] or ""),
                }
            cursor = (str(rows[-1]["canonical"]), str(rows[-1]["entity_kind"]))

    def graph_statements(self, limit: int = MAX_SUBGRAPH_EDGES) -> list[dict[str, Any]]:
        """Return every statement of this revision as a binary edge with evidence."""

        with self._session() as connection:
            rows = connection.execute(
                _GRAPH_STATEMENTS_SQL, (self._revision_key, limit)
            ).fetchall()
        return [
            {
                "source": str(row["src"]),
                "source_kind": str(row["src_kind"]),
                "target": str(row["dst"]),
                "target_kind": str(row["dst_kind"]),
                "predicate": str(row["predicate"]),
                "weight": row["weight"],
                "description": str(row["description"] or ""),
                "node_ids": sorted(int(n) for n in (row["node_nos"] or ())),
            }
            for row in rows
        ]

    def entities_for_nodes(self, node_ids: Sequence[int]) -> dict[int, list[str]]:
        wanted = _bounded_ids(node_ids)
        if not wanted:
            return {}
        with self._session() as connection:
            rows = connection.execute(
                "SELECT node.node_no, ent.canonical "
                "FROM docarag.entity_mention mention "
                "JOIN docarag.tree_node node "
                "  ON node.revision_key = mention.revision_key "
                " AND node.node_key = mention.node_key "
                "JOIN docarag.entity ent ON ent.entity_key = mention.entity_key "
                "WHERE mention.revision_key = %s AND node.node_no = ANY(%s)",
                (self._revision_key, wanted),
            ).fetchall()
        found: dict[int, list[str]] = {}
        for row in rows:
            found.setdefault(int(row["node_no"]), []).append(str(row["canonical"]))
        return found

    # -- plumbing -----------------------------------------------------------

    def _session(self):
        # The connection is borrowed for the life of a query and never owned
        # here: entering it as a context manager would end the caller's
        # transaction between two reads of the same slice.
        return _Borrowed(self._connect())

    def _query(self, sql: str, params: tuple[object, ...]) -> list[TreeNodeView]:
        with self._session() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_view(row) for row in rows]

    def _one(self, sql: str, params: tuple[object, ...]) -> Mapping[str, Any] | None:
        with self._session() as connection:
            return connection.execute(sql, params).fetchone()


class _Borrowed:
    """Adapt a plain connection to the context-manager protocol."""

    __slots__ = ("_connection",)

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def __enter__(self) -> Any:
        return self._connection

    def __exit__(self, *_exc: object) -> bool:
        return False


def _bounded_ids(values: Iterable[int]) -> list[int]:
    seen: dict[int, None] = {}
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        seen.setdefault(value)
        if len(seen) >= MAX_NODES_PER_READ:
            break
    return list(seen)


def _view(row: Mapping[str, Any]) -> TreeNodeView:
    return TreeNodeView(
        index_id=int(row["node_no"]),
        type=str(row["node_kind"]),
        depth=int(row["depth"] or 0),
        parent_id=int(row["parent_no"]) if row.get("parent_no") is not None else None,
        summary=str(row.get("summary") or ""),
        evidence=(row.get("attrs") or {}).get("evidence"),
        outline_node=bool(row.get("is_outline")),
        meta_info=NodeMeta(
            content=str(row.get("content") or row.get("label") or ""),
            page_idx=row.get("page_no"),
            img_path=row.get("cas_uri"),
            caption=str(row.get("caption") or ""),
            footnote=str(row.get("footnote") or ""),
            table_body=str(row.get("table_body") or ""),
            block_uid=row.get("block_uid"),
            block_revision_uid=row.get("block_revision_uid"),
            page_uid=row.get("page_uid"),
            geometry=row.get("geometry"),
        ),
    )


# A node's fields plus the block channels retrieval reads.  The text channels
# are pulled by kind rather than joined row-wise, so one node stays one row.
_NODE_COLUMNS = """
    node.node_no,
    node.node_kind,
    node.depth,
    node.label,
    node.attrs,
    parent.node_no AS parent_no,
    summary_blob.body AS summary,
    blk.block_uid,
    blk.block_revision_uid,
    blk.page_uid,
    blk.page_no,
    blk.geometry,
    (node.parent_key IS NULL OR node.node_kind ILIKE '%%title%%') AS is_outline,
    feed.retrieval_text AS content,
    _channel(node.revision_key, node.block_key, 'caption') AS caption,
    _channel(node.revision_key, node.block_key, 'footnote') AS footnote,
    _channel(node.revision_key, node.block_key, 'html') AS table_body,
    crop.cas_uri
"""

_NODE_SOURCE = """
FROM docarag.tree_node node
LEFT JOIN docarag.tree_node parent
    ON parent.revision_key = node.revision_key AND parent.node_key = node.parent_key
LEFT JOIN docarag.text_blob summary_blob ON summary_blob.blob_key = node.summary_blob_key
LEFT JOIN docarag.block blk
    ON blk.revision_key = node.revision_key AND blk.block_key = node.block_key
LEFT JOIN docarag.node_retrieval feed
    ON feed.revision_key = node.revision_key AND feed.node_key = node.node_key
LEFT JOIN LATERAL (
    SELECT asset.cas_uri
    FROM docarag.block_asset link
    JOIN docarag.asset asset ON asset.asset_key = link.asset_key
    WHERE link.revision_key = node.revision_key AND link.block_key = node.block_key
    LIMIT 1
) crop ON true
"""

# Postgres has no inline helper, so the channel lookup is written out once and
# formatted into each statement.
_CHANNEL = """(
    SELECT body FROM docarag.text_blob
    JOIN docarag.block_representation rep ON rep.blob_key = text_blob.blob_key
    WHERE rep.revision_key = node.revision_key
      AND rep.block_key = node.block_key
      AND rep.rep_kind = '{kind}'
    LIMIT 1
)"""


def _sql(body: str) -> str:
    rendered = _NODE_COLUMNS
    for kind in ("caption", "footnote", "html"):
        rendered = rendered.replace(
            f"_channel(node.revision_key, node.block_key, '{kind}')",
            _CHANNEL.format(kind=kind),
        )
    return f"SELECT {rendered}{_NODE_SOURCE}{body}"


_NODES_SQL = _sql("WHERE node.revision_key = %s ORDER BY node.node_no")
_NODES_BY_NO_SQL = _sql(
    "WHERE node.revision_key = %s AND node.node_no = ANY(%s) ORDER BY node.node_no"
)
_ROOT_SQL = _sql("WHERE node.revision_key = %s AND node.parent_key IS NULL LIMIT 1")
_CHILDREN_SQL = _sql(
    "WHERE node.revision_key = %s AND parent.node_no = %s ORDER BY node.sibling_no"
)
_SIBLINGS_SQL = _sql(
    "WHERE node.revision_key = %s AND node.node_no <> %s AND node.parent_key = ("
    "  SELECT self.parent_key FROM docarag.tree_node self"
    "  WHERE self.revision_key = node.revision_key AND self.node_no = %s"
    ") ORDER BY node.sibling_no"
)
_AT_DEPTH_SQL = _sql(
    "WHERE node.revision_key = %s AND node.depth = %s ORDER BY node.node_no"
)
_BY_KIND_SQL = _sql(
    "WHERE node.revision_key = %s AND node.node_kind = %s ORDER BY node.node_no"
)
# Ancestors and descendants are ltree containment, so neither recurses.
_ANCESTORS_SQL = _sql(
    "WHERE node.revision_key = %s AND node.path @> ("
    "  SELECT self.path FROM docarag.tree_node self"
    "  WHERE self.revision_key = node.revision_key AND self.node_no = %s"
    ") ORDER BY node.depth"
)
_SUBTREE_SQL = _sql(
    "WHERE node.revision_key = %s AND EXISTS ("
    "  SELECT 1 FROM docarag.tree_node section"
    "  WHERE section.revision_key = node.revision_key"
    "    AND section.node_no = ANY(%s)"
    "    AND node.path <@ section.path"
    ") ORDER BY node.node_no"
)

_STRUCTURE_ELEMENTS_SQL = """
SELECT
    node.node_no,
    s.structure_key,
    s.structure_kind,
    el.element_kind,
    el.content,
    el.role
FROM docarag.tree_node node
JOIN docarag.block_structure s
    ON s.revision_key = node.revision_key AND s.block_key = node.block_key
JOIN docarag.block_element el
    ON el.revision_key = s.revision_key AND el.structure_key = s.structure_key
WHERE node.revision_key = %s AND node.node_no = ANY(%s)
ORDER BY s.structure_key, el.element_key
"""

_STRUCTURE_EDGES_SQL = """
SELECT
    s.structure_key,
    edge.edge_kind,
    edge.label,
    src.content AS src_content,
    dst.content AS dst_content
FROM docarag.tree_node node
JOIN docarag.block_structure s
    ON s.revision_key = node.revision_key AND s.block_key = node.block_key
JOIN docarag.block_element_edge edge
    ON edge.revision_key = s.revision_key AND edge.structure_key = s.structure_key
JOIN docarag.block_element src
    ON src.revision_key = edge.revision_key AND src.element_key = edge.src_key
JOIN docarag.block_element dst
    ON dst.revision_key = edge.revision_key AND dst.element_key = edge.dst_key
WHERE node.revision_key = %s AND node.node_no = ANY(%s)
ORDER BY s.structure_key, edge.ordinal, edge.src_key
"""


_GRAPH_ENTITIES_SQL = """
WITH revision_entities AS (
    SELECT entity_key FROM docarag.entity_mention WHERE revision_key = %s
    UNION
    SELECT entity_key FROM docarag.statement_role WHERE revision_key = %s
)
SELECT
    ent.canonical,
    ent.entity_kind,
    -- This revision's own wording first. ``entity.description_blob_key`` is
    -- corpus-scoped and last-writer-wins, so it holds whatever document
    -- mentioned the entity most recently -- 605 of one paper's 862 entities
    -- carried another revision's text that way. The corpus row remains the
    -- fallback for an entity a revision never described.
    COALESCE(own_blob.body, blob.body) AS description,
    mentioned.node_nos
FROM revision_entities re
JOIN docarag.entity ent ON ent.entity_key = re.entity_key
LEFT JOIN docarag.text_blob blob ON blob.blob_key = ent.description_blob_key
LEFT JOIN docarag.entity_description own
    ON own.revision_key = %s AND own.entity_key = ent.entity_key
LEFT JOIN docarag.text_blob own_blob ON own_blob.blob_key = own.description_blob_key
LEFT JOIN LATERAL (
    SELECT array_agg(DISTINCT node.node_no) AS node_nos
    FROM docarag.entity_mention mention
    JOIN docarag.tree_node node
        ON node.revision_key = mention.revision_key
       AND node.node_key = mention.node_key
    WHERE mention.revision_key = %s AND mention.entity_key = ent.entity_key
) mentioned ON true
ORDER BY ent.canonical
LIMIT %s
"""


#: The same rows, paged rather than capped.
#:
#: ``_GRAPH_ENTITIES_SQL`` bounds itself at ``MAX_GRAPH_ENTITIES`` because it
#: answers "show me this document's graph" -- a display, where 8,192 entities is
#: already more than anyone reads. The entity *index* is not a display: it must
#: carry every entity or a question can reach none of the ones it dropped. One
#: revision here holds 36,572, so the cap silently indexed 22% of it, and only
#: the attestation's point count caught the difference.
_GRAPH_ENTITIES_PAGE_SQL = """
WITH revision_entities AS (
    SELECT entity_key FROM docarag.entity_mention WHERE revision_key = %s
    UNION
    SELECT entity_key FROM docarag.statement_role WHERE revision_key = %s
)
SELECT
    ent.canonical,
    ent.entity_kind,
    COALESCE(own_blob.body, blob.body) AS description
FROM revision_entities re
JOIN docarag.entity ent ON ent.entity_key = re.entity_key
LEFT JOIN docarag.text_blob blob ON blob.blob_key = ent.description_blob_key
LEFT JOIN docarag.entity_description own
    ON own.revision_key = %s AND own.entity_key = ent.entity_key
LEFT JOIN docarag.text_blob own_blob ON own_blob.blob_key = own.description_blob_key
WHERE (ent.canonical, ent.entity_kind) > (%s, %s)
ORDER BY ent.canonical, ent.entity_kind
LIMIT %s
"""

_GRAPH_STATEMENTS_SQL = """
SELECT
    subject.canonical AS src,
    subject.entity_kind AS src_kind,
    object.canonical AS dst,
    object.entity_kind AS dst_kind,
    stmt.predicate,
    stmt.weight,
    blob.body AS description,
    evidence.node_nos
FROM docarag.statement stmt
JOIN docarag.statement_role subject_role
    ON subject_role.revision_key = stmt.revision_key
   AND subject_role.stmt_key = stmt.stmt_key
   AND subject_role.role = 'subject'
JOIN docarag.statement_role object_role
    ON object_role.revision_key = stmt.revision_key
   AND object_role.stmt_key = stmt.stmt_key
   AND object_role.role = 'object'
JOIN docarag.entity subject ON subject.entity_key = subject_role.entity_key
JOIN docarag.entity object ON object.entity_key = object_role.entity_key
LEFT JOIN docarag.text_blob blob ON blob.blob_key = stmt.description_blob_key
LEFT JOIN LATERAL (
    SELECT array_agg(DISTINCT node.node_no) AS node_nos
    FROM docarag.statement_evidence ev
    JOIN docarag.tree_node node
        ON node.revision_key = ev.revision_key AND node.node_key = ev.node_key
    WHERE ev.revision_key = stmt.revision_key AND ev.stmt_key = stmt.stmt_key
) evidence ON true
WHERE stmt.revision_key = %s
ORDER BY stmt.stmt_key
LIMIT %s
"""


_NODES_FOR_ENTITY_SQL = """
SELECT DISTINCT node.node_no
FROM docarag.entity ent
JOIN docarag.entity_mention mention ON mention.entity_key = ent.entity_key
JOIN docarag.tree_node node
    ON node.revision_key = mention.revision_key AND node.node_key = mention.node_key
WHERE mention.revision_key = %s
  AND ent.canonical = %s
  AND (%s::text IS NULL OR ent.entity_kind = %s)
ORDER BY node.node_no
"""

_SUBGRAPH_SQL = """
SELECT DISTINCT
    subject.canonical AS src,
    object.canonical AS dst,
    stmt.predicate,
    stmt.weight,
    blob.body AS description
FROM docarag.statement_evidence ev
JOIN docarag.tree_node node
    ON node.revision_key = ev.revision_key AND node.node_key = ev.node_key
JOIN docarag.statement stmt
    ON stmt.revision_key = ev.revision_key AND stmt.stmt_key = ev.stmt_key
JOIN docarag.statement_role subject_role
    ON subject_role.revision_key = stmt.revision_key
   AND subject_role.stmt_key = stmt.stmt_key
   AND subject_role.role = 'subject'
JOIN docarag.statement_role object_role
    ON object_role.revision_key = stmt.revision_key
   AND object_role.stmt_key = stmt.stmt_key
   AND object_role.role = 'object'
JOIN docarag.entity subject ON subject.entity_key = subject_role.entity_key
JOIN docarag.entity object ON object.entity_key = object_role.entity_key
LEFT JOIN docarag.text_blob blob ON blob.blob_key = stmt.description_blob_key
WHERE ev.revision_key = %s AND node.node_no = ANY(%s)
LIMIT %s
"""


__all__ = ["GraphEdge", "MAX_GRAPH_ENTITIES", "NodeMeta", "TreeNodeView", "TreeStore"]
