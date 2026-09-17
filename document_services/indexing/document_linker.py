"""Discover and materialize cross-document groupings and cross-docs relations.

Operates strictly on persisted Postgres state -- constructed after a
document's own graph-part persist has already run. Two ways to trigger it:

``link_document`` (auto): one just-persisted document against the whole corpus.
``link_documents`` (direct): one focal document -- just-ingested, or the one a
user chose "Link to..." on -- against a user-chosen, closed set of already-
ingested target documents, ignoring the rest of the corpus. Both funnel
through the same candidate/judge/write pipeline; only the search scope
differs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from retrieval_engine.configuration.knowledge_graph import GraphConfig
from retrieval_engine.configuration.system import SystemConfig
from retrieval_engine.configuration.vector_store import VDBConfig
from retrieval_engine.indexing.graph.entity_facets import (
    CANONICAL_FIELD,
    collapse_entity_hits,
    context_text,
    lexical_terms,
    lexical_text,
)
from retrieval_engine.indexing.vector.builder import corpus_entity_vector_scope
from retrieval_engine.providers.embeddings import TextEmbeddingProvider
from retrieval_engine.providers.language_model import LLM
from retrieval_engine.indexes.vector_store import VectorStore
from retrieval_engine.utils.bounded_executor import iter_bounded_ordered
from document_services.persistence.corpus_store import (
    CorpusStore, EntityView, GroupingMembership, entity_node_name, split_node_name,
)
from document_services.persistence.document_store import _open_connection
from document_services.persistence.contracts import CorpusScope
from document_services.persistence.postgres import PostgresLedger, resolve_database_url

logger = logging.getLogger(__name__)


# Document-structure container entities -- "Table 1", "Figure 2", "Formula
# (3)", a section heading -- name where something sits in THIS document's
# layout, not a real-world concept (retrieval_engine/indexing/graph/extractor.py assigns
# these kinds to the table/image/equation/section itself, separately from the
# semantic entities extracted from its content). Nearly every document has a
# generically named one of these, so matching them across documents produces
# false positives, not real links -- two unrelated papers' "Table 1" is not
# the same "Table 1". Excluded from cross-document linking on both sides: as
# the entity being linked, and as a candidate match already in the corpus.
STRUCTURAL_ENTITY_KINDS = frozenset({
    "TABLE", "IMAGE", "EQUATION_OR_FORMULA", "SECTION_TITLE",
})

# Ceiling on how many candidates one judge_pairs() call covers -- keeps the
# prompt and CrossDocBatchVerdict.verdicts bounded regardless of how many
# candidates one entity's search turns up (a restricted search's collapse
# limit scales with target-document count, see _search_cross_document_matches).
MAX_CROSS_DOC_BATCH_SIZE = 20


class GroupingAttachChoice(BaseModel):
    """One ambiguous grouping's attach-or-leave-alone verdict."""

    grouping_key: int
    attach: bool = Field(default=False)


class CrossDocPairVerdict(BaseModel):
    """One candidate pair's (A, B) verdict within a batched LLM call.

    Grouping and relation outcomes are independent and non-exclusive.
    ``grouping_attach`` only ever covers groupings the embedding-vs-centroid
    gate already flagged ambiguous -- confident and out-of-band groupings
    never reach the model. ``candidate_index`` ties this verdict back to the
    candidate it judges -- see CROSS_DOC_BATCH_PROMPT's [N] labels.
    """

    candidate_index: int = Field(default=-1)
    grouping_attach: list[GroupingAttachChoice] = Field(default_factory=list, max_length=16)
    warrants_new_grouping: bool = Field(default=False)
    new_grouping_label: str = Field(default="", max_length=256)
    new_grouping_description: str = Field(default="", max_length=2048)

    has_named_relation: bool = Field(default=False)
    relation_predicate: str = Field(default="", max_length=256)
    relation_subject: Literal["a", "b"] = Field(default="a")
    relation_directed: bool = Field(default=True)
    relation_description: str = Field(default="", max_length=2048)
    relation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CrossDocBatchVerdict(BaseModel):
    """Every candidate's verdict for one query entity (B), from one LLM call."""

    verdicts: list[CrossDocPairVerdict] = Field(default_factory=list, max_length=MAX_CROSS_DOC_BATCH_SIZE)


@dataclass(frozen=True, slots=True)
class CrossDocCandidate:
    entity_a: EntityView  # pre-existing corpus entity, the embedding hit
    entity_b: EntityView  # the entity being linked
    embedding_score: float
    # Which document actually anchored each side AT SEARCH TIME -- entity_key
    # alone can't answer that later, since entity resolution (a separate,
    # per-document process) can merge an entity with an unrelated document's
    # own mention of the same canonical name at any point, before or after
    # this candidate is judged. entity_a_document_version_id is only ever set
    # in restricted (documents/group) mode, where the search was scoped to
    # exactly one target document; auto mode has no single document to record.
    # entity_b_document_version_id is the revision being linked's own document.
    entity_a_document_version_id: str | None = None
    entity_b_document_version_id: str | None = None


@dataclass(slots=True)
class CrossDocLinkStats:
    entities_scanned: int = 0
    candidates_in_band: int = 0
    candidates_judged: int = 0
    llm_calls: int = 0
    groupings_created: int = 0
    groupings_attached: int = 0
    relations_written: int = 0

    def __iadd__(self, other: "CrossDocLinkStats") -> "CrossDocLinkStats":
        self.entities_scanned += other.entities_scanned
        self.candidates_in_band += other.candidates_in_band
        self.candidates_judged += other.candidates_judged
        self.llm_calls += other.llm_calls
        self.groupings_created += other.groupings_created
        self.groupings_attached += other.groupings_attached
        self.relations_written += other.relations_written
        return self


# One call judges every in-band candidate for a single query entity (B) at
# once -- mirrors kg_extractor's own one-call-per-node batching (extract
# every entity/relation a node contains in one completion) instead of the
# one-call-per-pair granularity this used to have, which made cross-linking
# issue far more LLM round-trips than KG extraction for a comparable amount
# of document content.
CROSS_DOC_BATCH_PROMPT = """\
-Goal-
Entity B is being linked into the corpus. Each candidate below is an
existing entity A that might connect to B. For EACH candidate, judge three
independent things: does B fit its listed groupings, do A and B warrant a
brand-new grouping (a shared topic category, e.g. "Optimization
algorithms" -- not a fact about this pair), and does a specific named
relation hold between them (e.g. "is a variant of"). A pair can warrant a
grouping, a relation, both, or neither -- never treat one as a fallback for
the other. Judge every candidate strictly on its own text; never let one
candidate's verdict lean on another's.

-Entity B (same for every candidate below)-
{b_text}

-Candidates-
{candidates_block}

-Output-
One verdict per candidate, `candidate_index` matching its `[N]` label
exactly, as a JSON object:

```json
{{
  "verdicts": [
    {{
      "candidate_index": "<int, matches [N]>",
      "grouping_attach": [
        {{"grouping_key": "<int, from that candidate's listed groupings>", "attach": "<bool>"}}
      ],
      "warrants_new_grouping": "<bool>",
      "new_grouping_label": "<string, empty if warrants_new_grouping is false>",
      "new_grouping_description": "<string, empty if warrants_new_grouping is false>",
      "has_named_relation": "<bool>",
      "relation_predicate": "<string, empty if has_named_relation is false>",
      "relation_subject": "<'a' or 'b'>",
      "relation_directed": "<bool>",
      "relation_description": "<string, empty if has_named_relation is false>",
      "relation_confidence": "<float 0.0-1.0, 0.0 if has_named_relation is false>"
    }}
  ]
}}
```

One `grouping_attach` entry per listed grouping (`[]` if none listed).
Only report what the candidate's own A/B text supports -- never fabricate.

-Example (one candidate, warranting both a grouping and a relation at once)-

B: Adam optimizer (TECHNOLOGY) -- adaptive learning-rate algorithm.
[0] A: gradient descent (TECHNOLOGY) -- minimizes a function via steepest
descent. Groupings: (none)

```json
{{"verdicts": [{{"candidate_index": 0, "grouping_attach": [],
"warrants_new_grouping": true, "new_grouping_label": "Optimization algorithms",
"new_grouping_description": "Algorithms that minimize a loss function during training.",
"has_named_relation": true, "relation_predicate": "is a variant of",
"relation_subject": "b", "relation_directed": true,
"relation_description": "Adam is a variant of gradient descent with adaptive per-parameter rates.",
"relation_confidence": 0.9}}]}}
```

-Task Execution-
Now judge every candidate listed above under -Candidates-.
"""


def _no_groupings_block() -> str:
    return "(none)"


class CrossDocLinker:
    """Cross-link one document's entities against a corpus scope."""

    __slots__ = (
        "_ledger", "_store", "_vdb", "_vdb_cfg", "_restricted_vdb_cache",
        "_embedder", "_llm", "_read_connection",
        "tenant_id", "corpus_id",
        "merge_threshold", "relatedness_floor",
        "candidate_top_k", "restricted_candidate_top_k", "description_refresh_every",
        "max_workers", "max_in_flight",
        "_store_lock",
        "_closed",
    )

    def __init__(
        self,
        ledger: PostgresLedger,
        store: CorpusStore,
        vdb: VectorStore,
        vdb_cfg: VDBConfig,
        embedder: TextEmbeddingProvider,
        llm: LLM,
        *,
        tenant_id: str,
        corpus_id: str,
        merge_threshold: float,
        relatedness_floor: float,
        candidate_top_k: int = 20,
        restricted_candidate_top_k: int = 6,
        description_refresh_every: int = 5,
        read_connection: object | None = None,
        max_workers: int = 4,
        max_in_flight: int | None = None,
    ) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise ValueError("corpus_id must be a non-empty string")
        if not (0.0 <= relatedness_floor < merge_threshold <= 1.0):
            raise ValueError(
                "require 0.0 <= relatedness_floor < merge_threshold <= 1.0"
            )
        if isinstance(candidate_top_k, bool) or candidate_top_k < 1:
            raise ValueError("candidate_top_k must be at least 1")
        if isinstance(restricted_candidate_top_k, bool) or restricted_candidate_top_k < 1:
            raise ValueError("restricted_candidate_top_k must be at least 1")
        if isinstance(description_refresh_every, bool) or description_refresh_every < 1:
            raise ValueError("description_refresh_every must be at least 1")
        if isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_in_flight is None:
            max_in_flight = max_workers
        if isinstance(max_in_flight, bool) or max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")

        self._ledger = ledger
        self._store = store
        self._vdb = vdb
        self._vdb_cfg = vdb_cfg
        self._restricted_vdb_cache = {}
        self._embedder = embedder
        self._llm = llm
        self.tenant_id = tenant_id
        self.corpus_id = corpus_id
        self.merge_threshold = merge_threshold
        self.relatedness_floor = relatedness_floor
        self.candidate_top_k = candidate_top_k
        self.restricted_candidate_top_k = restricted_candidate_top_k
        self.description_refresh_every = description_refresh_every
        self._read_connection = read_connection
        self.max_workers = max_workers
        self.max_in_flight = max_in_flight
        # self._store wraps one shared, non-pooled psycopg connection (see
        # CrossDocLinker.open) -- fine for the single-threaded callers this
        # class originally had, but a psycopg connection is not safe for
        # concurrent use from multiple threads. _link_one now runs entities
        # through a thread pool (mirrors kg_extractor's per-node concurrency),
        # so every read through self._store from a worker thread goes through
        # this lock. The reads it guards are small indexed lookups; the actual
        # per-entity cost -- the batched judge_pairs call and the embedding
        # call -- happens outside the lock and so still runs fully in parallel.
        self._store_lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        tenant_id: str,
        corpus_id: str,
        system_config: SystemConfig,
        database_url: str | None = None,
        merge_threshold: float | None = None,
        relatedness_floor: float | None = None,
        candidate_top_k: int = 20,
    ) -> "CrossDocLinker":
        """Build every collaborator from a loaded SystemConfig."""

        graph_config: GraphConfig = system_config.graph
        url = resolve_database_url(database_url)
        ledger = PostgresLedger(url)
        read_connection = _open_connection(url)
        store = CorpusStore(lambda: read_connection, tenant_id, corpus_id)
        embedder = TextEmbeddingProvider(
            model_name=graph_config.embedding_config.model_name,
            backend=graph_config.embedding_config.backend,
            max_length=graph_config.embedding_config.max_length,
            device=graph_config.embedding_config.device,
            api_base=graph_config.embedding_config.api_base,
            api_key=getattr(graph_config.embedding_config, "api_key", None),
        )
        vdb = VectorStore(
            embedding_model=embedder,
            qdrant_url=system_config.vdb.qdrant_url,
            collection_name=graph_config.entity_collection_name,
            scope_filter=corpus_entity_vector_scope(system_config.vdb),
            hybrid_entity_schema=True,
        )
        llm = LLM(llm_config=system_config.llm)
        return cls(
            ledger, store, vdb, system_config.vdb, embedder, llm,
            tenant_id=tenant_id, corpus_id=corpus_id,
            merge_threshold=(
                merge_threshold if merge_threshold is not None
                else graph_config.cross_doc_merge_threshold
            ),
            relatedness_floor=(
                relatedness_floor if relatedness_floor is not None
                else graph_config.cross_doc_relatedness_floor
            ),
            candidate_top_k=candidate_top_k,
            restricted_candidate_top_k=graph_config.cross_doc_link_restricted_candidate_top_k,
            read_connection=read_connection,
            max_workers=graph_config.cross_doc_link_max_workers,
            max_in_flight=graph_config.cross_doc_link_max_in_flight,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        owned = [self._vdb, self._embedder, self._llm]
        if self._read_connection is not None:
            owned.append(self._read_connection)
        for item in owned:
            close = getattr(item, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    # -- orchestration entrypoints --------------------------------------------

    def link_document(self, revision_key: int) -> CrossDocLinkStats:
        """Auto mode: cross-link one just-persisted document against the whole corpus."""

        return self._link_one(revision_key, restrict_to_revisions=None)

    def link_document_expanding(self, revision_key: int) -> CrossDocLinkStats:
        """Auto, but restricted to the existing mesh: this document's direct
        matches, plus whatever those documents are already cross-linked to.
        Skips documents with no existing cross-doc connection at all."""

        candidates = self.discover_candidates(revision_key)
        direct: set[int] = set()
        for candidate in candidates:
            direct |= self._store.revisions_for_entity(candidate.entity_a.entity_key)
        direct.discard(revision_key)

        expanded = set(direct)
        for other_revision in direct:
            expanded |= self._store.linked_revisions_for_revision(other_revision)
        expanded.discard(revision_key)

        if not expanded:
            return CrossDocLinkStats()
        return self.link_documents([revision_key, *sorted(expanded)])

    def link_documents(self, revision_keys: Sequence[int]) -> CrossDocLinkStats:
        """Direct mode: cross-link the focal document (first) against the rest.

        One-directional -- only the focal document's own entities are scanned,
        restricted to the target set. Every current caller (the ingest-time
        plan, the retroactive "Link to..." action, mesh expansion) passes
        [focal, *targets]: a document being compared against already-settled
        others, never a request to also mutually link those others against
        each other or against the focal document in reverse. Scanning every
        key against every other key -- what this used to do -- silently
        doubled (or worse, for a group of N targets) the entity scans and LLM
        judgments needed for exactly this shape of call, for no additional
        relations: A->B and B->A candidate search surface the same underlying
        pair, judged from both sides.
        """

        keys = [int(k) for k in revision_keys]
        if len(keys) < 2:
            raise ValueError("link_documents needs at least two documents")
        focal, *targets = keys
        return self._link_one(focal, restrict_to_revisions=frozenset(targets))

    def _link_one(
        self, revision_key: int, *, restrict_to_revisions: frozenset[int] | None,
    ) -> CrossDocLinkStats:
        entities = [
            entity
            for entity in self._store_call(self._store.entities_for_revision, revision_key)
            if entity.entity_kind not in STRUCTURAL_ENTITY_KINDS
        ]
        stats = CrossDocLinkStats(entities_scanned=len(entities))

        # Resolved once per document, not once per entity: the vector search
        # filters on document_version_id (what entity points are actually
        # written under -- see retrieval_engine/indexing/vector/builder.py::entity_vector_scope),
        # not revision_key, so a restricted link needs this translated exactly
        # once before any entity's search runs.
        restrict_to_document_versions = None
        if restrict_to_revisions is not None:
            restrict_to_document_versions = frozenset(
                docv for docv in (
                    self._store_call(self._store.document_version_id_for_revision, rk)
                    for rk in restrict_to_revisions
                )
                if docv is not None
            )
        # This revision's own document -- recorded on every candidate found
        # for it, so a written relation's provenance survives any later,
        # unrelated entity-name merge with some other document.
        own_document_version_id = self._store_call(
            self._store.document_version_id_for_revision, revision_key,
        )

        # One entity's candidate search + LLM judgment is independent of every
        # other entity's -- the same independence kg_extractor.py already
        # exploits per tree node -- so this runs through the same bounded
        # thread pool instead of one entity, and one LLM round-trip, at a time.
        for outcome in iter_bounded_ordered(
            entities,
            lambda entity: self._link_one_entity(
                entity, restrict_to_revisions, restrict_to_document_versions,
                own_document_version_id,
            ),
            max_workers=self.max_workers,
            max_in_flight=self.max_in_flight,
            thread_name_prefix="cross-doc-link",
        ):
            if outcome.ok:
                stats += outcome.unwrap()
            else:
                logger.error(
                    "Skipping cross-doc linking for entity %r after failure: %s",
                    outcome.item.canonical, outcome.error,
                )
        return stats

    def _link_one_entity(
        self,
        entity: EntityView,
        restrict_to_revisions: frozenset[int] | None,
        restrict_to_document_versions: frozenset[str] | None,
        own_document_version_id: str | None,
    ) -> CrossDocLinkStats:
        """One entity's full candidate-search-through-write pipeline -- the unit
        of work handed to each thread pool worker by _link_one."""

        stats = CrossDocLinkStats()
        candidates = [
            candidate
            for candidate in self._search_cross_document_matches(
                entity,
                restrict_to_revisions=restrict_to_revisions,
                restrict_to_document_versions=restrict_to_document_versions,
                own_document_version_id=own_document_version_id,
            )
            if self._is_cross_document(candidate.entity_a.entity_key, entity.entity_key)
        ]
        stats.candidates_in_band = len(candidates)
        if not candidates:
            return stats

        # Groupings and confident attaches are resolved locally, no LLM
        # needed -- only a candidate left ambiguous (or with no existing
        # groupings at all) needs a model judgment, collected into pending
        # and judged together below instead of one call per candidate.
        b_vec = None
        pending: list[tuple[CrossDocCandidate, list[GroupingMembership]]] = []
        for candidate in candidates:
            groupings = self._store_call(
                self._store.groupings_for_entity, candidate.entity_a.entity_key,
            )
            confident: list[GroupingMembership] = []
            ambiguous: list[GroupingMembership] = []
            if groupings:
                if b_vec is None:
                    # Computed at most once per entity, not once per
                    # candidate -- B's own embedding never changes across
                    # however many candidates it has.
                    b_vec = self._embedder.embed_texts([entity.canonical])[0]
                for grouping in groupings:
                    if not grouping.centroid:
                        ambiguous.append(grouping)
                        continue
                    fit = self._grouping_fit(b_vec, grouping)
                    (confident if fit >= self.relatedness_floor else ambiguous).append(grouping)

            for grouping in confident:
                self._attach_to_grouping(
                    candidate.entity_b, grouping, weight=candidate.embedding_score, member_vec=b_vec,
                )
                stats.groupings_attached += 1

            if ambiguous or not groupings:
                pending.append((candidate, ambiguous))

        stats.candidates_judged = len(pending)
        # One call judges every pending candidate for this entity at once --
        # mirrors kg_extractor's own one-call-per-node batching. Chunked so
        # the prompt and CrossDocBatchVerdict.verdicts stay bounded even when
        # a restricted search's collapse_limit (top_k * target count) makes
        # `pending` larger than one batch should be.
        for chunk_start in range(0, len(pending), MAX_CROSS_DOC_BATCH_SIZE):
            chunk = pending[chunk_start:chunk_start + MAX_CROSS_DOC_BATCH_SIZE]
            stats.llm_calls += 1
            verdicts = self.judge_pairs(entity, chunk)
            for index, (candidate, ambiguous) in enumerate(chunk):
                verdict = verdicts.get(index)
                if verdict is not None:
                    self._apply_verdict(candidate, ambiguous, verdict, stats)
        return stats

    def _apply_verdict(
        self,
        candidate: CrossDocCandidate,
        ambiguous: Sequence[GroupingMembership],
        verdict: CrossDocPairVerdict,
        stats: CrossDocLinkStats,
    ) -> None:
        by_key = {g.grouping_key: g for g in ambiguous}
        for choice in verdict.grouping_attach:
            grouping = by_key.get(choice.grouping_key)
            if choice.attach and grouping is not None:
                self._attach_to_grouping(candidate.entity_b, grouping, weight=candidate.embedding_score)
                stats.groupings_attached += 1

        if verdict.warrants_new_grouping:
            self._create_grouping(
                candidate, verdict.new_grouping_label, verdict.new_grouping_description,
            )
            stats.groupings_created += 1

        if verdict.has_named_relation:
            self._write_relation(candidate, verdict)
            stats.relations_written += 1

    def _store_call(self, fn, *args, **kwargs):
        """Every read through self._store, serialized.

        self._store wraps one shared, non-pooled psycopg connection (see
        CrossDocLinker.open) that is not safe for concurrent use from
        multiple threads. _link_one's workers all read through this store, so
        every call is routed through this lock -- it only guards small
        indexed lookups, not the LLM/embedding calls that make up the actual
        per-entity cost, so those still run fully in parallel.
        """

        with self._store_lock:
            return fn(*args, **kwargs)

    # -- candidate generation --------------------------------------------------

    def discover_candidates(
        self, revision_key: int, *, restrict_to_revisions: frozenset[int] | None = None,
    ) -> list[CrossDocCandidate]:
        found: list[CrossDocCandidate] = []
        for entity in self._store.entities_for_revision(revision_key):
            if entity.entity_kind in STRUCTURAL_ENTITY_KINDS:
                continue
            found.extend(
                self._search_cross_document_matches(
                    entity, restrict_to_revisions=restrict_to_revisions,
                )
            )
        return found

    def _vdb_for_document(self, document_version_id: str) -> VectorStore:
        """A VectorStore whose own scope is restricted to exactly one target
        document -- restricted (documents/group) mode's real isolation
        boundary, not just a metadata_filter kwarg a future call site could
        forget to pass. Shares self._vdb's underlying Qdrant client instead
        of opening a new connection per target document."""

        cached = self._restricted_vdb_cache.get(document_version_id)
        if cached is None:
            cached = VectorStore(
                embedding_model=self._embedder,
                collection_name=self._vdb.collection_name,
                scope_filter=corpus_entity_vector_scope(
                    self._vdb_cfg, document_version_ids=[document_version_id],
                ),
                hybrid_entity_schema=True,
                client=self._vdb.client,
                models=self._vdb._models,
            )
            self._restricted_vdb_cache[document_version_id] = cached
        return cached

    def _search_cross_document_matches(
        self,
        entity: EntityView,
        *,
        restrict_to_revisions: frozenset[int] | None,
        restrict_to_document_versions: frozenset[str] | None = None,
        own_document_version_id: str | None = None,
    ) -> list[CrossDocCandidate]:
        # Entity points embed the combined "Name: x\nType: y" form (see
        # kg_refiner.get_vdb_meta_data / get_node_name_from_entity), not the
        # bare canonical -- the query text has to match that shape to be
        # comparable against the stored name/context vectors.
        node_name = entity_node_name(entity.canonical, entity.entity_kind)
        # A restricted search already knows exactly which document it's
        # searching within (one target's worth of entities, not the whole
        # corpus) -- "up to 20 similar things" is sized for open-ended
        # discovery, and within one small target it mostly just inflates how
        # many candidates need an LLM judgment.
        top_k = self.restricted_candidate_top_k if restrict_to_document_versions else self.candidate_top_k
        search_kwargs = dict(
            name_text=node_name,
            context_text=context_text(node_name, entity.description),
            lexical_terms=lexical_terms(lexical_text(node_name, entity.description)),
            top_k=top_k,
            group_by=CANONICAL_FIELD,
            name_score_threshold=self.relatedness_floor,
            context_score_threshold=self.relatedness_floor,
        )
        if restrict_to_document_versions:
            # A restricted link (documents/group mode) already knows exactly
            # which document(s) it's comparing against -- searching the whole
            # corpus and discarding everything outside that set wastes the
            # search AND can silently miss a real match that just didn't make
            # the corpus-wide top-k. One search per target document instead,
            # each through a VectorStore scoped to exactly that document (see
            # _vdb_for_document) -- the restriction is the store's own scope,
            # not a per-call filter, so a target document can never leak into
            # another's share of candidates.
            hits = []
            for document_version_id in restrict_to_document_versions:
                scoped_vdb = self._vdb_for_document(document_version_id)
                for hit in scoped_vdb.search_entities(**search_kwargs):
                    # Tagged per-hit, before collapse_entity_hits picks the
                    # best-scoring one per entity, so the winning hit still
                    # carries which target document it was actually found in
                    # -- collapse_entity_hits copies the whole dict, extra
                    # keys and all.
                    hit = dict(hit)
                    hit["_source_document_version_id"] = document_version_id
                    hits.append(hit)
            # Collapse limit scales with the number of targets -- top_k caps
            # each target's own search, not the group's combined total, or a
            # group of several documents would arbitrarily lose a later
            # target's candidates to an earlier target's higher-scoring ones.
            collapse_limit = top_k * len(restrict_to_document_versions)
        else:
            # Auto mode: no target set to scope to -- this is a genuine
            # whole-corpus discovery search.
            hits = self._vdb.search_entities(**search_kwargs)
            collapse_limit = top_k
        collapsed = collapse_entity_hits(hits, limit=collapse_limit)
        candidates: list[CrossDocCandidate] = []
        for hit in collapsed:
            metadata = hit.get("metadata") or {}
            other_node_name = metadata.get("node_id") or metadata.get(CANONICAL_FIELD)
            if not other_node_name:
                continue
            canonical, entity_kind = split_node_name(str(other_node_name))
            if not canonical or entity_kind == "grouping":
                continue
            if entity_kind in STRUCTURAL_ENTITY_KINDS:
                continue
            score = 1.0 - float(hit["distance"])
            if not (self.relatedness_floor <= score < self.merge_threshold):
                continue
            if canonical == entity.canonical and entity_kind == entity.entity_kind:
                continue
            other_key = self._store_call(
                self._store.entity_key_for_canonical, canonical, entity_kind or "",
            )
            if other_key is None:
                continue
            if restrict_to_revisions is not None:
                other_revisions = self._store_call(self._store.revisions_for_entity, other_key)
                if other_revisions.isdisjoint(restrict_to_revisions):
                    continue
            candidates.append(
                CrossDocCandidate(
                    entity_a=EntityView(
                        entity_key=other_key, canonical=str(canonical),
                        entity_kind=str(entity_kind),
                        description=str(metadata.get("description", "")),
                    ),
                    entity_b=entity,
                    embedding_score=score,
                    entity_a_document_version_id=hit.get("_source_document_version_id"),
                    entity_b_document_version_id=own_document_version_id,
                )
            )
        return candidates

    def _is_cross_document(self, entity_a_key: int, entity_b_key: int) -> bool:
        revisions_a = self._store_call(self._store.revisions_for_entity, entity_a_key)
        revisions_b = self._store_call(self._store.revisions_for_entity, entity_b_key)
        return revisions_a.isdisjoint(revisions_b)

    # -- candidate judgment (batched) --------------------------------------------

    def judge_pairs(
        self,
        entity_b: EntityView,
        items: Sequence[tuple[CrossDocCandidate, Sequence[GroupingMembership]]],
    ) -> dict[int, CrossDocPairVerdict]:
        """One LLM call judging every pending candidate for one query entity
        (B) at once. A candidate the model's response omits is treated as "no
        relation, no grouping" -- never retried on its own, the same
        skip-and-move-on contract kg_extractor's own batching uses for a
        partial failure."""

        blocks = []
        for index, (candidate, ambiguous_groupings) in enumerate(items):
            groupings_block = (
                "\n".join(f"- ({g.grouping_key}) {g.label}: {g.description}" for g in ambiguous_groupings)
                if ambiguous_groupings else _no_groupings_block()
            )
            blocks.append(
                f"[{index}]\n"
                f"A: {candidate.entity_a.canonical} ({candidate.entity_a.entity_kind}) -- "
                f"{candidate.entity_a.description}\n"
                f"A's existing groupings where B's fit is ambiguous:\n"
                f"{groupings_block}"
            )
        prompt = CROSS_DOC_BATCH_PROMPT.format(
            b_text=f"{entity_b.canonical} ({entity_b.entity_kind}): {entity_b.description}",
            candidates_block="\n\n".join(blocks),
        )
        batch = self._llm.get_json_completion(prompt=prompt, schema=CrossDocBatchVerdict)
        return {
            verdict.candidate_index: verdict
            for verdict in batch.verdicts
            if 0 <= verdict.candidate_index < len(items)
        }

    # -- grouping attach-vs-create -----------------------------------------------

    def _grouping_fit(self, candidate_vec, grouping: GroupingMembership) -> float:
        import numpy as np

        centroid = np.asarray(grouping.centroid, dtype=float)
        vec = np.asarray(candidate_vec, dtype=float)
        denom = (np.linalg.norm(centroid) * np.linalg.norm(vec)) or 1.0
        return float((centroid @ vec) / denom)

    def _attach_to_grouping(
        self,
        member: EntityView,
        grouping: GroupingMembership,
        *,
        weight: float,
        member_vec=None,
    ) -> None:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        with self._ledger.transaction() as connection:
            self._ledger.write_grouping_membership(
                connection, scope,
                grouping_key=grouping.grouping_key, member_key=member.entity_key,
                weight=weight, producer="cross_doc_linker",
            )
            new_count = grouping.member_count + 1
            # A caller that already embedded this member (e.g. _link_one_entity,
            # to test grouping fit) passes it through instead of paying for the
            # same embedding twice.
            if member_vec is None:
                member_vec = self._embedder.embed_texts([member.canonical])[0]
            old_centroid = list(grouping.centroid) or list(member_vec)
            new_centroid = [
                (old_centroid[i] * grouping.member_count + float(member_vec[i])) / new_count
                for i in range(len(old_centroid))
            ]
            self._ledger.update_grouping_centroid(
                connection, scope,
                grouping_key=grouping.grouping_key, centroid=new_centroid, member_count=new_count,
            )
            if new_count % self.description_refresh_every == 0:
                self._refresh_grouping_description(
                    connection, scope, grouping.grouping_key, grouping.label,
                )

    def _create_grouping(
        self, candidate: CrossDocCandidate, label: str, description: str,
    ) -> int:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        a_vec = self._embedder.embed_texts([candidate.entity_a.canonical])[0]
        b_vec = self._embedder.embed_texts([candidate.entity_b.canonical])[0]
        centroid = [(float(a_vec[i]) + float(b_vec[i])) / 2.0 for i in range(len(a_vec))]

        # A candidate elsewhere in the corpus may already have minted a
        # grouping for this same topic -- reuse it instead of fragmenting one
        # concept across two hubs with the same label.
        best_match, best_fit = None, 0.0
        for existing in self._store_call(self._store.all_groupings):
            if not existing.centroid:
                continue
            fit = self._grouping_fit(centroid, existing)
            if fit > best_fit:
                best_match, best_fit = existing, fit
        if best_match is not None and best_fit >= self.merge_threshold:
            self._attach_to_grouping(candidate.entity_a, best_match, weight=1.0, member_vec=a_vec)
            self._attach_to_grouping(candidate.entity_b, best_match, weight=1.0, member_vec=b_vec)
            return best_match.grouping_key

        with self._ledger.transaction() as connection:
            return self._ledger.intern_grouping(
                connection, scope,
                label=label or f"{candidate.entity_a.canonical} / {candidate.entity_b.canonical}",
                description=description,
                founding_members=(candidate.entity_a.entity_key, candidate.entity_b.entity_key),
                centroid=centroid,
            )

    def _refresh_grouping_description(
        self, connection, scope: CorpusScope, grouping_key: int, label: str,
    ) -> None:
        members = self._store_call(self._store.members_for_grouping, grouping_key)
        names = ", ".join(m.canonical for m in members[:8])
        self._ledger.update_grouping_description(
            connection, scope, grouping_key=grouping_key,
            label=label, description=f"Concepts related to: {names}",
        )

    # -- cross-docs relation writer -----------------------------------------------

    def _write_relation(self, candidate: CrossDocCandidate, verdict: CrossDocPairVerdict) -> None:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        if verdict.relation_subject == "a":
            subject, obj = candidate.entity_a, candidate.entity_b
            subject_docv, obj_docv = (
                candidate.entity_a_document_version_id, candidate.entity_b_document_version_id,
            )
        else:
            subject, obj = candidate.entity_b, candidate.entity_a
            subject_docv, obj_docv = (
                candidate.entity_b_document_version_id, candidate.entity_a_document_version_id,
            )
        with self._ledger.transaction() as connection:
            self._ledger.write_cross_doc_relation(
                connection, scope,
                predicate=verdict.relation_predicate,
                directed=verdict.relation_directed,
                subject_key=subject.entity_key, object_key=obj.entity_key,
                weight=verdict.relation_confidence,
                confidence=candidate.embedding_score,
                producer="cross_doc_linker",
                source_document_version_id_subject=subject_docv,
                source_document_version_id_object=obj_docv,
                description=verdict.relation_description,
            )
