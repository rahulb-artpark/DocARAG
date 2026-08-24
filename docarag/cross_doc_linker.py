"""Discover and materialize cross-document groupings and cross-docs relations.

Operates strictly on persisted Postgres state -- constructed after a
document's own graph-part persist has already run. Two ways to trigger it:

``link_document`` (auto): one just-persisted document against the whole corpus.
``link_documents`` (direct): a user-chosen, closed set of already-ingested
documents against each other, ignoring the rest of the corpus. Both funnel
through the same candidate/judge/write pipeline; only the search scope differs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from Core.configs.graph_config import GraphConfig
from Core.configs.system_config import SystemConfig
from Core.pipelines.entity_facets import (
    CANONICAL_FIELD,
    collapse_entity_hits,
    context_text,
    lexical_terms,
    lexical_text,
)
from Core.pipelines.vdb_index import corpus_entity_vector_scope
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from docarag.corpus_store import (
    CorpusStore, EntityView, GroupingMembership, entity_node_name, split_node_name,
)
from docarag.document_store import _open_connection
from docarag.persistence.contracts import CorpusScope
from docarag.persistence.postgres import PostgresLedger, resolve_database_url


class GroupingAttachChoice(BaseModel):
    """One ambiguous grouping's attach-or-leave-alone verdict."""

    grouping_key: int
    attach: bool = Field(default=False)


class CrossDocPairVerdict(BaseModel):
    """One LLM call's joint verdict over a candidate pair (A, B).

    Grouping and relation outcomes are independent and non-exclusive.
    ``grouping_attach`` only ever covers groupings the embedding-vs-centroid
    gate already flagged ambiguous -- confident and out-of-band groupings
    never reach the model.
    """

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


@dataclass(frozen=True, slots=True)
class CrossDocCandidate:
    entity_a: EntityView  # pre-existing corpus entity, the embedding hit
    entity_b: EntityView  # the entity being linked
    embedding_score: float


@dataclass(slots=True)
class CrossDocLinkStats:
    entities_scanned: int = 0
    candidates_in_band: int = 0
    llm_calls: int = 0
    groupings_created: int = 0
    groupings_attached: int = 0
    relations_written: int = 0

    def __iadd__(self, other: "CrossDocLinkStats") -> "CrossDocLinkStats":
        self.entities_scanned += other.entities_scanned
        self.candidates_in_band += other.candidates_in_band
        self.llm_calls += other.llm_calls
        self.groupings_created += other.groupings_created
        self.groupings_attached += other.groupings_attached
        self.relations_written += other.relations_written
        return self


CROSS_DOC_PAIR_PROMPT = """\
-Goal-
Decide whether two entities from different documents in the same corpus are
meaningfully connected, and how.

-Entities-
A (already in the corpus): {a_text}
B (being linked): {b_text}

-Existing groupings A already belongs to that B is a borderline fit for-
{groupings_block}

-Task-
1. For each listed grouping, decide whether B genuinely fits that grouping's
   theme (attach) or not.
2. Separately, decide whether A and B share a facet not covered by any listed
   grouping -- if so, propose a new grouping with a short label and
   description.
3. Separately, decide whether a specific, nameable relationship exists between
   A and B (e.g. "is a specialization of", "is compared against",
   "supports"). If so, give the predicate, which of A or B is the subject,
   whether it is directed, a one-sentence description, and your confidence
   (0-1) in the relationship's strength.

Only report what the text supports. When in doubt, prefer no relation and no
new grouping over a fabricated one.
"""


def _no_groupings_block() -> str:
    return "(none)"


class CrossDocLinker:
    """Cross-link one document's entities against a corpus scope."""

    __slots__ = (
        "_ledger", "_store", "_vdb", "_embedder", "_llm", "_read_connection",
        "tenant_id", "corpus_id",
        "merge_threshold", "relatedness_floor",
        "candidate_top_k", "description_refresh_every",
        "_closed",
    )

    def __init__(
        self,
        ledger: PostgresLedger,
        store: CorpusStore,
        vdb: VectorStore,
        embedder: TextEmbeddingProvider,
        llm: LLM,
        *,
        tenant_id: str,
        corpus_id: str,
        merge_threshold: float,
        relatedness_floor: float,
        candidate_top_k: int = 20,
        description_refresh_every: int = 5,
        read_connection: object | None = None,
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
        if isinstance(description_refresh_every, bool) or description_refresh_every < 1:
            raise ValueError("description_refresh_every must be at least 1")

        self._ledger = ledger
        self._store = store
        self._vdb = vdb
        self._embedder = embedder
        self._llm = llm
        self.tenant_id = tenant_id
        self.corpus_id = corpus_id
        self.merge_threshold = merge_threshold
        self.relatedness_floor = relatedness_floor
        self.candidate_top_k = candidate_top_k
        self.description_refresh_every = description_refresh_every
        self._read_connection = read_connection
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
            ledger, store, vdb, embedder, llm,
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
            read_connection=read_connection,
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

    def link_documents(self, revision_keys: Sequence[int]) -> CrossDocLinkStats:
        """Direct mode: cross-link exactly this user-chosen set against each other."""

        keys = [int(k) for k in revision_keys]
        if len(keys) < 2:
            raise ValueError("link_documents needs at least two documents")
        total = CrossDocLinkStats()
        for revision_key in keys:
            restrict = frozenset(keys) - {revision_key}
            total += self._link_one(revision_key, restrict_to_revisions=restrict)
        return total

    def _link_one(
        self, revision_key: int, *, restrict_to_revisions: frozenset[int] | None,
    ) -> CrossDocLinkStats:
        stats = CrossDocLinkStats()
        for entity in self._store.entities_for_revision(revision_key):
            stats.entities_scanned += 1
            for candidate in self._search_cross_document_matches(
                entity, restrict_to_revisions=restrict_to_revisions,
            ):
                if not self._is_cross_document(candidate.entity_a.entity_key, entity.entity_key):
                    continue
                stats.candidates_in_band += 1
                self._process_candidate(candidate, stats)
        return stats

    # -- candidate generation --------------------------------------------------

    def discover_candidates(
        self, revision_key: int, *, restrict_to_revisions: frozenset[int] | None = None,
    ) -> list[CrossDocCandidate]:
        found: list[CrossDocCandidate] = []
        for entity in self._store.entities_for_revision(revision_key):
            found.extend(
                self._search_cross_document_matches(
                    entity, restrict_to_revisions=restrict_to_revisions,
                )
            )
        return found

    def _search_cross_document_matches(
        self, entity: EntityView, *, restrict_to_revisions: frozenset[int] | None,
    ) -> list[CrossDocCandidate]:
        # Entity points embed the combined "Name: x\nType: y" form (see
        # kg_refiner.get_vdb_meta_data / get_node_name_from_entity), not the
        # bare canonical -- the query text has to match that shape to be
        # comparable against the stored name/context vectors.
        node_name = entity_node_name(entity.canonical, entity.entity_kind)
        hits = self._vdb.search_entities(
            name_text=node_name,
            context_text=context_text(node_name, entity.description),
            lexical_terms=lexical_terms(lexical_text(node_name, entity.description)),
            top_k=self.candidate_top_k,
            group_by=CANONICAL_FIELD,
            name_score_threshold=self.relatedness_floor,
            context_score_threshold=self.relatedness_floor,
        )
        collapsed = collapse_entity_hits(hits, limit=self.candidate_top_k)
        candidates: list[CrossDocCandidate] = []
        for hit in collapsed:
            metadata = hit.get("metadata") or {}
            other_node_name = metadata.get("node_id") or metadata.get(CANONICAL_FIELD)
            if not other_node_name:
                continue
            canonical, entity_kind = split_node_name(str(other_node_name))
            if not canonical or entity_kind == "grouping":
                continue
            score = 1.0 - float(hit["distance"])
            if not (self.relatedness_floor <= score < self.merge_threshold):
                continue
            if canonical == entity.canonical and entity_kind == entity.entity_kind:
                continue
            other_key = self._store.entity_key_for_canonical(canonical, entity_kind or "")
            if other_key is None:
                continue
            if restrict_to_revisions is not None:
                if self._store.revisions_for_entity(other_key).isdisjoint(restrict_to_revisions):
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
                )
            )
        return candidates

    def _is_cross_document(self, entity_a_key: int, entity_b_key: int) -> bool:
        return self._store.revisions_for_entity(entity_a_key).isdisjoint(
            self._store.revisions_for_entity(entity_b_key)
        )

    # -- per-candidate processing -----------------------------------------------

    def _process_candidate(self, candidate: CrossDocCandidate, stats: CrossDocLinkStats) -> None:
        groupings = self._store.groupings_for_entity(candidate.entity_a.entity_key)
        b_vec = self._embedder.embed_texts([candidate.entity_b.canonical])[0]

        confident: list[GroupingMembership] = []
        ambiguous: list[GroupingMembership] = []
        for grouping in groupings:
            if not grouping.centroid:
                ambiguous.append(grouping)
                continue
            fit = self._grouping_fit(b_vec, grouping)
            (confident if fit >= self.relatedness_floor else ambiguous).append(grouping)

        for grouping in confident:
            self._attach_to_grouping(candidate.entity_b, grouping, weight=candidate.embedding_score)
            stats.groupings_attached += 1

        verdict: CrossDocPairVerdict | None = None
        if ambiguous or not groupings:
            verdict = self.judge_pair(candidate, ambiguous)
            stats.llm_calls += 1

        if verdict is None:
            return

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

    def judge_pair(
        self, candidate: CrossDocCandidate, ambiguous_groupings: Sequence[GroupingMembership],
    ) -> CrossDocPairVerdict:
        groupings_block = (
            "\n".join(f"- ({g.grouping_key}) {g.label}: {g.description}" for g in ambiguous_groupings)
            if ambiguous_groupings else _no_groupings_block()
        )
        prompt = CROSS_DOC_PAIR_PROMPT.format(
            a_text=f"{candidate.entity_a.canonical} ({candidate.entity_a.entity_kind}): {candidate.entity_a.description}",
            b_text=f"{candidate.entity_b.canonical} ({candidate.entity_b.entity_kind}): {candidate.entity_b.description}",
            groupings_block=groupings_block,
        )
        return self._llm.get_json_completion(prompt=prompt, schema=CrossDocPairVerdict)

    # -- grouping attach-vs-create -----------------------------------------------

    def _grouping_fit(self, candidate_vec, grouping: GroupingMembership) -> float:
        import numpy as np

        centroid = np.asarray(grouping.centroid, dtype=float)
        vec = np.asarray(candidate_vec, dtype=float)
        denom = (np.linalg.norm(centroid) * np.linalg.norm(vec)) or 1.0
        return float((centroid @ vec) / denom)

    def _attach_to_grouping(
        self, member: EntityView, grouping: GroupingMembership, *, weight: float,
    ) -> None:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        with self._ledger.transaction() as connection:
            self._ledger.write_grouping_membership(
                connection, scope,
                grouping_key=grouping.grouping_key, member_key=member.entity_key,
                weight=weight, producer="cross_doc_linker",
            )
            new_count = grouping.member_count + 1
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
                self._refresh_grouping_description(connection, scope, grouping.grouping_key)

    def _create_grouping(
        self, candidate: CrossDocCandidate, label: str, description: str,
    ) -> int:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        a_vec = self._embedder.embed_texts([candidate.entity_a.canonical])[0]
        b_vec = self._embedder.embed_texts([candidate.entity_b.canonical])[0]
        centroid = [(float(a_vec[i]) + float(b_vec[i])) / 2.0 for i in range(len(a_vec))]
        with self._ledger.transaction() as connection:
            return self._ledger.intern_grouping(
                connection, scope,
                label=label or f"{candidate.entity_a.canonical} / {candidate.entity_b.canonical}",
                description=description,
                founding_members=(candidate.entity_a.entity_key, candidate.entity_b.entity_key),
                centroid=centroid,
            )

    def _refresh_grouping_description(self, connection, scope: CorpusScope, grouping_key: int) -> None:
        members = self._store.members_for_grouping(grouping_key)
        label = ", ".join(m.canonical for m in members[:8])
        self._ledger.update_grouping_description(
            connection, scope, grouping_key=grouping_key,
            label=label, description=f"Concepts related to: {label}",
        )

    # -- cross-docs relation writer -----------------------------------------------

    def _write_relation(self, candidate: CrossDocCandidate, verdict: CrossDocPairVerdict) -> None:
        scope = CorpusScope(tenant_id=self.tenant_id, corpus_id=self.corpus_id)
        subject, obj = (
            (candidate.entity_a, candidate.entity_b) if verdict.relation_subject == "a"
            else (candidate.entity_b, candidate.entity_a)
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
                description=verdict.relation_description,
            )
