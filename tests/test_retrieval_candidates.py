from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import math
import unittest
from unittest.mock import patch

from docarag.domain.base import ContractValidationError
from docarag.retrieval_candidates import (
    ABSOLUTE_MAX_LEGS,
    ABSOLUTE_MAX_RRF_K,
    MAX_CANONICAL_REVISIONS_PER_HIT,
    MAX_SELECTED_DOCUMENTS,
    MAX_SUPPORTING_HITS_PER_CONTRIBUTION,
    CandidateFusionResult,
    CandidateHit,
    CandidateLeg,
    CandidateQueryScope,
    DocumentEvidenceBinding,
    FusedCandidate,
    FusionBounds,
    LegRankTrace,
    PlannedRetrievalLeg,
    RRFContribution,
    fuse_candidates,
)


class RetrievalCandidateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = FusionBounds(
            max_legs=4,
            max_hits_per_leg=8,
            max_output=5,
        )
        self.document_bindings = (
            self._binding("doc-a", "doc-a-v2"),
            self._binding("doc-b", "doc-b-v1"),
        )
        default_plan = PlannedRetrievalLeg(
            leg_id="dense-text",
            leg_kind="dense",
            backend_id="dense",
            index_revision_id="dense-r4",
            leg_config_sha256=self._sha("dense-text:config"),
            max_hits=self.bounds.max_hits_per_leg,
            budget_ms=2_000,
        )
        self.scope = self._scope((default_plan,))

    @staticmethod
    def _sha(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _binding(
        self,
        document: str,
        version: str,
        *,
        graph_release: str | None = None,
        canonical_revision_ids: tuple[str, ...] = (),
        canonical_representation_ids: tuple[str, ...] = (),
        canonical_asset_ids: tuple[str, ...] = (),
    ) -> DocumentEvidenceBinding:
        return DocumentEvidenceBinding(
            document_id=document,
            document_version_id=version,
            source_sha256=self._sha(f"{document}:{version}:source"),
            canonical_artifact_sha256=self._sha(
                f"{document}:{version}:canonical"
            ),
            tree_revision_id=f"tree-{document}-{version}",
            evidence_graph_release_id=graph_release,
            evidence_graph_release_sha256=(
                self._sha(f"{document}:{version}:{graph_release}:evidence-graph")
                if graph_release is not None
                else None
            ),
            canonical_revision_ids=canonical_revision_ids,
            canonical_representation_ids=canonical_representation_ids,
            canonical_asset_ids=canonical_asset_ids,
        )

    def _scope(
        self,
        plans: tuple[PlannedRetrievalLeg, ...],
        *,
        bindings: tuple[DocumentEvidenceBinding, ...] | None = None,
    ) -> CandidateQueryScope:
        ready: dict[str, str] = {}
        for plan in plans:
            previous = ready.setdefault(plan.backend_id, plan.index_revision_id)
            if previous != plan.index_revision_id:
                raise AssertionError("test scope cannot ready two revisions per backend")
        return CandidateQueryScope(
            tenant_id="tenant-a",
            corpus_id="corpus-a",
            filter_identity="filter-v3",
            policy_identity="policy-v7",
            query_sha256=self._sha("query"),
            retrieval_config_sha256=self._sha("retrieval-config"),
            document_bindings=bindings or self.document_bindings,
            planned_legs=plans,
            ready_index_revisions=ready,
        )

    def _scope_for(
        self,
        legs: tuple[CandidateLeg, ...],
        *,
        bounds: FusionBounds | None = None,
    ) -> CandidateQueryScope:
        effective_bounds = bounds or self.bounds
        bindings: list[DocumentEvidenceBinding] = []
        for binding in self.document_bindings:
            matching_hits = tuple(
                hit
                for leg in legs
                for hit in leg.hits
                if hit.document_id == binding.document_id
                and hit.evidence_binding_id == binding.evidence_binding_id
            )
            bindings.append(
                replace(
                    binding,
                    canonical_revision_ids=tuple(
                        revision_id
                        for hit in matching_hits
                        for revision_id in hit.canonical_revision_ids
                    ),
                    canonical_representation_ids=tuple(
                        representation_id
                        for hit in matching_hits
                        for representation_id in hit.canonical_representation_ids
                    ),
                    canonical_asset_ids=tuple(
                        asset_id
                        for hit in matching_hits
                        for asset_id in hit.canonical_asset_ids
                    ),
                )
            )
        return self._scope(
            tuple(
                PlannedRetrievalLeg(
                    leg_id=leg.leg_id,
                    leg_kind=leg.leg_kind,
                    backend_id=leg.backend_id,
                    index_revision_id=leg.index_revision_id,
                    leg_config_sha256=self._sha(f"{leg.leg_id}:config"),
                    max_hits=effective_bounds.max_hits_per_leg,
                    budget_ms=2_000,
                )
                for leg in legs
            ),
            bindings=tuple(bindings),
        )

    def _hit(
        self,
        hit_id: str,
        evidence_id: str,
        rank: int,
        *,
        backend: str = "dense",
        revision: str | None = None,
        score: float = 0.5,
        evidence_kind: str = "canonical",
        tenant: str = "tenant-a",
        corpus: str = "corpus-a",
        document: str = "doc-a",
        document_version: str = "doc-a-v2",
        policy: str = "policy-v7",
        filter_id: str = "filter-v3",
        tree_node: str | None = None,
        representation: str | None = None,
        asset: str | None = None,
        leg_kind: str | None = None,
        graph_release: str | None = None,
    ) -> CandidateHit:
        if revision is None:
            revision = "dense-r4" if backend == "dense" else "sparse-r9"
        evidence = (
            {"canonical_evidence_id": evidence_id}
            if evidence_kind == "canonical"
            else {"derived_evidence_id": evidence_id}
        )
        binding = self._binding(
            document, document_version, graph_release=graph_release
        )
        if leg_kind is None:
            leg_kind = "dense" if backend == "dense" else "sparse"
        return CandidateHit(
            hit_id=hit_id,
            tenant_id=tenant,
            corpus_id=corpus,
            document_id=document,
            document_version_id=document_version,
            source_sha256=binding.source_sha256,
            canonical_artifact_sha256=binding.canonical_artifact_sha256,
            tree_revision_id=binding.tree_revision_id,
            evidence_graph_release_id=binding.evidence_graph_release_id,
            evidence_graph_release_sha256=binding.evidence_graph_release_sha256,
            tree_node_id=tree_node or f"tree-{hit_id}",
            modality="table" if representation else "image" if asset else "text",
            backend_id=backend,
            leg_kind=leg_kind,
            index_revision_id=revision,
            raw_score=score,
            rank=rank,
            payload_identity=self._sha(f"payload-{hit_id}"),
            filter_identity=filter_id,
            policy_identity=policy,
            canonical_revision_ids=(f"revision-{evidence_id}",),
            canonical_representation_ids=(representation,) if representation else (),
            canonical_asset_ids=(asset,) if asset else (),
            selected_representation_ids=(representation,) if representation else (),
            selected_asset_ids=(asset,) if asset else (),
            **evidence,
        )

    def _leg(
        self,
        leg_id: str,
        hits: tuple[CandidateHit, ...],
        *,
        backend: str = "dense",
        revision: str | None = None,
        leg_kind: str | None = None,
    ) -> CandidateLeg:
        if revision is None:
            revision = "dense-r4" if backend == "dense" else "sparse-r9"
        if leg_kind is None:
            leg_kind = "dense" if backend == "dense" else "sparse"
        return CandidateLeg(
            leg_id=leg_id,
            leg_kind=leg_kind,
            backend_id=backend,
            index_revision_id=revision,
            query_sha256=self._sha("query"),
            retrieval_config_sha256=self._sha("retrieval-config"),
            leg_config_sha256=self._sha(f"{leg_id}:config"),
            hits=hits,
            status="success" if hits else "empty",
            latency_ms=7,
            attempt_count=1,
        )

    def _fuse(
        self,
        legs: tuple[CandidateLeg, ...],
        *,
        bounds: FusionBounds | None = None,
        scope: CandidateQueryScope | None = None,
        rrf_k: int = 60,
    ) -> CandidateFusionResult:
        effective_bounds = bounds or self.bounds
        return fuse_candidates(
            legs,
            scope=scope or self._scope_for(legs, bounds=effective_bounds),
            bounds=effective_bounds,
            rrf_k=rrf_k,
        )

    def test_candidate_and_full_fusion_round_trip_are_immutable(self) -> None:
        hit = self._hit(
            "dense-1",
            "block-table-1",
            1,
            representation="repr-table-grid",
            asset="asset-table-crop",
        )
        self.assertEqual(hit, CandidateHit.from_json(hit.to_json()))
        self.assertEqual("canonical", hit.evidence_kind)
        self.assertEqual("block-table-1", hit.evidence_id)
        with self.assertRaises(FrozenInstanceError):
            hit.rank = 3  # type: ignore[misc]
        with self.assertRaises(TypeError):
            self.scope.allowed_document_versions["doc-c"] = "doc-c-v1"  # type: ignore[index]

        result = self._fuse(
            (self._leg("dense-text", (hit,)),),
            rrf_k=60,
        )
        restored = CandidateFusionResult.from_json(result.to_json())
        self.assertEqual(result, restored)
        self.assertEqual(result.to_json(), restored.to_json())
        self.assertEqual(
            ("repr-table-grid",),
            result.candidates[0].representative_hit.selected_representation_ids,
        )
        self.assertEqual(
            ("asset-table-crop",),
            result.candidates[0].representative_hit.selected_asset_ids,
        )

        tampered = result.to_dict()
        tampered["fusion_id"] = "candidate_fusion_" + "0" * 32
        with self.assertRaisesRegex(ContractValidationError, "fusion_id"):
            CandidateFusionResult.from_dict(tampered)

        false_count = result.to_dict()
        false_count["eligible_candidate_count"] = 2
        with self.assertRaisesRegex(
            ContractValidationError, "eligible_candidate_count"
        ):
            CandidateFusionResult.from_dict(false_count)

    def test_ordinary_rrf_is_deterministic_and_deduplicates_evidence(self) -> None:
        dense = self._leg(
            "dense-text",
            (
                self._hit("dense-b", "block-b", 2, score=999.0),
                self._hit(
                    "dense-a",
                    "block-a",
                    1,
                    score=-10.0,
                    tree_node="tree-a-dense",
                    representation="repr-a",
                ),
            ),
        )
        sparse = self._leg(
            "sparse-bm25",
            (
                self._hit(
                    "sparse-a-duplicate",
                    "block-a",
                    3,
                    backend="sparse",
                    score=10_000.0,
                    tree_node="tree-a-other",
                ),
                self._hit(
                    "sparse-a",
                    "block-a",
                    2,
                    backend="sparse",
                    score=-5.0,
                    asset="asset-a-crop",
                ),
                self._hit(
                    "sparse-b", "block-b", 1, backend="sparse", score=-100.0
                ),
            ),
            backend="sparse",
        )

        scope = self._scope_for((dense, sparse))
        first = self._fuse((sparse, dense), scope=scope, rrf_k=60)
        second = self._fuse((dense, sparse), scope=scope, rrf_k=60)

        self.assertEqual(first.to_json(), second.to_json())
        restored = CandidateFusionResult.from_json(first.to_json())
        self.assertEqual(first, restored)
        self.assertEqual(2, len(restored.leg_traces))
        # A and B have the same fused score.  Evidence identity, not raw score or
        # caller insertion order, is the deterministic tie break.
        self.assertEqual(
            ["block-a", "block-b"],
            [candidate.evidence_id for candidate in first.candidates],
        )
        self.assertAlmostEqual(
            1 / 61 + 1 / 62,
            first.candidates[0].fused_score,
        )
        self.assertEqual(
            ("sparse-a", "sparse-a-duplicate"),
            first.candidates[0].contributions[1].supporting_hit_ids,
        )
        self.assertEqual(2, len(first.candidates))
        self.assertEqual(2, first.eligible_candidate_count)
        self.assertEqual(
            ("tree-a-dense", "tree-a-other", "tree-sparse-a"),
            first.candidates[0].tree_node_ids,
        )
        self.assertEqual(
            ("repr-a",), first.candidates[0].canonical_representation_ids
        )
        self.assertEqual(
            ("asset-a-crop",), first.candidates[0].canonical_asset_ids
        )
        self.assertEqual(
            ("repr-a",), first.candidates[0].selected_representation_ids
        )
        self.assertEqual(
            ("asset-a-crop",), first.candidates[0].selected_asset_ids
        )
        self.assertEqual(
            ("image", "table", "text"), first.candidates[0].modalities
        )

    def test_canonical_and_derived_identities_do_not_collapse_each_other(self) -> None:
        leg = self._leg(
            "dense-text",
            (
                self._hit("canonical", "shared-id", 1),
                self._hit("derived", "shared-id", 2, evidence_kind="derived"),
                self._hit(
                    "canonical-other-tree",
                    "shared-id",
                    3,
                    tree_node="tree-other",
                ),
            ),
        )

        result = self._fuse((leg,), rrf_k=60)

        self.assertEqual(2, result.eligible_candidate_count)
        self.assertEqual(
            [("canonical", "shared-id"), ("derived", "shared-id")],
            [(item.evidence_kind, item.evidence_id) for item in result.candidates],
        )
        canonical = result.candidates[0]
        self.assertEqual(
            ("canonical", "canonical-other-tree"),
            canonical.contributions[0].supporting_hit_ids,
        )

    def test_exact_scope_filters_are_applied_after_tenant_isolation(self) -> None:
        dense = self._leg(
            "dense-text",
            (
                self._hit("eligible", "block-a", 1),
                self._hit(
                    "no-access", "block-b", 2, document="doc-private"
                ),
                self._hit(
                    "stale-document",
                    "block-c",
                    3,
                    document_version="doc-a-v1",
                ),
                self._hit("wrong-policy", "block-d", 4, policy="policy-old"),
                self._hit("wrong-filter", "block-e", 5, filter_id="filter-old"),
                self._hit(
                    "stale-release",
                    "block-f",
                    6,
                    graph_release="release-stale",
                ),
            ),
        )

        result = self._fuse((dense,))

        self.assertEqual(["block-a"], [item.evidence_id for item in result.candidates])
        traces = {trace.leg_id: trace for trace in result.leg_traces}
        self.assertEqual(
            {
                "document_access": 1,
                "document_version": 1,
                "filter_identity": 1,
                "policy_identity": 1,
                "evidence_graph_release": 1,
            },
            dict(traces["dense-text"].filtered_counts),
        )
        self.assertEqual(("eligible",), tuple(hit.hit_id for hit in traces["dense-text"].eligible_hits))

    def test_mixed_tenant_or_corpus_fails_the_entire_fusion(self) -> None:
        cases = (
            self._hit("tenant-leak", "block-a", 1, tenant="tenant-b"),
            self._hit("corpus-leak", "block-a", 1, corpus="corpus-b"),
        )
        for hit in cases:
            with self.subTest(hit_id=hit.hit_id):
                with self.assertRaisesRegex(ContractValidationError, "mixed query scope"):
                    self._fuse(
                        (self._leg("dense-text", (hit,)),), scope=self.scope
                    )

    def test_malformed_rank_and_score_fail_closed(self) -> None:
        for score in (math.nan, math.inf, -math.inf, 10**10_000, True, "0.2"):
            with self.subTest(score=score):
                with self.assertRaisesRegex(ContractValidationError, "raw_score"):
                    self._hit("bad-score", "block-a", 1, score=score)  # type: ignore[arg-type]
        for rank in (0, -1, True, 1.5):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(ContractValidationError, "rank"):
                    self._hit("bad-rank", "block-a", rank)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ContractValidationError, "absolute limit"):
            self._hit("bad-rank", "block-a", 10_001)

        with self.assertRaisesRegex(ContractValidationError, "contiguous"):
            self._leg(
                "rank-gap",
                (
                    self._hit("first", "block-a", 1),
                    self._hit("third", "block-b", 3),
                ),
            )
        invalid_identity = self._hit("base", "block-a", 1).to_dict()
        invalid_identity["canonical_evidence_id"] = None
        with self.assertRaisesRegex(ContractValidationError, "exactly one"):
            CandidateHit.from_dict(invalid_identity)

    def test_duplicate_conflicting_hit_ids_fail_and_exact_duplicates_do_not_inflate(self) -> None:
        first = self._hit("reused-hit", "block-a", 1)
        conflict = self._hit("reused-hit", "block-b", 1)
        with self.assertRaisesRegex(ContractValidationError, "duplicate conflicting hit_id"):
            legs = (
                self._leg("dense-a", (first,)),
                self._leg("dense-b", (conflict,)),
            )
            self._fuse(legs)

        legs = (
            self._leg("dense-b", (first,)),
            self._leg("dense-a", (first,)),
        )
        result = self._fuse(legs)
        self.assertEqual(1 / 61, result.candidates[0].fused_score)
        traces = {trace.leg_id: trace for trace in result.leg_traces}
        self.assertEqual((first,), traces["dense-a"].eligible_hits)
        self.assertEqual(
            {"duplicate_hit_id": 1}, dict(traces["dense-b"].filtered_counts)
        )

    def test_result_constructor_revalidates_rank_provenance_and_material_union(self) -> None:
        legs = (
                self._leg(
                    "dense-text",
                    (
                        self._hit("a", "block-a", 1, representation="repr-a"),
                        self._hit("b", "block-b", 2),
                    ),
                ),
                self._leg(
                    "sparse-bm25",
                    (
                        self._hit(
                            "sparse-a",
                            "block-a",
                            1,
                            backend="sparse",
                            asset="asset-a",
                        ),
                        self._hit("sparse-b", "block-b", 2, backend="sparse"),
                    ),
                    backend="sparse",
                ),
        )
        result = self._fuse(legs)
        candidate = result.candidates[0]
        contribution = candidate.contributions[0]

        cases: tuple[tuple[str, FusedCandidate, str], ...] = (
            (
                "missing-leg",
                replace(
                    candidate,
                    contributions=(
                        replace(contribution, leg_id="missing-leg"),
                        *candidate.contributions[1:],
                    ),
                ),
                "contribution leg",
            ),
            (
                "wrong-rank",
                replace(
                    candidate,
                    contributions=(
                        replace(contribution, selected_rank=2),
                        *candidate.contributions[1:],
                    ),
                ),
                "rank/raw score",
            ),
            (
                "wrong-support",
                replace(
                    candidate,
                    contributions=(
                        replace(
                            contribution,
                            supporting_hit_ids=(
                                contribution.selected_hit_id,
                                "b",
                            ),
                        ),
                        *candidate.contributions[1:],
                    ),
                ),
                "supporting hit IDs",
            ),
            (
                "wrong-rrf",
                replace(
                    candidate,
                    contributions=(
                        replace(
                            contribution,
                            rrf_contribution=contribution.rrf_contribution * 2,
                        ),
                        *candidate.contributions[1:],
                    ),
                ),
                "RRF contribution",
            ),
            (
                "wrong-score",
                replace(candidate, fused_score=candidate.fused_score * 2),
                "fused_score",
            ),
            (
                "false-material",
                replace(
                    candidate,
                    tree_node_ids=candidate.tree_node_ids + ("tree-fabricated",),
                ),
                "tree_node_ids",
            ),
        )
        for name, false_candidate, expected_error in cases:
            with self.subTest(name=name):
                false_candidates = (false_candidate, *result.candidates[1:])
                with self.assertRaisesRegex(ContractValidationError, expected_error):
                    replace(result, candidates=false_candidates)

        duplicate = replace(candidate, fused_rank=2)
        with self.assertRaisesRegex(ContractValidationError, "unique evidence keys"):
            replace(result, candidates=(candidate, duplicate))

    def test_exact_release_binding_prevents_cross_release_fusion(self) -> None:
        dense = self._leg(
            "dense-text", (self._hit("dense", "block-a", 1),)
        )
        sparse = self._leg(
            "sparse-bm25",
            (
                self._hit(
                    "stale",
                    "block-a",
                    1,
                    backend="sparse",
                    graph_release="release-stale",
                ),
            ),
            backend="sparse",
        )

        result = self._fuse((dense, sparse))

        self.assertEqual(1, len(result.candidates))
        self.assertEqual(1 / 61, result.candidates[0].fused_score)
        traces = {trace.leg_id: trace for trace in result.leg_traces}
        self.assertEqual(
            {"evidence_graph_release": 1},
            dict(traces["sparse-bm25"].filtered_counts),
        )
        self.assertEqual(
            self.document_bindings[0].evidence_binding_id,
            result.candidates[0].representative_hit.evidence_binding_id,
        )

    def test_base_search_works_before_kg_and_graph_search_requires_adoption(self) -> None:
        dense = self._leg(
            "dense-text", (self._hit("dense", "block-a", 1),)
        )
        bm25_hit = self._hit(
            "bm25",
            "block-a",
            1,
            backend="bm25",
            revision="bm25-r2",
            leg_kind="bm25",
        )
        bm25 = self._leg(
            "bm25-text",
            (bm25_hit,),
            backend="bm25",
            revision="bm25-r2",
            leg_kind="bm25",
        )

        base_result = self._fuse((dense, bm25))
        self.assertEqual(1, len(base_result.candidates))
        self.assertIsNone(
            base_result.candidates[0].evidence_graph_release_id
        )
        self.assertTrue(
            all(
                binding.evidence_graph_release_id is None
                for binding in base_result.scope.document_bindings
            )
        )

        with self.assertRaisesRegex(
            ContractValidationError, "require an adopted evidence graph release"
        ):
            self._hit(
                "entity-missing-release",
                "entity-a",
                1,
                backend="graph-index",
                revision="graph-r1",
                leg_kind="entity",
                evidence_kind="derived",
            )

        graph_plan = PlannedRetrievalLeg(
            leg_id="entity-search",
            leg_kind="entity",
            backend_id="graph-index",
            index_revision_id="graph-r1",
            leg_config_sha256=self._sha("entity-search:config"),
            max_hits=4,
            budget_ms=2_000,
        )
        with self.assertRaisesRegex(
            ContractValidationError, "require an adopted evidence graph release"
        ):
            self._scope((graph_plan,))

        entity_hit = self._hit(
            "entity-a",
            "entity-a",
            1,
            backend="graph-index",
            revision="graph-r1",
            leg_kind="entity",
            evidence_kind="derived",
            graph_release="graph-release-r1",
        )
        entity_leg = self._leg(
            "entity-search",
            (entity_hit,),
            backend="graph-index",
            revision="graph-r1",
            leg_kind="entity",
        )
        adopted = self._binding(
            "doc-a",
            "doc-a-v2",
            graph_release="graph-release-r1",
            canonical_revision_ids=entity_hit.canonical_revision_ids,
        )
        entity_scope = self._scope((graph_plan,), bindings=(adopted,))

        graph_result = self._fuse((entity_leg,), scope=entity_scope)
        self.assertEqual("graph-release-r1", graph_result.candidates[0].evidence_graph_release_id)
        self.assertEqual((), graph_result.candidates[0].selected_asset_ids)

    def test_material_selection_and_payload_digest_are_fail_closed(self) -> None:
        hit = self._hit("a", "block-a", 1, representation="repr-a")
        with self.assertRaisesRegex(ContractValidationError, "subset"):
            replace(hit, selected_representation_ids=("repr-fabricated",))
        with self.assertRaisesRegex(ContractValidationError, "payload_identity"):
            replace(hit, payload_identity="not-a-sha")

        legitimate_leg = self._leg("dense-text", (hit,))
        scope = self._scope_for((legitimate_leg,))
        fabricated_hit = replace(
            hit,
            canonical_representation_ids=("repr-fabricated",),
            selected_representation_ids=(),
        )
        filtered = self._fuse(
            (self._leg("dense-text", (fabricated_hit,)),), scope=scope
        )
        self.assertEqual((), filtered.candidates)
        self.assertEqual(
            {"canonical_material": 1},
            dict(filtered.leg_traces[0].filtered_counts),
        )

        result = self._fuse((legitimate_leg,), scope=scope)
        candidate = result.candidates[0]
        fabricated = replace(
            candidate,
            canonical_representation_ids=candidate.canonical_representation_ids
            + ("repr-fabricated",),
        )
        with self.assertRaisesRegex(
            ContractValidationError, "canonical_representation_ids"
        ):
            replace(result, candidates=(fabricated,))

    def test_exact_leg_roster_records_empty_timeout_and_rejects_clones(self) -> None:
        dense = self._leg(
            "dense-text", (self._hit("dense", "block-a", 1),)
        )
        timeout = CandidateLeg(
            leg_id="sparse-bm25",
            leg_kind="sparse",
            backend_id="sparse",
            index_revision_id="sparse-r9",
            query_sha256=self._sha("query"),
            retrieval_config_sha256=self._sha("retrieval-config"),
            leg_config_sha256=self._sha("sparse-bm25:config"),
            hits=(),
            status="timeout",
            latency_ms=2_050,
            attempt_count=2,
            error_code="deadline_exceeded",
        )
        scope = self._scope_for((dense, timeout))

        result = self._fuse((dense, timeout), scope=scope)
        traces = {trace.leg_id: trace for trace in result.leg_traces}
        self.assertEqual("success", traces["dense-text"].status)
        self.assertEqual("timeout", traces["sparse-bm25"].status)
        self.assertEqual(2_050, traces["sparse-bm25"].latency_ms)
        self.assertEqual(2_000, traces["sparse-bm25"].budget_ms)
        self.assertEqual("deadline_exceeded", traces["sparse-bm25"].error_code)

        with self.assertRaisesRegex(ContractValidationError, "exact planned leg roster"):
            self._fuse((dense,), scope=scope)
        clone = replace(dense, leg_id="dense-clone")
        with self.assertRaisesRegex(ContractValidationError, "exact planned leg roster"):
            self._fuse((dense, clone), scope=self.scope)
        with self.assertRaisesRegex(ContractValidationError, "query/config identity"):
            self._fuse(
                (replace(dense, query_sha256=self._sha("wrong-query")),),
                scope=self.scope,
            )
        with self.assertRaisesRegex(ContractValidationError, "privacy-safe"):
            replace(timeout, error_code="raw exception: user@example.com")

    def test_trace_rejects_rank_beyond_submitted_count(self) -> None:
        hit = replace(self._hit("far-rank", "block-a", 1), rank=10_000)
        plan = self.scope.planned_legs[0]
        with self.assertRaisesRegex(
            ContractValidationError, "rank cannot exceed submitted_hit_count"
        ):
            LegRankTrace(
                leg_id=plan.leg_id,
                leg_kind=plan.leg_kind,
                backend_id=plan.backend_id,
                index_revision_id=plan.index_revision_id,
                plan_leg_id=plan.plan_leg_id,
                status="success",
                latency_ms=1,
                budget_ms=plan.budget_ms,
                attempt_count=1,
                planned_max_hits=10_000,
                submitted_hit_count=1,
                eligible_hits=(hit,),
                filtered_counts={},
            )

    def test_nested_cardinality_is_checked_before_deserialization(self) -> None:
        hit = self._hit("a", "block-a", 1)
        oversized_hit = hit.to_dict()
        oversized_hit["canonical_revision_ids"] = [
            "revision"
        ] * (MAX_CANONICAL_REVISIONS_PER_HIT + 1)
        with self.assertRaisesRegex(ContractValidationError, "cardinality"):
            CandidateHit.from_dict(oversized_hit)

        leg = self._leg("dense-text", (hit,))
        oversized_leg = leg.to_dict()
        oversized_leg["hits"] = [hit.to_dict()] * 10_001
        with self.assertRaisesRegex(ContractValidationError, "cardinality"):
            CandidateLeg.from_dict(oversized_leg)

        oversized_scope = self.scope.to_dict()
        oversized_scope["document_bindings"] = [
            self.document_bindings[0].to_dict()
        ] * (MAX_SELECTED_DOCUMENTS + 1)
        with self.assertRaisesRegex(ContractValidationError, "cardinality"):
            CandidateQueryScope.from_dict(oversized_scope)

        contribution = self._fuse((leg,)).candidates[0].contributions[0]
        oversized_contribution = contribution.to_dict()
        oversized_contribution["supporting_hit_ids"] = [
            contribution.selected_hit_id
        ] * (MAX_SUPPORTING_HITS_PER_CONTRIBUTION + 1)
        with self.assertRaisesRegex(ContractValidationError, "cardinality"):
            RRFContribution.from_dict(oversized_contribution)

    def test_request_bounds_preflight_before_nested_result_construction(self) -> None:
        small = FusionBounds(max_legs=1, max_hits_per_leg=2, max_output=1)
        leg = self._leg(
            "dense-text",
            (
                self._hit("a", "block-a", 1),
                self._hit("b", "block-b", 2),
            ),
        )
        result = self._fuse((leg,), bounds=small)

        too_many_candidates = result.to_dict()
        candidate_payload = too_many_candidates["candidates"][0]  # type: ignore[index]
        too_many_candidates["candidates"] = [candidate_payload, candidate_payload]
        with patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(ContractValidationError, "cardinality"):
                CandidateFusionResult.from_dict(too_many_candidates)

        trace_over_plan = result.to_dict()
        trace = trace_over_plan["leg_traces"][0]  # type: ignore[index]
        trace["planned_max_hits"] = 3
        trace["submitted_hit_count"] = 3
        trace["eligible_hits"] = [leg.hits[0].to_dict()] * 3
        with patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "request max_hits"
            ):
                CandidateFusionResult.from_dict(trace_over_plan)

        false_eligible_count = result.to_dict()
        false_eligible_count["eligible_candidate_count"] = 0
        with patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "eligible_candidate_count"
            ):
                CandidateFusionResult.from_dict(false_eligible_count)

        missing_trace = result.to_dict()
        missing_trace["leg_traces"] = []
        with patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "exact retrieval plan"
            ):
                CandidateFusionResult.from_dict(missing_trace)

    def test_each_dto_caps_serialized_bytes_before_json_decode(self) -> None:
        hit = self._hit("a", "block-a", 1, representation="repr-a")
        leg = self._leg("dense-text", (hit,))
        result = self._fuse((leg,))
        values = (
            result.scope.document_bindings[0],
            hit,
            result.scope.planned_legs[0],
            result.scope,
            leg,
            result.bounds,
            result.leg_traces[0],
            result.candidates[0].contributions[0],
            result.candidates[0],
            result,
        )

        for value in values:
            dto = type(value)
            serialized = value.to_json()
            encoded_size = len(serialized.encode("utf-8"))
            with self.subTest(dto=dto.__name__), patch.object(
                dto, "MAX_JSON_BYTES", encoded_size - 1
            ), patch(
                "docarag.retrieval_candidates.parse_json_object",
                side_effect=AssertionError("JSON decoding must not run"),
            ):
                with self.assertRaisesRegex(
                    ContractValidationError, "serialized JSON exceeds"
                ):
                    dto.from_json(serialized)

        # Character length alone is insufficient for a multibyte text payload.
        with patch.object(FusionBounds, "MAX_JSON_BYTES", 4), patch(
            "docarag.retrieval_candidates.parse_json_object",
            side_effect=AssertionError("JSON decoding must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "serialized JSON exceeds"
            ):
                FusionBounds.from_json('"\u00e9\u00e9"')

        with patch(
            "docarag.retrieval_candidates.parse_json_object",
            side_effect=AssertionError("JSON decoding must not run"),
        ):
            with self.assertRaisesRegex(ContractValidationError, "valid UTF-8"):
                FusionBounds.from_json('"' + chr(0xD800) + '"')

    def test_planned_legs_reject_duplicate_backend_semantics(self) -> None:
        plan = self.scope.planned_legs[0]
        exact_clone = replace(plan, leg_id="dense-text-clone")
        with self.assertRaisesRegex(
            ContractValidationError, "duplicate backend query semantics"
        ):
            self._scope((plan, exact_clone))

        distinct_limit = replace(
            plan,
            leg_id="dense-text-wider",
            max_hits=plan.max_hits - 1,
        )
        with self.assertRaisesRegex(
            ContractValidationError, "duplicate backend query semantics"
        ):
            self._scope((plan, distinct_limit))

        distinct_query = replace(
            distinct_limit,
            leg_config_sha256=self._sha("dense-text:alternate-field"),
        )
        scope = self._scope((plan, distinct_query))
        self.assertEqual(2, len(scope.planned_legs))

    def test_fused_material_bounds_apply_to_unique_union(self) -> None:
        bounds = FusionBounds(max_legs=1, max_hits_per_leg=3, max_output=1)
        repeated = tuple(
            self._hit(
                f"same-{rank}",
                "same-block",
                rank,
                tree_node="same-tree",
                representation="same-repr",
                asset="same-asset",
            )
            for rank in range(1, 4)
        )
        leg = self._leg("dense-text", repeated)

        with patch(
            "docarag.retrieval_candidates.MAX_BINDINGS_PER_FUSED_CANDIDATE", 2
        ):
            result = self._fuse((leg,), bounds=bounds)

        candidate = result.candidates[0]
        self.assertEqual(("same-tree",), candidate.tree_node_ids)
        self.assertEqual(("revision-same-block",), candidate.canonical_revision_ids)
        self.assertEqual(("same-repr",), candidate.canonical_representation_ids)
        self.assertEqual(("same-asset",), candidate.canonical_asset_ids)
        self.assertEqual(
            ("same-1", "same-2", "same-3"),
            candidate.contributions[0].supporting_hit_ids,
        )

        distinct_nodes = tuple(
            replace(hit, tree_node_id=f"tree-{index}")
            for index, hit in enumerate(repeated, start=1)
        )
        distinct_leg = self._leg("dense-text", distinct_nodes)
        with patch(
            "docarag.retrieval_candidates.MAX_BINDINGS_PER_FUSED_CANDIDATE", 2
        ), self.assertRaisesRegex(ContractValidationError, "tree_node_ids"):
            self._fuse((distinct_leg,), bounds=bounds)

    def test_aggregate_byte_preflight_runs_before_nested_dtos(self) -> None:
        scope_payload = self.scope.to_dict()
        with patch(
            "docarag.retrieval_candidates.MAX_SCOPE_IDENTITY_UTF8_BYTES", 1
        ), patch.object(
            DocumentEvidenceBinding,
            "from_dict",
            side_effect=AssertionError("binding construction must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "query scope identity payload"
            ):
                CandidateQueryScope.from_dict(scope_payload)

        result = self._fuse(
            (self._leg("dense-text", (self._hit("a", "block-a", 1),)),)
        )
        with patch(
            "docarag.retrieval_candidates.MAX_TRACE_IDENTITY_UTF8_BYTES", 1
        ), patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(
                ContractValidationError, "fusion trace identity payload"
            ):
                CandidateFusionResult.from_dict(result.to_dict())

    def test_trace_declared_count_bounds_hits_before_deserialization(self) -> None:
        result = self._fuse(
            (self._leg("dense-text", (self._hit("a", "block-a", 1),)),)
        )
        trace_payload = result.leg_traces[0].to_dict()
        trace_payload["eligible_hits"] = [
            result.leg_traces[0].eligible_hits[0].to_dict(),
            result.leg_traces[0].eligible_hits[0].to_dict(),
        ]
        with patch.object(
            CandidateHit,
            "from_dict",
            side_effect=AssertionError("nested hit construction must not run"),
        ):
            with self.assertRaisesRegex(ContractValidationError, "cardinality"):
                LegRankTrace.from_dict(trace_payload)

    def test_standalone_fused_candidate_id_binds_all_evidence_materials(self) -> None:
        result = self._fuse(
            (self._leg("dense-text", (self._hit("a", "block-a", 1),)),)
        )
        candidate_payload = result.candidates[0].to_dict()
        candidate_payload["canonical_representation_ids"] = ["fabricated-repr"]

        with self.assertRaisesRegex(
            ContractValidationError, "fused_candidate_id"
        ):
            FusedCandidate.from_dict(candidate_payload)

    def test_identifier_size_is_bounded_by_utf8_bytes(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "UTF-8 bytes"):
            replace(self._hit("a", "block-a", 1), hit_id="é" * 257)
        with self.assertRaisesRegex(ContractValidationError, "UTF-8 bytes"):
            self._binding("d" * 513, "version")
        with patch(
            "docarag.retrieval_candidates.MAX_SCOPE_IDENTITY_UTF8_BYTES", 1
        ):
            with self.assertRaisesRegex(ContractValidationError, "scope identity"):
                self._scope(self.scope.planned_legs)
        with patch(
            "docarag.retrieval_candidates.MAX_TRACE_IDENTITY_UTF8_BYTES", 1
        ):
            with self.assertRaisesRegex(ContractValidationError, "identity payload"):
                self._fuse(
                    (
                        self._leg(
                            "dense-text", (self._hit("a", "block-a", 1),)
                        ),
                    ),
                    scope=self.scope,
                )

    def test_input_and_output_bounds_are_enforced_before_work(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "finite sequence"):
            fuse_candidates(
                (leg for leg in ()),  # type: ignore[arg-type]
                scope=self.scope,
                bounds=self.bounds,
            )
        with self.assertRaisesRegex(ContractValidationError, "max_legs"):
            fuse_candidates(
                tuple(self._leg(f"leg-{index}", ()) for index in range(5)),
                scope=self.scope,
                bounds=self.bounds,
            )
        small = FusionBounds(max_legs=1, max_hits_per_leg=2, max_output=1)
        too_many_hits = self._leg(
            "dense-text",
            tuple(
                self._hit(f"hit-{index}", f"block-{index}", index + 1)
                for index in range(3)
            ),
        )
        with self.assertRaisesRegex(ContractValidationError, "bounded plan"):
            fuse_candidates(
                (too_many_hits,),
                scope=self._scope_for((too_many_hits,), bounds=small),
                bounds=small,
            )
        with self.assertRaisesRegex(ContractValidationError, "absolute limit"):
            FusionBounds(
                max_legs=ABSOLUTE_MAX_LEGS + 1,
                max_hits_per_leg=1,
                max_output=1,
            )
        with self.assertRaisesRegex(ContractValidationError, "rrf_k"):
            fuse_candidates(
                (),
                scope=self.scope,
                bounds=self.bounds,
                rrf_k=ABSOLUTE_MAX_RRF_K + 1,
            )

        legs = (
                self._leg(
                    "dense-text",
                    (
                        self._hit("a", "block-a", 1),
                        self._hit("b", "block-b", 2),
                    ),
                ),
        )
        result = self._fuse(legs, bounds=small)
        self.assertEqual(2, result.eligible_candidate_count)
        self.assertEqual(1, len(result.candidates))


if __name__ == "__main__":
    unittest.main()
