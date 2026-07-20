import heapq
import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import networkx as nx
import logging

from Core.provider.embedding import TextEmbeddingProvider, MMRerankerProvider
from Core.provider.rerank import TextRerankerProvider
from Core.Index.Tree import TreeNode, NodeType
from Core.utils.table_utils import table2text
from Core.utils.utils import num_tokens, TextProcessor
from Core.prompts.gbc_prompt import TEXT_RERANKER_PROMPT, MM_RERANKER_INSTRUCTION
from Core.rag.gbc_utils import (
    enhance_graph_with_semantic_links,
    merge_ranker_scores,
    calculate_skyline,
)

log = logging.getLogger(__name__)
MAX_SEMANTIC_RERANK_CHARS = 8192
MAX_SEMANTIC_RERANK_ELEMENTS = 256
SEMANTIC_SELECTOR_METADATA_FIELDS = (
    "semantic_substructure_id",
    "semantic_structure_kind",
    "semantic_structure_availability",
    "semantic_structure_producer_kind",
    "semantic_sha256",
    "semantic_comparison_sha256",
    "semantic_element_id",
    "semantic_element_kind",
    "semantic_element_local_key",
    "semantic_element_content_sha256",
    "semantic_element_source_selector_count",
    "semantic_element_relation_count",
    "semantic_element_role",
    "semantic_element_row_index",
    "semantic_element_column_index",
    "semantic_element_row_span",
    "semantic_element_column_span",
    "semantic_element_confidence",
    "semantic_bbox_x0",
    "semantic_bbox_y0",
    "semantic_bbox_x1",
    "semantic_bbox_y1",
    "semantic_coordinate_space",
    "semantic_coordinate_origin",
)


@dataclass(frozen=True)
class HybridRetrievalBinding:
    """Immutable source/index identity needed by candidate-contract fusion."""

    tenant_id: str
    corpus_id: str
    document_id: str
    document_version_id: str
    source_sha256: str
    canonical_artifact_sha256: str
    tree_revision_id: str
    dense_index_revision_id: str


class _CandidateContractUnavailable(ValueError):
    """A legacy tree node lacks the identity required by the v1 contract."""


class Retriever:
    def __init__(
        self,
        variant: str,
        reranker: TextRerankerProvider,
        # mm_reranker: MMRerankerProvider,
        embedder: TextEmbeddingProvider,
        alpha: float = 0.85,
        topk_ent: int = 20,
        x_percentile: float = 0.85,
        topk: int = 10,
        topk_relative_floor: float = 0.5,
        topk_min: int = 3,
        candidate_pool_size: int = 512,
        tree_vdb: Any | None = None,
        retrieval_binding: HybridRetrievalBinding | None = None,
    ):
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ValueError("topk must be a positive integer")
        if (
            isinstance(topk_relative_floor, bool)
            or not isinstance(topk_relative_floor, (int, float))
            or not 0.0 < topk_relative_floor <= 1.0
        ):
            raise ValueError("topk_relative_floor must be a number in (0, 1]")
        if isinstance(topk_min, bool) or not isinstance(topk_min, int) or topk_min < 1:
            raise ValueError("topk_min must be a positive integer")
        if (
            isinstance(candidate_pool_size, bool)
            or not isinstance(candidate_pool_size, int)
            or not 1 <= candidate_pool_size <= 4096
        ):
            raise ValueError("candidate_pool_size must be an integer in [1, 4096]")
        if candidate_pool_size < topk:
            raise ValueError("candidate_pool_size must be greater than or equal to topk")

        self.variant = variant
        self.reranker: TextRerankerProvider = reranker
        # self.mm_reranker: MMRerankerProvider = mm_reranker
        self.embedder: TextEmbeddingProvider = embedder
        self.alpha: float = alpha
        self.topk_ent: int = topk_ent
        self.x_percentile: float = x_percentile
        self.topk: int = topk
        self.topk_relative_floor: float = float(topk_relative_floor)
        self.topk_min: int = topk_min
        self.candidate_pool_size: int = candidate_pool_size
        self.tree_vdb = tree_vdb
        self.retrieval_binding = retrieval_binding
        self.last_fusion_result = None
        self.last_semantic_selector_by_node: dict[int, dict[str, object]] = {}

    @staticmethod
    def _bounded_scalar_text(value: object, *, max_chars: int = 512) -> str:
        """Project nested semantic content without serializing a full graph."""

        parts: list[str] = []
        chars = 0
        visited = 0
        stack = [value]
        while stack and visited < 128 and chars < max_chars:
            current = stack.pop()
            visited += 1
            if isinstance(current, str):
                text = current.strip()
            elif isinstance(current, (int, float)) and not isinstance(current, bool):
                text = str(current)
            elif isinstance(current, Mapping):
                values = [
                    current[key]
                    for key in sorted(current, key=lambda item: str(item))[:32]
                ]
                stack.extend(reversed(values))
                continue
            elif isinstance(current, Sequence) and not isinstance(
                current, (str, bytes, bytearray)
            ):
                stack.extend(reversed(list(current[:32])))
                continue
            else:
                continue
            if not text:
                continue
            remaining = max_chars - chars
            text = text[:remaining]
            parts.append(text)
            chars += len(text)
        return " ".join(parts)[:max_chars]

    @staticmethod
    def _semantic_rerank_text(node: TreeNode) -> str:
        raw_structures = getattr(node.meta_info, "semantic_substructures", None)
        if not isinstance(raw_structures, list):
            return ""
        parts: list[str] = []
        chars = 0
        element_count = 0
        for structure in raw_structures[:32]:
            if not isinstance(structure, Mapping):
                continue
            structure_kind = structure.get("kind")
            raw_elements = structure.get("elements")
            if not isinstance(raw_elements, list):
                continue
            for element in raw_elements:
                if element_count >= MAX_SEMANTIC_RERANK_ELEMENTS:
                    break
                element_count += 1
                if not isinstance(element, Mapping):
                    continue
                context = [
                    value
                    for value in (
                        structure_kind,
                        element.get("kind"),
                        element.get("role"),
                        element.get("local_key"),
                    )
                    if isinstance(value, str) and value
                ]
                content = Retriever._bounded_scalar_text(element.get("content"))
                if content:
                    context.append(content)
                rendered = " | ".join(context)
                if not rendered:
                    continue
                remaining = MAX_SEMANTIC_RERANK_CHARS - chars
                if remaining <= 0:
                    break
                rendered = rendered[:remaining]
                parts.append(rendered)
                chars += len(rendered)
            if element_count >= MAX_SEMANTIC_RERANK_ELEMENTS or chars >= MAX_SEMANTIC_RERANK_CHARS:
                break
        return "\n".join(parts)[:MAX_SEMANTIC_RERANK_CHARS]

    @staticmethod
    def _node_rerank_text(node: TreeNode) -> str:
        """Return a safe text projection for every BookRAG evidence modality."""

        meta = node.meta_info
        values: list[str] = []
        if node.type == NodeType.TABLE:
            try:
                table_text = table2text(meta.model_dump())
            except (AttributeError, KeyError, TypeError, ValueError):
                table_text = ""
            if isinstance(table_text, str) and table_text.strip():
                values.append(table_text)
        for field_name in ("content", "caption", "footnote"):
            value = getattr(meta, field_name, None)
            if isinstance(value, str) and value.strip() and value not in values:
                values.append(value)
        semantic_text = Retriever._semantic_rerank_text(node)
        if semantic_text:
            values.append(semantic_text)
        return "\n".join(values)

    @staticmethod
    def _query_terms(query: str) -> tuple[str, ...]:
        """Return at most 64 stable Unicode query terms."""

        terms: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"[^\W_]+", query.casefold(), flags=re.UNICODE):
            term = match.group(0)
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) == 64:
                break
        return tuple(terms)

    def _bm25_candidates(
        self,
        subtree_nodes: List[TreeNode],
        query: str,
    ) -> List[Tuple[TreeNode, float]]:
        """Rank the selected subtree with bounded exact query-term BM25.

        Only frequencies for the at-most-64 query terms are retained.  The two
        linear scans avoid allocating a token matrix for a large document; the
        result heap is bounded by ``candidate_pool_size``.
        """

        query_terms = self._query_terms(query)
        if not query_terms or not subtree_nodes:
            return []
        query_set = set(query_terms)
        document_frequency = {term: 0 for term in query_terms}
        total_document_length = 0

        for node in subtree_nodes:
            present: set[str] = set()
            length = 0
            text = self._node_rerank_text(node).casefold()
            for match in re.finditer(r"[^\W_]+", text, flags=re.UNICODE):
                term = match.group(0)
                length += 1
                if term in query_set:
                    present.add(term)
            total_document_length += length
            for term in present:
                document_frequency[term] += 1

        document_count = len(subtree_nodes)
        average_length = total_document_length / max(1, document_count)
        if average_length <= 0:
            return []
        idf = {
            term: math.log(
                1.0
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            for term in query_terms
        }
        k1 = 1.2
        b = 0.75
        heap: list[tuple[float, int, int, TreeNode]] = []
        limit = self.candidate_pool_size
        for position, node in enumerate(subtree_nodes):
            frequencies = {term: 0 for term in query_terms}
            length = 0
            text = self._node_rerank_text(node).casefold()
            for match in re.finditer(r"[^\W_]+", text, flags=re.UNICODE):
                term = match.group(0)
                length += 1
                if term in frequencies:
                    frequencies[term] += 1
            normalization = k1 * (1.0 - b + b * length / average_length)
            score = 0.0
            for term, frequency in frequencies.items():
                if frequency:
                    score += idf[term] * (
                        frequency * (k1 + 1.0) / (frequency + normalization)
                    )
            if score <= 0.0:
                continue
            item = (score, -position, position, node)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

        return [
            (node, score)
            for score, _negative_position, _position, node in sorted(
                heap,
                key=lambda item: (-item[0], item[2]),
            )
        ]

    @staticmethod
    def _node_id_from_hit(hit: Mapping[str, Any]) -> int | None:
        metadata = hit.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        value = metadata.get("node_id")
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            return int(value)
        return None

    def _dense_candidates(
        self,
        subtree_nodes: List[TreeNode],
        query: str,
    ) -> List[Tuple[TreeNode, Mapping[str, Any]]]:
        """Search attested text and visual TreeVDB representations."""

        if self.tree_vdb is None or not subtree_nodes:
            return []
        allowed = {node.index_id: node for node in subtree_nodes}
        # Search a bounded oversample because the source-scoped Qdrant index is
        # document-wide while section selection may authorize one subtree.
        search_limit = min(
            4096,
            max(self.candidate_pool_size, 4 * self.candidate_pool_size),
        )
        rows = self.tree_vdb.search(query_text=query, top_k=search_limit)
        candidates: list[tuple[TreeNode, Mapping[str, Any]]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("TreeVDB search must return mapping records")
            node_id = self._node_id_from_hit(row)
            node = allowed.get(node_id)
            if node is None:
                continue
            candidates.append((node, row))
            if len(candidates) == self.candidate_pool_size:
                break
        return candidates

    @staticmethod
    def _sha256_json(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _node_material_ids(node: TreeNode, field_name: str) -> tuple[str, ...]:
        values = getattr(node.meta_info, field_name, None) or ()
        if not isinstance(values, (list, tuple)):
            raise _CandidateContractUnavailable(
                f"tree node {node.index_id} has invalid {field_name}"
            )
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return tuple(result)

    def _candidate_hit(
        self,
        *,
        node: TreeNode,
        rank: int,
        backend_id: str,
        leg_kind: str,
        index_revision_id: str,
        raw_score: float,
        payload: object,
        hit_id: str,
        filter_identity: str,
        policy_identity: str,
    ):
        from docarag.retrieval_candidates import CandidateHit

        binding = self.retrieval_binding
        if binding is None:
            raise _CandidateContractUnavailable("retrieval binding is unavailable")
        meta = node.meta_info
        direct_block = getattr(meta, "canonical_block_id", None)
        direct_revision = getattr(meta, "canonical_revision_id", None)
        derived_id = getattr(meta, "derived_evidence_id", None)
        source_revisions = self._node_material_ids(
            node, "source_canonical_revision_ids"
        )
        if isinstance(direct_block, str) and direct_block and isinstance(
            direct_revision, str
        ) and direct_revision:
            evidence = {"canonical_evidence_id": direct_block}
            canonical_revisions = (direct_revision,)
        elif isinstance(derived_id, str) and derived_id and source_revisions:
            evidence = {"derived_evidence_id": derived_id}
            canonical_revisions = source_revisions
        else:
            raise _CandidateContractUnavailable(
                f"tree node {node.index_id} lacks canonical/derived evidence identity"
            )

        representations = self._node_material_ids(
            node, "canonical_representation_ids"
        )
        assets = self._node_material_ids(node, "canonical_asset_ids")
        selected_representations = tuple(
            value
            for value in self._node_material_ids(
                node, "selected_representation_ids"
            )
            if value in representations
        )
        selected_assets = tuple(
            value
            for value in self._node_material_ids(node, "selected_asset_ids")
            if value in assets
        )
        modality = (
            node.type.value if isinstance(node.type, NodeType) else str(node.type)
        )
        if isinstance(payload, Mapping):
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                candidate_modality = metadata.get("modality")
                if isinstance(candidate_modality, str) and candidate_modality:
                    modality = candidate_modality

        return CandidateHit(
            hit_id=hit_id,
            tenant_id=binding.tenant_id,
            corpus_id=binding.corpus_id,
            document_id=binding.document_id,
            document_version_id=binding.document_version_id,
            source_sha256=binding.source_sha256,
            canonical_artifact_sha256=binding.canonical_artifact_sha256,
            tree_revision_id=binding.tree_revision_id,
            tree_node_id=str(node.index_id),
            modality=modality,
            backend_id=backend_id,
            leg_kind=leg_kind,
            index_revision_id=index_revision_id,
            raw_score=raw_score,
            rank=rank,
            payload_identity=self._sha256_json(payload),
            filter_identity=filter_identity,
            policy_identity=policy_identity,
            canonical_revision_ids=canonical_revisions,
            canonical_representation_ids=representations,
            canonical_asset_ids=assets,
            selected_representation_ids=selected_representations,
            selected_asset_ids=selected_assets,
            **evidence,
        )

    def _contract_fusion(
        self,
        *,
        subtree_nodes: List[TreeNode],
        query: str,
        dense_candidates: List[Tuple[TreeNode, Mapping[str, Any]]],
        bm25_candidates: List[Tuple[TreeNode, float]],
        dense_error: bool,
    ) -> List[TreeNode]:
        """Translate live results to the backend-neutral candidate contract."""

        from docarag.retrieval_candidates import (
            CandidateLeg,
            CandidateQueryScope,
            DocumentEvidenceBinding,
            FusionBounds,
            PlannedRetrievalLeg,
            fuse_candidates,
        )

        binding = self.retrieval_binding
        if binding is None:
            raise _CandidateContractUnavailable("retrieval binding is unavailable")
        filter_digest = hashlib.sha256()
        for node in subtree_nodes:
            filter_digest.update(str(node.index_id).encode("ascii"))
            filter_digest.update(b"\0")
        filter_identity = f"subtree-{filter_digest.hexdigest()}"
        policy_identity = "bookrag-hybrid-qdrant-bm25-v1"
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        retrieval_config_sha256 = self._sha256_json(
            {
                "candidate_pool_size": self.candidate_pool_size,
                "dense_oversample": 4,
                "rrf_k": 60,
                "policy": policy_identity,
            }
        )
        dense_backend = "qdrant-tree-dense"
        bm25_backend = "bookrag-tree-bm25"
        dense_leg_config = self._sha256_json(
            {"backend": dense_backend, "index": binding.dense_index_revision_id}
        )
        bm25_leg_config = self._sha256_json(
            {"backend": bm25_backend, "tree": binding.tree_revision_id}
        )

        dense_hits = []
        for rank, (node, row) in enumerate(dense_candidates, start=1):
            distance = row.get("distance")
            if isinstance(distance, bool) or not isinstance(distance, numbers.Real):
                raise ValueError("TreeVDB hit distance must be numeric")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                row_id = self._sha256_json(row)
            dense_hits.append(
                self._candidate_hit(
                    node=node,
                    rank=rank,
                    backend_id=dense_backend,
                    leg_kind="dense",
                    index_revision_id=binding.dense_index_revision_id,
                    raw_score=1.0 - float(distance),
                    payload=row,
                    hit_id=f"dense-{row_id}",
                    filter_identity=filter_identity,
                    policy_identity=policy_identity,
                )
            )
        bm25_hits = [
            self._candidate_hit(
                node=node,
                rank=rank,
                backend_id=bm25_backend,
                leg_kind="bm25",
                index_revision_id=binding.tree_revision_id,
                raw_score=score,
                payload={"node_id": node.index_id, "score": score},
                hit_id=f"bm25-{node.index_id}-{self._sha256_json(score)[:16]}",
                filter_identity=filter_identity,
                policy_identity=policy_identity,
            )
            for rank, (node, score) in enumerate(bm25_candidates, start=1)
        ]

        revision_ids = tuple(
            sorted(
                {
                    value
                    for hit in (*dense_hits, *bm25_hits)
                    for value in hit.canonical_revision_ids
                }
            )
        )
        representation_ids = tuple(
            sorted(
                {
                    value
                    for hit in (*dense_hits, *bm25_hits)
                    for value in hit.canonical_representation_ids
                }
            )
        )
        asset_ids = tuple(
            sorted(
                {
                    value
                    for hit in (*dense_hits, *bm25_hits)
                    for value in hit.canonical_asset_ids
                }
            )
        )
        document_binding = DocumentEvidenceBinding(
            document_id=binding.document_id,
            document_version_id=binding.document_version_id,
            source_sha256=binding.source_sha256,
            canonical_artifact_sha256=binding.canonical_artifact_sha256,
            tree_revision_id=binding.tree_revision_id,
            canonical_revision_ids=revision_ids,
            canonical_representation_ids=representation_ids,
            canonical_asset_ids=asset_ids,
        )
        plans = (
            PlannedRetrievalLeg(
                leg_id="dense-tree",
                leg_kind="dense",
                backend_id=dense_backend,
                index_revision_id=binding.dense_index_revision_id,
                leg_config_sha256=dense_leg_config,
                max_hits=self.candidate_pool_size,
                budget_ms=30_000,
            ),
            PlannedRetrievalLeg(
                leg_id="bm25-tree",
                leg_kind="bm25",
                backend_id=bm25_backend,
                index_revision_id=binding.tree_revision_id,
                leg_config_sha256=bm25_leg_config,
                max_hits=self.candidate_pool_size,
                budget_ms=30_000,
            ),
        )
        scope = CandidateQueryScope(
            tenant_id=binding.tenant_id,
            corpus_id=binding.corpus_id,
            filter_identity=filter_identity,
            policy_identity=policy_identity,
            query_sha256=query_sha256,
            retrieval_config_sha256=retrieval_config_sha256,
            document_bindings=(document_binding,),
            planned_legs=plans,
            ready_index_revisions={
                dense_backend: binding.dense_index_revision_id,
                bm25_backend: binding.tree_revision_id,
            },
        )
        dense_status = (
            "error" if dense_error else "success" if dense_hits else "empty"
        )
        legs = (
            CandidateLeg(
                leg_id="dense-tree",
                leg_kind="dense",
                backend_id=dense_backend,
                index_revision_id=binding.dense_index_revision_id,
                query_sha256=query_sha256,
                retrieval_config_sha256=retrieval_config_sha256,
                leg_config_sha256=dense_leg_config,
                hits=tuple(dense_hits),
                status=dense_status,
                error_code="backend_error" if dense_error else None,
            ),
            CandidateLeg(
                leg_id="bm25-tree",
                leg_kind="bm25",
                backend_id=bm25_backend,
                index_revision_id=binding.tree_revision_id,
                query_sha256=query_sha256,
                retrieval_config_sha256=retrieval_config_sha256,
                leg_config_sha256=bm25_leg_config,
                hits=tuple(bm25_hits),
                status="success" if bm25_hits else "empty",
            ),
        )
        bounds = FusionBounds(
            max_legs=2,
            max_hits_per_leg=self.candidate_pool_size,
            max_output=self.candidate_pool_size,
        )
        result = fuse_candidates(legs, scope=scope, bounds=bounds, rrf_k=60)
        self.last_fusion_result = result
        by_id = {str(node.index_id): node for node in subtree_nodes}
        selected: list[TreeNode] = []
        seen: set[int] = set()
        for candidate in result.candidates:
            for node_id in candidate.tree_node_ids:
                node = by_id.get(node_id)
                if node is None or node.index_id in seen:
                    continue
                seen.add(node.index_id)
                selected.append(node)
                if len(selected) == self.candidate_pool_size:
                    return selected
        return selected

    def _internal_rrf(
        self,
        dense_candidates: List[Tuple[TreeNode, Mapping[str, Any]]],
        bm25_candidates: List[Tuple[TreeNode, float]],
    ) -> List[TreeNode]:
        """Compatibility RRF for historical trees without canonical identity."""

        scores: dict[int, float] = {}
        nodes: dict[int, TreeNode] = {}
        for candidates in (dense_candidates, bm25_candidates):
            seen_leg: set[int] = set()
            for rank, (node, _raw) in enumerate(candidates, start=1):
                if node.index_id in seen_leg:
                    continue
                seen_leg.add(node.index_id)
                nodes[node.index_id] = node
                scores[node.index_id] = scores.get(node.index_id, 0.0) + 1.0 / (
                    60 + rank
                )
        ranked_ids = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
        return [nodes[node_id] for node_id in ranked_ids[: self.candidate_pool_size]]

    def _hybrid_candidates(
        self,
        subtree_nodes: List[TreeNode],
        query: str,
    ) -> List[TreeNode]:
        """Fuse dense/visual Qdrant and BM25 before bounded model reranking."""

        bm25_candidates = self._bm25_candidates(subtree_nodes, query)
        dense_error = False
        try:
            dense_candidates = self._dense_candidates(subtree_nodes, query)
        except Exception:
            dense_error = True
            dense_candidates = []
            log.exception("TreeVDB dense/visual retrieval failed; continuing with BM25")

        selected: list[TreeNode]
        try:
            selected = self._contract_fusion(
                subtree_nodes=subtree_nodes,
                query=query,
                dense_candidates=dense_candidates,
                bm25_candidates=bm25_candidates,
                dense_error=dense_error,
            )
        except _CandidateContractUnavailable:
            self.last_fusion_result = None
            selected = self._internal_rrf(dense_candidates, bm25_candidates)

        # Keep deterministic source coverage when both retrieval legs return a
        # small set (or a legacy tree cannot participate in contract fusion).
        seen = {node.index_id for node in selected}
        for node in self._bounded_text_candidates(subtree_nodes, query):
            if node.index_id in seen:
                continue
            selected.append(node)
            seen.add(node.index_id)
            if len(selected) == self.candidate_pool_size:
                break
        return selected[: self.candidate_pool_size]

    def _bounded_text_candidates(
        self, subtree_nodes: List[TreeNode], query: str
    ) -> List[TreeNode]:
        """Preselect across the full subtree in bounded memory.

        This compatibility/coverage leg prevents a vector miss from reducing a
        long selected subtree to its prefix. It retains positive lexical
        matches and fills the remainder with deterministic positions spanning
        the selected structure. Runtime is linear in inspected text plus
        ``O(N log M)`` heap work and memory is ``O(M)``, where ``M`` is
        ``candidate_pool_size``.
        """

        limit = self.candidate_pool_size
        if len(subtree_nodes) <= limit:
            return list(subtree_nodes)

        query_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in re.findall(r"\w+", query.casefold(), flags=re.UNICODE):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            query_terms.append(term)
            if len(query_terms) >= 64:
                break

        # Reserve at least a quarter of the pool for structural coverage. A
        # semantic vector/BM25 index will eventually replace this compatibility
        # preselector; the reservation prevents lexical blind spots meanwhile.
        relevance_limit = max(1, limit - max(1, limit // 4))
        relevant: list[tuple[int, int]] = []
        if query_terms:
            normalized_query = " ".join(query_terms)
            heap: list[tuple[int, int, int]] = []
            for position, node in enumerate(subtree_nodes):
                text = self._node_rerank_text(node).casefold()
                if not text:
                    continue
                text_terms = set(re.findall(r"\w+", text, flags=re.UNICODE))
                matched = sum(term in text_terms for term in query_terms)
                if matched == 0:
                    continue
                phrase_bonus = int(bool(normalized_query) and normalized_query in text)
                score = matched * 2 + phrase_bonus
                item = (score, -position, position)
                if len(heap) < relevance_limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            relevant = [(score, position) for score, _negative, position in heap]

        selected_positions = {position for _score, position in relevant}
        remaining = limit - len(selected_positions)
        if remaining > 0:
            node_count = len(subtree_nodes)
            if remaining == 1:
                coverage_positions = [node_count // 2]
            else:
                coverage_positions = [
                    round(index * (node_count - 1) / (remaining - 1))
                    for index in range(remaining)
                ]
            selected_positions.update(coverage_positions)

        # Rounding and overlap with lexical matches can leave holes. Fill them
        # in stable source order without growing beyond the configured bound.
        if len(selected_positions) < limit:
            for position in range(len(subtree_nodes)):
                selected_positions.add(position)
                if len(selected_positions) >= limit:
                    break

        return [
            subtree_nodes[position]
            for position in sorted(selected_positions)[:limit]
        ]

    def _confidence_filtered(
        self,
        ranked: List[Tuple[int, float]],
        *,
        limit: int | None = None,
        min_k: int | None = None,
    ) -> List[Tuple[int, float]]:
        """Keep the highest-scoring candidates, but stop at a real drop-off.

        ``limit`` (default ``self.topk``) is an upper bound, not a quota:
        reranker scores are calibrated relevance probabilities, so padding
        out to the limit with whatever scored next sends the answer model
        noise it has to spend tokens (and sometimes trust) discounting.
        Candidates scoring below ``topk_relative_floor`` of the best match
        are dropped, down to a floor of ``min_k`` (default ``self.topk_min``)
        so a query with uniformly modest scores still returns an answer.
        This only ever shrinks the selection -- it can never add a candidate
        the limit alone would have excluded.
        """

        limit = self.topk if limit is None else limit
        min_k = self.topk_min if min_k is None else min_k
        bounded = ranked[:limit]
        if len(bounded) <= min_k:
            return bounded
        top_score = bounded[0][1]
        floor = top_score * self.topk_relative_floor
        kept = [item for item in bounded if item[1] >= floor]
        if len(kept) < min_k:
            kept = bounded[:min_k]
        return kept

    def text_reranker(
        self, subtree_nodes: List[TreeNode], query: str
    ) -> List[Tuple[int, float]]:
        """
        Rerank the original text in the subtree based on their relevance to the query.
        Use Reranker model
        """
        candidates = self._hybrid_candidates(subtree_nodes, query)
        if not candidates:
            return []

        max_length = getattr(self.reranker, "max_length", None)
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise ValueError("reranker.max_length must be an integer")
        effective_token_length = max_length - 1000
        if effective_token_length <= 0:
            raise ValueError(
                "reranker.max_length must be greater than the 1000-token "
                "prompt reserve"
            )

        doc_text = []
        tree_ids = [node.index_id for node in candidates]
        for node in candidates:
            text = self._node_rerank_text(node)
            if num_tokens(text) > effective_token_length:
                text = TextProcessor.split_text_into_chunks(
                    text=text, max_length=effective_token_length
                )
                text = text[0]  # use the first chunk
            doc_text.append(text)

        try:
            scores = self.reranker.rerank(
                query=query, documents=doc_text, instruction=TEXT_RERANKER_PROMPT
            )
            try:
                score_count = len(scores)
            except TypeError as exc:
                raise ValueError(
                    "reranker must return a sized sequence of scores"
                ) from exc
            if score_count != len(tree_ids):
                raise ValueError(
                    "reranker returned an unexpected number of scores: "
                    f"expected {len(tree_ids)}, got {score_count}"
                )

            validated_scores = []
            for position, score in enumerate(scores):
                if (
                    isinstance(score, bool)
                    or not isinstance(score, numbers.Real)
                    or not math.isfinite(float(score))
                ):
                    raise ValueError(
                        "reranker returned a non-finite numeric score at "
                        f"position {position}"
                    )
                validated_scores.append(float(score))
        finally:
            self.reranker.clean_cache()

        ranked_res = sorted(
            zip(tree_ids, validated_scores), key=lambda item: (-item[1], item[0])
        )

        return ranked_res

    # def multimodal_reranker(
    #     self, subtree_nodes: List[TreeNode], query: str
    # ) -> List[Tuple[int, float]]:
    #     doc_mm = []
    #     tree_ids = [node.index_id for node in subtree_nodes]
    #     for node in subtree_nodes:
    #         node_type = node.type
    #         tmp_dict = {}
    #         if node_type == NodeType.TABLE or node_type == NodeType.IMAGE:
    #             img_path = node.meta_info.img_path
    #             tmp_dict["img_path"] = img_path
    #         text = node.meta_info.content
    #         tmp_dict["content"] = text
    #         tmp_dict["node_id"] = node.index_id
    #         doc_mm.append(tmp_dict)

    #     scores = self.mm_reranker.rerank_documents(
    #         query=query, doc_list=doc_mm, instruction=MM_RERANKER_INSTRUCTION
    #     )
    #     ranked_res = sorted(zip(tree_ids, scores), key=lambda x: x[1], reverse=True)
    #     # Return the ranked results as a list of tuples (tree_id, score)
    #     return ranked_res

    def graph_reranker(
        self,
        subgraph: nx.Graph,
        ent_map: Dict[str, List[str]],
        subtree_nodes: List[TreeNode],
    ) -> Tuple[List[Tuple[int, float]], List[str]]:
        ents_sim = {}
        for q_ent_name, gbc_ents in ent_map.items():
            for node_name in gbc_ents:
                if node_name not in subgraph.nodes():
                    continue
                # compute similarity between the query entity and the GBC entity
                sim = self.embedder.compute_texts_sim(q_ent_name, node_name)
                positive_sim = (sim + 1) / 2
                ents_sim[node_name] = ents_sim.get(node_name, 0.0) + positive_sim

        total_sim_sum = sum(ents_sim.values())

        # Normalize the similarity scores
        personalization_vector = {}
        if total_sim_sum > 0:
            for ent, score in ents_sim.items():
                personalization_vector[ent] = score / total_sim_sum
        else:
            return [], []

        pagerank_alpha: float = self.alpha
        # --- 步骤 1 & 2: 增强图并计算 PageRank 分数 ---
        enhanced_graph = enhance_graph_with_semantic_links(
            graph=subgraph, embedder=self.embedder, x_percentile=self.x_percentile
        )

        pagerank_scores = nx.pagerank(
            enhanced_graph,
            alpha=pagerank_alpha,
            personalization=personalization_vector,
            weight="weight",  # 可以考虑使用相似度作为边的权重
        )

        # sort the ranked list by score and filter by topk_ent
        res_entities = [(ent, score) for ent, score in pagerank_scores.items()]
        res_entities = sorted(res_entities, key=lambda item: (-item[1], item[0]))
        res_entities = res_entities[: self.topk_ent]
        res_entities = [ent for ent, _ in res_entities]
        log.info(
            f"Graph reranker: Retrieved {len(res_entities)} entities with topk={self.topk_ent}."
        )

        # --- 步骤 3 & 4: 聚合分数到 Tree Node ID 并排序返回 ---
        # 3.1 创建一个目标 tree_id 的集合，用于快速查找
        target_tree_ids = {node.index_id for node in subtree_nodes}
        # 3.2 Only create candidates backed by at least one source-linked entity.
        # Text fusion can still introduce graphless candidates in standard mode;
        # graph-only mode must not return arbitrary zero-score tree nodes.
        tree_node_scores = {}

        # 3.3 遍历 PageRank 的结果
        for entity_name, score in pagerank_scores.items():
            # 从图中获取该实体节点的属性
            # .get('source_ids', set()) 是一种安全的方式，防止节点没有该属性时出错
            node_attributes = subgraph.nodes.get(entity_name, {})
            source_ids = node_attributes.get("source_ids", set())

            # 遍历该实体的所有 source_id
            for source_id in source_ids:
                # 如果这个 source_id 是我们关心的目标 tree_id
                if source_id in target_tree_ids:
                    # 将该实体的 PageRank 分数累加到对应的 tree_node_scores 中
                    tree_node_scores[source_id] = (
                        tree_node_scores.get(source_id, 0.0) + score
                    )

        # 4.1 将聚合后的分数字典转换为 (id, score) 的元组列表
        aggregated_ranked_list = [
            (tree_id, score)
            for tree_id, score in tree_node_scores.items()
            if score > 0
        ]

        # 4.2 按聚合后的分数降序排序
        sorted_ranked_list = sorted(
            aggregated_ranked_list, key=lambda item: (-item[1], item[0])
        )

        return sorted_ranked_list, res_entities

    def skyline_filter(
        self,
        sub_query: str,
        subtree_nodes: List[TreeNode],
        subgraph: nx.Graph,
        start_ent_map: Dict[str, List[str]],
    ) -> Tuple[List[int], List[str]]:
        if len(subtree_nodes) == 0:
            log.info("No subtree nodes available for reranking.")
            return [], []

        if self.variant == "wo_graph":
            log.info("Variant 'wo_graph' selected: Skipping graph reranker.")
            # Only use text reranker
            text_rerank_res = self.text_reranker(subtree_nodes, sub_query)
            tree_node_ids = [
                node_id for node_id, _ in self._confidence_filtered(text_rerank_res)
            ]

            # use start_ent_map as the returned entities
            res_entities = []
            seen_entities = set()
            for _, gbc_ents in start_ent_map.items():
                for node_name in gbc_ents:
                    if node_name not in seen_entities:
                        seen_entities.add(node_name)
                        res_entities.append(node_name)

            return tree_node_ids, res_entities

        if self.variant == "wo_text":
            log.info("Variant 'wo_text' selected: Skipping text reranker.")
            # Only use graph reranker
            graph_rerank_res, res_entities = self.graph_reranker(
                subgraph, start_ent_map, subtree_nodes
            )
            tree_node_ids = [node_id for node_id, _ in graph_rerank_res[: self.topk]]

            return tree_node_ids, res_entities

        # 2. Use Three Layer Reranker to select most relevant TreeNodes in the subtree.
        # 2.1 PPR to rank the most relevant TreeNodes in the subtree.
        graph_rerank_res, res_entities = self.graph_reranker(
            subgraph, start_ent_map, subtree_nodes
        )

        # 2.2 Rerank with text reranker model.
        text_rerank_res = self.text_reranker(subtree_nodes, sub_query)

        # 2.3 Rerank with Multimodal method.
        # mm_rerank_res = self.multimodal_reranker(subtree_nodes, sub_query)

        # Combine the results from three rerankers
        # merged_scores: Dict[int, List[float]] = merge_ranker_scores(
        #     graph_rerank_res, text_rerank_res, mm_rerank_res
        # )

        merged_scores: Dict[int, List[float]] = merge_ranker_scores(
            graph_rerank_res, text_rerank_res
        )
        sel_tree_nodes = calculate_skyline(merged_scores, max_results=self.topk)
        tree_node_ids = [node["node_id"] for node in sel_tree_nodes]

        if len(tree_node_ids) < self.topk and len(merged_scores) >= self.topk:
            log.info(
                f"Skyline returned only {len(tree_node_ids)} nodes. "
                f"Activating fallback to meet minimum of {self.topk}."
            )

            filled = self._fill_to_limit(
                tree_node_ids,
                graph_rerank_res,
                text_rerank_res,
                limit=self.topk,
            )
            # Skyline members are the algorithm's actual multi-criteria pick
            # and are never trimmed; only the padding added purely to reach
            # the numeric quota is subject to the relevance floor.
            skyline_ids = set(tree_node_ids)
            padding_ids = [node_id for node_id in filled if node_id not in skyline_ids]
            text_score_by_id = dict(text_rerank_res)
            padding_scored = sorted(
                ((node_id, text_score_by_id.get(node_id, 0.0)) for node_id in padding_ids),
                key=lambda item: (-item[1], item[0]),
            )
            kept_padding = {
                node_id
                for node_id, _ in self._confidence_filtered(
                    padding_scored,
                    limit=len(padding_scored),
                    min_k=max(0, self.topk_min - len(skyline_ids)),
                )
            }
            tree_node_ids = [
                node_id
                for node_id in filled
                if node_id in skyline_ids or node_id in kept_padding
            ]
            log.info(
                "Fallback selected %s unique nodes (%s padding entries dropped "
                "below the confidence floor).",
                len(tree_node_ids),
                len(filled) - len(tree_node_ids),
            )

        return tree_node_ids[: self.topk], res_entities

    @staticmethod
    def _fill_to_limit(
        selected_ids: List[int],
        *ranked_results: Iterable[Tuple[int, float]],
        limit: int,
    ) -> List[int]:
        """Deterministically fill a skyline without ever exceeding ``limit``."""

        result = []
        seen = set()

        def add(node_id: int) -> bool:
            if node_id in seen:
                return False
            seen.add(node_id)
            result.append(node_id)
            return len(result) >= limit

        for node_id in selected_ids:
            if add(node_id):
                return result

        iterators = [iter(items) for items in ranked_results]
        while iterators:
            active_iterators = []
            for iterator in iterators:
                try:
                    node_id, _ = next(iterator)
                except StopIteration:
                    continue
                active_iterators.append(iterator)
                if add(node_id):
                    return result
            iterators = active_iterators

        return result
