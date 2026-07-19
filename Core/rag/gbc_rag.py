from collections import defaultdict
from bisect import bisect_left
import heapq
import itertools
import re
from typing import Any, Callable, List, Mapping, Tuple, Dict, Optional

from Core.Index.Tree import TreeNode, NodeType
from Core.rag.base_rag import BaseRAG
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.rerank import TextRerankerProvider
from Core.provider.embedding import MMRerankerProvider
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.Index.GBCIndex import GBC
from Core.prompts.gbc_prompt import (
    LLM_EXPANSION_SELECT_PROMPT,
    QuestionEntity,
    QuestionEntityExtraction,
    QUESTION_ENT_PROMPT,
    QUESTION_ENTITY_TYPES,
    SecEXPSelection,
)
from Core.rag.gbc_answer import AnswerAgent
from Core.rag.gbc_plan import (
    PlanningBudgetExceeded,
    TaskPlanner,
    PlanResult,
)
from Core.rag.gbc_retrieval import HybridRetrievalBinding, Retriever
from docarag.asset_material import AssetMaterialResolver
from docarag.bookrag_projection import BookRAGAssetMaterializationLease
from docarag.domain.evidence import AssetKind, AssetRef, DocumentRef
from docarag.evidence_hydration import CanonicalEvidenceHydrator

from Core.rag.gbc_utils import (
    GBCRAGContext,
    SubStep,
    filter_tree_nodes,
)


import json
import networkx as nx

import logging

log = logging.getLogger(__name__)


SECTION_SELECTION_POLICY = (
    "linked-all_then_lexical-path_source-stratified-v2"
)
MAX_ANSWER_ASSETS = 64
MAX_ANSWER_ASSET_BYTES = 32 * 1024 * 1024
MAX_ANSWER_ASSET_TOTAL_BYTES = 128 * 1024 * 1024


class AnswerAssetMaterializationError(RuntimeError):
    """Canonical visual evidence could not be byte-verified for generation."""


class SectionSelectionBudgetExceeded(PlanningBudgetExceeded):
    """The prompt budget cannot represent all source-linked sections."""


def _citation_validation_from_partials(partials: object) -> Dict[str, Any]:
    if not isinstance(partials, list):
        return {}
    for partial in partials:
        if not isinstance(partial, dict):
            continue
        validation = partial.get("answer_citation_validation")
        if isinstance(validation, dict):
            return dict(validation)
    return {}


def _citation_bindings_from_validation(
    validation: object,
) -> Dict[str, str | None]:
    """Recover the canonical label map serialized by the answer validator."""

    if not isinstance(validation, dict):
        return {}
    allowed = validation.get("allowed_bindings")
    if not isinstance(allowed, list):
        return {}
    bindings: Dict[str, str | None] = {}
    for item in allowed:
        if not isinstance(item, dict):
            continue
        label = item.get("evidence_label")
        citation_id = item.get("citation_id")
        if not isinstance(label, str):
            continue
        normalized_id = citation_id if isinstance(citation_id, str) else None
        if label in bindings and bindings[label] != normalized_id:
            raise ValueError(
                f"answer validation contains conflicting binding for {label}"
            )
        bindings[label] = normalized_id
    return bindings


class GBCRAG(BaseRAG):
    """
    GBC RAG (Graph-Based Contextual Retrieval Augmented Generation) class.
    This class is designed to handle the retrieval and generation of responses
    based on a graph-based context.
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        config: GBCRAGConfig,
        gbc_index: GBC,
        tree_vdb: Any,
        asset_cas_root: str | None = None,
    ):
        super().__init__(
            llm,
            name="GBC RAG",
            description="Graph-Based Contextual Retrieval Augmented Generation",
        )
        self.vlm = vlm
        self.cfg = config
        self.variant = self.cfg.variant
        if not gbc_index:
            raise ValueError("GBC index must be provided for GBCRAG.")
        if tree_vdb is None:
            raise ValueError("Attested TreeVDB must be provided for GBCRAG.")
        self.gbc_index = gbc_index
        self.tree_vdb = tree_vdb
        self.tree_embedder = getattr(tree_vdb, "embedding_model", None)
        if asset_cas_root is not None and (
            not isinstance(asset_cas_root, str) or not asset_cas_root
        ):
            raise ValueError("asset_cas_root must be a non-empty path or None")
        self.asset_cas_root = asset_cas_root
        self._query_asset_leases: list[BookRAGAssetMaterializationLease] | None = None
        self._query_asset_attestations: Dict[str, object] = {}
        self._query_asset_refs: Dict[str, AssetRef] = {}
        self._query_asset_total_bytes = 0
        if self.asset_cas_root is not None:
            # Bind and validate the root before any model generation. No asset
            # bytes are read until a retrieved evidence item selects them.
            AssetMaterialResolver(
                content_addressed_root=self.asset_cas_root,
                max_assets=MAX_ANSWER_ASSETS,
                max_asset_bytes=MAX_ANSWER_ASSET_BYTES,
                max_total_bytes=MAX_ANSWER_ASSET_TOTAL_BYTES,
            )
        self.embedder = self.gbc_index.embedder if self.gbc_index else None
        self.reranker = TextRerankerProvider(
            model_name=self.cfg.reranker_config.model_name,
            max_length=self.cfg.reranker_config.max_length,
            device=self.cfg.reranker_config.device,
            backend=self.cfg.reranker_config.backend,
            api_base=self.cfg.reranker_config.api_base,
            request_timeout_seconds=(
                self.cfg.reranker_config.request_timeout_seconds
            ),
            use_flash_attention=self.cfg.reranker_config.use_flash_attention,
            flash_attention_fallback=(
                self.cfg.reranker_config.flash_attention_fallback
            ),
        )
        # GBC RAG config
        self.threshold_e = self.cfg.sim_threshold_e
        self.select_depth = self.cfg.select_depth
        self.max_retry = self.cfg.max_retry

        # Agents
        self.planner = TaskPlanner(
            llm=self.llm,
            max_query_chars=self.cfg.planner_max_query_chars,
            max_subquestions=self.cfg.planner_max_subquestions,
            max_subquestion_chars=self.cfg.planner_max_subquestion_chars,
            max_filters=self.cfg.planner_max_filters,
            max_filter_value_chars=self.cfg.planner_max_filter_value_chars,
        )
        evidence_hydrator = None
        tree_index = self.gbc_index.TreeIndex
        retrieval_binding = self._hybrid_retrieval_binding(tree_index, tree_vdb)
        if tree_index is not None:
            tree_meta = tree_index.meta_info
            for field_name in (
                "document_version_id",
                "source_sha256",
                "source_uri",
            ):
                value = getattr(tree_meta, field_name, None)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "canonical BookRAG tree is missing source identity: "
                        f"{field_name}"
                    )
            evidence_hydrator = CanonicalEvidenceHydrator(
                self.gbc_index.store.evidence_catalog(),
                node_resolver=lambda node_ids, tree=tree_index: tree.get_nodes_data(
                    list(node_ids)
                ),
                expected_document_version_id=tree_meta.document_version_id,
                expected_source_sha256=tree_meta.source_sha256,
                expected_source_uri=tree_meta.source_uri,
                max_evidence=64,
                max_sources_per_evidence=64,
                max_total_sources=512,
            )
        self.evidence_hydrator = evidence_hydrator
        self.answer = AnswerAgent(
            llm=self.llm,
            vlm=self.vlm,
            variant=self.variant,
        )
        self.retriever = Retriever(
            variant=self.variant,
            reranker=self.reranker,
            # mm_reranker=self.mm_reranker,
            embedder=self.embedder,
            alpha=self.cfg.alpha,
            topk_ent=self.cfg.topk_ent,
            x_percentile=self.cfg.x_percentile,
            topk=self.cfg.topk,
            candidate_pool_size=self.cfg.candidate_pool_size,
            tree_vdb=self.tree_vdb,
            retrieval_binding=retrieval_binding,
        )
        self.retrieval_callback: Optional[
            Callable[[GBCRAGContext, SubStep], None]
        ] = None

    @staticmethod
    def _hybrid_retrieval_binding(tree_index, tree_vdb) -> HybridRetrievalBinding | None:
        """Bind candidate fusion to the exact canonical tree/vector revision."""

        if tree_index is None:
            raise ValueError("GBCRAG requires a loaded BookRAG tree")
        tree_meta = tree_index.meta_info
        scope = getattr(tree_vdb, "scope_filter", None)
        if not isinstance(scope, dict):
            raise ValueError("TreeVDB is missing its immutable source scope")
        for field_name in (
            "tenant_id",
            "corpus_id",
            "document_version_id",
            "index_revision_id",
            "source_sha256",
        ):
            value = scope.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"TreeVDB scope is missing {field_name}")
        if scope.get("index_kind") != "tree":
            raise ValueError("GBCRAG requires the attested tree vector scope")
        if scope["source_sha256"] != tree_meta.source_sha256:
            raise ValueError("TreeVDB source SHA-256 differs from the BookRAG tree")
        if scope["document_version_id"] != tree_meta.document_version_id:
            raise ValueError("TreeVDB document version differs from the BookRAG tree")

        canonical_sha256 = tree_meta.canonical_artifact_sha256
        source_uri = tree_meta.source_uri
        if (
            not isinstance(canonical_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", canonical_sha256) is None
            or not isinstance(source_uri, str)
            or not source_uri
        ):
            # Historical BookRAG trees remain queryable through deterministic
            # internal RRF, but cannot claim the canonical candidate contract.
            log.warning(
                "BookRAG tree has no canonical artifact identity; hybrid retrieval "
                "will use compatibility fusion without a CandidateFusionResult"
            )
            return None
        document = DocumentRef.identified(
            content_sha256=scope["source_sha256"],
            mime_type="application/pdf",
            source_uri=source_uri,
            display_name=tree_meta.file_name,
        )
        if document.document_version_id != scope["document_version_id"]:
            raise ValueError(
                "BookRAG document identity differs from the TreeVDB document version"
            )
        revision_id = scope["index_revision_id"]
        tree_revision_id = getattr(tree_vdb, "tree_revision_id", None)
        if not isinstance(tree_revision_id, str) or not tree_revision_id:
            # Compatibility for injected test stores. The production resource
            # loader always supplies the attested tree semantic digest.
            tree_revision_id = f"tree-{revision_id}"
        return HybridRetrievalBinding(
            tenant_id=scope["tenant_id"],
            corpus_id=scope["corpus_id"],
            document_id=document.document_id,
            document_version_id=document.document_version_id,
            source_sha256=scope["source_sha256"],
            canonical_artifact_sha256=canonical_sha256,
            tree_revision_id=tree_revision_id,
            dense_index_revision_id=revision_id,
        )

    def _notify_retrieval_ready(self, context: GBCRAGContext, step: SubStep) -> None:
        """Optional UI hook fired after retrieval and before answer generation."""
        if not self.retrieval_callback:
            return
        try:
            self.retrieval_callback(context, step)
        except Exception as e:
            log.warning(f"Retrieval callback failed: {e}")

    def _get_entity_embed_text(self, entity: QuestionEntity) -> str:
        return f"Name: {entity.entity_name}\nType: {entity.entity_type}"

    def _configured_int(self, name: str, default: int) -> int:
        value = getattr(getattr(self, "cfg", None), name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _query_limits_trace(self) -> Dict[str, int]:
        trace = self.planner.limits_trace()
        for field, default in (
            ("query_entity_limit", 16),
            ("query_entity_name_chars", 256),
            ("query_entity_type_chars", 64),
            ("query_entity_candidate_limit", 5),
            ("section_selection_candidate_limit", 48),
            ("section_selection_supplementary_limit", 8),
            ("section_selection_title_chars", 256),
            ("section_selection_path_depth", 8),
            ("section_selection_prompt_chars", 65536),
        ):
            trace[field] = self._configured_int(field, default)
        return trace

    @staticmethod
    def _bounded_results(items: object, *, limit: int, label: str) -> list:
        """Materialize at most ``limit + 1`` items and reject contract overflow."""

        if items is None:
            return []
        try:
            bounded = list(itertools.islice(iter(items), limit + 1))
        except TypeError as exc:
            raise ValueError(f"{label} must be iterable") from exc
        if len(bounded) > limit:
            raise PlanningBudgetExceeded(
                f"{label} returned more than the configured limit of {limit}"
            )
        return bounded

    def _validate_entity(self, entity: object, *, position: int) -> QuestionEntity:
        name = getattr(entity, "entity_name", None)
        entity_type = getattr(entity, "entity_type", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"query entity {position} has an empty name")
        if not isinstance(entity_type, str) or not entity_type.strip():
            raise ValueError(f"query entity {position} has an empty type")
        name_limit = self._configured_int("query_entity_name_chars", 256)
        type_limit = self._configured_int("query_entity_type_chars", 64)
        if len(name) > name_limit:
            raise PlanningBudgetExceeded(
                f"query entity {position} name contains {len(name)} characters, "
                f"exceeding the configured limit of {name_limit}"
            )
        if len(entity_type) > type_limit:
            raise PlanningBudgetExceeded(
                f"query entity {position} type contains {len(entity_type)} "
                f"characters, exceeding the configured limit of {type_limit}"
            )
        return QuestionEntity(entity_name=name, entity_type=entity_type)

    def _entity_map(
        self, entities: List[QuestionEntity], force_one: bool = False
    ) -> Dict[str, List[str]]:
        """
        Maps entities to their corresponding IDs in the GBC index.
        Use vdb to find the entity in GBC index.
        """
        entity_limit = self._configured_int("query_entity_limit", 16)
        bounded_entities = self._bounded_results(
            entities,
            limit=entity_limit,
            label="query entity extraction",
        )
        validated_entities = [
            self._validate_entity(entity, position=index)
            for index, entity in enumerate(bounded_entities, start=1)
        ]
        entities_str = [
            self._get_entity_embed_text(entity) for entity in validated_entities
        ]
        Qent_GBCent_map = defaultdict(list)
        res_list = []
        for ent_str in entities_str:
            query_res = self._bounded_results(
                self.gbc_index.entity_vdb.search(query_text=ent_str, top_k=2),
                limit=2,
                label="entity-vector search",
            )
            if not query_res:
                log.info(f"No GBC entity candidates found for '{ent_str}'.")
                continue
            min_distance = query_res[0]["distance"]
            retrieve_name = query_res[0]["metadata"].get("entity_name")
            retrieve_type = query_res[0]["metadata"].get("entity_type")
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                retrieve_name, retrieve_type
            )
            if min_distance < self.threshold_e:
                Qent_GBCent_map[ent_str].append(node_name)
                log.info(f"Entity '{ent_str}' mapped to GBC entity: {node_name}")
            else:
                res_list.append((ent_str, node_name, min_distance))

        if force_one and len(Qent_GBCent_map) == 0 and len(res_list) > 0:
            # `force_one` is retained as a compatibility argument, but must not
            # override the configured quality threshold.  An unrelated graph
            # seed is worse than an empty entity leg because it can promote an
            # entire wrong subtree.  Section/text candidates remain available
            # to the caller when no entity is sufficiently close.
            closest = min(res_list, key=lambda item: item[2])
            log.info(
                "No entity candidate met threshold %.4f; closest candidate "
                "for '%s' was '%s' at distance %.4f and was rejected.",
                self.threshold_e,
                closest[0],
                closest[1],
                closest[2],
            )

        return Qent_GBCent_map

    def _get_query_entity(self, query: str) -> Dict[str, List[str]]:
        """
        Get the entity mapping for the query.
        """

        if not isinstance(query, str) or not query.strip():
            log.info("Skip entity mapping for an empty query.")
            return {}

        max_query_chars = self._configured_int("planner_max_query_chars", 4096)
        if len(query) > max_query_chars:
            raise PlanningBudgetExceeded(
                f"query contains {len(query)} characters, exceeding the "
                f"configured limit of {max_query_chars}"
            )

        # 1. retrieval relevent entities from the query
        candidate_limit = self._configured_int("query_entity_candidate_limit", 5)
        retrieval_ents = self._bounded_results(
            self.gbc_index.entity_vdb.search(
                query_text=query,
                top_k=candidate_limit,
            ),
            limit=candidate_limit,
            label="query entity candidate search",
        )
        retrieval_node_names = set()
        retrieval_nodes = []
        for ent_info in retrieval_ents:
            ent_name = ent_info["metadata"].get("entity_name")
            ent_type = ent_info["metadata"].get("entity_type")
            node_dict = {
                "entity_name": ent_name,
                "entity_type": ent_type,
            }
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                ent_name, ent_type
            )
            if node_name not in retrieval_node_names:
                retrieval_node_names.add(node_name)
                retrieval_nodes.append(node_dict)

        # 2. llm generate and select entities from the query
        prompt = QUESTION_ENT_PROMPT.format(
            input_text=query,
            entity_types=", ".join(QUESTION_ENTITY_TYPES),
            retrieved_entities=json.dumps(retrieval_nodes, ensure_ascii=False),
        )
        res_entities = []
        try:
            res: QuestionEntityExtraction = self.llm.get_json_completion(
                prompt, QuestionEntityExtraction
            )
            if res and res.entities:
                entity_limit = self._configured_int("query_entity_limit", 16)
                res_entities = self._bounded_results(
                    res.entities,
                    limit=entity_limit,
                    label="query entity extraction",
                )
                res_entities = [
                    self._validate_entity(entity, position=index)
                    for index, entity in enumerate(res_entities, start=1)
                ]
                entities_name = [entity.entity_name for entity in res_entities]
                log.info(f"Extracted entities: {entities_name}")
            else:
                log.info("No entities extracted from the query.")

        except PlanningBudgetExceeded:
            raise
        except Exception as e:
            log.error(f"Error during entity extraction: {e}")

        if len(res_entities) == 0:
            # use the retrieval entities if no entity is extracted by llm
            log.info("Use the question as the entity.")
            res_entities = [
                QuestionEntity(entity_name=query, entity_type="TASK_OR_PROBLEM")
            ]

        Qent_GBCent_map = defaultdict(list)
        remain_ents = []
        seen_extracted_entities = set()
        for res_ent in res_entities:
            normalized_name = res_ent.entity_name.lower()
            normalized_type = res_ent.entity_type.upper().replace(" ", "_")
            normalized_key = (normalized_name, normalized_type)
            if normalized_key in seen_extracted_entities:
                continue
            seen_extracted_entities.add(normalized_key)
            res_ent = QuestionEntity(
                entity_name=normalized_name,
                entity_type=normalized_type,
            )
            ent_node_name = self.gbc_index.GraphIndex.get_node_name_from_entity(res_ent)
            if ent_node_name in retrieval_node_names:
                Qent_GBCent_map[ent_node_name].append(ent_node_name)
                log.info(
                    f"Entity '{ent_node_name}' mapped to GBC entity: {ent_node_name}"
                )
            else:
                remain_ents.append(res_ent)

        if remain_ents:
            remain_map = self._entity_map(remain_ents, force_one=False)
            for k, v in remain_map.items():
                Qent_GBCent_map[k].extend(v)

        return Qent_GBCent_map

    def link_tree_node(self, entities_map: Dict[str, List[str]]) -> List[dict]:
        """
        Get the tree nodes for the given entities.
        """
        tree_node_cnt = defaultdict(list)
        all_map_nodenames = []
        seen_node_names = set()
        for ent_list in entities_map.values():
            for ent in ent_list:
                if ent not in seen_node_names:
                    seen_node_names.add(ent)
                    all_map_nodenames.append(ent)
        if not all_map_nodenames:
            log.warning("No entities found in the mapping.")
            return []

        for node_name in all_map_nodenames:
            tree_node_set = self.gbc_index.GraphIndex.NodeName2TreeNodes(node_name)
            for node_id in sorted(tree_node_set):
                tree_node_cnt[node_id].append(node_name)

        tree_nodes = [
            {
                "index_id": node_id,
                "map_cnt": len(link_ents),
                "linked_entities": link_ents,
            }
            for node_id, link_ents in sorted(
                tree_node_cnt.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ]

        if not tree_nodes:
            log.warning("No tree nodes found for the given entities.")
            return []

        log.info(f"Retrieved {len(tree_nodes)} tree nodes based on entity mapping.")
        return tree_nodes

    def link_section(self, tree_nodes: List[dict]) -> Dict[int, List[str]]:
        """
        Get the linked section TreeNode ids from the tree nodes.
        given the tree nodes, get the linked section TreeNode ids (specific depth).
        return the Dict: section_id --> [linked_entity1, linked_entity2, ...]
        """
        sec_entity_map = defaultdict(list)
        for node in tree_nodes:
            node_idx = node["index_id"]
            ancestor = self.gbc_index.TreeIndex.get_ancestor_at_depth(
                node_idx, self.select_depth
            )
            ancestor_idx = ancestor.index_id if ancestor else None
            node_ents = node["linked_entities"]
            if ancestor_idx:
                sec_entity_map[ancestor_idx].extend(node_ents)

        for sec_id, val in sec_entity_map.items():
            sec_entity_map[sec_id] = list(dict.fromkeys(val))

        log.info(
            f"Found {len(sec_entity_map)} linked sections at depth {self.select_depth}."
        )
        return sec_entity_map

    def prep_SecSel_prompt(
        self,
        query: str,
        link_nodes: List[TreeNode] = None,
        remain_nodes: List[TreeNode] = None,
        sec_entity_map: Dict[int, List[str]] = None,
        iter_context: Optional[SubStep] = None,
    ) -> str:
        """Prepare one bounded, source-ordered section-selection prompt."""

        link_nodes = list(link_nodes or [])
        remain_nodes = list(remain_nodes or [])
        candidate_limit = self._configured_int(
            "section_selection_candidate_limit", 48
        )
        if len(link_nodes) + len(remain_nodes) > candidate_limit:
            raise SectionSelectionBudgetExceeded(
                "section-selection prompt contains "
                f"{len(link_nodes) + len(remain_nodes)} candidates, exceeding "
                f"the configured limit of {candidate_limit}"
            )
        max_query_chars = self._configured_int("planner_max_query_chars", 4096)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("section-selection query must be a non-empty string")
        if len(query) > max_query_chars:
            raise PlanningBudgetExceeded(
                f"query contains {len(query)} characters, exceeding the "
                f"configured limit of {max_query_chars}"
            )

        title_limit = self._configured_int("section_selection_title_chars", 256)
        path_depth_limit = self._configured_int(
            "section_selection_path_depth", 8
        )
        truncated_fields = 0

        def bounded_title(value: object) -> str:
            nonlocal truncated_fields
            text = value if isinstance(value, str) else ""
            if len(text) <= title_limit:
                return text
            truncated_fields += 1
            return text[: title_limit - 1] + "…"

        def prep_nodes_json(
            nodes: List[TreeNode], sec_entity_map: Dict[int, List[str]] = None
        ) -> str:
            nonlocal truncated_fields
            node_infos = []
            for node in nodes:
                sec_idx = node.index_id
                section_title = bounded_title(node.meta_info.content)
                sec_path = self.gbc_index.TreeIndex.get_path_from_root(sec_idx)
                if len(sec_path) > path_depth_limit:
                    # Keep the document root context and the most local
                    # structural ancestors.  The trace records this loss.
                    truncated_fields += 1
                    if path_depth_limit == 1:
                        sec_path = [sec_path[-1]]
                    else:
                        sec_path = [
                            sec_path[0],
                            *sec_path[-(path_depth_limit - 1) :],
                        ]
                title_path_obj = [
                    bounded_title(path_node.meta_info.content)
                    for path_node in sec_path
                ]
                sec_info = {
                    "id": sec_idx,
                    "title": section_title,
                    "path": title_path_obj,
                }
                if sec_entity_map and sec_idx in sec_entity_map:
                    entities_str = bounded_title(
                        ", ".join(sec_entity_map[sec_idx])
                    )
                    sec_info["contained_entities"] = entities_str
                node_infos.append(sec_info)

            sec_info_str = json.dumps(
                node_infos,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return sec_info_str

        link_sec_str = (
            prep_nodes_json(link_nodes, sec_entity_map=sec_entity_map)
            if link_nodes
            else "[]"
        )
        remain_sec_str = (
            prep_nodes_json(remain_nodes, sec_entity_map=None) if remain_nodes else "[]"
        )
        query_prompt = LLM_EXPANSION_SELECT_PROMPT.format(
            user_question=query,
            primary_candidates_json=link_sec_str,
            remaining_sections_json=remain_sec_str,
            max_supplementary_sections=self._configured_int(
                "section_selection_supplementary_limit", 8
            ),
        )

        prompt_limit = self._configured_int(
            "section_selection_prompt_chars", 65536
        )
        if iter_context is not None:
            iter_context.section_prompt_chars = len(query_prompt)
            iter_context.section_prompt_truncated_fields = truncated_fields
        if len(query_prompt) > prompt_limit:
            if iter_context is not None:
                iter_context.section_selection_status = "prompt_budget_exceeded"
                iter_context.section_selection_budget_exhausted = True
            raise SectionSelectionBudgetExceeded(
                "rendered section-selection prompt contains "
                f"{len(query_prompt)} characters, exceeding the configured "
                f"limit of {prompt_limit}"
            )

        return query_prompt

    @staticmethod
    def _query_tokens(text: object) -> set[str]:
        if not isinstance(text, str):
            return set()
        return {
            token
            for token in re.findall(r"[^\W_]+", text.casefold())
            if len(token) > 1
        }

    def _bounded_section_candidates(
        self,
        *,
        query: str,
        remain_sections: List[TreeNode],
        capacity: int,
        source_positions: Dict[int, int],
        linked_positions: List[int],
    ) -> List[TreeNode]:
        """Select deterministic top-M structural candidates in bounded memory.

        Three quarters of the budget is assigned to lexical title/path fit and
        source proximity to linked evidence.  The remainder is filled from
        source-order strata so a long document is not reduced to its prefix
        when headings share no query token. Complexity is
        ``O(S * (D + log M))`` time and ``O(S + M)`` temporary memory for
        ``S`` sections, bounded path depth ``D`` and configured candidate
        capacity ``M <= 256``.
        """

        if capacity <= 0 or not remain_sections:
            return []

        query_tokens = self._query_tokens(query)
        title_limit = self._configured_int("section_selection_title_chars", 256)
        path_depth_limit = self._configured_int(
            "section_selection_path_depth", 8
        )
        sorted_linked_positions = sorted(linked_positions)

        def linked_distance(position: int) -> int:
            if not sorted_linked_positions:
                return position
            insertion = bisect_left(sorted_linked_positions, position)
            distances = []
            if insertion < len(sorted_linked_positions):
                distances.append(sorted_linked_positions[insertion] - position)
            if insertion:
                distances.append(position - sorted_linked_positions[insertion - 1])
            return min(distances)

        records: List[Tuple[Tuple[int, ...], TreeNode]] = []
        for node in remain_sections:
            title = (node.meta_info.content or "")[:title_limit]
            path = self.gbc_index.TreeIndex.get_path_from_root(node.index_id)
            path = path[-path_depth_limit:]
            path_text = " ".join(
                (path_node.meta_info.content or "")[:title_limit]
                for path_node in path
            )
            title_overlap = len(query_tokens & self._query_tokens(title))
            path_overlap = len(query_tokens & self._query_tokens(path_text))
            position = source_positions[node.index_id]
            proximity = linked_distance(position)
            # ``heapq.nsmallest`` makes the negated relevance fields highest
            # priority and keeps every tie stable in source order.
            rank_key = (
                -title_overlap,
                -path_overlap,
                proximity,
                position,
                node.index_id,
            )
            records.append((rank_key, node))

        target_count = min(capacity, len(records))
        relevance_count = max(1, (target_count * 3 + 3) // 4)
        relevance_top = heapq.nsmallest(
            relevance_count,
            records,
            key=lambda record: record[0],
        )
        selected: Dict[int, TreeNode] = {
            node.index_id: node for _, node in relevance_top
        }

        # Add deterministic source-stratified coverage. Assign every available
        # section to its nearest stratum in one O(S log M) pass, retain the best
        # candidate per stratum, then use one source-order fallback scan if a
        # stratum happened to contain only relevance-selected sections.
        coverage_needed = target_count - len(selected)
        section_count = len(remain_sections)
        coverage_targets = [
            min(
                section_count - 1,
                ((2 * stratum + 1) * section_count)
                // (2 * max(1, coverage_needed)),
            )
            for stratum in range(coverage_needed)
        ]
        best_by_stratum: Dict[int, Tuple[Tuple[int, int, int], TreeNode]] = {}
        for local_position, node in enumerate(remain_sections):
            if node.index_id in selected or not coverage_targets:
                continue
            insertion = bisect_left(coverage_targets, local_position)
            stratum_candidates = []
            if insertion < len(coverage_targets):
                stratum_candidates.append(insertion)
            if insertion:
                stratum_candidates.append(insertion - 1)
            stratum = min(
                stratum_candidates,
                key=lambda index: (
                    abs(local_position - coverage_targets[index]),
                    index,
                ),
            )
            candidate_rank = (
                abs(local_position - coverage_targets[stratum]),
                source_positions[node.index_id],
                node.index_id,
            )
            current = best_by_stratum.get(stratum)
            if current is None or candidate_rank < current[0]:
                best_by_stratum[stratum] = (candidate_rank, node)

        for stratum in sorted(best_by_stratum):
            node = best_by_stratum[stratum][1]
            selected[node.index_id] = node

        if len(selected) < target_count:
            for node in remain_sections:
                selected.setdefault(node.index_id, node)
                if len(selected) == target_count:
                    break

        return sorted(
            selected.values(),
            key=lambda node: (source_positions[node.index_id], node.index_id),
        )

    def llm_section_selection(
        self,
        query: str,
        tree_nodes: List[dict],
        iter_context: Optional[SubStep] = None,
    ) -> None:
        """
        Use LLM to select the most relevant section based on the query and Section info.
        """
        if iter_context is None:
            raise ValueError("iter_context is required for traced section selection")
        sec_entity_map = self.link_section(tree_nodes)
        requested_link_ids = set(sec_entity_map)

        all_sections = list(
            self.gbc_index.TreeIndex.get_nodes_at_depth(self.select_depth)
        )
        source_positions: Dict[int, int] = {}
        for position, section in enumerate(all_sections):
            if section.index_id in source_positions:
                raise ValueError(
                    f"duplicate section index_id {section.index_id} at selection depth"
                )
            source_positions[section.index_id] = position
        unknown_link_ids = requested_link_ids - source_positions.keys()
        if unknown_link_ids:
            raise ValueError(
                "entity-linked sections are absent from the selected tree depth: "
                + ", ".join(str(value) for value in sorted(unknown_link_ids))
            )

        # Set membership makes the partition O(S + L), not O(S * L).
        link_secs = [
            section
            for section in all_sections
            if section.index_id in requested_link_ids
        ]
        remain_secs = [
            section
            for section in all_sections
            if section.index_id not in requested_link_ids
        ]
        link_section_ids = [section.index_id for section in link_secs]
        candidate_limit = self._configured_int(
            "section_selection_candidate_limit", 48
        )
        iter_context.section_selection_policy = SECTION_SELECTION_POLICY
        iter_context.section_total_count = len(all_sections)
        iter_context.section_linked_count = len(link_secs)
        iter_context.section_candidate_limit = candidate_limit
        iter_context.linked_section_ids = link_section_ids

        if len(link_secs) > candidate_limit:
            # Linked sections are source-derived evidence, not optional model
            # suggestions.  Never hide an arbitrary suffix to make the prompt
            # fit; reject the request before selection or subtree retrieval.
            iter_context.section_selection_status = "linked_budget_exceeded"
            iter_context.section_selection_budget_exhausted = True
            iter_context.section_omitted_count = len(all_sections)
            raise SectionSelectionBudgetExceeded(
                f"{len(link_secs)} source-linked sections exceed the configured "
                f"section candidate limit of {candidate_limit}; no linked "
                "evidence was silently dropped"
            )

        remaining_capacity = candidate_limit - len(link_secs)
        prompt_remain_secs = self._bounded_section_candidates(
            query=query,
            remain_sections=remain_secs,
            capacity=remaining_capacity,
            source_positions=source_positions,
            linked_positions=[source_positions[value] for value in link_section_ids],
        )
        iter_context.section_prompt_primary_count = len(link_secs)
        iter_context.section_prompt_remaining_count = len(prompt_remain_secs)
        iter_context.section_prompt_candidate_count = (
            len(link_secs) + len(prompt_remain_secs)
        )
        iter_context.section_prompt_candidate_ids = [
            *link_section_ids,
            *(section.index_id for section in prompt_remain_secs),
        ]
        iter_context.section_omitted_count = len(remain_secs) - len(
            prompt_remain_secs
        )

        if len(remain_secs) == 0:
            log.info("No remaining sections to select from. Skipping LLM expansion.")
            iter_context.supplementary_ids = []
            iter_context.selected_explanation = (
                "No remaining sections for supplementary selection."
            )
            iter_context.retrieval_sec_ids = link_section_ids
            iter_context.section_selection_status = "no_remaining_sections"
            return

        if not prompt_remain_secs:
            # This occurs when linked evidence consumes the complete prompt
            # budget.  It is safe to continue because every linked section is
            # represented and will be retrieved; the omitted supplementary
            # scope is explicit in the trace.
            iter_context.supplementary_ids = []
            iter_context.selected_explanation = (
                "All section-selection capacity was consumed by source-linked "
                "sections; supplementary selection was not attempted."
            )
            iter_context.retrieval_sec_ids = link_section_ids
            iter_context.section_selection_status = "linked_scope_fills_budget"
            return

        query_prompt = self.prep_SecSel_prompt(
            query=query,
            link_nodes=link_secs,
            remain_nodes=prompt_remain_secs,
            sec_entity_map=sec_entity_map,
            iter_context=iter_context,
        )
        sel_ids = []
        explanation = "Error or no valid response from LLM during section expansion."

        prompt_remain_ids = {sec.index_id for sec in prompt_remain_secs}
        try:
            res: SecEXPSelection = self.llm.get_json_completion(
                query_prompt, SecEXPSelection
            )
            if res:
                explanation = res.explanation
                if res.supplementary_ids:
                    supplementary_limit = self._configured_int(
                        "section_selection_supplementary_limit", 8
                    )
                    if len(res.supplementary_ids) > supplementary_limit:
                        iter_context.section_selection_status = (
                            "model_selection_budget_exceeded"
                        )
                        iter_context.section_selection_budget_exhausted = True
                        raise PlanningBudgetExceeded(
                            "section selection returned "
                            f"{len(res.supplementary_ids)} IDs, exceeding the "
                            f"configured limit of {supplementary_limit}"
                        )
                    if len(set(res.supplementary_ids)) != len(
                        res.supplementary_ids
                    ):
                        iter_context.section_selection_status = (
                            "invalid_duplicate_model_ids"
                        )
                        raise ValueError(
                            "section selection returned duplicate supplementary IDs"
                        )
                    invalid_ids = set(res.supplementary_ids) - prompt_remain_ids
                    if invalid_ids:
                        iter_context.section_selection_status = (
                            "invalid_out_of_pool_model_ids"
                        )
                        raise ValueError(
                            "section selection returned IDs outside the bounded "
                            "prompt candidate pool: "
                            + ", ".join(
                                str(value) for value in sorted(invalid_ids)
                            )
                        )
                    sel_ids.extend(res.supplementary_ids)

                    if sel_ids:
                        log.info(f"LLM selected {len(sel_ids)} supplementary sections.")
                    else:
                        log.info("LLM did not select any valid supplementary sections.")
                else:
                    log.info("LLM did not select any supplementary sections.")

        except (PlanningBudgetExceeded, ValueError):
            raise
        except Exception as e:
            log.error(f"Error occurred during section selection: {e}")
            iter_context.section_selection_status = "model_error_linked_only"
            if not link_section_ids:
                raise RuntimeError(
                    "section selection failed and no source-linked section is "
                    "available for fail-closed retrieval"
                ) from e

        iter_context.supplementary_ids = sel_ids
        iter_context.selected_explanation = explanation

        if iter_context.section_selection_status == "not_started":
            iter_context.section_selection_status = (
                "selected" if sel_ids else "no_supplementary_selected"
            )

        retrieval_sec_ids = list(dict.fromkeys(link_section_ids + sel_ids))
        iter_context.retrieval_sec_ids = retrieval_sec_ids
        log.info(
            f"LLM selected {len(sel_ids)} supplementary sections, total {len(retrieval_sec_ids)} sections for retrieval."
        )

    @staticmethod
    def _asset_is_image_compatible(asset: AssetRef) -> bool:
        media_type = (asset.media_type or "").casefold()
        return media_type.startswith("image/") or asset.kind in {
            AssetKind.REGION_CROP,
            AssetKind.EXTRACTED_IMAGE,
            AssetKind.PAGE_RENDER,
        }

    @staticmethod
    def _visual_evidence_node(node: Mapping[str, Any]) -> bool:
        node_type = node.get("type")
        if isinstance(node_type, NodeType):
            node_type = node_type.value
        if node_type in {
            NodeType.IMAGE.value,
            NodeType.TABLE.value,
            NodeType.EQUATION.value,
        }:
            return True
        subtype = node.get("semantic_subtype")
        if not isinstance(subtype, str):
            return False
        normalized = re.sub(r"[\s-]+", "_", subtype.strip().casefold())
        return normalized in {
            "chart",
            "diagram",
            "figure",
            "flow_chart",
            "flowchart",
            "image",
            "picture",
            "scan_region",
            "screenshot",
            "sketch",
            "workflow",
        }

    def _answer_asset_refs(self, node: Mapping[str, Any]) -> tuple[AssetRef, ...]:
        """Select exact canonical image material for one answer evidence item."""

        if not self._visual_evidence_node(node):
            return ()
        raw_sources = node.get("canonical_sources")
        if not isinstance(raw_sources, list):
            raise AnswerAssetMaterializationError(
                "visual evidence is missing hydrated canonical_sources"
            )
        declared_ids = node.get("canonical_asset_ids") or []
        if not isinstance(declared_ids, list) or any(
            not isinstance(value, str) or not value for value in declared_ids
        ):
            raise AnswerAssetMaterializationError(
                "visual evidence has invalid canonical_asset_ids"
            )
        selected_ids = node.get("selected_asset_ids") or []
        if not isinstance(selected_ids, list) or any(
            not isinstance(value, str) or not value for value in selected_ids
        ):
            raise AnswerAssetMaterializationError(
                "visual evidence has invalid selected_asset_ids"
            )
        declared = set(declared_ids)
        selected = set(selected_ids)
        if not selected.issubset(declared):
            raise AnswerAssetMaterializationError(
                "selected answer assets are outside canonical_asset_ids"
            )

        by_id: Dict[str, AssetRef] = {}
        for source_position, source in enumerate(raw_sources):
            if not isinstance(source, Mapping):
                raise AnswerAssetMaterializationError(
                    f"canonical_sources[{source_position}] must be an object"
                )
            raw_assets = source.get("assets", [])
            if not isinstance(raw_assets, list):
                raise AnswerAssetMaterializationError(
                    f"canonical_sources[{source_position}].assets must be an array"
                )
            for asset_position, raw_asset in enumerate(raw_assets):
                if not isinstance(raw_asset, Mapping):
                    raise AnswerAssetMaterializationError(
                        "canonical asset entry must be an object"
                    )
                try:
                    asset = AssetRef.from_dict(raw_asset)
                except Exception as exc:
                    raise AnswerAssetMaterializationError(
                        "hydrated canonical asset is invalid"
                    ) from exc
                if asset.asset_id not in declared:
                    raise AnswerAssetMaterializationError(
                        "hydrated asset is outside canonical_asset_ids"
                    )
                previous = by_id.get(asset.asset_id)
                if previous is not None and previous != asset:
                    raise AnswerAssetMaterializationError(
                        "one canonical asset ID resolves to conflicting references"
                    )
                by_id[asset.asset_id] = asset

        if set(by_id) != declared:
            raise AnswerAssetMaterializationError(
                "canonical_asset_ids do not resolve exactly through canonical_sources"
            )
        if selected:
            assets = [by_id[asset_id] for asset_id in selected_ids]
            if any(not self._asset_is_image_compatible(asset) for asset in assets):
                raise AnswerAssetMaterializationError(
                    "selected canonical answer asset is not image-compatible"
                )
            return tuple(assets)

        # Older canonical tree projections may have withheld img_path because
        # no indexing-time lease was supplied. Select one stable canonical crop
        # now so tables, equations, diagrams, and images can still reach VLM.
        compatible = sorted(
            (asset for asset in by_id.values() if self._asset_is_image_compatible(asset)),
            key=lambda asset: (
                {
                    AssetKind.REGION_CROP: 0,
                    AssetKind.EXTRACTED_IMAGE: 1,
                    AssetKind.PAGE_RENDER: 2,
                    AssetKind.SOURCE: 3,
                }[asset.kind],
                asset.asset_id,
            ),
        )
        return tuple(compatible[:1])

    def _begin_query_asset_materialization(self) -> None:
        if self._query_asset_leases is not None:
            raise RuntimeError("GBCRAG query asset lease is already active")
        self._query_asset_leases = []
        self._query_asset_attestations = {}
        self._query_asset_refs = {}
        self._query_asset_total_bytes = 0

    def _finish_query_asset_materialization(self) -> None:
        leases = getattr(self, "_query_asset_leases", None)
        self._query_asset_leases = None
        self._query_asset_attestations = {}
        self._query_asset_refs = {}
        self._query_asset_total_bytes = 0
        if leases is None:
            return
        first_error: BaseException | None = None
        for lease in reversed(leases):
            try:
                lease.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    log.exception("Additional answer asset lease close failed")
        if first_error is not None:
            raise AnswerAssetMaterializationError(
                "canonical answer asset changed before generation completed"
            ) from first_error

    def _materialize_answer_assets(
        self, tree_data: List[Dict[str, Any]]
    ) -> None:
        """Lease bounded canonical CAS paths and attach verified prompt material."""

        assets_by_node: Dict[int, tuple[AssetRef, ...]] = {}
        new_assets: Dict[str, AssetRef] = {}
        for node in tree_data:
            refs = self._answer_asset_refs(node)
            if not refs:
                continue
            node_id = node.get("index_id")
            if isinstance(node_id, bool) or not isinstance(node_id, int):
                raise AnswerAssetMaterializationError(
                    "retrieved visual evidence has an invalid tree node ID"
                )
            assets_by_node[node_id] = refs
            for asset in refs:
                bound = self._query_asset_refs.get(asset.asset_id)
                if bound is not None and bound != asset:
                    raise AnswerAssetMaterializationError(
                        "one answer asset ID resolves to conflicting references"
                    )
                pending = new_assets.get(asset.asset_id)
                if pending is not None and pending != asset:
                    raise AnswerAssetMaterializationError(
                        "one pending answer asset ID resolves to conflicting references"
                    )
                if bound is None:
                    new_assets.setdefault(asset.asset_id, asset)

        if not assets_by_node:
            return
        if self._query_asset_leases is None:
            raise AnswerAssetMaterializationError(
                "answer asset materialization requires an active GBCRAG query"
            )
        if self.asset_cas_root is None:
            log.warning(
                "Skipping verified visual material for %d node(s); "
                "DOCARAG_ASSET_CAS_ROOT is not set. Text evidence (table cells, "
                "OCR, captions) still grounds the answer.",
                len(assets_by_node),
            )
            return

        if new_assets:
            remaining_assets = MAX_ANSWER_ASSETS - len(self._query_asset_refs)
            remaining_bytes = (
                MAX_ANSWER_ASSET_TOTAL_BYTES - self._query_asset_total_bytes
            )
            if len(new_assets) > remaining_assets or remaining_bytes < 1:
                raise AnswerAssetMaterializationError(
                    "answer asset selection exceeds the per-query asset limits"
                )
            resolver = AssetMaterialResolver(
                content_addressed_root=self.asset_cas_root,
                max_assets=remaining_assets,
                max_asset_bytes=MAX_ANSWER_ASSET_BYTES,
                max_total_bytes=remaining_bytes,
            )
            ordered = tuple(new_assets[key] for key in sorted(new_assets))
            try:
                lease = BookRAGAssetMaterializationLease(resolver, ordered)
            except Exception as exc:
                raise AnswerAssetMaterializationError(
                    "canonical answer asset materialization failed"
                ) from exc
            attestations = {
                asset.asset_id: lease.attestation_for(asset) for asset in ordered
            }
            self._query_asset_leases.append(lease)
            self._query_asset_refs.update(new_assets)
            self._query_asset_attestations.update(attestations)
            self._query_asset_total_bytes += sum(
                attestation.size_bytes for attestation in attestations.values()
            )

        for node in tree_data:
            node_id = node.get("index_id")
            refs = assets_by_node.get(node_id)
            if not refs:
                continue
            attestations = [
                self._query_asset_attestations[asset.asset_id] for asset in refs
            ]
            material_payloads = [attestation.to_dict() for attestation in attestations]
            verified_ids = [attestation.asset_id for attestation in attestations]
            node["img_path"] = attestations[0].materialized_path
            node["answer_selected_asset_ids"] = verified_ids
            node["asset_material_status"] = "verified"
            node["asset_materialization_attestations"] = material_payloads
            citation = node.get("citation")
            if not isinstance(citation, dict):
                raise AnswerAssetMaterializationError(
                    "verified answer asset requires a canonical citation"
                )
            citation["asset_material_status"] = "verified"
            citation["asset_material_scope"] = "answer_selected_assets"
            citation["verified_asset_ids"] = verified_ids
            citation["asset_materialization_attestations"] = material_payloads
            locations = citation.get("locations")
            if isinstance(locations, list):
                verified_set = set(verified_ids)
                for location in locations:
                    if not isinstance(location, dict):
                        continue
                    location_assets = location.get("assets")
                    if not isinstance(location_assets, list):
                        continue
                    location_ids = {
                        asset.get("asset_id")
                        for asset in location_assets
                        if isinstance(asset, Mapping)
                    }
                    matched = [
                        payload
                        for payload in material_payloads
                        if payload["asset_id"] in location_ids & verified_set
                    ]
                    if matched:
                        location["asset_material_status"] = "verified"
                        location["asset_material_scope"] = "answer_selected_assets"
                        location["verified_asset_ids"] = [
                            payload["asset_id"] for payload in matched
                        ]
                        location["asset_materialization_attestations"] = matched

    def _section_titles(self) -> List[str]:
        """Return the heading titles the planner is allowed to filter on."""

        return [
            node.meta_info.content
            for node in self.gbc_index.TreeIndex.get_nodes(hasRoot=False)
            if node.type == NodeType.TITLE
            and isinstance(node.meta_info.content, str)
            and node.meta_info.content.strip()
        ]

    def _process_retrieved_nodes(
        self, tree_data: List[Dict[str, Any]], iter_context: SubStep
    ) -> None:
        """Processes and categorizes retrieved nodes into the iteration context."""
        if self.evidence_hydrator is not None:
            tree_data = self.evidence_hydrator.hydrate(tree_data)
            self._materialize_answer_assets(tree_data)
        iter_context.retrieval_nodes = tree_data

        image_nodes = [node for node in tree_data if node["type"] == NodeType.IMAGE]
        text_nodes = [node for node in tree_data if node["type"] != NodeType.IMAGE]

        iter_context.iteration_image_nodes = image_nodes
        iter_context.iteration_text_nodes = text_nodes

    def _attach_candidate_fusion_trace(
        self, tree_data: List[Dict[str, Any]]
    ) -> None:
        """Persist the validated dense/BM25 rank trace with retrieved evidence."""

        fusion = getattr(self.retriever, "last_fusion_result", None)
        if fusion is None:
            return
        legs = {trace.leg_id: trace for trace in fusion.leg_traces}
        by_node_id: Dict[int, Dict[str, Any]] = {}
        for candidate in fusion.candidates:
            trace = {
                "fusion_id": fusion.fusion_id,
                "fused_candidate_id": candidate.fused_candidate_id,
                "fused_rank": candidate.fused_rank,
                "fused_score": candidate.fused_score,
                "modalities": list(candidate.modalities),
                "selected_representation_ids": list(
                    candidate.selected_representation_ids
                ),
                "selected_asset_ids": list(candidate.selected_asset_ids),
                "contributions": [
                    {
                        "leg_id": contribution.leg_id,
                        "leg_kind": legs[contribution.leg_id].leg_kind,
                        "backend_id": legs[contribution.leg_id].backend_id,
                        "selected_rank": contribution.selected_rank,
                        "selected_raw_score": contribution.selected_raw_score,
                        "rrf_contribution": contribution.rrf_contribution,
                    }
                    for contribution in candidate.contributions
                ],
            }
            for node_id in candidate.tree_node_ids:
                if isinstance(node_id, str) and re.fullmatch(r"-?\d+", node_id):
                    by_node_id[int(node_id)] = trace
        for node in tree_data:
            trace = by_node_id.get(node.get("index_id"))
            if trace is not None:
                node["candidate_fusion"] = trace

    def get_GBC_info(self, iter_context: SubStep) -> None:
        """
        1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        2. Use Three Layer Reranker to select most relevant TreeNodes in the subtree.
            2.1 PPR to rank the most relevant TreeNodes in the subtree.
            2.2 Rerank with text reranker model.
            2.3 Rerank with Multimodal method.
            Then: Skyline algorithm to select the most relevant TreeNodes.
        3. Combine the connected TreeNodes and Subgraph info to form the final GBC data info.
        """

        # 1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        # Get the subtree rooted at the selected section ID
        if self.variant == "wo_selector":
            log.info("Variant 'wo_selector' selected")
            subtree_nodes = self.gbc_index.TreeIndex.get_nodes(hasRoot=False)
        else:
            log.info(f"Using {self.variant} variant for retrieval.")
            retrieval_sec_ids = iter_context.retrieval_sec_ids
            subtree_nodes = self.gbc_index.TreeIndex.get_subtree_nodes(
                retrieval_sec_ids
            )

        subtree_ids = [node.index_id for node in subtree_nodes]

        subgraph: nx.Graph = self.gbc_index.GraphIndex.get_kg_subgraph(subtree_ids)

        start_ent_map = iter_context.gbc_entity_map

        tree_node_ids, res_entities = self.retriever.skyline_filter(
            iter_context.sub_query, subtree_nodes, subgraph, start_ent_map
        )

        log.info(f"After skyline filtering, select {len(tree_node_ids)} TreeNodes")

        Graph_data = self.gbc_index.GraphIndex.get_subgraph_data(res_entities)
        iter_context.iteration_graph_nodes = Graph_data.get("nodes", [])
        iter_context.iteration_graph_edges = Graph_data.get("edges", [])

        tree_data = self.gbc_index.TreeIndex.get_nodes_data(tree_node_ids)
        self._attach_candidate_fusion_trace(tree_data)
        self._process_retrieved_nodes(tree_data, iter_context)

    def _retrieve(
        self,
        query: str,
        iter_context: SubStep = None,
    ) -> None:
        """
        GBC retrieval following the steps:
        1. Extract entities from the query.
        2. Get the section nodes based on the entities.
        3. Use LLM to select the most relevant section based on the query and Section info.
        4. Use graph-based retrieval on the subgraph projected by the subtree (Select Section).

        iter_context: IterationStep, Iteration context for the current step.
        """

        Qent_GBCent_map = self._get_query_entity(query)
        iter_context.gbc_entity_map = Qent_GBCent_map

        tree_nodes = self.link_tree_node(Qent_GBCent_map)
        iter_context.linked_tree_nodes = tree_nodes

        # 3. Use LLM to select the most relevant section or supplementary sections
        if self.variant == "wo_selector":
            log.info("Variant 'wo_selector' selected: Skipping LLM section selection.")
            iter_context.retrieval_sec_ids = [
                self.gbc_index.TreeIndex.root_node.index_id
            ]
        else:
            self.llm_section_selection(query, tree_nodes, iter_context)

        # 4. Graph-based retrieval on subgraph projected by the subtree (Select Section)
        self.get_GBC_info(iter_context)

    def process_analysis(self, context: GBCRAGContext, query_analysis: PlanResult):
        log.info(f"Query analysis type: {query_analysis.query_type}")
        planner = getattr(self, "planner", None)
        if planner is not None:
            # Revalidate at the execution boundary.  This also protects
            # callers that deserialize a cached/model-produced PlanResult and
            # invoke ``process_analysis`` directly.
            query_analysis = planner.validate_plan(query_analysis)

        if query_analysis.query_type == "simple":
            query = query_analysis.original_query
            current_step = SubStep(sub_query=query, sub_number=1)
            self._retrieve(query, current_step)
            self._notify_retrieval_ready(context, current_step)

            final_answer, partial_answers = self.answer.answer_simple_question(
                query=query,
                retrieved_nodes=current_step.retrieval_nodes,
                entities=current_step.iteration_graph_nodes,
                relations=current_step.iteration_graph_edges,
                require_citations=self.evidence_hydrator is not None,
            )
            current_step.partial_answers = partial_answers
            current_step.generated_answer = final_answer
            current_step.answer_validation = _citation_validation_from_partials(
                partial_answers
            )

            context.iterations.append(current_step)
            context.final_answer = final_answer
            context.answer_validation = current_step.answer_validation
        elif query_analysis.query_type == "complex":
            # 1. Separate retrieval tasks from the full plan
            retrieval_tasks = [
                sub_q
                for sub_q in query_analysis.sub_questions
                if sub_q.type == "retrieval"
            ]

            # 2. Execute each retrieval task and collect the results
            sub_question_results = []
            for i, task in enumerate(retrieval_tasks):
                sub_question = task.question
                current_step = SubStep(sub_query=sub_question, sub_number=i + 1)
                self._retrieve(sub_question, current_step)
                self._notify_retrieval_ready(context, current_step)

                sub_answer, partial_answers = self.answer.answer_simple_question(
                    query=sub_question,
                    retrieved_nodes=current_step.retrieval_nodes,
                    entities=current_step.iteration_graph_nodes,
                    relations=current_step.iteration_graph_edges,
                    require_citations=self.evidence_hydrator is not None,
                    evidence_scope=f"Q{i + 1}",
                )
                current_step.partial_answers = partial_answers
                current_step.generated_answer = sub_answer
                current_step.answer_validation = _citation_validation_from_partials(
                    partial_answers
                )
                context.iterations.append(current_step)

                sub_question_results.append(
                    {
                        "question": sub_question,
                        "answer": sub_answer,
                        "answer_validation": current_step.answer_validation,
                        "evidence_bindings": _citation_bindings_from_validation(
                            current_step.answer_validation
                        ),
                    }
                )
            final_answer, final_validation = (
                self.answer.answer_complex_question_with_validation(
                    original_query=query_analysis.original_query,
                    sub_question_plan=query_analysis.sub_questions,
                    sub_question_results=sub_question_results,
                    require_citations=self.evidence_hydrator is not None,
                )
            )
            context.final_answer = final_answer
            context.answer_validation = final_validation

        elif query_analysis.query_type == "global":
            # Create a step for the global operation
            current_step = SubStep(
                sub_query=query_analysis.original_query, sub_number=1
            )

            # 1. Filter the tree nodes based on the plan's filters
            filtered_nodes: List[TreeNode] = filter_tree_nodes(
                self.gbc_index.TreeIndex, query_analysis.filters
            )
            operation = query_analysis.operation.upper()
            filtered_count = len(filtered_nodes)
            current_step.global_filtered_count = filtered_count
            current_step.global_evidence_limit = self.cfg.global_evidence_limit
            log.info("Global filter resulted in %d nodes.", filtered_count)

            # 2. Perform the specified operation
            if operation == "COUNT":
                # COUNT is exact and needs neither hydration nor a model call.
                # Avoid loading every canonical record merely to count an
                # already materialized filtered tree scope.
                count_result = filtered_count
                # You can format this into a more natural sentence if desired
                final_answer = (
                    f"Based on my analysis of the document, I found {count_result} items"
                    f" that answer the question: '{query_analysis.original_query}'"
                )

                current_step.partial_answers = [
                    {"source": "Direct Count", "content": final_answer}
                ]
                current_step.answer_validation = {
                    "status": "deterministic_count",
                    "valid": True,
                    "citations_required": False,
                }
                current_step.retrieval_nodes = []
                self._notify_retrieval_ready(context, current_step)
            elif filtered_count > self.cfg.global_evidence_limit:
                # A silent prefix/sample is inaccurate, while one model call
                # per item is an unbounded latency and cost failure. Until the
                # durable asynchronous hierarchical Map/Reduce worker lands,
                # fail explicitly and preserve the full scope in the trace.
                current_step.global_budget_exhausted = True
                current_step.retrieval_nodes = []
                final_answer = (
                    "This global query matched "
                    f"{filtered_count} evidence items, exceeding the synchronous "
                    f"limit of {self.cfg.global_evidence_limit}. Narrow the page, "
                    "section, or modality filter; no partial answer was generated."
                )
                current_step.partial_answers = [
                    {
                        "source": "Global evidence budget",
                        "content": final_answer,
                        "matched_count": filtered_count,
                        "evidence_limit": self.cfg.global_evidence_limit,
                        "budget_exhausted": True,
                    }
                ]
                current_step.answer_validation = {
                    "status": "not_generated_budget_exhausted",
                    "valid": False,
                    "citations_required": False,
                }
                self._notify_retrieval_ready(context, current_step)
            else:  # For LIST, SUMMARIZE, ANALYZE
                filter_nodes_ids = [node.index_id for node in filtered_nodes]
                tree_data = self.gbc_index.TreeIndex.get_nodes_data(filter_nodes_ids)
                self._process_retrieved_nodes(tree_data, current_step)
                self._notify_retrieval_ready(context, current_step)
                # Call the dedicated global answer agent method
                final_answer, partials = self.answer.answer_global_question(
                    original_query=query_analysis.original_query,
                    operation=operation,
                    filtered_nodes=current_step.retrieval_nodes,
                    require_citations=self.evidence_hydrator is not None,
                )
                current_step.partial_answers = partials
                current_step.answer_validation = _citation_validation_from_partials(
                    partials
                )

            current_step.generated_answer = final_answer
            context.iterations.append(current_step)
            context.final_answer = final_answer
            context.answer_validation = current_step.answer_validation
        else:
            log.warning(f"Unknown query type: {query_analysis.query_type}")
            context.final_answer = "I'm sorry, I cannot process this query."

    def _create_augmented_prompt(self, query: str) -> str:
        pass

    def generation(self, query: str, query_output_dir: str):
        self._begin_query_asset_materialization()
        query_failed = False
        try:
            self.planner.validate_query(query)
            context = GBCRAGContext(query=query)
            context.planning_limits = self._query_limits_trace()

            if self.variant == "wo_plan":
                log.info("Variant 'wo_plan' selected: Skipping LLM planning.")
                query_analysis = PlanResult(
                    query_type="simple",
                    original_query=query,
                )
            else:
                query_analysis: PlanResult = self.planner.analyze(
                    query, section_titles=self._section_titles()
                )

            context.plan = query_analysis
            self.process_analysis(context, query_analysis)

            log.info(f"Final answer for query '{query}': {context.final_answer}")
            retrieval_ids = self._save_retrieval_res(context, query_output_dir)

            return context.final_answer, retrieval_ids
        except BaseException:
            query_failed = True
            raise
        finally:
            try:
                # Re-hash every handed-out CAS path after the answer model has
                # consumed it. A mismatch prevents a successful answer return.
                self._finish_query_asset_materialization()
            except BaseException:
                if query_failed:
                    log.exception(
                        "Answer asset lease verification also failed while the "
                        "query was already failing"
                    )
                else:
                    raise

    def _save_retrieval_res(self, context: GBCRAGContext, query_output_dir: str):
        retrieval_ids = []

        # direct save the context to a json file
        retrieval_save_res = query_output_dir / "retrieval_res.json"
        context_dict = context.model_dump()
        with open(retrieval_save_res, "w", encoding="utf-8") as f:
            json.dump(context_dict, f, indent=2, ensure_ascii=False)
        log.info(f"Retrieval results saved to {retrieval_save_res}")

        # use the tree nodes as retrieval ids
        retrieval_ids = []
        for iter_step in context.iterations:
            text_nodes = iter_step.iteration_text_nodes
            if text_nodes:
                for node in text_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)
            image_nodes = iter_step.iteration_image_nodes
            if image_nodes:
                for node in image_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)

        retrieval_ids = sorted(retrieval_ids)

        return retrieval_ids

    def close(self):
        """Release rerank, attested tree, entity/graph, and model resources."""

        try:
            self._finish_query_asset_materialization()
        except Exception:
            log.exception("Failed to re-verify active answer asset leases on close")
        self.asset_cas_root = None
        reranker = getattr(self, "reranker", None)
        self.reranker = None
        tree_embedder = getattr(self, "tree_embedder", None)
        self.tree_embedder = None
        tree_vdb = getattr(self, "tree_vdb", None)
        self.tree_vdb = None
        retriever = getattr(self, "retriever", None)
        if retriever is not None:
            retriever.tree_vdb = None
            retriever.retrieval_binding = None
        gbc_index = getattr(self, "gbc_index", None)
        self.gbc_index = None
        self.embedder = None
        try:
            self._close_owned_resources(
                ("GBC reranker", reranker),
                ("TreeVDB embedder", tree_embedder),
                ("TreeVDB", tree_vdb),
                ("GBC index", gbc_index),
            )
        finally:
            super().close()
