from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from Core.Index.Tree import TreeNode, DocumentTree, NodeType
from Core.provider.embedding import TextEmbeddingProvider
from Core.rag.gbc_plan import Filter, PlanResult, parse_page_filter_value
import math
import logging


log = logging.getLogger(__name__)


class SubStep(BaseModel):
    sub_query: str
    sub_number: int
    gbc_entity_map: Dict[str, List[str]] = Field(default_factory=dict)
    linked_tree_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    linked_section_ids: List[int] = Field(default_factory=list)

    supplementary_ids: List[int] = Field(default_factory=list)
    selected_explanation: str = ""
    section_selection_status: str = "not_started"
    section_selection_policy: str = ""
    section_total_count: int = 0
    section_linked_count: int = 0
    section_candidate_limit: int = 0
    section_prompt_primary_count: int = 0
    section_prompt_remaining_count: int = 0
    section_prompt_candidate_count: int = 0
    section_prompt_candidate_ids: List[int] = Field(default_factory=list)
    section_omitted_count: int = 0
    section_prompt_chars: int = 0
    section_prompt_truncated_fields: int = 0
    section_selection_budget_exhausted: bool = False

    # store the retrieval info for visualization
    retrieval_sec_ids: List[int] = Field(default_factory=list)
    retrieval_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_text_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_image_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_graph_nodes: Union[str | List[Any]] = Field(default_factory=list)
    iteration_graph_edges: Union[str | List[Any]] = Field(default_factory=list)

    partial_answers: Union[str | List[Any]] = Field(default_factory=list)
    generated_answer: str = ""
    answer_validation: Dict[str, Any] = Field(default_factory=dict)
    global_filtered_count: int = 0
    global_evidence_limit: int = 0
    global_budget_exhausted: bool = False


class GBCRAGContext(BaseModel):
    query: str
    used_sec_ids: List[int] = Field(default_factory=list)
    all_sec_ids: List[int] = Field(default_factory=list)

    # Plan
    plan: PlanResult = None
    planning_limits: Dict[str, int] = Field(default_factory=dict)

    # Interation/ subquestion result
    iterations: List[SubStep] = Field(default_factory=list)

    # Output
    final_answer: str = ""
    answer_validation: Dict[str, Any] = Field(default_factory=dict)


# --- Pydantic Models for Structured Output (已更新) ---


def filter_tree_nodes(
    tree_index: DocumentTree,
    filters: List[Filter],
) -> List[TreeNode]:
    filtered_nodes = tree_index.get_nodes(hasRoot=False)
    for f in filters:
        f_type = f.filter_type
        f_value = f.filter_value
        if f_type == "document":
            # Explicit no-op scope. Whole-document requests must never be
            # represented by an invalid/missing structural value.
            continue
        if f_type in {"image", "table", "equation", "title"}:
            f_type = NodeType(f_type)
            filtered_nodes = [node for node in filtered_nodes if node.type == f_type]
        elif f_type == "page":
            start_page, end_page = parse_page_filter_value(f_value)
            filtered_nodes = [
                node
                for node in filtered_nodes
                if start_page <= node.meta_info.page_idx + 1 <= end_page
            ]
        elif f_type == "section":
            matched_nodes = []
            if not isinstance(f_value, str) or not f_value.strip():
                raise ValueError(
                    "section filter_value must be a non-empty section name"
                )
            section_name = f_value.strip().lower()
            for node in filtered_nodes:
                node_idx = node.index_id
                root_path = tree_index.get_path_from_root(node_idx)
                title_str = ""
                for pnode in root_path:
                    if pnode.type == NodeType.TITLE:
                        title_str += pnode.meta_info.content.lower()
                if section_name in title_str:
                    matched_nodes.append(node)
            filtered_nodes = matched_nodes
        else:
            raise ValueError(f"unknown global filter type: {f_type!r}")

    return filtered_nodes


def merge_ranker_scores(
    *ranker_scores_lists: List[Tuple[int, float]]
) -> Dict[int, List[float]]:
    """
    将多个评分器（ranker）的评分列表合并为一个字典。

    Args:
        *ranker_scores_lists: 任意数量的评分列表。
            每个列表的格式为 [(node_id, score), ...]。

    Returns:
        一个字典，将每个 node_id 映射到一个包含所有评分器分数的列表。
        格式: {node_id: [score1, score2, ...]}
    """
    num_rankers = len(ranker_scores_lists)
    # 使用 lambda 创建一个长度为 num_rankers 的默认值列表
    merged_scores = defaultdict(lambda: [0.0] * num_rankers)

    # 遍历每个评分列表并填充分数
    for i, scores_list in enumerate(ranker_scores_lists):
        for node_id, score in scores_list:
            merged_scores[node_id][i] = score

    return dict(merged_scores)


def calculate_skyline(
    merged_scores: Dict[int, List[float]],
    max_results: Optional[int] = None,
) -> List[Dict[str, Union[int, List[float]]]]:
    """
    根据合并后的分数计算 skyline 集合。
    一个点 p 被另一个点 s "支配" (dominate)，当且仅当 s 在所有维度上都不劣于 p，
    并且至少在一个维度上严格优于 p。

    Args:
        merged_scores: 一个字典，包含节点ID及其多维度的分数。
            格式: {node_id: [score1, score2, ...]}

    Returns:
        skyline_nodes: 不被任何其他节点支配的节点列表。
            格式: [{'node_id': id, 'dims': [score1, score2, ...]}, ...]
    """
    if max_results is not None and (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or max_results < 1
    ):
        raise ValueError("max_results must be a positive integer or None")

    # Normalize and validate dimensions before sorting. A deterministic score
    # order plus node ID is also the deterministic preference order when a
    # Pareto frontier contains more nodes than the configured result budget.
    nodes = []
    num_dims = None
    for node_id, raw_scores in merged_scores.items():
        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError) as error:
            raise ValueError("skyline scores must be finite numbers") from error
        if num_dims is None:
            num_dims = len(scores)
        elif len(scores) != num_dims:
            raise ValueError("all skyline score vectors must have equal dimensions")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("skyline scores must be finite numbers")
        nodes.append((node_id, scores))

    nodes.sort(key=lambda item: tuple(-score for score in item[1]) + (item[0],))

    # 如果节点数少于2，则它们都在skyline中
    if len(nodes) < 2:
        return [
            {"node_id": nid, "dims": scores}
            for nid, scores in nodes[:max_results]
        ]

    # 获取维度数量
    assert num_dims is not None

    # The production fusion path has two dimensions (graph and text). After
    # sorting by the first dimension, a point is on the exact frontier iff its
    # second score is strictly greater than every earlier first-score group.
    # Equal score vectors remain separate evidence nodes, matching Pareto's
    # strict-dominance definition. This is O(n log n), including sorting.
    if num_dims == 2:
        skyline = []
        best_second = -math.inf
        index = 0
        while index < len(nodes):
            first_score = nodes[index][1][0]
            group_end = index + 1
            while (
                group_end < len(nodes)
                and nodes[group_end][1][0] == first_score
            ):
                group_end += 1

            group_best_second = nodes[index][1][1]
            if group_best_second > best_second:
                for node_id, scores in nodes[index:group_end]:
                    if scores[1] != group_best_second:
                        break
                    skyline.append({"node_id": node_id, "dims": scores})
                    if max_results is not None and len(skyline) >= max_results:
                        return skyline
            best_second = max(best_second, group_best_second)
            index = group_end
        return skyline

    # Generic fallback for future fusions with three or more rankers.
    skyline = []
    for p_id, p_scores in nodes:
        is_dominated = False
        # 2. 检查当前节点 p 是否被已有的 skyline 节点 s 支配
        for s_node in skyline:
            s_scores = s_node["dims"]

            # s 在所有维度上都不劣于 p
            all_dims_ge = all(s_scores[i] >= p_scores[i] for i in range(num_dims))
            # s 至少在一个维度上严格优于 p
            any_dim_gt = any(s_scores[i] > p_scores[i] for i in range(num_dims))

            if all_dims_ge and any_dim_gt:
                is_dominated = True
                break  # 已被支配，无需再比较

        # 3. 如果未被任何 skyline 中的节点支配，则将其加入 skyline
        if not is_dominated:
            new_skyline = []
            for s_node in skyline:
                s_scores = s_node["dims"]
                # 检查 s 是否被 p 支配
                if not (
                    all(p_scores[i] >= s_scores[i] for i in range(num_dims))
                    and any(p_scores[i] > s_scores[i] for i in range(num_dims))
                ):
                    new_skyline.append(s_node)

            # 将 p 加入新的 skyline
            new_skyline.append({"node_id": p_id, "dims": p_scores})
            skyline = new_skyline

    # Keep the generic path semantically exact before applying its presentation
    # bound.  The optimized, bounded production path above covers the two
    # graph/text dimensions; future higher-dimensional fusion should not make
    # its Pareto result depend on an internal early-exit assumption.
    skyline.sort(
        key=lambda item: tuple(-score for score in item["dims"])
        + (item["node_id"],)
    )
    return skyline[:max_results]


def enhance_graph_with_semantic_links(
    graph: nx.Graph,
    embedder: TextEmbeddingProvider,
    x_percentile: float = 0.85,
    max_nodes: int = 512,
) -> nx.Graph:
    """Add bounded query-time semantic links to a selected graph scope.

    The legacy implementation allocates a dense ``n x n`` similarity matrix.
    Keep that optional enhancement strictly bounded until the durable index can
    provide offline ANN neighbors. Larger scopes retain their evidence-backed
    graph rather than risking quadratic query memory and latency.
    """
    if 1 < x_percentile <= 100:
        log.warning(
            "Interpreting legacy x_percentile=%s as a percentage; use a 0..1 fraction",
            x_percentile,
        )
        x_percentile /= 100
    if not 0 <= x_percentile <= 1:
        raise ValueError("x_percentile must be a fraction in [0, 1]")

    enhanced_graph = graph.copy()
    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    # 如果节点数不足以添加边，直接返回副本
    if len(nodes) <= 1:
        return enhanced_graph

    if n_nodes > max_nodes:
        log.warning(
            "Skipping query-time semantic graph augmentation for %s nodes; "
            "bounded limit is %s. Original evidence-backed edges are retained.",
            n_nodes,
            max_nodes,
        )
        return enhanced_graph

    log.info("Enhancing selected graph scope with %s nodes", n_nodes)

    try:
        node_embeddings = embedder.embed_texts(nodes)
    except Exception as e:
        log.warning("Semantic graph augmentation embedding failed: %s", e)
        return enhanced_graph

    # 计算所有节点对之间的余弦相似度矩阵
    # sim_matrix[i][j] 是 nodes[i] 和 nodes[j] 之间的相似度
    sim_matrix = cosine_similarity(node_embeddings)

    # 2.1 创建一个从节点名到矩阵索引的映射，用于高效查找
    node_to_idx = {node_name: i for i, node_name in enumerate(nodes)}
    # 2.2 遍历原始图的边，提取它们的相似度
    existing_edge_similarities = []
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            similarity = sim_matrix[idx_u, idx_v]
            existing_edge_similarities.append(similarity)

    # 2.3 基于提取出的相似度计算阈值
    if existing_edge_similarities:
        scores_array = np.array(existing_edge_similarities)
        similarity_threshold = np.percentile(scores_array, x_percentile * 100)
    else:
        # 如果原始图没有边，无法计算基准，设置一个非常高的阈值
        # 这样就不会添加任何新边，是安全的默认行为
        similarity_threshold = 1.0

    log.info(f"Base similarities on {len(existing_edge_similarities)} existing edges.")
    log.info(
        f"Calculated similarity threshold at {x_percentile*100:.0f}th percentile: {similarity_threshold:.4f}"
    )

    if n_nodes > 0:
        # 平均度数 = 2 * |E| / |V|
        average_degree = (2 * graph.number_of_edges()) / n_nodes
        # 应用 ceiling 并确保最小值至少为1
        max_new_edges = max(1, math.ceil(average_degree))
    else:
        max_new_edges = 1  # 对于空图的安全默认值

    log.info(f"Original graph average degree: {average_degree:.2f}")
    log.info(f"Global limit for adding new edges per node set to: {max_new_edges}")

    for i, source_node in enumerate(nodes):
        new_edges_added_for_node = 0

        similarities = sorted(
            zip(sim_matrix[i], nodes), key=lambda x: x[0], reverse=True
        )

        for sim_score, target_node in similarities:
            if new_edges_added_for_node >= max_new_edges:
                break
            if source_node == target_node:
                continue

            # 条件1: 相似度高于全局阈值
            # 条件2: 边在 *原始* 图中不存在
            if sim_score > similarity_threshold and not graph.has_edge(
                source_node, target_node
            ):
                if not enhanced_graph.has_edge(source_node, target_node):
                    enhanced_graph.add_edge(
                        source_node, target_node, weight=float(sim_score)
                    )
                    new_edges_added_for_node += 1

    original_edges = graph.number_of_edges()
    new_edges = enhanced_graph.number_of_edges()
    log.info(
        f"Enhancement complete. Edges increased from {original_edges} to {new_edges}."
    )

    return enhanced_graph
