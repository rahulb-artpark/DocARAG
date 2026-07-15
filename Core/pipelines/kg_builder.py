from Core.Index.Tree import DocumentTree, NodeType
from Core.Index.Graph import Graph
from Core.pipelines.kg_extractor import KGExtractor
from Core.pipelines.kg_refiner import KGRefiner
from Core.configs.system_config import SystemConfig
from Core.configs.graph_config import graph_artifact_variant
from Core.pipelines.vdb_index import tree_vector_scope

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker

import logging

log = logging.getLogger(__name__)

# print log for test
from rich.logging import RichHandler

import os
import time
from itertools import batched

# log_dir = "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/logs"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# log_file = os.path.join(log_dir, f"kg_builder_{time.strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     level="INFO",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         RichHandler(rich_tracebacks=True),  # RichHandler 会继续使用自己的漂亮格式
#         logging.FileHandler(
#             log_file, encoding="utf-8"
#         ),  # FileHandler 会使用上面定义的 format
#     ],
# )


def build_knowledge_graph(tree: DocumentTree, cfg: SystemConfig):
    """
    Build a knowledge graph from the given document tree.

    :param tree: DocumentTree object containing the document structure.
    :param graph_config: GraphConfig object containing configuration for the graph.
    :return: A tuple containing the KGExtractor and KGRefiner instances.
    """
    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.graph.image_description_force else None

    # try load_the graph if constructed before
    # graph_path = os.path.join(cfg.save_path, Graph._DATA_FILE)
    # if os.path.exists(graph_path):
    #     log.info(f"Loading existing knowledge graph from {graph_path}...")
    #     graph_index = Graph.load_from_dir(cfg.save_path)
    #     return graph_index
    # else:
    #     log.info("No existing knowledge graph found. Creating a new one...")

    variant = graph_artifact_variant(cfg.graph)

    # Try to load the graph index if it already exists
    target_graph_file = Graph._get_filename(variant)
    target_graph_path = os.path.join(cfg.save_path, target_graph_file)

    if os.path.exists(target_graph_path):
        log.info(
            f"Existing knowledge graph detected at {target_graph_path}. Trying to load..."
        )
        try:
            graph_index = Graph.load_from_dir(cfg.save_path, variant=variant)
            log.info("Loaded existing knowledge graph successfully. Skip rebuilding.")
            return graph_index
        except Exception as e:
            log.warning(
                f"Failed to load existing graph from {target_graph_path}, will rebuild. Reason: {e}"
            )
    else:
        log.info(
            f"No existing knowledge graph found at {target_graph_path}. Creating a new one..."
        )

    graph_index = Graph(save_path=cfg.save_path, variant=variant)

    kg_extractor = KGExtractor(
        cfg_graph=cfg.graph, llm=llm, vlm=vlm, save_path=cfg.save_path
    )
    entity_scope = {
        **tree_vector_scope(tree, cfg.vdb),
        "index_kind": "entity",
    }
    kg_refiner = KGRefiner(
        llm=llm,
        graph_config=cfg.graph,
        graph_index=graph_index,
        save_path=cfg.save_path,
        vdb_config=cfg.vdb,
        scope_filter=entity_scope,
        g=cfg.graph.g,
    )

    try:
        def refine_extraction(res):
            if cfg.graph.refine_type == "basic":
                kg_refiner.basic_kg_refiner(
                    entities=res.get("entities", []),
                    relationships=res.get("relations", []),
                    source_id=res.get("node_idx", -1),
                )
            elif cfg.graph.refine_type == "advanced":
                kg_refiner.advanced_kg_refiner(
                    entities=res.get("entities", []),
                    relationships=res.get("relations", []),
                    source_id=res.get("node_idx", -1),
                )

        # Keep at most one configured microbatch of extraction results in memory.
        # Each successful node has its own durable extraction file before refinement,
        # so a failed later node can be resumed without publishing an incomplete KG.
        source_nodes = (node for node in tree.nodes if node is not tree.root_node)
        extracted_count = 0
        for node_batch in batched(source_nodes, cfg.graph.extraction_batch_size):
            title_work = []
            ordinary_nodes = []
            for node in node_batch:
                if node.type == NodeType.TITLE:
                    title_work.append(
                        (
                            node,
                            tree.get_path_from_root(node.index_id),
                            tree.get_sibling_nodes(node.index_id),
                        )
                    )
                else:
                    ordinary_nodes.append(node)

            batch_results = []
            if title_work:
                batch_results.extend(kg_extractor.iter_extract_titles(title_work))
            if ordinary_nodes:
                batch_results.extend(kg_extractor.iter_extract_kg(ordinary_nodes))

            # Preserve the legacy deterministic node order even though title and
            # non-title provider streams are executed separately within a batch.
            batch_results.sort(key=lambda item: item.get("node_idx", -1))
            for result in batch_results:
                refine_extraction(result)
                extracted_count += 1
            log.info(
                "Committed KG extraction/refinement for %d/%d tree nodes",
                extracted_count,
                max(0, len(tree.nodes) - 1),
            )

        log.info("Knowledge graph extraction completed.")
        log.info("Extracted %d nodes from the document tree.", extracted_count)

        token_tracker = TokenTracker.get_instance()
        kg_extraction_cost = token_tracker.record_stage("kg_extraction")
        log.info(f"Knowledge graph extraction cost: {kg_extraction_cost}")

        kg_refiner.refine_entities()
        kg_refiner.refine_relation()

        log.info("Knowledge graph refinement completed.")
        kg_refinement_cost = token_tracker.record_stage("kg_refinement")
        log.info(f"Knowledge graph refinement cost: {kg_refinement_cost}")
        return graph_index
    finally:
        kg_refiner.close()
    # graph_index.save_graph()


if __name__ == "__main__":
    # We test the knowledge graph builder here
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(cfg.save_path))

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # Build the knowledge graph
    graph_index = build_knowledge_graph(tree_index, cfg)
    graph_index.save_graph()
    print("Knowledge graph built successfully.")
