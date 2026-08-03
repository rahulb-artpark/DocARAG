from Core.Index.Tree import DocumentTree, NodeType
from Core.Index.Graph import Graph
from Core.pipelines.kg_extractor import KGExtractor, is_extractable
from Core.pipelines.kg_refiner import KGRefiner
from Core.configs.system_config import SystemConfig
from Core.configs.graph_config import graph_artifact_variant
from Core.configs.llm_config import indexing_llm_config
from Core.pipelines.kg_refinement_state import (
    clear_state as clear_refinement_state,
    load_state as load_refinement_state,
    save_state as save_refinement_state,
)
from Core.pipelines.vdb_index import tree_vector_scope

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker

import logging

log = logging.getLogger(__name__)

# print log for test
from rich.logging import RichHandler

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from itertools import batched


EXTRACTION_GAPS_EVENT = "kg.extraction.gaps"


def extraction_gaps_payload(tree, failed_node_ids, *, variant=None) -> dict:
    """Which nodes the graph does not cover, as the stage's own record.

    Stopping the stage instead was worse than the loss it guarded against: a
    handful of figures a provider refused would leave a document with no index
    at all, and resuming cannot help when the refusal is deterministic. So the
    graph is published without them and the gap is stated rather than left for
    a reader to notice.

    It is stated in Postgres, on the graph stage's own event stream, because
    that is where this run's state lives. An earlier version of this wrote a
    JSON file beside the graph, which put run state back in exactly the place
    the store exists to replace -- and a file beside a graph is free to
    disagree with the database, which is how a whole class of these bugs
    starts.
    """

    by_id = {node.index_id: node for node in tree.nodes}
    missing = []
    for node_id in sorted(failed_node_ids):
        node = by_id.get(node_id)
        missing.append(
            {
                "node_idx": node_id,
                "node_type": getattr(getattr(node, "type", None), "value", None),
            }
        )
    return {
        "variant": variant,
        "extractable_nodes": sum(
            1 for node in tree.nodes if node is not tree.root_node
        ),
        "missing_count": len(missing),
        "missing": missing,
    }

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


def build_knowledge_graph(tree: DocumentTree, cfg: SystemConfig, stage=None):
    """
    Build a knowledge graph from the given document tree.

    :param tree: DocumentTree object containing the document structure.
    :param stage: the durable stage whose event stream records this run's state.
    :param graph_config: GraphConfig object containing configuration for the graph.
    :return: A tuple containing the KGExtractor and KGRefiner instances.
    """
    llm = LLM(indexing_llm_config(cfg.llm))
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

    resumed = load_refinement_state(cfg.save_path, variant=variant)
    if resumed is None:
        graph_index = Graph(save_path=cfg.save_path, variant=variant)
        refined_node_ids: set[int] = set()
    else:
        graph_index = resumed.graph
        refined_node_ids = set(resumed.refined_node_ids)

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
    if resumed is not None:
        # KGRefiner resets the entity collection on construction, so the vector
        # store has to be rebuilt from the restored graph before any resolution.
        kg_refiner.entity_alias_map.update(resumed.alias_map)
        kg_refiner.rebuild_vdb_from_graph()

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

        def extract_batch(node_batch):
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
            return batch_results

        # Page furniture is excluded by decision, not by capability, so it is
        # dropped here instead of being sent into extraction where the resulting
        # error would be indistinguishable from a node that genuinely failed.
        extractable_nodes = [
            node
            for node in tree.nodes
            if node is not tree.root_node and is_extractable(node.type)
        ]
        skipped = max(0, len(tree.nodes) - 1 - len(extractable_nodes))
        if skipped:
            log.info(
                "Skipping %d page-furniture node(s); %d node(s) are extractable",
                skipped,
                len(extractable_nodes),
            )

        # Each successful node has its own durable extraction file before refinement,
        # so a failed later node can be resumed without publishing an incomplete KG.
        source_nodes = (
            node for node in extractable_nodes if node.index_id not in refined_node_ids
        )
        extractable_total = len(extractable_nodes)
        extracted_count = len(refined_node_ids)
        failed_node_ids: set[int] = set()
        node_batches = batched(source_nodes, cfg.graph.extraction_batch_size)

        # Refinement is serial and holds one provider slot, leaving the rest of
        # max_workers idle. Extracting the next batch while the current one is
        # being refined keeps them busy; the provider's own admission semaphore
        # still caps total concurrency, so this cannot oversubscribe the model.
        # At most two microbatches of extraction results are held at once.
        def commit_batches(batches):
            nonlocal extracted_count
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kg-extraction"
            ) as extraction_pump:
                # The pump thread is where iter_bounded_ordered copies the context
                # it hands to its own workers. Submitting bare would copy this
                # thread's empty context instead of the run's, and every record
                # the extractor emits would be filtered out of the run log --
                # which is how a whole document's failed nodes once went unseen.
                def submit(batch):
                    return extraction_pump.submit(copy_context().run, extract_batch, batch)

                upcoming = next(batches, None)
                pending = submit(upcoming) if upcoming is not None else None
                while pending is not None:
                    batch_results = pending.result()
                    upcoming = next(batches, None)
                    pending = submit(upcoming) if upcoming is not None else None

                    for result in batch_results:
                        node_idx = result.get("node_idx", -1)
                        if result.get("extraction_failed"):
                            # Leaving it out of refined_node_ids is what lets a
                            # retry re-attempt it; counting it as extracted here
                            # is what used to hide the loss.
                            failed_node_ids.add(node_idx)
                            continue
                        refine_extraction(result)
                        refined_node_ids.add(node_idx)
                        failed_node_ids.discard(node_idx)
                        extracted_count += 1
                    save_refinement_state(
                        cfg.save_path,
                        graph=graph_index,
                        alias_map=kg_refiner.entity_alias_map,
                        refined_node_ids=refined_node_ids,
                        variant=variant,
                    )
                    log.info(
                        "Committed KG extraction/refinement for %d/%d tree nodes",
                        extracted_count,
                        extractable_total,
                    )
                    kg_refiner.er_stats.log_snapshot(len(graph_index.kg.nodes))

        commit_batches(node_batches)

        if failed_node_ids:
            # A transient provider failure is the common case, and re-running the
            # whole stage to recover a handful of nodes is expensive, so retry
            # them once here before deciding the graph cannot be published.
            log.warning(
                "Retrying KG extraction for %d node(s) that failed: %s",
                len(failed_node_ids),
                sorted(failed_node_ids),
            )
            retry_nodes = [
                node for node in extractable_nodes if node.index_id in failed_node_ids
            ]
            commit_batches(batched(retry_nodes, cfg.graph.extraction_batch_size))

        # The gap is published with the graph rather than in place of it. It is
        # logged at error level and written down, so it is visible without being
        # fatal -- the failure mode this replaced left a document unindexable.
        if failed_node_ids:
            log.error(
                "Knowledge graph is missing %d node(s) whose extraction failed twice: %s. "
                "Recorded on the graph stage as %s; the graph is published without them.",
                len(failed_node_ids),
                sorted(failed_node_ids),
                EXTRACTION_GAPS_EVENT,
            )
        if stage is not None:
            # Recorded on every run, so "no gaps" is a statement the run made
            # rather than the absence of one.
            stage.checkpoint(
                EXTRACTION_GAPS_EVENT,
                extraction_gaps_payload(tree, failed_node_ids, variant=variant),
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
        clear_refinement_state(cfg.save_path)
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
