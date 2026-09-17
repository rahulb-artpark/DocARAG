import os

from retrieval_engine.rag.base_strategy import BaseRAG
from retrieval_engine.rag.grounding import NO_RETRIEVED_EVIDENCE_ANSWER
from retrieval_engine.providers.language_model import LLM
from retrieval_engine.providers.vision_model import VLM
from retrieval_engine.indexes.vector_store import VectorStore
from retrieval_engine.configuration.strategies.gbc_vanilla import GBCVanillaConfig

from retrieval_engine.utils.text import TextProcessor


import json

import logging

log = logging.getLogger(__name__)


class GBCVanillaRAG(BaseRAG):
    """
    GBC Vanilla RAG class.
    Vanilla RAG based on GBC index,
    Directly retrieves the most relevant TreeNodes and Entities based on the similarity.
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM | None,
        config: GBCVanillaConfig,
        tree_vdb: VectorStore = None,
        graph_vdb: VectorStore = None,
    ):
        super().__init__(
            llm,
            name="GBC Vanilla RAG",
            description="Vanilla RAG based on GBC index",
        )
        self.vlm = vlm
        self.cfg = config
        self.tree_vdb = tree_vdb
        self.graph_vdb = graph_vdb
        self.embedder = self.tree_vdb.embedding_model if self.tree_vdb else None

        self.topk = self.cfg.topk
        self.topk_ent = self.cfg.topk_ent

    def _retrieve(self, query: str):
        # 1. Retrieve relevant TreeNodes using the tree_vdb
        tree_retrieval_res = self.tree_vdb.search(query_text=query, top_k=self.topk)

        # 2. Retrieve relevant Entities using the graph_vdb
        graph_retrieval_res = self.graph_vdb.search(
            query_text=query, top_k=self.topk_ent
        )
        return tree_retrieval_res, graph_retrieval_res

    def _create_augmented_prompt(
        self, query: str, tree_retrieval_res: list, graph_retrieval_res: list
    ):
        context_text = "Please refer to the following background information to answer the question. You should base your answer strictly on the provided information and not supplement it with outside knowledge. If the background information is insufficient to answer the question, please state that the provided information is not enough.\n\n--- Background Information ---\n"
        question_text = f"--- User Question ---\n{query}\n\n"
        context_text += question_text
        if tree_retrieval_res is None and graph_retrieval_res is None:
            context_text += "No relevant documents found.\n"
            return context_text

        if len(graph_retrieval_res) > 0:
            context_text += "\n--- Retrieved Entities ---\n"
            for i, ent in enumerate(graph_retrieval_res):
                context_text += f"Entity {i+1}: {ent['content']}\n"

        if len(tree_retrieval_res) > 0:
            context_text += "\n--- Retrieved Documents ---\n"
            for i, doc in enumerate(tree_retrieval_res):
                context_text += f"Text {i+1}: {doc['content']}\n"

        context_text = TextProcessor.split_text_into_chunks(
            text=context_text, max_length=self.llm.config.max_tokens - 400
        )
        context_text = context_text[0]  # take the first chunk only
        return context_text

    def generation(self, query: str, query_output_dir: str):

        tree_retrieval_res, graph_retrieval_res = self._retrieve(query)

        if len(tree_retrieval_res) == 0 and len(graph_retrieval_res) == 0:
            log.warning(f"No relevant information retrieved for query: {query}")
            return NO_RETRIEVED_EVIDENCE_ANSWER, []

        augmented_prompt = self._create_augmented_prompt(
            query=query,
            tree_retrieval_res=tree_retrieval_res,
            graph_retrieval_res=graph_retrieval_res,
        )

        final_answer = self.llm.get_completion(augmented_prompt, json_response=False)

        retrieval_ids = self._save_retrieval_res(
            tree_retrieval_res,
            graph_retrieval_res,
            query_output_dir,
        )
        return final_answer, retrieval_ids
        

    def _save_retrieval_res(self, tree_nodes, graph_nodes, query_output_dir: str):
        def _get_meta_info(doc):
            if "metadata" in doc:
                meta = doc.get("metadata", {})
                if "node_id" in meta:
                    node_id = meta["node_id"]
                elif "chunk_id" in meta:
                    node_id = meta["chunk_id"]
                else:
                    node_id = meta.get("node_id", meta.get("id", None))
            elif "node_id" in doc:
                node_id = doc["node_id"]
            else:
                node_id = doc["id"]
            return node_id

        retrieval_ids = []
        for doc in tree_nodes:
            node_id = _get_meta_info(doc)
            meta_info_dict = {
                "id": node_id,
                "content": doc["content"],
            }
            retrieval_ids.append(node_id)
            node_file_path = query_output_dir / f"tree_{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)

        for doc in graph_nodes:
            node_id = _get_meta_info(doc)
            meta_info_dict = {
                "id": node_id,
                "content": doc["content"],
            }
            retrieval_ids.append(node_id)
            node_file_path = query_output_dir / f"graph_{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)
        return retrieval_ids

    def close(self):
        try:
            embedder = getattr(self, "embedder", None)
            self.embedder = None
            stores = []
            for attribute in ("tree_vdb", "graph_vdb"):
                store = getattr(self, attribute, None)
                setattr(self, attribute, None)
                stores.append((attribute, store))
            self._close_owned_resources(
                ("shared vector embedder", embedder),
                *stores,
            )
        finally:
            super().close()
