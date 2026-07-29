import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from Core.configs.rag.mm_config import MMConfig
from Core.provider.embedding import MultimodalEmbeddingProvider
from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from Core.provider.vlm import VLM
from Core.rag.base_rag import BaseRAG
from Core.rag.grounding import NO_RETRIEVED_EVIDENCE_ANSWER

log = logging.getLogger(__name__)


MAX_VLM_IMAGES = 2
NO_USABLE_EVIDENCE_ANSWER = NO_RETRIEVED_EVIDENCE_ANSWER


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retrieval_modality(metadata: Dict[str, Any]) -> str:
    """Read the new explicit modality while retaining legacy ``type`` support."""

    modality = metadata.get("modality", metadata.get("type", "text"))
    if hasattr(modality, "value"):
        modality = modality.value
    return modality if modality in {"text", "image"} else "text"


class MMRAG(BaseRAG):
    """
    一个封装了检索、增强、生成完整流程的多模态RAG流水线。
    """

    def __init__(
        self,
        config: MMConfig,
        vector_store: VectorStore,
        llm: LLM,
        vlm: VLM,
        topk: int = 3,
    ):
        super().__init__(
            llm=llm,
            name="MM RAG",
            description="Multimodal Vanilla Retrieval Augmented Generation",
        )
        self.cfg = config
        self.vlm = vlm
        self.vdb = vector_store
        log.info("MultimodalRAGPipeline initialized.")
        self.topk = topk

    def _retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata_filter: Dict[str, Any] | None = None,
        document_version_id: str | None = None,
    ):
        optional_filters: Dict[str, Any] = {}
        if metadata_filter is not None:
            optional_filters["metadata_filter"] = metadata_filter
        if document_version_id is not None:
            optional_filters["document_version_id"] = document_version_id
        return self.vdb.search(query_text=query, top_k=top_k, **optional_filters)

    def _select_usable_evidence(
        self,
        retrieved_docs,
        *,
        max_images: int = MAX_VLM_IMAGES,
    ) -> List[Dict[str, Any]]:
        """Return the exact bounded evidence set that may reach generation.

        Asset validation, VLM image limits, prompting, auditing, and returned
        node IDs must all operate on the same list.  Otherwise an unavailable
        or changed image can be reported as used even though the model never
        received it.
        """

        usable: List[Dict[str, Any]] = []
        selected_images = 0
        for doc in retrieved_docs or ():
            if not isinstance(doc, dict) or not isinstance(doc.get("metadata"), dict):
                log.warning("Skipping malformed multimodal retrieval record")
                continue
            content_type = _retrieval_modality(doc["metadata"])
            content = doc.get("content")
            if content_type == "text":
                if not isinstance(content, str) or not content.strip():
                    log.warning(
                        "Skipping empty text retrieval record %s", doc.get("id")
                    )
                    continue
                usable.append(doc)
                continue

            if selected_images >= max_images:
                log.info(
                    "Skipping retrieval image %s because the VLM image limit is %d",
                    doc.get("id"),
                    max_images,
                )
                continue
            if not isinstance(content, str) or not content:
                log.warning("Skipping image retrieval record with no asset path")
                continue
            image_path = Path(content)
            try:
                if not image_path.is_file():
                    log.warning(
                        "Skipping missing retrieval asset for vector record %s",
                        doc.get("id"),
                    )
                    continue
                expected_sha256 = doc["metadata"].get("asset_sha256")
                if expected_sha256 is not None:
                    actual_sha256 = _file_sha256(image_path)
                    if actual_sha256 != expected_sha256:
                        log.error(
                            "Skipping changed retrieval asset for vector record %s",
                            doc.get("id"),
                        )
                        continue
            except OSError:
                log.exception(
                    "Skipping unreadable retrieval asset for vector record %s",
                    doc.get("id"),
                )
                continue
            usable.append(doc)
            selected_images += 1
        return usable

    def _format_augmented_prompt(self, query: str, retrieved_docs) -> tuple[str, list]:
        context_text = (
            "Please refer to the following background information to answer "
            "the question.\n\n--- Background Information ---\n"
        )
        context_images = []
        question_text = f"--- User Question ---\n{query}\n\n"
        if retrieved_docs is None:
            context_text += "No relevant documents found.\n"
            context_text += question_text
            return context_text, context_images

        for i, doc in enumerate(retrieved_docs):
            content_type = _retrieval_modality(doc["metadata"])
            if content_type == "image":
                image_path = doc["content"]
                context_images.append(image_path)
                context_text += (
                    f"Image {i+1}: A relevant image is provided at the path: "
                    f"{image_path}\n"
                )
            else:
                context_text += f"Text {i+1}: {doc['content']}\n"

        context_text += question_text

        return context_text, context_images

    def _create_augmented_prompt(
        self, query: str, retrieved_docs=None
    ) -> tuple[str, list]:
        """Build a prompt from validated evidence (compatibility entry point)."""

        if retrieved_docs is None:
            return self._format_augmented_prompt(query, None)
        usable_docs = self._select_usable_evidence(retrieved_docs)
        return self._format_augmented_prompt(query, usable_docs)

    def _save_retrieval_res(self, context_nodes, query_output_dir) -> List[Dict]:
        retrieval_ids = []
        seen_node_ids = set()
        output_dir = Path(query_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for doc in context_nodes:
            content_type = _retrieval_modality(doc["metadata"])
            if content_type not in ["text", "image"]:
                log.warning(
                    f"Unsupported content type: {content_type}. Skipping this document."
                )
                continue
            node_id = doc["metadata"].get("node_id")

            meta_info_dict = {
                "node_id": node_id,
                "vector_record_id": doc.get("id"),
                "type": content_type,
                "modality": content_type,
                "node_type": doc["metadata"].get("node_type"),
                "content": doc["content"],
            }
            for trace_field in (
                "document_version_id",
                "canonical_block_id",
                "canonical_revision_id",
                "derived_evidence_id",
                "observation_id",
                "page_id",
                "page_idx",
                "source_sha256",
                "asset_sha256",
            ):
                if trace_field in doc["metadata"]:
                    meta_info_dict[trace_field] = doc["metadata"][trace_field]
            if content_type == "image":
                img_path = doc["content"]
                meta_info_dict["img_path"] = img_path

            if node_id is not None and node_id not in seen_node_ids:
                retrieval_ids.append(node_id)
                seen_node_ids.add(node_id)
            # Text and visual representations of the same semantic node are
            # separate vector records. Key audit files by the immutable record
            # identity so one modality cannot overwrite another, and hash the
            # external value so it can never become a path component.
            record_identity = str(doc.get("id") or "")
            if not record_identity:
                record_identity = json.dumps(
                    {
                        "node_id": node_id,
                        "modality": content_type,
                        "content": doc.get("content"),
                        "metadata": doc.get("metadata"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
            record_digest = hashlib.sha256(
                record_identity.encode("utf-8")
            ).hexdigest()[:24]
            node_file_path = output_dir / f"retrieval-{record_digest}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)

        log.info("Saved retrieval results to output directory.")

        return retrieval_ids

    def generation(
        self,
        query: str,
        query_output_dir: str,
        *,
        metadata_filter: Dict[str, Any] | None = None,
        document_version_id: str | None = None,
    ) -> tuple:
        """
        Generates an answer for a given query and returns the answer along with the context used.
        Returns:
            Tuple[str, List[Any]]: A tuple containing the final answer string and a list of the context nodes.
        """
        retrieved_docs = self._retrieve(
            query,
            top_k=self.topk,
            metadata_filter=metadata_filter,
            document_version_id=document_version_id,
        )
        if not retrieved_docs:
            log.warning("No multimodal retrieval evidence found")
            return NO_RETRIEVED_EVIDENCE_ANSWER, []

        usable_docs = self._select_usable_evidence(retrieved_docs)
        if not usable_docs:
            # Retrieval happened, but none of its output is safe/usable.  Do
            # not turn this integrity failure into an ungrounded LLM call or
            # claim that the rejected vector records were used.
            log.warning("All retrieved multimodal evidence was rejected")
            return NO_USABLE_EVIDENCE_ANSWER, []

        context_text, context_images = self._format_augmented_prompt(
            query, usable_docs
        )

        if len(context_images) > 0:
            # if there are images, use VLM to generate the answer
            final_answer = self.vlm.generate(
                prompt_or_memory=context_text, images=context_images
            )
        else:
            # if no images, fallback to LLM generation
            final_answer = self.llm.get_completion(context_text, json_response=False)
        retrieval_ids = self._save_retrieval_res(
            usable_docs, query_output_dir=query_output_dir
        )
        return final_answer, retrieval_ids

    def run(self, query: str, query_output_dir: str) -> Dict[str, Any]:
        answer, retrieved_docs = self.generation(query, query_output_dir)
        return {"answer": answer, "retrieved_docs": retrieved_docs}

    def close(self):
        store = getattr(self, "vdb", None)
        self.vdb = None
        embedding_model = getattr(store, "embedding_model", None)
        try:
            if store is not None and store.embedding_model is embedding_model:
                # Break the store reference before closing so its own cleanup
                # cannot hide or retain the heavyweight provider.
                store.embedding_model = None
            self._close_owned_resources(("vector embedder", embedding_model))
            if isinstance(embedding_model, MultimodalEmbeddingProvider):
                log.info("Closed MultimodalEmbeddingProvider and released model weights.")
        finally:
            try:
                self._close_owned_resources(("vector store", store))
            finally:
                super().close()
        log.info("MultimodalRAGPipeline resources have been released.")
