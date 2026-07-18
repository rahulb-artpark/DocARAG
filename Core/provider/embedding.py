from __future__ import annotations

import os

import numpy as np
from typing import Any, List, Optional, Dict
import math
from tqdm import tqdm
import json

from abc import ABC, abstractmethod

import logging
import gc
import weakref
from contextlib import contextmanager
from threading import Condition, Event, Lock, RLock

from Core.configs.llm_config import _normalize_api_base

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    MM_EMBEDDER: bool = False

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        """释放模型和相关资源。"""
        pass


class _GmeSharedRuntime:
    """One heavyweight GME runtime shared by explicit provider handles.

    The owner count is protected by ``GmeEmbeddingProvider._pool_lock``.  The
    condition protects in-flight calls so the final owner cannot unload model
    weights while a request is still using them.
    """

    def __init__(
        self,
        model_name: str,
        device_strategy: str,
        torch_dtype: str = "bfloat16",
        *,
        max_active_calls: int = 1,
    ) -> None:
        if (
            isinstance(max_active_calls, bool)
            or not isinstance(max_active_calls, int)
            or max_active_calls < 1
        ):
            raise ValueError("max_active_calls must be at least 1")
        self.model_name = model_name
        self.device_strategy = device_strategy
        self.torch_dtype = torch_dtype
        self.max_active_calls = max_active_calls
        self.owner_count = 0
        self._active_calls = 0
        self._waiting_calls = 0
        self._closing = False
        self._condition = Condition()
        self.torch = None
        self.torch_functional = None
        self.model = None
        try:
            self._load_model()
        except BaseException:
            # A failed load is never inserted into the shared pool.  Release
            # any partially allocated CUDA/model state before propagating it.
            try:
                self.release_weights()
            except Exception:
                log.exception("Failed to clean a partially loaded GME runtime")
            raise

    def _load_model(self) -> None:
        import torch
        from torch.nn import functional as torch_functional
        from transformers import AutoModel

        self.torch = torch
        self.torch_functional = torch_functional
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self._resolved_dtype(torch),
            trust_remote_code=True,
            device_map=self.device_strategy,
            use_fast=True,
        )
        self.model.eval()

    def _resolved_dtype(self, torch) -> Any:
        """Resolve the load dtype, defaulting away from a float32 checkpoint.

        gme-Qwen2-VL ships float32 weights, which cost twice the VRAM its
        architecture needs and leave no room beside a large local LLM.
        """

        if self.torch_dtype in (None, "", "auto"):
            return "auto"
        dtype = getattr(torch, self.torch_dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.torch_dtype}")
        return dtype

    def acquire_call(self, *, cancelled=None) -> None:
        """Admit one bounded inference call or reject a closing owner.

        A single admitted call is the conservative default because concurrent
        forward passes on one local model can multiply transient GPU memory.
        ``cancelled`` belongs to the provider handle, allowing its ``close``
        path to wake a queued call without holding the provider state lock.
        """

        is_cancelled = cancelled or (lambda: False)
        with self._condition:
            while (
                not self._closing
                and self.model is not None
                and not is_cancelled()
                and self._active_calls >= self.max_active_calls
            ):
                self._waiting_calls += 1
                # This notification also makes admission state observable to
                # deterministic lifecycle tests and operational diagnostics.
                self._condition.notify_all()
                try:
                    self._condition.wait()
                finally:
                    self._waiting_calls -= 1

            if is_cancelled():
                raise RuntimeError("GME embedding provider closed while waiting")
            if self._closing or self.model is None:
                raise RuntimeError("GME embedding runtime is closing")
            self._active_calls += 1

    def release_call(self) -> None:
        with self._condition:
            self._active_calls -= 1
            if self._active_calls < 0:
                # This indicates an internal lifecycle error, not a caller
                # input error.  Restore a valid counter before failing loudly.
                self._active_calls = 0
                raise RuntimeError("GME runtime call lease released twice")
            self._condition.notify_all()

    def notify_waiters(self) -> None:
        """Wake admission waiters after a provider handle is closed."""

        with self._condition:
            self._condition.notify_all()

    def wait_until_idle(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            while self._active_calls:
                self._condition.wait()

    def release_weights(self) -> None:
        model = self.model
        self.model = None
        torch_runtime = self.torch
        self.torch = None
        self.torch_functional = None

        if model is not None:
            del model
        gc.collect()
        if torch_runtime is not None and torch_runtime.cuda.is_available():
            torch_runtime.cuda.empty_cache()


class GmeEmbeddingProvider(BaseEmbedder):
    """An owned handle over a keyed, process-local GME model runtime.

    Constructing a provider acquires one ownership reference. Providers with
    the same ``(model_name, device)`` reuse the heavyweight model, but remain
    distinct handles and must each be closed. The last close waits for active
    calls and then unloads the shared weights. Different configurations never
    silently share a model.

    ``_instance`` remains only as a compatibility hook for old manually
    constructed test doubles; normal construction no longer uses a singleton.
    """

    MM_EMBEDDER: bool = True

    _instance = None
    _pool_lock = RLock()
    _runtime_pool: dict[tuple[str, str], _GmeSharedRuntime] = {}

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(torch_dtype, str) or not torch_dtype.strip():
            raise ValueError("torch_dtype must be a non-empty string")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_name = model_name.strip()
        self.device_strategy = device.strip().lower()
        self.torch_dtype = torch_dtype.strip()
        self._pool_key = (self.model_name, self.device_strategy, self.torch_dtype)
        self._state_lock = Lock()
        self._closed_event = Event()
        self._runtime = None
        self._closed = True
        self._initialized = False
        self._torch = None
        self._torch_functional = None

        log.info(
            "Acquiring GME runtime for model=%s device=%s",
            self.model_name,
            self.device_strategy,
        )
        with type(self)._pool_lock:
            runtime = type(self)._runtime_pool.get(self._pool_key)
            if runtime is None:
                log.info("Loading a new shared GME model runtime...")
                try:
                    runtime = _GmeSharedRuntime(
                        self.model_name,
                        self.device_strategy,
                        self.torch_dtype,
                    )
                except BaseException:
                    log.error(
                        "Error loading GME model '%s'.",
                        self.model_name,
                        exc_info=True,
                    )
                    raise
                type(self)._runtime_pool[self._pool_key] = runtime
                log.info("GME model loaded successfully.")
            else:
                log.info("Reusing compatible shared GME model runtime.")
            runtime.owner_count += 1

        self._runtime = runtime
        # These aliases preserve the historical inspection surface while all
        # operations and shutdown decisions remain bound to ``_runtime``.
        self.model = runtime.model
        self._torch = runtime.torch
        self._torch_functional = runtime.torch_functional
        self._initialized = True
        self._closed = False

    @contextmanager
    def _use_runtime(self):
        with self._state_lock:
            runtime = self._runtime
            if self._closed or runtime is None:
                raise RuntimeError("GME embedding provider is closed")
            closed_event = self._closed_event

        # Admission can wait behind another handle. Do not retain the provider
        # lock: close() must be able to cancel this waiter and release its
        # ownership without deadlocking on the gate.
        runtime.acquire_call(cancelled=closed_event.is_set)
        try:
            yield runtime
        finally:
            runtime.release_call()

    @staticmethod
    def _normalize_with_runtime(
        runtime: _GmeSharedRuntime,
        embeddings: Any,
    ) -> np.ndarray:
        functional = runtime.torch_functional
        if functional is None:
            raise RuntimeError("GME embedding provider is closed")
        normalized_embeddings = functional.normalize(embeddings, p=2, dim=-1)
        return normalized_embeddings.detach().float().cpu().numpy()

    def embed_texts(
        self, texts: List[str], instruction: Optional[str] = None, batch_size=4
    ) -> np.ndarray:
        """
        为一批文本生成嵌入向量。

        Args:
            texts (List[str]): 需要编码的文本列表。
            instruction (Optional[str]): 用于指导文本嵌入生成的指令。如果为None，则使用模型的默认指令。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_texts, embedding_dim)。
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        with self._use_runtime() as runtime:
            torch_runtime = runtime.torch
            all_embeddings = []
            with torch_runtime.no_grad():
                for i in tqdm(
                    range(0, len(texts), batch_size), desc="Embedding Texts"
                ):
                    batch_texts = texts[i : i + batch_size]

                    embeddings_tensor = runtime.model.get_text_embeddings(
                        texts=batch_texts, instruction=instruction
                    )
                    all_embeddings.append(embeddings_tensor.cpu())

            # 将所有批次的结果拼接成一个Tensor
            full_embeddings_tensor = torch_runtime.cat(all_embeddings, dim=0)

            return self._normalize_with_runtime(runtime, full_embeddings_tensor)

    def embed_images(self, images: List[str], batch_size: int = 4) -> np.ndarray:
        """
        为一批图像生成嵌入向量（已分批处理）。

        Args:
            images (List[str]): 需要编码的图像路径列表。
            batch_size (int): 每个批次处理的图像数量，默认为16。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_images, embedding_dim)。
        """
        if not images or not isinstance(images, list):
            raise ValueError("Input 'images' must be a non-empty list of image paths.")

        with self._use_runtime() as runtime:
            torch_runtime = runtime.torch
            all_embeddings = []
            with torch_runtime.no_grad():
                for i in tqdm(
                    range(0, len(images), batch_size), desc="Embedding Images"
                ):
                    batch_images = images[i : i + batch_size]

                    embeddings_tensor = runtime.model.get_image_embeddings(
                        images=batch_images
                    )
                    all_embeddings.append(embeddings_tensor.cpu())

            full_embeddings_tensor = torch_runtime.cat(all_embeddings, dim=0)

            return self._normalize_with_runtime(runtime, full_embeddings_tensor)

    def embed_fused(
        self, images: List[str], texts: List[str], batch_size: int = 4
    ) -> np.ndarray:
        """
        为一一对应的文本-图像对生成融合嵌入向量（已分批处理）。

        Args:
            texts (List[str]): 文本列表。
            images (List[str]): 图像路径列表。两个列表长度必须相等。
            batch_size (int): 每个批次处理的图文对数量，默认为4

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_pairs, embedding_dim)。
        """
        if not texts or not images or len(texts) != len(images):
            raise ValueError(
                "'texts' and 'images' must be non-empty lists of the same length."
            )

        with self._use_runtime() as runtime:
            torch_runtime = runtime.torch
            all_embeddings = []
            with torch_runtime.no_grad():
                for i in tqdm(
                    range(0, len(texts), batch_size), desc="Embedding Fused"
                ):
                    batch_texts = texts[i : i + batch_size]
                    batch_images = images[i : i + batch_size]

                    embeddings_tensor = runtime.model.get_fused_embeddings(
                        texts=batch_texts, images=batch_images
                    )
                    all_embeddings.append(embeddings_tensor.cpu())

            full_embeddings_tensor = torch_runtime.cat(all_embeddings, dim=0)

            return self._normalize_with_runtime(runtime, full_embeddings_tensor)

    @classmethod
    def _release_runtime(
        cls,
        key: tuple[str, str],
        runtime: _GmeSharedRuntime,
    ) -> None:
        with cls._pool_lock:
            pooled = cls._runtime_pool.get(key)
            if pooled is not runtime:
                raise RuntimeError("GME runtime ownership does not match its pool key")
            runtime.owner_count -= 1
            if runtime.owner_count < 0:
                runtime.owner_count = 0
                raise RuntimeError("GME runtime ownership released twice")
            if runtime.owner_count:
                return

            # Keep the key reserved while waiting so a new owner cannot race
            # with teardown and temporarily allocate duplicate GPU weights.
            runtime.wait_until_idle()
            try:
                runtime.release_weights()
            finally:
                cls._runtime_pool.pop(key, None)

    def _close_legacy_unpooled_instance(self) -> None:
        """Close old manually-constructed doubles without reviving singleton use."""

        if not getattr(self, "_initialized", False):
            return
        torch_runtime = getattr(self, "_torch", None)
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch_runtime is not None and torch_runtime.cuda.is_available():
            torch_runtime.cuda.empty_cache()
        self._torch = None
        self._torch_functional = None
        self._initialized = False
        if type(self)._instance is self:
            type(self)._instance = None

    def close(self) -> None:
        """Release this handle and unload weights only after the last owner."""

        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            self._close_legacy_unpooled_instance()
            return

        with state_lock:
            if self._closed:
                return
            self._closed = True
            self._closed_event.set()
            runtime = self._runtime
            self._runtime = None
            key = self._pool_key
            if hasattr(self, "model"):
                del self.model
            self._torch = None
            self._torch_functional = None
            self._initialized = False

        if runtime is not None:
            runtime.notify_waiters()
            self._release_runtime(key, runtime)
        log.info("GmeEmbeddingProvider handle closed.")

    def clear_cache(self) -> None:
        """
        手动清理PyTorch的CUDA缓存和Python的垃圾回收。
        注意：这是一个同步操作，可能会影响性能，请仅在必要时（如处理完一个大任务后）调用。
        """
        log.info("Clearing CUDA cache...")
        with self._use_runtime() as runtime:
            torch_runtime = runtime.torch
            if torch_runtime is not None and torch_runtime.cuda.is_available():
                torch_runtime.cuda.empty_cache()
        gc.collect()
        log.info("CUDA cache cleared.")

    def rerank_documents(
        self,
        query: str,
        doc_list: List[Dict[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Rerank a list of documents based on a query and an optional instruction.
        """
        if not query:
            raise ValueError("Query cannot be empty.")
        if not doc_list:
            return []

        log.info(
            f"Reranking {len(doc_list)} documents based on query and instruction..."
        )

        query_embedding = self.embed_texts([query], instruction=instruction)[0]

        # 2. 按类型对文档进行分组，同时记录原始索引
        text_only_group = []
        image_only_group = []
        fused_group = []

        for i, doc in enumerate(doc_list):
            text = doc.get("content")
            image = doc.get("img_path")

            if text and image:
                fused_group.append({"index": i, "text": text, "image": image})
            elif text:
                text_only_group.append({"index": i, "text": text})
            elif image:
                image_only_group.append({"index": i, "image": image})

        # 初始化最终分数列表
        scores = [0.0] * len(doc_list)

        # 3. 对每个分组进行批量嵌入和相似度计算
        if text_only_group:
            log.info(f"Processing {len(text_only_group)} text-only documents...")
            texts = [item["text"] for item in text_only_group]
            doc_embeddings = self.embed_texts(texts)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(text_only_group):
                scores[item["index"]] = similarities[i]

        if image_only_group:
            log.info(f"Processing {len(image_only_group)} image-only documents...")
            images = [item["image"] for item in image_only_group]
            doc_embeddings = self.embed_images(images)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(image_only_group):
                scores[item["index"]] = similarities[i]

        if fused_group:
            log.info(f"Processing {len(fused_group)} fused (text+image) documents...")
            texts = [item["text"] for item in fused_group]
            images = [item["image"] for item in fused_group]
            doc_embeddings = self.embed_fused(texts=texts, images=images)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(fused_group):
                scores[item["index"]] = similarities[i]

        log.info("Reranking complete.")
        return scores


class MMRerankerProvider(BaseEmbedder):
    MM_EMBEDDER: bool = True

    def __init__(
        self,
        index_path: str,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "auto",
        *,
        gme_provider: Optional[GmeEmbeddingProvider] = None,
        owns_gme_provider: Optional[bool] = None,
    ):
        if owns_gme_provider is not None and not isinstance(
            owns_gme_provider, bool
        ):
            raise ValueError("owns_gme_provider must be a boolean when provided")
        if gme_provider is None and owns_gme_provider is False:
            raise ValueError(
                "an internally-created GME provider must be owned by the reranker"
            )

        self.index_path = index_path
        self.doc_metadata = {}
        self.doc_embeddings = np.array([])
        self.question_embeddings = {}
        self.gme_provider = gme_provider
        # Injected resources are borrowed unless ownership is explicitly
        # transferred. Internally-created provider handles are always owned.
        self._owns_gme_provider = (
            True if gme_provider is None else bool(owns_gme_provider)
        )
        self._closed = False

        log.info(f"Initializing MMRerankerProvider from path: {self.index_path}")
        try:
            self._load_embeddings()
            if self.gme_provider is None:
                self.gme_provider = GmeEmbeddingProvider(
                    model_name=model_name,
                    device=device,
                )
        except BaseException:
            provider = self.gme_provider
            self.gme_provider = None
            self.doc_metadata.clear()
            self.doc_embeddings = np.array([])
            self.question_embeddings.clear()
            self._closed = True
            if self._owns_gme_provider and provider is not None:
                try:
                    provider.close()
                except Exception:
                    log.exception(
                        "Failed to close owned GME provider after reranker startup failure"
                    )
            raise

    def _load_embeddings(self):
        """
        Loads metadata and embeddings for documents and questions from disk.
        """
        # --- Load Document Embeddings ---
        doc_metadata_path = os.path.join(self.index_path, "mm_node_metadata.json")
        doc_embeddings_path = os.path.join(self.index_path, "mm_embeddings.npy")

        if os.path.exists(doc_metadata_path) and os.path.exists(doc_embeddings_path):
            log.info("Loading document metadata and embeddings...")
            with open(doc_metadata_path, "r", encoding="utf-8") as f:
                doc_metadata_list = json.load(f)

            # Create a dictionary for fast O(1) lookup by node_id
            self.doc_metadata = {item["node_id"]: item for item in doc_metadata_list}

            self.doc_embeddings = np.load(doc_embeddings_path)
            log.info(
                f"Loaded {len(self.doc_metadata)} document metadata entries and {self.doc_embeddings.shape[0]} document embeddings."
            )
        else:
            log.warning("Document metadata or embedding files not found")

        # --- Load Question Embeddings ---
        q_metadata_path = os.path.join(self.index_path, "mm_question_metadata.json")
        q_embeddings_path = os.path.join(self.index_path, "mm_question_embeddings.npy")

        if os.path.exists(q_metadata_path) and os.path.exists(q_embeddings_path):
            log.info("Loading question metadata and embeddings...")
            with open(q_metadata_path, "r", encoding="utf-8") as f:
                q_metadata = json.load(f)
            q_embeddings = np.load(q_embeddings_path)

            # Create a dictionary mapping question text to its embedding for fast lookup
            for row in q_metadata:
                question_text = row["question"]
                embedding_idx = row["question_embedding_idx"]
                self.question_embeddings[question_text] = q_embeddings[embedding_idx]
                # The query key must include the instruction, as it was used during pre-computation
                # query_key = question_text + self.RERANKER_INSTRUCTION
                # self.question_embeddings[query_key] = q_embeddings[embedding_idx]

            log.info(
                f"Loaded {len(self.question_embeddings)} pre-computed question embeddings."
            )
        else:
            log.warning("Question metadata or embedding files not found.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("MMRerankerProvider does not support text embedding.")

    def embed_images(self, images: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "MMRerankerProvider does not support image embedding."
        )

    def embed_fused(self, images: List[str], texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "MMRerankerProvider does not support fused embedding."
        )

    def rerank_documents(
        self,
        query: str,
        doc_list: List[Dict[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        if self._closed or self.gme_provider is None:
            raise RuntimeError("MM reranker provider is closed")
        if not query:
            raise ValueError("Query cannot be empty.")
        if not doc_list:
            return []

        # 1. Get the pre-computed query embedding
        if query in self.question_embeddings.keys():
            query_embedding = self.question_embeddings[query]
        else:
            query_embedding = self.gme_provider.embed_texts(
                [query], instruction=instruction
            )[0]

        if query_embedding is None:
            log.error(
                f"Query '{query}' not found in pre-computed question embeddings. Returning zero scores."
            )
            return [0.0] * len(doc_list)

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        if self.doc_embeddings.size == 0:
            embedding_dim = query_embedding.shape[
                0
            ]  # Get dimension from query embedding
        else:
            embedding_dim = self.doc_embeddings.shape[1]

        # 2. Prepare document embeddings for batch calculation
        # Create a matrix to hold all document embeddings. Initialize with zeros.
        doc_embeddings_matrix = np.zeros(
            (len(doc_list), embedding_dim), dtype=query_embedding.dtype
        )

        # Collect valid embedding indices and their corresponding positions in doc_list
        valid_doc_indices = []
        valid_embedding_indices = []

        for i, doc in enumerate(doc_list):
            node_id = doc.get("node_id")
            metadata = self.doc_metadata.get(node_id)

            if metadata and metadata.get("embedding_idx") is not None:
                # skip documents without valid embedding
                embedding_idx_in_storage = metadata["embedding_idx"]
                valid_doc_indices.append(i)  # Original position in doc_list
                valid_embedding_indices.append(embedding_idx_in_storage)

        if valid_doc_indices:
            doc_embeddings_matrix[valid_doc_indices] = self.doc_embeddings[
                valid_embedding_indices
            ]

        # 3. Perform batch similarity calculation
        similarities = (doc_embeddings_matrix @ query_embedding).tolist()

        log.info(f"Successfully reranked {len(doc_list)} documents in batch mode.")
        return similarities

    def close(self) -> None:
        """Release in-memory data and this reranker's owned GME handle once."""

        if self._closed:
            return
        self._closed = True
        self.doc_metadata.clear()
        self.doc_embeddings = np.array([])
        self.question_embeddings.clear()
        provider = self.gme_provider
        self.gme_provider = None
        if self._owns_gme_provider and provider is not None:
            provider.close()


class TextEmbeddingProvider(BaseEmbedder):

    MM_EMBEDDER: bool = False
    _instances: "weakref.WeakSet[TextEmbeddingProvider]" = weakref.WeakSet()

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        backend: str = "local",
        device: Optional[str] = None,
        max_length: int = 8192,
        api_base: str = None,
        api_key: Optional[str] = None,
        num_ctx: Optional[int] = None,
        *,
        client: Optional[object] = None,
        owns_client: Optional[bool] = None,
    ):

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(backend, str):
            raise ValueError("backend must be 'local', 'ollama', or 'openai'")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be at least 1")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        api_base = _normalize_api_base(api_base)
        if owns_client is not None and not isinstance(owns_client, bool):
            raise ValueError("owns_client must be a boolean when provided")

        self._closed = False
        self.model_name = model_name.strip()
        self.backend = backend.strip().lower()
        self.max_length = max_length
        self.num_ctx = num_ctx
        self._torch = None
        self._torch_functional = None

        if self.backend == "local":
            if api_key or api_base:
                raise ValueError(
                    "api_key and api_base are not supported by the local backend"
                )
            if client is not None or owns_client is not None:
                raise ValueError("client settings are only supported by remote backends")
            device = device or "auto"
            if not isinstance(device, str) or not device.strip():
                raise ValueError("device must be a non-empty string")

            try:
                import torch
                from torch.nn import functional as torch_functional
                from modelscope import AutoModel, AutoTokenizer

                self._torch = torch
                self._torch_functional = torch_functional
                if device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device

                log.info(
                    "Using local backend (ModelScope) on device: %s", self.device
                )
                log.info("Loading model: %s...", self.model_name)

                # 根据官方示例，使用AutoTokenizer和AutoModel，并设置padding_side
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, padding_side="left", trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self.device)
                self.model.eval()
            except Exception:
                log.exception(
                    "Error loading local embedding model '%s'.", self.model_name
                )
                torch_runtime = self._torch
                if hasattr(self, "model"):
                    del self.model
                if hasattr(self, "tokenizer"):
                    del self.tokenizer
                gc.collect()
                if (
                    torch_runtime is not None
                    and "cuda" in str(getattr(self, "device", ""))
                    and torch_runtime.cuda.is_available()
                ):
                    torch_runtime.cuda.empty_cache()
                self._torch = None
                self._torch_functional = None
                raise

            log.info("Local model loaded successfully.")

        elif self.backend == "ollama":
            if device is not None:
                raise ValueError("device is only supported by the local backend")
            self.device = "ollama_service"
            if client is None:
                import ollama

                parameters: Dict[str, Any] = {}
                if api_base:
                    parameters["host"] = api_base
                if api_key:
                    parameters["headers"] = {
                        "Authorization": f"Bearer {api_key}"
                    }
                self.client = ollama.Client(**parameters)
                self._owns_client = True if owns_client is None else owns_client
            else:
                self.client = client
                self._owns_client = False if owns_client is None else owns_client
        elif self.backend == "openai":
            if device is not None:
                raise ValueError("device is only supported by the local backend")
            if client is None:
                from openai import OpenAI

                parameters: Dict[str, Any] = {}
                if api_key is not None:
                    parameters["api_key"] = api_key
                if api_base is not None:
                    parameters["base_url"] = api_base
                self.client = OpenAI(**parameters)
                self._owns_client = True if owns_client is None else owns_client
            else:
                self.client = client
                self._owns_client = False if owns_client is None else owns_client
        else:
            raise ValueError(
                f"Unsupported backend: '{self.backend}'. Choose 'local', 'ollama', or 'openai'."
            )
        self._instances.add(self)

    @classmethod
    def close_instance(cls) -> None:
        """Backward-compatible shutdown of every live provider instance."""

        for instance in list(cls._instances):
            instance.close()

    def close(self) -> None:
        """Release local weights or remote client resources idempotently."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        backend = getattr(self, "backend", None)
        if backend == "local":
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
        elif backend in {"openai", "ollama"}:
            client = getattr(self, "client", None)
            if (
                client is not None
                and getattr(self, "_owns_client", True)
                and hasattr(client, "close")
            ):
                client.close()
            self.client = None

        torch_runtime = getattr(self, "_torch", None)
        self._torch = None
        self._torch_functional = None
        gc.collect()
        device = str(getattr(self, "device", ""))
        if (
            backend == "local"
            and "cuda" in device
            and torch_runtime is not None
            and torch_runtime.cuda.is_available()
        ):
            torch_runtime.cuda.empty_cache()
        log.info("TextEmbeddingProvider closed for model %s.", self.model_name)

    def _last_token_pool(
        self, last_hidden_states: Any, attention_mask: Any
    ) -> Any:
        torch_runtime = self._torch
        if torch_runtime is None:
            raise RuntimeError("Local embedding provider is closed")
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch_runtime.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _normalize(self, embeddings: Any) -> np.ndarray:
        """
        对PyTorch张量进行L2归一化，并转换为numpy数组。
        """
        if self._torch_functional is None:
            raise RuntimeError("Local embedding provider is closed")
        # 使用torch.nn.functional.normalize进行归一化
        normalized_embeddings = self._torch_functional.normalize(
            embeddings, p=2, dim=1
        )
        # 移动到CPU并转换为numpy数组
        return normalized_embeddings.detach().float().cpu().numpy()

    @staticmethod
    def _normalize_numpy(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize remote embeddings without importing PyTorch."""

        values = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(values, ord=2, axis=1, keepdims=True)
        return values / np.maximum(norms, np.float32(1e-12))

    def _embed_options(self) -> dict:
        if self.num_ctx is None:
            return {}
        return {"options": {"num_ctx": self.num_ctx}}

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        为一批文本生成嵌入向量。

        Args:
            texts (List[str]): 需要编码的文本列表。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_texts, embedding_dim)。
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        if self.backend == "local":
            torch_runtime = self._torch
            if torch_runtime is None:
                raise RuntimeError("Local embedding provider is closed")
            batch_size = 16
            all_embeddings = []
            n = len(texts)
            if n <= batch_size:
                batch_dict = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict.to(self.device)
                with torch_runtime.no_grad():
                    outputs = self.model(**batch_dict)
                embeddings_tensor = self._last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch_runtime.cat(all_embeddings, dim=0)
            else:
                num_batches = math.ceil(n / batch_size)
                for i in tqdm(
                    range(0, n, batch_size),
                    desc="Embedding batches",
                    total=num_batches,
                ):
                    batch_texts = texts[i : i + batch_size]
                    batch_dict = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch_dict.to(self.device)
                    with torch_runtime.no_grad():
                        outputs = self.model(**batch_dict)
                    embeddings_tensor = self._last_token_pool(
                        outputs.last_hidden_state, batch_dict["attention_mask"]
                    )
                    all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch_runtime.cat(all_embeddings, dim=0)

        elif self.backend == "ollama":
            # Always use the configured client. The module-level Ollama helper
            # silently targets its default localhost and used to ignore
            # ``EmbeddingConfig.api_base`` entirely.
            embed = getattr(self.client, "embed", None)
            if callable(embed):
                response = embed(
                    model=self.model_name, input=texts, **self._embed_options()
                )
                if isinstance(response, dict):
                    all_embeddings = response.get("embeddings")
                else:
                    all_embeddings = getattr(response, "embeddings", None)
                if not isinstance(all_embeddings, list) or len(all_embeddings) != len(
                    texts
                ):
                    raise ValueError(
                        "Ollama embedding response did not match the input batch"
                    )
            else:
                # Compatibility with Ollama SDKs predating ``Client.embed``.
                legacy_embed = getattr(self.client, "embeddings", None)
                if not callable(legacy_embed):
                    raise RuntimeError("Ollama client does not provide an embedding API")
                all_embeddings = []
                for text in texts:
                    response = legacy_embed(model=self.model_name, prompt=text)
                    embedding = (
                        response.get("embedding")
                        if isinstance(response, dict)
                        else getattr(response, "embedding", None)
                    )
                    if not isinstance(embedding, list):
                        raise ValueError("Ollama returned an invalid embedding response")
                    all_embeddings.append(embedding)
            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            return self._normalize_numpy(embeddings_np)

        elif self.backend == "openai":
            BATCH_SIZE = 8
            n = len(texts)
            all_embeddings = []
            num_batches = math.ceil(n / BATCH_SIZE)
            for i in tqdm(
                range(0, n, BATCH_SIZE), desc="Embedding texts", total=num_batches
            ):
                chunk = texts[i : i + BATCH_SIZE]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk,
                )
                embeddings_from_this_batch = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings_from_this_batch)

            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            return self._normalize_numpy(embeddings_np)

        # Local outputs retain the original torch normalization path.
        return self._normalize(embeddings_tensor)

    def compute_texts_sim(self, text1: str, text2: str) -> float:
        """
        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.
        Returns:
            float: Cosine similarity between the two text embeddings, in the range [-1.0, 1.0].
        """
        embeddings = self.embed_texts([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]
        similarity = vec1 @ vec2

        return float(similarity)


if __name__ == "__main__":
    # Example usage
    text_embedder = TextEmbeddingProvider(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        # model_name="Qwen/Qwen3-Embedding-4B",
        backend="local",
        device="cuda:6",
    )
    texts = ["你好，世界！", "这是一个测试。"]
    embeddings = text_embedder.embed_texts(texts)
    print("Text embeddings shape:", embeddings.shape)
    # test the cuda storage

    mm_embedder = GmeEmbeddingProvider(
        model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device="cuda:6",
    )
    texts = ["你好，世界！", "这是一个测试。"]
    images = [
        "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/03fc225967e86ac2ca3f1dc1afe2f2657bf2298c14ad952bbf6f931574888e3c.jpg",
        "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/76f83ac5b4a7fd38c31a353466fec43927b5fed31f433753e8ece01795097112.jpg",
    ]
    text_embeddings = mm_embedder.embed_texts(texts)
    image_embeddings = mm_embedder.embed_images(images)
    print("Text embeddings shape:", text_embeddings.shape)
    print("Image embeddings shape:", image_embeddings.shape)
    fused_embeddings = mm_embedder.embed_fused(images=images, texts=texts)
    print("Fused embeddings shape:", fused_embeddings.shape)

    from Core.prompts.gbc_prompt import MM_RERANKER_INSTRUCTION

    docs = [
        {
            "content": "Table 1: The inference seconds over 500 sentences (run on single V100 GPU). Here LLaMA is extremely slow since we set batch size as 1 due to memory limit. <table><tr><td>Dataset (Task)</td><td>Roberta</td><td>T5</td><td>LLaMA</td><td>CODEX</td></tr><tr><td>FewNERD (NER)</td><td>2.8</td><td>39.4</td><td>1135.4</td><td>179.4</td></tr><tr><td>TACREV (RE)</td><td>1.4</td><td>45.6</td><td>1144.9</td><td>151.6</td></tr><tr><td>ACE05 (ED)</td><td>6.6</td><td>62.5</td><td>733.4</td><td>171.7</td></tr></table>",
            "img_path": "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/76f83ac5b4a7fd38c31a353466fec43927b5fed31f433753e8ece01795097112.jpg",
        },
        {
            "content": "Table 2: Comparative ratios of negative to positive samples across various datasets and subsets. We set fixed threshold  $\\tau$  here for simplicity.",
            "img_path": "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/34eaca8085bae127fb28d014cc4951fec10866cb37e58c65f4906989a5a1806a.jpg",
        },
        {
            "content": "The study highlights the significant impact of climate change on global food security, emphasizing rising temperatures, shifting precipitation patterns, and increased frequency of extreme weather events as major threats to agricultural productivity. It also underscores the vulnerability of smallholder farmers and the urgent need for adaptive strategies, including sustainable farming practices, improved water management, and investment in resilient crop varieties. The findings call for immediate and coordinated global action to mitigate risks and ensure long-term food stability."
        },
    ]
    query = "What is the inference time of LLaMA on FewNERD dataset?"
    rerank_scores = mm_embedder.rerank_documents(
        query=query,
        doc_list=docs,
        instruction=MM_RERANKER_INSTRUCTION,
    )
    print("Rerank scores:", rerank_scores)

    # clip_embedder = ChineseClipModel(model_name="ViT-B-16")
    # image_embeddings = clip_embedder.embed_images(["path/to/image1.jpg", "path/to/image2.jpg"])
    # print("Image embeddings shape:", image_embeddings.shape)
