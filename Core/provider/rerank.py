from __future__ import annotations

from Core.configs.llm_config import _normalize_api_base
from typing import TYPE_CHECKING, Any, List, Dict, Optional
import importlib
import math
from tqdm import tqdm
import logging
import gc

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from Core.Index.Graph import Entity


class TextRerankerProvider:
    """
    一个用于文本重排（Reranking）的封装类，基于Qwen3-Reranker模型。

    该类封装了模型加载、输入格式化、批处理和分数计算的逻辑，
    提供一个简单易用的接口来计算查询与一组文档之间的相关性得分。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: Optional[str] = None,
        max_length: int = 4096,
        use_flash_attention: Optional[bool] = None,
        torch_dtype: Any = None,
        backend: str = "local",
        api_base: str = None,
        request_timeout_seconds: Optional[float] = None,
        flash_attention_fallback: Optional[bool] = None,
        *,
        session: Optional[object] = None,
        owns_session: Optional[bool] = None,
    ):
        """
        初始化Reranker Provider。

        Args:
            model_name (str): 要加载的ModelScope或Hugging Face模型名称。
            device (str): 运行模型的设备 ('auto', 'cuda', 'cpu')。
            max_length (int): 模型的最大序列长度。
            use_flash_attention (bool): 是否尝试使用Flash Attention 2以提升性能。
            torch_dtype (torch.dtype): 模型加载时使用的数据类型，如 torch.bfloat16。
            backend (str): 后端类型，支持 'local' 和 'vllm'。
            api_base (str): 如果使用 'vllm' 后端，必须提供API基础URL。
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(backend, str):
            raise ValueError("backend must be 'local' or 'vllm'")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be at least 1")
        if owns_session is not None and not isinstance(owns_session, bool):
            raise ValueError("owns_session must be a boolean when provided")
        self.model_name = model_name.strip()
        self.max_length = max_length
        self.backend = backend.strip().lower()
        self._closed = False

        # ==========================================================
        # vLLM 后端逻辑
        # ==========================================================
        if self.backend == "vllm":
            if device is not None:
                raise ValueError("device is only supported by the local reranker backend")
            if use_flash_attention is not None:
                raise ValueError(
                    "use_flash_attention is only supported by the local reranker backend"
                )
            if flash_attention_fallback is not None:
                raise ValueError(
                    "flash_attention_fallback is only supported by the local reranker backend"
                )
            if torch_dtype is not None:
                raise ValueError("torch_dtype is only supported by the local backend")
            if request_timeout_seconds is None:
                request_timeout_seconds = 30.0
            if (
                isinstance(request_timeout_seconds, bool)
                or not isinstance(request_timeout_seconds, (int, float))
                or not math.isfinite(request_timeout_seconds)
                or request_timeout_seconds <= 0
            ):
                raise ValueError(
                    "request_timeout_seconds must be finite and greater than 0"
                )
            self.request_timeout_seconds = float(request_timeout_seconds)
            if not isinstance(api_base, str) or not api_base.strip():
                raise ValueError("api_base must be provided for the 'vllm' backend.")
            normalized_api_base = _normalize_api_base(api_base)
            if normalized_api_base.endswith("/rerank"):
                self.rerank_url = normalized_api_base
            else:
                self.rerank_url = f"{normalized_api_base}/rerank"
            if session is None:
                import requests

                self.session = requests.Session()
                self._owns_session = True if owns_session is None else owns_session
            else:
                self.session = session
                self._owns_session = False if owns_session is None else owns_session
            # Do not log the endpoint: deployment URLs can contain credentials.
            log.info("Using remote reranker backend.")
        # ==========================================================
        # 本地后端逻辑 (将原有代码移入此分支)
        # ==========================================================
        elif self.backend == "local":
            if session is not None or owns_session is not None:
                raise ValueError("session settings are only supported by the vllm backend")
            if api_base is not None and str(api_base).strip():
                raise ValueError("api_base is only supported by the vllm backend")
            if request_timeout_seconds is not None:
                raise ValueError(
                    "request_timeout_seconds is only supported by the vllm backend"
                )
            device = device or "auto"
            if not isinstance(device, str) or not device.strip():
                raise ValueError("device must be a non-empty string")
            import torch

            self._torch = torch
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            if use_flash_attention is None:
                use_flash_attention = "cuda" in self.device
            if not isinstance(use_flash_attention, bool):
                raise ValueError("use_flash_attention must be a boolean")
            if use_flash_attention:
                if "cuda" not in self.device:
                    raise ValueError("use_flash_attention requires a CUDA device")
                flash_attention_fallback = (
                    True
                    if flash_attention_fallback is None
                    else flash_attention_fallback
                )
            elif flash_attention_fallback is not None:
                raise ValueError(
                    "flash_attention_fallback requires use_flash_attention=true"
                )
            if flash_attention_fallback is not None and not isinstance(
                flash_attention_fallback, bool
            ):
                raise ValueError("flash_attention_fallback must be a boolean")

            log.info("Using local reranker on device %s", self.device)

            model_args = {"trust_remote_code": True}
            if torch_dtype is not None:
                model_args["torch_dtype"] = torch_dtype
            if "cuda" in self.device:
                model_args.setdefault("torch_dtype", torch.bfloat16)
                if use_flash_attention:
                    try:
                        # Importing catches missing wheels and binary/ABI failures
                        # before any tokenizer or model weights are allocated.
                        importlib.import_module("flash_attn")
                    except Exception as error:
                        if not flash_attention_fallback:
                            raise RuntimeError(
                                "Flash Attention 2 was requested but is unavailable"
                            ) from error
                        log.warning(
                            "Flash Attention 2 unavailable; using default attention (%s)",
                            type(error).__name__,
                        )
                    else:
                        model_args["attn_implementation"] = "flash_attention_2"
                        log.info("Using Flash Attention 2.")

            from modelscope import AutoModelForCausalLM, AutoTokenizer

            # 加载分词器，Qwen Reranker要求左填充
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )

            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name, **model_args)
                .to(self.device)
                .eval()
            )

            log.info("Reranker model loaded successfully.")

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Choose 'local' or 'vllm'."
            )
        self._define_prompt_template()

    def clean_cache(self):
        if self.backend == "local":
            log.info("Cleaning cache...")
            gc.collect()
            if "cuda" in self.device:
                self._torch.cuda.empty_cache()
            log.info("Cache cleaned.")
        else:
            log.info(f"{self.backend} backend requires no local cache cleaning.")

    def close(self) -> None:
        """
        close the reranker provider and release resources.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        log.info("Closing TextRerankerProvider.")

        if self.backend == "local":
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer

            if "cuda" in self.device:
                log.info("Reranker: Emptying CUDA cache.")
                self._torch.cuda.empty_cache()

            gc.collect()
            log.info("Local reranker resources released.")

        elif self.backend == "vllm":
            if (
                getattr(self, "session", None) is not None
                and getattr(self, "_owns_session", True)
            ):
                self.session.close()
            self.session = None
            log.info("vLLM backend session closed.")

        log.info("TextRerankerProvider closed.")

    def _define_prompt_template(self):
        """
        定义Qwen3-Reranker所需的特定提示词模板。
        """
        # --- 步骤1: 定义所有后端都需要的字符串模板 ---
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        # --- 步骤2: 仅为 local 后端预编码 token ---
        if self.backend == "local":
            # 获取"yes"和"no"的token ID
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            # 预先编码模板，以备后续拼接
            self.prefix_tokens = self.tokenizer.encode(
                self.prefix, add_special_tokens=False
            )
            self.suffix_tokens = self.tokenizer.encode(
                self.suffix, add_special_tokens=False
            )

    def _format_instruction(
        self, query: str, doc: str, instruction: Optional[str]
    ) -> str:
        """
        根据指令、查询和文档格式化单个输入对。
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_batch(self, batch_pairs: List[str]) -> Dict[str, Any]:
        """
        对一批格式化后的文本对进行分词、添加特殊token和填充。
        """
        # 1. 对核心内容进行分词
        inputs = self.tokenizer(
            batch_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )

        # 2. 手动添加前缀和后缀token
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = (
                self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
            )

        # 3. 进行填充
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            # max_length=self.max_length
        )

        # 4. 移动到设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    def _compute_scores_from_inputs(
        self, inputs: Dict[str, Any]
    ) -> List[float]:
        """
        从处理好的输入中计算相关性得分。
        """
        # 模型前向传播，获取最后一个token的logits
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        # 提取"yes"和"no"的logits
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]

        # 计算Log-Softmax并转换为概率
        scores_tensor = self._torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = self._torch.nn.functional.log_softmax(
            scores_tensor, dim=1
        )

        # 返回"yes"的概率作为最终得分
        final_scores = log_softmax_scores[:, 1].exp().tolist()

        # 【关键修改】显式删除大的中间张量
        del (
            outputs,
            logits,
            true_vector,
            false_vector,
            scores_tensor,
            log_softmax_scores,
        )

        return final_scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 4,
    ) -> List[float]:
        """
        计算单个查询与多个文档之间的相关性得分。

        Args:
            query (str): 单个查询文本。
            documents (List[str]): 需要与查询进行比较的文档列表。
            instruction (Optional[str]): 任务指令，如果为None则使用默认值。
            batch_size (int): 处理时的批处理大小。

        Returns:
            List[float]: 返回一个与文档列表对应的得分列表，每个分数在0到1之间。
        """
        if not documents or not isinstance(documents, list):
            raise ValueError("Input 'documents' must be a non-empty list of strings.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("documents must contain only strings")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be at least 1")

        if self.backend == "vllm":

            if instruction is None:
                # 使用默认的 instruction
                instruction = "Given a web search query, retrieve relevant passages that answer the query"

            # 格式化查询 (注意 API 可能需要列表，所以我们把单个查询放入列表)
            formatted_query = self.query_template.format(
                prefix=self.prefix, instruction=instruction, query=query
            )

            # 1. 构建 payload，使用原始文本，无需客户端模板
            all_formatted_documents = [
                self.document_template.format(doc=doc, suffix=self.suffix)
                for doc in documents
            ]

            
            all_scores: List[float] = []
            num_docs = len(documents)
            num_batches = math.ceil(num_docs / batch_size)

            for i in tqdm(
                range(0, num_docs, batch_size),
                desc="Reranking Batches (vLLM)",
                total=num_batches,
                disable=num_docs < batch_size,
            ):
                batch_formatted_documents = all_formatted_documents[
                    i : i + batch_size
                ]
                payload = {
                    "model": self.model_name,
                    "query": formatted_query,
                    "documents": batch_formatted_documents,
                }
                response = self.session.post(
                    self.rerank_url,
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()

                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Remote reranker returned a non-object response")
                results = data.get("results")
                if not isinstance(results, list):
                    raise ValueError("Remote reranker response has no results list")
                expected_indices = set(range(len(batch_formatted_documents)))
                by_index: Dict[int, float] = {}
                for result in results:
                    if not isinstance(result, dict):
                        raise ValueError("Remote reranker returned an invalid result")
                    index = result.get("index")
                    score = result.get("relevance_score")
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index not in expected_indices
                        or index in by_index
                    ):
                        raise ValueError("Remote reranker returned invalid indices")
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(score)
                    ):
                        raise ValueError("Remote reranker returned an invalid score")
                    by_index[index] = float(score)
                if set(by_index) != expected_indices:
                    raise ValueError("Remote reranker returned incomplete results")
                all_scores.extend(by_index[index] for index in sorted(by_index))

            return all_scores


        elif self.backend == "local":
            # 1. 创建所有的查询-文档对
            pairs = [
                self._format_instruction(query, doc, instruction) for doc in documents
            ]

            # 2. 分批处理并计算得分
            all_scores = []
            num_docs = len(documents)
            num_batches = math.ceil(num_docs / batch_size)

            # 【关键修改】初始化计数器和阈值
            processed_content_count = 0
            # 当累计处理的字符数超过此阈值时，清理缓存
            CACHE_CLEAN_THRESHOLD = 4000 * 5  # 假设平均一个词约5个字符，可调整

            for i in tqdm(
                range(0, num_docs, batch_size),
                desc="Reranking Batches",
                total=num_batches,
                disable=num_docs < batch_size,
            ):
                batch_pairs = pairs[i : i + batch_size]

                # 处理当前批次
                inputs = self._process_batch(batch_pairs)

                # 计算并收集得分
                scores = self._compute_scores_from_inputs(inputs)
                all_scores.extend(scores)

                # 【关键修改】累加处理过的文本量
                # 使用sum(len(p) for p in batch_pairs)来估算字符数，计算非常快
                processed_content_count += sum(len(p) for p in batch_pairs)

                # 【关键修改】检查是否达到清理阈值
                if processed_content_count >= CACHE_CLEAN_THRESHOLD:
                    del inputs, batch_pairs, scores

                    log.info(
                        f"Processed content count ({processed_content_count}) reached threshold. Cleaning cache."
                    )
                    self.clean_cache()
                    # 重置计数器
                    processed_content_count = 0
                    continue

                del inputs, batch_pairs, scores

            return all_scores


# For test


def format_entity_for_reranking(entity: Entity) -> str:
    """Formats the first three fields of an Entity object into a single,
    English-keyed string for scoring."""
    return f"Name: {entity.entity_name}\nType: {entity.entity_type}\nDescription: {entity.description}"


def test_example_1(reranker: TextRerankerProvider, instruction: str):
    from Core.Index.Graph import Entity

    query_entity = Entity(
        entity_name="Michael Jordan",
        entity_type="Person",
        description="A famous basketball player, widely regarded as the greatest of all time.",
    )

    # 4. 定义10个从VDB中召回的“候选实体”，这是一个混合了正确、模糊和错误结果的列表
    candidate_entities = [
        # 正确匹配
        Entity(
            entity_name="Michael Jordan",
            entity_type="Basketball Player",
            description="Michael Jeffrey Jordan, also known by his initials MJ, is an American businessman and former professional basketball player for the Chicago Bulls.",
        ),
        # 名称相同，但实体不同 (关键的模糊项)
        Entity(
            entity_name="Michael B. Jordan",
            entity_type="Actor",
            description="An American actor known for his roles in films like Black Panther and Creed.",
        ),
        Entity(
            entity_name="Michael I. Jordan",
            entity_type="Professor",
            description="A renowned researcher in machine learning and artificial intelligence at UC Berkeley.",
        ),
        # 主题相关，但实体不同
        Entity(
            entity_name="Scottie Pippen",
            entity_type="Basketball Player",
            description="A teammate of Michael Jordan on the Chicago Bulls during their championship runs.",
        ),
        Entity(
            entity_name="Chicago Bulls",
            entity_type="Sports Team",
            description="The NBA team where Michael Jordan played most of his career and won six championships.",
        ),
        # 高度相关品牌，但非本人
        Entity(
            entity_name="Air Jordan",
            entity_type="Brand",
            description="A brand of athletic footwear and apparel produced by Nike, Inc., created for Michael Jordan.",
        ),
        # 另一个正确的匹配，但描述略有不同
        Entity(
            entity_name="Michael Jordan",
            entity_type="Athlete",
            description="Played 15 seasons in the NBA, winning six championships with the Chicago Bulls. His biography is 'The Last Dance'.",
        ),
        # 名称部分匹配
        Entity(
            entity_name="Jordan Peele",
            entity_type="Director",
            description="An American actor, comedian, and filmmaker, known for directing horror films like 'Get Out'.",
        ),
        # 完全不相关
        Entity(
            entity_name="Eiffel Tower",
            entity_type="Landmark",
            description="A wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        ),
        # 另一个著名但无关的Michael
        Entity(
            entity_name="Michael Jackson",
            entity_type="Musician",
            description="Known as the 'King of Pop', one of the most significant cultural figures of the 20th century.",
        ),
    ]

    # 5. 格式化输入以调用rerank方法
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]

    # 6. 调用rerank方法
    scores = reranker.rerank(
        query=query_string,
        documents=document_strings,
        instruction=instruction,
    )

    # 7. 将实体对象和得分结合并排序
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    # 8. 打印结果
    print("\n--- Entity Reranking Results ---")
    print(f"Query Entity: {query_entity.entity_name} ({query_entity.description})\n")
    print("Ranked candidate entities from VDB:")
    for entity, score in ranked_results:
        # 清晰地打印出实体名称和类型，以便于观察排序效果
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )

    pass


# --- 测试用例 2: 不存在正确匹配的实体 ---
def test_example_2(reranker: "TextRerankerProvider", instruction: str):
    from Core.Index.Graph import Entity

    """
    运行实体解析的第二个测试用例，该场景中不存在任何正确的匹配项。
    查询: Katsushika Hokusai
    """
    print("\n" + "=" * 50)
    print("--- Running Test Example 2: Katsushika Hokusai (No Exact Match) ---")
    print("=" * 50)

    # 1. 定义查询实体
    query_entity = Entity(
        entity_name="Katsushika Hokusai",
        entity_type="Artist",
        description="A Japanese ukiyo-e artist of the Edo period, best known for the woodblock print series 'Thirty-six Views of Mount Fuji'.",
    )

    # 2. 定义5个不匹配的候选实体
    candidate_entities = [
        Entity(
            entity_name="Utagawa Hiroshige",
            entity_type="Artist",
            description="A Japanese ukiyo-e artist, considered the last great master of that tradition.",
        ),
        Entity(
            entity_name="The Great Wave off Kanagawa",
            entity_type="Artwork",
            description="A woodblock print by the Japanese ukiyo-e artist Hokusai.",
        ),
        Entity(
            entity_name="Matsuo Bashō",
            entity_type="Poet",
            description="The most famous poet of the Edo period in Japan. Recognized as the master of haiku.",
        ),
        Entity(
            entity_name="Akira Kurosawa",
            entity_type="Film Director",
            description="A highly influential Japanese filmmaker who directed films like 'Seven Samurai'.",
        ),
        Entity(
            entity_name="Hokkaido",
            entity_type="Location",
            description="Japan's second largest island and the largest and northernmost prefecture.",
        ),
    ]

    # 3. 格式化并调用rerank
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]
    scores = reranker.rerank(
        query=query_string, documents=document_strings, instruction=instruction
    )

    # 4. 排序并打印结果
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    print(f"\nQuery Entity: {query_entity.entity_name}\n")
    print("Ranked candidate entities:")
    for entity, score in ranked_results:
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )


# --- 测试用例 3: 完全不相关的候选实体 (健全性检查) ---
def test_example_3(reranker: "TextRerankerProvider", instruction: str):
    from Core.Index.Graph import Entity

    """
    运行实体解析的第三个测试用例，该场景中所有候选实体都与查询实体完全不相关。
    这是一个健全性检查，预期所有分数都应极低。
    查询: Tokyo Tower
    """
    print("\n" + "=" * 50)
    print(
        "--- Running Test Example 3: Tokyo Tower (Sanity Check - No Related Matches) ---"
    )
    print("=" * 50)

    # 1. 定义一个非常明确的查询实体
    query_entity = Entity(
        entity_name="Tokyo Tower",
        entity_type="Landmark / Communications Tower",
        description="A communications and observation tower in the Shiba-koen district of Minato, Tokyo, Japan. At 333 meters, it is the second-tallest structure in Japan.",
    )

    # 2. 定义5个完全不相关的候选实体
    candidate_entities = [
        Entity(
            entity_name="Ramen",
            entity_type="Food",
            description="A Japanese noodle dish. It consists of Chinese-style wheat noodles served in a broth, often flavored with soy sauce or miso.",
        ),
        Entity(
            entity_name="Sony Group Corporation",
            entity_type="Company",
            description="A Japanese multinational conglomerate corporation headquartered in Minato, Tokyo, known for electronics like the PlayStation.",
        ),
        Entity(
            entity_name="Samurai",
            entity_type="Historical Class",
            description="The military nobility and officer caste of medieval and early-modern Japan.",
        ),
        Entity(
            entity_name="Statue of Liberty",
            entity_type="Landmark",
            description="A colossal neoclassical sculpture on Liberty Island in New York Harbor in the United States.",
        ),
        Entity(
            entity_name="Godzilla",
            entity_type="Fictional Character (Kaiju)",
            description="A fictional monster, or kaiju, originating from a series of Japanese films. It is often seen attacking Tokyo.",
        ),
    ]

    # 3. 格式化并调用rerank
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]
    scores = reranker.rerank(
        query=query_string, documents=document_strings, instruction=instruction
    )

    # 4. 排序并打印结果
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    print(f"\nQuery Entity: {query_entity.entity_name}\n")
    print("Ranked candidate entities:")
    for entity, score in ranked_results:
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )


# --- 使用示例 ---
if __name__ == "__main__":
    import torch

    # 1. 初始化Reranker
    # 如果GPU内存有限，可以尝试更小的数据类型，如 torch.float16
    try:
        reranker = TextRerankerProvider(
            # model_name="Qwen/Qwen3-Reranker-0.6B",
            model_name="Qwen/Qwen3-Reranker-4B",
            torch_dtype=torch.bfloat16,
            device="cuda:2",
        )
    except Exception as e:
        print(
            f"Failed to initialize reranker, maybe due to GPU memory limitations. Error: {e}"
        )
        reranker = None

    if reranker:
        # 2. 定义实体排序任务的Instruction
        # 这个Instruction明确了任务目标：判断是否为同一个实体，而不仅仅是主题相关。
        entity_resolution_instruction = (
            "You are an expert entity resolution system. Your task is to verify if a 'Candidate Entity' "
            "is an exact match to a 'Query Entity'. Both entities were extracted by an LLM, so you must "
            "be robust to common extraction variations.\n\n"
            "1. **Analyze the Entity Name**: Be aware that names may not be identical. They can be "
            "abbreviations (e.g., 'LLM' vs 'Large Language Model'), aliases, or have minor variations. "
            "Your goal is to determine if these different name strings refer to the same concept.\n\n"
            "2. **Verify Entity Type Compatibility**: This is a critical check. The `entity_type` fields "
            "must be logically compatible. An entity of type 'Person' cannot match an entity of type 'Artwork', "
            "'Company', or 'Location'. For example, 'Artist' and 'Painter' are compatible types, but 'Artist' "
            "and 'Artwork' are fundamentally different and **cannot be a match**. Use this as a strong filter.\n\n"
            "3. **Synthesize the Description**: The descriptions are extracted from different text passages "
            "and will naturally differ. Do not simply compare the text. Instead, synthesize the information "
            "from both descriptions to see if they build a coherent and consistent profile of a single, "
            "unique real-world entity.\n\n"
            "4. **Assume Non-match by Default**: It is expected that in most cases, the candidate entities "
            "are **NOT** a match. Your default stance should be to assign a low score. Only give a high "
            "score when there is conclusive evidence of a perfect identity match after accounting for the "
            "potential variations in name and description.\n\n"
            "A high score signifies a confirmed identity. A low score is required for all other cases, "
            "including entities that are merely related, similar, or share a common topic."
        )

        test_example_1(
            reranker=reranker,
            instruction=entity_resolution_instruction,
        )

        test_example_2(reranker, entity_resolution_instruction)
        test_example_3(reranker, entity_resolution_instruction)
