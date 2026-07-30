from __future__ import annotations

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.configs.graph_config import GraphConfig
from Core.Index.Graph import Entity, Relationship, SetEncoder
from Core.Index.Tree import TreeNode, NodeType
from Core.prompts.kg_prompt import (
    DEFAULT_ENTITY_TYPES,
    ENTITY_EXTRACTION,
    ENTITY_CONTINUE_EXTRACTION,
    ENTITY_IF_LOOP_EXTRACTION,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    EQUATION_ENTITY_EXTRACTION,
    IMAGE_ENTITY_EXTRACTION,
    TABLE_DESCRIPTION_EXTRACTION,
    TABLE_BODY_EXTRACTION,
    TABLE_ENTITY_EXTRACTION,
    SECTION_ENTITY_EXTRACTION,
    ExtractionResult,
    EntityExtractionResult,
    FormulaExtractionResult,
)
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.utils.utils import (
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    TextProcessor,
    num_tokens,
)
from Core.utils.table_utils import (
    create_hierarchical_headers,
    identify_header_rows,
    parse_html_table_to_grid,
)
from Core.utils.bounded_executor import iter_bounded_ordered

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Iterable, Iterator, List, Dict, Tuple, Any, Union, final
import hashlib
import logging
import re
import json
import os
import tempfile
from urllib.parse import urlsplit, urlunsplit
from nltk.metrics.distance import edit_distance

logger = logging.getLogger(__name__)


KG_CHECKPOINT_SCHEMA_VERSION = "docarag.kg-extraction-checkpoint.v1"

_TEXTUAL_KG_NODE_TYPES = frozenset(
    {
        NodeType.TEXT,
        NodeType.LIST,
        NodeType.CODE,
        NodeType.ALGORITHM,
        NodeType.ABSTRACT,
        NodeType.REFERENCE,
    }
)
_VISUAL_KG_NODE_TYPES = frozenset(
    {NodeType.IMAGE, NodeType.CHART, NodeType.DIAGRAM}
)

# Reachable node types deliberately kept out of entity extraction. A type in
# neither KG set is skipped in silence, exactly as in the index, so the omissions
# are named here and a test holds the two lists together.
#
# The furniture reasons were decided when furniture got its own node type. The
# other three were not written down at the time and are recorded here from the
# behaviour, so that changing them is a decision rather than an accident.
_NOT_EXTRACTED_NODE_TYPES = frozenset(
    {
        # Page furniture names no entity the document is about: a running header
        # would contribute the publisher's name to every page it appears on.
        NodeType.HEADER,
        NodeType.FOOTER,
        NodeType.ASIDE,
        NodeType.DISCARDED,
        # A table's evidence is its cells and an equation's is LaTeX. Neither
        # reads as prose naming entities, and the paragraph that introduces them
        # is extracted in their place.
        NodeType.TABLE,
        NodeType.EQUATION,
        # A heading names the section its children discuss, so extracting from it
        # duplicates entities those children already yield.
        NodeType.TITLE,
    }
)

_SENSITIVE_CONFIG_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "password",
        "secret",
        "client_secret",
        "credential",
        "credentials",
        "authorization",
        "proxy_authorization",
        "cookie",
        "headers",
    }
)


def _is_sensitive_config_field(name: Any) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_CONFIG_FIELDS
        or normalized == "token"
        or normalized.endswith("_token")
        or normalized.endswith("_api_key")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_private_key")
        or normalized.endswith("_credential")
        or normalized.endswith("_credentials")
    )


def _is_url_config_field(name: Any) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return (
        normalized in {"api_base", "base_url", "endpoint", "server_url", "url"}
        or normalized.endswith("_api_base")
        or normalized.endswith("_endpoint")
        or normalized.endswith("_url")
        or normalized.endswith("_uri")
    )


def _sanitize_endpoint(value: str) -> str:
    """Retain endpoint routing identity without persisting URL credentials."""

    has_authority_marker = "://" in value
    parse_value = value if has_authority_marker else f"//{value}"
    try:
        parsed = urlsplit(parse_value)
        hostname = parsed.hostname
        if not hostname:
            return value.split("?", 1)[0].split("#", 1)[0]
        host = f"[{hostname}]" if ":" in hostname else hostname
        port = parsed.port
        netloc = f"{host}:{port}" if port is not None else host
        sanitized = urlunsplit(
            (parsed.scheme if has_authority_marker else "", netloc, parsed.path, "", "")
        )
        return sanitized if has_authority_marker else sanitized.removeprefix("//")
    except ValueError:
        # Even malformed endpoints must not retain query/fragment credentials.
        without_suffix = value.split("?", 1)[0].split("#", 1)[0]
        if "@" in without_suffix:
            prefix, suffix = without_suffix.rsplit("@", 1)
            if ":" in prefix:
                return suffix
        return without_suffix


def _canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by checkpoint identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class KGExtractionError(RuntimeError):
    """One source node could not produce a trustworthy KG observation."""

    def __init__(self, node_id: int, modality: str, error: Exception):
        self.node_id = node_id
        self.modality = modality
        self.error = error
        super().__init__(
            f"KG extraction failed for {modality} node {node_id}: {error}"
        )


# 定义一个函数，用于通过空格和所有非字母数字的符号来分割字符串
def split_by_space_and_symbols(text):
    # re.split(r'\W+', text) 会按所有非字母数字字符（包括符号和下划线）进行分割
    # 过滤掉分割后可能产生的空字符串
    words = [word for word in re.split(r"\W+", text) if word]
    return words


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, node: TreeNode):
        pass


class LocalExtractor(BaseExtractor):
    """
    Extractor that uses local NLP libraries (mainly spaCy) to extract entities and relations.
    """

    def __init__(
        self, model_name: str = "en_core_web_trf", device: str = "cpu"
    ):
        """
        Initializes the extractor with GPU support.

        Args:
            model_name (str): The name of the spaCy transformer model.
        """
        import spacy

        if device == "cuda":
            spacy.require_gpu()
            logger.info("GPU activated for spaCy.")
        elif device == "auto":
            using_gpu = spacy.prefer_gpu()
            logger.info("spaCy device selected: %s", "cuda" if using_gpu else "cpu")
        else:
            logger.info("spaCy device selected: cpu")

        logger.info(f"Loading spaCy model '{model_name}'...")
        self.nlp = spacy.load(model_name)
        logger.info("Model loaded successfully.")

    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Extracts named entities from the doc and returns a structured list."""
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
        return entities

    def _extract_relations(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """
        Extracts relation triples (subject, relation, object) from the doc.
        This is the core method based on dependency parsing.
        """
        relations = []
        # For efficient lookup, create a mapping from a token's index to its entity span
        token_index_to_entity = {token.i: ent for ent in doc.ents for token in ent}

        for token in doc:
            # Rule 1: Look for verbs as potential relations
            if token.pos_ == "VERB":
                subjects = []
                objects = []

                # Iterate over the verb's children to find subjects and objects
                for child in token.children:
                    # Find subject (nsubj) or passive subject (nsubjpass)
                    if "nsubj" in child.dep_:
                        # Find the full entity span corresponding to the subject token
                        if child.i in token_index_to_entity:
                            subjects.append(token_index_to_entity[child.i])

                    # Find object (dobj), attribute (attr), or object of a preposition (pobj)
                    elif "obj" in child.dep_ or "attr" in child.dep_:
                        # Find the full entity span corresponding to the object token
                        if child.i in token_index_to_entity:
                            # Case A: Direct object (dobj) or attribute (attr)
                            # The relation is the verb lemma itself
                            relation_phrase = token.lemma_
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                        # Case B: Object of a preposition (pobj)
                        elif child.dep_ == "pobj":
                            # 'child' is the object entity, 'child.head' is the preposition
                            preposition = child.head
                            # The relation phrase is the verb's lemma + the preposition's text
                            relation_phrase = f"{token.lemma_} {preposition.text}"
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                # Combine all found subjects and objects
                for subj_ent in subjects:
                    for rel, obj_ent in objects:
                        # Avoid duplicates and self-referential relations
                        if subj_ent != obj_ent:
                            relations.append((subj_ent.text, rel, obj_ent.text))

        # Remove duplicates
        return list(set(relations))

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        """
        Executes the full entity and relation extraction pipeline.

        Args:
            text (str): The English text to be processed.

        Returns:
            Dict[str, Any]: A dictionary containing lists of 'entities' and 'relations'.
        """
        text = node.meta_info.content
        node_idx = node.index_id
        doc = self.nlp(text)
        raw_entities = self._extract_entities(doc)
        raw_relations = self._extract_relations(doc)
        entities = [
            Entity(
                entity_name=entity["text"],
                entity_type=entity["label"],
                description=(
                    f"Named entity recognized by the local parser as "
                    f"{entity['label']}."
                ),
                source_ids={node_idx},
            )
            for entity in raw_entities
        ]
        relations = [
            Relationship(
                src_entity_name=subject,
                tgt_entity_name=object_name,
                relation_name=predicate,
                weight=1.0,
                description=f"{subject} {predicate} {object_name}",
                source_ids={node_idx},
            )
            for subject, predicate, object_name in raw_relations
        ]

        return {
            "entities": entities,
            "relations": relations,
            "node_idx": node_idx,
        }

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        return self.extract(node)

    def extract_with_textacy(self, text: str) -> List[Tuple[str, str, str]]:
        """
        A simpler method: Use the Textacy library to directly extract SVO triples.
        This can serve as a quick baseline or a simplified approach.
        """
        import textacy.extract

        doc = self.nlp(text)
        svo_triples = textacy.extract.subject_verb_object_triples(doc)

        # Format Textacy's output into (subject_text, verb_text, object_text)
        relations = []
        for triple in svo_triples:
            subj = " ".join(t.text for t in triple.subject)
            verb = " ".join(t.text for t in triple.verb)
            obj = " ".join(t.text for t in triple.object)
            relations.append((subj, verb, obj))

        return relations


class LLMExtractor(BaseExtractor):
    def __init__(
        self,
        graph_config: GraphConfig,
        llm: LLM,
        vlm: VLM = None,
    ):
        self.llm = llm
        self.max_gleaning = graph_config.max_gleaning
        self.graph_config: GraphConfig = graph_config
        if self.graph_config.image_description_force:
            self.vlm = vlm
        else:
            self.vlm = None

    @classmethod
    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content,
        )

    def _extract_records_from_text(self, chunk_text: str):
        """
        Extract entity and relationship from chunk, which is used for the GraphRAG.
        Please refer to the following references:
        1. https://github.com/gusye1234/nano-graphrag
        2. https://github.com/HKUDS/LightRAG/tree/main
        """
        context = self._build_context_for_entity_extraction(chunk_text)
        prompt = ENTITY_EXTRACTION.format(**context)

        working_memory = Memory()

        working_memory.add(Message(content=prompt, role="user"))
        final_result = self.llm.get_completion(prompt)
        working_memory.add(Message(content=final_result, role="assistant"))

        for glean_idx in range(self.max_gleaning):
            working_memory.add(Message(content=ENTITY_CONTINUE_EXTRACTION, role="user"))
            glean_result = self.llm.get_completion(working_memory)

            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result

            if glean_idx == self.max_gleaning - 1:
                break

            working_memory.add(Message(content=ENTITY_IF_LOOP_EXTRACTION, role="user"))

            if_loop_result = self.llm.get_completion(working_memory)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        working_memory.clear()

        return split_string_by_multi_markers(
            final_result, [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER]
        )

    @classmethod
    def _handle_single_entity_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Entity, None]:

        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1])
        if not entity_name.strip():
            return None

        entity = Entity(
            entity_name=entity_name,
            entity_type=clean_str(record_attributes[2]),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

        return entity

    def _handle_single_relationship_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_entity_name=clean_str(record_attributes[1]),
            tgt_entity_name=clean_str(record_attributes[2]),
            weight=(
                float(record_attributes[-1])
                if is_float_regex(record_attributes[-1])
                else 1.0
            ),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

    def _build_graph_from_records(self, records: list[str], chunk_key: int):
        entities_list = []
        relationships_list = []
        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(
                match.group(1), [DEFAULT_TUPLE_DELIMITER]
            )
            entity = self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                entities_list.append(entity)
                continue

            relationship = self._handle_single_relationship_extraction(
                record_attributes, chunk_key
            )

            if relationship is not None:
                relationships_list.append(relationship)

        return entities_list, relationships_list

    def _extract_kg_from_text(self, node: TreeNode):
        content_texts = node.meta_info.content
        processor = TextProcessor()
        split_tokens = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(ENTITY_CONTINUE_EXTRACTION + ENTITY_EXTRACTION)
        )
        chunks = processor.split_text_into_chunks(
            text=content_texts, max_length=split_tokens
        )
        res_entities = []
        res_relation = []
        for text in chunks:
            records = self._extract_records_from_text(text)
            entities, relations = self._build_graph_from_records(records, node.index_id)
            res_entities.extend(entities)
            res_relation.extend(relations)
        return res_entities, res_relation

    def _extract_kg_table_step1(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """
        第一步：从描述中提取实体。
        """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()

        column_headers = []
        column_headers = []
        if num_header_rows > 0 and grid:
            column_headers = create_hierarchical_headers(grid, num_header_rows)

        input_data = {"description": description, "column_headers": column_headers}
        input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        prompt = TABLE_DESCRIPTION_EXTRACTION.format(
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_json=input_json_str,  # 假设 prompt 现在的占位符是 input_text
        )

        llm_entities_raw = []
        try:
            res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                prompt, schema=EntityExtractionResult
            )
            if res and res.entities:
                llm_entities_raw = res.entities
        except Exception as e:
            logger.error(f"Step 1 LLM call failed for node {node.index_id}: {e}")

        # 2. 验证并分离 LLM 的提取结果
        table_entity_from_llm = next(
            (ent for ent in llm_entities_raw if ent.entity_type == "TABLE"), None
        )
        other_entities_from_llm = [
            ent for ent in llm_entities_raw if ent.entity_type != "TABLE"
        ]

        # 3. 决定最终的 TABLE 实体（LLM版本 vs. Fallback版本）
        final_table_entity: Entity
        if table_entity_from_llm:
            # 如果LLM成功提取了TABLE实体，则使用它
            logger.info(
                f"LLM successfully extracted TABLE entity for node {node.index_id}."
            )
            final_table_entity = Entity(
                entity_name=table_entity_from_llm.entity_name,
                entity_type=table_entity_from_llm.entity_type,
                description=table_entity_from_llm.description,
                source_ids={node.index_id},
            )
        else:
            # 如果LLM未能提取，或者没有描述，或者调用失败，则创建我们的 fallback 实体
            logger.warning(
                f"LLM failed to provide a TABLE entity for node {node.index_id}. Creating a fallback."
            )
            if description:
                table_name_prefix = " ".join(description.split()[:8])
                table_desc = f"A data table described as: {description}"
            else:
                table_name_prefix = f"Node {node.index_id}"
                table_desc = "A table with no available description."

            final_table_entity = Entity(
                entity_name=f"Table: {table_name_prefix}...",
                entity_type="TABLE",
                description=table_desc,
                source_ids={node.index_id},
            )

        final_other_entities = [
            Entity(
                entity_name=ent.entity_name,
                entity_type=ent.entity_type,
                description=ent.description,
                source_ids={node.index_id},
            )
            for ent in other_entities_from_llm
        ]

        final_entities = [final_table_entity] + final_other_entities

        return final_entities

    def _split_table_into_batches(
        self,
        grid: List[List[str]],
        num_header_rows: int,
        # description 和 column_headers 的token开销将在主函数中计算
        max_tokens_for_rows: int,
    ) -> List[List[str]]:
        # 1. 提取并转换表格主体为行字符串
        body_rows = grid[num_header_rows:]
        # 我们使用 " | " 作为分隔符，这与我们prompt中的示例一致
        row_strings = [" | ".join(cell.strip() for cell in row) for row in body_rows]

        all_batches = []
        current_batch = []
        current_batch_tokens = 0

        for row_str in row_strings:
            row_tokens = num_tokens(row_str)

            # 如果单个行就超过了限制，将其单独放入一个批次。
            # 这是一个边缘情况，表示批处理大小可能需要调整。
            if row_tokens > max_tokens_for_rows:
                # 如果当前批次有内容，先提交
                if current_batch:
                    all_batches.append(current_batch)
                # 将超长的行单独作为一个批次
                all_batches.append([row_str])
                # 重置当前批次
                current_batch = []
                current_batch_tokens = 0
                continue

            # 如果将当前行加入批次会超出限制
            if current_batch_tokens + row_tokens > max_tokens_for_rows:
                # 提交当前已满的批次
                all_batches.append(current_batch)
                # 用当前行开始一个新的批次
                current_batch = [row_str]
                current_batch_tokens = row_tokens
            else:
                # 将当前行加入批次
                current_batch.append(row_str)
                current_batch_tokens += row_tokens

        # 3. 不要忘记提交最后一个批次
        if current_batch:
            all_batches.append(current_batch)

        return all_batches

    def _create_prompts_from_batches(
        self, batches: List[List[str]], description: str, column_headers: List[str]
    ) -> List[str]:
        """根据批次数据、描述和列表头，创建完整的prompt列表。"""
        prompts = []
        for batch in batches:
            # 1. 构建输入JSON对象
            input_data = {
                "description": description,
                "column_headers": column_headers,
                "rows_batch": batch,  # batch现在是行字符串的列表
            }
            # 2. 序列化为JSON字符串
            input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

            # 3. 格式化主Prompt模板
            # 假设 DEFAULT_ENTITY_TYPES 和 TABLE_BODY_EXTRACTION 已定义
            prompt = TABLE_BODY_EXTRACTION.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=input_json_str,
            )
            prompts.append(prompt)

        return prompts

    def _extract_kg_table_step2(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """ """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()
        column_headers = create_hierarchical_headers(grid, num_header_rows)

        prompt_overhead = num_tokens(description) + num_tokens(
            json.dumps(column_headers)
        )
        max_tokens_for_rows = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(TABLE_BODY_EXTRACTION)
            - prompt_overhead
        )

        batches = self._split_table_into_batches(
            grid, num_header_rows, max_tokens_for_rows
        )
        prompts = self._create_prompts_from_batches(
            batches, description, column_headers
        )

        logger.info(
            f"Node {node.index_id}: Table body split into {len(prompts)} batches for processing."
        )

        all_body_entities = []
        for i, prompt in enumerate(prompts):
            logger.info(
                f"Processing batch {i+1}/{len(prompts)} for node {node.index_id}..."
            )
            try:
                res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                    prompt, schema=EntityExtractionResult
                )
                if res and res.entities:
                    batch_entities = [
                        Entity(
                            entity_name=ent.entity_name,
                            entity_type=ent.entity_type,
                            description=ent.description,
                            source_ids={node.index_id},
                        )
                        for ent in res.entities
                    ]
                    all_body_entities.extend(batch_entities)
            except Exception as e:
                raise KGExtractionError(
                    node.index_id,
                    "table",
                    RuntimeError(f"body batch {i + 1}/{len(prompts)} failed: {e}"),
                ) from e

        return all_body_entities

    def _extract_kg_from_table(self, node: TreeNode):

        table_body = node.meta_info.table_body
        grid = parse_html_table_to_grid(table_body)
        num_header_rows = identify_header_rows(grid)

        desc_entities = self._extract_kg_table_step1(node, grid, num_header_rows)
        body_entities = self._extract_kg_table_step2(node, grid, num_header_rows)

        table_entity = desc_entities[0]
        other_step1_entities = desc_entities[1:]

        final_other_entities = []
        seen_entity_names = {ent.entity_name for ent in other_step1_entities}

        for ent in body_entities:
            if ent.entity_name not in seen_entity_names:
                final_other_entities.append(ent)
                seen_entity_names.add(ent.entity_name)

        final_entities = [table_entity] + other_step1_entities + final_other_entities

        final_relations = [
            Relationship(
                src_entity_name=table_entity.entity_name,
                tgt_entity_name=ent.entity_name,
                weight=9.0,
                description=f"Table '{table_entity.entity_name}' contains data about '{ent.entity_name}'.",
                source_ids={node.index_id},
            )
            for ent in final_entities
            if ent.entity_type != "TABLE"
        ]
        logger.info(
            f"Total relations created: {len(final_relations)} in Table node: {node.index_id}"
        )

        return final_entities, final_relations

    def _extract_kg_from_image(self, node: TreeNode):
        if self.graph_config.image_description_force == False:
            logger.warning(
                "Image description force is disabled, skipping image extraction."
            )
            return [], []
        image_description = node.meta_info.content
        image_path = node.meta_info.img_path
        prompt = IMAGE_ENTITY_EXTRACTION.format(
            image_description=image_description,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
        )
        try:
            res: Union[None | EntityExtractionResult] = self.vlm.generate_json(
                prompt_or_memory=prompt,
                images=[image_path],
                schema=EntityExtractionResult,
            )
            if res is None:
                logger.warning("No response from LLM.")
                return [], []

            entities = res.entities if res.entities else []
            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]

            image_entity = next(
                (ent for ent in final_entities if ent.entity_type == "IMAGE"), None
            )
            # check image_entity the first two words in the description or not
            if image_entity is not None:
                if (
                    split_by_space_and_symbols(image_entity.entity_name)[:2]
                    != split_by_space_and_symbols(image_description)[:2]
                ):
                    image_entity = None
                    logger.info(
                        "Image entity name does not match the original image description."
                    )
                    logger.info("Creating a default image entity instead.")

            if image_entity is None:
                # create a default image_entity if not found,
                # use the first 5 words of the description as the name
                # use the description as the description
                logger.warning(
                    "No image entity found in the extracted entities. Creating a default image entity."
                )

                image_name = "Image " + " ".join(image_description.split()[:5])
                description = "Original image description: " + image_description
                image_entity = Entity(
                    entity_name=image_name,
                    entity_type="IMAGE",
                    description=description,
                    source_ids={node.index_id},
                )
                final_entities.append(image_entity)

            # create relationship from the image entity to all other entities
            final_relations = [
                Relationship(
                    src_entity_name=image_entity.entity_name,
                    tgt_entity_name=ent.entity_name,
                    weight=9.0,  # Default weight
                    description=f"Image entity {image_entity.entity_name} related to {ent.entity_name}",
                    source_ids={node.index_id},
                )
                for ent in final_entities
                if ent.entity_name != image_entity.entity_name
            ]

            return final_entities, final_relations

        except Exception as e:
            logger.exception(f"Error extracting entities and relations from image: {e}")
            raise KGExtractionError(node.index_id, "image", e) from e

    def _extract_kg_from_equation(self, node: TreeNode):
        latex_text = node.meta_info.content
        try:
            prompt = EQUATION_ENTITY_EXTRACTION.format(formula_latex_code=latex_text)
            res: Union[None | FormulaExtractionResult] = self.llm.get_json_completion(
                prompt, schema=FormulaExtractionResult
            )
            if res is None:
                logger.info("LLM returned no response for equation extraction.")
                logger.info("Create a default entity for the equation.")

                equation_text = latex_text.split()[:8]
                equation_text = " ".join(equation_text)

                final_entity = Entity(
                    entity_name=f"Equation: {equation_text}",
                    entity_type="EQUATION_OR_FORMULA",
                    description=f"A formula represented by LaTeX code: {latex_text}",
                    source_ids={node.index_id},
                )
                return [final_entity], []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type="EQUATION_OR_FORMULA",
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in res.entities
            ]

            return final_entities, []

        except Exception as e:
            logger.exception(
                f"Error extracting entities and relations from equation: {e}"
            )
            raise KGExtractionError(node.index_id, "equation", e) from e

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        try:
            if node.type not in NodeType:
                raise ValueError(f"Unsupported node type: {node.type}")
            if node.type in _TEXTUAL_KG_NODE_TYPES:
                entities, relations = self._extract_kg_from_text(node)
            elif node.type in _VISUAL_KG_NODE_TYPES:
                entities, relations = self._extract_kg_from_image(node)
            elif node.type == NodeType.TABLE:
                entities, relations = self._extract_kg_from_table(node)
            elif node.type == NodeType.EQUATION:
                entities, relations = self._extract_kg_from_equation(node)
            else:
                raise ValueError(f"Unsupported node type: {node.type}")

            return {
                "entities": entities,
                "relations": relations,
                "node_idx": node.index_id,
            }

        except Exception as e:
            logger.exception(f"Error extracting entities and relation: {e}")
            if isinstance(e, KGExtractionError):
                raise
            modality = getattr(node.type, "value", str(node.type))
            raise KGExtractionError(node.index_id, modality, e) from e
        finally:
            logger.info(
                f"Finish extracting entities and relations in TreeNode {node.index_id}."
            )

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        title_path_obj = [
            {"depth": i, "title": node.meta_info.content}
            for i, node in enumerate(title_path)
        ]
        # find previous and next section title in sibling nodes
        # previou section index_id is small than current node index_id
        # next section index_id is large than current node index_id
        # sibling_nodes is sorted by index_id
        previous_section_title = ""
        next_section_title = ""
        for sibling in sibling_nodes:
            if sibling.index_id < node.index_id and not previous_section_title:
                previous_section_title = sibling.meta_info.content
            elif sibling.index_id > node.index_id and not next_section_title:
                next_section_title = sibling.meta_info.content
                break

        prep_prompt = {
            "context": {
                "previous_section_title": (
                    previous_section_title if previous_section_title else "null"
                ),
                "title_path": title_path_obj,
                "next_section_title": (
                    next_section_title if next_section_title else "null"
                ),
            },
            "title_to_process": node.meta_info.content,
        }
        input_json = json.dumps(prep_prompt, ensure_ascii=False, indent=2)
        prompt = SECTION_ENTITY_EXTRACTION.format(
            input_json=input_json, entity_types=",".join(DEFAULT_ENTITY_TYPES)
        )
        try:
            res = self.llm.get_json_completion(prompt=prompt, schema=ExtractionResult)
            if res is None:
                raise ValueError("LLM returned no title extraction response")
            entities = res.entities if res.entities else []
            relations = res.relationships if res.relationships else []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]
            final_relations = [
                Relationship(
                    src_entity_name=rel.src_entity_name,
                    tgt_entity_name=rel.tgt_entity_name,
                    weight=rel.weight,
                    description=rel.description,
                    source_ids={node.index_id},
                )
                for rel in relations
            ]
            return {
                "entities": final_entities,
                "relations": final_relations,
                "node_idx": node.index_id,
            }
        except Exception as e:
            logger.exception(f"Error extracting entities and relations from title: {e}")
            raise KGExtractionError(node.index_id, "title", e) from e


class KGExtractor:
    def __init__(
        self,
        cfg_graph: GraphConfig,
        llm: LLM = None,
        vlm: VLM = None,
        save_path: str = None,
        # force_rebuild: bool = True,
        force_rebuild: bool = False,
    ):
        self.cfg_graph = cfg_graph
        extractor_type = cfg_graph.extractor_type
        if extractor_type == "local":
            self.extractor = LocalExtractor(
                cfg_graph.local_model_name, device=cfg_graph.local_device
            )
        elif extractor_type == "llm":
            self.extractor = LLMExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)

        self.llm = llm
        self.vlm = vlm
        self.save_path = os.path.join(save_path, "kg_extractor_res")
        os.makedirs(self.save_path, exist_ok=True)
        logger.info(f"KG extraction results will be saved to {self.save_path}")
        self.force_rebuild = force_rebuild

    def res_repair(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert all entity names and types to lower case in the result dictionary.
        This is useful for normalization and comparison purposes.
        """

        def _clean_name(name: str) -> str:
            """
            Normalizes a string by converting to lowercase, stripping whitespace,
            and consolidating multiple spaces into a single space.
            """
            cleaned_name = name.lower()
            cleaned_name = cleaned_name.strip()
            cleaned_name = re.sub(r"\s+", " ", cleaned_name)
            return cleaned_name

        def relation_name_refine(relation_entity_name, entity_name_set, res_obj):
            if relation_entity_name in entity_name_set:
                return relation_entity_name

            # Find the most similar entity name by computing edit distance
            logger.info(
                f"Relation entity '{relation_entity_name}' not found in entities."
            )
            logger.info(f"Available entities: {', '.join(entity_name_set)}")

            # Calculate normalized similarity
            most_similar_entity = None
            if entity_name_set:
                most_similar_entity = min(
                    entity_name_set,
                    key=lambda ent: edit_distance(ent, relation_entity_name),
                )
                distance = edit_distance(most_similar_entity, relation_entity_name)
                max_len = max(len(most_similar_entity), len(relation_entity_name))
                similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
            else:
                # If the entity set was empty, similarity is effectively 0 to force creation
                similarity = 0.0

            if most_similar_entity and similarity >= 0.9:
                logger.info(
                    f"Similarity to '{most_similar_entity}' is {similarity:.2f}. Using it instead."
                )
                return most_similar_entity
            else:
                logger.info(
                    f"Similarity to closest entity is {similarity:.2f} (< 0.9). Creating a new 'UNKNOWN' entity for '{relation_entity_name}'."
                )

                # Create the new entity object
                new_entity = Entity(
                    entity_name=relation_entity_name,
                    entity_type="UNKNOWN",
                    description="",  # Description is empty as requested
                    source_ids={res_obj["node_idx"]},
                )

                # Add the new entity to the main list and the name to the set
                res_obj.get("entities", []).append(new_entity)
                entity_name_set.add(relation_entity_name)

                return relation_entity_name

        entity_name_set = set()
        entity_map: dict[tuple[str, str], Entity] = {}
        entity_descriptions: dict[tuple[str, str], list[str]] = {}
        for entity in res.get("entities", []):
            entity.entity_name = _clean_name(entity.entity_name)
            # Strip before collapsing spaces: the type is half the node id, and
            # a model returning " METHOD " would otherwise become "_METHOD_",
            # a distinct type whose id the vector store then rejects for
            # surrounding whitespace -- the failure that ended a run at 12,352
            # of 16,091 nodes.
            entity.entity_type = entity.entity_type.strip().upper()
            entity.entity_type = entity.entity_type.replace(" ", "_")
            entity_name_set.add(entity.entity_name)
            key = (entity.entity_name, entity.entity_type)
            existing_entity = entity_map.get(key)
            if existing_entity is None:
                entity_map[key] = entity
                entity_descriptions[key] = [entity.description]
                continue

            entity_descriptions[key].append(entity.description)
            existing_entity.source_ids.update(entity.source_ids)
            logger.info(
                "Aggregated duplicate entity: %s of type %s",
                entity.entity_name,
                entity.entity_type,
            )

        # Dict insertion order preserves the first occurrence of every entity.
        # Building a replacement avoids skipped duplicates from list.remove while
        # iterating. Joining each description group once keeps work linear in the
        # number of entities plus the total description size.
        for key, entity in entity_map.items():
            entity.description = " ".join(entity_descriptions[key])
        res["entities"] = list(entity_map.values())

        valid_relations = []
        relation_map = {}
        for relation in res.get("relations", []):
            src_name = _clean_name(relation.src_entity_name)
            tgt_name = _clean_name(relation.tgt_entity_name)

            repaired_src_name = relation_name_refine(src_name, entity_name_set, res)
            repaired_tgt_name = relation_name_refine(tgt_name, entity_name_set, res)

            # Check if the relation already exists in the map
            relation_key = (
                repaired_src_name,
                repaired_tgt_name,
                relation.relation_name,
            )
            if relation_key in relation_map:
                # If the relation is same as an existing one, remove the new one
                continue
            else:
                # Add the relation to the map
                relation_map[relation_key] = relation

            # The refine function should now always return a valid name
            relation.src_entity_name = repaired_src_name
            relation.tgt_entity_name = repaired_tgt_name
            valid_relations.append(relation)

        # Replace the old relations list with the filtered and repaired one
        res["relations"] = valid_relations

        return res

    @classmethod
    def _checkpoint_value(cls, value: Any) -> Any:
        """Convert config values to deterministic, secret-free JSON data."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return cls._checkpoint_value(value.value)
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: cls._checkpoint_config_item(
                    field.name, getattr(value, field.name)
                )
                for field in fields(value)
                if not _is_sensitive_config_field(field.name)
            }
        if hasattr(value, "model_dump"):
            return cls._checkpoint_value(value.model_dump())
        if isinstance(value, dict):
            return {
                str(key): cls._checkpoint_config_item(key, item)
                for key, item in value.items()
                if not _is_sensitive_config_field(key)
            }
        if isinstance(value, (list, tuple)):
            return [cls._checkpoint_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            converted = [cls._checkpoint_value(item) for item in value]
            return sorted(converted, key=_canonical_json)
        if hasattr(value, "__dict__"):
            return {
                str(key): cls._checkpoint_config_item(key, item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
                and not _is_sensitive_config_field(key)
            }
        return {
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}"
        }

    @classmethod
    def _checkpoint_config_item(cls, name: Any, value: Any) -> Any:
        if _is_url_config_field(name) and isinstance(value, str):
            return _sanitize_endpoint(value)
        return cls._checkpoint_value(value)

    @classmethod
    def _model_checkpoint_identity(cls, model: Any) -> Any:
        if model is None:
            return None
        config = getattr(model, "config", None)
        return {
            "controller": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": cls._checkpoint_value(config),
        }

    @staticmethod
    def _node_type(node: TreeNode) -> str:
        node_type = node.type
        return node_type.value if isinstance(node_type, Enum) else str(node_type)

    @staticmethod
    def _value_digest(value: Any) -> Dict[str, Any]:
        return {
            "present": value is not None,
            "sha256": _sha256_json(value),
        }

    @staticmethod
    def _file_identity(path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "not-configured"}

        path_text = os.fspath(path)
        path_digest = _sha256_text(path_text)
        digest = hashlib.sha256()
        try:
            with open(path_text, "rb") as image_file:
                while True:
                    chunk = image_file.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            return {
                "status": "present",
                "path_sha256": path_digest,
                "bytes_sha256": digest.hexdigest(),
                "size": os.path.getsize(path_text),
            }
        except OSError as error:
            return {
                "status": "unreadable",
                "path_sha256": path_digest,
                "error_type": type(error).__name__,
            }

    @classmethod
    def _node_evidence_identity(cls, node: TreeNode) -> Dict[str, Any]:
        meta = node.meta_info
        source_observation_ids = list(meta.source_observation_ids or [])
        source_observation_ids.sort()
        source_canonical_block_ids = list(meta.source_canonical_block_ids or [])
        source_canonical_revision_ids = list(
            meta.source_canonical_revision_ids or []
        )
        identity: Dict[str, Any] = {
            "node_idx": node.index_id,
            "node_type": cls._node_type(node),
            "source": {
                "source_sha256": meta.source_sha256,
                "document_version_id": meta.document_version_id,
                "canonical_block_id": meta.canonical_block_id,
                "canonical_revision_id": meta.canonical_revision_id,
                "observation_id": meta.observation_id,
                "source_observation_ids": source_observation_ids,
                "derived_evidence_id": meta.derived_evidence_id,
                "source_canonical_block_ids": source_canonical_block_ids,
                "source_canonical_revision_ids": source_canonical_revision_ids,
                "canonical_representation_ids": cls._value_digest(
                    meta.canonical_representation_ids
                ),
                "canonical_asset_ids": cls._value_digest(meta.canonical_asset_ids),
                "selected_representation_ids": cls._value_digest(
                    meta.selected_representation_ids
                ),
                "selected_asset_ids": cls._value_digest(meta.selected_asset_ids),
                "source_evidence_refs": cls._value_digest(meta.source_evidence_refs),
                "parser_provenance": cls._value_digest(meta.parser_provenance),
                "quality_status": meta.quality_status,
                "reconciliation_method": meta.reconciliation_method,
                "transformations": cls._value_digest(meta.transformations),
            },
            "page": {
                "page_idx": meta.page_idx,
                "page_id": meta.page_id,
                "page_path": cls._value_digest(meta.page_path),
                "source_geometry": cls._value_digest(meta.source_geometry),
            },
            "evidence": {
                "content": cls._value_digest(meta.content),
                "table_body": cls._value_digest(meta.table_body),
                "caption": cls._value_digest(meta.caption),
                "footnote": cls._value_digest(meta.footnote),
            },
        }
        if cls._node_type(node) in {item.value for item in _VISUAL_KG_NODE_TYPES}:
            identity["evidence"]["image"] = cls._file_identity(meta.img_path)
        return identity

    def _graph_checkpoint_identity(self) -> Dict[str, Any]:
        setting_names = (
            "extractor_type",
            "local_model_name",
            "local_device",
            "image_description_force",
            "max_gleaning",
            "extraction_max_workers",
            "extraction_max_in_flight",
            "extraction_batch_size",
        )
        return {
            name: self._checkpoint_value(getattr(self.cfg_graph, name, None))
            for name in setting_names
        }

    def _prompt_checkpoint_identity(self, node_type: str, kind: str) -> Dict[str, str]:
        if getattr(self.cfg_graph, "extractor_type", None) != "llm":
            return {}

        if kind == "title":
            templates = {"SECTION_ENTITY_EXTRACTION": SECTION_ENTITY_EXTRACTION}
        elif node_type in {item.value for item in _TEXTUAL_KG_NODE_TYPES}:
            templates = {
                "ENTITY_EXTRACTION": ENTITY_EXTRACTION,
                "ENTITY_CONTINUE_EXTRACTION": ENTITY_CONTINUE_EXTRACTION,
                "ENTITY_IF_LOOP_EXTRACTION": ENTITY_IF_LOOP_EXTRACTION,
            }
        elif node_type == NodeType.TABLE.value:
            templates = {
                "TABLE_DESCRIPTION_EXTRACTION": TABLE_DESCRIPTION_EXTRACTION,
                "TABLE_BODY_EXTRACTION": TABLE_BODY_EXTRACTION,
            }
        elif node_type in {item.value for item in _VISUAL_KG_NODE_TYPES}:
            templates = {"IMAGE_ENTITY_EXTRACTION": IMAGE_ENTITY_EXTRACTION}
        elif node_type == NodeType.EQUATION.value:
            templates = {"EQUATION_ENTITY_EXTRACTION": EQUATION_ENTITY_EXTRACTION}
        else:
            templates = {}

        templates["extraction_protocol"] = _canonical_json(
            {
                "entity_types": DEFAULT_ENTITY_TYPES,
                "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
                "record_delimiter": DEFAULT_RECORD_DELIMITER,
                "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
            }
        )
        return {name: _sha256_text(template) for name, template in templates.items()}

    @classmethod
    def _title_context_identity(cls, nodes: Iterable[TreeNode]) -> List[Dict[str, Any]]:
        # Title extraction uses only each context node's order/index and displayed
        # text.  Hash exactly that prompt-shaping state; reading sibling image bytes
        # here would make every title cache lookup scale with unrelated assets.
        return [
            {
                "node_idx": context_node.index_id,
                "node_type": cls._node_type(context_node),
                "content": cls._value_digest(context_node.meta_info.content),
            }
            for context_node in nodes
        ]

    def _checkpoint_identity(
        self,
        node: TreeNode,
        *,
        kind: str,
        title_path: Iterable[TreeNode] = (),
        sibling_nodes: Iterable[TreeNode] = (),
    ) -> Dict[str, Any]:
        llm = getattr(self, "llm", None)
        vlm = getattr(self, "vlm", None)
        if llm is None:
            llm = getattr(self.extractor, "llm", None)
        if vlm is None:
            vlm = getattr(self.extractor, "vlm", None)

        node_type = self._node_type(node)
        identity = {
            "kind": kind,
            "node": self._node_evidence_identity(node),
            "models": {
                "llm": self._model_checkpoint_identity(llm),
                "vlm": self._model_checkpoint_identity(vlm),
            },
            "graph_extraction": self._graph_checkpoint_identity(),
            "prompt_sha256": self._prompt_checkpoint_identity(node_type, kind),
        }
        if kind == "title":
            identity["title_context"] = {
                "title_path": self._title_context_identity(title_path),
                "sibling_nodes": self._title_context_identity(sibling_nodes),
            }
        return identity

    def _checkpoint_path(self, node_idx: int) -> str:
        return os.path.join(self.save_path, f"kg_extractor_res_{node_idx}.json")

    @staticmethod
    def _payload_to_objects(payload: Dict[str, Any], node_idx: int) -> Dict[str, Any]:
        if payload.get("node_idx") != node_idx:
            raise ValueError(
                f"payload node_idx {payload.get('node_idx')!r} does not match "
                f"requested node_idx {node_idx!r}"
            )
        raw_entities = payload.get("entities")
        raw_relations = payload.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise ValueError("checkpoint payload must contain entity and relation lists")

        entities = [
            Entity(
                entity_name=entity["entity_name"],
                entity_type=entity["entity_type"],
                description=entity["description"],
                source_ids=set(entity.get("source_ids", [])),
            )
            for entity in raw_entities
        ]
        relations = [
            Relationship(
                src_entity_name=relation["src_entity_name"],
                tgt_entity_name=relation["tgt_entity_name"],
                relation_name=relation.get("relation_name", ""),
                weight=relation.get("weight", 0.0),
                description=relation.get("description", ""),
                source_ids=set(relation.get("source_ids", [])),
            )
            for relation in raw_relations
        ]
        return {"entities": entities, "relations": relations, "node_idx": node_idx}

    @staticmethod
    def _atomic_write_json(path: str, value: Dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            os.fchmod(descriptor, 0o600)
            checkpoint_file = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = -1
            with checkpoint_file:
                json.dump(
                    value,
                    checkpoint_file,
                    ensure_ascii=False,
                    indent=2,
                    cls=SetEncoder,
                )
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def save_tmp_res(
        self,
        res: Dict[str, Any],
        node_idx: int,
        checkpoint_identity: Dict[str, Any] | None = None,
    ) -> None:
        """Atomically save a source- and model-bound KG checkpoint envelope."""

        if checkpoint_identity is None:
            raise ValueError("checkpoint identity is required for KG checkpoint writes")
        if res.get("node_idx") != node_idx:
            raise ValueError("cannot checkpoint a result for a different node_idx")

        raw_payload = {
            "entities": [entity.model_dump() for entity in res["entities"]],
            "relations": [relation.model_dump() for relation in res["relations"]],
            "node_idx": res["node_idx"],
        }
        payload = self._checkpoint_value(raw_payload)
        if not isinstance(payload, dict):
            raise TypeError("KG checkpoint payload did not normalize to an object")
        envelope = {
            "schema_version": KG_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": _sha256_json(checkpoint_identity),
            "payload_sha256": _sha256_json(payload),
            "identity": checkpoint_identity,
            "payload": payload,
        }
        file_path = self._checkpoint_path(node_idx)
        self._atomic_write_json(file_path, envelope)
        logger.info("Saved KG extraction checkpoint to %s.", file_path)

    def load_tmp_res(
        self,
        node_idx: int,
        checkpoint_identity: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Load a checkpoint only when its full extraction identity matches."""

        file_path = self._checkpoint_path(node_idx)
        if self.force_rebuild or not os.path.exists(file_path):
            return None
        if checkpoint_identity is None:
            logger.warning(
                "Ignoring unbound KG checkpoint read for node %s; recomputation is required.",
                node_idx,
            )
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as checkpoint_file:
                envelope = json.load(checkpoint_file)
            if not isinstance(envelope, dict):
                raise ValueError("checkpoint envelope is not an object")
            if envelope.get("schema_version") != KG_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("legacy or unsupported checkpoint schema")
            saved_identity = envelope.get("identity")
            saved_identity_hash = envelope.get("identity_sha256")
            if not isinstance(saved_identity, dict) or not isinstance(
                saved_identity_hash, str
            ):
                raise ValueError("checkpoint identity is incomplete")
            if _sha256_json(saved_identity) != saved_identity_hash:
                raise ValueError("checkpoint identity digest is invalid")
            expected_identity_hash = _sha256_json(checkpoint_identity)
            if saved_identity_hash != expected_identity_hash:
                raise ValueError("checkpoint extraction identity does not match")
            if saved_identity != checkpoint_identity:
                raise ValueError("checkpoint extraction identity is inconsistent")
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload is missing")
            payload_hash = envelope.get("payload_sha256")
            if not isinstance(payload_hash, str):
                raise ValueError("checkpoint payload digest is missing")
            if _sha256_json(payload) != payload_hash:
                raise ValueError("checkpoint payload digest is invalid")
            result = self._payload_to_objects(payload, node_idx)
        except (
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            logger.warning(
                "Ignoring invalid KG checkpoint %s for node %s: %s",
                file_path,
                node_idx,
                error,
            )
            return None

        logger.info("Loaded KG extraction checkpoint from %s.", file_path)
        return result

    def extract_kg(self, node: TreeNode) -> Dict[str, Any]:
        checkpoint_identity = self._checkpoint_identity(node, kind="node")
        tmp_res = self.load_tmp_res(node.index_id, checkpoint_identity)
        if tmp_res is not None:
            return self.res_repair(tmp_res)

        kg = self.extractor.extract(node)
        res = self.res_repair(kg)
        final_identity = self._checkpoint_identity(node, kind="node")
        if final_identity != checkpoint_identity:
            raise KGExtractionError(
                node.index_id,
                self._node_type(node),
                RuntimeError("checkpoint identity changed during extraction"),
            )
        self.save_tmp_res(res, node.index_id, final_identity)
        return res

    def extract_title(
        self,
        node: TreeNode,
        title_path: List[TreeNode],
        sibling_nodes: List[TreeNode],
    ) -> Dict[str, Any]:
        checkpoint_identity = self._checkpoint_identity(
            node,
            kind="title",
            title_path=title_path,
            sibling_nodes=sibling_nodes,
        )
        tmp_res = self.load_tmp_res(node.index_id, checkpoint_identity)
        if tmp_res is not None:
            return self.res_repair(tmp_res)

        kg = self.extractor.extract_title(node, title_path, sibling_nodes)
        logger.info(
            f"Extracted title entities and relations for node {node.index_id}."
        )

        res = self.res_repair(kg)
        final_identity = self._checkpoint_identity(
            node,
            kind="title",
            title_path=title_path,
            sibling_nodes=sibling_nodes,
        )
        if final_identity != checkpoint_identity:
            raise KGExtractionError(
                node.index_id,
                "title",
                RuntimeError("checkpoint identity changed during extraction"),
            )
        self.save_tmp_res(res, node.index_id, final_identity)
        return res

    def iter_extract_kg(
        self,
        nodes: Iterable[TreeNode],
        max_workers: int | None = None,
        max_in_flight: int | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield node results in input order with bounded submission.

        A node whose extraction fails after retries is logged and skipped with an
        empty result so one unreliable node cannot abort the whole document. Its
        checkpoint is not committed, so a later run re-attempts it.
        """

        worker_limit = max_workers or self.cfg_graph.extraction_max_workers
        in_flight_limit = (
            max_in_flight
            if max_in_flight is not None
            else self.cfg_graph.extraction_max_in_flight
        )
        for outcome in iter_bounded_ordered(
            nodes,
            self.extract_kg,
            max_workers=worker_limit,
            max_in_flight=in_flight_limit,
            thread_name_prefix="kg-extraction",
        ):
            if not outcome.ok:
                assert outcome.error is not None
                node = outcome.item
                logger.error(
                    "Skipping KG extraction for node %s after failure: %s",
                    node.index_id,
                    outcome.error,
                )
                yield {"entities": [], "relations": [], "node_idx": node.index_id}
                continue
            yield outcome.unwrap()

    def batch_extract_kg(
        self,
        nodes: Iterable[TreeNode],
        max_workers: int | None = None,
        max_in_flight: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility materialization over :meth:`iter_extract_kg`."""

        return list(
            self.iter_extract_kg(
                nodes,
                max_workers=max_workers,
                max_in_flight=max_in_flight,
            )
        )

    def iter_extract_titles(
        self,
        work_items: Iterable[tuple[TreeNode, List[TreeNode], List[TreeNode]]],
        max_workers: int | None = None,
        max_in_flight: int | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield title results with the same bounded/fail-fast contract."""

        worker_limit = max_workers or self.cfg_graph.extraction_max_workers
        in_flight_limit = (
            max_in_flight
            if max_in_flight is not None
            else self.cfg_graph.extraction_max_in_flight
        )

        def extract(
            item: tuple[TreeNode, List[TreeNode], List[TreeNode]],
        ) -> Dict[str, Any]:
            node, title_path, sibling_nodes = item
            return self.extract_title(node, title_path, sibling_nodes)

        for outcome in iter_bounded_ordered(
            work_items,
            extract,
            max_workers=worker_limit,
            max_in_flight=in_flight_limit,
            thread_name_prefix="kg-title-extraction",
        ):
            if not outcome.ok:
                assert outcome.error is not None
                node = outcome.item[0]
                logger.error(
                    "Skipping KG extraction for title node %s after failure: %s",
                    node.index_id,
                    outcome.error,
                )
                yield {"entities": [], "relations": [], "node_idx": node.index_id}
                continue
            yield outcome.unwrap()

    def batch_extract_titles(
        self,
        nodes: Iterable[TreeNode],
        title_paths: Iterable[List[TreeNode]],
        sibling_nodes_list: Iterable[List[TreeNode]],
        max_workers: int | None = None,
        max_in_flight: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility materialization over :meth:`iter_extract_titles`."""

        return list(
            self.iter_extract_titles(
                zip(nodes, title_paths, sibling_nodes_list),
                max_workers=max_workers,
                max_in_flight=max_in_flight,
            )
        )


if __name__ == "__main__":
    # Test
    from Core.configs.system_config import load_system_config

    config = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml")

    llm = LLM(config.llm)
    vlm = VLM(config.vlm)
    save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    kg_extractor = KGExtractor(config.graph, llm, vlm, save_path=save_path)

    from Core.Index.Tree import DocumentTree

    tmp_path = "/mnt/data/wangshu/mmrag/MMLongBench-Doc/index/011bb5d3-e95e-5fd2-b9a6-0e0f980e7024"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))

    # select node for testing
    # 1. one long text node with more than 1000 characters
    # 2. one table node
    # 3. one image node
    # 4. one equation node
    text_node = None
    image_node = None
    table_node = None
    equation_node = None
    title_node = None

    for node in tree_index.nodes:
        if (
            text_node == None
            and node.type == NodeType.TEXT
            and len(node.meta_info.content) > 1000
        ):
            print(
                f"Selected text node: {node.index_id} with content length {len(node.meta_info.content)}"
            )
            text_node = node
        if image_node == None and node.type == NodeType.IMAGE:
            print(
                f"Selected image node: {node.index_id} with image path {node.meta_info.img_path}"
            )
            image_node = node
        if table_node == None and node.type == NodeType.TABLE:
            print(
                f"Selected table node: {node.index_id} with caption '{node.meta_info.caption}'"
            )
            table_node = node
        if equation_node == None and node.type == NodeType.EQUATION:
            print(
                f"Selected equation node: {node.index_id} with content '{node.meta_info.content}'"
            )
            equation_node = node
        if title_node == None and node.type == NodeType.TITLE:
            print(
                f"Selected title node: {node.index_id} with content '{node.meta_info.content}'"
            )
            title_node = node

    def print_entity_name(res: dict):
        for ent in res.get("entities", []):
            print(
                f"Entity: {ent.entity_name}, Type: {ent.entity_type}, Description: {ent.description[:50]}..."
            )

    # text_kg = kg_extractor.extract_kg(text_node)
    # print("Extracted KG from text node:")
    # print(
    #     f"Extract {len(text_kg['entities'])} entities and {len(text_kg['relations'])} relations."
    # )
    # print_entity_name(text_kg)

    image_kg = kg_extractor.extract_kg(image_node)
    print("Extracted KG from image node:")
    print(
        f"Extract {len(image_kg['entities'])} entities and {len(image_kg['relations'])} relations."
    )
    print_entity_name(image_kg)

    # table_kg = kg_extractor.extract_kg(table_node)
    # print("Extracted KG from table node:")
    # print(
    #     f"Extract {len(table_kg['entities'])} entities and {len(table_kg['relations'])} relations."
    # )
    # print_entity_name(table_kg)

    # equation_kg = kg_extractor.extract_kg(equation_node)
    # print("Extracted KG from equation node:")
    # print(
    #     f"Extract {len(equation_kg['entities'])} entities and {len(equation_kg['relations'])} relations."
    # )
    # print_entity_name(equation_kg)

    # title_kg = kg_extractor.extract_title(
    #     title_node,
    #     tree_index.get_path_from_root(title_node.index_id),
    #     tree_index.get_sibling_nodes(title_node.index_id),
    # )
    # print("Extracted KG from title node:")
    # print(
    #     f"Extract {len(title_kg['entities'])} entities and {len(title_kg['relations'])} relations."
    # )
    # print_entity_name(title_kg)
