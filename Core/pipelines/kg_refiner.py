from xmlrpc.client import Boolean

from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from Core.configs.graph_config import GraphConfig
from Core.configs.vdb_config import VDBConfig
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.rerank import TextRerankerProvider
from Core.Index.Graph import Entity, Relationship, Graph
from Core.prompts.kg_prompt import (
    SUMMARIZE_ENTITY,
    DEFAULT_ENTITY_TYPES,
    MergedEntitySchema,
    ENTITY_RESOLUATION_PROMPT,
    ERExtractSel,
    ER_RERANK_INSTRUCTION,
    DESCRIPTION_SYNTHESIS,
)
from Core.utils.utils import truncate_description
from Core.utils.bounded_executor import iter_bounded_ordered


from collections import defaultdict
from typing import Optional, List
import json
import logging
import gc

log = logging.getLogger(__name__)


class KGRefinementError(RuntimeError):
    """An entity could not be refined without losing evidence."""

    def __init__(self, entity_name: str, error: Exception):
        self.entity_name = entity_name
        self.error = error
        super().__init__(f"KG refinement failed for entity {entity_name!r}: {error}")


class KGRefiner:
    """
    A class to refine knowledge graphs (KG).
    Including the basic and advanced refinement methods.
    For the basic refinement, it merges entities with the same name.
    For the advanced refinement, it performs entity resolution.
    """

    # The separator used to merge entity descriptions
    _DESCRIPTION_SEP_ = "<SEP>"

    def __init__(
        self,
        llm: LLM,
        graph_config: GraphConfig,
        graph_index: Graph,
        save_path: str,
        vdb_config: VDBConfig,
        scope_filter: dict[str, str],
        g: float = 0.6,
    ):
        self.llm = llm
        self.graph_index = graph_index
        self.graph_config = graph_config
        self.scope_filter = dict(scope_filter)
        # Similarity search configuration:
        # `g` is configurable from init; others are fixed defaults.
        self.similar_search_topk = 10
        self.similar_search_distance_threshold = 0.2
        self.similar_search_mink = 1
        self.similar_search_g = g
        log.info(f"KGRefiner initialized with g={g}")

        self.embedder = None
        self.reranker = None
        self.vdb = None
        self.entity_to_vdb_id: dict[str, str] = defaultdict(str)
        self.entity_alias_map: dict[str, str] = defaultdict(str)
        try:
            self.embedder = TextEmbeddingProvider(
                model_name=graph_config.embedding_config.model_name,
                backend=graph_config.embedding_config.backend,
                max_length=graph_config.embedding_config.max_length,
                device=graph_config.embedding_config.device,
                api_base=graph_config.embedding_config.api_base,
                api_key=getattr(graph_config.embedding_config, "api_key", None),
            )
            self.reranker = TextRerankerProvider(
                model_name=graph_config.reranker_config.model_name,
                device=graph_config.reranker_config.device,
                max_length=graph_config.reranker_config.max_length,
                backend=graph_config.reranker_config.backend,
                api_base=graph_config.reranker_config.api_base,
                request_timeout_seconds=(
                    graph_config.reranker_config.request_timeout_seconds
                ),
                use_flash_attention=(
                    graph_config.reranker_config.use_flash_attention
                ),
                flash_attention_fallback=(
                    graph_config.reranker_config.flash_attention_fallback
                ),
            )
            self.vdb = VectorStore(
                embedding_model=self.embedder,
                qdrant_url=vdb_config.qdrant_url,
                collection_name=graph_config.entity_collection_name,
                api_key_env=vdb_config.api_key_env,
                timeout_seconds=vdb_config.timeout_seconds,
                scope_filter=self.scope_filter,
            )
            self.vdb.reset()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """
        Correctly closes all resources used by the KGRefiner, including the
        Embedder, Reranker
        """
        log.info("Closing KGRefiner and all its resources...")

        for resource_name in ("reranker", "vdb", "embedder"):
            resource = getattr(self, resource_name, None)
            if resource is not None and hasattr(resource, "close"):
                try:
                    resource.close()
                except Exception:
                    log.exception("Failed to close KG resource %s", resource_name)
            setattr(self, resource_name, None)

        if hasattr(self, "entity_to_vdb_id"):
            self.entity_to_vdb_id.clear()
        if hasattr(self, "entity_alias_map"):
            self.entity_alias_map.clear()

        log.info("Performing final cleanup...")
        gc.collect()

        log.info("✅ KGRefiner resources closed successfully.")

    def get_latest_entity_name(self, node_name: str) -> str:
        if node_name not in self.entity_alias_map:
            raise ValueError(
                f"Entity name '{node_name}' not found in the alias map. "
                "Please ensure the entity has been processed before."
            )

        path: list[str] = []
        positions: dict[str, int] = {}
        current = node_name
        while True:
            if current in positions:
                cycle_start = positions[current]
                cycle = path[cycle_start:] + [current]
                raise ValueError(
                    "Entity alias cycle detected: " + " -> ".join(cycle)
                )
            positions[current] = len(path)
            path.append(current)

            if current not in self.entity_alias_map:
                raise ValueError(
                    f"Entity alias '{path[-2]}' points to missing entity "
                    f"'{current}'."
                )
            latest_node_name = self.entity_alias_map[current]
            if not latest_node_name:
                raise ValueError(
                    f"Entity alias '{current}' has an empty destination."
                )
            if latest_node_name == current:
                root = current
                break
            current = latest_node_name

        # Union-find-style path compression makes future lookups effectively
        # constant time while avoiding recursion depth failures on long chains.
        for alias in path:
            self.entity_alias_map[alias] = root
        return root

    def _concatenated_merge(self, old_entity: Entity, new_entity: Entity) -> Entity:
        """Merge under the old entity's identity, keeping both descriptions."""

        return Entity(
            entity_name=old_entity.entity_name,
            entity_type=old_entity.entity_type,
            description=(
                old_entity.description
                + self._DESCRIPTION_SEP_
                + new_entity.description
            ),
            source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
        )

    def entity_merge(
        self,
        old_entity: Entity,
        new_entity: Entity,
        merged_to_old_entity: Boolean = False,
    ) -> Entity:
        """
        Merges two entities into one by summarizing their descriptions and updating the graph index.
        Args:
            old_entity (Entity): The old entity to merge.
            new_entity (Entity): The new entity to merge with the old entity.
        Returns:
            Entity: The merged entity with updated description and source IDs.
        """
        old_node_name = self.graph_index.get_node_name_from_entity(old_entity)
        new_node_name = self.graph_index.get_node_name_from_entity(new_entity)
        entity_c: Optional[Entity] = None

        if (old_node_name == new_node_name) or merged_to_old_entity:
            log.info("merged directly")
            merged_entity = self._concatenated_merge(old_entity, new_entity)
        else:
            old_entity_dict = old_entity.model_dump(exclude={"source_ids"})
            old_entity_dict["description"] = truncate_description(
                old_entity_dict["description"], max_words=200
            )

            new_entity_dict = new_entity.model_dump(exclude={"source_ids"})
            new_entity_dict["description"] = truncate_description(
                new_entity_dict["description"], max_words=200
            )

            prompt = SUMMARIZE_ENTITY.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=json.dumps(
                    {
                        "entity_1": old_entity_dict,
                        "entity_2": new_entity_dict,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            try:
                res_entity: MergedEntitySchema = self.llm.get_json_completion(
                    prompt=prompt, schema=MergedEntitySchema
                )
            except Exception as error:
                # Coreference was already decided; only the synthesized name is
                # lost, so keep the merge rather than failing the whole stage.
                log.warning(
                    "Entity summarization failed for '%s' + '%s' (%s); merging directly.",
                    old_entity.entity_name,
                    new_entity.entity_name,
                    type(error).__name__,
                )
                merged_entity = self._concatenated_merge(old_entity, new_entity)
            else:
                log.info("merged by LLM summarization")
                res_entity.entity_name = res_entity.entity_name.lower()
                res_entity.entity_type = res_entity.entity_type.upper()
                res_entity.entity_type = res_entity.entity_type.replace(" ", "_")

                merged_entity = Entity(
                    entity_name=res_entity.entity_name,
                    entity_type=res_entity.entity_type,
                    description=(
                        old_entity.description
                        + self._DESCRIPTION_SEP_
                        + new_entity.description
                    ),
                    source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
                )

                merged_node_name = self.graph_index.get_node_name_from_entity(
                    merged_entity
                )
                if (
                    merged_node_name != old_node_name
                    and merged_node_name in self.graph_index.kg
                ):
                    # The name the LLM chose is already a distinct entity, so
                    # that entity is folded in as well.
                    entity_c = self.graph_index.get_entity_by_node_name(
                        merged_node_name
                    )
                    log.info(
                        f"Entity '{merged_node_name}' already exists in the graph. "
                        "Merging it with the old entity.\n"
                    )
                    log.info(
                        f"Old entity: {old_entity.entity_name} ({old_entity.entity_type}), \n"
                        f"New entity: {new_entity.entity_name} ({new_entity.entity_type}), \n"
                        f"Entity C: {entity_c.entity_name} ({entity_c.entity_type})"
                    )
                    merged_entity.description += (
                        self._DESCRIPTION_SEP_ + entity_c.description
                    )
                    merged_entity.source_ids = set(merged_entity.source_ids).union(
                        entity_c.source_ids
                    )

        # Mutations begin here. Nothing above touches the graph or the vector
        # store, so a failed summarization leaves both consistent.
        self.delete_entity_from_vdb(old_entity)
        if entity_c is not None:
            self.delete_entity_from_vdb(entity_c)
            self.graph_index.update_entity(
                old_entity_name=entity_c.entity_name,
                old_entity_type=entity_c.entity_type,
                new_entity=old_entity,
            )

        self.graph_index.update_entity(
            old_entity_name=old_entity.entity_name,
            old_entity_type=old_entity.entity_type,
            new_entity=merged_entity,
        )

        log.info(
            f"Merged entity '{old_entity.entity_name}' with '{new_entity.entity_name}'. \n"
            f"Old entity type: '{old_entity.entity_type}', \n"
            f"New entity name: '{merged_entity.entity_name}', \n"
            f"New entity type: '{merged_entity.entity_type}', \n"
        )

        merged_node_name = self.graph_index.get_node_name_from_entity(merged_entity)
        self.entity_alias_map[old_node_name] = merged_node_name
        self.entity_alias_map[new_node_name] = merged_node_name
        self.entity_alias_map[merged_node_name] = merged_node_name

        return merged_entity

    def basic_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        Merges entities if they have the same entity name and updates relationships accordingly.
        Args:
            entities (List[Entity]): List of entities to merge.
            relationships (List[Relationship]): List of relationships to update.
            source_id (int): The source ID of this extracted Sub KG.
        """

        # Create a map from original entity name to final entity  (if merged)
        entity_map: dict[str, str] = {}
        add_entity_list = []
        for entity in entities:
            entity_node_name = self.graph_index.get_node_name_from_entity(entity=entity)
            # If the entity is not in the graph index, add it
            # Otherwise, merge it with the existing entity
            if entity_node_name not in self.graph_index.kg:
                self.graph_index.add_and_link(tree_node_id=source_id, entities=entity)
                entity_map[entity.entity_name] = entity
                add_entity_list.append(entity)
            else:
                # Merge with existing entity
                existing_entity = self.graph_index.get_entity(
                    entity.entity_name, entity.entity_type
                )
                merged_entity = self.entity_merge(existing_entity, entity)
                entity_map[existing_entity.entity_name] = merged_entity
                add_entity_list.append(merged_entity)

        # Add the new entities to the vector database
        self.add_entities_to_vdb(add_entity_list)

        # Update relationships
        for rel in relationships:
            if rel.src_entity_name in entity_map:
                rel.src_entity_name = entity_map[rel.src_entity_name].entity_name
                src_type = entity_map[rel.src_entity_name].entity_type
            if rel.tgt_entity_name in entity_map:
                rel.tgt_entity_name = entity_map[rel.tgt_entity_name].entity_name
                tgt_type = entity_map[rel.tgt_entity_name].entity_type
            self.graph_index.add_kg_edge(rel=rel, src_type=src_type, tgt_type=tgt_type)

    def get_vdb_meta_data(self, entity: Entity) -> dict:
        """
        Generates metadata for the entity to be stored in the vector database.
        Args:
            entity (Entity): The entity to generate metadata for.
        Returns:
            dict: The metadata dictionary without source_ids.
            since vdb does not support list type.
        """
        return {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "node_id": self.graph_index.get_node_name_from_entity(entity),
            **self.scope_filter,
        }

    def add_entities_to_vdb(self, entities: List[Entity]) -> None:
        """
        Adds a list of entities to the vector database.
        Args:
            entities (List[Entity]): The list of entities to add to the vector database.
        """
        if not entities:
            return

        # deduplicated entities
        entity_map = {}
        for entity in entities:
            node_name = self.graph_index.get_node_name_from_entity(entity)
            if node_name not in entity_map:
                entity_map[node_name] = entity
            else:
                # If the entity already exists, select longer description
                existing_entity = entity_map[node_name]
                if len(entity.description) > len(existing_entity.description):
                    existing_entity.description = entity.description

        entities = list(entity_map.values())

        embed_texts = []
        metadatas = []
        for ent in entities:
            node_name = self.graph_index.get_node_name_from_entity(ent)
            if node_name in self.entity_to_vdb_id:
                log.info(
                    f"Entity '{node_name}' already exists in the vector database."
                    "Skipping adding it again."
                )
                continue
            embed_texts.append(node_name)
            metadatas.append(self.get_vdb_meta_data(ent))
        if not embed_texts:
            return

        vdbids: List[str] = self.vdb.add_texts(
            texts=embed_texts,
            metadatas=metadatas,
            strict_source_identity=True,
        )
        for embed_text, vdbid in zip(embed_texts, vdbids):
            if embed_text in self.entity_to_vdb_id:
                log.warning(
                    f"Entity '{embed_text}' already exists in the vector database. "
                    "Overwriting the existing entry."
                )
            self.entity_to_vdb_id[embed_text] = vdbid

    def delete_entity_from_vdb(self, old_entity: Entity) -> None:
        """
        Deletes an entity from the vector database.
        Args:
            entity (Entity): The entity to delete from the vector database.
        """
        embed_text = self.graph_index.get_node_name_from_entity(old_entity)
        vdbid = self.entity_to_vdb_id.get(embed_text, None)
        if vdbid is not None:
            self.vdb.delete_text_by_ids(ids=[vdbid])
            del self.entity_to_vdb_id[embed_text]
            log.info(f"delete entity {embed_text} from vector database.")
        else:
            log.info(
                f"Entity '{old_entity.entity_name}' with type '{old_entity.entity_type}' "
                f"not found in the vector database. Cannot delete."
            )
            log.info("this may cause add duplicate entities later.")

    def search_similar_entities(self, entity: Entity) -> List[Entity]:
        """
        Searches for similar entities in the vector database based on the entity's text information.
        This method is the core method for entity resolution.
        1. First retrieval topk Entities from the vector database.
        2. use the reranker to score these Entities
        3. Gradient-based similar Entities selection.
        4. 1) If all the Entities are not similar enough (With low score), return empty list.
        2) If there are some similar Entities, return the gradient-based truncated list (one or more).
        3) If all Entities are selected, return empty list. This means all Entities are similar enough.

        Args:
            entity (Entity): The entity to search for similar entities.
        Returns:
            List[Entity]: A list of similar entities or empty list if none found.
        """
        topk = self.similar_search_topk
        distance_threshold = self.similar_search_distance_threshold
        mink = self.similar_search_mink
        g = self.similar_search_g

        embed_text = self.graph_index.get_node_name_from_entity(entity)
        similar_entities = self.vdb.search(embed_text, top_k=topk)
        min_distance = (
            similar_entities[0]["distance"] if similar_entities else float("inf")
        )
        if min_distance > distance_threshold:
            log.info(
                f"No similar entities found for '{entity.entity_name}' with type '{entity.entity_type}'. "
                f"Minimum distance: {min_distance}, threshold: {distance_threshold}."
            )
            return []

        def metadata_str(meta_data: dict):
            description = meta_data.get("description", "")

            max_words = 1000
            max_chars = 10000

            words = description.split()
            if len(words) > max_words:
                description = " ".join(words[:max_words]) + "..."

            if len(description) > max_chars:
                description = description[:max_chars] + "..."

            entity_str = (
                f"Name: {meta_data.get('entity_name', '')}\n"
                f"Type: {meta_data.get('entity_type', '')}\n"
                f"Description: {description}"
            )
            return entity_str

        similar_entities_str = [
            metadata_str(ent["metadata"]) for ent in similar_entities
        ]

        scores = self.reranker.rerank(
            query=embed_text,
            documents=similar_entities_str,
            instruction=ER_RERANK_INSTRUCTION,
        )
        self.reranker.clean_cache()

        ranked_results = sorted(
            zip(similar_entities, scores), key=lambda x: x[1], reverse=True
        )

        # 4.1 max score < 0.5 not similar enough, return empty list
        if not ranked_results or ranked_results[0][1] < 0.5:
            return []

        # 4.2 gradient-based selection
        # add first min_k entities to the selection list
        sel_entities = ranked_results[:mink]
        score_remain = sel_entities[-1][1]  # the score of the last selected entity

        # add the remaining entities based on the gradient-based selection
        for ent, score in ranked_results[mink:]:
            if score >= score_remain * g:
                sel_entities.append((ent, score))
                score_remain = score
            else:
                break

        if len(sel_entities) == len(ranked_results):
            # 4.3 If all entities are selected, return empty list
            return []

        log.info(f"After gradient-select, found {len(sel_entities)} similar entities.")

        res_entities = []
        for ent, _ in sel_entities:
            entity_name = ent["metadata"].get("entity_name", "")
            entity_type = ent["metadata"].get("entity_type", "")
            res_entities.append(self.graph_index.get_entity(entity_name, entity_type))
        return res_entities

    def _prepare_selection_input(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> str:
        """Formats the entities into the JSON structure required by the prompt."""

        # Give each similar entity a temporary ID (index)
        candidates_with_ids = []
        for i, entity in enumerate(similar_entities):
            entity_dict = entity.model_dump(exclude={"source_ids"})
            if "description" in entity_dict and entity_dict["description"]:
                entity_dict["description"] = truncate_description(
                    entity_dict["description"]
                )
            entity_dict["id"] = i
            candidates_with_ids.append(entity_dict)

        input_data = {
            "new_entity": new_entity.model_dump(exclude={"source_ids"}),
            "candidate_entities": candidates_with_ids,
        }

        return json.dumps(input_data, indent=2, ensure_ascii=False)

    def er_selection_by_llm(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> Optional[Entity]:
        # 1. Prepare the input for the LLM
        input_json_str = self._prepare_selection_input(new_entity, similar_entities)
        prompt = ENTITY_RESOLUATION_PROMPT.format(input_json=input_json_str)

        # 2. Call the LLM
        try:
            res: ERExtractSel = self.llm.get_json_completion(
                prompt=prompt, schema=ERExtractSel
            )
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return None

        # 3. Parse the LLM response
        select_id = res.select_id

        # 4. Return the result
        if select_id == -1:
            log.info(
                f"LLM did not select any similar entity for the entity:\n {new_entity.entity_name} "
            )
            log.info(f"Reason:\n {res.explanation}")
            return None

        if 0 <= select_id < len(similar_entities):
            # Log the selection and reason
            log.info(
                f"LLM selected entity ID: {select_id}, " f"Reason: {res.explanation}"
            )
            # Log the new entity and the selected similar entity
            log.info("New Entity Info:")
            log.info(
                f"Entity Name: {new_entity.entity_name}, Entity Type: {new_entity.entity_type}"
            )
            log.info(f"LLM selected Entity Info:")
            log.info(
                f"Entity Name: {similar_entities[select_id].entity_name}, Entity Type: {similar_entities[select_id].entity_type}"
            )

            return similar_entities[select_id]
        else:
            print(f"Warning: LLM returned an out-of-bounds ID: {select_id}")
            return None

    def entity_resolution(self, new_entity: Entity) -> Entity:
        """
        Resolves the new entity by comparing it with similar entities.
        Merge the new entity with the most similar one if they are true duplicated entity.

        Args:
            new_entity (Entity): The new entity to resolve.
        Returns:
            Entity: The resolved entity, which may be a merged entity or the new entity itself.
        """

        # 1.1 the same entity name and entity type, merge them directly
        node_name = self.graph_index.get_node_name_from_entity(entity=new_entity)
        if node_name in self.graph_index.kg:
            # If the entity already exists in the graph index with the same type, merge them
            existing_entity = self.graph_index.get_entity(
                new_entity.entity_name, new_entity.entity_type
            )

            merged_entity = self.entity_merge(existing_entity, new_entity)
            return merged_entity

        # 1.2 If the entity have a same name with existing entity, but merged before.
        # Merged them directly
        if node_name in self.entity_alias_map.keys():
            latest_entity_name = self.get_latest_entity_name(node_name=node_name)
            log.info(
                f"Entity '{new_entity.entity_name}' with type '{new_entity.entity_type}' "
                f"has been merged before. Merging with the existing entity."
                f"Latest node: {latest_entity_name}"
            )
            existing_entity = self.graph_index.get_entity_by_node_name(
                latest_entity_name
            )

            merged_entity = self.entity_merge(
                existing_entity, new_entity, merged_to_old_entity=True
            )

            return merged_entity

        # 2. search similar entities in the vector database
        similar_entities = self.search_similar_entities(new_entity)
        if len(new_entity.source_ids) != 1:
            raise ValueError(
                f"Expected exactly one source_id, but found {len(new_entity.source_ids)}."
            )

        source_id = next(iter(new_entity.source_ids))

        if len(similar_entities) == 0:
            # 2.1 No similar entities found, add the new entity directly
            self.graph_index.add_and_link(tree_node_id=source_id, entities=new_entity)
            return new_entity

        # 2.2 If similar entities are found, use the LLM to determine if exist one of them is the same entity as the new one.
        sel_existing_entity = self.er_selection_by_llm(
            new_entity=new_entity, similar_entities=similar_entities
        )
        if sel_existing_entity is None:
            # If no similar entity is selected, add the new entity directly
            self.graph_index.add_and_link(tree_node_id=source_id, entities=new_entity)
            return new_entity
        else:
            # If a similar entity is selected, merge the new entity with it
            merged_entity: Entity = self.entity_merge(sel_existing_entity, new_entity)

            return merged_entity

    def process_unknown_entities(
        self, unknown_entities: List[Entity], entity_map: dict[str, Entity]
    ) -> dict[str, Entity]:
        log.info(f"Processing unknown entities, length: {len(unknown_entities)}")
        if unknown_entities:
            unknown_vdb_entities = []
            for entity in unknown_entities:
                # Perform entity resolution for unknown entities
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                unknown_vdb_entities.append(new_entity)
            # Add the resolved unknown entities to the vector database
            self.add_entities_to_vdb(unknown_vdb_entities)
        return entity_map

    def process_relationships(
        self, relationships: List[Relationship], entity_map: dict[str, Entity]
    ) -> None:
        """
        Processes relationships by updating source and target entity names based on the entity map.
        And adds them to the graph index.
        Args:
            relationships (List[Relationship]): List of relationships to process.
            entity_map (dict[str, Entity]): Map of old entity names to new entities.
        """
        for k, v in entity_map.items():
            node_name = self.graph_index.get_node_name_from_entity(v)
            if node_name not in self.graph_index.kg:
                new_node_name = self.get_latest_entity_name(node_name=node_name)
                entity_map[k] = self.graph_index.get_entity_by_node_name(new_node_name)
                log.info(
                    f"Entity '{v.entity_name}' with type '{v.entity_type}' not found in the graph index. "
                    f"Using the latest entity '{new_node_name}' instead."
                )

        for rel in relationships:
            old_src_name = rel.src_entity_name
            old_tgt_name = rel.tgt_entity_name
            src_type = None
            tgt_type = None
            if old_src_name in entity_map:
                rel.src_entity_name = entity_map[old_src_name].entity_name
                src_type = entity_map[old_src_name].entity_type
            if old_tgt_name in entity_map:
                rel.tgt_entity_name = entity_map[old_tgt_name].entity_name
                tgt_type = entity_map[old_tgt_name].entity_type
            if src_type is None or tgt_type is None:
                log.info(
                    f"Relationship {rel} has missing entity types. "
                    "Skipping this relationship."
                )
                continue
            else:
                self.graph_index.add_kg_edge(
                    rel=rel, src_type=src_type, tgt_type=tgt_type
                )

    def _debug_check_num(self):
        num_node_graph = len(self.graph_index.kg.nodes())
        num_node_vdb = self.vdb.count()
        if num_node_graph != num_node_vdb:
            log.warning(
                f"Number of nodes in the graph index ({num_node_graph}) "
                f"does not match the number of nodes in the vector database ({num_node_vdb})."
            )
            print("warning here")
        else:
            log.info(
                f"graph and vdb contain the same number of nodes: {num_node_graph}."
            )

    def advanced_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        Refines the knowledge graph by advanced entity resolution and relationship updates.
        Args:
            entities (List[Entity]): List of entities to refine.
            relationships (List[Relationship]): List of relationships to update.
            source_id (int): The source ID of this extracted tree node.
        """
        log.info(
            f"--------------------\n"
            f"Starting advanced knowledge graph refinement for source ID: {source_id}\n"
            f"with {len(entities)} entities and {len(relationships)} relationships."
        )

        # map the old entity name to the new entity name after resolution
        entity_map: dict[str, Entity] = {}

        # 1. for the first time to refine the KG, the vector database and graph index are initialized.
        if self.vdb.count() <= 10:
            # If the vector database is empty or has very few entities, we can skip entity resolution.
            # Not entity resolution for normal entities.

            add_entities = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type != "UNKNOWN":
                    # For normal entities, we can add them directly to the vector database and graph index.
                    add_entities.append(entity)
                    entity_map[entity.entity_name] = entity
                else:
                    unknown_entities.append(entity)

            # add to vdb and graph
            self.add_entities_to_vdb(entities)
            self.graph_index.add_and_link(tree_node_id=source_id, entities=entities)

            # For unknown entities, we need to resoluation them
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # Update relationships based on the entity map
            self.process_relationships(relationships, entity_map)
        else:
            # 2. For each entity, perform resolution and update the graph index.
            new_entity_list = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type == "UNKNOWN":
                    # 2.1 for unknown entity type, perform resolution
                    # For unknown entities, we need to resolve them later.
                    unknown_entities.append(entity)
                    continue

                # 2.2 for other entity types, perform resolution
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                new_entity_list.append(new_entity)

            # 2.3 Add the resolved entities to the vector database
            # Since the ER should not be performed within the same chunk
            # The new entities should not be in the vector database yet.
            self.add_entities_to_vdb(new_entity_list)

            # 2.4 Address the unknown entities
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # 3. Update relationships based on the resolved entities
            self.process_relationships(
                relationships=relationships, entity_map=entity_map
            )

        # for debug check the number of nodes in graph and vdb
        self._debug_check_num()

    def _synthesize_entity_description(self, entity: Entity) -> str:
        """Run the model only; graph and vector-store changes stay serialized."""

        log.info(
            f"Refining entity description for {entity.entity_name} of type {entity.entity_type}."
        )
        json_entity = entity.model_dump(exclude={"source_ids"})
        prompt = DESCRIPTION_SYNTHESIS.format(
            input_json=json.dumps(json_entity, indent=2, ensure_ascii=False)
        )
        refined_description = self.llm.get_completion(
            prompt=prompt, json_response=False
        )
        if not refined_description or not refined_description.strip():
            raise ValueError("LLM returned an empty entity description")
        return refined_description

    def _apply_refined_description(
        self, entity: Entity, refined_description: str
    ) -> Entity:
        previous_entity = entity.model_copy(deep=True)
        node_name = self.graph_index.get_node_name_from_entity(entity)
        vdb_id = self.entity_to_vdb_id.get(node_name)
        entity.description = refined_description
        graph_updated = False
        try:
            self.graph_index.update_entity(
                old_entity_name=entity.entity_name,
                old_entity_type=entity.entity_type,
                new_entity=entity,
            )
            graph_updated = True
            if vdb_id is None:
                # A missing mapping is repaired once. The embedding input is the
                # stable node name; normal refinement only updates metadata.
                self.add_entities_to_vdb([entity])
            else:
                self.vdb.update_text_metadatas(
                    ids=[vdb_id],
                    metadatas=[self.get_vdb_meta_data(entity)],
                )
        except Exception:
            entity.description = previous_entity.description
            if graph_updated:
                try:
                    self.graph_index.update_entity(
                        old_entity_name=previous_entity.entity_name,
                        old_entity_type=previous_entity.entity_type,
                        new_entity=previous_entity,
                    )
                except Exception:
                    log.exception(
                        "Failed to roll back graph state for entity %s",
                        previous_entity.entity_name,
                    )
            raise
        log.info(
            f"Entity {entity.entity_name} description refined successfully."
        )
        return entity

    def refine_entity_description(self, entity: Entity) -> Entity:
        """Compatibility wrapper for one fail-fast, serialized refinement."""

        try:
            description = self._synthesize_entity_description(entity)
            return self._apply_refined_description(entity, description)
        except KGRefinementError:
            raise
        except Exception as error:
            raise KGRefinementError(entity.entity_name, error) from error

    def refine_entities(self):
        merged_entity_set = set()

        def entities_to_refine():
            for node_name in self.entity_alias_map.keys():
                latest_entity_name = self.get_latest_entity_name(node_name)
                if latest_entity_name in merged_entity_set:
                    continue
                merged_entity_set.add(latest_entity_name)
                entity = self.graph_index.get_entity_by_node_name(latest_entity_name)
                if self._DESCRIPTION_SEP_ in entity.description:
                    yield entity

        refined_count = 0
        outcomes = iter_bounded_ordered(
            entities_to_refine(),
            self._synthesize_entity_description,
            max_workers=self.graph_config.refinement_max_workers,
            max_in_flight=self.graph_config.refinement_max_in_flight,
            thread_name_prefix="kg-refinement",
        )
        for outcome in outcomes:
            if not outcome.ok:
                assert outcome.error is not None
                raise KGRefinementError(
                    outcome.item.entity_name, outcome.error
                ) from outcome.error
            try:
                self._apply_refined_description(outcome.item, outcome.unwrap())
            except KGRefinementError:
                raise
            except Exception as error:
                raise KGRefinementError(
                    outcome.item.entity_name, error
                ) from error
            refined_count += 1
        if refined_count == 0:
            log.info("No entities need to be refined.")
            return
        log.info(
            "Refined %d entity descriptions and synchronized vector metadata.",
            refined_count,
        )
        self._debug_check_num()
        return

    def refine_relation(self):
        # delete self loop in graph index
        self.graph_index.remove_self_loops()
