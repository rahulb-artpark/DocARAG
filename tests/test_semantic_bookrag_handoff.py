from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from Core.Index.Tree import DocumentTree, NodeType
from Core.pipelines import vdb_index
from Core.pipelines.pdf_refiner import merge_tables_and_mark_invalid
from Core.pipelines.tree_node_builder import create_node_by_type
from docarag.bookrag_projection import project_canonical_blocks
from docarag.canonical_artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
)
from docarag.domain import (
    BlockType,
    BoundingBox,
    CoordinateOrigin,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    ParserBlockObservation,
    ParserProvenance,
    Representation,
    RepresentationKind,
    SemanticElement,
    SemanticElementKind,
    SemanticSourceSelector,
    SemanticStructureKind,
    SemanticSubstructure,
    SourceGeometry,
    content_sha256,
)
from docarag.reconciliation import reconcile_observations
from docarag.evidence_hydration import (
    CanonicalEvidenceHydrator,
    EvidenceHydrationError,
)


DOCUMENT_VERSION_ID = "semantic-bookrag-document-v1"


def _semantic_receipt(structure) -> dict[str, object]:
    return {
        "substructure_id": structure.substructure_id,
        "semantic_sha256": structure.semantic_sha256,
        "comparison_sha256": structure.comparison_sha256,
        "source_observation_ids": list(structure.source_observation_ids),
        "source_substructure_ids": list(structure.source_substructure_ids),
    }


def _semantic_observation(
    *,
    page_index: int,
    native_id: str,
    block_type: BlockType,
    representation_kind: RepresentationKind,
    representation_content: object,
    structure_kind: SemanticStructureKind,
    element_kind: SemanticElementKind,
    element_content: str | None = None,
) -> tuple[ParserBlockObservation, SemanticSubstructure]:
    representation = Representation(
        kind=representation_kind,
        content=representation_content,  # type: ignore[arg-type]
        media_type=(
            "application/json"
            if representation_kind is RepresentationKind.TABLE_GRID
            else "text/plain"
        ),
        confidence=0.95,
    )
    selector = SemanticSourceSelector(
        representation_id=representation.representation_id,
        native_selector=f"/semantic/{native_id}/elements/0",
    )
    element_kwargs: dict[str, object] = {}
    if element_kind is SemanticElementKind.TABLE_CELL:
        element_kwargs.update(
            row_index=0,
            column_index=0,
            row_span=1,
            column_span=1,
            role="column_header",
        )
    element = SemanticElement(
        local_key=f"{native_id}-element",
        kind=element_kind,
        source_selectors=(selector,),
        content=(
            element_content
            if element_content is not None
            else {
                SemanticElementKind.TABLE_CELL: "Metric",
                SemanticElementKind.FORMULA_SYMBOL: "x",
                SemanticElementKind.DIAGRAM_ELEMENT: "Input",
            }[element_kind]
        ),
        **element_kwargs,  # type: ignore[arg-type]
    )
    raw_structure = SemanticSubstructure(
        kind=structure_kind,
        elements=(element,),
    )
    observation = ParserBlockObservation(
        document_version_id=DOCUMENT_VERSION_ID,
        page_id=PageRef(DOCUMENT_VERSION_ID, page_index).page_id,
        page_index=page_index,
        provenance=ParserProvenance(
            parser_name="semantic-fixture",
            parser_version="1",
            model_version="semantic-fixture-1",
            config_sha256=content_sha256({"fixture": native_id}),
            parser_native_id=native_id,
            raw_artifact_uri=f"artifact://semantic/{native_id}.json",
        ),
        block_type=block_type,
        representations=(representation,),
        geometry=SourceGeometry(
            coordinate_space=CoordinateSpace.NORMALIZED,
            coordinate_origin=CoordinateOrigin.TOP_LEFT,
            bbox=BoundingBox(0.1, 0.1, 0.9, 0.4),
        ),
        reading_order=page_index + 1,
        confidence=0.95,
        semantic_substructures=(raw_structure,),
    )
    return observation, raw_structure


class SemanticBookRAGHandoffTests(unittest.TestCase):
    def _projected_cases(self):
        cases = (
            dict(
                page_index=0,
                native_id="table",
                block_type=BlockType.TABLE,
                representation_kind=RepresentationKind.TABLE_GRID,
                representation_content={"cells": [["Metric"]]},
                structure_kind=SemanticStructureKind.TABLE,
                element_kind=SemanticElementKind.TABLE_CELL,
            ),
            dict(
                page_index=1,
                native_id="formula",
                block_type=BlockType.EQUATION,
                representation_kind=RepresentationKind.LATEX,
                representation_content=r"x^2",
                structure_kind=SemanticStructureKind.FORMULA,
                element_kind=SemanticElementKind.FORMULA_SYMBOL,
            ),
            dict(
                page_index=2,
                native_id="diagram",
                block_type=BlockType.DIAGRAM,
                representation_kind=RepresentationKind.VISUAL_DESCRIPTION,
                representation_content="Input node",
                structure_kind=SemanticStructureKind.DIAGRAM,
                element_kind=SemanticElementKind.DIAGRAM_ELEMENT,
            ),
        )
        observations = []
        raw_by_observation_id = {}
        for case in cases:
            observation, raw_structure = _semantic_observation(**case)
            observations.append(observation)
            raw_by_observation_id[observation.observation_id] = raw_structure
        canonical_blocks = reconcile_observations(tuple(observations)).canonical_blocks
        document = DocumentRef(
            document_id="semantic-bookrag-document",
            document_version_id=DOCUMENT_VERSION_ID,
            content_sha256="7" * 64,
            mime_type="application/pdf",
            source_uri="file:///semantic-bookrag.pdf",
            page_count=3,
        )
        rows = project_canonical_blocks(document, canonical_blocks)
        return document, canonical_blocks, raw_by_observation_id, rows

    def test_projected_graphs_survive_node_pickle_and_retrieval(self) -> None:
        document, blocks, raw_by_observation_id, rows = self._projected_cases()

        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "file_name": "semantic-bookrag.pdf",
                    "source_uri": document.source_uri,
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            node_ids: list[int] = []
            for row in rows:
                node = create_node_by_type(row, isTitle=False)
                tree.add_node(node)
                tree.root_node.add_child(node)
                node_ids.append(node.index_id)
                self.assertEqual(
                    row["semantic_substructures"],
                    node.meta_info.semantic_substructures,
                )
                source_ref = node.meta_info.source_evidence_refs[0]
                self.assertNotIn("semantic_substructures", source_ref)
                self.assertEqual(
                    row["source_evidence_refs"][0][
                        "semantic_substructure_refs"
                    ],
                    source_ref["semantic_substructure_refs"],
                )

            tree.save_to_file()
            restored = DocumentTree.load_from_file(
                DocumentTree.get_save_path(directory)
            )
            retrieval_by_block = {
                item["canonical_block_id"]: item
                for item in restored.get_nodes_data(node_ids)
            }

        expected_node_types = {
            BlockType.TABLE: NodeType.TABLE,
            BlockType.EQUATION: NodeType.EQUATION,
            BlockType.DIAGRAM: NodeType.IMAGE,
        }
        for block in blocks:
            with self.subTest(block_type=block.block_type.value):
                retrieved = retrieval_by_block[block.block_id]
                expected_structure = block.semantic_substructures[0]
                payload = retrieved["semantic_substructures"][0]
                self.assertEqual(expected_node_types[block.block_type], retrieved["type"])
                self.assertEqual(expected_structure.to_dict(), payload)
                self.assertEqual(block.block_id, retrieved["canonical_block_id"])
                self.assertEqual(block.revision_id, retrieved["canonical_revision_id"])
                self.assertEqual(
                    list(block.source_observation_ids),
                    payload["source_observation_ids"],
                )
                raw_structure = raw_by_observation_id[
                    block.source_observation_ids[0]
                ]
                self.assertEqual(
                    [raw_structure.substructure_id],
                    payload["source_substructure_ids"],
                )
                self.assertEqual(
                    expected_structure.substructure_id,
                    payload["substructure_id"],
                )
                source_ref = retrieved["source_evidence_refs"][0]
                self.assertNotIn("semantic_substructures", source_ref)
                self.assertEqual(
                    _semantic_receipt(expected_structure),
                    source_ref["semantic_substructure_refs"][0],
                )

    def test_shared_metadata_path_covers_every_legacy_node_branch(self) -> None:
        _document, _blocks, _raw, rows = self._projected_cases()
        semantic_payload = rows[0]["semantic_substructures"]
        source_refs = rows[0]["source_evidence_refs"]
        cases = (
            ({"type": "text", "text": "body"}, False, NodeType.TEXT),
            (
                {"type": "text", "text": "heading", "text_level": 1},
                True,
                NodeType.TITLE,
            ),
            ({"type": "image", "image_caption": ["image"]}, False, NodeType.IMAGE),
            ({"type": "table", "table_body": "<table></table>"}, False, NodeType.TABLE),
            ({"type": "equation", "text": "x"}, False, NodeType.EQUATION),
            ({"type": "future", "text": "fallback"}, False, NodeType.TEXT),
        )
        for pdf_id, (base, is_title, expected_type) in enumerate(cases, start=1):
            with self.subTest(expected_type=expected_type, is_title=is_title):
                row = {
                    **base,
                    "pdf_id": pdf_id,
                    "page_idx": 0,
                    "semantic_substructures": semantic_payload,
                    "source_evidence_refs": source_refs,
                }
                node = create_node_by_type(row, isTitle=is_title)
                self.assertEqual(expected_type, node.type)
                self.assertEqual(semantic_payload, node.meta_info.semantic_substructures)
                self.assertEqual(source_refs, node.meta_info.source_evidence_refs)

    def test_derived_table_keeps_two_exact_graphs_instead_of_claiming_one(self) -> None:
        document, blocks, _raw, rows = self._projected_cases()
        target = deepcopy(next(row for row in rows if row["type"] == "table"))
        continuation_observation, _raw_structure = _semantic_observation(
            page_index=3,
            native_id="table-continuation",
            block_type=BlockType.TABLE,
            representation_kind=RepresentationKind.TABLE_GRID,
            representation_content={"cells": [["Value"]]},
            structure_kind=SemanticStructureKind.TABLE,
            element_kind=SemanticElementKind.TABLE_CELL,
        )
        continuation_block = reconcile_observations(
            (continuation_observation,)
        ).canonical_blocks[0]
        expanded_document = DocumentRef(
            document_id=document.document_id,
            document_version_id=document.document_version_id,
            content_sha256=document.content_sha256,
            mime_type=document.mime_type,
            source_uri=document.source_uri,
            page_count=4,
        )
        continuation = project_canonical_blocks(
            expanded_document, (continuation_block,)
        )[0]
        expected_source_graphs = [
            deepcopy(target["semantic_substructures"]),
            deepcopy(continuation["semantic_substructures"]),
        ]
        # Explicit compatibility input: the immediately preceding compact
        # receipt schema lacked comparison_sha256. Refinement must upgrade it
        # from the matching full top-level graph, never guess it.
        for source_ref in (
            target["source_evidence_refs"][0],
            continuation["source_evidence_refs"][0],
        ):
            source_ref["semantic_substructure_refs"][0].pop(
                "comparison_sha256"
            )
        expected_source_receipts = [
            [
                _semantic_receipt(source_blocks_structure)
                for source_blocks_structure in source.semantic_substructures
            ]
            for source in (
                next(
                    block
                    for block in blocks
                    if block.block_type is BlockType.TABLE
                ),
                continuation_block,
            )
        ]

        merge_tables_and_mark_invalid(target, continuation)

        self.assertEqual(2, len(target["source_evidence_refs"]))
        self.assertEqual(
            expected_source_receipts,
            [
                ref["semantic_substructure_refs"]
                for ref in target["source_evidence_refs"]
            ],
        )
        self.assertTrue(
            all(
                "semantic_substructures" not in ref
                for ref in target["source_evidence_refs"]
            )
        )
        expected_union = sorted(
            [*expected_source_graphs[0], *expected_source_graphs[1]],
            key=lambda item: item["substructure_id"],
        )
        self.assertEqual(expected_union, target["semantic_substructures"])
        self.assertEqual(
            2,
            len(
                {
                    item["source_substructure_ids"][0]
                    for item in target["semantic_substructures"]
                }
            ),
        )

        source_blocks = (
            next(block for block in blocks if block.block_type is BlockType.TABLE),
            continuation_block,
        )
        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "source_uri": expanded_document.source_uri,
                    "source_sha256": expanded_document.content_sha256,
                    "document_version_id": expanded_document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            node = create_node_by_type(target, isTitle=False)
            tree.add_node(node)
            tree.root_node.add_child(node)
            record = tree.get_nodes_data([node.index_id])[0]
            artifact = Path(directory) / "derived-canonical-evidence.json"
            artifact.write_text(
                json.dumps(
                    {
                        "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                        "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
                        "document": expanded_document.to_dict(),
                        "parse_request_id": "semantic-derived-test",
                        "block_count": len(source_blocks),
                        "canonical_blocks": [
                            block.to_dict() for block in source_blocks
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            hydrated = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda _ids: [deepcopy(record)],
            ).hydrate([deepcopy(record)])[0]

        self.assertEqual(
            [item["substructure_id"] for item in expected_union],
            [
                item["substructure_id"]
                for item in hydrated["semantic_substructures"]
            ],
        )
        self.assertEqual(2, hydrated["canonical_source_count"])

    def test_hydration_retains_graphs_and_rejects_tampering_or_duplication(self) -> None:
        document, blocks, _raw, rows = self._projected_cases()
        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "source_uri": document.source_uri,
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            for row in rows:
                node = create_node_by_type(row, isTitle=False)
                tree.add_node(node)
                tree.root_node.add_child(node)
            records = tree.get_nodes_data(
                [node.index_id for node in tree.get_nodes()]
            )
            artifact = Path(directory) / "canonical-evidence.json"
            artifact.write_text(
                json.dumps(
                    {
                        "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                        "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
                        "document": document.to_dict(),
                        "parse_request_id": "semantic-handoff-test",
                        "block_count": len(blocks),
                        "canonical_blocks": [block.to_dict() for block in blocks],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            hydrator = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda ids: [
                    deepcopy(records[index - 1]) for index in ids
                ],
            )
            hydrated = hydrator.hydrate(deepcopy(records))
            self.assertEqual(
                [row["semantic_substructures"] for row in rows],
                [item["semantic_substructures"] for item in hydrated],
            )

            selected_structure = blocks[0].semantic_substructures[0]
            selected_element = selected_structure.elements[0]
            semantic_candidate = deepcopy(records[0])
            semantic_candidate.update(
                {
                    "semantic_substructure_id": selected_structure.substructure_id,
                    "semantic_structure_kind": selected_structure.kind.value,
                    "semantic_sha256": selected_structure.semantic_sha256,
                    "semantic_comparison_sha256": (
                        selected_structure.comparison_sha256
                    ),
                    "semantic_element_id": selected_element.element_id,
                    "semantic_element_kind": selected_element.kind.value,
                    "semantic_element_content_sha256": content_sha256(
                        selected_element.content
                    ),
                }
            )
            selected = hydrator.hydrate([semantic_candidate])[0]
            self.assertEqual(
                selected_element.to_dict(), selected["selected_semantic_element"]
            )
            self.assertEqual(
                selected_structure.to_dict(),
                selected["selected_semantic_substructure"],
            )
            tampered_selector = deepcopy(semantic_candidate)
            tampered_selector["semantic_element_content_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                EvidenceHydrationError, "mismatched field"
            ):
                hydrator.hydrate([tampered_selector])

            legacy_tree_record = deepcopy(records[0])
            legacy_tree_record["source_evidence_refs"][0][
                "semantic_substructure_refs"
            ][0].pop("comparison_sha256")
            legacy_hydrator = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda _ids: [legacy_tree_record],
            )
            legacy_candidate = deepcopy(records[0])
            legacy_candidate.pop("source_evidence_refs")
            self.assertEqual(
                records[0]["semantic_substructures"],
                legacy_hydrator.hydrate([legacy_candidate])[0][
                    "semantic_substructures"
                ],
            )

            bad_comparison_tree = deepcopy(records[0])
            bad_comparison_tree["source_evidence_refs"][0][
                "semantic_substructure_refs"
            ][0]["comparison_sha256"] = "0" * 64
            bad_comparison_hydrator = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda _ids: [bad_comparison_tree],
            )
            candidate = deepcopy(records[0])
            candidate.pop("source_evidence_refs")
            with self.assertRaisesRegex(
                EvidenceHydrationError, "semantic_substructure_refs"
            ):
                bad_comparison_hydrator.hydrate([candidate])

            tampered_tree = deepcopy(records[0])
            tampered_tree["semantic_substructures"][0]["metadata"] = {
                "tampered": True
            }
            tampered_hydrator = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda _ids: [tampered_tree],
            )
            candidate = deepcopy(records[0])
            candidate.pop("semantic_substructures")
            with self.assertRaisesRegex(
                EvidenceHydrationError, "semantic substructures"
            ):
                tampered_hydrator.hydrate([candidate])

            duplicated_tree = deepcopy(records[0])
            duplicated_tree["source_evidence_refs"][0][
                "semantic_substructures"
            ] = deepcopy(duplicated_tree["semantic_substructures"])
            duplicated_hydrator = CanonicalEvidenceHydrator.from_artifact(
                artifact,
                node_resolver=lambda _ids: [duplicated_tree],
            )
            candidate = deepcopy(records[0])
            candidate.pop("source_evidence_refs")
            with self.assertRaisesRegex(EvidenceHydrationError, "duplicates"):
                duplicated_hydrator.hydrate([candidate])

    def test_vector_metadata_keeps_only_bounded_semantic_receipts(self) -> None:
        large_label = "diagram-label-" + ("x" * 60_000)
        observation, _raw_structure = _semantic_observation(
            page_index=0,
            native_id="large-diagram",
            block_type=BlockType.DIAGRAM,
            representation_kind=RepresentationKind.VISUAL_DESCRIPTION,
            representation_content="Large workflow node",
            structure_kind=SemanticStructureKind.DIAGRAM,
            element_kind=SemanticElementKind.DIAGRAM_ELEMENT,
            element_content=large_label,
        )
        block = reconcile_observations((observation,)).canonical_blocks[0]
        document = DocumentRef(
            document_id="large-semantic-vector-document",
            document_version_id=DOCUMENT_VERSION_ID,
            content_sha256="8" * 64,
            mime_type="application/pdf",
            source_uri="file:///large-semantic-vector.pdf",
            page_count=1,
        )
        row = project_canonical_blocks(document, (block,))[0]
        full_graph_bytes = len(
            json.dumps(row["semantic_substructures"], sort_keys=True).encode("utf-8")
        )
        self.assertGreater(full_graph_bytes, 50_000)

        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            node = create_node_by_type(row, isTitle=False)
            tree.add_node(node)
            tree.root_node.add_child(node)
            records = list(vdb_index.iter_tree_index_records(tree))

        self.assertEqual(2, len(records))
        metadata = records[0].metadata
        metadata_bytes = len(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        self.assertLess(metadata_bytes, 2_048)
        self.assertNotIn("semantic_substructures", metadata)
        self.assertNotIn("source_evidence_refs", metadata)
        self.assertEqual(1, metadata["semantic_substructure_count"])
        compact_refs = json.loads(metadata["semantic_substructure_refs_json"])
        self.assertEqual(
            [
                {
                    "substructure_id": block.semantic_substructures[0].substructure_id,
                    "semantic_sha256": block.semantic_substructures[0].semantic_sha256,
                    "comparison_sha256": block.semantic_substructures[
                        0
                    ].comparison_sha256,
                }
            ],
            compact_refs,
        )
        self.assertEqual(
            hashlib.sha256(
                metadata["semantic_substructure_refs_json"].encode("utf-8")
            ).hexdigest(),
            metadata["semantic_substructure_refs_sha256"],
        )
        self.assertTrue(
            all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
        )
        semantic_record = records[1]
        self.assertEqual("text", semantic_record.modality)
        self.assertEqual(
            block.semantic_substructures[0].substructure_id,
            semantic_record.metadata["semantic_substructure_id"],
        )
        self.assertEqual(
            block.semantic_substructures[0].elements[0].element_id,
            semantic_record.metadata["semantic_element_id"],
        )
        self.assertLessEqual(
            len(semantic_record.value),
            vdb_index.MAX_SEMANTIC_ELEMENT_EMBED_TEXT_CHARS,
        )

        # A tree written during the short-lived duplicated-reference shape is
        # still readable, but its full source-reference graph is never copied
        # into vector metadata.
        legacy_row = deepcopy(row)
        legacy_source_ref = legacy_row["source_evidence_refs"][0]
        legacy_source_ref.pop("semantic_substructure_refs")
        legacy_source_ref["semantic_substructures"] = deepcopy(
            legacy_row["semantic_substructures"]
        )
        with tempfile.TemporaryDirectory() as directory:
            legacy_tree = DocumentTree(
                meta_dict={
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            legacy_node = create_node_by_type(legacy_row, isTitle=False)
            legacy_tree.add_node(legacy_node)
            legacy_tree.root_node.add_child(legacy_node)
            legacy_metadata = list(
                vdb_index.iter_tree_index_records(legacy_tree)
            )[0].metadata
        self.assertEqual(
            metadata["semantic_substructure_refs_json"],
            legacy_metadata["semantic_substructure_refs_json"],
        )
        self.assertNotIn("semantic_substructures", legacy_metadata)
        self.assertNotIn("source_evidence_refs", legacy_metadata)

        legacy_compact_row = deepcopy(row)
        legacy_compact_row["source_evidence_refs"][0][
            "semantic_substructure_refs"
        ][0].pop("comparison_sha256")
        with tempfile.TemporaryDirectory() as directory:
            legacy_compact_tree = DocumentTree(
                meta_dict={
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            legacy_compact_node = create_node_by_type(
                legacy_compact_row, isTitle=False
            )
            legacy_compact_tree.add_node(legacy_compact_node)
            legacy_compact_tree.root_node.add_child(legacy_compact_node)
            upgraded_metadata = list(
                vdb_index.iter_tree_index_records(legacy_compact_tree)
            )[0].metadata
        self.assertEqual(
            metadata["semantic_substructure_refs_json"],
            upgraded_metadata["semantic_substructure_refs_json"],
        )

        mismatched_row = deepcopy(row)
        mismatched_row["source_evidence_refs"][0][
            "semantic_substructure_refs"
        ][0]["comparison_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            mismatched_tree = DocumentTree(
                meta_dict={
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            mismatched_node = create_node_by_type(
                mismatched_row, isTitle=False
            )
            mismatched_tree.add_node(mismatched_node)
            mismatched_tree.root_node.add_child(mismatched_node)
            with self.assertRaisesRegex(
                vdb_index.VectorIndexBindingError,
                "differ from source evidence receipts",
            ):
                list(vdb_index.iter_tree_index_records(mismatched_tree))

        # A syntactically valid digest carried only by a compact source receipt
        # is not proof of semantic truth.  V3 must verify it against a full
        # graph before putting it into retrieval metadata.
        source_only_row = deepcopy(row)
        source_only_row.pop("semantic_substructures")
        source_only_row["source_evidence_refs"][0][
            "semantic_substructure_refs"
        ][0]["comparison_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            source_only_tree = DocumentTree(
                meta_dict={
                    "source_sha256": document.content_sha256,
                    "document_version_id": document.document_version_id,
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            source_only_node = create_node_by_type(
                source_only_row, isTitle=False
            )
            source_only_tree.add_node(source_only_node)
            source_only_tree.root_node.add_child(source_only_node)
            with self.assertRaisesRegex(
                vdb_index.VectorIndexBindingError,
                "source-only compact digests are unverified",
            ):
                list(vdb_index.iter_tree_index_records(source_only_tree))

    def test_vector_semantic_receipt_limit_fails_before_index_write(self) -> None:
        receipts = [
            {
                "substructure_id": f"sstruct:{index:064x}",
                "semantic_sha256": f"{index + 1:064x}",
                "comparison_sha256": f"{index + 2:064x}",
                "source_observation_ids": [f"observation:{index:064x}"],
                "source_substructure_ids": [f"sstruct:{index + 1:064x}"],
            }
            for index in range(vdb_index.MAX_VECTOR_SEMANTIC_REFS + 1)
        ]
        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={
                    "source_sha256": "9" * 64,
                    "document_version_id": "receipt-limit-document",
                },
                cfg=SimpleNamespace(save_path=directory),
            )
            node = create_node_by_type(
                {
                    "type": "text",
                    "text": "bounded receipts",
                    "pdf_id": 1,
                    "page_idx": 0,
                    "source_evidence_refs": [
                        {"semantic_substructure_refs": receipts}
                    ],
                },
                isTitle=False,
            )
            tree.add_node(node)
            tree.root_node.add_child(node)
            with self.assertRaisesRegex(
                vdb_index.VectorIndexBindingError,
                "semantic reference limit",
            ):
                list(vdb_index.iter_tree_index_records(tree))

    def test_tree_unpickled_from_pre_field_state_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tree = DocumentTree(
                meta_dict={"source_sha256": "a" * 64},
                cfg=SimpleNamespace(save_path=directory),
            )
            node = create_node_by_type(
                {"type": "text", "text": "legacy", "pdf_id": 1, "page_idx": 0},
                isTitle=False,
            )
            tree.add_node(node)
            tree.root_node.add_child(node)

            # Simulate the instance dictionary stored by a pickle created before
            # MetaInfo gained the optional semantic_substructures field.
            node.meta_info.__dict__.pop("semantic_substructures", None)
            path = Path(DocumentTree.get_save_path(directory))
            with path.open("wb") as handle:
                pickle.dump(tree, handle)

            restored = DocumentTree.load_from_file(str(path))
            retrieval = restored.get_nodes_data([node.index_id])[0]

        self.assertEqual("legacy", retrieval["content"])
        self.assertNotIn("semantic_substructures", retrieval)


if __name__ == "__main__":
    unittest.main()
