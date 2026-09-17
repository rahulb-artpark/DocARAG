from __future__ import annotations

import unittest

from document_services.evidence.projections.bookrag import project_canonical_blocks
from document_services.domain import (
    BlockType,
    BoundingBox,
    CanonicalBlock,
    ContractValidationError,
    CoordinateOrigin,
    CoordinateSpace,
    DocumentRef,
    MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE,
    MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
    MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE,
    PageRef,
    ParserBlockObservation,
    ParserProvenance,
    QualityStatus,
    Representation,
    RepresentationKind,
    SemanticElement,
    SemanticElementKind,
    SemanticProducerKind,
    SemanticRelation,
    SemanticRelationKind,
    SemanticSourceSelector,
    SemanticStructureKind,
    SemanticStructureAvailability,
    SemanticSubstructure,
    SourceGeometry,
    content_sha256,
)
from document_services.parsing.reconciliation import reconcile_observations


DOCUMENT_VERSION = "semantic-document-version"


def _geometry(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    origin: CoordinateOrigin = CoordinateOrigin.TOP_LEFT,
) -> SourceGeometry:
    return SourceGeometry(
        coordinate_space=CoordinateSpace.NORMALIZED,
        bbox=BoundingBox(x0, y0, x1, y1),
        coordinate_origin=origin,
    )


def _representation(
    content: object = "A B",
    *,
    kind: RepresentationKind = RepresentationKind.TEXT,
) -> Representation:
    return Representation(
        kind=kind,
        content=content,  # type: ignore[arg-type]
        media_type=(
            "application/json"
            if kind
            in {
                RepresentationKind.TABLE_GRID,
                RepresentationKind.DIAGRAM_GRAPH,
                RepresentationKind.STRUCTURED_DATA,
            }
            else "text/plain"
        ),
        confidence=0.95,
    )


def _observation(
    parser: str,
    native_id: str,
    *,
    representation: Representation,
    structure: SemanticSubstructure | None,
    block_type: BlockType = BlockType.TABLE,
) -> ParserBlockObservation:
    return ParserBlockObservation(
        document_version_id=DOCUMENT_VERSION,
        page_id=PageRef(DOCUMENT_VERSION, 0).page_id,
        page_index=0,
        provenance=ParserProvenance(
            parser_name=parser,
            parser_version="1",
            model_version=f"{parser}-model",
            config_sha256=content_sha256({"parser": parser}),
            parser_native_id=native_id,
            raw_artifact_uri=f"artifact://{parser}/{native_id}.json",
        ),
        block_type=block_type,
        representations=(representation,),
        geometry=_geometry(0.1, 0.1, 0.8, 0.5),
        reading_order=1,
        confidence=0.95,
        semantic_substructures=() if structure is None else (structure,),
    )


def _table_structure(representation: Representation) -> SemanticSubstructure:
    first_source = SemanticSourceSelector(
        representation_id=representation.representation_id,
        native_selector="/table_cells/0",
        geometry=_geometry(0.1, 0.1, 0.4, 0.2),
    )
    second_source = SemanticSourceSelector(
        representation_id=representation.representation_id,
        native_selector="/table_cells/1",
        geometry=_geometry(0.4, 0.1, 0.8, 0.2),
    )
    first = SemanticElement(
        local_key="r0c0",
        kind=SemanticElementKind.TABLE_CELL,
        source_selectors=(first_source,),
        content="Quarter",
        role="column_header",
        row_index=0,
        column_index=0,
        row_span=2,
        column_span=1,
        confidence=0.97,
    )
    second = SemanticElement(
        local_key="r0c1",
        kind=SemanticElementKind.TABLE_CELL,
        source_selectors=(second_source,),
        content="Revenue",
        role="column_header",
        row_index=0,
        column_index=1,
        row_span=1,
        column_span=1,
        confidence=0.96,
    )
    return SemanticSubstructure(
        kind=SemanticStructureKind.TABLE,
        elements=(second, first),
        relations=(
            SemanticRelation(
                kind=SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                source_element_id=first.element_id,
                target_element_id=second.element_id,
                source_selector_ids=(
                    second_source.selector_id,
                    first_source.selector_id,
                ),
                confidence=0.94,
            ),
        ),
        metadata={"row_count": 2, "column_count": 2},
    )


def _vendor_table_structure(
    representation: Representation,
    *,
    vendor: str,
    selector_root: str,
    producer_kind: SemanticProducerKind,
    changed_content: bool = False,
    omit_last_relation: bool = False,
) -> SemanticSubstructure:
    values = (("Quarter", "Revenue"), ("Q1", "99" if changed_content else "42"))
    elements: list[SemanticElement] = []
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            selector = SemanticSourceSelector(
                representation_id=representation.representation_id,
                native_selector=f"{selector_root}/{row_index}/{column_index}",
                metadata={
                    "vendor": vendor,
                    "native_coordinate": f"{row_index}:{column_index}",
                },
            )
            elements.append(
                SemanticElement(
                    local_key=(
                        f"r{row_index}c{column_index}"
                        if vendor == "docling"
                        else f"cell-{row_index}-{column_index}"
                    ),
                    kind=SemanticElementKind.TABLE_CELL,
                    source_selectors=(selector,),
                    content=value,
                    role="column_header" if row_index == 0 else "body",
                    row_index=row_index,
                    column_index=column_index,
                    row_span=1,
                    column_span=1,
                    confidence=0.99 if vendor == "docling" else 0.87,
                    metadata={
                        "vendor": vendor,
                        "native_cell_id": f"{vendor}-{row_index}-{column_index}",
                    },
                )
            )
    by_position = {
        (element.row_index, element.column_index): element for element in elements
    }
    relations: list[SemanticRelation] = []
    for row_index in range(2):
        left = by_position[(row_index, 0)]
        right = by_position[(row_index, 1)]
        relations.append(
            SemanticRelation(
                kind=SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                source_element_id=left.element_id,
                target_element_id=right.element_id,
                source_selector_ids=(
                    left.source_selectors[0].selector_id,
                    right.source_selectors[0].selector_id,
                ),
                confidence=0.98 if vendor == "docling" else 0.82,
                metadata={"vendor": vendor, "native_edge": f"right-{row_index}"},
            )
        )
    for column_index in range(2):
        above = by_position[(0, column_index)]
        below = by_position[(1, column_index)]
        relations.append(
            SemanticRelation(
                kind=SemanticRelationKind.TABLE_ADJACENT_BELOW,
                source_element_id=above.element_id,
                target_element_id=below.element_id,
                source_selector_ids=(
                    above.source_selectors[0].selector_id,
                    below.source_selectors[0].selector_id,
                ),
                confidence=0.97 if vendor == "docling" else 0.81,
                metadata={"vendor": vendor, "native_edge": f"below-{column_index}"},
            )
        )
    if omit_last_relation:
        relations.pop()
    return SemanticSubstructure(
        kind=SemanticStructureKind.TABLE,
        elements=tuple(elements),
        relations=tuple(relations),
        producer_kind=producer_kind,
        source_field_paths=(
            "data.table_cells" if vendor == "docling" else "table_grid.rows",
        ),
        metadata=(
            {
                "row_count": 2,
                "column_count": 2,
                "parser": "docling",
                "normalizer": "docling-table.v1",
            }
            if vendor == "docling"
            else {
                "num_rows": 2,
                "num_cols": 2,
                "parser": "mineru",
                "normalizer": "mineru-grid.v3",
            }
        ),
    )


def _two_element_structure(
    representation: Representation,
    *,
    structure_kind: SemanticStructureKind,
    element_kind: SemanticElementKind,
    relation_kind: SemanticRelationKind,
    normalized_edge_type: str | None = None,
) -> SemanticSubstructure:
    left_source = SemanticSourceSelector(
        representation_id=representation.representation_id,
        native_selector="/elements/left",
    )
    right_source = SemanticSourceSelector(
        representation_id=representation.representation_id,
        native_selector="/elements/right",
    )
    left = SemanticElement(
        local_key="left",
        kind=element_kind,
        source_selectors=(left_source,),
        content="left",
        geometry=_geometry(0.1, 0.1, 0.3, 0.2),
    )
    right = SemanticElement(
        local_key="right",
        kind=element_kind,
        source_selectors=(right_source,),
        content="right",
        geometry=_geometry(0.5, 0.1, 0.7, 0.2),
    )
    return SemanticSubstructure(
        kind=structure_kind,
        elements=(left, right),
        relations=(
            SemanticRelation(
                kind=relation_kind,
                source_element_id=left.element_id,
                target_element_id=right.element_id,
                source_selector_ids=(
                    left_source.selector_id,
                    right_source.selector_id,
                ),
                label="edge",
                metadata=(
                    {"normalized_edge_type": normalized_edge_type}
                    if normalized_edge_type is not None
                    else {}
                ),
            ),
        ),
    )


class SemanticSubstructureContractTests(unittest.TestCase):
    def test_table_round_trip_reconciliation_and_projection_preserve_lineage(self) -> None:
        representation = _representation(
            {"cells": [["Quarter", "Revenue"]]},
            kind=RepresentationKind.TABLE_GRID,
        )
        raw_structure = _table_structure(representation)
        observation = _observation(
            "mineru", "table-1", representation=representation, structure=raw_structure
        )

        restored_observation = ParserBlockObservation.from_json(observation.to_json())
        self.assertEqual(observation, restored_observation)
        self.assertEqual(observation.content_sha256, restored_observation.content_sha256)

        block = reconcile_observations((observation,)).canonical_blocks[0]
        self.assertEqual(1, len(block.semantic_substructures))
        canonical_structure = block.semantic_substructures[0]
        self.assertEqual((observation.observation_id,), canonical_structure.source_observation_ids)
        self.assertEqual((raw_structure.substructure_id,), canonical_structure.source_substructure_ids)
        self.assertEqual(raw_structure.semantic_sha256, canonical_structure.semantic_sha256)
        self.assertEqual(block, CanonicalBlock.from_json(block.to_json()))

        document = DocumentRef(
            document_id="semantic-document",
            document_version_id=DOCUMENT_VERSION,
            content_sha256="7" * 64,
            mime_type="application/pdf",
            source_uri="file:///semantic.pdf",
            page_count=1,
        )
        row = project_canonical_blocks(document, (block,))[0]
        self.assertEqual(
            [canonical_structure.substructure_id],
            row["canonical_semantic_substructure_ids"],
        )
        self.assertEqual(
            canonical_structure.to_dict(), row["semantic_substructures"][0]
        )
        source_ref = row["source_evidence_refs"][0]
        self.assertNotIn("semantic_substructures", source_ref)
        self.assertEqual(
            {
                "substructure_id": canonical_structure.substructure_id,
                "semantic_sha256": canonical_structure.semantic_sha256,
                "comparison_sha256": canonical_structure.comparison_sha256,
                "source_observation_ids": list(
                    canonical_structure.source_observation_ids
                ),
                "source_substructure_ids": list(
                    canonical_structure.source_substructure_ids
                ),
            },
            source_ref["semantic_substructure_refs"][0],
        )

    def test_all_semantic_families_have_typed_elements_and_edges(self) -> None:
        representation = _representation("x y")
        cases = (
            (
                SemanticStructureKind.FORMULA,
                SemanticElementKind.FORMULA_SYMBOL,
                SemanticRelationKind.FORMULA_SUPERSCRIPT_OF,
            ),
            (
                SemanticStructureKind.DIAGRAM,
                SemanticElementKind.DIAGRAM_ELEMENT,
                SemanticRelationKind.DIAGRAM_CONNECTS_TO,
            ),
            (
                SemanticStructureKind.FLOWCHART,
                SemanticElementKind.FLOWCHART_ELEMENT,
                SemanticRelationKind.FLOW_BRANCH_TRUE,
            ),
            (
                SemanticStructureKind.LAYOUT,
                SemanticElementKind.LAYOUT_REGION,
                SemanticRelationKind.READING_NEXT,
            ),
        )
        for structure_kind, element_kind, relation_kind in cases:
            with self.subTest(kind=structure_kind):
                structure = _two_element_structure(
                    representation,
                    structure_kind=structure_kind,
                    element_kind=element_kind,
                    relation_kind=relation_kind,
                )
                self.assertEqual(
                    structure, SemanticSubstructure.from_json(structure.to_json())
                )

    def test_producer_partial_availability_and_field_paths_are_explicit(self) -> None:
        representation = _representation("x")
        selector = SemanticSourceSelector(
            representation_id=representation.representation_id,
            char_start=0,
            char_end=1,
        )
        element = SemanticElement(
            local_key="x",
            kind=SemanticElementKind.FORMULA_SYMBOL,
            source_selectors=(selector,),
            content="x",
        )
        with self.assertRaisesRegex(ContractValidationError, "source_field_paths"):
            SemanticSubstructure(
                kind=SemanticStructureKind.FORMULA,
                elements=(element,),
            )
        structure = SemanticSubstructure(
            kind=SemanticStructureKind.FORMULA,
            elements=(element,),
            producer_kind=SemanticProducerKind.MODEL_DERIVED,
            availability=SemanticStructureAvailability.PARTIAL,
            source_field_paths=("representations[latex].characters[0:1]",),
        )
        restored = SemanticSubstructure.from_json(structure.to_json())
        self.assertEqual(SemanticProducerKind.MODEL_DERIVED, restored.producer_kind)
        self.assertEqual(
            SemanticStructureAvailability.PARTIAL, restored.availability
        )
        partial_observation = _observation(
            "formula-model",
            "partial-formula",
            representation=representation,
            structure=structure,
            block_type=BlockType.EQUATION,
        )
        partial_block = reconcile_observations(
            (partial_observation,)
        ).canonical_blocks[0]
        self.assertEqual(QualityStatus.DEGRADED, partial_block.quality_status)
        with self.assertRaisesRegex(ContractValidationError, "at least one element"):
            SemanticSubstructure(
                kind=SemanticStructureKind.FORMULA,
                elements=(),
                source_field_paths=("representations[latex]",),
            )

    def test_spatial_above_respects_both_coordinate_origins(self) -> None:
        representation = _representation("regions")
        for origin, source_y, target_y in (
            (CoordinateOrigin.TOP_LEFT, 0.1, 0.7),
            (CoordinateOrigin.BOTTOM_LEFT, 0.7, 0.1),
        ):
            with self.subTest(origin=origin):
                source_selector = SemanticSourceSelector(
                    representation_id=representation.representation_id,
                    native_selector=f"/{origin.value}/source",
                )
                target_selector = SemanticSourceSelector(
                    representation_id=representation.representation_id,
                    native_selector=f"/{origin.value}/target",
                )
                source = SemanticElement(
                    local_key="source",
                    kind=SemanticElementKind.LAYOUT_REGION,
                    source_selectors=(source_selector,),
                    geometry=_geometry(
                        0.1,
                        source_y,
                        0.2,
                        source_y + 0.05,
                        origin=origin,
                    ),
                )
                target = SemanticElement(
                    local_key="target",
                    kind=SemanticElementKind.LAYOUT_REGION,
                    source_selectors=(target_selector,),
                    geometry=_geometry(
                        0.1,
                        target_y,
                        0.2,
                        target_y + 0.05,
                        origin=origin,
                    ),
                )
                structure = SemanticSubstructure(
                    kind=SemanticStructureKind.LAYOUT,
                    elements=(source, target),
                    relations=(
                        SemanticRelation(
                            kind=SemanticRelationKind.SPATIAL_ABOVE,
                            source_element_id=source.element_id,
                            target_element_id=target.element_id,
                            source_selector_ids=(
                                source_selector.selector_id,
                                target_selector.selector_id,
                            ),
                        ),
                    ),
                )
                self.assertEqual(
                    SemanticRelationKind.SPATIAL_ABOVE,
                    structure.relations[0].kind,
                )

    def test_one_root_field_path_covers_more_than_256_exact_table_selectors(self) -> None:
        representation = _representation(
            {"cells": [[str(index) for index in range(257)]]},
            kind=RepresentationKind.TABLE_GRID,
        )
        elements = tuple(
            SemanticElement(
                local_key=f"cell-{index}",
                kind=SemanticElementKind.TABLE_CELL,
                source_selectors=(
                    SemanticSourceSelector(
                        representation_id=representation.representation_id,
                        native_selector=f"/table/cells/0/{index}",
                    ),
                ),
                content=str(index),
                row_index=0,
                column_index=index,
                row_span=1,
                column_span=1,
            )
            for index in range(257)
        )
        structure = SemanticSubstructure(
            kind=SemanticStructureKind.TABLE,
            elements=elements,
            source_field_paths=("/table/cells",),
        )
        self.assertEqual(("/table/cells",), structure.source_field_paths)
        self.assertEqual(257, len(structure.elements))

    def test_identical_parser_graphs_merge_but_disagreement_remains_separate(self) -> None:
        representation = _representation(
            {"cells": [["A", "B"]]}, kind=RepresentationKind.TABLE_GRID
        )
        structure = _table_structure(representation)
        docling = _observation(
            "docling", "table-a", representation=representation, structure=structure
        )
        mineru = _observation(
            "mineru", "table-b", representation=representation, structure=structure
        )
        merged = reconcile_observations((mineru, docling)).canonical_blocks[0]
        self.assertEqual(1, len(merged.semantic_substructures))
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            set(merged.semantic_substructures[0].source_observation_ids),
        )
        self.assertEqual(
            (structure.substructure_id,),
            merged.semantic_substructures[0].source_substructure_ids,
        )

        changed_element = SemanticElement(
            local_key="only",
            kind=SemanticElementKind.TABLE_CELL,
            source_selectors=(
                SemanticSourceSelector(
                    representation_id=representation.representation_id,
                    native_selector="/changed",
                ),
            ),
            content="changed",
            row_index=0,
            column_index=0,
            row_span=1,
            column_span=1,
        )
        changed = SemanticSubstructure(
            kind=SemanticStructureKind.TABLE, elements=(changed_element,)
        )
        paddle = _observation(
            "paddleocr-vl",
            "table-c",
            representation=representation,
            structure=changed,
        )
        disagreed = reconcile_observations((docling, paddle)).canonical_blocks[0]
        self.assertEqual(2, len(disagreed.semantic_substructures))
        self.assertEqual(QualityStatus.NEEDS_REVIEW, disagreed.quality_status)
        self.assertEqual(
            "table",
            disagreed.metadata["semantic_substructure_conflicts"][0]["kind"],
        )

        layout = _two_element_structure(
            representation,
            structure_kind=SemanticStructureKind.LAYOUT,
            element_kind=SemanticElementKind.LAYOUT_REGION,
            relation_kind=SemanticRelationKind.READING_NEXT,
        )
        layout_observation = _observation(
            "layout-parser",
            "layout-a",
            representation=representation,
            structure=layout,
        )
        different_kinds = reconcile_observations(
            (docling, layout_observation)
        ).canonical_blocks[0]
        self.assertEqual(QualityStatus.ACCEPTED, different_kinds.quality_status)
        self.assertNotIn(
            "semantic_substructure_conflicts", different_kinds.metadata
        )

    def test_source_neutral_digest_merges_realistic_cross_parser_tables(self) -> None:
        docling_representation = Representation(
            kind=RepresentationKind.TABLE_GRID,
            content={
                "num_rows": 2,
                "num_cols": 2,
                "table_cells": [
                    {"text": "Quarter", "row": 0, "column": 0},
                    {"text": "Revenue", "row": 0, "column": 1},
                    {"text": "Q1", "row": 1, "column": 0},
                    {"text": "42", "row": 1, "column": 1},
                ],
            },
            media_type="application/json",
            confidence=0.99,
            metadata={"vendor": "docling", "source_field": "data"},
        )
        mineru_representation = Representation(
            kind=RepresentationKind.TABLE_GRID,
            content={"rows": [["Quarter", "Revenue"], ["Q1", "42"]]},
            media_type="application/json",
            confidence=0.88,
            metadata={"vendor": "mineru", "source_field": "table_grid"},
        )
        docling_structure = _vendor_table_structure(
            docling_representation,
            vendor="docling",
            selector_root="/table_cells",
            producer_kind=SemanticProducerKind.PARSER_NATIVE,
        )
        mineru_structure = _vendor_table_structure(
            mineru_representation,
            vendor="mineru",
            selector_root="/rows",
            producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
        )

        self.assertNotEqual(
            docling_representation.representation_id,
            mineru_representation.representation_id,
        )
        self.assertNotEqual(
            docling_structure.semantic_sha256,
            mineru_structure.semantic_sha256,
        )
        self.assertNotEqual(
            docling_structure.substructure_id,
            mineru_structure.substructure_id,
        )
        self.assertEqual(
            docling_structure.comparison_sha256,
            mineru_structure.comparison_sha256,
        )
        self.assertEqual(
            docling_structure.to_comparison_dict(),
            mineru_structure.to_comparison_dict(),
        )
        restored = SemanticSubstructure.from_json(docling_structure.to_json())
        self.assertEqual(
            docling_structure.comparison_sha256, restored.comparison_sha256
        )
        legacy_payload = docling_structure.to_dict()
        legacy_payload.pop("comparison_sha256")
        self.assertEqual(
            docling_structure,
            SemanticSubstructure.from_dict(legacy_payload),
        )
        tampered = docling_structure.to_dict()
        tampered["comparison_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ContractValidationError, "comparison_sha256 mismatch"
        ):
            SemanticSubstructure.from_dict(tampered)

        docling = _observation(
            "docling",
            "table-native",
            representation=docling_representation,
            structure=docling_structure,
        )
        mineru = _observation(
            "mineru",
            "table-grid",
            representation=mineru_representation,
            structure=mineru_structure,
        )
        result = reconcile_observations((mineru, docling))
        self.assertEqual(1, len(result.canonical_blocks))
        self.assertFalse(result.issues)
        merged = result.canonical_blocks[0]
        self.assertEqual(QualityStatus.ACCEPTED, merged.quality_status)
        self.assertEqual(1, len(merged.semantic_substructures))
        self.assertNotIn("semantic_substructure_conflicts", merged.metadata)
        match = merged.metadata["match_evidence"][0]
        self.assertTrue(match["semantic_structure_override"])
        self.assertTrue(match["representation_content_conflict"])
        self.assertEqual("same_type_semantic_structure", match["reason"])

        canonical_structure = merged.semantic_substructures[0]
        self.assertEqual(
            docling_structure.comparison_sha256,
            canonical_structure.comparison_sha256,
        )
        # The full graph remains one exact source assertion.  Selectors from
        # the other equivalent parser are not laundered into it.
        self.assertEqual(1, len(canonical_structure.source_substructure_ids))
        selector_vendors = {
            selector.metadata["vendor"]
            for element in canonical_structure.elements
            for selector in element.source_selectors
        }
        self.assertEqual(1, len(selector_vendors))

        equivalence = merged.metadata[
            "semantic_substructure_equivalence_sources"
        ][canonical_structure.substructure_id]
        assertions = equivalence["source_assertions"]
        self.assertEqual(
            {docling.observation_id, mineru.observation_id},
            {item["observation_id"] for item in assertions},
        )
        self.assertEqual(
            {docling_structure.substructure_id, mineru_structure.substructure_id},
            {item["substructure_id"] for item in assertions},
        )
        self.assertEqual(
            {docling_structure.semantic_sha256, mineru_structure.semantic_sha256},
            {item["semantic_sha256"] for item in assertions},
        )
        self.assertEqual(
            {False, True},
            {item["represented_by_canonical_graph"] for item in assertions},
        )

    def test_source_neutral_digest_keeps_changed_content_and_topology_separate(self) -> None:
        docling_representation = _representation(
            {"vendor": "docling", "table": "same logical grid"},
            kind=RepresentationKind.TABLE_GRID,
        )
        mineru_representation = _representation(
            {"vendor": "mineru", "table": "same logical grid"},
            kind=RepresentationKind.TABLE_GRID,
        )
        base = _vendor_table_structure(
            docling_representation,
            vendor="docling",
            selector_root="/table_cells",
            producer_kind=SemanticProducerKind.PARSER_NATIVE,
        )
        changed_content = _vendor_table_structure(
            mineru_representation,
            vendor="mineru",
            selector_root="/rows",
            producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
            changed_content=True,
        )
        changed_topology = _vendor_table_structure(
            mineru_representation,
            vendor="mineru",
            selector_root="/rows",
            producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
            omit_last_relation=True,
        )
        self.assertNotEqual(
            base.comparison_sha256, changed_content.comparison_sha256
        )
        self.assertNotEqual(
            base.comparison_sha256, changed_topology.comparison_sha256
        )

        docling = _observation(
            "docling",
            "base-table",
            representation=docling_representation,
            structure=base,
        )
        for native_id, changed in (
            ("changed-content", changed_content),
            ("changed-topology", changed_topology),
        ):
            with self.subTest(native_id=native_id):
                other = _observation(
                    "mineru",
                    native_id,
                    representation=mineru_representation,
                    structure=changed,
                )
                reconciled = reconcile_observations((docling, other))
                self.assertEqual(1, len(reconciled.canonical_blocks))
                block = reconciled.canonical_blocks[0]
                self.assertEqual(2, len(block.semantic_substructures))
                self.assertEqual(QualityStatus.NEEDS_REVIEW, block.quality_status)
                conflict = block.metadata["semantic_substructure_conflicts"][0]
                self.assertEqual("table", conflict["kind"])
                self.assertTrue(
                    all(
                        assertion["comparison_sha256"]
                        for assertion in conflict["observation_assertions"]
                    )
                )

    def test_explicit_diagram_edge_type_disagreement_requires_review(self) -> None:
        representation = _representation(
            {
                "nodes": [
                    {"id": "manager", "label": "Manager"},
                    {"id": "report", "label": "Report"},
                ],
                "edges": [["manager", "report"]],
            },
            kind=RepresentationKind.DIAGRAM_GRAPH,
        )
        depends_on = _two_element_structure(
            representation,
            structure_kind=SemanticStructureKind.DIAGRAM,
            element_kind=SemanticElementKind.DIAGRAM_ELEMENT,
            relation_kind=SemanticRelationKind.DIAGRAM_CONNECTS_TO,
            normalized_edge_type="depends_on",
        )
        reports_to = _two_element_structure(
            representation,
            structure_kind=SemanticStructureKind.DIAGRAM,
            element_kind=SemanticElementKind.DIAGRAM_ELEMENT,
            relation_kind=SemanticRelationKind.DIAGRAM_CONNECTS_TO,
            normalized_edge_type="reports_to",
        )
        self.assertNotEqual(
            depends_on.comparison_sha256, reports_to.comparison_sha256
        )

        reconciled = reconcile_observations(
            (
                _observation(
                    "docling",
                    "depends-on",
                    representation=representation,
                    structure=depends_on,
                    block_type=BlockType.DIAGRAM,
                ),
                _observation(
                    "mineru",
                    "reports-to",
                    representation=representation,
                    structure=reports_to,
                    block_type=BlockType.DIAGRAM,
                ),
            )
        )
        self.assertEqual(1, len(reconciled.canonical_blocks))
        block = reconciled.canonical_blocks[0]
        self.assertEqual(QualityStatus.NEEDS_REVIEW, block.quality_status)
        self.assertEqual(2, len(block.semantic_substructures))
        self.assertEqual(
            "diagram",
            block.metadata["semantic_substructure_conflicts"][0]["kind"],
        )

    def test_character_selectors_cannot_escape_or_target_structured_content(self) -> None:
        text = _representation("abc")
        beyond = SemanticSubstructure(
            kind=SemanticStructureKind.FORMULA,
            elements=(
                SemanticElement(
                    local_key="beyond",
                    kind=SemanticElementKind.FORMULA_SYMBOL,
                    source_selectors=(
                        SemanticSourceSelector(
                            representation_id=text.representation_id,
                            native_selector="/text",
                            char_start=0,
                            char_end=4,
                        ),
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ContractValidationError, "exceeds"):
            _observation(
                "docling",
                "char-beyond",
                representation=text,
                structure=beyond,
                block_type=BlockType.EQUATION,
            )

        structured = _representation(
            {"symbol": "x"}, kind=RepresentationKind.STRUCTURED_DATA
        )
        non_string = SemanticSubstructure(
            kind=SemanticStructureKind.FORMULA,
            elements=(
                SemanticElement(
                    local_key="structured",
                    kind=SemanticElementKind.FORMULA_SYMBOL,
                    source_selectors=(
                        SemanticSourceSelector(
                            representation_id=structured.representation_id,
                            native_selector="/symbol",
                            char_start=0,
                            char_end=1,
                        ),
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ContractValidationError, "string representation"):
            _observation(
                "mineru",
                "char-structured",
                representation=structured,
                structure=non_string,
                block_type=BlockType.EQUATION,
            )

    def test_invalid_spans_kinds_endpoints_and_selectors_fail_closed(self) -> None:
        representation = _representation("cell")
        selector = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector="/cell",
        )
        with self.assertRaisesRegex(ContractValidationError, "positive row/column spans"):
            SemanticElement(
                local_key="bad-span",
                kind=SemanticElementKind.TABLE_CELL,
                source_selectors=(selector,),
                row_index=0,
                column_index=0,
                row_span=0,
                column_span=1,
            )
        diagram_element = SemanticElement(
            local_key="diagram",
            kind=SemanticElementKind.DIAGRAM_ELEMENT,
            source_selectors=(selector,),
        )
        with self.assertRaisesRegex(ContractValidationError, "incompatible element"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE, elements=(diagram_element,)
            )

        huge = SemanticElement(
            local_key="huge",
            kind=SemanticElementKind.TABLE_CELL,
            source_selectors=(selector,),
            row_index=0,
            column_index=0,
            row_span=MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE + 1,
            column_span=1,
        )
        with self.assertRaisesRegex(ContractValidationError, "position limit"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE, elements=(huge,)
            )

        overlap_a = SemanticElement(
            local_key="overlap-a",
            kind=SemanticElementKind.TABLE_CELL,
            source_selectors=(selector,),
            row_index=0,
            column_index=0,
            row_span=1,
            column_span=2,
        )
        overlap_b = SemanticElement(
            local_key="overlap-b",
            kind=SemanticElementKind.TABLE_CELL,
            source_selectors=(selector,),
            row_index=0,
            column_index=1,
            row_span=1,
            column_span=1,
        )
        with self.assertRaisesRegex(ContractValidationError, "overlapping cell spans"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE,
                elements=(overlap_a, overlap_b),
            )

        gap_source = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector="/gap",
        )
        gap = SemanticElement(
            local_key="gap",
            kind=SemanticElementKind.TABLE_CELL,
            source_selectors=(gap_source,),
            row_index=0,
            column_index=3,
            row_span=1,
            column_span=1,
        )
        with self.assertRaisesRegex(ContractValidationError, "contradicts"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE,
                elements=(overlap_a, gap),
                relations=(
                    SemanticRelation(
                        kind=SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                        source_element_id=overlap_a.element_id,
                        target_element_id=gap.element_id,
                        source_selector_ids=(
                            selector.selector_id,
                            gap_source.selector_id,
                        ),
                    ),
                ),
            )

        table = _table_structure(representation)
        first = table.elements[0]
        with self.assertRaisesRegex(ContractValidationError, "endpoint is outside"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE,
                elements=table.elements,
                relations=(
                    SemanticRelation(
                        kind=SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                        source_element_id=first.element_id,
                        target_element_id="selem_" + "0" * 32,
                        source_selector_ids=(selector.selector_id,),
                    ),
                ),
            )

        left_source = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector="/layout/left",
        )
        right_source = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector="/layout/right",
        )
        left = SemanticElement(
            local_key="layout-left",
            kind=SemanticElementKind.LAYOUT_REGION,
            source_selectors=(left_source,),
            geometry=_geometry(0.1, 0.1, 0.2, 0.2),
        )
        right = SemanticElement(
            local_key="layout-right",
            kind=SemanticElementKind.LAYOUT_REGION,
            source_selectors=(right_source,),
            geometry=_geometry(0.8, 0.1, 0.9, 0.2),
        )
        with self.assertRaisesRegex(ContractValidationError, "contradicts"):
            SemanticSubstructure(
                kind=SemanticStructureKind.LAYOUT,
                elements=(left, right),
                relations=(
                    SemanticRelation(
                        kind=SemanticRelationKind.SPATIAL_RIGHT_OF,
                        source_element_id=left.element_id,
                        target_element_id=right.element_id,
                        source_selector_ids=(
                            left_source.selector_id,
                            right_source.selector_id,
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(ContractValidationError, "selector is outside"):
            SemanticSubstructure(
                kind=SemanticStructureKind.TABLE,
                elements=table.elements,
                relations=(
                    SemanticRelation(
                        kind=SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                        source_element_id=table.elements[0].element_id,
                        target_element_id=table.elements[1].element_id,
                        source_selector_ids=(
                            SemanticSourceSelector(native_selector="/foreign").selector_id,
                        ),
                    ),
                ),
            )

    def test_foreign_material_and_lineage_laundering_fail_closed(self) -> None:
        representation = _representation("owned")
        foreign = _representation("foreign")
        element = SemanticElement(
            local_key="foreign",
            kind=SemanticElementKind.FORMULA_SYMBOL,
            source_selectors=(
                SemanticSourceSelector(
                    representation_id=foreign.representation_id,
                    native_selector="/symbol",
                ),
            ),
        )
        structure = SemanticSubstructure(
            kind=SemanticStructureKind.FORMULA, elements=(element,)
        )
        with self.assertRaisesRegex(ContractValidationError, "outside its observation"):
            _observation(
                "docling",
                "foreign",
                representation=representation,
                structure=structure,
                block_type=BlockType.EQUATION,
            )

        valid = _two_element_structure(
            representation,
            structure_kind=SemanticStructureKind.FORMULA,
            element_kind=SemanticElementKind.FORMULA_SYMBOL,
            relation_kind=SemanticRelationKind.FORMULA_NEXT,
        )
        observation = _observation(
            "docling",
            "valid",
            representation=representation,
            structure=valid,
            block_type=BlockType.EQUATION,
        )
        bound = valid.with_source_lineage(
            source_observation_ids=(observation.observation_id,)
        )
        with self.assertRaisesRegex(ContractValidationError, "must be unbound"):
            _observation(
                "docling",
                "laundered",
                representation=representation,
                structure=bound,
                block_type=BlockType.EQUATION,
            )
        outsider = "obs_" + "0" * 32
        outsider_bound = valid.with_source_lineage(
            source_observation_ids=(outsider,)
        )
        with self.assertRaisesRegex(ContractValidationError, "must belong to the block"):
            CanonicalBlock(
                canonical_key="bad-lineage",
                revision=1,
                document_version_id=DOCUMENT_VERSION,
                page_id=PageRef(DOCUMENT_VERSION, 0).page_id,
                page_index=0,
                block_type=BlockType.EQUATION,
                representations=(representation,),
                source_observation_ids=(observation.observation_id,),
                reconciliation_method="fixture",
                quality_status=QualityStatus.ACCEPTED,
                semantic_substructures=(outsider_bound,),
            )

    def test_nested_and_collection_limits_reject_before_materialization(self) -> None:
        representation = _representation("bounded")
        structure = _two_element_structure(
            representation,
            structure_kind=SemanticStructureKind.LAYOUT,
            element_kind=SemanticElementKind.LAYOUT_REGION,
            relation_kind=SemanticRelationKind.READING_NEXT,
        )
        payload = structure.to_dict()
        payload["elements"] = [
            structure.elements[0].to_dict()
        ] * (MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE + 1)
        with self.assertRaisesRegex(ContractValidationError, "limit"):
            SemanticSubstructure.from_dict(payload)

        observation = _observation(
            "docling",
            "bounded",
            representation=representation,
            structure=None,
            block_type=BlockType.PARAGRAPH,
        )
        observation_payload = observation.to_dict()
        observation_payload["semantic_substructures"] = [
            structure.to_dict()
        ] * (MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK + 1)
        with self.assertRaisesRegex(ContractValidationError, "limit"):
            ParserBlockObservation.from_dict(observation_payload)

        nested: object = "leaf"
        for _ in range(34):
            nested = {"next": nested}
        with self.assertRaisesRegex(ContractValidationError, "depth"):
            SemanticSourceSelector(native_selector="/deep", metadata={"v": nested})

    def test_empty_extension_is_serialization_and_hash_neutral(self) -> None:
        representation = _representation("legacy")
        observation = _observation(
            "docling",
            "legacy",
            representation=representation,
            structure=None,
            block_type=BlockType.PARAGRAPH,
        )
        self.assertNotIn("semantic_substructures", observation.to_dict())
        expected_observation_hash = content_sha256(
            {
                "document_version_id": observation.document_version_id,
                "page_id": observation.page_id,
                "page_index": observation.page_index,
                "provenance": observation.provenance.to_dict(),
                "block_type": observation.block_type.value,
                "semantic_subtype": observation.semantic_subtype,
                "geometry": observation.geometry.to_dict(),
                "reading_order": observation.reading_order,
                "representations": [representation.to_dict()],
                "assets": [],
                "relations": [],
                "confidence": observation.confidence,
                "metadata": {},
            }
        )
        self.assertEqual(expected_observation_hash, observation.content_sha256)

        block = reconcile_observations((observation,)).canonical_blocks[0]
        self.assertNotIn("semantic_substructures", block.to_dict())
        self.assertNotIn("semantic_substructure_sources", block.metadata)
        expected_block_hash = content_sha256(
            {
                "block_id": block.block_id,
                "block_type": block.block_type.value,
                "semantic_subtype": block.semantic_subtype,
                "geometry": block.geometry.to_dict(),
                "reading_order": block.reading_order,
                "representations": [item.to_dict() for item in block.representations],
                "assets": [],
                "source_observation_ids": block.source_observation_ids,
                "reconciliation_method": block.reconciliation_method,
                "quality_status": block.quality_status.value,
                "parent_block_id": block.parent_block_id,
                "metadata": block.metadata,
            }
        )
        self.assertEqual(expected_block_hash, block.content_sha256)
        self.assertEqual(block, CanonicalBlock.from_dict(block.to_dict()))


if __name__ == "__main__":
    unittest.main()
