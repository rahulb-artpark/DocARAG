from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from docarag.artifact_store import LocalContentAddressedArtifactStore
from docarag.bookrag_projection import (
    BookRAGAssetMaterializationLease,
    project_canonical_blocks,
)
from docarag.canonical_artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
)
from docarag.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    BoundingBox,
    CanonicalBlock,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
)
from docarag.evidence_hydration import (
    CITATION_SCHEMA_VERSION,
    CanonicalEvidenceHydrator,
    EvidenceHydrationError,
)
from docarag.derived_identity import (
    derived_evidence_id,
    derived_identity_payload,
    derived_output_identity,
)


SOURCE_SHA256 = "a" * 64


def make_document() -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=SOURCE_SHA256,
        mime_type="application/pdf",
        source_uri="s3://truth/research-report.pdf",
        display_name="research-report.pdf",
        page_count=3,
    )


def geometry(page_index: int) -> SourceGeometry:
    return SourceGeometry(
        coordinate_space=CoordinateSpace.PAGE_POINTS,
        bbox=BoundingBox(10.0, 20.0 + page_index, 300.0, 180.0),
        page_width=612.0,
        page_height=792.0,
    )


def make_block(
    document: DocumentRef,
    key: str,
    *,
    page_index: int,
    block_type: BlockType,
    representations: tuple[Representation, ...],
    assets: tuple[AssetRef, ...] = (),
    semantic_subtype: str | None = None,
) -> CanonicalBlock:
    return CanonicalBlock(
        canonical_key=key,
        revision=1,
        document_version_id=document.document_version_id,
        page_id=PageRef(
            document_version_id=document.document_version_id,
            page_index=page_index,
        ).page_id,
        page_index=page_index,
        block_type=block_type,
        semantic_subtype=semantic_subtype,
        geometry=geometry(page_index),
        reading_order=float(page_index),
        representations=representations,
        assets=assets,
        source_observation_ids=(f"observation-{key}",),
        reconciliation_method="fixture",
        quality_status=QualityStatus.ACCEPTED,
    )


def tree_record(document: DocumentRef, block: CanonicalBlock, index_id: int) -> dict:
    row = project_canonical_blocks(document, (block,))[0]
    node_type = row["type"]
    if node_type == "text":
        content = row["text"]
    elif node_type == "table":
        content = " ".join(row["table_caption"]) + " ".join(
            row["table_footnote"]
        )
    elif node_type == "equation":
        content = row["text"]
    else:
        content = " ".join(row["image_caption"]) + " ".join(
            row["image_footnote"]
        )
    record = {
        "index_id": index_id,
        "type": node_type,
        "content": content,
        "page": block.page_index,
        "page_id": block.page_id,
        "document_version_id": document.document_version_id,
        "source_uri": document.source_uri,
        "source_sha256": document.content_sha256,
        "canonical_block_id": block.block_id,
        "canonical_revision_id": block.revision_id,
        "source_observation_ids": list(block.source_observation_ids),
        "source_evidence_refs": deepcopy(row["source_evidence_refs"]),
        "canonical_representation_ids": list(
            row["canonical_representation_ids"]
        ),
        "canonical_asset_ids": list(row["canonical_asset_ids"]),
        "selected_representation_ids": list(
            row["selected_representation_ids"]
        ),
        "selected_asset_ids": list(row["selected_asset_ids"]),
        "canonical_relations": deepcopy(row["canonical_relations"]),
        "geometry": block.geometry.to_dict(),
    }
    # Mirrors evidence_hydration._CROP_BEARING_NODE_TYPES: a chart and a
    # diagram carry a crop exactly as an image does.
    if node_type in {"table", "equation", "image", "chart", "diagram"}:
        record["img_path"] = row["img_path"]
    if node_type == "table":
        record.update(
            {
                "table_body": row["table_body"],
                "caption": " ".join(row["table_caption"]),
                "footnote": " ".join(row["table_footnote"]),
            }
        )
    return record


class EvidenceHydrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = make_document()
        self.table = make_block(
            self.document,
            "table",
            page_index=0,
            block_type=BlockType.TABLE,
            representations=(
                Representation(
                    kind=RepresentationKind.TABLE_GRID,
                    content=[["Metric", "Value"], ["Accuracy", "99%"]],
                    media_type="application/json",
                ),
                Representation(
                    kind=RepresentationKind.CAPTION,
                    content="Evaluation metrics",
                    media_type="text/plain",
                ),
            ),
            assets=(
                AssetRef(
                    kind=AssetKind.REGION_CROP,
                    uri="/evidence/table.png",
                    media_type="image/png",
                    sha256="b" * 64,
                ),
            ),
        )
        self.formula = make_block(
            self.document,
            "formula",
            page_index=1,
            block_type=BlockType.EQUATION,
            representations=(
                Representation(
                    kind=RepresentationKind.LATEX,
                    content=r"E = mc^2",
                    media_type="application/x-latex",
                ),
            ),
            assets=(
                AssetRef(
                    kind=AssetKind.REGION_CROP,
                    uri="/evidence/formula.png",
                    media_type="image/png",
                    sha256="c" * 64,
                ),
            ),
        )
        self.diagram = make_block(
            self.document,
            "diagram",
            page_index=2,
            block_type=BlockType.DIAGRAM,
            semantic_subtype="workflow",
            representations=(
                Representation(
                    kind=RepresentationKind.VISUAL_DESCRIPTION,
                    content="Input flows to validation, then indexing.",
                    media_type="text/plain",
                ),
                Representation(
                    kind=RepresentationKind.OCR_TEXT,
                    content="INPUT VALIDATE INDEX",
                    media_type="text/plain",
                ),
            ),
            assets=(
                AssetRef(
                    kind=AssetKind.REGION_CROP,
                    uri="/evidence/workflow.png",
                    media_type="image/png",
                    sha256="d" * 64,
                ),
            ),
        )
        self.blocks = (self.table, self.formula, self.diagram)
        self.records = {
            index: tree_record(self.document, block, index)
            for index, block in enumerate(self.blocks, start=1)
        }
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.artifact = Path(self.temporary_directory.name) / "canonical_blocks.json"
        self._write_artifact(self.blocks)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_artifact(self, blocks: tuple[CanonicalBlock, ...]) -> None:
        self.artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                    "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
                    "document": self.document.to_dict(),
                    "parse_request_id": "request-fixture",
                    "block_count": len(blocks),
                    "canonical_blocks": [block.to_dict() for block in blocks],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def hydrator(self, **limits) -> CanonicalEvidenceHydrator:
        return CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: [deepcopy(self.records[item]) for item in ids],
            expected_document_version_id=self.document.document_version_id,
            expected_source_sha256=self.document.content_sha256,
            expected_source_uri=self.document.source_uri,
            **limits,
        )

    def test_hydrates_a_node_whose_image_asset_was_leased_at_index_time(self):
        # An index-time projection holds a materialization lease and records the
        # leased crop in ``selected_asset_ids``.  This hydrator is deliberately
        # reference-only, so it can never reproduce that selection; it must
        # still accept the node instead of reporting a projection mismatch.
        store = LocalContentAddressedArtifactStore(
            Path(self.temporary_directory.name) / "asset_cas", create=True
        )
        stored = store.publish(b"\x89PNG\r\n\x1a\n workflow", media_type="image/png")
        asset = stored.to_asset_ref(kind=AssetKind.REGION_CROP)
        leased_diagram = make_block(
            self.document,
            "diagram",
            page_index=2,
            block_type=BlockType.DIAGRAM,
            semantic_subtype="workflow",
            representations=(
                Representation(
                    kind=RepresentationKind.OCR_TEXT,
                    content="INPUT VALIDATE INDEX",
                    media_type="text/plain",
                ),
            ),
            assets=(asset,),
        )
        lease = BookRAGAssetMaterializationLease(
            store.make_asset_resolver(), (asset,)
        )
        with lease:
            row = project_canonical_blocks(
                self.document,
                (leased_diagram,),
                asset_materialization_lease=lease,
            )[0]
        self.assertEqual([asset.asset_id], list(row["selected_asset_ids"]))

        record = {
            **tree_record(self.document, leased_diagram, 1),
            "selected_asset_ids": list(row["selected_asset_ids"]),
        }
        self._write_artifact((leased_diagram,))
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: [deepcopy(record) for _ in ids],
            expected_document_version_id=self.document.document_version_id,
        )

        hydrated = hydrator.hydrate([{**deepcopy(record), "distance": 0.1}])

        self.assertEqual("INPUT VALIDATE INDEX", hydrated[0]["content"])

    def test_node_citing_an_asset_the_block_does_not_own_is_rejected(self):
        record = {
            **deepcopy(self.records[3]),
            "selected_asset_ids": ["asset_forged"],
        }
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: [deepcopy(record) for _ in ids],
            expected_document_version_id=self.document.document_version_id,
        )

        with self.assertRaisesRegex(
            EvidenceHydrationError, "not canonical block assets"
        ):
            hydrator.hydrate([{**deepcopy(record), "distance": 0.1}])

    def test_hydrates_table_formula_and_diagram_without_mutating_candidates(self):
        candidates = [
            {**deepcopy(self.records[index]), "distance": float(index) / 10}
            for index in (1, 2, 3)
        ]
        before = deepcopy(candidates)

        hydrated = self.hydrator().hydrate(candidates)

        self.assertEqual(before, candidates)
        self.assertEqual(3, len(hydrated))
        for item, block in zip(hydrated, self.blocks, strict=True):
            self.assertEqual(CITATION_SCHEMA_VERSION, item["citation"]["schema_version"])
            self.assertEqual(SOURCE_SHA256, item["citation"]["source_sha256"])
            self.assertEqual(block.page_id, item["citation"]["page_ids"][0])
            self.assertEqual(
                block.geometry.to_dict(), item["citation"]["locations"][0]["geometry"]
            )
            self.assertEqual(block.to_dict(), item["canonical_sources"][0])
            self.assertEqual(float(item["index_id"]) / 10, item["distance"])
            self.assertEqual(
                "reference_only", item["citation"]["asset_material_status"]
            )
        self.assertIn("<table>", hydrated[0]["table_body"])
        self.assertEqual("", hydrated[0]["img_path"])
        self.assertEqual(r"E = mc^2", hydrated[1]["content"])
        self.assertEqual("", hydrated[1]["img_path"])
        self.assertEqual("workflow", hydrated[2]["canonical_sources"][0]["semantic_subtype"])
        self.assertEqual("", hydrated[2]["img_path"])

    def test_derived_evidence_resolves_every_source_in_declared_order(self):
        refs = []
        for block in (self.table, self.formula):
            refs.append(
                {
                    "canonical_block_id": block.block_id,
                    "canonical_revision_id": block.revision_id,
                    "observation_id": block.source_observation_ids[0],
                    "source_observation_ids": list(block.source_observation_ids),
                    "document_version_id": self.document.document_version_id,
                    "page_id": block.page_id,
                    "page_idx": block.page_index,
                    "geometry": block.geometry.to_dict(),
                }
            )
        derived = {
            "index_id": 9,
            "type": "text",
            "content": "The table value is defined by the equation.",
            "page": self.table.page_index,
            "page_id": self.table.page_id,
            "document_version_id": self.document.document_version_id,
            "source_uri": self.document.source_uri,
            "source_sha256": self.document.content_sha256,
            "derived_evidence_id": "pending",
            "source_canonical_block_ids": [
                self.table.block_id,
                self.formula.block_id,
            ],
            "source_canonical_revision_ids": [
                self.table.revision_id,
                self.formula.revision_id,
            ],
            "source_observation_ids": [
                *self.table.source_observation_ids,
                *self.formula.source_observation_ids,
            ],
            "source_evidence_refs": refs,
            "transformations": ["bookrag.fixture_merge.v1"],
            "geometry": self.table.geometry.to_dict(),
        }
        derived["derived_evidence_id"] = derived_evidence_id(
            derived_identity_payload(
                output=derived_output_identity(
                    content_type=derived["type"],
                    text=derived["content"],
                ),
                source_revision_ids=derived["source_canonical_revision_ids"],
                source_document_version_ids=[self.document.document_version_id],
                source_block_ids=derived["source_canonical_block_ids"],
                source_observation_ids=derived["source_observation_ids"],
                source_references=refs,
                transformations=derived["transformations"],
            )
        )
        self.records[9] = deepcopy(derived)

        item = self.hydrator().hydrate([deepcopy(derived)])[0]

        self.assertEqual(2, item["canonical_source_count"])
        self.assertEqual(
            [self.table.revision_id, self.formula.revision_id],
            item["citation"]["canonical_revision_ids"],
        )
        self.assertEqual([0, 1], item["citation"]["page_indices"])
        self.assertEqual(
            ["bookrag.fixture_merge.v1"], item["citation"]["transformations"]
        )

        tampered = deepcopy(derived)
        tampered["content"] = "The derived output was changed after identity binding."
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda _ids: [deepcopy(tampered)],
        )
        with self.assertRaisesRegex(EvidenceHydrationError, "derived evidence digest"):
            hydrator.hydrate([deepcopy(tampered)])

    def test_rejects_retrieval_tree_and_canonical_lineage_mismatches(self):
        tampered_candidate = deepcopy(self.records[2])
        tampered_candidate["content"] = "E = not source truth"
        with self.assertRaisesRegex(EvidenceHydrationError, "authoritative field"):
            self.hydrator().hydrate([tampered_candidate])

        tampered_tree = deepcopy(self.records[2])
        tampered_tree["source_sha256"] = "f" * 64
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda _ids: [tampered_tree],
        )
        with self.assertRaisesRegex(EvidenceHydrationError, "source_sha256"):
            hydrator.hydrate([deepcopy(self.records[2])])

        tampered_material_ref = deepcopy(self.records[2])
        tampered_material_ref["source_evidence_refs"][0][
            "representation_ids"
        ] = ["representation_missing"]
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda _ids: [tampered_material_ref],
        )
        candidate = deepcopy(self.records[2])
        candidate.pop("source_evidence_refs")
        with self.assertRaisesRegex(EvidenceHydrationError, "representation_ids"):
            hydrator.hydrate([candidate])

        tampered_asset_ref = deepcopy(self.records[2])
        tampered_asset_ref["source_evidence_refs"][0]["asset_ids"] = [
            "asset_missing"
        ]
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda _ids: [tampered_asset_ref],
        )
        candidate = deepcopy(self.records[2])
        candidate.pop("source_evidence_refs")
        with self.assertRaisesRegex(EvidenceHydrationError, "asset_ids"):
            hydrator.hydrate([candidate])

        tampered_selected_material = deepcopy(self.records[2])
        tampered_selected_material["selected_representation_ids"] = [
            "representation_missing"
        ]
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda _ids: [tampered_selected_material],
        )
        candidate = deepcopy(self.records[2])
        candidate.pop("selected_representation_ids")
        with self.assertRaisesRegex(
            EvidenceHydrationError, "selected_representation_ids"
        ):
            hydrator.hydrate([candidate])

        tampered_revision = deepcopy(self.records[2])
        tampered_revision["canonical_revision_id"] = "blockrev_missing"
        self.records[2] = tampered_revision
        with self.assertRaisesRegex(EvidenceHydrationError, "absent"):
            self.hydrator().hydrate([tampered_revision])

    def test_explicit_bounds_fail_before_unbounded_resolution(self):
        calls = []
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: calls.append(ids) or [],
            max_evidence=1,
        )
        with self.assertRaisesRegex(EvidenceHydrationError, "max_evidence"):
            hydrator.hydrate([self.records[1], self.records[2]])
        self.assertEqual([], calls)

        with self.assertRaisesRegex(EvidenceHydrationError, "max_canonical_blocks"):
            CanonicalEvidenceHydrator.from_artifact(
                self.artifact,
                node_resolver=lambda _ids: [],
                max_canonical_blocks=2,
            )

    def test_candidates_require_source_identity_and_well_formed_derived_id(self):
        calls = []
        hydrator = CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: calls.append(ids) or [],
        )
        missing_source = deepcopy(self.records[1])
        missing_source.pop("source_sha256")
        with self.assertRaisesRegex(EvidenceHydrationError, "source_sha256"):
            hydrator.hydrate([missing_source])

        malformed_derived = deepcopy(self.records[1])
        malformed_derived.pop("canonical_block_id")
        malformed_derived.pop("canonical_revision_id")
        malformed_derived["derived_evidence_id"] = "derived:NOT-A-DIGEST"
        with self.assertRaisesRegex(EvidenceHydrationError, "64 lowercase hex"):
            hydrator.hydrate([malformed_derived])
        self.assertEqual([], calls)

    def test_artifact_read_fails_if_descriptor_metadata_changes(self):
        real_stat = self.artifact.stat()
        before = SimpleNamespace(
            st_mode=real_stat.st_mode,
            st_dev=real_stat.st_dev,
            st_ino=real_stat.st_ino,
            st_size=real_stat.st_size,
            st_mtime_ns=real_stat.st_mtime_ns,
            st_ctime_ns=real_stat.st_ctime_ns,
        )
        after = SimpleNamespace(**vars(before))
        after.st_mtime_ns += 1
        with patch(
            "docarag.evidence_hydration.os.fstat",
            side_effect=(before, after),
        ):
            with self.assertRaisesRegex(EvidenceHydrationError, "changed while"):
                CanonicalEvidenceHydrator.from_artifact(
                    self.artifact,
                    node_resolver=lambda _ids: [],
                )

    def test_does_not_accept_wrong_tree_root_identity(self):
        with self.assertRaisesRegex(EvidenceHydrationError, "tree source_uri"):
            CanonicalEvidenceHydrator.from_artifact(
                self.artifact,
                node_resolver=lambda _ids: [],
                expected_source_uri="s3://wrong/document.pdf",
            )


if __name__ == "__main__":
    unittest.main()
