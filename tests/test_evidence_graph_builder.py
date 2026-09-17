from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from unittest.mock import patch
import unittest

from document_services.domain.base import (
    canonical_json_bytes,
    content_sha256,
    json_ready,
    stable_id,
)
from document_services.domain.evidence import (
    AssetKind,
    AssetRef,
    BlockType,
    CanonicalBlock,
    DocumentRef,
    QualityStatus,
    Representation,
    RepresentationKind,
)
from document_services.domain.knowledge import (
    EvidenceAnchor as DomainEvidenceAnchor,
    EvidenceGraphRelease,
)
from document_services.evidence.graph_builder import (
    BUILDER_SCHEMA_VERSION,
    DEFAULT_MAX_RELEASE_BYTES,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_UNSPECIFIED_RELATION_TYPE,
    CanonicalMaterialSelection,
    EvidenceGraphBuildError,
    EvidenceGraphBuildErrorCode,
    EvidenceGraphBuildLimits,
    EvidenceGraphReleaseBinding,
    ImmutableArtifactRef,
    LegacyKGCheckpointInput,
    SelectedAssetMaterializationReceipt,
    TreeNodeEvidenceDescriptor,
    build_evidence_graph_release,
    compute_canonical_input_sha256,
    compute_tree_descriptor_sha256,
    iter_canonical_input_artifact_bytes,
    iter_tree_descriptor_artifact_bytes,
)
from document_services.evidence.projections.legacy_graph import (
    LegacyProjectionPolicy,
    project_bookrag_legacy_graph,
)


_FIXTURE_ASSET_BYTES = b"docarag-fixture-workflow-image-bytes"
_FIXTURE_ASSET_SHA256 = hashlib.sha256(_FIXTURE_ASSET_BYTES).hexdigest()


def _value_digest(value: object) -> dict[str, object]:
    return {"present": value is not None, "sha256": content_sha256(value)}


def _document() -> DocumentRef:
    return DocumentRef.identified(
        content_sha256="a" * 64,
        mime_type="application/pdf",
        source_uri="cas://sha256/" + "a" * 64,
        document_id="document-builder-test",
        page_count=1,
    )


def _blocks(
    document: DocumentRef,
    *,
    asset_sha256: str | None = _FIXTURE_ASSET_SHA256,
    asset_uri: str | None = None,
) -> tuple[CanonicalBlock, CanonicalBlock]:
    page_id = stable_id("page", document.document_version_id, 0)
    text = CanonicalBlock(
        canonical_key="text-1",
        revision=1,
        document_version_id=document.document_version_id,
        page_id=page_id,
        page_index=0,
        block_type=BlockType.PARAGRAPH,
        representations=(
            Representation(
                RepresentationKind.TEXT,
                "Alpha provides input to Beta.",
                "text/plain",
            ),
        ),
        source_observation_ids=("observation-text",),
        reconciliation_method="fixture",
        quality_status=QualityStatus.ACCEPTED,
    )
    uri = asset_uri
    if uri is None:
        uri = (
            f"cas://sha256/{asset_sha256}"
            if asset_sha256 is not None
            else "file:///unbound/figure.png"
        )
    figure = CanonicalBlock(
        canonical_key="figure-1",
        revision=1,
        document_version_id=document.document_version_id,
        page_id=page_id,
        page_index=0,
        block_type=BlockType.DIAGRAM,
        representations=(
            Representation(
                RepresentationKind.VISUAL_DESCRIPTION,
                "Alpha to Beta workflow",
                "text/plain",
            ),
        ),
        assets=(
            AssetRef(
                AssetKind.REGION_CROP,
                uri,
                media_type="image/png",
                sha256=asset_sha256,
            ),
        ),
        source_observation_ids=("observation-figure",),
        reconciliation_method="fixture",
        quality_status=QualityStatus.ACCEPTED,
        semantic_subtype="workflow",
    )
    return text, figure


def _selections(
    blocks: tuple[CanonicalBlock, CanonicalBlock],
    *,
    representation_id: str | None = None,
    asset_id: str | None = None,
) -> tuple[CanonicalMaterialSelection, ...]:
    figure = blocks[1]
    return (
        CanonicalMaterialSelection(
            canonical_revision_id=figure.revision_id,
            representation_id=(
                representation_id
                if representation_id is not None
                else figure.representations[0].representation_id
            ),
            metadata={"role": "vlm_context"},
        ),
        CanonicalMaterialSelection(
            canonical_revision_id=figure.revision_id,
            asset_id=(asset_id if asset_id is not None else figure.assets[0].asset_id),
            metadata={"role": "vlm_image"},
        ),
    )


def _checkpoint_identity(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, CanonicalBlock],
    *,
    legacy_node_index: int,
    node_type: str,
    checkpoint_kind: str,
    selected_representation_ids: tuple[str, ...],
    selected_asset_ids: tuple[str, ...],
    model_revision: str = "fixture-v1",
    image_bytes_sha256: str | None = None,
) -> dict[str, object]:
    figure = blocks[1]
    source = {
        "source_sha256": document.content_sha256,
        "document_version_id": document.document_version_id,
        "canonical_block_id": figure.block_id,
        "canonical_revision_id": figure.revision_id,
        "observation_id": "observation-figure",
        "source_observation_ids": ["observation-figure"],
        "derived_evidence_id": None,
        "source_canonical_block_ids": [figure.block_id],
        "source_canonical_revision_ids": [figure.revision_id],
        "canonical_representation_ids": _value_digest(
            [item.representation_id for item in figure.representations]
        ),
        "canonical_asset_ids": _value_digest(
            [item.asset_id for item in figure.assets]
        ),
        "selected_representation_ids": _value_digest(
            list(selected_representation_ids)
        ),
        "selected_asset_ids": _value_digest(list(selected_asset_ids)),
        "source_evidence_refs": _value_digest(
            [
                {
                    "canonical_revision_id": figure.revision_id,
                    "selected_representation_ids": list(selected_representation_ids),
                    "selected_asset_ids": list(selected_asset_ids),
                }
            ]
        ),
        "parser_provenance": _value_digest(
            {"parser_name": "fixture", "model_version": "fixture-parser-v1"}
        ),
        "quality_status": "accepted",
        "reconciliation_method": "fixture",
        "transformations": _value_digest([]),
    }
    evidence: dict[str, object] = {
        "content": _value_digest("Alpha to Beta workflow"),
        "table_body": _value_digest(None),
        "caption": _value_digest("Alpha to Beta workflow"),
        "footnote": _value_digest(""),
    }
    if node_type == "image":
        selected_asset = blocks[1].assets[0]
        evidence["image"] = {
            "status": "present",
            "path_sha256": "c" * 64,
            "bytes_sha256": (
                image_bytes_sha256
                if image_bytes_sha256 is not None
                else selected_asset.sha256
            ),
            "size": len(_FIXTURE_ASSET_BYTES),
        }
    identity: dict[str, object] = {
        "kind": checkpoint_kind,
        "node": {
            "node_idx": legacy_node_index,
            "node_type": node_type,
            "source": source,
            "page": {
                "page_idx": 0,
                "page_id": figure.page_id,
                "page_path": _value_digest("/immutable/pages/page-0.png"),
                "source_geometry": _value_digest(None),
            },
            "evidence": evidence,
        },
        "models": {
            "llm": {
                "controller": "retrieval_engine.providers.language_model.LLM",
                "config": {
                    "backend": "fixture",
                    "model_name": model_revision,
                    "temperature": 0.0,
                },
            },
            "vlm": {
                "controller": "retrieval_engine.providers.vision_model.VLM",
                "config": {
                    "backend": "fixture",
                    "model_name": model_revision,
                    "temperature": 0.0,
                },
            },
        },
        "graph_extraction": {
            "extractor_type": "llm",
            "local_model_name": "unused",
            "local_device": "cpu",
            "image_description_force": True,
            "max_gleaning": 1,
            "extraction_max_workers": 2,
            "extraction_max_in_flight": 2,
            "extraction_batch_size": 8,
        },
        "prompt_sha256": {
            "IMAGE_ENTITY_EXTRACTION": "d" * 64,
            "extraction_protocol": "e" * 64,
        },
    }
    if checkpoint_kind == "title":
        identity["title_context"] = {"title_path": [], "sibling_nodes": []}
    return identity


def _descriptor(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, CanonicalBlock],
    *,
    tree_node_id: str = "opaque-tree-node-7",
    legacy_node_index: int = 7,
    node_type: str = "image",
    checkpoint_kind: str = "node",
    selections: tuple[CanonicalMaterialSelection, ...] | None = None,
    selected_representation_ids: tuple[str, ...] | None = None,
    selected_asset_ids: tuple[str, ...] | None = None,
    model_revision: str = "fixture-v1",
    image_bytes_sha256: str | None = None,
) -> TreeNodeEvidenceDescriptor:
    values = selections or _selections(blocks)
    representation_ids = selected_representation_ids
    if representation_ids is None:
        representation_ids = tuple(
            item.representation_id
            for item in values
            if item.representation_id is not None
        )
    asset_ids = selected_asset_ids
    if asset_ids is None:
        asset_ids = tuple(item.asset_id for item in values if item.asset_id is not None)
    identity = _checkpoint_identity(
        document,
        blocks,
        legacy_node_index=legacy_node_index,
        node_type=node_type,
        checkpoint_kind=checkpoint_kind,
        selected_representation_ids=representation_ids,
        selected_asset_ids=asset_ids,
        model_revision=model_revision,
        image_bytes_sha256=image_bytes_sha256,
    )
    return TreeNodeEvidenceDescriptor(
        tree_node_id=tree_node_id,
        legacy_node_index=legacy_node_index,
        node_type=node_type,
        checkpoint_kind=checkpoint_kind,
        expected_checkpoint_identity_sha256=content_sha256(identity),
        selections=values,
        selected_representation_ids=representation_ids,
        selected_asset_ids=asset_ids,
    )


def _entity(name: str, entity_type: str, node_index: int) -> dict[str, object]:
    return {
        "entity_name": name,
        "entity_type": entity_type,
        "description": f"Raw description of {name}",
        "source_ids": [node_index],
    }


def _relation(
    source: str,
    target: str,
    relation: str,
    node_index: int,
    *,
    weight: float,
) -> dict[str, object]:
    return {
        "src_entity_name": source,
        "tgt_entity_name": target,
        "relation_name": relation,
        "weight": weight,
        "description": f"Raw {source} {relation or 'unspecified'} {target}",
        "source_ids": [node_index],
    }


def _envelope(
    tree_node_id: str,
    identity: dict[str, object],
    payload: dict[str, object],
) -> LegacyKGCheckpointInput:
    return LegacyKGCheckpointInput.from_mapping(
        tree_node_id=tree_node_id,
        envelope={
            "schema_version": LEGACY_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": content_sha256(identity),
            "payload_sha256": content_sha256(payload),
            "identity": identity,
            "payload": payload,
        },
    )


def _artifact_bytes(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, ...],
    descriptors: tuple[TreeNodeEvidenceDescriptor, ...],
    *,
    tree_revision_id: str = "tree-revision-fixture",
) -> tuple[bytes, bytes]:
    canonical = b"".join(
        iter_canonical_input_artifact_bytes(document, blocks)
    )
    tree = b"".join(
        iter_tree_descriptor_artifact_bytes(
            descriptors,
            tree_revision_id=tree_revision_id,
        )
    )
    return canonical, tree


class _MemoryArtifactReader:
    def __init__(self, values: tuple[bytes, ...]) -> None:
        self.values = {
            f"cas://sha256/{hashlib.sha256(value).hexdigest()}": value
            for value in values
        }

    def iter_bytes(
        self,
        artifact: ImmutableArtifactRef,
        *,
        max_bytes: int,
    ):
        value = self.values[artifact.uri]
        for offset in range(0, len(value), 17):
            yield value[offset : offset + 17]


def _checkpoint(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, CanonicalBlock],
    descriptor: TreeNodeEvidenceDescriptor,
    *,
    entities: list[dict[str, object]] | None = None,
    relations: list[dict[str, object]] | None = None,
    model_revision: str = "fixture-v1",
    image_bytes_sha256: str | None = None,
    identity: dict[str, object] | None = None,
) -> LegacyKGCheckpointInput:
    checkpoint_identity = identity or _checkpoint_identity(
        document,
        blocks,
        legacy_node_index=descriptor.legacy_node_index,
        node_type=descriptor.node_type,
        checkpoint_kind=descriptor.checkpoint_kind,
        selected_representation_ids=descriptor.selected_representation_ids,
        selected_asset_ids=descriptor.selected_asset_ids,
        model_revision=model_revision,
        image_bytes_sha256=image_bytes_sha256,
    )
    payload = {
        "entities": entities
        if entities is not None
        else [
            _entity("Alpha", "PROCESS", descriptor.legacy_node_index),
            _entity("Beta", "PROCESS", descriptor.legacy_node_index),
        ],
        "relations": relations
        if relations is not None
        else [
            _relation("Alpha", "Beta", "feeds", descriptor.legacy_node_index, weight=0.91),
            _relation("Alpha", "Beta", "feeds", descriptor.legacy_node_index, weight=0.91),
            _relation("Beta", "Alpha", "feeds", descriptor.legacy_node_index, weight=0.44),
        ],
        "node_idx": descriptor.legacy_node_index,
    }
    return _envelope(descriptor.tree_node_id, checkpoint_identity, payload)


def _binding(
    document: DocumentRef,
    blocks: tuple[CanonicalBlock, ...],
    descriptors: tuple[TreeNodeEvidenceDescriptor, ...],
) -> EvidenceGraphReleaseBinding:
    canonical_bytes, tree_bytes = _artifact_bytes(document, blocks, descriptors)
    canonical_digest = hashlib.sha256(canonical_bytes).hexdigest()
    tree_digest = hashlib.sha256(tree_bytes).hexdigest()
    asset_by_id = {
        asset.asset_id: asset
        for block in blocks
        for asset in block.assets
    }
    selected_asset_ids = {
        asset_id
        for descriptor in descriptors
        for asset_id in descriptor.selected_asset_ids
    }
    receipts = tuple(
        SelectedAssetMaterializationReceipt(
            asset_id=asset_id,
            artifact=ImmutableArtifactRef(
                uri=f"cas://sha256/{asset_by_id[asset_id].sha256}",
                sha256=asset_by_id[asset_id].sha256,
                size_bytes=len(_FIXTURE_ASSET_BYTES),
            ),
            media_type=asset_by_id[asset_id].media_type,
        )
        for asset_id in sorted(selected_asset_ids)
        if asset_id in asset_by_id and asset_by_id[asset_id].sha256 is not None
    )
    return EvidenceGraphReleaseBinding(
        build_spec_sha256="1" * 64,
        canonical_artifact=ImmutableArtifactRef(
            uri=f"cas://sha256/{canonical_digest}",
            sha256=canonical_digest,
            size_bytes=len(canonical_bytes),
        ),
        canonical_input_sha256=compute_canonical_input_sha256(document, blocks),
        tree_revision_id="tree-revision-fixture",
        tree_artifact=ImmutableArtifactRef(
            uri=f"cas://sha256/{tree_digest}",
            sha256=tree_digest,
            size_bytes=len(tree_bytes),
        ),
        tree_descriptor_sha256=compute_tree_descriptor_sha256(
            descriptors,
            tree_revision_id="tree-revision-fixture",
        ),
        selected_asset_receipts=receipts,
        metadata={"profile": "fixture"},
    )


class EvidenceGraphReleaseBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = _document()
        self.blocks = _blocks(self.document)
        self.descriptor = _descriptor(self.document, self.blocks)
        self.checkpoint = _checkpoint(self.document, self.blocks, self.descriptor)

    def _build(self, **overrides: object) -> EvidenceGraphRelease:
        values: dict[str, object] = {
            "document": self.document,
            "canonical_blocks": self.blocks,
            "tree_nodes": (self.descriptor,),
            "checkpoints": (self.checkpoint,),
        }
        values.update(overrides)
        if "binding" not in values:
            values["binding"] = _binding(
                values["document"],  # type: ignore[arg-type]
                tuple(values["canonical_blocks"]),  # type: ignore[arg-type]
                tuple(values["tree_nodes"]),  # type: ignore[arg-type]
            )
        if "artifact_reader" not in values:
            values["artifact_reader"] = _MemoryArtifactReader(
                (
                    *_artifact_bytes(
                        values["document"],  # type: ignore[arg-type]
                        tuple(values["canonical_blocks"]),  # type: ignore[arg-type]
                        tuple(values["tree_nodes"]),  # type: ignore[arg-type]
                    ),
                    _FIXTURE_ASSET_BYTES,
                )
            )
        return build_evidence_graph_release(**values)  # type: ignore[arg-type]

    def test_builds_deterministic_release_with_exact_whole_material_assertions(self) -> None:
        release = self._build()
        rebuilt = self._build(canonical_blocks=tuple(reversed(self.blocks)))

        self.assertEqual(release, rebuilt)
        self.assertEqual(release, EvidenceGraphRelease.from_json(release.to_json()))
        self.assertEqual(release.metadata["builder"]["schema_version"], BUILDER_SCHEMA_VERSION)
        self.assertEqual(
            "coarse_whole_node", release.metadata["builder"]["evidence_precision"]
        )
        self.assertFalse(release.metadata["builder"]["authority_eligible"])
        self.assertEqual(len(release.anchors), 2)
        self.assertEqual(len(release.mentions), 2)
        self.assertEqual(len(release.relations), 3)
        self.assertEqual(len(release.gt_links), 5)
        self.assertEqual(len({item.relation_id for item in release.relations}), 3)
        self.assertTrue(all(anchor.char_span is None for anchor in release.anchors))
        self.assertTrue(all(anchor.bbox is None for anchor in release.anchors))
        asset_anchor = next(item for item in release.anchors if item.asset_id is not None)
        self.assertEqual(asset_anchor.metadata["asset_ref"], self.blocks[1].assets[0].to_dict())
        self.assertEqual(release.mentions[0].attributes["description"], release.mentions[0].attributes["legacy_assertion"]["description"])
        self.assertEqual(
            {relation.attributes["weight"] for relation in release.relations},
            {0.91, 0.44},
        )
        self.assertLessEqual(len(release.to_json().encode("utf-8")), DEFAULT_MAX_RELEASE_BYTES)

    def test_empty_real_bookrag_relation_name_maps_losslessly_to_sentinel(self) -> None:
        checkpoint = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            relations=[_relation("Alpha", "Beta", "", 7, weight=2.75)],
        )
        release = self._build(checkpoints=(checkpoint,))
        relation = release.relations[0]
        self.assertEqual(relation.relation_type, LEGACY_UNSPECIFIED_RELATION_TYPE)
        self.assertEqual(relation.attributes["legacy_relation_name"], "")
        self.assertEqual(relation.attributes["description"], "Raw Alpha unspecified Beta")
        self.assertEqual(relation.attributes["weight"], 2.75)
        projected = project_bookrag_legacy_graph(
            release,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases={self.descriptor.tree_node_id: 7},
        ).to_dict()
        self.assertEqual(projected["graph"]["nodes"][0]["description"], release.mentions[0].attributes["description"])
        assertion = projected["graph"]["links"][0]["assertions"][0]["relation"]
        self.assertEqual(assertion["attributes"]["weight"], 2.75)
        self.assertEqual(assertion["attributes"]["legacy_relation_name"], "")

    def test_fails_closed_on_missing_or_ambiguous_local_endpoint(self) -> None:
        missing = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            entities=[_entity("Alpha", "PROCESS", 7)],
            relations=[_relation("Alpha", "Beta", "feeds", 7, weight=1.0)],
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(missing,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.MISSING_ENDPOINT)

        ambiguous = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            entities=[
                _entity("Alpha", "PROCESS", 7),
                _entity("Alpha", "SYSTEM", 7),
                _entity("Beta", "PROCESS", 7),
            ],
            relations=[_relation("Alpha", "Beta", "feeds", 7, weight=1.0)],
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(ambiguous,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.AMBIGUOUS_ENDPOINT)

    def test_shared_source_locus_reuses_anchor_but_keeps_separate_tree_links(self) -> None:
        second = _descriptor(
            self.document,
            self.blocks,
            tree_node_id="opaque-tree-node-8",
            legacy_node_index=8,
        )
        release = self._build(
            tree_nodes=(second, self.descriptor),
            checkpoints=(
                _checkpoint(self.document, self.blocks, second),
                self.checkpoint,
            ),
        )
        self.assertEqual(len(release.anchors), 2)
        self.assertEqual(len(release.mentions), 4)
        self.assertEqual(len(release.relations), 6)
        self.assertEqual(len(release.gt_links), 10)
        self.assertEqual(
            {link.tree_node_id for link in release.gt_links},
            {self.descriptor.tree_node_id, second.tree_node_id},
        )

    def test_checkpoint_integrity_source_binding_and_missing_material_fail(self) -> None:
        tampered_payload = json.loads(self.checkpoint.encoded_envelope)
        tampered_payload["payload_sha256"] = "f" * 64
        tampered = LegacyKGCheckpointInput.from_mapping(
            tree_node_id=self.descriptor.tree_node_id,
            envelope=tampered_payload,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(tampered,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY)

        wrong_source = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            entities=[_entity("Alpha", "PROCESS", 99)],
            relations=[],
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(wrong_source,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)

        selections = _selections(self.blocks, representation_id="repr_missing")
        missing_descriptor = _descriptor(
            self.document, self.blocks, selections=selections
        )
        missing_checkpoint = _checkpoint(
            self.document, self.blocks, missing_descriptor
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                tree_nodes=(missing_descriptor,), checkpoints=(missing_checkpoint,)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.MATERIAL_BINDING)

    def test_actual_nested_identity_is_strict_and_stale_checkpoint_is_rejected(self) -> None:
        stale = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            model_revision="fixture-v2",
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(stale,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)
        self.assertIn("authorized", str(raised.exception))

        malformed_identity = _checkpoint_identity(
            self.document,
            self.blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=self.descriptor.selected_representation_ids,
            selected_asset_ids=self.descriptor.selected_asset_ids,
        )
        malformed_identity.pop("graph_extraction")
        authorized_malformed = replace(
            self.descriptor,
            expected_checkpoint_identity_sha256=content_sha256(malformed_identity),
        )
        malformed = _checkpoint(
            self.document,
            self.blocks,
            authorized_malformed,
            identity=malformed_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                tree_nodes=(authorized_malformed,), checkpoints=(malformed,)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY)

        missing_selection_identity = _checkpoint_identity(
            self.document,
            self.blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=self.descriptor.selected_representation_ids,
            selected_asset_ids=self.descriptor.selected_asset_ids,
        )
        missing_selection_identity["node"]["source"].pop("selected_asset_ids")
        authorized_missing = replace(
            self.descriptor,
            expected_checkpoint_identity_sha256=content_sha256(
                missing_selection_identity
            ),
        )
        missing_selection_checkpoint = _checkpoint(
            self.document,
            self.blocks,
            authorized_missing,
            identity=missing_selection_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                tree_nodes=(authorized_missing,),
                checkpoints=(missing_selection_checkpoint,),
            )
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY
        )

    def test_ordered_selected_id_digest_and_image_bytes_must_match(self) -> None:
        figure = self.blocks[1]
        second_representation = Representation(
            RepresentationKind.OCR_TEXT,
            "OCR Alpha to Beta workflow",
            "text/plain",
        )
        two_rep_figure = CanonicalBlock(
            canonical_key=figure.canonical_key,
            revision=figure.revision,
            document_version_id=figure.document_version_id,
            page_id=figure.page_id,
            page_index=figure.page_index,
            block_type=figure.block_type,
            representations=(figure.representations[0], second_representation),
            assets=figure.assets,
            source_observation_ids=figure.source_observation_ids,
            reconciliation_method=figure.reconciliation_method,
            quality_status=figure.quality_status,
            semantic_subtype=figure.semantic_subtype,
        )
        ordered_blocks = (self.blocks[0], two_rep_figure)
        ordered_selections = (
            CanonicalMaterialSelection(
                canonical_revision_id=two_rep_figure.revision_id,
                representation_id=figure.representations[0].representation_id,
            ),
            CanonicalMaterialSelection(
                canonical_revision_id=two_rep_figure.revision_id,
                representation_id=second_representation.representation_id,
            ),
            CanonicalMaterialSelection(
                canonical_revision_id=two_rep_figure.revision_id,
                asset_id=two_rep_figure.assets[0].asset_id,
            ),
        )
        attested_order = (
            second_representation.representation_id,
            figure.representations[0].representation_id,
        )
        ordered_descriptor = _descriptor(
            self.document,
            ordered_blocks,
            selections=ordered_selections,
            selected_representation_ids=attested_order,
        )
        ordered_checkpoint = _checkpoint(
            self.document, ordered_blocks, ordered_descriptor
        )
        ordered_release = self._build(
            canonical_blocks=ordered_blocks,
            tree_nodes=(ordered_descriptor,),
            checkpoints=(ordered_checkpoint,),
        )
        self.assertEqual(len(ordered_release.anchors), 3)

        wrong_order_identity = _checkpoint_identity(
            self.document,
            ordered_blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=tuple(reversed(attested_order)),
            selected_asset_ids=ordered_descriptor.selected_asset_ids,
        )
        authorized_wrong_order = replace(
            ordered_descriptor,
            expected_checkpoint_identity_sha256=content_sha256(wrong_order_identity),
        )
        wrong_order_checkpoint = _checkpoint(
            self.document,
            ordered_blocks,
            authorized_wrong_order,
            identity=wrong_order_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                canonical_blocks=ordered_blocks,
                tree_nodes=(authorized_wrong_order,),
                checkpoints=(wrong_order_checkpoint,),
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)

        wrong_selection_identity = _checkpoint_identity(
            self.document,
            self.blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=(),
            selected_asset_ids=self.descriptor.selected_asset_ids,
        )
        authorized_wrong = replace(
            self.descriptor,
            expected_checkpoint_identity_sha256=content_sha256(wrong_selection_identity),
        )
        checkpoint = _checkpoint(
            self.document,
            self.blocks,
            authorized_wrong,
            identity=wrong_selection_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(tree_nodes=(authorized_wrong,), checkpoints=(checkpoint,))
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)

        absent_identity = _checkpoint_identity(
            self.document,
            self.blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=self.descriptor.selected_representation_ids,
            selected_asset_ids=self.descriptor.selected_asset_ids,
        )
        absent_identity["node"]["source"]["selected_representation_ids"] = _value_digest(None)
        authorized_absent = replace(
            self.descriptor,
            expected_checkpoint_identity_sha256=content_sha256(absent_identity),
        )
        absent_checkpoint = _checkpoint(
            self.document,
            self.blocks,
            authorized_absent,
            identity=absent_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                tree_nodes=(authorized_absent,), checkpoints=(absent_checkpoint,)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)

        wrong_image_identity = _checkpoint_identity(
            self.document,
            self.blocks,
            legacy_node_index=7,
            node_type="image",
            checkpoint_kind="node",
            selected_representation_ids=self.descriptor.selected_representation_ids,
            selected_asset_ids=self.descriptor.selected_asset_ids,
            image_bytes_sha256="9" * 64,
        )
        authorized_wrong_image = replace(
            self.descriptor,
            expected_checkpoint_identity_sha256=content_sha256(wrong_image_identity),
        )
        wrong_image = _checkpoint(
            self.document,
            self.blocks,
            authorized_wrong_image,
            identity=wrong_image_identity,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                tree_nodes=(authorized_wrong_image,), checkpoints=(wrong_image,)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_BINDING)
        self.assertIn("image checkpoint bytes", str(raised.exception))

    def test_selected_assets_require_sha256_and_digest_consistent_cas_uri(self) -> None:
        for blocks in (
            _blocks(self.document, asset_sha256=None),
            _blocks(
                self.document,
                asset_sha256="b" * 64,
                asset_uri="file:///mutable/figure.png",
            ),
        ):
            with self.subTest(uri=blocks[1].assets[0].uri):
                descriptor = _descriptor(self.document, blocks)
                checkpoint = _checkpoint(self.document, blocks, descriptor)
                with self.assertRaises(EvidenceGraphBuildError) as raised:
                    self._build(
                        canonical_blocks=blocks,
                        tree_nodes=(descriptor,),
                        checkpoints=(checkpoint,),
                    )
                self.assertEqual(
                    raised.exception.code, EvidenceGraphBuildErrorCode.MATERIAL_BINDING
                )

    def test_v1_rejects_every_unattested_fine_locator(self) -> None:
        figure = self.blocks[1]
        base = {
            "canonical_revision_id": figure.revision_id,
            "representation_id": figure.representations[0].representation_id,
        }
        for field_name, value in (
            ("char_span", (0, 2)),
            ("table_cell", (0, 0)),
            ("bbox", (0.1, 0.1, 0.2, 0.2)),
            ("diagram_element_id", "diamond-1"),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(EvidenceGraphBuildError) as raised:
                    CanonicalMaterialSelection(**base, **{field_name: value})
                self.assertEqual(
                    raised.exception.code, EvidenceGraphBuildErrorCode.MATERIAL_BINDING
                )

    def test_semantic_canonical_and_tree_binding_digests_are_enforced(self) -> None:
        binding = _binding(self.document, self.blocks, (self.descriptor,))
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=replace(binding, canonical_input_sha256="f" * 64)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.DOCUMENT_BINDING)
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=replace(binding, tree_descriptor_sha256="f" * 64)
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.TREE_BINDING)
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=replace(binding, tree_revision_id="tree-revision-other")
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.TREE_BINDING)

    def test_public_semantic_artifact_helpers_are_bounded(self) -> None:
        with self.assertRaisesRegex(
            EvidenceGraphBuildError, "canonical_input_bytes"
        ):
            b"".join(
                iter_canonical_input_artifact_bytes(
                    self.document,
                    self.blocks,
                    maximum_bytes=1,
                )
            )

        with self.assertRaisesRegex(EvidenceGraphBuildError, "semantic record bytes"):
            b"".join(
                iter_tree_descriptor_artifact_bytes(
                    (self.descriptor,),
                    tree_revision_id="tree-revision-fixture",
                    maximum_record_bytes=1,
                )
            )

    def test_artifact_digests_are_verified_from_bounded_cas_bytes(self) -> None:
        binding = _binding(self.document, self.blocks, (self.descriptor,))
        values = _artifact_bytes(self.document, self.blocks, (self.descriptor,))
        corrupted_reader = _MemoryArtifactReader(values)
        corrupted_reader.values[binding.canonical_artifact.uri] = (
            b"x" * binding.canonical_artifact.size_bytes
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(binding=binding, artifact_reader=corrupted_reader)
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY
        )

        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=binding,
                artifact_reader=_MemoryArtifactReader(values),
            )
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY
        )
        self.assertIn("selected_asset", str(raised.exception))

        canonical_bytes, tree_bytes = values
        unrelated = b"valid but unrelated canonical CAS bytes"
        unrelated_digest = hashlib.sha256(unrelated).hexdigest()
        unrelated_ref = ImmutableArtifactRef(
            uri=f"cas://sha256/{unrelated_digest}",
            sha256=unrelated_digest,
            size_bytes=len(unrelated),
        )
        unrelated_reader = _MemoryArtifactReader((unrelated, tree_bytes))
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=replace(binding, canonical_artifact=unrelated_ref),
                artifact_reader=unrelated_reader,
            )
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.DOCUMENT_BINDING
        )
        self.assertIn("semantic input snapshot", str(raised.exception))

        wrong_size = replace(
            binding.canonical_artifact,
            size_bytes=binding.canonical_artifact.size_bytes + 1,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                binding=replace(binding, canonical_artifact=wrong_size),
                artifact_reader=_MemoryArtifactReader(values),
            )
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY
        )

    def test_checkpoint_byte_limit_precedes_json_materialization(self) -> None:
        invalid_but_bounded = LegacyKGCheckpointInput(
            tree_node_id=self.descriptor.tree_node_id,
            encoded_envelope=b"{" + (b"x" * 256),
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(
                checkpoints=(invalid_but_bounded,),
                limits=EvidenceGraphBuildLimits(max_checkpoint_bytes=32),
            )
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED
        )
        self.assertIn("checkpoint_bytes", str(raised.exception))

    def test_collection_material_byte_checkpoint_and_release_limits_are_enforced(self) -> None:
        second = _descriptor(
            self.document,
            self.blocks,
            tree_node_id="opaque-tree-node-8",
            legacy_node_index=8,
        )
        cases = (
            (EvidenceGraphBuildLimits(max_nodes=1), {"tree_nodes": (self.descriptor, second)}, "tree_nodes"),
            (
                EvidenceGraphBuildLimits(max_checkpoints=1),
                {
                    "tree_nodes": (self.descriptor, second),
                    "checkpoints": (
                        self.checkpoint,
                        _checkpoint(self.document, self.blocks, second),
                    ),
                },
                "checkpoints",
            ),
            (EvidenceGraphBuildLimits(max_anchors=1), {}, "anchors"),
            (EvidenceGraphBuildLimits(max_mentions=1), {}, "mentions"),
            (EvidenceGraphBuildLimits(max_relations=2), {}, "relations"),
            (EvidenceGraphBuildLimits(max_links=4), {}, "links"),
            (EvidenceGraphBuildLimits(max_materials=2), {}, "materials"),
            (EvidenceGraphBuildLimits(max_selections=1), {}, "selections"),
            (EvidenceGraphBuildLimits(max_checkpoint_bytes=1), {}, "checkpoint_bytes"),
            (EvidenceGraphBuildLimits(max_total_checkpoint_bytes=1), {}, "total_checkpoint_bytes"),
            (EvidenceGraphBuildLimits(max_canonical_input_bytes=1), {}, "canonical_input_bytes"),
            (
                EvidenceGraphBuildLimits(max_canonical_artifact_bytes=1),
                {},
                "canonical_artifact_bytes",
            ),
            (
                EvidenceGraphBuildLimits(max_tree_artifact_bytes=1),
                {},
                "tree_artifact_bytes",
            ),
            (EvidenceGraphBuildLimits(max_tree_descriptor_bytes=1), {}, "tree_descriptor_bytes"),
            (EvidenceGraphBuildLimits(max_release_bytes=1), {}, "release_bytes"),
        )
        for limits, overrides, expected_field in cases:
            with self.subTest(expected_field=expected_field):
                with self.assertRaises(EvidenceGraphBuildError) as raised:
                    self._build(limits=limits, **overrides)
                self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.LIMIT_EXCEEDED)
                self.assertIn(expected_field, str(raised.exception))

    def test_non_finite_json_and_unequal_generated_id_collisions_fail(self) -> None:
        envelope = json.loads(self.checkpoint.encoded_envelope)
        assert isinstance(envelope, dict)
        envelope["payload"]["relations"][0]["weight"] = float("nan")
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            LegacyKGCheckpointInput.from_mapping(
                tree_node_id=self.descriptor.tree_node_id,
                envelope=envelope,
            )
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY)

        page_id = stable_id("page", self.document.document_version_id, 0)
        other_representation = Representation(
            RepresentationKind.TEXT, "Different content", "text/plain"
        )
        other_block = CanonicalBlock(
            canonical_key="collision",
            revision=1,
            document_version_id=self.document.document_version_id,
            page_id=page_id,
            page_index=0,
            block_type=BlockType.PARAGRAPH,
            representations=(other_representation,),
            source_observation_ids=("observation-collision",),
            reconciliation_method="fixture",
            quality_status=QualityStatus.ACCEPTED,
        )
        object.__setattr__(
            other_representation,
            "representation_id",
            self.blocks[0].representations[0].representation_id,
        )
        collision_blocks = (*self.blocks, other_block)
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(canonical_blocks=collision_blocks)
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.MATERIAL_BINDING)

        calls = 0

        def colliding_anchor(**kwargs: object) -> DomainEvidenceAnchor:
            nonlocal calls
            anchor = DomainEvidenceAnchor(**kwargs)  # type: ignore[arg-type]
            calls += 1
            object.__setattr__(anchor, "anchor_id", "anchor_forced_collision")
            return anchor

        with patch(
            "document_services.evidence.graph_builder.EvidenceAnchor",
            side_effect=colliding_anchor,
        ):
            with self.assertRaises(EvidenceGraphBuildError) as raised:
                self._build()
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(raised.exception.code, EvidenceGraphBuildErrorCode.MATERIAL_BINDING)

    def test_empty_artifact_chunks_bool_node_index_and_absurd_weights_fail_closed(self) -> None:
        class EmptyChunkReader(_MemoryArtifactReader):
            def iter_bytes(self, artifact, *, max_bytes):
                yield b""
                yield from super().iter_bytes(artifact, max_bytes=max_bytes)

        values = _artifact_bytes(self.document, self.blocks, (self.descriptor,))
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(artifact_reader=EmptyChunkReader(values))
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.ARTIFACT_INTEGRITY
        )
        self.assertIn("empty", str(raised.exception))

        bool_envelope = json.loads(self.checkpoint.encoded_envelope)
        bool_envelope["payload"]["node_idx"] = True
        bool_envelope["payload_sha256"] = content_sha256(bool_envelope["payload"])
        bool_checkpoint = LegacyKGCheckpointInput.from_mapping(
            tree_node_id=self.descriptor.tree_node_id,
            envelope=bool_envelope,
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(bool_checkpoint,))
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.CHECKPOINT_INTEGRITY
        )

        huge_weight = _checkpoint(
            self.document,
            self.blocks,
            self.descriptor,
            relations=[
                _relation("Alpha", "Beta", "feeds", 7, weight=10**400)
            ],
        )
        with self.assertRaises(EvidenceGraphBuildError) as raised:
            self._build(checkpoints=(huge_weight,))
        self.assertEqual(
            raised.exception.code, EvidenceGraphBuildErrorCode.INVALID_ASSERTION
        )
        self.assertIn("absolute value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
