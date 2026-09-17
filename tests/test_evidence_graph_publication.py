from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from document_services.persistence.artifact_store import LocalContentAddressedArtifactStore
from document_services.domain.base import ContractValidationError, canonical_json_bytes
from document_services.domain.knowledge import (
    EntityMention,
    EvidenceAnchor,
    EvidenceGraphRelease,
    GTLink,
    RelationAssertion,
)
from document_services.evidence.graph_checkpoint import (
    CHECKPOINT_MEDIA_TYPE,
    EvidenceGraphAdoptionAuthorization,
    EvidenceGraphCheckpointError,
    EvidenceGraphCheckpointErrorCode,
    EvidenceGraphCheckpointRef,
    EvidenceGraphCheckpointStore,
)
from document_services.evidence.projections.legacy_graph import (
    LegacyGraphProjectionCompatibilityError,
    LegacyGraphProjectionLimitError,
    LegacyProjectionLimits,
    LegacyProjectionLossCode,
    LegacyProjectionPolicy,
    LossyLegacyGraphProjectionError,
    project_bookrag_legacy_graph,
)


_TREE_ALIASES = {"tree-table": 10, "tree-diagram": 11}


class _ExactAdoptionAuthority:
    def authorization_for(
        self, reference: EvidenceGraphCheckpointRef
    ) -> EvidenceGraphAdoptionAuthorization:
        return EvidenceGraphAdoptionAuthorization.from_reference(
            reference,
            authority_id="unit-test-adoption-ledger",
        )


class _FixedAdoptionAuthority:
    def __init__(self, authorization: EvidenceGraphAdoptionAuthorization) -> None:
        self.authorization = authorization

    def authorization_for(
        self, reference: EvidenceGraphCheckpointRef
    ) -> EvidenceGraphAdoptionAuthorization:
        del reference
        return self.authorization


class _DenyingAdoptionAuthority:
    def authorization_for(
        self, reference: EvidenceGraphCheckpointRef
    ) -> EvidenceGraphAdoptionAuthorization | None:
        del reference
        return None


def _release() -> EvidenceGraphRelease:
    table_anchor = EvidenceAnchor(
        canonical_revision_id="canonical-table-v1",
        representation_id="representation-table-grid",
        table_cell=(1, 2),
    )
    diagram_anchor = EvidenceAnchor(
        canonical_revision_id="canonical-diagram-v1",
        asset_id="asset-diagram",
        page_id="page-2",
        bbox=(0.1, 0.2, 0.8, 0.9),
        diagram_element_id="decision-1",
    )
    revenue = EntityMention(
        label="Revenue",
        entity_type="metric",
        extraction_revision_id="extractor-table-v1",
        evidence_anchor_ids=(table_anchor.anchor_id,),
        extraction_confidence=0.92,
        attributes={"description": "Reported revenue"},
    )
    approval = EntityMention(
        label="Approve budget",
        entity_type="process_step",
        extraction_revision_id="extractor-diagram-v1",
        evidence_anchor_ids=(diagram_anchor.anchor_id,),
        extraction_confidence=0.81,
    )
    forward_a = RelationAssertion(
        source_mention_id=revenue.mention_id,
        target_mention_id=approval.mention_id,
        relation_type="feeds",
        extraction_revision_id="extractor-diagram-v1",
        evidence_anchor_ids=(diagram_anchor.anchor_id,),
        extraction_confidence=0.88,
        assertion_key="diagram-edge-1",
    )
    forward_b = RelationAssertion(
        source_mention_id=revenue.mention_id,
        target_mention_id=approval.mention_id,
        relation_type="supports",
        extraction_revision_id="extractor-diagram-v1",
        evidence_anchor_ids=(diagram_anchor.anchor_id,),
        extraction_confidence=0.77,
        assertion_key="diagram-edge-2",
    )
    reverse = RelationAssertion(
        source_mention_id=approval.mention_id,
        target_mention_id=revenue.mention_id,
        relation_type="feeds",
        extraction_revision_id="extractor-table-v1",
        evidence_anchor_ids=(table_anchor.anchor_id,),
        extraction_confidence=0.61,
        assertion_key="table-edge-3",
    )
    mention_link = GTLink(
        graph_object_id=revenue.mention_id,
        tree_node_id="tree-table",
        role="defined_in",
        evidence_anchor_ids=(table_anchor.anchor_id,),
        link_confidence=0.96,
    )
    relation_link = GTLink(
        graph_object_id=forward_a.relation_id,
        tree_node_id="tree-diagram",
        role="asserted_in",
        evidence_anchor_ids=(diagram_anchor.anchor_id,),
        link_confidence=0.89,
    )
    return EvidenceGraphRelease(
        document_id="document-1",
        document_version_id="document-version-1",
        source_sha256="0" * 64,
        build_spec_sha256="1" * 64,
        canonical_artifact_sha256="2" * 64,
        tree_revision_id="tree-revision-1",
        tree_artifact_sha256="3" * 64,
        graph_revision_id="graph-revision-1",
        graph_artifact_sha256="4" * 64,
        mapping_revision_id="mapping-revision-1",
        mapping_artifact_sha256="5" * 64,
        canonical_revision_ids=("canonical-table-v1", "canonical-diagram-v1"),
        representation_ids=("representation-table-grid",),
        asset_ids=("asset-diagram",),
        tree_node_ids=("tree-table", "tree-diagram"),
        anchors=(table_anchor, diagram_anchor),
        mentions=(revenue, approval),
        relations=(forward_a, forward_b, reverse),
        gt_links=(mention_link, relation_link),
        metadata={"producer": "unit-test"},
    )


class EvidenceGraphCheckpointTests(unittest.TestCase):
    def test_atomic_publication_is_content_addressed_and_idempotent(self) -> None:
        release = _release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence-graph-checkpoints"
            store = EvidenceGraphCheckpointStore(
                root,
                adoption_authority=_ExactAdoptionAuthority(),
            )

            first = store.publish(release)
            second = store.publish(release)

            self.assertEqual(first.reference, second.reference)
            self.assertEqual(first.path, second.path)
            self.assertEqual(first.path.read_bytes(), canonical_json_bytes(release.to_dict()))
            self.assertEqual(
                first.reference.checkpoint_sha256,
                hashlib.sha256(first.path.read_bytes()).hexdigest(),
            )
            self.assertEqual(first.reference.checkpoint_sha256, first.path.name)
            self.assertEqual(0o400, first.path.stat().st_mode & 0o777)
            self.assertEqual(release, store.read(first.reference))
            self.assertEqual(release, store.adopt(first.reference, expected_release=release))

    def test_reference_round_trip_carries_exact_source_canonical_and_tgm_binding(self) -> None:
        release = _release()
        reference = EvidenceGraphCheckpointStore.reference_for(release)
        restored = EvidenceGraphCheckpointRef.from_json(reference.to_json())
        authorization = EvidenceGraphAdoptionAuthorization.from_reference(
            reference,
            authority_id="unit-test-adoption-ledger",
        )
        restored_authorization = EvidenceGraphAdoptionAuthorization.from_json(
            authorization.to_json()
        )

        self.assertEqual(reference, restored)
        self.assertEqual(authorization, restored_authorization)
        self.assertEqual(reference.release_id, authorization.release_id)
        self.assertEqual(
            reference.release_content_sha256,
            authorization.release_content_sha256,
        )
        self.assertEqual(reference.checkpoint_sha256, authorization.checkpoint_sha256)
        self.assertEqual(
            reference.binding.binding_sha256,
            authorization.binding_sha256,
        )
        binding = restored.binding
        self.assertEqual(release.source_sha256, binding.source_sha256)
        self.assertEqual(release.canonical_artifact_sha256, binding.canonical_artifact_sha256)
        self.assertEqual(release.canonical_revision_ids, binding.canonical_revision_ids)
        self.assertEqual(release.tree_revision_id, binding.tree_revision_id)
        self.assertEqual(release.tree_artifact_sha256, binding.tree_artifact_sha256)
        self.assertEqual(release.graph_revision_id, binding.graph_revision_id)
        self.assertEqual(release.graph_artifact_sha256, binding.graph_artifact_sha256)
        self.assertEqual(release.mapping_revision_id, binding.mapping_revision_id)
        self.assertEqual(release.mapping_artifact_sha256, binding.mapping_artifact_sha256)

        tampered = reference.to_dict()
        tampered["binding"]["tree_revision_id"] = "other-tree"  # type: ignore[index]
        with self.assertRaisesRegex(ContractValidationError, "binding_sha256"):
            EvidenceGraphCheckpointRef.from_dict(tampered)

        tampered_authorization = authorization.to_dict()
        tampered_authorization["binding_sha256"] = "f" * 64
        with self.assertRaisesRegex(ContractValidationError, "authorization_id"):
            EvidenceGraphAdoptionAuthorization.from_dict(tampered_authorization)

    def test_concurrent_publishers_converge_on_one_verified_checkpoint(self) -> None:
        release = _release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = EvidenceGraphCheckpointStore(root)
            with ThreadPoolExecutor(max_workers=8) as executor:
                publications = tuple(executor.map(store.publish, (release,) * 24))

            self.assertEqual(1, len({item.reference for item in publications}))
            self.assertEqual(1, len({item.path for item in publications}))
            final_files = [path for path in root.rglob("*") if path.is_file()]
            self.assertEqual([publications[0].path], final_files)
            self.assertFalse(
                any(path.name.startswith(".publish-") for path in root.rglob("*"))
            )
            self.assertEqual(release, store.read(publications[0].reference))

    def test_adoption_rejects_any_different_release_or_binding(self) -> None:
        release = _release()
        changed = replace(
            release,
            tree_revision_id="tree-revision-2",
            tree_artifact_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceGraphCheckpointStore(Path(directory) / "checkpoints")
            published = store.publish(release)

            with self.assertRaisesRegex(
                EvidenceGraphCheckpointError, "exact expected release"
            ):
                store.adopt(published.reference, expected_release=changed)

            changed_binding = replace(
                published.reference.binding,
                graph_revision_id="graph-revision-other",
            )
            wrong_reference = replace(published.reference, binding=changed_binding)
            with self.assertRaisesRegex(EvidenceGraphCheckpointError, "binding"):
                store.read(wrong_reference)

    def test_adoption_requires_external_exact_authorization(self) -> None:
        release = _release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = EvidenceGraphCheckpointStore(root)
            published = store.publish(release)

            with self.assertRaises(EvidenceGraphCheckpointError) as missing:
                store.adopt(published.reference, expected_release=release)
            self.assertEqual(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORITY_REQUIRED,
                missing.exception.code,
            )

            authorization = EvidenceGraphAdoptionAuthorization.from_reference(
                published.reference,
                authority_id="unit-test-adoption-ledger",
            )
            mismatches = (
                replace(
                    authorization,
                    release_id="kgrelease_" + "0" * 32,
                ),
                replace(authorization, release_content_sha256="a" * 64),
                replace(authorization, checkpoint_sha256="b" * 64),
                replace(authorization, binding_sha256="c" * 64),
            )
            for mismatch in mismatches:
                with self.subTest(field=mismatch.authorization_id):
                    authorized_store = EvidenceGraphCheckpointStore(
                        root,
                        create=False,
                        adoption_authority=_FixedAdoptionAuthority(mismatch),
                    )
                    with self.assertRaises(EvidenceGraphCheckpointError) as rejected:
                        authorized_store.adopt(
                            published.reference,
                            expected_release=release,
                        )
                    self.assertEqual(
                        EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_MISMATCH,
                        rejected.exception.code,
                    )

            denying_store = EvidenceGraphCheckpointStore(
                root,
                create=False,
                adoption_authority=_DenyingAdoptionAuthority(),
            )
            with self.assertRaises(EvidenceGraphCheckpointError) as denied:
                denying_store.adopt(published.reference, expected_release=release)
            self.assertEqual(
                EvidenceGraphCheckpointErrorCode.ADOPTION_AUTHORIZATION_ERROR,
                denied.exception.code,
            )

    def test_inspection_only_builder_release_can_publish_but_cannot_be_adopted(self) -> None:
        release = replace(
            _release(),
            metadata={
                "builder": {
                    "evidence_precision": "coarse_whole_node",
                    "authority_eligible": False,
                    "authority_ineligibility_reason": "missing exact material-use receipts",
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceGraphCheckpointStore(Path(directory) / "checkpoints")
            published = store.publish(release)

            self.assertEqual(release, store.read(published.reference))
            with self.assertRaisesRegex(
                EvidenceGraphCheckpointError, "inspection-only"
            ):
                store.adopt(published.reference, expected_release=release)

    def test_corrupt_or_domain_invalid_checkpoint_fails_closed(self) -> None:
        release = _release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = EvidenceGraphCheckpointStore(root)
            published = store.publish(release)

            os.chmod(published.path, 0o600)
            published.path.write_bytes(b"{}")
            with self.assertRaises(EvidenceGraphCheckpointError):
                store.read(published.reference)

        # A hash-valid CAS object is still rejected when its domain IDs or links
        # do not validate.  Byte integrity alone is not adoption authority.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = EvidenceGraphCheckpointStore(root)
            payload = release.to_dict()
            payload["mentions"][0]["evidence_anchor_ids"] = ["missing-anchor"]  # type: ignore[index]
            malformed = canonical_json_bytes(payload)
            raw_store = LocalContentAddressedArtifactStore(root, create=False)
            stored = raw_store.publish(malformed, media_type=CHECKPOINT_MEDIA_TYPE)
            reference = replace(
                EvidenceGraphCheckpointStore.reference_for(release),
                checkpoint_sha256=stored.sha256,
                size_bytes=stored.size_bytes,
            )
            with self.assertRaisesRegex(EvidenceGraphCheckpointError, "valid evidence-graph"):
                store.read(reference)

    def test_huge_integer_confidence_is_a_typed_invalid_checkpoint(self) -> None:
        release = _release()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = EvidenceGraphCheckpointStore(root)
            payload = release.to_dict()
            payload["mentions"][0]["extraction_confidence"] = 10**400  # type: ignore[index]
            malformed = canonical_json_bytes(payload)
            raw_store = LocalContentAddressedArtifactStore(root, create=False)
            stored = raw_store.publish(malformed, media_type=CHECKPOINT_MEDIA_TYPE)
            reference = replace(
                EvidenceGraphCheckpointStore.reference_for(release),
                checkpoint_sha256=stored.sha256,
                size_bytes=stored.size_bytes,
            )

            with self.assertRaises(EvidenceGraphCheckpointError) as raised:
                store.read(reference)
            self.assertEqual(
                EvidenceGraphCheckpointErrorCode.INVALID_RELEASE,
                raised.exception.code,
            )


class LegacyEvidenceGraphProjectionTests(unittest.TestCase):
    @staticmethod
    def _mention_only_links(release: EvidenceGraphRelease) -> tuple[GTLink, ...]:
        mention_ids = {mention.mention_id for mention in release.mentions}
        return tuple(
            link for link in release.gt_links if link.graph_object_id in mention_ids
        )

    def test_strict_projection_rejects_directed_parallel_semantic_loss(self) -> None:
        with self.assertRaises(LossyLegacyGraphProjectionError) as raised:
            project_bookrag_legacy_graph(
                _release(), tree_node_aliases=_TREE_ALIASES
            )

        codes = {loss.code for loss in raised.exception.losses}
        self.assertIn(LegacyProjectionLossCode.DIRECTED_ASSERTIONS, codes)
        self.assertIn(LegacyProjectionLossCode.PARALLEL_ASSERTIONS, codes)
        self.assertIn(LegacyProjectionLossCode.OPPOSITE_DIRECTIONS, codes)
        self.assertIn(LegacyProjectionLossCode.MULTIPLE_RELATION_TYPES, codes)
        self.assertIn(LegacyProjectionLossCode.RELATION_TREE_LINKS, codes)
        self.assertIn(LegacyProjectionLossCode.OPAQUE_NODE_IDS, codes)

    def test_marked_projection_retains_every_assertion_direction_and_gt_link(self) -> None:
        release = _release()
        result = project_bookrag_legacy_graph(
            release,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        payload = result.to_dict()

        self.assertTrue(result.lossy)
        self.assertTrue(payload["lossy"])
        self.assertTrue(payload["read_only_derived"])
        self.assertEqual(release.release_id, payload["authority"]["release_id"])
        self.assertEqual(release.content_sha256, payload["authority"]["release_content_sha256"])
        self.assertFalse(payload["graph"]["directed"])
        self.assertFalse(payload["graph"]["multigraph"])
        self.assertEqual(2, len(payload["graph"]["nodes"]))
        self.assertEqual(1, len(payload["graph"]["links"]))
        self.assertFalse(result.retrieval_safe)
        self.assertFalse(payload["retrieval_use"]["safe"])
        self.assertIn(
            LegacyProjectionLossCode.OPAQUE_NODE_IDS,
            {loss.code for loss in result.losses},
        )
        self.assertEqual(
            {anchor.anchor_id for anchor in release.anchors},
            set(payload["evidence_anchor_catalog"]),
        )

        bundle = payload["graph"]["links"][0]
        self.assertEqual(3, bundle["assertion_count"])
        assertions = bundle["assertions"]
        relation_payloads = [item["relation"] for item in assertions]
        self.assertEqual(
            {relation.relation_id for relation in release.relations},
            {item["relation_id"] for item in relation_payloads},
        )
        self.assertEqual(
            {
                (relation.source_mention_id, relation.target_mention_id)
                for relation in release.relations
            },
            {
                (item["source_mention_id"], item["target_mention_id"])
                for item in relation_payloads
            },
        )
        projected_gt_links = {
            link["gt_link_id"]
            for node in payload["graph"]["nodes"]
            for link in node["gt_links"]
        } | {
            link["gt_link_id"]
            for assertion in assertions
            for link in assertion["gt_links"]
        }
        self.assertEqual(
            {link.gt_link_id for link in release.gt_links}, projected_gt_links
        )
        self.assertEqual(
            [
                next(
                    link.graph_object_id
                    for link in release.gt_links
                    if link.graph_object_id
                    in {relation.relation_id for relation in release.relations}
                )
            ],
            payload["relation_tree2kg"]["11"],
        )
        self.assertEqual(_TREE_ALIASES, payload["tree_node_aliases"])
        # The projection is dependency-free ordinary JSON, not NetworkX/Pydantic.
        self.assertEqual(payload, json.loads(result.to_json()))
        json.dumps(payload)

        # Verify the compatibility shape against BookRAG's actual loader.  The
        # resulting graph is read-only: assertions remain source DTOs on the
        # bundled legacy edge, while the evidence release stays authoritative.
        try:
            from retrieval_engine.indexes.graph import Graph
        except ModuleNotFoundError as exc:
            if exc.name == "networkx":
                self.skipTest("BookRAG runtime dependencies are not installed")
            raise
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "graph_data.json"
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            loaded = Graph.load_from_dir(directory)
            self.assertEqual(2, loaded.kg.number_of_nodes())
            self.assertEqual(1, loaded.kg.number_of_edges())
            self.assertEqual(3, next(iter(loaded.kg.edges(data=True)))[2]["assertion_count"])
            self.assertEqual(
                payload["tree2kg"]["10"], sorted(loaded.tree2kg[10])
            )

    def test_empty_relation_graph_remains_inspection_only_and_traceable(self) -> None:
        release = _release()
        mention_links = tuple(
            link for link in release.gt_links if link.graph_object_id in {
                mention.mention_id for mention in release.mentions
            }
        )
        lossless = replace(release, relations=(), gt_links=mention_links)

        with self.assertRaises(LossyLegacyGraphProjectionError) as raised:
            project_bookrag_legacy_graph(lossless, tree_node_aliases=_TREE_ALIASES)
        self.assertIn(
            LegacyProjectionLossCode.OPAQUE_NODE_IDS,
            {loss.code for loss in raised.exception.losses},
        )

        result = project_bookrag_legacy_graph(
            lossless,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        payload = result.to_dict()

        self.assertTrue(result.lossy)
        self.assertFalse(result.retrieval_safe)
        self.assertFalse(payload["retrieval_use"]["safe"])
        self.assertEqual([], payload["graph"]["links"])
        self.assertEqual(
            [mention_links[0].graph_object_id], payload["tree2kg"]["10"]
        )
        with self.assertRaises(LossyLegacyGraphProjectionError):
            result.to_retrieval_dict()

    def test_single_legacy_assertion_preserves_exact_description_weight_and_raw_name(self) -> None:
        release = _release()
        mention = release.mentions[0]
        relation = RelationAssertion(
            source_mention_id=mention.mention_id,
            target_mention_id=mention.mention_id,
            relation_type="legacy_unspecified",
            extraction_revision_id="legacy-checkpoint-v1",
            evidence_anchor_ids=mention.evidence_anchor_ids,
            attributes={
                "legacy_relation_name": "",
                "description": "Exact legacy relation description",
                "weight": 2.75,
            },
        )
        exact_release = replace(
            release,
            relations=(relation,),
            gt_links=self._mention_only_links(release),
        )

        result = project_bookrag_legacy_graph(
            exact_release,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        payload = result.to_dict()
        edge = payload["graph"]["links"][0]

        self.assertTrue(result.lossy)
        self.assertFalse(result.retrieval_safe)
        self.assertFalse(payload["retrieval_use"]["safe"])
        self.assertEqual("reject", payload["retrieval_use"]["default_action"])
        self.assertEqual(mention.mention_id, edge["source"])
        self.assertEqual(mention.mention_id, edge["target"])
        self.assertEqual("", edge["relation_name"])
        self.assertEqual("Exact legacy relation description", edge["description"])
        self.assertEqual(2.75, edge["weight"])
        self.assertTrue(edge["legacy_edge_fields_exact"])
        self.assertFalse(edge["docarag_retrieval_safe"])
        self.assertEqual(
            {
                "relation_name": "",
                "description": "Exact legacy relation description",
                "weight": 2.75,
                "exact": True,
                "field_exactness": {
                    "relation_name": True,
                    "description": True,
                    "weight": True,
                },
            },
            edge["assertions"][0]["legacy_compatibility_fields"],
        )
        with self.assertRaises(LossyLegacyGraphProjectionError):
            result.to_retrieval_dict()

    def test_marked_directed_assertion_keeps_fields_but_retrieval_export_fails_closed(
        self,
    ) -> None:
        release = _release()
        original = release.relations[0]
        relation = replace(
            original,
            attributes={
                "legacy_relation_name": "feeds_raw",
                "description": "Revenue feeds the approval step",
                "weight": 4,
            },
        )
        directed_release = replace(
            release,
            relations=(relation,),
            gt_links=self._mention_only_links(release),
        )

        result = project_bookrag_legacy_graph(
            directed_release,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        payload = result.to_dict()
        edge = payload["graph"]["links"][0]

        self.assertFalse(result.retrieval_safe)
        self.assertFalse(payload["retrieval_use"]["safe"])
        self.assertEqual("reject", payload["retrieval_use"]["default_action"])
        self.assertEqual(relation.source_mention_id, edge["source"])
        self.assertEqual(relation.target_mention_id, edge["target"])
        self.assertEqual("feeds_raw", edge["relation_name"])
        self.assertEqual("Revenue feeds the approval step", edge["description"])
        self.assertEqual(4, edge["weight"])
        self.assertTrue(edge["legacy_edge_fields_exact"])
        self.assertFalse(edge["docarag_retrieval_safe"])
        with self.assertRaises(LossyLegacyGraphProjectionError):
            result.to_retrieval_dict()

    def test_parallel_field_collapse_is_exactly_marked_and_assertions_remain_authoritative(
        self,
    ) -> None:
        release = _release()
        first = replace(
            release.relations[0],
            attributes={
                "legacy_relation_name": "feeds",
                "description": "First exact description",
                "weight": 1.25,
            },
        )
        second = replace(
            release.relations[1],
            attributes={
                "legacy_relation_name": "supports",
                "description": "Second exact description",
                "weight": 8.5,
            },
        )
        parallel_release = replace(
            release,
            relations=(first, second),
            gt_links=self._mention_only_links(release),
        )

        result = project_bookrag_legacy_graph(
            parallel_release,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        payload = result.to_dict()
        edge = payload["graph"]["links"][0]
        codes = {loss.code for loss in result.losses}

        self.assertIn(LegacyProjectionLossCode.PARALLEL_ASSERTIONS, codes)
        self.assertIn(LegacyProjectionLossCode.RELATION_NAME_COLLAPSE, codes)
        self.assertIn(
            LegacyProjectionLossCode.RELATION_DESCRIPTION_COLLAPSE, codes
        )
        self.assertIn(LegacyProjectionLossCode.RELATION_WEIGHT_COLLAPSE, codes)
        self.assertEqual("__see_assertions__", edge["relation_name"])
        self.assertEqual("__see_assertions__", edge["description"])
        self.assertEqual(0.0, edge["weight"])
        self.assertFalse(edge["legacy_edge_fields_exact"])
        self.assertEqual(
            {
                ("feeds", "First exact description", 1.25),
                ("supports", "Second exact description", 8.5),
            },
            {
                (
                    assertion["legacy_compatibility_fields"]["relation_name"],
                    assertion["legacy_compatibility_fields"]["description"],
                    assertion["legacy_compatibility_fields"]["weight"],
                )
                for assertion in edge["assertions"]
            },
        )
        with self.assertRaises(LossyLegacyGraphProjectionError):
            result.to_retrieval_dict()

    def test_opaque_tree_ids_require_a_total_bijective_legacy_alias_map(self) -> None:
        release = _release()
        with self.assertRaisesRegex(
            LegacyGraphProjectionCompatibilityError, "explicit tree_node_aliases"
        ):
            project_bookrag_legacy_graph(
                release, policy=LegacyProjectionPolicy.MARK
            )
        with self.assertRaisesRegex(
            LegacyGraphProjectionCompatibilityError, "cover exactly"
        ):
            project_bookrag_legacy_graph(
                release,
                policy=LegacyProjectionPolicy.MARK,
                tree_node_aliases={"tree-table": 10},
            )
        with self.assertRaisesRegex(
            LegacyGraphProjectionCompatibilityError, "one-to-one"
        ):
            project_bookrag_legacy_graph(
                release,
                policy=LegacyProjectionPolicy.MARK,
                tree_node_aliases={"tree-table": 10, "tree-diagram": 10},
            )

    def test_projection_limits_reject_before_compatibility_expansion(self) -> None:
        release = _release()
        cases = (
            ("max_nodes", 1, "nodes count"),
            ("max_relations", 1, "relations count"),
            ("max_anchors", 1, "anchors count"),
            ("max_gt_links", 1, "GT-links count"),
            ("max_tree_nodes", 1, "tree nodes count"),
            ("max_anchor_references", 1, "anchor reference count"),
        )
        for field_name, maximum, message in cases:
            with self.subTest(field=field_name):
                limits = replace(
                    LegacyProjectionLimits(), **{field_name: maximum}
                )
                with self.assertRaisesRegex(
                    LegacyGraphProjectionLimitError, message
                ):
                    # No aliases are supplied.  A limit failure must occur before
                    # tree compatibility mapping or JSON payload expansion.
                    project_bookrag_legacy_graph(
                        release,
                        policy=LegacyProjectionPolicy.MARK,
                        limits=limits,
                    )

        tiny_bytes = replace(
            LegacyProjectionLimits(),
            max_estimated_output_bytes=1,
            max_output_bytes=1,
        )
        with self.assertRaisesRegex(
            LegacyGraphProjectionLimitError, "estimated output"
        ):
            project_bookrag_legacy_graph(
                release,
                policy=LegacyProjectionPolicy.MARK,
                tree_node_aliases=_TREE_ALIASES,
                limits=tiny_bytes,
            )

    def test_projection_bounds_ignored_release_metadata_before_clone(self) -> None:
        release = replace(_release(), metadata={"padding": "x" * 4096})
        limits = replace(
            LegacyProjectionLimits(),
            max_release_metadata_bytes=1024,
        )

        with self.assertRaisesRegex(
            LegacyGraphProjectionLimitError, "release metadata size"
        ):
            # Missing aliases would fail later.  The metadata limit must win
            # before the release is cloned or compatibility is evaluated.
            project_bookrag_legacy_graph(
                release,
                policy=LegacyProjectionPolicy.MARK,
                limits=limits,
            )

    def test_projection_counts_metadata_in_estimated_memory_budget(self) -> None:
        release = replace(_release(), metadata={"padding": "x" * 4096})
        limits = replace(
            LegacyProjectionLimits(),
            max_release_metadata_bytes=8192,
            max_estimated_output_bytes=2048,
            max_output_bytes=2048,
        )

        with self.assertRaisesRegex(
            LegacyGraphProjectionLimitError, "estimated output"
        ):
            project_bookrag_legacy_graph(
                release,
                policy=LegacyProjectionPolicy.MARK,
                limits=limits,
            )

    def test_absurd_integer_weight_is_marked_invalid_without_float_overflow(self) -> None:
        release = _release()
        original = release.relations[0]
        relation = replace(
            original,
            attributes={
                "legacy_relation_name": "feeds",
                "description": "bounded compatibility test",
                "weight": 10**400,
            },
        )
        candidate = replace(
            release,
            relations=(relation,),
            gt_links=self._mention_only_links(release),
        )

        result = project_bookrag_legacy_graph(
            candidate,
            policy=LegacyProjectionPolicy.MARK,
            tree_node_aliases=_TREE_ALIASES,
        )
        edge = result.to_dict()["graph"]["links"][0]

        self.assertIn(
            LegacyProjectionLossCode.INVALID_EDGE_COMPATIBILITY_FIELDS,
            {loss.code for loss in result.losses},
        )
        self.assertFalse(edge["legacy_edge_fields_exact"])
        self.assertFalse(
            edge["assertions"][0]["legacy_compatibility_fields"][
                "field_exactness"
            ]["weight"]
        )
        with self.assertRaises(LossyLegacyGraphProjectionError):
            result.to_retrieval_dict()


if __name__ == "__main__":
    unittest.main()
