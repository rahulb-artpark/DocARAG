from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from retrieval_engine.indexes.tree import DocumentTree, NodeType, TreeNode
from retrieval_engine.indexing.tree import builder as doc_tree_builder
from retrieval_engine.indexing.tree.builder import (
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    CanonicalIngestionBindingError,
    CanonicalIngestionPolicy,
    CanonicalIngestionResult,
    build_tree_from_pdf,
)
from document_services.domain.base import CONTRACT_SCHEMA_VERSION
from document_services.evidence.projections.bookrag import project_to_bookrag_pdf_list
from document_services.domain import (
    BlockType,
    BoundingBox,
    CanonicalBlock,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    ParseRequest,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
)


class RecordingCanonicalProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, bool]] = []

    def ingest(self, cfg, *, reforce: bool = False):
        self.calls.append((cfg, reforce))
        return self.result


class _CapturedStage:
    enabled = True
    replayed = False

    def __init__(self, name: str, calls: list[tuple[str, object]]) -> None:
        self.name = name
        self.calls = calls

    def __enter__(self):
        self.calls.append(("enter", self.name))
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.calls.append(("exit", self.name))
        return False

    def checkpoint(self, name, payload=None):
        self.calls.append((f"checkpoint:{self.name}", name))

    def record_artifact(self, path, *, artifact_role, **kwargs):
        self.calls.append((f"artifact:{self.name}", artifact_role))

    def record_not_emitted(self, *, artifact_role, **kwargs):
        self.calls.append((f"not-emitted:{self.name}", artifact_role))


class BookRAGIngestionSeamTests(unittest.TestCase):
    def make_cfg(self, root: Path):
        source_bytes = b"BookRAG ingestion seam fixture"
        source_path = root / "source.pdf"
        source_path.write_bytes(source_bytes)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        return SimpleNamespace(
            source=SimpleNamespace(
                display_name="source.pdf",
                original_uri="file:///original/source.pdf",
                sha256=source_sha256,
                snapshot_path=str(source_path),
            ),
            pdf_path=str(source_path),
            save_path=str(root / "run"),
            tree=SimpleNamespace(use_vlm=False, node_summary=False),
            llm=SimpleNamespace(),
            vlm=SimpleNamespace(),
            mineru=SimpleNamespace(
                backend="pipeline",
                method="auto",
                server_url=None,
                lang="en",
                shard_pages=10,
            ),
        )

    @staticmethod
    def document(cfg, *, source_sha256: str | None = None) -> DocumentRef:
        return DocumentRef.identified(
            content_sha256=source_sha256 or cfg.source.sha256,
            mime_type="application/pdf",
            source_uri=cfg.source.original_uri,
            display_name=cfg.source.display_name,
            page_count=1,
        )

    @staticmethod
    def block(
        document: DocumentRef,
        *,
        quality_status: QualityStatus = QualityStatus.ACCEPTED,
    ) -> CanonicalBlock:
        text = Representation(
            kind=RepresentationKind.TEXT,
            content="Canonical routed content",
            media_type="text/plain",
            confidence=0.95,
        )
        return CanonicalBlock(
            canonical_key="canonical-ingestion-fixture",
            revision=2,
            document_version_id=document.document_version_id,
            page_id=PageRef(document.document_version_id, 0).page_id,
            page_index=0,
            block_type=BlockType.PARAGRAPH,
            representations=(text,),
            source_observation_ids=("obs_routed_fixture",),
            reconciliation_method="routed_fixture.v1",
            quality_status=quality_status,
            geometry=SourceGeometry(
                coordinate_space=CoordinateSpace.NORMALIZED,
                bbox=BoundingBox(0.1, 0.2, 0.8, 0.3),
            ),
            reading_order=1.0,
            metadata={"source_sha256": document.content_sha256},
        )

    @staticmethod
    def artifact(root: Path, document: DocumentRef, block: CanonicalBlock) -> Path:
        path = root / "canonical-result.json"
        path.write_text(
            json.dumps(
                {
                    "document": document.to_dict(),
                    "canonical_blocks": [block.to_dict()],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def pipeline_patches(self, captured: dict[str, object]):
        def refine(pdf_list, llm):
            captured["refiner_input"] = pdf_list
            captured["llm"] = llm
            return pdf_list

        def outline(pdf_list, llm):
            captured["outline_input"] = pdf_list
            return [], frozenset()

        def construct(*, tree_index, pdf_list, title_outline, rejected_title_ids):
            captured["tree_input"] = pdf_list
            captured["title_outline"] = title_outline
            captured["rejected_title_ids"] = rejected_title_ids
            return tree_index

        tracker = mock.Mock()
        tracker.record_stage.return_value = {"total": 0}
        return (
            mock.patch.object(doc_tree_builder, "LLM", return_value="llm-sentinel"),
            mock.patch.object(doc_tree_builder, "pdf_info_refiner", side_effect=refine),
            mock.patch.object(
                doc_tree_builder,
                "extract_pdf_outline_in_chunks",
                side_effect=outline,
            ),
            mock.patch.object(
                doc_tree_builder,
                "construct_tree_index",
                side_effect=construct,
            ),
            mock.patch.object(
                doc_tree_builder.TokenTracker,
                "get_instance",
                return_value=tracker,
            ),
            mock.patch.object(doc_tree_builder.DocumentTree, "save_to_file"),
        )

    @staticmethod
    def cached_tree(
        cfg,
        document: DocumentRef,
        artifact: Path,
        *,
        canonical_revision_id: str,
    ) -> DocumentTree:
        metadata = {
            "file_name": document.display_name,
            "file_path": document.source_uri,
            "source_uri": document.source_uri,
            "source_sha256": document.content_sha256,
            "snapshot_path": cfg.pdf_path,
            "document_version_id": document.document_version_id,
            "canonical_artifact_path": str(artifact.resolve()),
            "canonical_artifact_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
            "canonical_artifact_size_bytes": artifact.stat().st_size,
        }
        tree = DocumentTree(meta_dict=metadata, cfg=cfg)
        node = TreeNode(
            {
                **metadata,
                "pdf_id": 1,
                "page_idx": 0,
                "canonical_revision_id": canonical_revision_id,
                "content": "cached canonical content",
            }
        )
        node.type = NodeType.TEXT
        tree.add_node(node)
        tree.root_node.add_child(node)
        Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
        tree.save_to_file()
        return tree

    def test_cached_tree_reuse_is_bound_to_source_and_current_provider_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            document = self.document(cfg)
            block = self.block(document)
            artifact = self.artifact(root, document, block)
            expected_tree = self.cached_tree(
                cfg,
                document,
                artifact,
                canonical_revision_id=block.revision_id,
            )
            provider = RecordingCanonicalProvider(
                SimpleNamespace(
                    document=document,
                    canonical_blocks=(block,),
                    canonical_artifact_path=artifact,
                )
            )

            restored = build_tree_from_pdf(cfg, parsing_provider=provider)

            self.assertEqual(document.document_version_id, restored.meta_info.document_version_id)
            self.assertEqual([(cfg, False)], provider.calls)
            self.assertEqual(
                expected_tree.get_nodes()[0].meta_info.canonical_revision_id,
                restored.get_nodes()[0].meta_info.canonical_revision_id,
            )

            restored.get_nodes()[0].meta_info.canonical_revision_id = "stale-revision"
            restored.save_to_file()
            with self.assertRaisesRegex(
                CanonicalIngestionBindingError,
                "canonical lineage differs",
            ):
                build_tree_from_pdf(cfg, parsing_provider=provider)

            cfg.source.sha256 = "f" * 64
            with self.assertRaisesRegex(
                CanonicalIngestionBindingError,
                "source_sha256 differs",
            ):
                build_tree_from_pdf(cfg)

    def test_injected_provider_projects_canonical_blocks_into_unchanged_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            document = self.document(cfg)
            block = self.block(document)
            artifact = self.artifact(root, document, block)
            audit = root / "routed-parsing-audit.json"
            audit.write_bytes(b'{"schema":"routed-audit-fixture.v1"}\n')
            provider = RecordingCanonicalProvider(
                SimpleNamespace(
                    document=document,
                    canonical_blocks=(block,),
                    canonical_artifact_path=artifact,
                    audit_artifact_path=audit,
                )
            )
            expected = project_to_bookrag_pdf_list(document, (block,))
            captured: dict[str, object] = {}
            stage_calls: list[tuple[str, object]] = []
            patches = self.pipeline_patches(captured)

            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "parse_doc",
                        side_effect=AssertionError("MinerU must be bypassed"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "merge_middle_content",
                        side_effect=AssertionError("MinerU must be bypassed"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "bind_canonical_evidence",
                        side_effect=AssertionError("MinerU binding must be bypassed"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "durable_stage",
                        side_effect=lambda name, **_kwargs: _CapturedStage(
                            name, stage_calls
                        ),
                    )
                )
                tree = build_tree_from_pdf(
                    cfg,
                    reforce=True,
                    parsing_provider=provider,
                )

            self.assertEqual([(cfg, True)], provider.calls)
            self.assertEqual(expected, captured["refiner_input"])
            self.assertIs(captured["refiner_input"], captured["outline_input"])
            self.assertIs(captured["outline_input"], captured["tree_input"])
            self.assertEqual([], captured["title_outline"])
            self.assertEqual(block.block_id, expected[0]["canonical_block_id"])
            self.assertEqual(block.revision_id, expected[0]["canonical_revision_id"])
            self.assertEqual(
                document.document_version_id,
                tree.meta_info.document_version_id,
            )
            self.assertEqual(document.content_sha256, tree.meta_info.source_sha256)
            self.assertEqual(str(artifact.resolve()), tree.meta_info.canonical_artifact_path)
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                tree.meta_info.canonical_artifact_sha256,
            )
            self.assertEqual(
                artifact.stat().st_size,
                tree.meta_info.canonical_artifact_size_bytes,
            )
            self.assertEqual(
                str(audit.resolve()), tree.meta_info.routed_audit_artifact_path
            )
            self.assertEqual(
                hashlib.sha256(audit.read_bytes()).hexdigest(),
                tree.meta_info.routed_audit_artifact_sha256,
            )
            self.assertEqual(
                audit.stat().st_size,
                tree.meta_info.routed_audit_artifact_size_bytes,
            )
            self.assertEqual(
                tree.meta_info.routed_audit_artifact_sha256,
                tree.root_node.meta_info.routed_audit_artifact_sha256,
            )
            self.assertEqual(
                tree.meta_info.document_version_id,
                tree.root_node.meta_info.document_version_id,
            )
            self.assertEqual(
                ["parse", "tree", "summary"],
                [value for action, value in stage_calls if action == "enter"],
            )
            self.assertEqual(
                {
                    ("artifact:parse", "canonical-ingestion"),
                    ("artifact:parse", "routed-parsing-audit"),
                    ("artifact:summary", "bookrag-tree"),
                    ("artifact:summary", "bookrag-tree-summary"),
                },
                {
                    item
                    for item in stage_calls
                    if item[0].startswith("artifact:")
                },
            )

    def test_default_path_still_loads_and_binds_mineru_merged_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            document = self.document(cfg)
            canonical = self.block(document)
            merged_path = (
                Path(cfg.save_path) / "auto" / "source_merged_content.json"
            )
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            legacy = [{"type": "text", "text": "MinerU content", "page_idx": 0}]
            merged_path.write_text(json.dumps(legacy), encoding="utf-8")
            request = ParseRequest(
                document=document,
                pages=(PageRef(document.document_version_id, 0),),
            )
            captured: dict[str, object] = {}
            patches = self.pipeline_patches(captured)

            def bind(
                _cfg,
                pdf_list,
                *,
                merged_content_path,
                document_page_count=None,
                debug_artifact_suffix="",
                canonical_artifact_filename="canonical_blocks.json",
            ):
                self.assertEqual(str(merged_path), merged_content_path)
                pdf_list[0]["canonical_block_id"] = canonical.block_id
                pdf_list[0]["canonical_revision_id"] = canonical.revision_id
                canonical_path = Path(cfg.save_path) / canonical_artifact_filename
                canonical_path.write_text(
                    json.dumps(
                        {
                            "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
                            "document": document.to_dict(),
                            "parse_request_id": request.request_id,
                            "parse_result_sha256": "0" * 64,
                            "parser_manifest": {},
                            "block_count": 1,
                            "canonical_blocks": [canonical.to_dict()],
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return request, (canonical,), False

            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "parse_doc",
                        side_effect=AssertionError(
                            "cached MinerU content must be reused"
                        ),
                    )
                )
                binder = stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "bind_canonical_evidence",
                        side_effect=bind,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "project_to_bookrag_pdf_list",
                        side_effect=AssertionError(
                            "default MinerU rows are already normalized"
                        ),
                    )
                )
                tree = build_tree_from_pdf(cfg)

            binder.assert_called_once()
            self.assertEqual(canonical.block_id, captured["refiner_input"][0]["canonical_block_id"])
            self.assertEqual(
                document.document_version_id,
                tree.meta_info.document_version_id,
            )
            self.assertEqual(
                str((Path(cfg.save_path) / "canonical_blocks.json").resolve()),
                tree.meta_info.canonical_artifact_path,
            )

    def test_cross_shard_relation_gap_falls_back_to_one_whole_document_bind(self):
        # Two shards (shard_pages=1, two pages) with the FIRST shard's
        # bind_canonical_evidence call reporting a cross-page relation gap.
        # build_tree_from_pdf must fall back to one additional whole-document
        # call instead of merging the (now-superseded) per-shard fragments.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            cfg.mineru.shard_pages = 1
            document = self.document(cfg)
            canonical = self.block(document)
            merged_path = (
                Path(cfg.save_path) / "auto" / "source_merged_content.json"
            )
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            legacy = [
                {"type": "text", "text": "page 0 content", "page_idx": 0},
                {"type": "text", "text": "page 1 content", "page_idx": 1},
            ]
            merged_path.write_text(json.dumps(legacy), encoding="utf-8")
            request = ParseRequest(
                document=document,
                pages=(PageRef(document.document_version_id, 0),),
            )
            captured: dict[str, object] = {}
            patches = self.pipeline_patches(captured)
            bind_calls: list[list[dict]] = []

            def bind(
                _cfg,
                pdf_list,
                *,
                merged_content_path,
                document_page_count=None,
                debug_artifact_suffix="",
                canonical_artifact_filename="canonical_blocks.json",
            ):
                bind_calls.append([dict(block) for block in pdf_list])
                for block in pdf_list:
                    block["canonical_block_id"] = canonical.block_id
                    block["canonical_revision_id"] = canonical.revision_id
                canonical_path = Path(cfg.save_path) / canonical_artifact_filename
                canonical_path.write_text(
                    json.dumps(
                        {
                            "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
                            "document": document.to_dict(),
                            "parse_request_id": request.request_id,
                            "parse_result_sha256": "0" * 64,
                            "parser_manifest": {},
                            "block_count": len(pdf_list),
                            "canonical_blocks": [canonical.to_dict()],
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                # Only the very first call (shard 0) reports the gap.
                return request, (canonical,), len(bind_calls) == 1

            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "parse_doc",
                        side_effect=AssertionError(
                            "cached MinerU content must be reused"
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder, "bind_canonical_evidence", side_effect=bind,
                    )
                )
                merger = stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "merge_canonical_evidence_shards",
                        side_effect=AssertionError(
                            "merge must be skipped when falling back"
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doc_tree_builder,
                        "project_to_bookrag_pdf_list",
                        side_effect=AssertionError(
                            "default MinerU rows are already normalized"
                        ),
                    )
                )
                build_tree_from_pdf(cfg)

            # Two shard calls (one block each) plus one fallback call with
            # both blocks together.
            self.assertEqual(3, len(bind_calls))
            self.assertEqual(1, len(bind_calls[0]))
            self.assertEqual(1, len(bind_calls[1]))
            self.assertEqual(2, len(bind_calls[2]))
            self.assertEqual({0, 1}, {b["page_idx"] for b in bind_calls[2]})
            merger.assert_not_called()

    def test_provider_source_mismatch_stops_before_bookrag_or_mineru(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            other_document = self.document(cfg, source_sha256="f" * 64)
            block = self.block(other_document)
            artifact = self.artifact(root, other_document, block)
            provider = RecordingCanonicalProvider(
                SimpleNamespace(
                    document=other_document,
                    canonical_blocks=(block,),
                    canonical_artifact_path=artifact,
                )
            )

            with (
                mock.patch.object(
                    doc_tree_builder,
                    "parse_doc",
                    side_effect=AssertionError("MinerU must be bypassed"),
                ),
                mock.patch.object(
                    doc_tree_builder,
                    "LLM",
                    side_effect=AssertionError("downstream stages must not start"),
                ),
            ):
                with self.assertRaisesRegex(
                    CanonicalIngestionBindingError,
                    "source SHA-256 differs",
                ):
                    build_tree_from_pdf(
                        cfg,
                        reforce=True,
                        parsing_provider=provider,
                    )

            self.assertEqual([(cfg, True)], provider.calls)

    def test_nonaccepted_canonical_quality_requires_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            document = self.document(cfg)

            for status in (QualityStatus.DEGRADED, QualityStatus.NEEDS_REVIEW):
                with self.subTest(status=status.value):
                    block = self.block(document, quality_status=status)
                    artifact = self.artifact(root, document, block)
                    provider = RecordingCanonicalProvider(
                        SimpleNamespace(
                            document=document,
                            canonical_blocks=(block,),
                            canonical_artifact_path=artifact,
                        )
                    )
                    with self.assertRaisesRegex(
                        CanonicalIngestionBindingError,
                        "quality statuses blocked by policy",
                    ):
                        build_tree_from_pdf(
                            cfg,
                            reforce=True,
                            parsing_provider=provider,
                        )

            degraded = self.block(
                document,
                quality_status=QualityStatus.DEGRADED,
            )
            artifact = self.artifact(root, document, degraded)
            provider = RecordingCanonicalProvider(
                SimpleNamespace(
                    document=document,
                    canonical_blocks=(degraded,),
                    canonical_artifact_path=artifact,
                )
            )
            captured: dict[str, object] = {}
            with ExitStack() as stack:
                for patcher in self.pipeline_patches(captured):
                    stack.enter_context(patcher)
                build_tree_from_pdf(
                    cfg,
                    reforce=True,
                    parsing_provider=provider,
                    ingestion_policy=CanonicalIngestionPolicy(
                        allowed_quality_statuses=(
                            QualityStatus.ACCEPTED,
                            QualityStatus.DEGRADED,
                        )
                    ),
                )

            self.assertEqual(
                QualityStatus.DEGRADED.value,
                captured["refiner_input"][0]["quality_status"],
            )

    def test_duplicate_current_and_revision_identities_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_cfg(root)
            document = self.document(cfg)
            block = self.block(document)
            artifact = self.artifact(root, document, block)
            next_revision = replace(block, revision=block.revision + 1)

            with self.assertRaisesRegex(
                CanonicalIngestionBindingError,
                "multiple current revisions",
            ):
                CanonicalIngestionResult(
                    document=document,
                    canonical_blocks=(block, next_revision),
                    canonical_artifact_path=artifact,
                )

            with self.assertRaisesRegex(
                CanonicalIngestionBindingError,
                "multiple current revisions|duplicate revision identities",
            ):
                CanonicalIngestionResult(
                    document=document,
                    canonical_blocks=(block, block),
                    canonical_artifact_path=artifact,
                )

            other_document = self.document(cfg, source_sha256="e" * 64)
            other_block = self.block(other_document)
            with self.assertRaisesRegex(
                CanonicalIngestionBindingError,
                "blocks belong to another document version",
            ):
                CanonicalIngestionResult(
                    document=document,
                    canonical_blocks=(other_block,),
                    canonical_artifact_path=artifact,
                )


if __name__ == "__main__":
    unittest.main()
