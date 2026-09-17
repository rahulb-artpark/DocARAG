from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from document_services.evidence import provenance
from document_services.indexing import recovery
from document_services.operations import environment_probe as runtime_probe
from document_services.runtime.bookrag_bridge import inspect_run_readiness


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_qdrant_attestation(
    path: Path,
    *,
    revision_id: str,
    document_version_id: str,
    source_sha256: str,
    index_kind: str,
    collection_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "docarag.test-qdrant-index.v1",
        "document_version_id": document_version_id,
        "index_revision_id": revision_id,
        "scope_filter": {
            "tenant_id": "default",
            "corpus_id": "default",
            "document_version_id": document_version_id,
            "index_revision_id": revision_id,
            "source_sha256": source_sha256,
            "index_kind": index_kind,
        },
        "point_count": 1,
        "collection_name": collection_name,
        "qdrant_endpoint_sha256": hashlib.sha256(
            b"http://qdrant.test:6333"
        ).hexdigest(),
    }
    write_json(
        path,
        {
            **identity,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "state": "content_attested",
        },
    )


def terminal_state(
    *, revision_id: str, attempt_id: str, manifest: bytes
) -> dict[str, object]:
    artifact = json.loads(manifest)
    return {
        "schema_version": 1,
        "revision_id": revision_id,
        "attempt_id": attempt_id,
        "overall_status": "succeeded",
        "artifact_manifest": "artifact-manifest.json",
        "artifact_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "artifact_set_sha256": artifact["artifact_set_sha256"],
        "stages": {
            stage: {"status": "succeeded"}
            for stage in ("tree", "graph", "vdb")
        },
    }


def bind_terminal_state(run: Path, revision_id: str, attempt_id: str) -> None:
    manifest = (run / "artifact-manifest.json").read_bytes()
    state = terminal_state(
        revision_id=revision_id, attempt_id=attempt_id, manifest=manifest
    )
    write_json(run / "index-state.json", state)
    write_json(run / "attempts" / attempt_id / "state.json", state)


def make_published_run(
    base: Path,
    *,
    name: str = "run",
    bm25: bool = False,
    summary_checkpoint: bool = False,
    canonical_evidence: bool = False,
) -> tuple[Path, str, str]:
    revision_id = f"revision-{name}"
    document_version_id = f"document-version-{name}"
    attempt_id = f"attempt-{name}"
    run = base / "runs" / "document" / revision_id
    run.mkdir(parents=True)

    source_bytes = b"%PDF-1.4\nimmutable source\n"
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = base / "runs" / "_sources" / "sha256" / f"{source_sha}.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_bytes)
    write_json(
        run / "run-manifest.json",
        {
            "schema_version": 1,
            "revision_id": revision_id,
            "document_id": f"document-{name}",
            "document_version_id": document_version_id,
            "tenant_id": "default",
            "corpus_id": "default",
            "source": {
                "sha256": source_sha,
                "size_bytes": len(source_bytes),
                "snapshot_path": str(source),
            },
        },
    )

    initial_state = {
        "schema_version": 1,
        "revision_id": revision_id,
        "attempt_id": attempt_id,
        "overall_status": "running",
        "stages": {
            stage: {"status": "succeeded"}
            for stage in ("tree", "graph", "vdb")
        },
    }
    write_json(run / "index-state.json", initial_state)

    runtime_identity = {
        "python": {"implementation": "CPython", "version": "test"},
        "platform": {"system": "test"},
        "packages": {},
        "gpus": [],
        "environment": {"CUDA_VISIBLE_DEVICES": None},
    }
    runtime = {
        "schema_version": 1,
        "revision_id": revision_id,
        "attempt_id": attempt_id,
        **runtime_identity,
    }
    runtime["environment_fingerprint_sha256"] = (
        provenance.environment_fingerprint_sha256(runtime)
    )
    write_json(run / "attempts" / attempt_id / "runtime.json", runtime)

    (run / "tree.pkl").write_bytes(b"trusted-tree-pickle")
    write_json(run / "tree.json", {"nodes": [{"index_id": 0}]})
    write_json(
        run / "graph_data.json",
        {
            "graph": {"nodes": [{"id": "entity"}], "links": []},
            "tree2kg": {"0": ["entity"]},
        },
    )
    (run / "run_config.yaml").write_text(
        f"index_type: {'bm25' if bm25 else 'gbc'}\n"
        "source:\n"
        f'  sha256: "{source_sha}"\n'
        f'  document_version_id: "{document_version_id}"\n'
        f'  index_revision_id: "{revision_id}"\n'
        "vdb:\n"
        '  tenant_id: "default"\n'
        '  corpus_id: "default"\n'
        '  qdrant_url: "http://qdrant.test:6333"\n'
        '  api_key_env: "QDRANT_API_KEY"\n'
        "  timeout_seconds: 1.0\n",
        encoding="utf-8",
    )
    if summary_checkpoint:
        (run / "summary-checkpoints.sqlite3").write_bytes(b"durable-summary-cache")
    if canonical_evidence:
        write_json(
            run / "canonical_blocks.json",
            {
                "schema_version": "docarag.parser-contract.v1",
                "document_version_id": document_version_id,
                "blocks": [],
            },
        )

    retrieval_directory = run / "Tree_vdb"
    retrieval_attestation = None
    lexical_store = None
    if bm25:
        retrieval_directory.mkdir()
        lexical_store = retrieval_directory / "bm25_index.pkl"
        lexical_store.write_bytes(b"trusted-bm25-pickle")
    else:
        retrieval_attestation = (
            retrieval_directory / "docarag-qdrant-index.ready.json"
        )
        write_qdrant_attestation(
            retrieval_attestation,
            revision_id=revision_id,
            document_version_id=document_version_id,
            source_sha256=source_sha,
            index_kind="tree",
            collection_name="bookrag-tree",
        )
    entity_attestation = (
        run
        / "entity_attestation_test"
        / "docarag-qdrant-index.ready.json"
    )
    write_qdrant_attestation(
        entity_attestation,
        revision_id=revision_id,
        document_version_id=document_version_id,
        source_sha256=source_sha,
        index_kind="entity",
        collection_name="bookrag-entities",
    )

    provenance.publish_artifact_manifest(
        run,
        revision_id=revision_id,
        attempt_id=attempt_id,
        retrieval_attestation=retrieval_attestation,
        lexical_store=lexical_store,
        graph_artifact=run / "graph_data.json",
        entity_attestation=entity_attestation,
    )
    bind_terminal_state(run, revision_id, attempt_id)
    return run, revision_id, attempt_id


def rewrite_manifest_and_rebind(
    run: Path,
    revision_id: str,
    attempt_id: str,
    payload: dict[str, object],
) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    for path in (
        run / "artifact-manifest.json",
        run / "attempts" / attempt_id / "artifact-manifest.json",
    ):
        path.chmod(0o600)
        path.write_bytes(encoded)
    bind_terminal_state(run, revision_id, attempt_id)


class ProvenanceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        provenance._sha256_for_stat.cache_clear()
        provenance._json_for_stat.cache_clear()
        qdrant = patch(
            "retrieval_engine.utils.artifact_paths.qdrant_collection_state",
            return_value=SimpleNamespace(exists=True, point_count=1),
        )
        qdrant.start()
        self.addCleanup(qdrant.stop)

    def test_valid_publication_passes_online_and_deep_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = make_published_run(Path(directory))

            self.assertTrue(provenance.verify_query_artifacts(run).valid)
            self.assertTrue(provenance.verify_artifact_manifest(run, deep=True).valid)
            self.assertTrue(inspect_run_readiness(run, purpose="ask").ready)

    def test_summary_checkpoint_is_bound_into_published_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = make_published_run(
                Path(directory), summary_checkpoint=True
            )

            manifest = json.loads(
                (run / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            published = {entry["path"] for entry in manifest["files"]}
            self.assertIn("summary-checkpoints.sqlite3", published)
            self.assertTrue(
                provenance.verify_artifact_manifest(run, deep=True).valid
            )

    def test_canonical_evidence_is_a_hashed_query_artifact_role(self):
        with tempfile.TemporaryDirectory() as directory:
            run, revision_id, attempt_id = make_published_run(
                Path(directory), canonical_evidence=True
            )

            manifest = json.loads(
                (run / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["roles"]["canonical_evidence"],
                "canonical_blocks.json",
            )
            self.assertTrue(provenance.verify_query_artifacts(run).valid)

            canonical = run / "canonical_blocks.json"
            original = canonical.read_bytes()
            document_version_id = json.loads(
                (run / "run-manifest.json").read_text(encoding="utf-8")
            )["document_version_id"]
            replacement = original.replace(
                document_version_id.encode(), b"f" * len(document_version_id)
            )
            self.assertEqual(len(original), len(replacement))
            canonical.write_bytes(replacement)

            verification = provenance.verify_query_artifacts(run)
            self.assertFalse(verification.valid)
            self.assertIn(
                "artifact digest differs for canonical_blocks.json",
                verification.errors,
            )

    def test_runtime_probe_fingerprints_all_distributions_and_gpu_visibility(self):
        class Distribution:
            def __init__(self, name: str, version: str):
                self.metadata = {"Name": name}
                self.version = version

        installed = [
            Distribution("Torch", "2.8.0"),
            Distribution("private_plugin.extra", "1.2.3"),
        ]
        with (
            patch(
                "document_services.operations.environment_probe.metadata.distributions",
                return_value=installed,
            ),
            patch("document_services.operations.environment_probe.gpu_identity", return_value=[]),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}),
        ):
            payload = runtime_probe.build_payload(
                revision_id="revision", attempt_id="attempt"
            )

        self.assertEqual(
            {"private-plugin-extra": "1.2.3", "torch": "2.8.0"},
            payload["packages"],
        )
        self.assertEqual(
            "2,3", payload["environment"]["CUDA_VISIBLE_DEVICES"]
        )
        self.assertEqual(
            provenance.environment_fingerprint_sha256(payload),
            payload["environment_fingerprint_sha256"],
        )

    def test_publication_requires_run_manifest_and_matching_runtime_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, revision_id, attempt_id = make_published_run(base)
            retrieval = run / "Tree_vdb" / "docarag-qdrant-index.ready.json"
            graph = run / "graph_data.json"
            entity = (
                run
                / "entity_attestation_test"
                / "docarag-qdrant-index.ready.json"
            )

            (run / "run-manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "run manifest"):
                provenance.publish_artifact_manifest(
                    run,
                    revision_id=revision_id,
                    attempt_id=attempt_id,
                    retrieval_attestation=retrieval,
                    graph_artifact=graph,
                    entity_attestation=entity,
                )

            with self.assertRaisesRegex(recovery.RecoveryError, "source-of-truth"):
                recovery.plan_recovery(run)

    def test_recovery_treats_broken_manifest_symlink_as_unsafe(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "artifact-manifest.json").symlink_to("missing-manifest.json")

            plan = recovery.plan_recovery(run)

            self.assertEqual("tree", plan.from_stage)
            self.assertIn("unsafe", plan.reason)

    def test_publication_rejects_forged_runtime_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            run, revision_id, attempt_id = make_published_run(Path(directory))
            runtime_path = run / "attempts" / attempt_id / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["packages"] = {"torch": "forged"}
            write_json(runtime_path, runtime)
            state_path = run / "index-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["overall_status"] = "running"
            state.pop("artifact_manifest", None)
            state.pop("artifact_manifest_sha256", None)
            state.pop("artifact_set_sha256", None)
            write_json(state_path, state)

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                provenance.publish_artifact_manifest(
                    run,
                    revision_id=revision_id,
                    attempt_id=attempt_id,
                    retrieval_attestation=(
                        run / "Tree_vdb" / "docarag-qdrant-index.ready.json"
                    ),
                    graph_artifact=run / "graph_data.json",
                    entity_attestation=(
                        run
                        / "entity_attestation_test"
                        / "docarag-qdrant-index.ready.json"
                    ),
                )

    def test_verifier_recomputes_runtime_fingerprint_after_coherent_rebinding(self):
        with tempfile.TemporaryDirectory() as directory:
            run, revision_id, attempt_id = make_published_run(Path(directory))
            runtime_path = run / "attempts" / attempt_id / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["packages"] = {"torch": "forged"}
            write_json(runtime_path, runtime)

            artifact_path = run / "artifact-manifest.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["runtime_manifest"]["sha256"] = provenance.sha256_file(
                runtime_path
            )
            rewrite_manifest_and_rebind(run, revision_id, attempt_id, artifact)

            result = provenance.verify_query_artifacts(run)

            self.assertFalse(result.valid)
            self.assertIn(
                "runtime environment fingerprint is inconsistent", result.errors
            )

    def test_verifier_binds_terminal_revision_attempt_and_manifest_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, attempt_id = make_published_run(Path(directory))
            state_path = run / "attempts" / attempt_id / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["attempt_id"] = "different-attempt"
            write_json(state_path, state)
            attempt_manifest = (
                run / "attempts" / attempt_id / "artifact-manifest.json"
            )
            attempt_manifest.chmod(0o600)
            attempt_manifest.write_text("{}\n", encoding="utf-8")

            result = provenance.verify_query_artifacts(run)

            self.assertFalse(result.valid)
            self.assertIn("terminal attempt state attempt ID differs", result.errors)
            self.assertIn(
                "root and attempt-local artifact manifests differ", result.errors
            )

    def test_published_run_cannot_fall_back_to_legacy_when_state_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = make_published_run(Path(directory))
            (run / "index-state.json").unlink()

            readiness = inspect_run_readiness(run, purpose="ask")

            self.assertFalse(readiness.ready)
            self.assertIn("terminal index state", readiness.missing[0])

    def test_verifier_requires_run_manifest_and_root_state_digest_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            missing_run_manifest, _, _ = make_published_run(
                base, name="missing-run-manifest"
            )
            (missing_run_manifest / "run-manifest.json").unlink()
            missing_result = provenance.verify_query_artifacts(
                missing_run_manifest
            )

            wrong_state, _, _ = make_published_run(base, name="wrong-state")
            state_path = wrong_state / "index-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["artifact_manifest_sha256"] = "0" * 64
            write_json(state_path, state)
            state_result = provenance.verify_query_artifacts(wrong_state)

            self.assertFalse(missing_result.valid)
            self.assertIn("run-manifest.json is missing or unsafe", missing_result.errors)
            self.assertFalse(state_result.valid)
            self.assertIn(
                "terminal index state artifact manifest digest differs",
                state_result.errors,
            )

    def test_verifier_rejects_source_snapshot_outside_this_runs_source_store(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, revision_id, attempt_id = make_published_run(base)
            run_manifest_path = run / "run-manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            original = Path(run_manifest["source"]["snapshot_path"])
            external = (
                base
                / "external"
                / "_sources"
                / "sha256"
                / original.name
            )
            external.parent.mkdir(parents=True)
            external.write_bytes(original.read_bytes())
            run_manifest["source"]["snapshot_path"] = str(external)
            write_json(run_manifest_path, run_manifest)

            artifact_path = run / "artifact-manifest.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["run_manifest"]["sha256"] = provenance.sha256_file(
                run_manifest_path
            )
            rewrite_manifest_and_rebind(run, revision_id, attempt_id, artifact)

            result = provenance.verify_query_artifacts(run)

            self.assertFalse(result.valid)
            self.assertIn("immutable source snapshot path is unsafe", result.errors)

    def test_online_readiness_rejects_same_size_query_artifact_tampering(self):
        artifacts = (
            "tree.pkl",
            "tree.json",
            "graph_data.json",
            "run_config.yaml",
            "Tree_vdb/bm25_index.pkl",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, relative in enumerate(artifacts):
                with self.subTest(artifact=relative):
                    run, _, _ = make_published_run(
                        base, name=f"tamper-{index}", bm25=True
                    )
                    self.assertTrue(provenance.verify_query_artifacts(run).valid)
                    target = run / relative
                    original = target.read_bytes()
                    old_stat = target.stat()
                    target.write_bytes(
                        bytes([original[0] ^ 1]) + original[1:]
                    )
                    os.utime(
                        target,
                        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns),
                    )

                    result = provenance.verify_query_artifacts(run)

                    self.assertFalse(result.valid)
                    self.assertTrue(
                        any("digest differs" in error for error in result.errors),
                        result.errors,
                    )
                    self.assertFalse(
                        inspect_run_readiness(run, purpose="ask").ready
                    )

    def test_online_verification_caches_only_bound_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = make_published_run(Path(directory), bm25=True)
            provenance._sha256_for_stat.cache_clear()
            original = provenance.sha256_file
            hashed: list[Path] = []

            def recording_hash(path: Path) -> str:
                hashed.append(Path(path))
                return original(Path(path))

            with patch("document_services.evidence.provenance.sha256_file", side_effect=recording_hash):
                first = provenance.verify_query_artifacts(run)
                first_count = len(hashed)
                second = provenance.verify_query_artifacts(run)

            self.assertTrue(first.valid, first.errors)
            self.assertTrue(second.valid, second.errors)
            self.assertEqual(first_count, len(hashed))
            self.assertIn(run / "Tree_vdb" / "bm25_index.pkl", hashed)
            self.assertIn(
                run
                / "entity_attestation_test"
                / "docarag-qdrant-index.ready.json",
                hashed,
            )

    def test_readiness_cache_key_tracks_same_size_loaded_artifact_change(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = make_published_run(Path(directory))
            before = provenance.query_readiness_stat_key(run)
            tree = run / "tree.pkl"
            original = tree.read_bytes()
            old_stat = tree.stat()
            tree.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            os.utime(tree, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

            after = provenance.query_readiness_stat_key(run)

            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
