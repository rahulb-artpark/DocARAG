from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docarag.artifact_roles import resolve_artifact_roles
from docarag.domain.evidence import DocumentRef


REPO = Path(__file__).resolve().parents[1]


class ArtifactRoleTests(unittest.TestCase):
    def test_resolver_uses_bookrag_graph_and_store_naming_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run with spaces"
            run.mkdir()

            source_config = REPO / "config/ollama-gbc.yaml"
            bound_config = Path(directory) / "bound-config.yaml"
            bound_config.write_text(
                source_config.read_text(encoding="utf-8")
                + "\nsource:\n"
                + '  display_name: "source.pdf"\n'
                + '  original_uri: "file:///source.pdf"\n'
                + f'  sha256: "{"a" * 64}"\n',
                encoding="utf-8",
            )

            roles = resolve_artifact_roles(bound_config, run)

            self.assertEqual(str(run / "graph_data.json"), roles["graph"])
            self.assertEqual("gbc", roles["index_type"])
            self.assertEqual("TreeVDB", roles["retrieval_collection"])
            self.assertEqual(
                str(
                    run
                    / "qdrant-tree-attestation"
                    / "docarag-qdrant-index.ready.json"
                ),
                roles["retrieval_attestation"],
            )
            # The name carries the vector schema: Qdrant cannot convert an
            # unnamed collection to named vectors, so a migration is a new
            # collection rather than a mutated one.
            self.assertEqual(
                "bookrag-entities-hybrid-v1", roles["entity_collection"]
            )
            self.assertTrue(
                Path(roles["entity_attestation"]).parent.name.startswith(
                    "entity_attestation_0.6_"
                )
            )
            self.assertEqual("a" * 64, roles["retrieval_scope"]["source_sha256"])
            self.assertEqual(
                run.name, roles["retrieval_scope"]["index_revision_id"]
            )
            self.assertNotEqual(
                run.name, roles["retrieval_scope"]["document_version_id"]
            )
            self.assertEqual("tree", roles["retrieval_scope"]["index_kind"])
            self.assertEqual("entity", roles["entity_scope"]["index_kind"])
            for role in ("graph", "entity_attestation", "retrieval_attestation"):
                Path(roles[role]).relative_to(run)

    def test_resolver_binds_source_identity_from_run_manifest(self):
        # The live index driver copies the user's config byte-for-byte (it is
        # content-addressed and legitimately carries no per-document `source`)
        # and records the admitted identity in run-manifest.json instead. The
        # resolver must recover that identity rather than demand it in the
        # config snapshot.
        with tempfile.TemporaryDirectory() as directory:
            revision_id = "b" * 64
            run = Path(directory) / revision_id
            run.mkdir()

            bare_config = Path(directory) / "system-config-snapshot.yaml"
            bare_config.write_text(
                (REPO / "config/ollama-gbc.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.assertNotIn("\nsource:", bare_config.read_text(encoding="utf-8"))

            source = REPO / "config/ollama-gbc.yaml"  # any stable file for a URI/name
            source_uri = source.resolve().as_uri()
            source_sha256 = "c" * 64
            document = DocumentRef.identified(
                content_sha256=source_sha256,
                mime_type="application/pdf",
                source_uri=source_uri,
                display_name=source.name,
            )
            (run / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "revision_id": revision_id,
                        "document_id": document.document_id,
                        "document_version_id": document.document_version_id,
                        "tenant_id": "acme",
                        "corpus_id": "handbooks",
                        "source": {
                            "path": str(source),
                            "uri": source_uri,
                            "sha256": source_sha256,
                            "snapshot_path": str(run / "snap.pdf"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            roles = resolve_artifact_roles(bare_config, run)

            self.assertEqual(source_sha256, roles["retrieval_scope"]["source_sha256"])
            self.assertEqual("acme", roles["retrieval_scope"]["tenant_id"])
            self.assertEqual("handbooks", roles["retrieval_scope"]["corpus_id"])
            self.assertEqual(
                document.document_version_id,
                roles["retrieval_scope"]["document_version_id"],
            )
            self.assertEqual(
                revision_id, roles["retrieval_scope"]["index_revision_id"]
            )

    def test_resolver_still_requires_identity_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            bare_config = Path(directory) / "system-config-snapshot.yaml"
            bare_config.write_text(
                (REPO / "config/ollama-gbc.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                resolve_artifact_roles(bare_config, run)


if __name__ == "__main__":
    unittest.main()
