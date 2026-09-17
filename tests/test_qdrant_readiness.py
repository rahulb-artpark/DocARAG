from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from retrieval_engine.utils.artifact_paths import qdrant_endpoint_sha256
from document_services.persistence.qdrant_readiness import (
    QdrantAttestationMismatch,
    validate_qdrant_attestation,
)


SOURCE_SHA256 = "a" * 64
REVISION_ID = "revision-a"
DOCUMENT_VERSION_ID = "document-version-a"


def _attestation(endpoint: str) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": "docarag.test-qdrant-index.v1",
        "source_sha256": SOURCE_SHA256,
        "document_version_id": DOCUMENT_VERSION_ID,
        "index_revision_id": REVISION_ID,
        "scope_filter": {
            "tenant_id": "tenant-a",
            "corpus_id": "corpus-a",
            "document_version_id": DOCUMENT_VERSION_ID,
            "index_revision_id": REVISION_ID,
            "source_sha256": SOURCE_SHA256,
            "index_kind": "tree",
        },
        "point_count": 3,
        "collection_name": "docarag-tree",
        "qdrant_endpoint_sha256": qdrant_endpoint_sha256(endpoint),
    }
    return {
        **identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "state": "content_attested",
    }


class QdrantReadinessTests(unittest.TestCase):
    def _write_fixture(
        self,
        directory: str,
        *,
        configured_endpoint: str,
        attested_endpoint: str,
    ) -> tuple[Path, Path]:
        root = Path(directory)
        config = root / "run_config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source": {
                        "sha256": SOURCE_SHA256,
                        "document_version_id": DOCUMENT_VERSION_ID,
                    },
                    "vdb": {
                        "tenant_id": "tenant-a",
                        "corpus_id": "corpus-a",
                        "qdrant_url": configured_endpoint,
                        "qdrant_url_env": "DOCARAG_TEST_QDRANT_URL",
                        "api_key_env": "DOCARAG_TEST_QDRANT_API_KEY",
                        "timeout_seconds": 2.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        attestation = root / "docarag-qdrant-index.ready.json"
        attestation.write_text(
            json.dumps(_attestation(attested_endpoint)), encoding="utf-8"
        )
        return config, attestation

    def test_environment_endpoint_and_live_count_must_both_match(self):
        with tempfile.TemporaryDirectory() as directory:
            config, attestation = self._write_fixture(
                directory,
                configured_endpoint="http://configured:6333",
                attested_endpoint="http://runtime:6333",
            )
            with (
                patch.dict(
                    os.environ,
                    {"DOCARAG_TEST_QDRANT_URL": "http://runtime:6333"},
                    clear=False,
                ),
                patch(
                    "document_services.persistence.qdrant_readiness.qdrant_collection_state",
                    return_value=SimpleNamespace(exists=True, point_count=3),
                ) as state,
            ):
                self.assertEqual(
                    3,
                    validate_qdrant_attestation(
                        config_path=config,
                        attestation_path=attestation,
                        revision_id=REVISION_ID,
                        expected_index_kind="tree",
                    ),
                )
            self.assertEqual("http://runtime:6333", state.call_args.kwargs["qdrant_url"])

    def test_modified_content_with_stale_self_digest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config, attestation = self._write_fixture(
                directory,
                configured_endpoint="http://qdrant:6333",
                attested_endpoint="http://qdrant:6333",
            )
            payload = json.loads(attestation.read_text(encoding="utf-8"))
            payload["point_count"] = 4
            attestation.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch(
                    "document_services.persistence.qdrant_readiness.qdrant_collection_state"
                ) as state,
                self.assertRaisesRegex(
                    QdrantAttestationMismatch, "self-digest mismatches"
                ),
            ):
                validate_qdrant_attestation(
                    config_path=config,
                    attestation_path=attestation,
                    revision_id=REVISION_ID,
                )
            state.assert_not_called()

    def test_environment_endpoint_mismatch_fails_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config, attestation = self._write_fixture(
                directory,
                configured_endpoint="http://configured:6333",
                attested_endpoint="http://configured:6333",
            )
            with (
                patch.dict(
                    os.environ,
                    {"DOCARAG_TEST_QDRANT_URL": "http://other:6333"},
                    clear=False,
                ),
                patch(
                    "document_services.persistence.qdrant_readiness.qdrant_collection_state"
                ) as state,
                self.assertRaisesRegex(
                    QdrantAttestationMismatch, "endpoint binding mismatches"
                ),
            ):
                validate_qdrant_attestation(
                    config_path=config,
                    attestation_path=attestation,
                    revision_id=REVISION_ID,
                )
            state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
