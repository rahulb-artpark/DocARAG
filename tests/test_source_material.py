from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from document_services.evidence.assets import AssetMaterialErrorCode
from document_services.domain import DocumentRef
from document_services.evidence.sources import (
    SOURCE_MATERIAL_BINDING_KEY,
    SourceMaterialError,
    SourceMaterialResolver,
    bind_source_material,
    local_source_material_binding,
)


class SourceMaterialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.payload = b"%PDF-1.7\nimmutable parser source fixture\n"
        self.path = self.root / "source.pdf"
        self.path.write_bytes(self.payload)
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.document = DocumentRef.identified(
            content_sha256=self.digest,
            mime_type="application/pdf",
            source_uri="s3://records/original/report.pdf",
            display_name="report.pdf",
            page_count=1,
        )

    def bound_document(self) -> DocumentRef:
        return bind_source_material(
            self.document,
            local_source_material_binding(
                self.path,
                sha256=self.digest,
                media_type="application/pdf",
            ),
        )

    def test_binding_preserves_logical_source_and_round_trips(self) -> None:
        bound = self.bound_document()

        self.assertEqual(self.document.source_uri, bound.source_uri)
        self.assertEqual(self.document.document_version_id, bound.document_version_id)
        self.assertIn(SOURCE_MATERIAL_BINDING_KEY, bound.metadata)
        restored = DocumentRef.from_dict(bound.to_dict())
        self.assertEqual(bound, restored)

    def test_resolver_returns_exact_bounded_byte_snapshot(self) -> None:
        material = SourceMaterialResolver(
            allowed_roots=(self.root,), max_source_bytes=1024
        ).resolve(self.bound_document(), include_bytes=True)

        self.assertEqual(self.payload, material.data)
        self.assertEqual(self.digest, material.sha256)
        self.assertEqual(self.path, material.path)
        self.assertEqual(self.document.source_uri, material.logical_source_uri)
        self.assertTrue(material.to_dict()["bytes_retained"])

    def test_missing_binding_never_falls_back_to_original_uri(self) -> None:
        with self.assertRaises(SourceMaterialError) as raised:
            SourceMaterialResolver(allowed_roots=(self.root,)).resolve(self.document)

        self.assertEqual("binding_required", raised.exception.code)

    def test_binding_identity_must_match_document(self) -> None:
        wrong = local_source_material_binding(
            self.path,
            sha256="0" * 64,
            media_type="application/pdf",
        )
        with self.assertRaises(SourceMaterialError) as raised:
            bind_source_material(self.document, wrong)

        self.assertEqual("identity_mismatch", raised.exception.code)

    def test_changed_bytes_fail_digest_verification(self) -> None:
        bound = self.bound_document()
        self.path.write_bytes(self.payload + b"changed")

        with self.assertRaises(SourceMaterialError) as raised:
            SourceMaterialResolver(allowed_roots=(self.root,)).resolve(bound)

        self.assertEqual("materialization_failed", raised.exception.code)
        self.assertEqual(
            AssetMaterialErrorCode.DIGEST_MISMATCH,
            raised.exception.material_code,
        )

    def test_out_of_root_and_size_limits_fail_closed(self) -> None:
        other = self.root / "other"
        other.mkdir()
        resolver = SourceMaterialResolver(
            allowed_roots=(other,), max_source_bytes=len(self.payload) - 1
        )
        with self.assertRaises(SourceMaterialError) as outside:
            resolver.resolve(self.bound_document())
        self.assertEqual(AssetMaterialErrorCode.PATH_ESCAPE, outside.exception.material_code)

        resolver = SourceMaterialResolver(
            allowed_roots=(self.root,), max_source_bytes=len(self.payload) - 1
        )
        with self.assertRaises(SourceMaterialError) as oversized:
            resolver.resolve(self.bound_document())
        self.assertEqual(AssetMaterialErrorCode.TOO_LARGE, oversized.exception.material_code)

    def test_different_existing_binding_cannot_replace_source(self) -> None:
        bound = self.bound_document()
        second = self.root / "second.pdf"
        second.write_bytes(self.payload)
        replacement = local_source_material_binding(
            second,
            sha256=self.digest,
            media_type="application/pdf",
        )
        with self.assertRaises(SourceMaterialError) as raised:
            bind_source_material(bound, replacement)
        self.assertEqual("binding_conflict", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
