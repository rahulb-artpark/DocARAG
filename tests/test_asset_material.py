from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from docarag.asset_material import (
    ASSET_MATERIAL_SCHEMA_VERSION,
    AssetMaterialErrorCode,
    AssetMaterialResolver,
    AssetMaterializationError,
)
from docarag.domain import AssetKind, AssetRef


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def asset(path_or_uri: str, data: bytes, *, media_type: str = "image/png") -> AssetRef:
    return AssetRef(
        kind=AssetKind.REGION_CROP,
        uri=path_or_uri,
        media_type=media_type,
        sha256=digest(data),
    )


class AssetMaterialResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "assets"
        self.root.mkdir()
        self.content_root = Path(self.temporary_directory.name) / "cas"
        self.content_root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def resolver(self, **overrides: object) -> AssetMaterialResolver:
        configuration: dict[str, object] = {
            "allowed_roots": (self.root,),
            "content_addressed_root": self.content_root,
            "max_assets": 8,
            "max_asset_bytes": 64 * 1024,
            "max_total_bytes": 128 * 1024,
            "chunk_size": 1024,
        }
        configuration.update(overrides)
        return AssetMaterialResolver(**configuration)  # type: ignore[arg-type]

    def test_local_file_is_stream_verified_without_retaining_bytes_by_default(self):
        payload = b"canonical diagram pixels" * 300
        path = self.root / "page-0001" / "diagram.png"
        path.parent.mkdir()
        path.write_bytes(payload)

        material = self.resolver().resolve(asset(os.fspath(path), payload))

        self.assertEqual("verified", material.status)
        self.assertEqual(digest(payload), material.sha256)
        self.assertEqual(len(payload), material.size_bytes)
        self.assertEqual(path, material.path)
        self.assertEqual("local_file", material.reference_kind)
        self.assertIsNone(material.data)
        self.assertEqual(
            {
                "schema_version": ASSET_MATERIAL_SCHEMA_VERSION,
                "status": "verified",
                "asset_id": material.asset_id,
                "asset_uri": os.fspath(path),
                "media_type": "image/png",
                "sha256": digest(payload),
                "size_bytes": len(payload),
                "verified_path": os.fspath(path),
                "reference_kind": "local_file",
                "bytes_retained": False,
                "verification_scope": "descriptor_read_snapshot",
                "byte_limit_scope": "encoded_file_bytes_only",
                "media_type_source": "asset_ref_declaration_unverified",
                "decoded_content_validation": "not_performed",
            },
            material.to_dict(),
        )

    def test_audit_metadata_does_not_claim_mime_or_decoded_image_validation(self):
        payload = b"compressed bytes are not decoded pixels"
        path = self.root / "declared-image.png"
        path.write_bytes(payload)

        metadata = self.resolver().resolve(
            asset(os.fspath(path), payload, media_type="image/png")
        ).to_dict()

        self.assertEqual("encoded_file_bytes_only", metadata["byte_limit_scope"])
        self.assertEqual(
            "asset_ref_declaration_unverified", metadata["media_type_source"]
        )
        self.assertEqual(
            "not_performed", metadata["decoded_content_validation"]
        )

    def test_explicit_byte_snapshot_is_immutable_and_batch_bounded(self):
        first_payload = b"first verified image"
        second_payload = b"second verified image"
        first_path = self.root / "first.png"
        second_path = self.root / "second.png"
        first_path.write_bytes(first_payload)
        second_path.write_bytes(second_payload)
        resolver = self.resolver(max_total_bytes=len(first_payload) + len(second_payload))

        materials = resolver.resolve_many(
            (
                asset(os.fspath(first_path), first_payload),
                asset(os.fspath(second_path), second_payload),
            ),
            include_bytes=True,
        )

        self.assertEqual((first_payload, second_payload), tuple(x.data for x in materials))
        first_path.write_bytes(b"changed after the bounded descriptor read")
        self.assertEqual(first_payload, materials[0].data)
        self.assertTrue(materials[0].to_dict()["bytes_retained"])

    def test_byte_snapshot_fast_path_reads_exact_bound_without_joining_chunks(self):
        payload = b"bounded immutable snapshot" * 127
        path = self.root / "fast-path.png"
        path.write_bytes(payload)
        requested_counts: list[int] = []
        real_read = os.read

        def recording_read(file_descriptor: int, count: int) -> bytes:
            requested_counts.append(count)
            return real_read(file_descriptor, count)

        with patch("docarag.asset_material.os.read", side_effect=recording_read):
            material = self.resolver(max_asset_bytes=len(payload)).resolve(
                asset(os.fspath(path), payload), include_bytes=True
            )

        self.assertEqual(payload, material.data)
        self.assertIsInstance(material.data, bytes)
        self.assertEqual([len(payload), 1], requested_counts)
        self.assertLessEqual(max(requested_counts), len(payload))

    def test_byte_snapshot_short_reads_use_bounded_presized_fallback(self):
        payload = b"short-read-equivalence" * 97
        path = self.root / "short-read.png"
        path.write_bytes(payload)
        requested_counts: list[int] = []
        real_read = os.read

        def short_read(file_descriptor: int, count: int) -> bytes:
            requested_counts.append(count)
            return real_read(file_descriptor, min(count, 17))

        with patch("docarag.asset_material.os.read", side_effect=short_read):
            material = self.resolver(chunk_size=31).resolve(
                asset(os.fspath(path), payload), include_bytes=True
            )

        self.assertEqual(payload, material.data)
        self.assertIsInstance(material.data, bytes)
        self.assertEqual(len(payload), requested_counts[0])
        self.assertTrue(all(count <= 31 for count in requested_counts[1:]))

    def test_byte_snapshot_detects_growth_with_one_bounded_sentinel_byte(self):
        payload = b"a" * 257
        path = self.root / "growing-snapshot.png"
        path.write_bytes(payload)
        real_read = os.read
        changed = False
        requested_counts: list[int] = []

        def grow_after_initial_read(file_descriptor: int, count: int) -> bytes:
            nonlocal changed
            requested_counts.append(count)
            chunk = real_read(file_descriptor, count)
            if chunk and not changed:
                changed = True
                with path.open("ab") as stream:
                    stream.write(b"x")
                    stream.flush()
                    os.fsync(stream.fileno())
            return chunk

        with patch(
            "docarag.asset_material.os.read", side_effect=grow_after_initial_read
        ):
            with self.assertRaises(AssetMaterializationError) as raised:
                self.resolver(max_asset_bytes=512).resolve(
                    asset(os.fspath(path), payload), include_bytes=True
                )

        self.assertEqual(
            AssetMaterialErrorCode.CHANGED_DURING_READ, raised.exception.code
        )
        self.assertEqual([len(payload), 1], requested_counts)

    def test_intermediate_fstat_failure_closes_every_opened_directory(self):
        payload = b"descriptor cleanup"
        path = self.root / "nested" / "asset.png"
        path.parent.mkdir()
        path.write_bytes(payload)
        resolver = self.resolver()
        real_open = os.open
        real_fstat = os.fstat
        real_close = os.close
        opened_descriptors: list[int] = []
        closed_descriptors: list[int] = []
        fstat_calls = 0

        def tracking_open(
            name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
            opened_descriptors.append(descriptor)
            return descriptor

        def failing_fstat(descriptor: int) -> os.stat_result:
            nonlocal fstat_calls
            fstat_calls += 1
            if fstat_calls == 2:
                raise OSError(errno.EIO, "simulated intermediate fstat failure")
            return real_fstat(descriptor)

        def tracking_close(descriptor: int) -> None:
            closed_descriptors.append(descriptor)
            real_close(descriptor)

        with (
            patch("docarag.asset_material.os.open", side_effect=tracking_open),
            patch("docarag.asset_material.os.fstat", side_effect=failing_fstat),
            patch("docarag.asset_material.os.close", side_effect=tracking_close),
        ):
            with self.assertRaises(AssetMaterializationError) as raised:
                resolver.resolve(asset(os.fspath(path), payload))

        self.assertEqual(AssetMaterialErrorCode.IO_ERROR, raised.exception.code)
        self.assertEqual(2, len(opened_descriptors))
        self.assertCountEqual(opened_descriptors, closed_descriptors)

    def test_content_addressed_references_use_configured_store_and_digest(self):
        payload = b"content-addressed table crop"
        sha256 = digest(payload)
        stored = self.content_root / sha256[:2] / sha256
        stored.parent.mkdir()
        stored.write_bytes(payload)
        reference = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=f"cas://sha256/{sha256}",
            media_type="image/png",
            sha256=sha256,
        )

        material = self.resolver().resolve(reference, include_bytes=True)

        self.assertEqual(stored, material.path)
        self.assertEqual("content_addressed", material.reference_kind)
        self.assertEqual(payload, material.data)

        sha_reference = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=f"sha256:{sha256}",
            media_type="image/png",
            sha256=sha256,
        )
        self.assertEqual(stored, self.resolver().resolve(sha_reference).path)

    def test_content_addressed_uri_must_match_declared_digest(self):
        reference = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=f"sha256:{'a' * 64}",
            media_type="image/png",
            sha256="b" * 64,
        )

        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver().resolve(reference)
        self.assertEqual(AssetMaterialErrorCode.DIGEST_MISMATCH, raised.exception.code)

    def test_rejects_missing_digest_remote_uri_and_relative_local_uri(self):
        missing_digest = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=os.fspath(self.root / "image.png"),
            media_type="image/png",
        )
        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver().resolve(missing_digest)
        self.assertEqual(AssetMaterialErrorCode.DIGEST_REQUIRED, raised.exception.code)

        for uri in ("https://example.invalid/image.png", "relative/image.png"):
            reference = AssetRef(
                kind=AssetKind.REGION_CROP,
                uri=uri,
                media_type="image/png",
                sha256="a" * 64,
            )
            with self.subTest(uri=uri), self.assertRaises(
                AssetMaterializationError
            ) as raised:
                self.resolver().resolve(reference)
            self.assertEqual(AssetMaterialErrorCode.UNSUPPORTED_URI, raised.exception.code)

    def test_rejects_path_escape_and_lexical_traversal(self):
        payload = b"outside"
        outside = Path(self.temporary_directory.name) / "outside.png"
        outside.write_bytes(payload)
        references = (
            asset(os.fspath(outside), payload),
            asset(os.fspath(self.root / ".." / "outside.png"), payload),
            asset(f"file://{self.root}/%2e%2e/outside.png", payload),
        )

        for reference in references:
            with self.subTest(uri=reference.uri), self.assertRaises(
                AssetMaterializationError
            ) as raised:
                self.resolver().resolve(reference)
            self.assertEqual(AssetMaterialErrorCode.PATH_ESCAPE, raised.exception.code)

    def test_rejects_final_and_intermediate_symbolic_links(self):
        payload = b"real pixels"
        real_file = self.root / "real.png"
        real_file.write_bytes(payload)
        final_link = self.root / "linked.png"
        final_link.symlink_to(real_file)
        real_directory = self.root / "real-directory"
        real_directory.mkdir()
        nested = real_directory / "nested.png"
        nested.write_bytes(payload)
        directory_link = self.root / "linked-directory"
        directory_link.symlink_to(real_directory, target_is_directory=True)

        for path in (final_link, directory_link / "nested.png"):
            with self.subTest(path=path), self.assertRaises(
                AssetMaterializationError
            ) as raised:
                self.resolver().resolve(asset(os.fspath(path), payload))
            self.assertEqual(AssetMaterialErrorCode.SYMBOLIC_LINK, raised.exception.code)

    def test_rejects_symbolic_allowlist_root(self):
        linked_root = Path(self.temporary_directory.name) / "linked-root"
        linked_root.symlink_to(self.root, target_is_directory=True)

        with self.assertRaises(AssetMaterializationError) as raised:
            AssetMaterialResolver(allowed_roots=(linked_root,))
        self.assertEqual(
            AssetMaterialErrorCode.INVALID_CONFIGURATION, raised.exception.code
        )

    def test_rejects_nonregular_and_oversize_targets_before_reading(self):
        directory = self.root / "directory.png"
        directory.mkdir()
        directory_reference = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=os.fspath(directory),
            media_type="image/png",
            sha256=hashlib.sha256(b"").hexdigest(),
        )
        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver().resolve(directory_reference)
        self.assertEqual(AssetMaterialErrorCode.NOT_REGULAR, raised.exception.code)

        payload = b"x" * 33
        oversized = self.root / "oversized.png"
        oversized.write_bytes(payload)
        with patch("docarag.asset_material.os.read", wraps=os.read) as read:
            with self.assertRaises(AssetMaterializationError) as raised:
                self.resolver(max_asset_bytes=32).resolve(
                    asset(os.fspath(oversized), payload)
                )
        self.assertEqual(AssetMaterialErrorCode.TOO_LARGE, raised.exception.code)
        read.assert_not_called()

    def test_rejects_digest_mismatch(self):
        path = self.root / "mismatch.png"
        path.write_bytes(b"actual bytes")
        reference = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=os.fspath(path),
            media_type="image/png",
            sha256=digest(b"different bytes"),
        )

        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver().resolve(reference)
        self.assertEqual(AssetMaterialErrorCode.DIGEST_MISMATCH, raised.exception.code)

    def test_rejects_change_during_streaming_read(self):
        original = b"a" * 4096
        path = self.root / "changing.png"
        path.write_bytes(original)
        reference = asset(os.fspath(path), original)
        real_read = os.read
        changed = False

        def changing_read(file_descriptor: int, count: int) -> bytes:
            nonlocal changed
            chunk = real_read(file_descriptor, count)
            if chunk and not changed:
                changed = True
                with path.open("r+b") as stream:
                    stream.seek(2048)
                    stream.write(b"b" * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
            return chunk

        with patch("docarag.asset_material.os.read", side_effect=changing_read):
            with self.assertRaises(AssetMaterializationError) as raised:
                self.resolver().resolve(reference)
        self.assertEqual(
            AssetMaterialErrorCode.CHANGED_DURING_READ, raised.exception.code
        )

    def test_rejects_path_replacement_during_read(self):
        original = b"a" * 4096
        path = self.root / "replaced.png"
        path.write_bytes(original)
        reference = asset(os.fspath(path), original)
        real_read = os.read
        replaced = False

        def replacing_read(file_descriptor: int, count: int) -> bytes:
            nonlocal replaced
            chunk = real_read(file_descriptor, count)
            if chunk and not replaced:
                replaced = True
                old_path = self.root / "old-inode.png"
                path.rename(old_path)
                path.write_bytes(original)
            return chunk

        with patch("docarag.asset_material.os.read", side_effect=replacing_read):
            with self.assertRaises(AssetMaterializationError) as raised:
                self.resolver().resolve(reference)
        self.assertEqual(
            AssetMaterialErrorCode.CHANGED_DURING_READ, raised.exception.code
        )

    def test_rejects_same_inode_mutation_after_descriptor_read(self):
        original = b"a" * 4096
        path = self.root / "late-mutation.png"
        path.write_bytes(original)
        reference = asset(os.fspath(path), original)
        real_fstat = os.fstat
        regular_fstats = 0

        def mutate_after_final_fstat(file_descriptor: int) -> os.stat_result:
            nonlocal regular_fstats
            result = real_fstat(file_descriptor)
            if stat.S_ISREG(result.st_mode):
                regular_fstats += 1
                if regular_fstats == 2:
                    with path.open("r+b") as stream:
                        stream.seek(0)
                        stream.write(b"b" * len(original))
                        stream.flush()
                        os.fsync(stream.fileno())
            return result

        with patch(
            "docarag.asset_material.os.fstat", side_effect=mutate_after_final_fstat
        ):
            with self.assertRaises(AssetMaterializationError) as raised:
                self.resolver().resolve(reference)
        self.assertEqual(
            AssetMaterialErrorCode.CHANGED_DURING_READ, raised.exception.code
        )

    def test_batch_limits_and_target_conflicts_fail_closed(self):
        payload = b"one"
        path = self.root / "one.png"
        path.write_bytes(payload)
        reference = asset(os.fspath(path), payload)
        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver(max_assets=1).resolve_many((reference, reference))
        self.assertEqual(AssetMaterialErrorCode.BATCH_LIMIT, raised.exception.code)

        second_payload = b"two-two"
        second_path = self.root / "two.png"
        second_path.write_bytes(second_payload)
        with self.assertRaises(AssetMaterializationError) as raised:
            self.resolver(max_total_bytes=len(payload)).resolve_many(
                (reference, asset(os.fspath(second_path), second_payload))
            )
        self.assertEqual(AssetMaterialErrorCode.BATCH_LIMIT, raised.exception.code)

        conflict = AssetRef(
            kind=AssetKind.REGION_CROP,
            uri=os.fspath(path),
            media_type="image/png",
            sha256="f" * 64,
        )
        resolver = self.resolver()
        with patch("docarag.asset_material.os.open", wraps=os.open) as opened:
            with self.assertRaises(AssetMaterializationError) as raised:
                resolver.resolve_many((reference, conflict))
        self.assertEqual(AssetMaterialErrorCode.TARGET_CONFLICT, raised.exception.code)
        opened.assert_not_called()

    def test_duplicate_target_is_read_once_but_preserves_input_order(self):
        payload = b"deduplicated read"
        path = self.root / "duplicate.png"
        path.write_bytes(payload)
        first = asset(os.fspath(path), payload)
        second = AssetRef(
            kind=AssetKind.PAGE_RENDER,
            uri=f"file://{path}",
            media_type="image/png",
            sha256=digest(payload),
        )
        real_read = os.read
        read_calls = 0

        def counting_read(file_descriptor: int, count: int) -> bytes:
            nonlocal read_calls
            read_calls += 1
            return real_read(file_descriptor, count)

        with patch("docarag.asset_material.os.read", side_effect=counting_read):
            materials = self.resolver().resolve_many((first, second))

        self.assertEqual((first.asset_id, second.asset_id), tuple(x.asset_id for x in materials))
        # One data chunk plus one EOF read, rather than two complete opens.
        self.assertEqual(2, read_calls)


if __name__ == "__main__":
    unittest.main()
