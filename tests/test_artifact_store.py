from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from docarag.artifact_store import (
    ArtifactStoreError,
    ArtifactStoreErrorCode,
    LocalContentAddressedArtifactStore,
    MinerUArchivePublicationLimits,
    make_mineru_archive_publisher,
    make_paddleocr_vl_artifact_sink,
)
from docarag.domain import AssetKind
from docarag.services.mineru import MinerUArchive
from docarag.services.paddleocr_vl import (
    PaddleOCRVLArtifact,
    PaddleOCRVLArtifactRole,
)


def _zip_bytes(members: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)
    return output.getvalue()


def _archive(data: bytes, members: tuple[str, ...]) -> MinerUArchive:
    member_sha256s: dict[str, str] = {}
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            for name in members:
                try:
                    member_data = archive.read(name)
                except KeyError:
                    member_data = b"declared member is absent"
                member_sha256s[name] = hashlib.sha256(member_data).hexdigest()
    except zipfile.BadZipFile:
        member_sha256s = {
            name: hashlib.sha256(b"invalid archive member").hexdigest()
            for name in members
        }
    return MinerUArchive(
        operation_id="operation-1",
        task_id="task-1",
        archive_sha256=hashlib.sha256(data).hexdigest(),
        data=data,
        members=members,
        member_sha256s=member_sha256s,
    )


class LocalContentAddressedArtifactStoreTests(unittest.TestCase):
    def test_configuration_requires_a_scoped_absolute_root(self):
        with self.assertRaises(ArtifactStoreError) as relative_error:
            LocalContentAddressedArtifactStore("relative/cas")
        self.assertEqual(
            ArtifactStoreErrorCode.INVALID_CONFIGURATION,
            relative_error.exception.code,
        )
        with self.assertRaises(ArtifactStoreError) as filesystem_error:
            LocalContentAddressedArtifactStore(Path(os.path.sep), create=False)
        self.assertEqual(
            ArtifactStoreErrorCode.INVALID_CONFIGURATION,
            filesystem_error.exception.code,
        )

    def test_publish_is_resolvable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cas"
            store = LocalContentAddressedArtifactStore(root)
            payload = b"canonical parser artifact"

            first = store.put_bytes(payload, media_type="application/octet-stream")
            second = store.publish(
                payload,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                media_type="application/octet-stream",
            )

            self.assertEqual(first, second)
            self.assertEqual(
                f"cas://sha256/{hashlib.sha256(payload).hexdigest()}", first.uri
            )
            self.assertEqual(payload, first.path.read_bytes())
            self.assertEqual(0, stat.S_IMODE(first.path.stat().st_mode) & 0o222)

            reference = first.to_asset_ref(kind=AssetKind.EXTRACTED_IMAGE)
            material = store.make_asset_resolver(
                max_total_bytes=1024
            ).resolve(reference, include_bytes=True)
            self.assertEqual(payload, material.data)
            self.assertEqual(first.sha256, material.sha256)
            self.assertEqual("content_addressed", material.reference_kind)

    def test_concurrent_publication_has_one_final_inode_and_no_staging_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(Path(temporary) / "cas")
            payload = b"same immutable bytes" * 1024

            with ThreadPoolExecutor(max_workers=12) as executor:
                results = tuple(executor.map(lambda _: store.publish(payload), range(48)))

            self.assertEqual(1, len({result.uri for result in results}))
            digest = results[0].sha256
            prefix = store.root / digest[:2]
            self.assertEqual([digest], sorted(path.name for path in prefix.iterdir()))
            self.assertEqual(payload, results[0].path.read_bytes())

    def test_digest_size_and_conflicting_target_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(
                Path(temporary) / "cas", max_artifact_bytes=4
            )
            with self.assertRaises(ArtifactStoreError) as mismatch:
                store.publish(b"data", expected_sha256="0" * 64)
            self.assertEqual(ArtifactStoreErrorCode.DIGEST_MISMATCH, mismatch.exception.code)
            with self.assertRaises(ArtifactStoreError) as too_large:
                store.publish(b"12345")
            self.assertEqual(ArtifactStoreErrorCode.TOO_LARGE, too_large.exception.code)

            payload = b"data"
            digest = hashlib.sha256(payload).hexdigest()
            prefix = store.root / digest[:2]
            prefix.mkdir(exist_ok=True)
            (prefix / digest).write_bytes(b"evil")
            with self.assertRaises(ArtifactStoreError) as conflict:
                store.publish(payload)
            self.assertEqual(ArtifactStoreErrorCode.TARGET_CONFLICT, conflict.exception.code)
            self.assertEqual(b"evil", (prefix / digest).read_bytes())

    def test_symlinked_root_or_digest_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            actual = temporary_path / "actual"
            actual.mkdir()
            alias = temporary_path / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ArtifactStoreError) as root_error:
                LocalContentAddressedArtifactStore(alias)
            self.assertEqual(ArtifactStoreErrorCode.UNSAFE_PATH, root_error.exception.code)

            root = temporary_path / "cas"
            store = LocalContentAddressedArtifactStore(root)
            payload = b"prefix must be a directory"
            digest = hashlib.sha256(payload).hexdigest()
            outside = temporary_path / "outside"
            outside.mkdir()
            (root / digest[:2]).symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ArtifactStoreError) as prefix_error:
                store.publish(payload)
            self.assertEqual(ArtifactStoreErrorCode.UNSAFE_PATH, prefix_error.exception.code)
            self.assertEqual([], list(outside.iterdir()))

    def test_same_size_mtime_restored_mutation_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(
                Path(temporary) / "cas", chunk_size=64
            )
            payload = b"a" * 4096
            stored = store.publish(payload)
            original_stat = stored.path.stat()
            real_read = os.read
            mutated = False

            def mutate_after_read(descriptor: int, maximum: int) -> bytes:
                nonlocal mutated
                chunk = real_read(descriptor, maximum)
                if chunk and not mutated:
                    mutated = True
                    os.chmod(stored.path, 0o600)
                    with stored.path.open("r+b") as output:
                        output.seek(0)
                        output.write(b"b" * len(chunk))
                        output.flush()
                        os.fsync(output.fileno())
                    os.utime(
                        stored.path,
                        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                    )
                    os.chmod(stored.path, 0o400)
                return chunk

            with patch("docarag.artifact_store.os.read", side_effect=mutate_after_read):
                with self.assertRaises(ArtifactStoreError) as conflict:
                    store.publish(payload)

            self.assertTrue(mutated)
            self.assertEqual(
                ArtifactStoreErrorCode.TARGET_CONFLICT, conflict.exception.code
            )
            self.assertNotEqual(payload, stored.path.read_bytes())


class MinerUArchivePublisherTests(unittest.TestCase):
    def test_publishes_archive_and_every_file_with_exact_resolvable_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(Path(temporary) / "cas")
            members = [
                ("doc/doc_content_list.json", b"[]"),
                ("doc/images/figure.png", b"image bytes"),
                ("doc/empty.txt", b""),
            ]
            encoded = _zip_bytes(members)
            published = make_mineru_archive_publisher(store)(
                _archive(encoded, tuple(name for name, _data in members))
            )

            self.assertEqual(hashlib.sha256(encoded).hexdigest(), published.archive_sha256)
            self.assertEqual(
                f"cas://sha256/{hashlib.sha256(encoded).hexdigest()}",
                published.archive_uri,
            )
            self.assertEqual({name for name, _data in members}, set(published.member_uris))
            resolver = store.make_asset_resolver(
                max_assets=8, max_total_bytes=len(encoded) + 1024
            )
            for name, expected in members:
                digest = published.member_uris[name].removeprefix("cas://sha256/")
                reference = store.publish(expected).to_asset_ref(
                    kind=AssetKind.EXTRACTED_IMAGE
                )
                self.assertEqual(published.member_uris[name], reference.uri)
                self.assertEqual(digest, reference.sha256)
                self.assertEqual(
                    expected,
                    resolver.resolve(reference, include_bytes=True).data,
                )

    def test_traversal_symlink_and_declared_member_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(Path(temporary) / "cas")
            publisher = make_mineru_archive_publisher(store)

            traversal = _zip_bytes([("../escape.txt", b"bad")])
            with self.assertRaises(ArtifactStoreError) as traversal_error:
                publisher(_archive(traversal, ("../escape.txt",)))
            self.assertEqual(
                ArtifactStoreErrorCode.INVALID_ARCHIVE, traversal_error.exception.code
            )

            symlink_output = BytesIO()
            with zipfile.ZipFile(symlink_output, "w") as archive:
                link = zipfile.ZipInfo("doc/link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "../../outside")
            symlink_data = symlink_output.getvalue()
            with self.assertRaises(ArtifactStoreError) as symlink_error:
                publisher(_archive(symlink_data, ("doc/link",)))
            self.assertEqual(
                ArtifactStoreErrorCode.INVALID_ARCHIVE, symlink_error.exception.code
            )

            regular = _zip_bytes([("doc/result.json", b"{}")])
            with self.assertRaises(ArtifactStoreError) as mismatch_error:
                publisher(_archive(regular, ("doc/other.json",)))
            self.assertEqual(
                ArtifactStoreErrorCode.INVALID_ARCHIVE, mismatch_error.exception.code
            )

            wrong_digest = MinerUArchive(
                operation_id="operation-1",
                task_id="task-1",
                archive_sha256=hashlib.sha256(regular).hexdigest(),
                data=regular,
                members=("doc/result.json",),
                member_sha256s={"doc/result.json": "0" * 64},
            )
            with self.assertRaises(ArtifactStoreError) as digest_error:
                publisher(wrong_digest)
            self.assertEqual(
                ArtifactStoreErrorCode.DIGEST_MISMATCH, digest_error.exception.code
            )

    def test_member_count_size_and_compression_ratio_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(Path(temporary) / "cas")
            encoded = _zip_bytes([("a.txt", b"a"), ("b.txt", b"b")])
            publisher = make_mineru_archive_publisher(
                store,
                limits=MinerUArchivePublicationLimits(
                    max_archive_bytes=1024 * 1024,
                    max_members=1,
                    max_member_bytes=1024,
                    max_total_uncompressed_bytes=1024,
                    max_compression_ratio=100,
                ),
            )
            with self.assertRaises(ArtifactStoreError) as count_error:
                publisher(_archive(encoded, ("a.txt", "b.txt")))
            self.assertEqual(ArtifactStoreErrorCode.ARCHIVE_LIMIT, count_error.exception.code)

            compressed = _zip_bytes([("large.txt", b"0" * 20_000)])
            ratio_limited = make_mineru_archive_publisher(
                store,
                limits=MinerUArchivePublicationLimits(
                    max_archive_bytes=1024 * 1024,
                    max_members=2,
                    max_member_bytes=30_000,
                    max_total_uncompressed_bytes=30_000,
                    max_compression_ratio=2,
                ),
            )
            with self.assertRaises(ArtifactStoreError) as ratio_error:
                ratio_limited(_archive(compressed, ("large.txt",)))
            self.assertEqual(ArtifactStoreErrorCode.ARCHIVE_LIMIT, ratio_error.exception.code)


class PaddleArtifactSinkTests(unittest.TestCase):
    def test_sink_returns_verified_cas_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalContentAddressedArtifactStore(Path(temporary) / "cas")
            sink = make_paddleocr_vl_artifact_sink(store)
            offered = PaddleOCRVLArtifact(
                role=PaddleOCRVLArtifactRole.MARKDOWN_IMAGE,
                request_id="request-1",
                document_version_id="document-version-1",
                page_index=3,
                service_log_id="log-1",
                name="images/chart.png",
                media_type="image/png",
                data=b"png fixture bytes",
            )

            stored = sink(offered)

            self.assertEqual(offered.sha256, stored.sha256)
            self.assertEqual("image/png", stored.media_type)
            self.assertEqual(f"cas://sha256/{offered.sha256}", stored.uri)
            reference = store.publish(
                offered.data,
                expected_sha256=offered.sha256,
                media_type=offered.media_type,
            ).to_asset_ref(kind=AssetKind.EXTRACTED_IMAGE)
            self.assertEqual(
                offered.data,
                store.make_asset_resolver(max_total_bytes=1024)
                .resolve(reference, include_bytes=True)
                .data,
            )


if __name__ == "__main__":
    unittest.main()
