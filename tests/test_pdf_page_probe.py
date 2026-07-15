from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from docarag.adapters.pymupdf_probe import PyMuPdfProbeBackend
from docarag.domain import ContractValidationError, DocumentRef
from docarag.page_probe import (
    PageProbeBackendError,
    PageProbeDependencyError,
    PageSelectionError,
    PdfPageSnapshot,
    PdfProbePolicy,
    SourceBindingError,
    SourceBoundPdfPageProbe,
    UnsupportedProbeSourceError,
)
from docarag.ports import PageProbe, PageRange


class _FakeSession:
    def __init__(self, snapshots: dict[int, PdfPageSnapshot], page_count: int) -> None:
        self.snapshots = snapshots
        self.page_count = page_count
        self.inspected: list[int] = []
        self.close_count = 0

    def inspect_page(self, page_index: int) -> PdfPageSnapshot:
        self.inspected.append(page_index)
        return self.snapshots[page_index]

    def close(self) -> None:
        self.close_count += 1


class _FakeBackend:
    backend_name = "fake-pdf"
    backend_version = "1.2.3"

    def __init__(self, snapshots: dict[int, PdfPageSnapshot], page_count: int) -> None:
        self.snapshots = snapshots
        self.page_count = page_count
        self.sessions: list[_FakeSession] = []

    def open(self, source_path: Path) -> _FakeSession:
        session = _FakeSession(self.snapshots, self.page_count)
        self.sessions.append(session)
        return session


def _snapshot(
    page_index: int,
    *,
    text: str = "Native text",
    text_boxes: tuple[tuple[float, float, float, float], ...] = ((0, 0, 50, 20),),
    image_boxes: tuple[tuple[float, float, float, float], ...] = (),
    width: float = 100.0,
    height: float = 100.0,
    rotation: int = 0,
) -> PdfPageSnapshot:
    return PdfPageSnapshot(
        page_index=page_index,
        width=width,
        height=height,
        rotation_degrees=rotation,
        native_text=text,
        text_boxes=text_boxes,
        image_boxes=image_boxes,
        font_count=2,
        metadata={"fixture": True},
    )


class SourceBoundPdfPageProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "source.pdf"
        self.content = b"%PDF-1.7\nsource-bound fake fixture\n%%EOF\n"
        self.path.write_bytes(self.content)

    def document(
        self,
        *,
        page_count: int | None = 3,
        digest: str | None = None,
        mime_type: str = "application/pdf",
        source_uri: str | None = None,
    ) -> DocumentRef:
        return DocumentRef.identified(
            content_sha256=digest or hashlib.sha256(self.content).hexdigest(),
            mime_type=mime_type,
            source_uri=source_uri or str(self.path),
            page_count=page_count,
        )

    def backend(self) -> _FakeBackend:
        return _FakeBackend({index: _snapshot(index) for index in range(3)}, 3)

    def test_port_is_runtime_checkable_and_range_is_half_open(self):
        probe = SourceBoundPdfPageProbe(self.backend())
        self.assertIsInstance(probe, PageProbe)
        self.assertEqual(
            (PageRange(2, 5).start_index, PageRange(2, 5).stop_index),
            (2, 5),
        )
        with self.assertRaises(ContractValidationError):
            PageRange(2, 2)
        with self.assertRaises(ContractValidationError):
            PageRange(True, 2)

    def test_streams_only_selected_pages_in_canonical_order(self):
        backend = self.backend()
        iterator = SourceBoundPdfPageProbe(backend).probe(
            self.document(), page_indices=[2, 0]
        )
        self.assertEqual(backend.sessions, [])

        first = next(iterator)
        self.assertEqual(first.page.page_index, 0)
        self.assertEqual(backend.sessions[0].inspected, [0])
        self.assertEqual(first.page.document_version_id, self.document().document_version_id)
        self.assertEqual(first.mime_type, "application/pdf")
        self.assertTrue(first.metadata["source_sha256_verified"])

        second = next(iterator)
        self.assertEqual(second.page.page_index, 2)
        self.assertEqual(backend.sessions[0].inspected, [0, 2])
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertEqual(backend.sessions[0].close_count, 1)

    def test_exact_rectangle_union_does_not_double_count_overlap(self):
        snapshot = _snapshot(
            0,
            text="same\nsame\nunique\ufffd\x01",
            text_boxes=((0, 0, 60, 50), (40, 0, 100, 50)),
            image_boxes=((0, 50, 75, 100), (25, 50, 100, 100)),
        )
        backend = _FakeBackend({0: snapshot}, 1)
        feature = next(
            SourceBoundPdfPageProbe(backend).probe(self.document(page_count=1))
        )
        self.assertAlmostEqual(feature.text_coverage_ratio, 0.5)
        self.assertAlmostEqual(feature.raster_image_coverage_ratio, 0.5)
        self.assertEqual(feature.native_character_count, len(snapshot.native_text))
        self.assertEqual(feature.native_word_count, 3)
        self.assertGreater(feature.duplicated_text_ratio, 0.0)
        self.assertGreater(feature.replacement_character_ratio, 0.0)
        self.assertGreater(feature.control_character_ratio, 0.0)
        self.assertEqual(feature.candidate_figure_regions, 2)

    def test_unknown_semantic_and_ocr_signals_remain_none(self):
        feature = next(
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(), page_indices=[0]
            )
        )
        for field_name in (
            "ocr_native_disagreement_ratio",
            "connected_component_density",
            "skew_degrees",
            "candidate_table_regions",
            "candidate_equation_regions",
            "candidate_chart_regions",
            "candidate_diagram_regions",
            "column_count",
            "is_slide_like",
            "is_photographed_page",
        ):
            with self.subTest(field=field_name):
                self.assertIsNone(getattr(feature, field_name))
        self.assertIn("candidate_table_regions", feature.metadata["signals_not_computed"])
        self.assertIn("vector_path_density", feature.metadata["unknown_signals"])

    def test_scanned_signal_is_positive_only_and_policy_bound(self):
        snapshots = {
            0: _snapshot(
                0,
                text="",
                text_boxes=(),
                image_boxes=((0, 0, 100, 80),),
            ),
            1: _snapshot(1, image_boxes=()),
        }
        backend = _FakeBackend(snapshots, 2)
        features = list(
            SourceBoundPdfPageProbe(backend).probe(self.document(page_count=2))
        )
        self.assertIs(features[0].is_scanned_page, True)
        self.assertIsNone(features[1].is_scanned_page)

        disabled = SourceBoundPdfPageProbe(
            _FakeBackend({0: snapshots[0]}, 1),
            policy=PdfProbePolicy(infer_scanned_pages=False),
        )
        self.assertIsNone(
            next(disabled.probe(self.document(page_count=1))).is_scanned_page
        )

    def test_hash_mismatch_fails_before_backend_open(self):
        backend = self.backend()
        with self.assertRaisesRegex(SourceBindingError, "SHA-256"):
            SourceBoundPdfPageProbe(backend).probe(
                self.document(digest="0" * 64)
            )
        self.assertEqual(backend.sessions, [])

    def test_source_mutation_is_detected_between_pages(self):
        backend = self.backend()
        iterator = SourceBoundPdfPageProbe(backend).probe(self.document())
        self.assertEqual(next(iterator).page.page_index, 0)
        self.path.write_bytes(self.content + b"changed")
        with self.assertRaisesRegex(SourceBindingError, "changed during probing"):
            next(iterator)
        self.assertEqual(backend.sessions[0].inspected, [0])
        self.assertEqual(backend.sessions[0].close_count, 1)

    def test_duplicate_indices_are_rejected_before_page_inspection(self):
        backend = self.backend()
        iterator = SourceBoundPdfPageProbe(backend).probe(
            self.document(), page_indices=[1, 1]
        )
        with self.assertRaisesRegex(PageSelectionError, "duplicate page index 1"):
            next(iterator)
        self.assertEqual(backend.sessions[0].inspected, [])
        self.assertEqual(backend.sessions[0].close_count, 1)

    def test_overlapping_and_out_of_range_ranges_are_rejected(self):
        backend = self.backend()
        overlap = SourceBoundPdfPageProbe(backend).probe(
            self.document(),
            page_ranges=[PageRange(0, 2), PageRange(1, 3)],
        )
        with self.assertRaisesRegex(PageSelectionError, "overlap"):
            next(overlap)
        self.assertEqual(backend.sessions[0].inspected, [])

        outside = SourceBoundPdfPageProbe(self.backend()).probe(
            self.document(), page_ranges=[PageRange(2, 4)]
        )
        with self.assertRaisesRegex(PageSelectionError, "exceeds"):
            next(outside)

    def test_disjoint_ranges_are_sorted_without_materializing_pages(self):
        backend = self.backend()
        features = list(
            SourceBoundPdfPageProbe(backend).probe(
                self.document(),
                page_ranges=[PageRange(2, 3), PageRange(0, 1)],
            )
        )
        self.assertEqual([item.page.page_index for item in features], [0, 2])
        self.assertEqual(backend.sessions[0].inspected, [0, 2])

    def test_page_count_and_snapshot_page_binding_are_authoritative(self):
        mismatch_backend = self.backend()
        iterator = SourceBoundPdfPageProbe(mismatch_backend).probe(
            self.document(page_count=2)
        )
        with self.assertRaisesRegex(SourceBindingError, "page_count"):
            next(iterator)
        self.assertEqual(mismatch_backend.sessions[0].inspected, [])

        wrong_page = _FakeBackend({0: _snapshot(1)}, 1)
        iterator = SourceBoundPdfPageProbe(wrong_page).probe(
            self.document(page_count=1)
        )
        with self.assertRaisesRegex(SourceBindingError, "bound to page"):
            next(iterator)

    def test_malformed_region_makes_only_its_coverage_unknown(self):
        malformed = PdfPageSnapshot(
            page_index=0,
            width=100,
            height=100,
            rotation_degrees=0,
            native_text="native",
            text_boxes=((0, 0, float("nan"), 10),),
            image_boxes=((0, 0, 100, 50),),
            font_count=1,
        )
        feature = next(
            SourceBoundPdfPageProbe(_FakeBackend({0: malformed}, 1)).probe(
                self.document(page_count=1)
            )
        )
        self.assertIsNone(feature.text_coverage_ratio)
        self.assertEqual(feature.raster_image_coverage_ratio, 0.5)
        self.assertEqual(feature.metadata["invalid_text_box_count"], 1)

    def test_features_are_deterministic_for_same_source_backend_and_policy(self):
        first = next(
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(), page_indices=[0]
            )
        )
        second = next(
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(), page_indices=[0]
            )
        )
        self.assertEqual(first.feature_id, second.feature_id)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_non_pdf_remote_and_ambiguous_selection_fail_closed(self):
        with self.assertRaises(UnsupportedProbeSourceError):
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(mime_type="text/plain")
            )
        with self.assertRaises(UnsupportedProbeSourceError):
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(source_uri="https://example.invalid/source.pdf")
            )
        with self.assertRaises(PageSelectionError):
            SourceBoundPdfPageProbe(self.backend()).probe(
                self.document(),
                page_indices=[0],
                page_ranges=[PageRange(0, 1)],
            )

    def test_local_file_uri_and_explicit_source_override_are_supported(self):
        file_uri_document = self.document(source_uri=self.path.as_uri())
        feature = next(
            SourceBoundPdfPageProbe(self.backend()).probe(
                file_uri_document, page_indices=[0]
            )
        )
        self.assertEqual(feature.page.document_version_id, file_uri_document.document_version_id)

        remote_identity = self.document(source_uri="s3://bucket/source.pdf")
        feature = next(
            SourceBoundPdfPageProbe(self.backend()).probe(
                remote_identity, source_path=self.path, page_indices=[0]
            )
        )
        self.assertEqual(feature.page.document_version_id, remote_identity.document_version_id)


class _FakePyMuPdfPage:
    cropbox = SimpleNamespace(width=200.0, height=100.0)
    cropbox_position = SimpleNamespace(x=10.0, y=20.0)
    rotation = 90

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def get_text(self, mode: str, *, sort: bool):
        self.calls.append(("get_text", mode, sort))
        return [
            (10, 20, 110, 70, "hello world", 0, 0),
            (10, 20, 210, 120, "<image metadata>", 1, 1),
        ]

    def get_image_info(self, *, hashes: bool, xrefs: bool):
        self.calls.append(("get_image_info", hashes, xrefs))
        return [{"bbox": (110, 20, 210, 120)}]

    def get_fonts(self, *, full: bool):
        self.calls.append(("get_fonts", full))
        return [(1, "ttf", "Type0", "Font", "F1"), (1, "ttf", "Type0", "Font", "F1")]

    def get_pixmap(self):
        raise AssertionError("page rendering must never be invoked")

    def get_textpage_ocr(self):
        raise AssertionError("OCR must never be invoked")

    def get_drawings(self):
        raise AssertionError("expanded vector extraction must never be invoked")


class _FakePyMuPdfDocument:
    is_pdf = True
    needs_pass = False
    page_count = 1

    def __init__(self) -> None:
        self.page = _FakePyMuPdfPage()
        self.closed = False

    def load_page(self, page_index: int) -> _FakePyMuPdfPage:
        if page_index != 0:
            raise IndexError(page_index)
        return self.page

    def close(self) -> None:
        self.closed = True


class PyMuPdfProbeBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "probe.pdf"
        self.content = b"%PDF-1.7\nfake\n%%EOF\n"
        self.path.write_bytes(self.content)

    def document(self) -> DocumentRef:
        return DocumentRef.identified(
            content_sha256=hashlib.sha256(self.content).hexdigest(),
            mime_type="application/pdf",
            source_uri=str(self.path),
            page_count=1,
        )

    def test_adapter_uses_metadata_only_apis_and_normalizes_cropbox_offset(self):
        fake_document = _FakePyMuPdfDocument()
        module = SimpleNamespace(
            __version__="9.8.7",
            open=lambda **kwargs: fake_document,
        )
        backend = PyMuPdfProbeBackend(module_loader=lambda: module)
        feature = next(SourceBoundPdfPageProbe(backend).probe(self.document()))

        self.assertEqual(feature.page.width, 200.0)
        self.assertEqual(feature.page.height, 100.0)
        self.assertEqual(feature.page.rotation_degrees, 90)
        self.assertAlmostEqual(feature.text_coverage_ratio, 0.25)
        self.assertAlmostEqual(feature.raster_image_coverage_ratio, 0.5)
        self.assertEqual(feature.font_count, 1)
        self.assertIsNone(feature.vector_path_density)
        self.assertEqual(feature.metadata["probe_backend_version"], "9.8.7")
        self.assertEqual(
            fake_document.page.calls,
            [
                ("get_text", "blocks", False),
                ("get_image_info", False, False),
                ("get_fonts", False),
            ],
        )
        self.assertTrue(fake_document.closed)

    def test_missing_dependency_has_an_actionable_error(self):
        def missing() -> object:
            raise ModuleNotFoundError("pymupdf")

        backend = PyMuPdfProbeBackend(module_loader=missing)
        iterator = SourceBoundPdfPageProbe(backend).probe(self.document())
        with self.assertRaisesRegex(PageProbeDependencyError, "PyMuPDF"):
            next(iterator)

    def test_non_pdf_and_password_protected_documents_are_closed(self):
        for attribute, expected in (("is_pdf", "not a PDF"), ("needs_pass", "password")):
            with self.subTest(attribute=attribute):
                fake_document = _FakePyMuPdfDocument()
                if attribute == "is_pdf":
                    fake_document.is_pdf = False
                else:
                    fake_document.needs_pass = True
                module = SimpleNamespace(open=lambda **kwargs: fake_document)
                backend = PyMuPdfProbeBackend(module_loader=lambda: module)
                with self.assertRaisesRegex(PageProbeBackendError, expected):
                    backend.open(self.path)
                self.assertTrue(fake_document.closed)


if __name__ == "__main__":
    unittest.main()
