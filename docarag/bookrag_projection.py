"""Loss-aware projection from canonical evidence to BookRAG ``pdf_list`` rows.

The canonical parser contract is richer than BookRAG's historical normalized
dictionary shape.  This module keeps the complete canonical representations,
assets, geometry, and lineage on every projected row while also populating the
small set of compatibility fields consumed by the existing tree builder.

The projector is intentionally dependency-light.  Its optional local-material
boundary uses DocARAG's bounded asset verifier, while keeping BookRAG, parser
SDKs, model frameworks, and storage clients out of the controller import path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
from enum import StrEnum
from html import escape as escape_html
from io import StringIO
import json
import os
from pathlib import PurePosixPath
import re
from string import punctuation
from types import MappingProxyType
from threading import RLock
from typing import NoReturn
from urllib.parse import unquote, urlsplit

from docarag.asset_material import AssetMaterialResolver, VerifiedAssetMaterial
from docarag.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    CanonicalBlock,
    DocumentRef,
    Representation,
    RepresentationKind,
)
from docarag.domain.base import json_ready


PROJECTION_NAME = "docarag.canonical-to-bookrag.v1"
BOOKRAG_ASSET_LEASE_SCHEMA_VERSION = "docarag.bookrag-asset-lease.v1"


class BookRAGImageMaterialPolicy(StrEnum):
    """Control whether BookRAG may receive a local image pathname.

    The default accepts only paths produced by an active verified canonical-CAS
    lease.  The long legacy member name is deliberately difficult to select by
    accident: rows produced under it are inspection exports, never indexing or
    evidence-authority inputs.
    """

    VERIFIED_CAS_LEASE_ONLY = "verified_cas_lease_only"
    LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS = (
        "legacy_inspection_allow_unverified_mutable_local_paths"
    )


class ProjectionErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    DOCUMENT_BINDING = "document_binding"
    DUPLICATE_BLOCK = "duplicate_block"
    INVALID_PROVENANCE = "invalid_provenance"
    INVALID_TABLE = "invalid_table"
    INVALID_ASSET_LEASE = "invalid_asset_lease"
    UNREPRESENTABLE_TEXT = "unrepresentable_text"
    UNREPRESENTABLE_TABLE = "unrepresentable_table"
    UNREPRESENTABLE_EQUATION = "unrepresentable_equation"
    UNREPRESENTABLE_VISUAL = "unrepresentable_visual"


class BookRAGProjectionError(ValueError):
    """Base failure carrying a stable machine-readable projection code."""

    def __init__(
        self,
        code: ProjectionErrorCode,
        message: str,
        *,
        block: CanonicalBlock | None = None,
    ) -> None:
        self.code = ProjectionErrorCode(code)
        self.block_id = block.block_id if block is not None else None
        self.page_index = block.page_index if block is not None else None
        suffix = ""
        if block is not None:
            suffix = f" [block_id={block.block_id}, page_index={block.page_index}]"
        super().__init__(f"{message}{suffix}")


class ProjectionBindingError(BookRAGProjectionError):
    """Source identity, block uniqueness, or provenance is inconsistent."""


class ProjectionEvidenceLossError(BookRAGProjectionError):
    """The legacy row would lose the block's only usable evidence."""


class ProjectionTableError(ProjectionEvidenceLossError):
    """A declared structured table cannot be converted without guessing."""


class BookRAGAssetLeaseError(ProjectionBindingError):
    """A local BookRAG image path is not bound to verified canonical CAS bytes."""


@dataclass(frozen=True, slots=True)
class BookRAGAssetMaterializationAttestation:
    """Audit record for one verified canonical-CAS-to-local-path binding.

    This attests encoded bytes only.  It deliberately makes no claim about the
    declared MIME type, decoded image dimensions, or decoder safety.
    """

    asset_id: str
    canonical_asset_uri: str
    sha256: str
    size_bytes: int
    materialized_path: str
    media_type: str | None
    reference_kind: str
    schema_version: str = field(
        default=BOOKRAG_ASSET_LEASE_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization attestation requires an asset_id",
            )
        if (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization attestation requires a lowercase SHA-256",
            )
        expected_uri = f"cas://sha256/{self.sha256}"
        if self.canonical_asset_uri != expected_uri:
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization URI is not digest-consistent canonical CAS",
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization size_bytes must be a non-negative integer",
            )
        if (
            not isinstance(self.materialized_path, str)
            or not self.materialized_path
            or not os.path.isabs(self.materialized_path)
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization path must be absolute",
            )
        if self.reference_kind != "content_addressed":
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "BookRAG asset leases accept content-addressed material only",
            )
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization media_type must be non-empty when provided",
            )

    @classmethod
    def from_verified_material(
        cls,
        asset: AssetRef,
        material: VerifiedAssetMaterial,
    ) -> "BookRAGAssetMaterializationAttestation":
        if (
            material.asset_id != asset.asset_id
            or material.asset_uri != asset.uri
            or material.sha256 != asset.sha256
            or material.media_type != asset.media_type
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "verified material does not bind the requested canonical AssetRef",
            )
        return cls(
            asset_id=asset.asset_id,
            canonical_asset_uri=asset.uri,
            sha256=material.sha256,
            size_bytes=material.size_bytes,
            materialized_path=os.fspath(material.path),
            media_type=material.media_type,
            reference_kind=material.reference_kind,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": "pre_use_verified_cas_lease",
            "asset_id": self.asset_id,
            "canonical_asset_uri": self.canonical_asset_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "materialized_path": self.materialized_path,
            "media_type": self.media_type,
            "reference_kind": self.reference_kind,
            "verification_scope": "bounded_descriptor_read_snapshot",
            "post_use_reverification": "required_on_lease_close_before_adoption",
            "byte_limit_scope": "encoded_file_bytes_only",
            "decoded_content_validation": "not_performed",
        }


class BookRAGAssetMaterializationLease:
    """Logical lifetime for CAS paths handed to legacy BookRAG consumers.

    Acquisition uses :class:`AssetMaterialResolver`, so callers provide no
    local path.  Only exact ``cas://sha256/<digest>`` assets are accepted.  The
    resolver's verified physical CAS path is suitable for ``img_path`` while
    the canonical URI and asset ID remain authoritative.  ``close()`` performs
    a second bounded descriptor read and fails if any path, size, or digest no
    longer matches; keep the lease active through KG extraction/checkpointing.

    The lease does not delete CAS objects.  It owns verification lifetime, not
    storage lifetime, which keeps concurrent readers and retries safe.
    """

    __slots__ = (
        "_resolver",
        "_assets",
        "_attestations",
        "_active",
        "_lock",
    )

    def __init__(
        self,
        resolver: AssetMaterialResolver,
        assets: Sequence[AssetRef],
    ) -> None:
        if not isinstance(resolver, AssetMaterialResolver):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "resolver must be an AssetMaterialResolver",
            )
        if not isinstance(assets, Sequence) or isinstance(
            assets, (str, bytes, bytearray)
        ):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "lease assets must be a sequence of AssetRef values",
            )
        unique: dict[str, AssetRef] = {}
        for index, asset in enumerate(assets):
            if not isinstance(asset, AssetRef):
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    f"lease assets[{index}] must be an AssetRef",
                )
            if asset.sha256 is None or asset.uri != f"cas://sha256/{asset.sha256}":
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "BookRAG image materialization requires an exact digest-bound "
                    "cas://sha256 AssetRef",
                )
            previous = unique.get(asset.asset_id)
            if previous is not None and previous != asset:
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "one asset_id resolves to conflicting canonical AssetRef values",
                )
            unique.setdefault(asset.asset_id, asset)
        if not unique:
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization lease must contain at least one asset",
            )
        ordered_assets = tuple(unique[key] for key in sorted(unique))
        materials = resolver.resolve_many(ordered_assets, include_bytes=False)
        attestations = {
            asset.asset_id: BookRAGAssetMaterializationAttestation.from_verified_material(
                asset, material
            )
            for asset, material in zip(ordered_assets, materials, strict=True)
        }
        self._resolver = resolver
        self._assets = MappingProxyType(
            {asset.asset_id: asset for asset in ordered_assets}
        )
        self._attestations = MappingProxyType(attestations)
        self._active = True
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def require_active(self) -> None:
        if not self.active:
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization lease is closed",
            )

    def has_asset(self, asset_id: str) -> bool:
        with self._lock:
            return self._active and asset_id in self._assets

    def attestation_for(
        self, asset: AssetRef
    ) -> BookRAGAssetMaterializationAttestation:
        if not isinstance(asset, AssetRef):
            raise BookRAGAssetLeaseError(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset materialization lookup requires an AssetRef",
            )
        with self._lock:
            if not self._active:
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "asset materialization lease is closed",
                )
            bound = self._assets.get(asset.asset_id)
            if bound is None or bound != asset:
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "asset is not exactly bound by this materialization lease",
                )
            return self._attestations[asset.asset_id]

    def verify(self) -> None:
        """Re-read all leased CAS paths and require the exact attestations."""

        with self._lock:
            if not self._active:
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "asset materialization lease is closed",
                )
            ordered_assets = tuple(
                self._assets[key] for key in sorted(self._assets)
            )
            materials = self._resolver.resolve_many(
                ordered_assets, include_bytes=False
            )
            current = {
                asset.asset_id: BookRAGAssetMaterializationAttestation.from_verified_material(
                    asset, material
                )
                for asset, material in zip(
                    ordered_assets, materials, strict=True
                )
            }
            if current != dict(self._attestations):
                raise BookRAGAssetLeaseError(
                    ProjectionErrorCode.INVALID_ASSET_LEASE,
                    "CAS materialization changed during the BookRAG lease",
                )

    def close(self) -> None:
        """Re-verify once, then revoke this lease's use of handed-out paths."""

        with self._lock:
            if not self._active:
                return
            try:
                self.verify()
            finally:
                self._active = False

    def __enter__(self) -> "BookRAGAssetMaterializationLease":
        self.require_active()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


_TEXT_BLOCKS = frozenset(
    {
        BlockType.DOC_TITLE,
        BlockType.TITLE,
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
        BlockType.CODE,
        BlockType.ALGORITHM,
        BlockType.ABSTRACT,
        BlockType.REFERENCE,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE,
        BlockType.UNKNOWN,
    }
)
_VISUAL_BLOCKS = frozenset({BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM})

# What a text-carrying block becomes in the tree. Anything absent keeps the
# historical "text", so a block type only gains its own node type when there
# is a downstream reason for it -- a summariser prompt, an index record shape,
# or a distinct way of rendering it in an answer.
_NODE_TYPE_BY_BLOCK = {
    BlockType.CODE: "code",
    BlockType.ALGORITHM: "algorithm",
    BlockType.ABSTRACT: "abstract",
    BlockType.REFERENCE: "reference",
}
# Visual blocks already project as "image"; chart and diagram are separated
# because a chart's values and a diagram's topology are read differently from
# a photograph.
_VISUAL_NODE_TYPE_BY_BLOCK = {
    BlockType.CHART: "chart",
    BlockType.DIAGRAM: "diagram",
}
_VISUAL_SUBTYPES = frozenset(
    {
        "chart",
        "diagram",
        "figure",
        "flow_chart",
        "flowchart",
        "image",
        "picture",
        "scan_region",
        "screenshot",
        "sketch",
        "workflow",
    }
)
_MARKDOWN_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_IMAGE_SUFFIXES = frozenset(
    {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_MARKDOWN_ESCAPABLE = frozenset(punctuation)


def _raise_binding(
    code: ProjectionErrorCode,
    message: str,
    block: CanonicalBlock | None = None,
) -> NoReturn:
    raise ProjectionBindingError(code, message, block=block)


def _raise_loss(
    code: ProjectionErrorCode,
    message: str,
    block: CanonicalBlock,
) -> NoReturn:
    raise ProjectionEvidenceLossError(code, message, block=block)


def _projection_sort_key(block: CanonicalBlock) -> tuple[object, ...]:
    return (
        block.page_index,
        block.reading_order is None,
        0.0 if block.reading_order is None else float(block.reading_order),
        block.block_id,
    )


def _primary_representation_id(
    block: CanonicalBlock, kind: RepresentationKind
) -> str | None:
    values = block.metadata.get("primary_representations")
    if not isinstance(values, Mapping):
        return None
    value = values.get(kind.value)
    return value if isinstance(value, str) and value else None


def _representation_sort_key(
    representation: Representation,
    *,
    primary_id: str | None = None,
) -> tuple[object, ...]:
    return (
        representation.representation_id != primary_id,
        representation.confidence is None,
        -(representation.confidence or 0.0),
        representation.representation_id,
        representation.content_sha256,
    )


def _representations_of_kind(
    block: CanonicalBlock, kind: RepresentationKind
) -> tuple[Representation, ...]:
    primary_id = _primary_representation_id(block, kind)
    values = tuple(item for item in block.representations if item.kind is kind)
    return tuple(
        sorted(values, key=lambda item: _representation_sort_key(item, primary_id=primary_id))
    )


def _select_textual_representation(
    block: CanonicalBlock,
    kinds: Sequence[RepresentationKind],
) -> Representation | None:
    for kind in kinds:
        candidates = _representations_of_kind(block, kind)
        for candidate in candidates:
            if isinstance(candidate.content, str) and candidate.content.strip():
                return candidate
    return None


def _text_entries(
    block: CanonicalBlock, kinds: Sequence[RepresentationKind]
) -> list[Representation]:
    result: list[Representation] = []
    seen: set[str] = set()
    for kind in kinds:
        for representation in _representations_of_kind(block, kind):
            content = representation.content
            if not isinstance(content, str) or not content.strip() or content in seen:
                continue
            result.append(representation)
            seen.add(content)
    return result


def _text_values(
    block: CanonicalBlock, kinds: Sequence[RepresentationKind]
) -> list[str]:
    return [str(item.content) for item in _text_entries(block, kinds)]


def _ordered_material_ids(values: Sequence[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _cell_text(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise ProjectionTableError(
        ProjectionErrorCode.INVALID_TABLE,
        f"{field} must be a scalar table-cell value",
    )


def _rectangular_rows(value: object, *, field: str) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            f"{field} must be an array of rows",
        )
    rows: list[list[str]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, Sequence) or isinstance(
            raw_row, (str, bytes, bytearray)
        ):
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{field}[{row_index}] must be an array",
            )
        rows.append(
            [
                _cell_text(item, field=f"{field}[{row_index}][{column_index}]")
                for column_index, item in enumerate(raw_row)
            ]
        )
    if not rows or not rows[0]:
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            f"{field} must contain at least one non-empty row",
        )
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            f"{field} must be rectangular",
        )
    return rows


def _matrix_to_html(rows: Sequence[Sequence[str]], *, header: bool = False) -> str:
    output = ["<table>"]
    start = 0
    if header:
        output.append("<thead><tr>")
        output.extend(f"<th>{escape_html(value)}</th>" for value in rows[0])
        output.append("</tr></thead>")
        start = 1
    output.append("<tbody>")
    for row in rows[start:]:
        output.append("<tr>")
        output.extend(f"<td>{escape_html(value)}</td>" for value in row)
        output.append("</tr>")
    output.append("</tbody></table>")
    return "".join(output)


def _descriptor_table_to_html(
    cells_value: object,
    *,
    num_rows: object = None,
    num_cols: object = None,
    field: str,
) -> str:
    if not isinstance(cells_value, Sequence) or isinstance(
        cells_value, (str, bytes, bytearray)
    ):
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            f"{field} must be an array of cell objects",
        )
    cells = list(cells_value)
    if not cells or any(not isinstance(cell, Mapping) for cell in cells):
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            f"{field} must contain cell objects",
        )

    normalized: list[dict[str, object]] = []
    max_row = 0
    max_col = 0
    for index, raw_cell in enumerate(cells):
        assert isinstance(raw_cell, Mapping)
        row = raw_cell.get("start_row_offset_idx", raw_cell.get("row", raw_cell.get("row_index")))
        column = raw_cell.get(
            "start_col_offset_idx", raw_cell.get("col", raw_cell.get("column_index"))
        )
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{field}[{index}] has an invalid start row",
            )
        if isinstance(column, bool) or not isinstance(column, int) or column < 0:
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{field}[{index}] has an invalid start column",
            )
        end_row = raw_cell.get("end_row_offset_idx")
        end_col = raw_cell.get("end_col_offset_idx")
        if end_row is None:
            row_span = raw_cell.get("row_span", 1)
            if isinstance(row_span, bool) or not isinstance(row_span, int) or row_span < 1:
                raise ProjectionTableError(
                    ProjectionErrorCode.INVALID_TABLE,
                    f"{field}[{index}] has an invalid row span",
                )
            end_row = row + row_span
        if end_col is None:
            col_span = raw_cell.get("col_span", 1)
            if isinstance(col_span, bool) or not isinstance(col_span, int) or col_span < 1:
                raise ProjectionTableError(
                    ProjectionErrorCode.INVALID_TABLE,
                    f"{field}[{index}] has an invalid column span",
                )
            end_col = column + col_span
        if (
            isinstance(end_row, bool)
            or not isinstance(end_row, int)
            or end_row <= row
            or isinstance(end_col, bool)
            or not isinstance(end_col, int)
            or end_col <= column
        ):
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{field}[{index}] has invalid exclusive end offsets",
            )
        text = _cell_text(
            raw_cell.get("text", raw_cell.get("value", raw_cell.get("content"))),
            field=f"{field}[{index}].text",
        )
        normalized.append(
            {
                "row": row,
                "column": column,
                "end_row": end_row,
                "end_col": end_col,
                "text": text,
                "header": bool(
                    raw_cell.get("column_header", False)
                    or raw_cell.get("row_header", False)
                    or raw_cell.get("header", False)
                ),
            }
        )
        max_row = max(max_row, end_row)
        max_col = max(max_col, end_col)

    def dimension(value: object, inferred: int, name: str) -> int:
        if value is None:
            return inferred
        if isinstance(value, bool) or not isinstance(value, int) or value < inferred:
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{name} must be an integer covering all declared cells",
            )
        return value

    rows = dimension(num_rows, max_row, "num_rows")
    columns = dimension(num_cols, max_col, "num_cols")
    starts: dict[tuple[int, int], Mapping[str, object]] = {}
    covered: set[tuple[int, int]] = set()
    for cell in sorted(
        normalized,
        key=lambda item: (int(item["row"]), int(item["column"])),
    ):
        start = (int(cell["row"]), int(cell["column"]))
        if start in starts or start in covered:
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                f"{field} contains overlapping cells",
            )
        starts[start] = cell
        for row in range(int(cell["row"]), int(cell["end_row"])):
            for column in range(int(cell["column"]), int(cell["end_col"])):
                position = (row, column)
                if position == start:
                    continue
                if position in starts or position in covered:
                    raise ProjectionTableError(
                        ProjectionErrorCode.INVALID_TABLE,
                        f"{field} contains overlapping spans",
                    )
                covered.add(position)

    output = ["<table><tbody>"]
    for row in range(rows):
        output.append("<tr>")
        for column in range(columns):
            position = (row, column)
            if position in covered:
                continue
            cell = starts.get(position)
            if cell is None:
                output.append("<td></td>")
                continue
            tag = "th" if cell["header"] else "td"
            attributes = ""
            rowspan = int(cell["end_row"]) - int(cell["row"])
            colspan = int(cell["end_col"]) - int(cell["column"])
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            output.append(
                f"<{tag}{attributes}>{escape_html(str(cell['text']))}</{tag}>"
            )
        output.append("</tr>")
    output.append("</tbody></table>")
    return "".join(output)


def _table_grid_to_html(content: object) -> str:
    if isinstance(content, Mapping):
        table_cells = content.get("table_cells")
        has_table_cells = table_cells is not None and not (
            isinstance(table_cells, Sequence)
            and not isinstance(table_cells, (str, bytes, bytearray))
            and not table_cells
        )
        if has_table_cells:
            return _descriptor_table_to_html(
                table_cells,
                num_rows=content.get("num_rows"),
                num_cols=content.get("num_cols"),
                field="table_grid.table_cells",
            )
        for key in ("rows", "grid", "cells"):
            value = content.get(key)
            if value is None:
                continue
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and not value
            ):
                continue
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and value
                and isinstance(value[0], Mapping)
            ):
                return _descriptor_table_to_html(
                    value,
                    num_rows=content.get("num_rows"),
                    num_cols=content.get("num_cols"),
                    field=f"table_grid.{key}",
                )
            return _matrix_to_html(
                _rectangular_rows(value, field=f"table_grid.{key}")
            )
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            "table_grid object requires rows, grid, cells, or table_cells",
        )
    return _matrix_to_html(_rectangular_rows(content, field="table_grid"))


def _split_markdown_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    trailing_backslashes = 0
    if value.endswith("|"):
        trailing_backslashes = len(value[:-1]) - len(value[:-1].rstrip("\\"))
    if value.endswith("|") and trailing_backslashes % 2 == 0:
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            if character not in _MARKDOWN_ESCAPABLE:
                current.append("\\")
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip().replace("<br>", "\n"))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip().replace("<br>", "\n"))
    return cells


def _markdown_to_html(content: str) -> str:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE, "Markdown table must not be empty"
        )
    rows = [_split_markdown_row(line) for line in lines]
    header = False
    if len(rows) >= 2 and all(
        _MARKDOWN_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in rows[1]
    ):
        if len(rows[1]) != len(rows[0]):
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                "Markdown separator must match the header column count",
            )
        header = True
        del rows[1]
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE,
            "Markdown table must have a consistent column count",
        )
    return _matrix_to_html(rows, header=header)


def _csv_to_html(content: str) -> str:
    try:
        rows = list(csv.reader(StringIO(content), strict=True))
    except csv.Error as exc:
        raise ProjectionTableError(
            ProjectionErrorCode.INVALID_TABLE, f"CSV table is invalid: {exc}"
        ) from exc
    normalized = _rectangular_rows(rows, field="csv")
    return _matrix_to_html(normalized)


def _table_body(
    block: CanonicalBlock,
) -> tuple[str | None, Representation | None, tuple[str, ...]]:
    converters = (
        (RepresentationKind.HTML, lambda value: value),
        (RepresentationKind.MARKDOWN, _markdown_to_html),
        (RepresentationKind.CSV, _csv_to_html),
        (RepresentationKind.TABLE_GRID, _table_grid_to_html),
    )
    conversion_errors: list[str] = []
    for kind, converter in converters:
        for representation in _representations_of_kind(block, kind):
            content = representation.content
            if kind is not RepresentationKind.TABLE_GRID and not isinstance(content, str):
                continue
            try:
                result = converter(content)  # type: ignore[arg-type]
            except ProjectionTableError as exc:
                conversion_errors.append(
                    f"{kind.value}:{representation.representation_id}: {exc}"
                )
                continue
            if isinstance(result, str) and result.strip():
                return result, representation, tuple(conversion_errors)
    return None, None, tuple(conversion_errors)


def _safe_local_asset_path(asset: AssetRef) -> str | None:
    uri = asset.uri
    if any(ord(character) < 32 or ord(character) == 127 for character in uri):
        return None
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return None
    if parsed.scheme:
        if parsed.scheme.casefold() != "file" or parsed.netloc.casefold() not in {
            "",
            "localhost",
        }:
            return None
        if parsed.query or parsed.fragment:
            return None
        encoded_path = parsed.path
    else:
        if parsed.netloc:
            return None
        encoded_path = uri
    try:
        path = unquote(encoded_path, errors="strict")
    except UnicodeDecodeError:
        return None
    normalized_path = path.replace("\\", "/")
    if (
        not path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or path.startswith("~")
        or normalized_path.startswith("//")
    ):
        return None
    normalized_parts = PurePosixPath(normalized_path).parts
    if ".." in normalized_parts:
        return None
    return path


def _asset_sort_key(asset: AssetRef) -> tuple[object, ...]:
    kind_rank = {
        AssetKind.REGION_CROP: 0,
        AssetKind.EXTRACTED_IMAGE: 1,
        AssetKind.PAGE_RENDER: 2,
        AssetKind.SOURCE: 3,
    }
    return (kind_rank[asset.kind], asset.asset_id)


def _asset_is_image_compatible(asset: AssetRef, path: str = "") -> bool:
    suffix = PurePosixPath(path.casefold()).suffix
    if (
        asset.media_type is not None
        and not asset.media_type.casefold().startswith("image/")
        and suffix not in _IMAGE_SUFFIXES
    ):
        return False
    if asset.kind is AssetKind.SOURCE and (
        asset.media_type is None
        or not asset.media_type.casefold().startswith("image/")
    ) and suffix not in _IMAGE_SUFFIXES:
        return False
    return bool(
        (asset.media_type or "").casefold().startswith("image/")
        or suffix in _IMAGE_SUFFIXES
        or asset.kind
        in {
            AssetKind.REGION_CROP,
            AssetKind.EXTRACTED_IMAGE,
            AssetKind.PAGE_RENDER,
        }
    )


def _select_local_image(
    block: CanonicalBlock,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None = None,
    *,
    image_material_policy: BookRAGImageMaterialPolicy = (
        BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
    ),
) -> tuple[
    str,
    AssetRef | None,
    BookRAGAssetMaterializationAttestation | None,
]:
    if asset_materialization_lease is not None:
        # Safe path: a caller cannot inject a pathname.  It can only lease an
        # exact canonical CAS AssetRef that AssetMaterialResolver has read and
        # hashed through its bound content-addressed root.
        asset_materialization_lease.require_active()
        for asset in sorted(block.assets, key=_asset_sort_key):
            if not asset_materialization_lease.has_asset(asset.asset_id):
                continue
            attestation = asset_materialization_lease.attestation_for(asset)
            if not _asset_is_image_compatible(
                asset, attestation.materialized_path
            ):
                continue
            return attestation.materialized_path, asset, attestation
        return "", None, None

    if (
        image_material_policy
        is BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
    ):
        return "", None, None

    # Explicit historical compatibility path. It is reference-only, is never
    # presented as verified material, and cannot support evidence authority.
    for asset in sorted(block.assets, key=_asset_sort_key):
        path = _safe_local_asset_path(asset)
        if path is None:
            continue
        if not _asset_is_image_compatible(asset, path):
            continue
        return path, asset, None
    return "", None, None


def _geometry_payload(
    block: CanonicalBlock,
) -> tuple[dict[str, object] | None, list[float] | None]:
    geometry = block.geometry
    if geometry is None:
        return None, None
    payload = geometry.to_dict()
    if geometry.bbox is not None:
        return payload, geometry.bbox.to_list()
    xs = [float(point.x) for point in geometry.polygon]
    ys = [float(point.y) for point in geometry.polygon]
    return payload, [min(xs), min(ys), max(xs), max(ys)]


def _source_parsers(block: CanonicalBlock) -> list[dict[str, object]]:
    raw = block.metadata.get("source_parsers")
    if raw is None:
        # ``CanonicalBlock.from_observation`` does not synthesize parser metadata.
        # Retain one explicit, non-speculative lineage record per observation
        # instead of returning an empty list or inventing parser/model details.
        return [
            {
                "observation_id": observation_id,
                "provenance_available": False,
            }
            for observation_id in sorted(block.source_observation_ids)
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        _raise_binding(
            ProjectionErrorCode.INVALID_PROVENANCE,
            "canonical metadata.source_parsers must be an array",
            block,
        )
    sources: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            _raise_binding(
                ProjectionErrorCode.INVALID_PROVENANCE,
                f"canonical metadata.source_parsers[{index}] must be an object",
                block,
            )
        observation_id = item.get("observation_id")
        if (
            not isinstance(observation_id, str)
            or observation_id not in block.source_observation_ids
        ):
            _raise_binding(
                ProjectionErrorCode.INVALID_PROVENANCE,
                "parser provenance references an observation outside the canonical block",
                block,
            )
        if observation_id in seen:
            _raise_binding(
                ProjectionErrorCode.INVALID_PROVENANCE,
                "parser provenance contains a duplicate observation",
                block,
            )
        seen.add(observation_id)
        ready = json_ready(item)
        assert isinstance(ready, dict)
        sources.append(ready)
    if seen != set(block.source_observation_ids):
        _raise_binding(
            ProjectionErrorCode.INVALID_PROVENANCE,
            "parser provenance does not cover every source observation",
            block,
        )
    return sorted(
        sources,
        key=lambda item: (
            str(item.get("observation_id", "")),
            str(item.get("parser_name", "")),
            str(item.get("parser_native_id", "")),
        ),
    )


def _projection_transformations(source_parsers: Sequence[Mapping[str, object]]) -> list[str]:
    values: set[str] = {PROJECTION_NAME}
    for source in source_parsers:
        transformations = source.get("transformations", ())
        if isinstance(transformations, Sequence) and not isinstance(
            transformations, (str, bytes, bytearray)
        ):
            values.update(
                item for item in transformations if isinstance(item, str) and item
            )
    return sorted(values)


def _base_row(
    document: DocumentRef,
    block: CanonicalBlock,
    *,
    pdf_id: int,
) -> dict[str, object]:
    geometry, bbox = _geometry_payload(block)
    source_parsers = _source_parsers(block)
    metadata = json_ready(block.metadata)
    assert isinstance(metadata, dict)
    selected_observation_id = metadata.get("selected_observation_id")
    if selected_observation_id is None:
        selected_observation_id = block.source_observation_ids[0]
    elif (
        not isinstance(selected_observation_id, str)
        or selected_observation_id not in block.source_observation_ids
    ):
        _raise_binding(
            ProjectionErrorCode.INVALID_PROVENANCE,
            "selected_observation_id must reference source_observation_ids",
            block,
        )
    semantic_subtype = block.semantic_subtype or block.block_type.value
    representations = [
        item.to_dict()
        for item in sorted(
            block.representations,
            key=lambda value: (
                value.kind.value,
                value.representation_id,
                value.confidence is None,
                -(value.confidence or 0.0),
            ),
        )
    ]
    assets = [item.to_dict() for item in sorted(block.assets, key=_asset_sort_key)]
    representation_ids = [item["representation_id"] for item in representations]
    asset_ids = [item["asset_id"] for item in assets]
    canonical_relations = [item.to_dict() for item in block.relations]
    semantic_substructures = [
        item.to_dict() for item in block.semantic_substructures
    ]
    semantic_substructure_refs = [
        {
            "substructure_id": item.substructure_id,
            "semantic_sha256": item.semantic_sha256,
            "comparison_sha256": item.comparison_sha256,
            "source_observation_ids": list(item.source_observation_ids),
            "source_substructure_ids": list(item.source_substructure_ids),
        }
        for item in block.semantic_substructures
    ]
    source_ref = {
        "document_version_id": document.document_version_id,
        "source_sha256": document.content_sha256,
        "page_id": block.page_id,
        "page_idx": block.page_index,
        "canonical_block_id": block.block_id,
        "canonical_revision_id": block.revision_id,
        "observation_id": selected_observation_id,
        "source_observation_ids": list(block.source_observation_ids),
        "geometry": geometry,
        "representation_ids": representation_ids,
        "asset_ids": asset_ids,
        "canonical_relations": canonical_relations,
    }
    if semantic_substructure_refs:
        source_ref["semantic_substructure_refs"] = semantic_substructure_refs
    middle_json = {
        "bbox": bbox,
        "geometry": geometry,
        "polygon": None if geometry is None else geometry["polygon"],
        "coordinate_space": None if geometry is None else geometry["coordinate_space"],
        "coordinate_origin": None if geometry is None else geometry["coordinate_origin"],
        "rotation_degrees": None if geometry is None else geometry["rotation_degrees"],
        "index": block.reading_order,
        "canonical_block_id": block.block_id,
        "canonical_revision_id": block.revision_id,
    }
    row = {
        "pdf_id": pdf_id,
        "page_idx": block.page_index,
        "middle_json": middle_json,
        "document_id": document.document_id,
        "document_version_id": document.document_version_id,
        "source_uri": document.source_uri,
        "source_sha256": document.content_sha256,
        "page_id": block.page_id,
        "canonical_block_id": block.block_id,
        "canonical_revision_id": block.revision_id,
        "canonical_key": block.canonical_key,
        "canonical_revision": block.revision,
        "canonical_content_sha256": block.content_sha256,
        "source_canonical_block_ids": [block.block_id],
        "source_canonical_revision_ids": [block.revision_id],
        "observation_id": selected_observation_id,
        "source_observation_ids": list(block.source_observation_ids),
        "source_evidence_refs": [source_ref],
        "parser_provenance": {
            "selected_observation_id": selected_observation_id,
            "sources": source_parsers,
        },
        "parser_provenance_list": source_parsers,
        "geometry": geometry,
        "semantic_subtype": semantic_subtype,
        "quality_status": block.quality_status.value,
        "reconciliation_method": block.reconciliation_method,
        "parent_block_id": block.parent_block_id,
        "reading_order": block.reading_order,
        "representations": representations,
        "assets": assets,
        "canonical_representation_ids": representation_ids,
        "canonical_asset_ids": asset_ids,
        "canonical_relations": canonical_relations,
        "canonical_metadata": metadata,
        "transformations": _projection_transformations(source_parsers),
    }
    if semantic_substructures:
        row["semantic_substructures"] = semantic_substructures
        row["canonical_semantic_substructure_ids"] = [
            item["substructure_id"] for item in semantic_substructures
        ]
    return row


def _project_text(block: CanonicalBlock, row: dict[str, object]) -> None:
    priorities: dict[BlockType, tuple[RepresentationKind, ...]] = {
        BlockType.TITLE: (
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
            RepresentationKind.MARKDOWN,
        ),
        BlockType.LIST_ITEM: (
            RepresentationKind.MARKDOWN,
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
        ),
        BlockType.CODE: (
            RepresentationKind.MARKDOWN,
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
        ),
        BlockType.CAPTION: (
            RepresentationKind.CAPTION,
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
        ),
        BlockType.FOOTNOTE: (
            RepresentationKind.FOOTNOTE,
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
        ),
    }
    selected = _select_textual_representation(
        block,
        priorities.get(
            block.block_type,
            (
                RepresentationKind.TEXT,
                RepresentationKind.OCR_TEXT,
                RepresentationKind.MARKDOWN,
                RepresentationKind.CAPTION,
                RepresentationKind.FOOTNOTE,
                RepresentationKind.VISUAL_DESCRIPTION,
            ),
        ),
    )
    if selected is None:
        _raise_loss(
            ProjectionErrorCode.UNREPRESENTABLE_TEXT,
            "text-compatible canonical block has no textual representation",
            block,
        )
    row.update(
        {
            "type": _NODE_TYPE_BY_BLOCK.get(block.block_type, "text"),
            "text": selected.content,
            "text_format": selected.kind.value,
            "selected_representation_id": selected.representation_id,
            "selected_representation_ids": [selected.representation_id],
            "selected_asset_ids": [],
        }
    )
    if block.block_type is BlockType.TITLE:
        level = block.metadata.get("level")
        if isinstance(level, int) and not isinstance(level, bool) and 1 <= level <= 6:
            row["text_level"] = level
        elif selected.kind is RepresentationKind.MARKDOWN:
            match = re.match(r"^(#{1,6})\s+", str(selected.content))
            if match:
                row["text_level"] = len(match.group(1))


def _project_table(
    block: CanonicalBlock,
    row: dict[str, object],
    *,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None,
    image_material_policy: BookRAGImageMaterialPolicy,
) -> None:
    body, selected, conversion_errors = _table_body(block)
    image_path, image_asset, image_attestation = _select_local_image(
        block,
        asset_materialization_lease,
        image_material_policy=image_material_policy,
    )
    caption_entries = _text_entries(block, (RepresentationKind.CAPTION,))
    footnote_entries = _text_entries(block, (RepresentationKind.FOOTNOTE,))
    if body is None and not image_path:
        if conversion_errors:
            raise ProjectionTableError(
                ProjectionErrorCode.INVALID_TABLE,
                "no declared table representation could be converted: "
                + "; ".join(conversion_errors),
                block=block,
            )
        _raise_loss(
            ProjectionErrorCode.UNREPRESENTABLE_TABLE,
            "table has neither structured table content nor a safe local crop",
            block,
        )
    row.update(
        {
            "type": "table",
            "table_body": body or "",
            "table_caption": [str(item.content) for item in caption_entries],
            "table_footnote": [str(item.content) for item in footnote_entries],
            "img_path": image_path,
            "selected_asset_id": image_asset.asset_id if image_asset else None,
            "selected_representation_id": (
                selected.representation_id if selected is not None else None
            ),
            "selected_representation_ids": _ordered_material_ids(
                [
                    selected.representation_id if selected is not None else None,
                    *(item.representation_id for item in caption_entries),
                    *(item.representation_id for item in footnote_entries),
                ]
            ),
            "selected_asset_ids": _ordered_material_ids(
                [image_asset.asset_id if image_asset else None]
            ),
            "table_body_source_kind": selected.kind.value if selected is not None else None,
            "table_conversion_errors": list(conversion_errors),
        }
    )
    if image_attestation is not None:
        row["asset_materialization_attestation"] = image_attestation.to_dict()


def _project_equation(
    block: CanonicalBlock,
    row: dict[str, object],
    *,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None,
    image_material_policy: BookRAGImageMaterialPolicy,
) -> None:
    selected = _select_textual_representation(
        block,
        (
            RepresentationKind.VLM_LATEX,
            RepresentationKind.LATEX,
            RepresentationKind.MATHML,
            RepresentationKind.TEXT,
            RepresentationKind.OCR_TEXT,
            RepresentationKind.MARKDOWN,
        ),
    )
    image_path, image_asset, image_attestation = _select_local_image(
        block,
        asset_materialization_lease,
        image_material_policy=image_material_policy,
    )
    if selected is None and not image_path:
        _raise_loss(
            ProjectionErrorCode.UNREPRESENTABLE_EQUATION,
            "equation has neither formula text nor a safe local crop",
            block,
        )
    row.update(
        {
            "type": "equation",
            "text": "" if selected is None else selected.content,
            "text_format": "image" if selected is None else selected.kind.value,
            "img_path": image_path,
            "selected_asset_id": image_asset.asset_id if image_asset else None,
            "selected_representation_id": (
                selected.representation_id if selected is not None else None
            ),
            "selected_representation_ids": _ordered_material_ids(
                [selected.representation_id if selected is not None else None]
            ),
            "selected_asset_ids": _ordered_material_ids(
                [image_asset.asset_id if image_asset else None]
            ),
        }
    )
    if image_attestation is not None:
        row["asset_materialization_attestation"] = image_attestation.to_dict()


def _project_visual(
    block: CanonicalBlock,
    row: dict[str, object],
    *,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None,
    image_material_policy: BookRAGImageMaterialPolicy,
) -> None:
    caption_entries = _text_entries(block, (RepresentationKind.CAPTION,))
    compatibility_entries = _text_entries(
        block,
        (
            RepresentationKind.CAPTION,
            RepresentationKind.VISUAL_DESCRIPTION,
            RepresentationKind.OCR_TEXT,
            RepresentationKind.TEXT,
        ),
    )
    footnote_entries = _text_entries(block, (RepresentationKind.FOOTNOTE,))
    captions = [str(item.content) for item in caption_entries]
    compatibility_text = [str(item.content) for item in compatibility_entries]
    footnotes = [str(item.content) for item in footnote_entries]
    image_path, image_asset, image_attestation = _select_local_image(
        block,
        asset_materialization_lease,
        image_material_policy=image_material_policy,
    )
    if not compatibility_text and not footnotes and not image_path:
        _raise_loss(
            ProjectionErrorCode.UNREPRESENTABLE_VISUAL,
            "visual block has neither textual evidence nor a safe local image",
            block,
        )
    row.update(
        {
            "type": _VISUAL_NODE_TYPE_BY_BLOCK.get(block.block_type, "image"),
            # BookRAG has only caption/footnote text channels for an image leaf.
            # Preserve every selected textual visual channel in the caption field;
            # the rich representations retain their original semantics alongside it.
            "image_caption": compatibility_text,
            "image_footnote": footnotes,
            "explicit_image_captions": captions,
            "img_path": image_path,
            "selected_asset_id": image_asset.asset_id if image_asset else None,
            "selected_representation_ids": _ordered_material_ids(
                [
                    *(item.representation_id for item in compatibility_entries),
                    *(item.representation_id for item in footnote_entries),
                ]
            ),
            "selected_asset_ids": _ordered_material_ids(
                [image_asset.asset_id if image_asset else None]
            ),
        }
    )
    if image_attestation is not None:
        row["asset_materialization_attestation"] = image_attestation.to_dict()


def _is_visual(block: CanonicalBlock) -> bool:
    if block.block_type in _VISUAL_BLOCKS:
        return True
    subtype = re.sub(
        r"[\s-]+",
        "_",
        (block.semantic_subtype or "").strip().casefold(),
    )
    return subtype in _VISUAL_SUBTYPES


def _project_block(
    document: DocumentRef,
    block: CanonicalBlock,
    *,
    pdf_id: int,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None,
    image_material_policy: BookRAGImageMaterialPolicy,
) -> dict[str, object]:
    row = _base_row(document, block, pdf_id=pdf_id)
    if block.block_type is BlockType.TABLE:
        _project_table(
            block,
            row,
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )
    elif block.block_type is BlockType.EQUATION:
        _project_equation(
            block,
            row,
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )
    elif _is_visual(block):
        _project_visual(
            block,
            row,
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )
    elif block.block_type in _TEXT_BLOCKS:
        _project_text(block, row)
    else:  # defensive against future additive enum values
        _raise_loss(
            ProjectionErrorCode.UNREPRESENTABLE_TEXT,
            f"unsupported canonical block type: {block.block_type.value}",
            block,
        )

    legacy_inspection = (
        image_material_policy
        is BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
    )
    if "img_path" not in row:
        image_material_status = "not_applicable"
    elif "asset_materialization_attestation" in row:
        image_material_status = "verified_cas_lease"
    elif row.get("img_path"):
        image_material_status = "legacy_inspection_unverified_mutable_local_path"
    else:
        image_material_status = "no_verified_local_image_selected"
    row.update(
        {
            "image_material_policy": image_material_policy.value,
            "image_material_status": image_material_status,
            "legacy_inspection_only": legacy_inspection,
        }
    )
    source_refs = row.get("source_evidence_refs")
    if isinstance(source_refs, list) and len(source_refs) == 1 and isinstance(
        source_refs[0], dict
    ):
        source_refs[0]["selected_representation_ids"] = list(
            row.get("selected_representation_ids", [])
        )
        source_refs[0]["selected_asset_ids"] = list(
            row.get("selected_asset_ids", [])
        )
    return row


def project_canonical_blocks(
    document: DocumentRef,
    blocks: Sequence[CanonicalBlock],
    *,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None = None,
    image_material_policy: BookRAGImageMaterialPolicy = (
        BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
    ),
    validate_relation_targets: bool = True,
) -> list[dict[str, object]]:
    """Project one current canonical revision per block into BookRAG rows.

    Output order is ``page_index``, known ``reading_order``, then stable
    ``block_id``.  ``pdf_id`` is one-based to match BookRAG's historical
    ``enumerate_pdf_list`` behavior.  Projection performs no filesystem I/O.
    An optional lease must already have verified exact canonical CAS bytes; its
    physical path is emitted solely for BookRAG's legacy ``img_path`` API.
    Without a lease the default withholds local paths. The explicitly named
    ``LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS`` policy is an
    inspection/migration escape hatch; every output row is marked
    ``legacy_inspection_only`` and is never indexing or evidence truth.
    """

    if not isinstance(document, DocumentRef):
        _raise_binding(
            ProjectionErrorCode.INVALID_INPUT, "document must be a DocumentRef"
        )
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes, bytearray)):
        _raise_binding(
            ProjectionErrorCode.INVALID_INPUT,
            "blocks must be a sequence of CanonicalBlock values",
        )
    if not isinstance(image_material_policy, BookRAGImageMaterialPolicy):
        _raise_binding(
            ProjectionErrorCode.INVALID_INPUT,
            "image_material_policy must be a BookRAGImageMaterialPolicy enum value",
        )
    if asset_materialization_lease is not None:
        if not isinstance(
            asset_materialization_lease, BookRAGAssetMaterializationLease
        ):
            _raise_binding(
                ProjectionErrorCode.INVALID_ASSET_LEASE,
                "asset_materialization_lease must be a BookRAGAssetMaterializationLease",
            )
        if (
            image_material_policy
            is BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS
        ):
            _raise_binding(
                ProjectionErrorCode.INVALID_INPUT,
                "verified asset_materialization_lease cannot be combined with the "
                "legacy-inspection image material policy",
            )
        asset_materialization_lease.require_active()

    ordered: list[CanonicalBlock] = []
    block_ids: set[str] = set()
    revision_ids: set[str] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, CanonicalBlock):
            _raise_binding(
                ProjectionErrorCode.INVALID_INPUT,
                f"blocks[{index}] must be a CanonicalBlock",
            )
        if block.document_version_id != document.document_version_id:
            _raise_binding(
                ProjectionErrorCode.DOCUMENT_BINDING,
                "canonical block belongs to a different document version",
                block,
            )
        if document.page_count is not None and block.page_index >= document.page_count:
            _raise_binding(
                ProjectionErrorCode.DOCUMENT_BINDING,
                "canonical block page exceeds document.page_count",
                block,
            )
        metadata_source_hash = block.metadata.get("source_sha256")
        if metadata_source_hash is not None and metadata_source_hash != document.content_sha256:
            _raise_binding(
                ProjectionErrorCode.DOCUMENT_BINDING,
                "canonical metadata source hash differs from DocumentRef",
                block,
            )
        if block.block_id in block_ids or block.revision_id in revision_ids:
            _raise_binding(
                ProjectionErrorCode.DUPLICATE_BLOCK,
                "canonical input contains a duplicate block revision",
                block,
            )
        block_ids.add(block.block_id)
        revision_ids.add(block.revision_id)
        ordered.append(block)

    if validate_relation_targets:
        for block in ordered:
            missing_targets = sorted(
                {
                    relation.target_block_id
                    for relation in block.relations
                    if relation.target_block_id not in block_ids
                }
            )
            if missing_targets:
                _raise_binding(
                    ProjectionErrorCode.INVALID_PROVENANCE,
                    "canonical relation target is outside the projected block set: "
                    + ", ".join(missing_targets),
                    block,
                )

    ordered.sort(key=_projection_sort_key)
    return [
        _project_block(
            document,
            block,
            pdf_id=index,
            asset_materialization_lease=asset_materialization_lease,
            image_material_policy=image_material_policy,
        )
        for index, block in enumerate(ordered, start=1)
    ]


def project_to_bookrag_pdf_list(
    document: DocumentRef,
    blocks: Sequence[CanonicalBlock],
    *,
    asset_materialization_lease: BookRAGAssetMaterializationLease | None = None,
    image_material_policy: BookRAGImageMaterialPolicy = (
        BookRAGImageMaterialPolicy.VERIFIED_CAS_LEASE_ONLY
    ),
) -> list[dict[str, object]]:
    """Explicitly named alias for callers at the BookRAG composition boundary."""

    return project_canonical_blocks(
        document,
        blocks,
        asset_materialization_lease=asset_materialization_lease,
        image_material_policy=image_material_policy,
    )


__all__ = [
    "BOOKRAG_ASSET_LEASE_SCHEMA_VERSION",
    "BookRAGImageMaterialPolicy",
    "PROJECTION_NAME",
    "BookRAGAssetLeaseError",
    "BookRAGAssetMaterializationAttestation",
    "BookRAGAssetMaterializationLease",
    "BookRAGProjectionError",
    "ProjectionBindingError",
    "ProjectionErrorCode",
    "ProjectionEvidenceLossError",
    "ProjectionTableError",
    "project_canonical_blocks",
    "project_to_bookrag_pdf_list",
]
