"""Bounded normalization of explicit parser-native semantic structures.

This module deliberately accepts only fields that a parser/worker emitted as
structured data.  It never reconstructs tables from HTML, formula graphs from
LaTeX/plain text, or diagram edges from prose, geometry, or node order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import TypeAlias
from xml.etree import ElementTree
from xml.parsers import expat

from docarag.domain.base import ContractValidationError, JsonValue, json_ready
from docarag.domain.evidence import (
    MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE,
    MAX_SEMANTIC_JSON_BYTES,
    MAX_SEMANTIC_JSON_DEPTH,
    MAX_SEMANTIC_JSON_NODES,
    MAX_SEMANTIC_RELATIONS_PER_STRUCTURE,
    MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT,
    MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE,
    SEMANTIC_NORMALIZED_EDGE_TYPE_KEY,
    Representation,
    SemanticElement,
    SemanticElementKind,
    SemanticProducerKind,
    SemanticRelation,
    SemanticRelationKind,
    SemanticSourceSelector,
    SemanticStructureAvailability,
    SemanticStructureKind,
    SemanticSubstructure,
    normalize_semantic_edge_type,
)


NORMALIZER_NAME = "explicit_parser_semantic_structure.v1"
MAX_MATHML_BYTES = 262_144
MAX_MATHML_ELEMENTS = MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE
MAX_MATHML_DEPTH = MAX_SEMANTIC_JSON_DEPTH

_JsonScalar: TypeAlias = str | int
_DTD_OR_ENTITY = re.compile(br"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    raise ContractValidationError(f"{field} must be an array")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ContractValidationError(f"{field} keys must be strings")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{field} must be a positive integer")
    return value


def _optional_probability(value: object, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ContractValidationError(f"{field} must be between zero and one")
    return float(value)


def _json_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def preflight_explicit_json(value: object, *, field: str) -> None:
    """Bound JSON depth, node count and encoded bytes before DTO recursion."""

    nodes = 0
    active: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    while stack:
        current, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(current))
            continue
        nodes += 1
        if nodes > MAX_SEMANTIC_JSON_NODES:
            raise ContractValidationError(
                f"{field} exceeds {MAX_SEMANTIC_JSON_NODES} JSON nodes"
            )
        if depth > MAX_SEMANTIC_JSON_DEPTH:
            raise ContractValidationError(
                f"{field} exceeds JSON depth {MAX_SEMANTIC_JSON_DEPTH}"
            )
        if current is None or isinstance(current, (str, bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ContractValidationError(f"{field} contains a non-finite number")
            continue
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise ContractValidationError(f"{field} contains a reference cycle")
            if any(not isinstance(key, str) for key in current):
                raise ContractValidationError(f"{field} contains a non-string key")
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend((item, depth + 1, False) for item in current.values())
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in active:
                raise ContractValidationError(f"{field} contains a reference cycle")
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend((item, depth + 1, False) for item in current)
            continue
        raise ContractValidationError(f"{field} contains a non-JSON value")
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractValidationError(f"{field} is not bounded JSON") from exc
    if len(encoded) > MAX_SEMANTIC_JSON_BYTES:
        raise ContractValidationError(
            f"{field} exceeds {MAX_SEMANTIC_JSON_BYTES} encoded bytes"
        )


def _table_role(cell: Mapping[str, object], field: str) -> str:
    explicit = cell.get("role")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ContractValidationError(f"{field}.role must be a non-empty string")
        return explicit.strip()
    flags: dict[str, bool] = {}
    for name in ("column_header", "row_header", "is_header"):
        value = cell.get(name)
        if value is not None and not isinstance(value, bool):
            raise ContractValidationError(f"{field}.{name} must be a boolean")
        flags[name] = value is True
    if flags["column_header"] and flags["row_header"]:
        return "column_and_row_header"
    if flags["column_header"]:
        return "column_header"
    if flags["row_header"]:
        return "row_header"
    if flags["is_header"]:
        return "header"
    return "body"


def _table_relations(
    cells: Sequence[SemanticElement],
) -> tuple[SemanticRelation, ...]:
    occupancy: dict[tuple[int, int], SemanticElement] = {}
    for cell in cells:
        assert cell.row_index is not None
        assert cell.column_index is not None
        assert cell.row_span is not None
        assert cell.column_span is not None
        for row in range(cell.row_index, cell.row_index + cell.row_span):
            for column in range(
                cell.column_index, cell.column_index + cell.column_span
            ):
                position = (row, column)
                if position in occupancy:
                    raise ContractValidationError(
                        "explicit table contains overlapping cell spans"
                    )
                occupancy[position] = cell

    pairs: set[tuple[SemanticRelationKind, str, str]] = set()
    for cell in cells:
        assert cell.row_index is not None
        assert cell.column_index is not None
        assert cell.row_span is not None
        assert cell.column_span is not None
        right_column = cell.column_index + cell.column_span
        for row in range(cell.row_index, cell.row_index + cell.row_span):
            target = occupancy.get((row, right_column))
            if target is not None and target.element_id != cell.element_id:
                pairs.add(
                    (
                        SemanticRelationKind.TABLE_ADJACENT_RIGHT,
                        cell.element_id,
                        target.element_id,
                    )
                )
        below_row = cell.row_index + cell.row_span
        for column in range(
            cell.column_index, cell.column_index + cell.column_span
        ):
            target = occupancy.get((below_row, column))
            if target is not None and target.element_id != cell.element_id:
                pairs.add(
                    (
                        SemanticRelationKind.TABLE_ADJACENT_BELOW,
                        cell.element_id,
                        target.element_id,
                    )
                )
    if len(pairs) > MAX_SEMANTIC_RELATIONS_PER_STRUCTURE:
        raise ContractValidationError("explicit table adjacency limit exceeded")
    by_id = {item.element_id: item for item in cells}
    return tuple(
        SemanticRelation(
            kind=kind,
            source_element_id=source_id,
            target_element_id=target_id,
            source_selector_ids=tuple(
                selector.selector_id
                for endpoint in (by_id[source_id], by_id[target_id])
                for selector in endpoint.source_selectors
            ),
            metadata={"derived_from": "explicit_cell_coordinates_and_spans"},
        )
        for kind, source_id, target_id in sorted(
            pairs, key=lambda item: (item[0].value, item[1], item[2])
        )
    )


def _finish_table(
    cells: Sequence[SemanticElement],
    *,
    row_count: int,
    column_count: int,
    source_field: str,
    parser_name: str,
) -> SemanticSubstructure:
    if not cells:
        raise ContractValidationError(f"{source_field} contains no table cells")
    if len(cells) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
        raise ContractValidationError(
            f"{source_field} exceeds {MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE} cells"
        )
    if row_count > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE // column_count:
        raise ContractValidationError(
            f"{source_field} dimensions exceed "
            f"{MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE} positions"
        )
    occupied = sum(
        int(cell.row_span or 0) * int(cell.column_span or 0) for cell in cells
    )
    availability = (
        SemanticStructureAvailability.COMPLETE
        if occupied == row_count * column_count
        else SemanticStructureAvailability.PARTIAL
    )
    return SemanticSubstructure(
        kind=SemanticStructureKind.TABLE,
        elements=tuple(cells),
        relations=_table_relations(cells),
        producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
        availability=availability,
        source_field_paths=(source_field,),
        metadata={
            "normalizer": NORMALIZER_NAME,
            "parser": parser_name,
            "row_count": row_count,
            "column_count": column_count,
            "cell_assertions": "parser_native",
            "adjacency": "deterministic_from_asserted_coordinates_and_spans",
        },
    )


def docling_table_substructure(
    data: Mapping[str, object],
    representation: Representation,
    *,
    confidence: float | None,
) -> SemanticSubstructure:
    """Normalize Docling's explicit ``data.table_cells`` contract."""

    preflight_explicit_json(data, field="table.data")
    rows = _positive_int(data.get("num_rows"), "table.data.num_rows")
    columns = _positive_int(data.get("num_cols"), "table.data.num_cols")
    if rows > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE // columns:
        raise ContractValidationError(
            "table.data dimensions exceed the semantic table position limit"
        )
    raw_cells = _sequence(data.get("table_cells"), "table.data.table_cells")
    if not raw_cells:
        raise ContractValidationError(
            "table.data.table_cells must be non-empty for typed table structure"
        )
    if len(raw_cells) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
        raise ContractValidationError("table.data.table_cells exceeds the cell limit")
    elements: list[SemanticElement] = []
    for index, raw_cell in enumerate(raw_cells):
        field = f"table.data.table_cells[{index}]"
        cell = _mapping(raw_cell, field)
        row = _non_negative_int(cell.get("start_row_offset_idx"), f"{field}.start_row_offset_idx")
        row_end = _positive_int(cell.get("end_row_offset_idx"), f"{field}.end_row_offset_idx")
        column = _non_negative_int(cell.get("start_col_offset_idx"), f"{field}.start_col_offset_idx")
        column_end = _positive_int(cell.get("end_col_offset_idx"), f"{field}.end_col_offset_idx")
        if row_end <= row or column_end <= column or row_end > rows or column_end > columns:
            raise ContractValidationError(f"{field} span falls outside the table")
        text = cell.get("text", "")
        if not isinstance(text, str):
            raise ContractValidationError(f"{field}.text must be a string")
        selector = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector=f"/table_cells/{index}",
            metadata={"raw_source_path": f"data.table_cells[{index}]"},
        )
        elements.append(
            SemanticElement(
                local_key=f"r{row}c{column}",
                kind=SemanticElementKind.TABLE_CELL,
                source_selectors=(selector,),
                content=text,
                role=_table_role(cell, field),
                row_index=row,
                column_index=column,
                row_span=row_end - row,
                column_span=column_end - column,
                confidence=confidence,
                metadata={"native_cell_index": index},
            )
        )
    return _finish_table(
        elements,
        row_count=rows,
        column_count=columns,
        source_field="data.table_cells",
        parser_name="docling",
    )


def _first_int(
    value: Mapping[str, object], names: Sequence[str], field: str, *, positive: bool
) -> int | None:
    present = [name for name in names if value.get(name) is not None]
    if not present:
        return None
    first = present[0]
    parser = _positive_int if positive else _non_negative_int
    parsed = parser(value[first], f"{field}.{first}")
    for name in present[1:]:
        if parser(value[name], f"{field}.{name}") != parsed:
            raise ContractValidationError(f"{field} contains conflicting {names}")
    return parsed


def explicit_table_grid_substructure(
    grid_value: object,
    representation: Representation,
    *,
    source_field: str,
    parser_name: str,
    confidence: float | None,
) -> SemanticSubstructure:
    """Normalize common, explicit JSON grid/cell shapes without HTML inference."""

    preflight_explicit_json(grid_value, field=source_field)
    grid = _mapping(grid_value, source_field)
    rows_name = "rows" if grid.get("rows") is not None else "grid"
    raw_rows_value = grid.get(rows_name)
    raw_rows = (
        ()
        if raw_rows_value is None
        else _sequence(raw_rows_value, f"{source_field}.{rows_name}")
    )
    if len(raw_rows) > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE:
        raise ContractValidationError(f"{source_field}.{rows_name} exceeds row limit")
    row_values: list[Sequence[object]] = []
    row_width = 0
    positions = 0
    for row_index, raw_row in enumerate(raw_rows):
        row = _sequence(raw_row, f"{source_field}.{rows_name}[{row_index}]")
        positions += len(row)
        if positions > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE:
            raise ContractValidationError(f"{source_field}.{rows_name} exceeds position limit")
        row_width = max(row_width, len(row))
        row_values.append(row)

    raw_cells_value = grid.get("cells")
    raw_cells = (
        ()
        if raw_cells_value is None
        else _sequence(raw_cells_value, f"{source_field}.cells")
    )
    if len(raw_cells) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
        raise ContractValidationError(f"{source_field}.cells exceeds cell limit")

    parsed_cells: list[tuple[Mapping[str, object], int, int, int, int, int]] = []
    maximum_row = 0
    maximum_column = 0
    for index, raw_cell in enumerate(raw_cells):
        field = f"{source_field}.cells[{index}]"
        cell = _mapping(raw_cell, field)
        row = _first_int(
            cell,
            ("row", "row_index", "start_row_offset_idx"),
            field,
            positive=False,
        )
        column = _first_int(
            cell,
            ("col", "column", "column_index", "start_col_offset_idx"),
            field,
            positive=False,
        )
        if row is None or column is None:
            raise ContractValidationError(f"{field} requires explicit row and column")
        declared_row_span = _first_int(
            cell, ("row_span", "rowspan"), field, positive=True
        )
        declared_column_span = _first_int(
            cell, ("col_span", "column_span", "colspan"), field, positive=True
        )
        row_span = 1 if declared_row_span is None else declared_row_span
        column_span = (
            1 if declared_column_span is None else declared_column_span
        )
        if cell.get("end_row_offset_idx") is not None:
            row_end = _positive_int(cell["end_row_offset_idx"], f"{field}.end_row_offset_idx")
            if row_end <= row or (
                declared_row_span is not None
                and row_end - row != declared_row_span
            ):
                raise ContractValidationError(f"{field} has conflicting row span")
            row_span = row_end - row
        if cell.get("end_col_offset_idx") is not None:
            column_end = _positive_int(cell["end_col_offset_idx"], f"{field}.end_col_offset_idx")
            if column_end <= column or (
                declared_column_span is not None
                and column_end - column != declared_column_span
            ):
                raise ContractValidationError(f"{field} has conflicting column span")
            column_span = column_end - column
        maximum_row = max(maximum_row, row + row_span)
        maximum_column = max(maximum_column, column + column_span)
        parsed_cells.append((cell, row, column, row_span, column_span, index))

    row_count = _first_int(
        grid, ("num_rows", "row_count"), source_field, positive=True
    )
    column_count = _first_int(
        grid, ("num_cols", "column_count"), source_field, positive=True
    )
    row_count = max(len(row_values), maximum_row) if row_count is None else row_count
    column_count = max(row_width, maximum_column) if column_count is None else column_count
    if row_count < 1 or column_count < 1:
        raise ContractValidationError(f"{source_field} requires non-empty dimensions")
    if row_count > MAX_SEMANTIC_TABLE_POSITIONS_PER_STRUCTURE // column_count:
        raise ContractValidationError(f"{source_field} dimensions exceed position limit")
    if len(row_values) > row_count or row_width > column_count:
        raise ContractValidationError(f"{source_field} rows exceed declared dimensions")

    elements: list[SemanticElement] = []
    if parsed_cells:
        for cell, row, column, row_span, column_span, index in parsed_cells:
            if row + row_span > row_count or column + column_span > column_count:
                raise ContractValidationError(f"{source_field}.cells[{index}] falls outside dimensions")
            selectors = [
                SemanticSourceSelector(
                    representation_id=representation.representation_id,
                    native_selector=f"/cells/{index}",
                    metadata={"raw_source_path": f"{source_field}.cells[{index}]"},
                )
            ]
            content: object = None
            for name in ("text", "content", "value"):
                if name in cell:
                    content = cell[name]
                    break
            if content is None and row < len(row_values) and column < len(row_values[row]):
                content = row_values[row][column]
                selectors.append(
                    SemanticSourceSelector(
                        representation_id=representation.representation_id,
                        native_selector=f"/{rows_name}/{row}/{column}",
                        metadata={"raw_source_path": f"{source_field}.{rows_name}[{row}][{column}]"},
                    )
                )
            elements.append(
                SemanticElement(
                    local_key=f"r{row}c{column}",
                    kind=SemanticElementKind.TABLE_CELL,
                    source_selectors=tuple(selectors),
                    content=json_ready(content),  # type: ignore[arg-type]
                    role=_table_role(cell, f"{source_field}.cells[{index}]"),
                    row_index=row,
                    column_index=column,
                    row_span=row_span,
                    column_span=column_span,
                    confidence=_optional_probability(cell.get("confidence", confidence), f"{source_field}.cells[{index}].confidence"),
                    metadata={"native_cell_index": index},
                )
            )
    else:
        if positions > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
            raise ContractValidationError(f"{source_field}.{rows_name} exceeds cell limit")
        for row, values in enumerate(row_values):
            for column, content in enumerate(values):
                selector = SemanticSourceSelector(
                    representation_id=representation.representation_id,
                    native_selector=f"/{rows_name}/{row}/{column}",
                    metadata={"raw_source_path": f"{source_field}.{rows_name}[{row}][{column}]"},
                )
                elements.append(
                    SemanticElement(
                        local_key=f"r{row}c{column}",
                        kind=SemanticElementKind.TABLE_CELL,
                        source_selectors=(selector,),
                        content=json_ready(content),  # type: ignore[arg-type]
                        role="body",
                        row_index=row,
                        column_index=column,
                        row_span=1,
                        column_span=1,
                        confidence=confidence,
                        metadata={"derived_cell_from": f"{rows_name}_position"},
                    )
                )
    return _finish_table(
        elements,
        row_count=row_count,
        column_count=column_count,
        source_field=source_field,
        parser_name=parser_name,
    )


def _node_reference(value: object, field: str) -> _JsonScalar:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ContractValidationError(f"{field} must be a string or integer node ID")
    if isinstance(value, str) and not value.strip():
        raise ContractValidationError(f"{field} must not be empty")
    return value


def _node_token(value: _JsonScalar) -> tuple[str, _JsonScalar]:
    return ("integer", value) if isinstance(value, int) else ("string", value)


def explicit_diagram_substructure(
    graph_value: object,
    representation: Representation,
    *,
    source_field: str,
    parser_name: str,
    confidence: float | None,
    flowchart: bool,
) -> SemanticSubstructure:
    """Normalize an explicit directed node/edge graph; never synthesize edges."""

    preflight_explicit_json(graph_value, field=source_field)
    graph = _mapping(graph_value, source_field)
    raw_nodes = graph.get("nodes")
    if raw_nodes is None:
        raise ContractValidationError(f"{source_field}.nodes is required")
    node_records: list[tuple[_JsonScalar, object, str, str | None]] = []
    if isinstance(raw_nodes, Mapping):
        if len(raw_nodes) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
            raise ContractValidationError(f"{source_field}.nodes exceeds node limit")
        for raw_key, raw_node in raw_nodes.items():
            key = _node_reference(raw_key, f"{source_field}.nodes key")
            selector = f"/nodes/{_json_pointer_token(raw_key)}"
            role = None
            if isinstance(raw_node, Mapping):
                role_value = raw_node.get("type", raw_node.get("kind", raw_node.get("role")))
                if role_value is not None:
                    if not isinstance(role_value, str) or not role_value.strip():
                        raise ContractValidationError(f"{source_field}{selector}.type must be text")
                    role = role_value.strip()
            node_records.append((key, raw_node, selector, role))
    else:
        nodes = _sequence(raw_nodes, f"{source_field}.nodes")
        if not nodes:
            raise ContractValidationError(f"{source_field}.nodes must not be empty")
        if len(nodes) > MAX_SEMANTIC_ELEMENTS_PER_STRUCTURE:
            raise ContractValidationError(f"{source_field}.nodes exceeds node limit")
        for index, raw_node in enumerate(nodes):
            role = None
            if isinstance(raw_node, Mapping):
                identity_fields = [
                    name for name in ("id", "node_id", "key") if raw_node.get(name) is not None
                ]
                key: _JsonScalar = index
                if identity_fields:
                    key = _node_reference(raw_node[identity_fields[0]], f"{source_field}.nodes[{index}].{identity_fields[0]}")
                    for name in identity_fields[1:]:
                        if _node_token(_node_reference(raw_node[name], f"{source_field}.nodes[{index}].{name}")) != _node_token(key):
                            raise ContractValidationError(f"{source_field}.nodes[{index}] has conflicting IDs")
                role_value = raw_node.get("type", raw_node.get("kind", raw_node.get("role")))
                if role_value is not None:
                    if not isinstance(role_value, str) or not role_value.strip():
                        raise ContractValidationError(f"{source_field}.nodes[{index}].type must be text")
                    role = role_value.strip()
            else:
                key = _node_reference(raw_node, f"{source_field}.nodes[{index}]")
            node_records.append((key, raw_node, f"/nodes/{index}", role))
    if not node_records:
        raise ContractValidationError(f"{source_field}.nodes must not be empty")

    raw_edges_value = graph.get("edges", ())
    raw_edges = _sequence(raw_edges_value, f"{source_field}.edges")
    if len(raw_edges) > MAX_SEMANTIC_RELATIONS_PER_STRUCTURE:
        raise ContractValidationError(f"{source_field}.edges exceeds edge limit")
    node_indices: dict[tuple[str, _JsonScalar], int] = {}
    for index, (key, _content, _selector, _role) in enumerate(node_records):
        token = _node_token(key)
        if token in node_indices:
            raise ContractValidationError(f"{source_field}.nodes contains duplicate IDs")
        node_indices[token] = index

    parsed_edges: list[
        tuple[
            int,
            int,
            str | None,
            str | None,
            str | None,
            float | None,
            int,
        ]
    ] = []
    edge_selectors_by_source: dict[int, list[SemanticSourceSelector]] = {}
    for index, raw_edge in enumerate(raw_edges):
        field = f"{source_field}.edges[{index}]"
        label: str | None = None
        edge_type: str | None = None
        normalized_edge_type: str | None = None
        edge_confidence = confidence
        if isinstance(raw_edge, Mapping):
            source_value = raw_edge.get("source", raw_edge.get("from"))
            target_value = raw_edge.get("target", raw_edge.get("to"))
            label_value = raw_edge.get("label")
            type_value = raw_edge.get("type", raw_edge.get("kind"))
            if label_value is not None:
                if not isinstance(label_value, str) or not label_value.strip():
                    raise ContractValidationError(f"{field}.label must be non-empty text")
                label = label_value.strip()
            if type_value is not None:
                if not isinstance(type_value, str) or not type_value.strip():
                    raise ContractValidationError(f"{field}.type must be non-empty text")
                edge_type = type_value.strip()
                normalized_edge_type = normalize_semantic_edge_type(edge_type)
            edge_confidence = _optional_probability(raw_edge.get("confidence", confidence), f"{field}.confidence")
        else:
            edge = _sequence(raw_edge, field)
            if len(edge) not in {2, 3}:
                raise ContractValidationError(f"{field} must contain source, target, and optional label")
            source_value, target_value = edge[0], edge[1]
            if len(edge) == 3:
                if not isinstance(edge[2], str) or not edge[2].strip():
                    raise ContractValidationError(f"{field}[2] must be a non-empty label")
                label = edge[2].strip()
        source_key = _node_reference(source_value, f"{field}.source")
        target_key = _node_reference(target_value, f"{field}.target")
        source_index = node_indices.get(_node_token(source_key))
        target_index = node_indices.get(_node_token(target_key))
        if source_index is None or target_index is None:
            raise ContractValidationError(f"{field} references an undeclared node")
        if source_index == target_index:
            raise ContractValidationError(f"{field} is a self edge")
        edge_selector = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector=f"/edges/{index}",
            metadata={"raw_source_path": f"{source_field}.edges[{index}]"},
        )
        edge_selectors_by_source.setdefault(source_index, []).append(edge_selector)
        parsed_edges.append(
            (
                source_index,
                target_index,
                label,
                edge_type,
                normalized_edge_type,
                edge_confidence,
                index,
            )
        )

    elements: list[SemanticElement] = []
    for index, (key, content, selector_path, role) in enumerate(node_records):
        selectors = [
            SemanticSourceSelector(
                representation_id=representation.representation_id,
                native_selector=selector_path,
                metadata={"raw_source_path": f"{source_field}{selector_path}"},
            ),
            *edge_selectors_by_source.get(index, ()),
        ]
        if len(selectors) > MAX_SEMANTIC_SOURCE_SELECTORS_PER_ELEMENT:
            raise ContractValidationError(
                f"{source_field} node {key!r} exceeds outgoing-edge selector limit"
            )
        elements.append(
            SemanticElement(
                local_key=f"{type(key).__name__}:{key}",
                kind=(
                    SemanticElementKind.FLOWCHART_ELEMENT
                    if flowchart
                    else SemanticElementKind.DIAGRAM_ELEMENT
                ),
                source_selectors=tuple(selectors),
                content=json_ready(content),  # type: ignore[arg-type]
                role=role,
                confidence=confidence,
                metadata={"native_node_id": key},
            )
        )
    relations: list[SemanticRelation] = []
    for (
        source_index,
        target_index,
        label,
        edge_type,
        normalized_edge_type,
        edge_confidence,
        edge_index,
    ) in parsed_edges:
        source = elements[source_index]
        target = elements[target_index]
        edge_selector = next(
            selector
            for selector in source.source_selectors
            if selector.native_selector == f"/edges/{edge_index}"
        )
        kind = SemanticRelationKind.DIAGRAM_CONNECTS_TO
        if flowchart:
            normalized = normalized_edge_type or ""
            if normalized in {"true", "yes", "branch_true", "condition_true"}:
                kind = SemanticRelationKind.FLOW_BRANCH_TRUE
            elif normalized in {"false", "no", "branch_false", "condition_false"}:
                kind = SemanticRelationKind.FLOW_BRANCH_FALSE
            else:
                kind = SemanticRelationKind.FLOW_NEXT
        relations.append(
            SemanticRelation(
                kind=kind,
                source_element_id=source.element_id,
                target_element_id=target.element_id,
                source_selector_ids=(edge_selector.selector_id,),
                label=label,
                confidence=edge_confidence,
                metadata={
                    "native_edge_index": edge_index,
                    **({"native_edge_type": edge_type} if edge_type is not None else {}),
                    **(
                        {SEMANTIC_NORMALIZED_EDGE_TYPE_KEY: normalized_edge_type}
                        if normalized_edge_type is not None
                        else {}
                    ),
                },
            )
        )
    return SemanticSubstructure(
        kind=(SemanticStructureKind.FLOWCHART if flowchart else SemanticStructureKind.DIAGRAM),
        elements=tuple(elements),
        relations=tuple(relations),
        producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
        availability=SemanticStructureAvailability.COMPLETE,
        source_field_paths=(source_field,),
        metadata={
            "normalizer": NORMALIZER_NAME,
            "parser": parser_name,
            "nodes": "parser_native",
            "edges": "parser_native_only",
        },
    )


def _mathml_preflight(xml_text: str, *, field: str) -> bytes:
    encoded = xml_text.encode("utf-8")
    if len(encoded) > MAX_MATHML_BYTES:
        raise ContractValidationError(f"{field} exceeds {MAX_MATHML_BYTES} UTF-8 bytes")
    if _DTD_OR_ENTITY.search(encoded):
        raise ContractValidationError(f"{field} must not contain DTD or entity declarations")
    count = 0
    depth = 0
    parser = expat.ParserCreate()

    def start(_name: str, _attrs: Mapping[str, str]) -> None:
        nonlocal count, depth
        count += 1
        depth += 1
        if count > MAX_MATHML_ELEMENTS:
            raise ContractValidationError(f"{field} exceeds {MAX_MATHML_ELEMENTS} XML elements")
        if depth > MAX_MATHML_DEPTH:
            raise ContractValidationError(f"{field} exceeds XML depth {MAX_MATHML_DEPTH}")

    def end(_name: str) -> None:
        nonlocal depth
        depth -= 1

    def reject(*_args: object) -> None:
        raise ContractValidationError(f"{field} contains a forbidden XML declaration")

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = lambda *_args: 0
    try:
        parser.Parse(encoded, True)
    except ContractValidationError:
        raise
    except expat.ExpatError as exc:
        raise ContractValidationError(f"{field} is invalid XML: {exc}") from exc
    if count < 1:
        raise ContractValidationError(f"{field} contains no XML elements")
    return encoded


def preflight_mathml(xml_text: object, *, field: str) -> None:
    """Reject oversized or hostile MathML before representation hashing."""

    if not isinstance(xml_text, str) or not xml_text.strip():
        raise ContractValidationError(f"{field} must be non-empty MathML text")
    _mathml_preflight(xml_text, field=field)


def _local_xml_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def mathml_formula_substructure(
    mathml: str,
    representation: Representation,
    *,
    source_field: str,
    parser_name: str,
    confidence: float | None,
) -> SemanticSubstructure:
    """Deterministically expose a bounded MathML XML tree as formula structure."""

    if not isinstance(mathml, str) or not mathml.strip():
        raise ContractValidationError(f"{source_field} must be non-empty MathML text")
    encoded = _mathml_preflight(mathml, field=source_field)
    try:
        root = ElementTree.fromstring(encoded)
    except ElementTree.ParseError as exc:
        raise ContractValidationError(f"{source_field} is invalid MathML: {exc}") from exc
    if _local_xml_name(root.tag).casefold() != "math":
        raise ContractValidationError(f"{source_field} root element must be math")
    if not list(root) and not (root.text is not None and root.text.strip()):
        raise ContractValidationError(f"{source_field} contains no formula content")

    records: list[tuple[ElementTree.Element, str, int | None, int]] = []
    stack: list[tuple[ElementTree.Element, str, int | None, int]] = [(root, "/math[1]", None, 0)]
    while stack:
        element, path, parent_index, child_index = stack.pop()
        record_index = len(records)
        records.append((element, path, parent_index, child_index))
        counts: dict[str, int] = {}
        children: list[tuple[ElementTree.Element, str, int, int]] = []
        for index, child in enumerate(list(element)):
            tag = _local_xml_name(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            children.append((child, f"{path}/{tag}[{counts[tag]}]", record_index, index))
        stack.extend(reversed(children))

    elements: list[SemanticElement] = []
    selectors: list[SemanticSourceSelector] = []
    for index, (element, path, _parent_index, _child_index) in enumerate(records):
        selector = SemanticSourceSelector(
            representation_id=representation.representation_id,
            native_selector=path,
            metadata={"raw_source_path": source_field, "selector_syntax": "xpath_subset"},
        )
        selectors.append(selector)
        content: dict[str, JsonValue] = {"tag": _local_xml_name(element.tag)}
        if element.attrib:
            content["attributes"] = dict(sorted(element.attrib.items()))
        if element.text is not None and element.text.strip():
            content["text"] = element.text.strip()
        if element.tail is not None and element.tail.strip():
            content["tail"] = element.tail.strip()
        elements.append(
            SemanticElement(
                local_key=f"xml:{index}",
                kind=SemanticElementKind.FORMULA_SYMBOL,
                source_selectors=(selector,),
                content=content,
                role=_local_xml_name(element.tag),
                confidence=confidence,
                metadata={"xml_document_order": index},
            )
        )

    relations: list[SemanticRelation] = []
    children_by_parent: dict[int, list[int]] = {}
    for child_index, (_element, _path, parent_index, _position) in enumerate(records):
        if parent_index is None:
            continue
        children_by_parent.setdefault(parent_index, []).append(child_index)
        relations.append(
            SemanticRelation(
                kind=SemanticRelationKind.FORMULA_ARGUMENT_OF,
                source_element_id=elements[child_index].element_id,
                target_element_id=elements[parent_index].element_id,
                source_selector_ids=(selectors[child_index].selector_id, selectors[parent_index].selector_id),
                metadata={"derived_from": "mathml_parent_child"},
            )
        )
    for parent_index, child_indices in children_by_parent.items():
        tag = _local_xml_name(records[parent_index][0].tag).casefold()
        specialized: list[tuple[int, int, SemanticRelationKind]] = []
        if tag == "msup" and len(child_indices) >= 2:
            specialized.append((child_indices[1], child_indices[0], SemanticRelationKind.FORMULA_SUPERSCRIPT_OF))
        elif tag == "msub" and len(child_indices) >= 2:
            specialized.append((child_indices[1], child_indices[0], SemanticRelationKind.FORMULA_SUBSCRIPT_OF))
        elif tag == "msubsup" and len(child_indices) >= 3:
            specialized.extend(
                (
                    (child_indices[1], child_indices[0], SemanticRelationKind.FORMULA_SUBSCRIPT_OF),
                    (child_indices[2], child_indices[0], SemanticRelationKind.FORMULA_SUPERSCRIPT_OF),
                )
            )
        for source_index, target_index, kind in specialized:
            relations.append(
                SemanticRelation(
                    kind=kind,
                    source_element_id=elements[source_index].element_id,
                    target_element_id=elements[target_index].element_id,
                    source_selector_ids=(selectors[source_index].selector_id, selectors[target_index].selector_id),
                    metadata={"derived_from": f"mathml_{tag}_child_positions"},
                )
            )
    if len(relations) > MAX_SEMANTIC_RELATIONS_PER_STRUCTURE:
        raise ContractValidationError(f"{source_field} exceeds formula relation limit")
    return SemanticSubstructure(
        kind=SemanticStructureKind.FORMULA,
        elements=tuple(elements),
        relations=tuple(relations),
        producer_kind=SemanticProducerKind.DETERMINISTIC_DERIVED,
        availability=SemanticStructureAvailability.COMPLETE,
        source_field_paths=(source_field,),
        metadata={
            "normalizer": NORMALIZER_NAME,
            "parser": parser_name,
            "source_format": "mathml_xml",
            "xml_limits": {
                "bytes": MAX_MATHML_BYTES,
                "elements": MAX_MATHML_ELEMENTS,
                "depth": MAX_MATHML_DEPTH,
            },
        },
    )
