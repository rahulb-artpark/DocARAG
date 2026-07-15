from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These modules define the stable application/domain boundary consumed by
# BookRAG. Vendor SDKs and infrastructure clients belong behind adapters and
# must never become transitive requirements of this layer.
BOUNDARY_PATHS = (
    *sorted((PROJECT_ROOT / "docarag" / "domain").glob("*.py")),
    *sorted((PROJECT_ROOT / "docarag" / "application").glob("*.py")),
    PROJECT_ROOT / "docarag" / "ports" / "parser.py",
    PROJECT_ROOT / "docarag" / "routing.py",
    PROJECT_ROOT / "docarag" / "reconciliation.py",
    PROJECT_ROOT / "docarag" / "retrieval_candidates.py",
    PROJECT_ROOT / "docarag" / "bookrag_projection.py",
    PROJECT_ROOT / "docarag" / "evidence_graph_builder.py",
    PROJECT_ROOT / "docarag" / "evidence_graph_checkpoint.py",
    PROJECT_ROOT / "docarag" / "legacy_graph_projection.py",
    PROJECT_ROOT / "docarag" / "parsing_orchestrator.py",
    PROJECT_ROOT / "docarag" / "composition.py",
    PROJECT_ROOT / "docarag" / "presets.py",
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "Core",
        "docling",
        "fastapi",
        "fitz",
        "magic_pdf",
        "mineru",
        "networkx",
        "numpy",
        "paddle",
        "paddleocr",
        "psycopg",
        "qdrant_client",
        "torch",
        "transformers",
        "vllm",
    }
)


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_domain_and_application_contracts_have_no_vendor_or_legacy_imports(self):
        violations: list[str] = []
        for path in BOUNDARY_PATHS:
            self.assertTrue(path.is_file(), f"missing boundary module: {path}")
            forbidden = sorted(imported_roots(path) & FORBIDDEN_IMPORT_ROOTS)
            if forbidden:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}: {', '.join(forbidden)}"
                )
        self.assertEqual(
            [],
            violations,
            "stable DocARAG boundaries imported infrastructure/vendor code",
        )


if __name__ == "__main__":
    unittest.main()
