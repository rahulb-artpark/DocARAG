"""Protect import boundaries, checkout discovery, and saved-index compatibility."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from retrieval_engine.indexes.bm25 import BM25
from retrieval_engine.indexes.tree import DocumentTree, MetaInfo, NodeType, TreeNode
from document_services.parsing.reconciliation import RECONCILIATION_SCHEMA_VERSION
from document_services.runtime.settings import repository_root
from document_services.application.contracts import (
    DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
    DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_first_party_imports_resolve_to_source_files(self):
        """Check every local import without importing optional model dependencies."""
        missing = []
        sources = [
            *(ROOT / "retrieval_engine").rglob("*.py"),
            *(ROOT / "document_services").rglob("*.py"),
            *(ROOT / "Scripts").rglob("*.py"),
            ROOT / "main.py",
        ]
        for source in sources:
            package = ".".join(source.relative_to(ROOT).parts[:-1])
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = "." * node.level + (node.module or "")
                    modules = [
                        importlib.util.resolve_name(name, package)
                        if node.level else name
                    ]
                for module in modules:
                    if module.partition(".")[0] not in {"retrieval_engine", "document_services"}:
                        continue
                    path = ROOT.joinpath(*module.split("."))
                    if not path.with_suffix(".py").is_file() and not (
                        path / "__init__.py"
                    ).is_file():
                        missing.append(f"{source.relative_to(ROOT)}:{node.lineno}: {module}")
        self.assertEqual([], missing)

    def test_checkout_discovery_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True), patch(
                "document_services.runtime.settings.Path.cwd", return_value=Path(directory)
            ):
                self.assertEqual(ROOT, repository_root())

    def test_reconciliation_schema_is_independent_of_module_location(self):
        self.assertEqual("docarag.reconciliation.v1", RECONCILIATION_SCHEMA_VERSION)

    def test_application_schema_ids_are_independent_of_package_name(self):
        self.assertEqual(
            "docarag.application.document-extraction.v1",
            DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            "docarag.application.document-extraction.v2",
            DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
        )


class SavedIndexCompatibilityTests(unittest.TestCase):
    def test_legacy_tree_pickle_restores_canonical_classes_and_topology(self):
        tree = DocumentTree(
            meta_dict={"file_name": "document.pdf"},
            cfg=SimpleNamespace(save_path="unused"),
        )
        child = TreeNode({"content": "Preserved evidence"})
        child.type = NodeType.TEXT
        tree.root_node.add_child(child)
        tree.nodes.append(child)

        # Protocol 0 records module names explicitly. Substitute the names
        # written by previous releases to exercise real legacy deserialization.
        payload = pickle.dumps(tree, protocol=0)
        self.assertIn(b"cretrieval_engine.indexes.tree\n", payload)
        for module in (b"Core.Index.Tree", b"Core.indexes.tree"):
            with self.subTest(module=module):
                legacy_payload = payload.replace(
                    b"cretrieval_engine.indexes.tree\n", b"c" + module + b"\n"
                )
                restored = pickle.loads(legacy_payload)
                self.assertIs(type(restored), DocumentTree)
                self.assertIs(type(restored.meta_info), MetaInfo)
                self.assertIs(type(restored.nodes[1]), TreeNode)
                self.assertIs(restored.nodes[1].type, NodeType.TEXT)
                self.assertIs(restored.nodes[1].parent, restored.root_node)
                self.assertEqual("Preserved evidence", restored.nodes[1].meta_info.content)

    def test_legacy_bm25_pickle_remains_searchable(self):
        index = BM25(["alpha evidence", "beta evidence"])
        index.initialize()
        payload = pickle.dumps(index, protocol=0)
        self.assertIn(b"cretrieval_engine.indexes.bm25\n", payload)
        for module in (b"Core.utils.bm25", b"Core.indexes.bm25"):
            with self.subTest(module=module):
                restored = pickle.loads(payload.replace(
                    b"cretrieval_engine.indexes.bm25\n", b"c" + module + b"\n"
                ))
                self.assertIs(type(restored), BM25)
                self.assertEqual(index.search("alpha", 1), restored.search("alpha", 1))


if __name__ == "__main__":
    unittest.main()
