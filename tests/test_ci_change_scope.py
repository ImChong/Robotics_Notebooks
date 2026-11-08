"""Regressions for CI ingest scope classification (issue #3507)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ci_change_scope as scope  # noqa: E402


class ChangeScopeTests(unittest.TestCase):
    def test_name_status_parses_deletions_and_renames(self) -> None:
        raw = b"A\0wiki/new.md\0D\0wiki/gone.md\0R100\0wiki/old.md\0wiki/renamed.md\0"
        self.assertEqual(
            scope.parse_name_status(raw),
            [
                ("A", ("wiki/new.md",)),
                ("D", ("wiki/gone.md",)),
                ("R100", ("wiki/old.md", "wiki/renamed.md")),
            ],
        )

    def test_ingest_only_for_markdown_content(self) -> None:
        result = scope.classify(
            [
                ("A", ("wiki/entities/paper-demo.md",)),
                ("M", ("sources/papers/demo.md",)),
                ("A", ("log.d/2026-11-08.md",)),
            ]
        )
        self.assertEqual(result["mode"], "ingest")
        self.assertEqual(result["changed_wiki"], ["wiki/entities/paper-demo.md"])

    def test_code_or_non_markdown_requires_full(self) -> None:
        for path in (
            "scripts/export_minimal.py",
            "tests/test_content_sync.py",
            "schema/topics.json",
            "docs/main.js",
            ".github/workflows/tests.yml",
            "wiki/image.png",
            "README.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(scope.classify([("M", (path,))])["mode"], "full")

    def test_rename_has_both_paths_and_removed_old_path(self) -> None:
        result = scope.classify(
            [("R100", ("wiki/entities/old.md", "wiki/entities/new.md"))]
        )
        self.assertEqual(result["mode"], "ingest")
        self.assertEqual(result["removed"], ["wiki/entities/old.md"])
        self.assertEqual(len(result["changed_wiki"]), 2)

    def test_empty_or_unparseable_diff_falls_back_to_full(self) -> None:
        self.assertEqual(scope.classify([])["mode"], "full")
        with self.assertRaises(ValueError):
            scope.parse_name_status(b"A\0wiki/file.md")
        with mock.patch.object(
            scope.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "git diff"),
        ):
            self.assertEqual(scope.detect("origin/main")["mode"], "full")


if __name__ == "__main__":
    unittest.main()
