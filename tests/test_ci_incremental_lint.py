"""Regression coverage for changed/affected wiki content selection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ci_incremental_lint as incremental  # noqa: E402


class IncrementalAffectedPagesTests(unittest.TestCase):
    def test_old_link_target_is_checked_after_link_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki_dir = root / "wiki" / "entities"
            wiki_dir.mkdir(parents=True)
            changed = wiki_dir / "changed.md"
            target = wiki_dir / "target.md"
            changed.write_text("No links now", encoding="utf-8")
            target.write_text("Existing page", encoding="utf-8")
            scope = {
                "changed_wiki": ["wiki/entities/changed.md"],
                "changed_sources": [],
                "removed": [],
                "base": "origin/main",
                "paths": ["wiki/entities/changed.md"],
            }
            with (
                mock.patch.object(incremental, "ROOT", root),
                mock.patch.object(incremental, "old_links", return_value={target.resolve()}),
            ):
                selected = incremental.affected_pages(scope, [changed, target], {})
            self.assertEqual(selected, [changed.resolve(), target.resolve()])

    def test_deleted_wiki_is_checked_via_inbound_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki_dir = root / "wiki"
            wiki_dir.mkdir()
            inbound = wiki_dir / "inbound.md"
            inbound.write_text("[old](old.md)", encoding="utf-8")
            scope = {
                "changed_wiki": ["wiki/old.md"],
                "changed_sources": [],
                "removed": ["wiki/old.md"],
                "base": "origin/main",
                "paths": ["wiki/old.md"],
            }
            with (
                mock.patch.object(incremental, "ROOT", root),
                mock.patch.object(incremental, "old_links", return_value=set()),
            ):
                selected = incremental.affected_pages(
                    scope,
                    [inbound],
                    {inbound.resolve(): ["wiki/old.md"]},
                )
            self.assertEqual(selected, [inbound.resolve()])

    def test_source_changes_recheck_original_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki_dir = root / "wiki"
            wiki_dir.mkdir()
            page = wiki_dir / "page.md"
            page.write_text(
                "---\noriginal: ../sources/papers/demo.md\n---\ncontent",
                encoding="utf-8",
            )
            scope = {
                "changed_wiki": [],
                "changed_sources": ["sources/papers/demo.md"],
                "removed": [],
                "base": "origin/main",
                "paths": ["sources/papers/demo.md"],
            }
            with mock.patch.object(incremental, "ROOT", root):
                selected = incremental.affected_pages(scope, [page], {})
            self.assertEqual(selected, [page.resolve()])


if __name__ == "__main__":
    unittest.main()
