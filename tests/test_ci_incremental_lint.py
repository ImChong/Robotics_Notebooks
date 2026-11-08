"""Regression coverage for changed/affected wiki content selection."""

from __future__ import annotations

import ast
import inspect
import sys
import tempfile
import textwrap
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


class IncrementalContradictionTests(unittest.TestCase):
    FACTS = {"demo": {"terms": ["PPO"], "pos_claims": ["PPO.*高"], "neg_claims": ["PPO.*低"]}}

    def test_only_facts_a_changed_page_takes_one_side_on_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wiki").mkdir()
            (root / "wiki" / "changed.md").write_text("PPO 样本效率低", encoding="utf-8")
            (root / "wiki" / "both.md").write_text("SAC 快，但 SAC 也慢", encoding="utf-8")
            facts = {
                **self.FACTS,
                "mention_only": {"terms": ["PPO"], "pos_claims": ["PPO.*快"], "neg_claims": []},
                "both_sides": {"terms": ["SAC"], "pos_claims": ["快"], "neg_claims": ["慢"]},
            }
            scope = {"changed_wiki": ["wiki/changed.md", "wiki/both.md", "wiki/deleted.md"]}
            with (
                mock.patch.object(incremental, "ROOT", root),
                mock.patch.object(incremental.wiki, "load_canonical_facts", return_value=facts),
            ):
                self.assertEqual(incremental.contradiction_fact_ids(scope), {"demo"})

    def test_selected_fact_is_compared_against_unchanged_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unchanged = Path(tmp) / "unchanged.md"
            changed = Path(tmp) / "changed.md"
            unchanged.write_text("PPO 样本效率高", encoding="utf-8")
            changed.write_text("PPO 样本效率低", encoding="utf-8")
            results = incremental.wiki._empty_results()
            with mock.patch.object(
                incremental.wiki, "load_canonical_facts", return_value=self.FACTS
            ):
                incremental.wiki._check_contradictions([unchanged, changed], results, set())
                self.assertEqual(results["contradictions"], [])
                incremental.wiki._check_contradictions([unchanged, changed], results, {"demo"})
            self.assertEqual(len(results["contradictions"]), 1)


def _called_checks(func) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(func)))):
        if isinstance(node, ast.Call):
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name.startswith("_check_"):
                names.add(name)
    return names


class FullLintParityTests(unittest.TestCase):
    def test_every_full_lint_check_is_run_or_explicitly_skipped(self) -> None:
        full = _called_checks(incremental.wiki.lint)
        ran = _called_checks(incremental.lint_changed)
        skipped = set(incremental.FULL_LINT_ONLY_CHECKS)
        self.assertEqual(full - ran, skipped, "update FULL_LINT_ONLY_CHECKS or lint_changed")
        self.assertFalse(ran & skipped)


if __name__ == "__main__":
    unittest.main()
