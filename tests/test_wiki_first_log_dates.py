"""wiki_first_log_dates：log.md 首次出现日期解析（新增/维护标签数据源）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_link_graph as glg


def _node(rel_path: str) -> dict[str, object]:
    return {"id": rel_path, "label": rel_path, "type": "concept"}


class WikiFirstLogDatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing_paths = [
            cand
            for cand in ("wiki/concepts/sim2real.md", "wiki/tasks/locomotion.md")
            if (glg.REPO_ROOT / cand).is_file()
        ]
        if len(self.existing_paths) < 2:
            self.skipTest("仓库中可用 wiki 节点不足以执行测试")
        self.nodes = [_node(rel) for rel in self.existing_paths]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._fake_log_path = Path(self._tmpdir.name) / "log.md"

    def _first_dates_from_text(self, raw_log: str) -> dict[str, str]:
        self._fake_log_path.write_text(raw_log, encoding="utf-8")
        with mock.patch.object(glg, "LOG_MD_PATH", self._fake_log_path):
            return glg.wiki_first_log_dates(self.nodes)

    def test_first_appearance_uses_oldest_log_day(self) -> None:
        rel = self.existing_paths[0]
        out = self._first_dates_from_text(
            f"## [2026-05-28] ingest\n- 接入 {rel}\n\n## [2026-05-01] ingest\n- 接入 {rel}\n"
        )
        self.assertEqual(out[rel], "2026-05-01")

    def test_ignores_lint_and_structural_glob_patterns(self) -> None:
        rel, other = self.existing_paths[0], self.existing_paths[1]
        for entry_header in (
            "## [2026-05-27] lint | scripts/lint_wiki.py — paper 元数据\n- 针对 `wiki/entities/paper-*.md` 做检查\n",
            "## [2026-05-01] structural | Top-100 论文枢纽补齐「结论」\n- 扫描 `wiki/entities/paper-*.md` 与 `wiki/queries/*`\n",
        ):
            out = self._first_dates_from_text(
                f"## [2026-05-28] ingest\n- 新建 {rel}\n\n{entry_header}"
            )
            self.assertEqual(out.get(rel), "2026-05-28")
            self.assertNotIn(other, out)

    def test_wiki_node_action(self) -> None:
        cases = [
            (
                "wiki/tasks/locomotion.md",
                "2026-05-01",
                {"wiki/tasks/locomotion.md": "2026-05-01"},
                None,
                "added",
            ),
            (
                "wiki/tasks/locomotion.md",
                "2026-05-28",
                {"wiki/tasks/locomotion.md": "2026-05-01"},
                None,
                "maintained",
            ),
            (
                "wiki/tasks/unknown.md",
                "2026-05-28",
                {"wiki/tasks/locomotion.md": "2026-05-01"},
                None,
                None,
            ),
            (
                "wiki/concepts/sim2real.md",
                "2026-04-10",
                {},
                {"wiki/concepts/sim2real.md": "2026-04-10"},
                "added",
            ),
            (
                "wiki/concepts/sim2real.md",
                "2026-05-01",
                {},
                {"wiki/concepts/sim2real.md": "2026-04-10"},
                "maintained",
            ),
        ]
        for rel, day, first_dates, git_dates, expected in cases:
            self.assertEqual(glg._wiki_node_action(rel, day, first_dates, git_dates), expected)

    def test_git_added_date_is_birth_day_and_shallow_fallback(self) -> None:
        """有 git 加入日时以其为新建日；浅克隆不可信时回退日志首日。"""
        rel = self.existing_paths[0]
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-05-28", {rel: "2026-05-28"}, {rel: "2026-04-01"}),
            "maintained",
        )
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-04-01", {rel: "2026-05-28"}, {rel: "2026-04-01"}),
            "added",
        )
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-05-28", {rel: "2026-05-01"}, {rel: "2026-05-28"}),
            "added",
        )
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-05-29", {rel: "2026-05-01"}, {rel: "2026-05-28"}),
            "maintained",
        )

        with (
            mock.patch.object(glg, "_git_is_shallow", return_value=True),
            mock.patch.object(glg, "_WIKI_GIT_ADDED_DATES_CACHE", None),
        ):
            self.assertEqual(glg.wiki_git_added_dates(force_refresh=True), {})
        self.assertEqual(glg._wiki_node_action(rel, "2026-05-28", {rel: "2026-05-28"}, {}), "added")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
