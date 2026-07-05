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
        self.existing_paths: list[str] = []
        for candidate in [
            "wiki/concepts/sim2real.md",
            "wiki/tasks/locomotion.md",
        ]:
            if (glg.REPO_ROOT / candidate).is_file():
                self.existing_paths.append(candidate)
        if len(self.existing_paths) < 2:
            self.skipTest("仓库中可用 wiki 节点不足以执行测试")
        self.nodes = [_node(rel) for rel in self.existing_paths]
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._fake_log_path = Path(self._tmpdir.name) / "log.md"

    def _write_log(self, blocks: list[tuple[str, list[str]]]) -> None:
        chunks: list[str] = []
        for log_date, paths in blocks:
            lines = [f"## [{log_date}] ingest"]
            for p in paths:
                lines.append(f"- 接入 {p}")
            chunks.append("\n".join(lines) + "\n")
        self._fake_log_path.write_text("\n".join(chunks), encoding="utf-8")

    def test_first_appearance_uses_oldest_log_day(self) -> None:
        rel = self.existing_paths[0]
        self._write_log(
            [
                ("2026-05-28", [rel]),
                ("2026-05-01", [rel]),
            ]
        )
        with mock.patch.object(glg, "LOG_MD_PATH", self._fake_log_path):
            out = glg.wiki_first_log_dates(self.nodes)
        self.assertEqual(out[rel], "2026-05-01")

    def test_ignores_lint_glob_patterns(self) -> None:
        rel = self.existing_paths[0]
        self._write_log(
            [
                ("2026-05-28", [rel]),
                (
                    "2026-05-27",
                    [],
                ),
            ]
        )
        lint_block = (
            "## [2026-05-27] lint | scripts/lint_wiki.py — paper 元数据\n"
            "- 针对 `wiki/entities/paper-*.md` 做检查\n"
        )
        self._fake_log_path.write_text(
            self._fake_log_path.read_text() + lint_block,
            encoding="utf-8",
        )
        with mock.patch.object(glg, "LOG_MD_PATH", self._fake_log_path):
            out = glg.wiki_first_log_dates(self.nodes)
        self.assertEqual(out.get(rel), "2026-05-28")

    def test_wiki_node_action(self) -> None:
        first_dates = {"wiki/tasks/locomotion.md": "2026-05-01"}
        self.assertEqual(
            glg._wiki_node_action("wiki/tasks/locomotion.md", "2026-05-01", first_dates),
            "added",
        )
        self.assertEqual(
            glg._wiki_node_action("wiki/tasks/locomotion.md", "2026-05-28", first_dates),
            "maintained",
        )
        self.assertIsNone(glg._wiki_node_action("wiki/tasks/unknown.md", "2026-05-28", first_dates))

    def test_git_fallback_when_log_has_no_first_date(self) -> None:
        git_dates = {"wiki/concepts/sim2real.md": "2026-04-10"}
        rel = "wiki/concepts/sim2real.md"
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-04-10", {}, git_dates),
            "added",
        )
        self.assertEqual(
            glg._wiki_node_action(rel, "2026-05-01", {}, git_dates),
            "maintained",
        )

    def test_log_first_date_takes_priority_over_git(self) -> None:
        rel = self.existing_paths[0]
        self.assertEqual(
            glg._wiki_node_action(
                rel,
                "2026-05-28",
                {rel: "2026-05-28"},
                {rel: "2026-04-01"},
            ),
            "added",
        )
        self.assertEqual(
            glg._wiki_node_action(
                rel,
                "2026-05-28",
                {rel: "2026-05-01"},
                {rel: "2026-05-28"},
            ),
            "maintained",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
