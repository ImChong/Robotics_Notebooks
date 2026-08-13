"""git 驱动的 wiki 活动时间线 / 最新新增节点。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_link_graph as glg


def _node(rel_path: str, label: str = "", type_: str = "concept") -> dict[str, object]:
    return {"id": rel_path, "label": label or rel_path, "type": type_}


class ParseWikiGitNameStatusTest(unittest.TestCase):
    def test_added_last_and_daily_touches(self) -> None:
        rel = "wiki/concepts/sim2real.md"
        other = "wiki/tasks/locomotion.md"
        log_text = (
            f"{glg._GIT_LOG_BOUNDARY}2026-05-28\n"
            f"M\t{rel}\n"
            f"{glg._GIT_LOG_BOUNDARY}2026-05-01\n"
            f"A\t{rel}\n"
            f"A\t{other}\n"
        )
        history = glg._parse_wiki_git_name_status(log_text, [rel, other])
        self.assertEqual(history.added_dates[rel], "2026-05-01")
        self.assertEqual(history.added_dates[other], "2026-05-01")
        self.assertEqual(history.last_dates[rel], "2026-05-28")
        self.assertEqual(history.last_dates[other], "2026-05-01")
        self.assertEqual(history.touches_by_date["2026-05-28"], [rel])
        self.assertEqual(history.touches_by_date["2026-05-01"], [rel, other])

    def test_rename_counts_as_touch_of_current_path(self) -> None:
        cur = "wiki/concepts/sim2real.md"
        log_text = (
            f"{glg._GIT_LOG_BOUNDARY}2026-05-28\n"
            f"R100\twiki/concepts/old-name.md\t{cur}\n"
            f"{glg._GIT_LOG_BOUNDARY}2026-05-01\n"
            "A\twiki/concepts/old-name.md\n"
        )
        history = glg._parse_wiki_git_name_status(log_text, [cur])
        self.assertEqual(history.added_dates[cur], "2026-05-01")
        self.assertEqual(history.last_dates[cur], "2026-05-28")
        self.assertEqual(history.touches_by_date["2026-05-28"], [cur])
        self.assertEqual(history.touches_by_date["2026-05-01"], [cur])


class WikiActivityFromGitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing_paths: list[str] = []
        for candidate in [
            "wiki/concepts/sim2real.md",
            "wiki/concepts/system-identification.md",
            "wiki/tasks/locomotion.md",
        ]:
            if (glg.REPO_ROOT / candidate).is_file():
                self.existing_paths.append(candidate)
        if len(self.existing_paths) < 2:
            self.skipTest("仓库中可用 wiki 节点不足以执行测试")
        self.nodes = [_node(rel) for rel in self.existing_paths]

    def test_added_vs_maintained_from_git_touches(self) -> None:
        rel = self.existing_paths[0]
        other = self.existing_paths[1]
        history = glg.WikiGitHistory(
            added_dates={rel: "2026-05-27", other: "2026-05-27"},
            last_dates={rel: "2026-05-28", other: "2026-05-27"},
            touches_by_date={
                "2026-05-28": [rel],
                "2026-05-27": [rel, other],
            },
        )
        out = glg.wiki_activity_from_git(self.nodes, history=history)
        by_date = {d["date"]: d for d in out}
        self.assertEqual(by_date["2026-05-27"]["added_count"], 2)
        self.assertEqual(by_date["2026-05-28"]["maintained_count"], 1)
        self.assertEqual(by_date["2026-05-28"]["nodes"][0]["action"], "maintained")

    def test_wiki_activity_falls_back_to_log_when_git_empty(self) -> None:
        rel = self.existing_paths[0]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake_log = Path(tmp.name) / "log.md"
        fake_log.write_text(f"## [2026-05-28] ingest\n- 接入 {rel}\n", encoding="utf-8")
        empty = glg.WikiGitHistory()
        with (
            mock.patch.object(glg, "collect_wiki_git_history", return_value=empty),
            mock.patch.object(glg, "LOG_MD_PATH", fake_log),
            mock.patch.object(glg, "wiki_git_added_dates", return_value={rel: "2026-05-28"}),
        ):
            out = glg.wiki_activity(self.nodes)
        self.assertEqual(out[0]["date"], "2026-05-28")
        self.assertEqual(out[0]["nodes"][0]["path"], rel)


class LatestWikiNodesFromGitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing_paths: list[str] = []
        for candidate in [
            "wiki/concepts/sim2real.md",
            "wiki/concepts/system-identification.md",
            "wiki/tasks/locomotion.md",
        ]:
            if (glg.REPO_ROOT / candidate).is_file():
                self.existing_paths.append(candidate)
        if len(self.existing_paths) < 2:
            self.skipTest("仓库中可用 wiki 节点不足以执行测试")
        self.nodes = [_node(rel) for rel in self.existing_paths]

    def test_prefers_added_and_skips_maintenance_only_days(self) -> None:
        added = self.existing_paths[0]
        maintained = self.existing_paths[1]
        history = glg.WikiGitHistory(
            added_dates={added: "2026-05-27", maintained: "2026-05-01"},
            last_dates={added: "2026-05-28", maintained: "2026-05-28"},
            touches_by_date={
                "2026-05-28": [maintained],
                "2026-05-27": [added],
            },
        )
        out = glg.latest_wiki_nodes_from_git(
            self.nodes, max_items=5, window_days=30, history=history
        )
        self.assertEqual([item["path"] for item in out], [added])
        self.assertEqual(out[0]["source"], "git")
        self.assertEqual(out[0]["action"], "added")
        self.assertEqual(out[0]["recency"], "2026-05-27")

    def test_window_excludes_old_added_nodes(self) -> None:
        recent = self.existing_paths[0]
        old = self.existing_paths[1]
        history = glg.WikiGitHistory(
            added_dates={recent: "2026-05-28", old: "2025-01-01"},
            last_dates={recent: "2026-05-28", old: "2025-01-01"},
            touches_by_date={
                "2026-05-28": [recent],
                "2025-01-01": [old],
            },
        )
        out = glg.latest_wiki_nodes_from_git(
            self.nodes, max_items=10, window_days=30, history=history
        )
        self.assertEqual([item["path"] for item in out], [recent])

    def test_max_items_caps_added_list(self) -> None:
        history = glg.WikiGitHistory(
            added_dates={p: "2026-05-28" for p in self.existing_paths},
            last_dates={p: "2026-05-28" for p in self.existing_paths},
            touches_by_date={"2026-05-28": list(self.existing_paths)},
        )
        out = glg.latest_wiki_nodes_from_git(
            self.nodes, max_items=1, window_days=30, history=history
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source"], "git")

    def test_roadmap_added_paths_appear(self) -> None:
        roadmap_rel = "roadmap/depth-real2sim.md"
        if not (glg.REPO_ROOT / roadmap_rel).is_file():
            self.skipTest("缺少 roadmap/depth-real2sim.md")
        wiki_rel = self.existing_paths[0]
        nodes = self.nodes + [_node(roadmap_rel, "Real2Sim 纵深", "roadmap_page")]
        history = glg.WikiGitHistory(
            added_dates={roadmap_rel: "2026-07-23", wiki_rel: "2026-07-23"},
            last_dates={roadmap_rel: "2026-07-23", wiki_rel: "2026-07-23"},
            touches_by_date={"2026-07-23": [roadmap_rel, wiki_rel]},
        )
        out = glg.latest_wiki_nodes_from_git(nodes, max_items=5, window_days=30, history=history)
        paths = [item["path"] for item in out]
        self.assertIn(roadmap_rel, paths)
        roadmap_item = next(item for item in out if item["path"] == roadmap_rel)
        self.assertEqual(roadmap_item["type"], "roadmap_page")
        self.assertEqual(roadmap_item["detail_id"], "roadmap-depth-real2sim")


class WikiLastActivityDatesTest(unittest.TestCase):
    def test_prefers_git_last_dates(self) -> None:
        rel = "wiki/concepts/sim2real.md"
        if not (glg.REPO_ROOT / rel).is_file():
            self.skipTest("缺少 sim2real 页")
        nodes = [_node(rel)]
        history = glg.WikiGitHistory(
            added_dates={rel: "2026-05-01"},
            last_dates={rel: "2026-05-28"},
            touches_by_date={"2026-05-28": [rel]},
        )
        with mock.patch.object(glg, "collect_wiki_git_history", return_value=history):
            out = glg.wiki_last_activity_dates(nodes)
        self.assertEqual(out[rel], "2026-05-28")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
