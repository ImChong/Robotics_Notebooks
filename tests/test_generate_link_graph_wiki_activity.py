"""wiki_activity_from_log：首页热力图按日活动数据（全量日志、同日去重）单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_link_graph as glg


def _node(rel_path: str, label: str = "", type_: str = "concept") -> dict[str, object]:
    return {"id": rel_path, "label": label or rel_path, "type": type_}


def _wiki_path_for(rel: str) -> Path:
    return glg.REPO_ROOT / rel


class WikiActivityFromLogTest(unittest.TestCase):
    """覆盖按日聚合、同日多块合并去重、升序输出与节点瘦身字段。"""

    def setUp(self) -> None:
        # 选仓库中真实存在的 wiki 路径，避免与文件存在性校验冲突。
        self.existing_paths: list[str] = []
        for candidate in [
            "wiki/concepts/sim2real.md",
            "wiki/concepts/system-identification.md",
            "wiki/tasks/locomotion.md",
            "wiki/methods/reinforcement-learning.md",
        ]:
            if _wiki_path_for(candidate).is_file():
                self.existing_paths.append(candidate)
        if len(self.existing_paths) < 3:
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

    def _patched_log(self):
        return mock.patch.object(glg, "LOG_MD_PATH", self._fake_log_path)

    def test_days_sorted_ascending_across_full_history(self) -> None:
        old_date = "2025-01-01"  # 远早于 latest_wiki_nodes 的 30 天窗口，仍应保留
        self._write_log(
            [
                ("2026-05-28", self.existing_paths[:1]),
                (old_date, self.existing_paths[1:2]),
            ]
        )
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        self.assertEqual([d["date"] for d in out], [old_date, "2026-05-28"])
        self.assertEqual([d["count"] for d in out], [1, 1])

    def test_same_day_blocks_merge_and_dedupe(self) -> None:
        self._write_log(
            [
                ("2026-05-28", self.existing_paths[:2]),
                ("2026-05-28", self.existing_paths[1:3]),  # 与上一块重叠 1 条
            ]
        )
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["count"], 3)
        detail_ids = [n["detail_id"] for n in out[0]["nodes"]]
        self.assertEqual(len(detail_ids), len(set(detail_ids)))

    def test_same_node_can_repeat_across_days(self) -> None:
        self._write_log(
            [
                ("2026-05-28", self.existing_paths[:1]),
                ("2026-05-27", self.existing_paths[:1]),
            ]
        )
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        self.assertEqual([d["count"] for d in out], [1, 1])

    def test_node_payload_is_slim(self) -> None:
        self._write_log([("2026-05-28", self.existing_paths[:1])])
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        node = out[0]["nodes"][0]
        allowed = {"detail_id", "label", "type", "action", "has_repo", "community_label"}
        self.assertTrue(set(node).issubset(allowed))

    def test_has_repo_flag_when_wiki_links_sources_repos(self) -> None:
        repo_rel = "wiki/entities/caveman.md"
        if not _wiki_path_for(repo_rel).is_file():
            self.skipTest("caveman 实体页不存在")
        nodes = [_node(repo_rel, type_="entity")]
        nodes[0]["community"] = "community-0"
        community_labels = {"community-0": "编码代理（Coding Agents） 社区"}
        self._write_log([("2026-05-28", [repo_rel])])
        with self._patched_log():
            out = glg.wiki_activity_from_log(nodes, community_labels=community_labels)
        node = out[0]["nodes"][0]
        self.assertTrue(node.get("has_repo"))
        self.assertEqual(node.get("community_label"), "编码代理（Coding Agents） 社区")

    def test_community_label_omitted_for_other_bucket(self) -> None:
        rel = self.existing_paths[0]
        nodes = [_node(rel)]
        nodes[0]["community"] = glg.OTHER_COMMUNITY_ID
        community_labels = {glg.OTHER_COMMUNITY_ID: glg.OTHER_COMMUNITY_LABEL}
        self._write_log([("2026-05-28", [rel])])
        with self._patched_log():
            out = glg.wiki_activity_from_log(nodes, community_labels=community_labels)
        self.assertNotIn("community_label", out[0]["nodes"][0])

    def test_has_repo_omitted_when_no_repo_source(self) -> None:
        rel = next(
            (
                p
                for p in self.existing_paths
                if not glg.wiki_has_repo_source(_wiki_path_for(p).read_text(encoding="utf-8"))
            ),
            None,
        )
        if not rel:
            self.skipTest("无不含 sources/repos 引用的测试用 wiki 页")
        self._write_log([("2026-05-28", [rel])])
        nodes = [_node(rel)]
        with self._patched_log():
            out = glg.wiki_activity_from_log(nodes)
        self.assertNotIn("has_repo", out[0]["nodes"][0])

    def test_action_added_vs_maintained(self) -> None:
        rel = self.existing_paths[0]
        other = self.existing_paths[1]
        self._write_log(
            [
                ("2026-05-28", [rel]),
                ("2026-05-27", [rel, other]),
            ]
        )
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        by_date = {d["date"]: d for d in out}
        self.assertEqual(by_date["2026-05-27"]["added_count"], 2)
        self.assertEqual(by_date["2026-05-27"]["nodes"][0]["action"], "added")
        self.assertEqual(by_date["2026-05-28"]["maintained_count"], 1)
        self.assertEqual(by_date["2026-05-28"]["nodes"][0]["action"], "maintained")

    def test_nodes_export_all_entries_and_count_matches(self) -> None:
        many = sorted(
            str(p.relative_to(glg.REPO_ROOT)).replace("\\", "/") for p in glg.WIKI_DIR.rglob("*.md")
        )[:35]
        if len(many) < 10:
            self.skipTest("仓库 wiki 页面数不足以覆盖多日节点导出")
        nodes = [_node(rel) for rel in many]
        self._write_log([("2026-05-28", many)])
        with self._patched_log():
            out = glg.wiki_activity_from_log(nodes)
        self.assertEqual(out[0]["count"], len(many))
        self.assertEqual(len(out[0]["nodes"]), len(many))

    def test_unknown_paths_and_empty_days_are_dropped(self) -> None:
        self._write_log(
            [
                ("2026-05-28", ["wiki/concepts/definitely-not-a-page.md"]),
                ("2026-05-27", self.existing_paths[:1]),
            ]
        )
        with self._patched_log():
            out = glg.wiki_activity_from_log(self.nodes)
        self.assertEqual([d["date"] for d in out], ["2026-05-27"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
