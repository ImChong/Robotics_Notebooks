"""V23: latest_wiki_nodes 时间窗口与最大项数可配置（CLI / 环境变量）单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_link_graph as glg


def _node(rel_path: str, label: str = "", type_: str = "concept") -> dict[str, object]:
    return {"id": rel_path, "label": label or rel_path, "type": type_}


def _wiki_path_for(rel: str) -> Path:
    return glg.REPO_ROOT / rel


class LatestWikiNodesWindowTest(unittest.TestCase):
    """覆盖 V23 新增 max_items / window_days 行为。"""

    def setUp(self) -> None:
        # 选两条仓库中真实存在的 wiki 路径，避免与文件存在性校验冲突。
        self.existing_paths: list[str] = []
        for candidate in [
            "wiki/concepts/sim2real.md",
            "wiki/concepts/system-identification.md",
            "wiki/tasks/locomotion.md",
            "wiki/methods/reinforcement-learning.md",
        ]:
            if _wiki_path_for(candidate).is_file():
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

    def _patched_log(self):
        return mock.patch.object(glg, "LOG_MD_PATH", self._fake_log_path)

    def test_max_items_caps_single_day(self) -> None:
        self._write_log([("2026-05-28", self.existing_paths)])
        with self._patched_log():
            out = glg.latest_wiki_nodes_from_log(self.nodes, max_items=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["recency"], "2026-05-28")
        self.assertEqual(out[0]["source"], "log.md")

    def test_window_spans_multiple_days(self) -> None:
        self._write_log(
            [
                ("2026-05-28", self.existing_paths[:1]),
                ("2026-05-27", self.existing_paths[1:2]),
            ]
        )
        with self._patched_log():
            out = glg.latest_wiki_nodes_from_log(self.nodes, max_items=5, window_days=30)
        self.assertEqual(len(out), 2)
        recencies = [item["recency"] for item in out]
        self.assertEqual(recencies, ["2026-05-28", "2026-05-27"])

    def test_window_excludes_outside_range(self) -> None:
        old_date = "2025-01-01"  # 远早于 30 天窗口
        self._write_log(
            [
                ("2026-05-28", self.existing_paths[:1]),
                (old_date, self.existing_paths[1:2]),
            ]
        )
        with self._patched_log():
            out = glg.latest_wiki_nodes_from_log(self.nodes, max_items=10, window_days=30)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["recency"], "2026-05-28")

    def test_roadmap_depth_paths_appear_in_latest_nodes(self) -> None:
        """纵深路线 roadmap/depth-*.md 也应进入首页最新知识节点清单。"""
        roadmap_rel = "roadmap/depth-real2sim.md"
        if not _wiki_path_for(roadmap_rel).is_file():
            self.skipTest("缺少 roadmap/depth-real2sim.md")
        nodes = self.nodes + [_node(roadmap_rel, "Real2Sim 纵深", "roadmap_page")]
        self._write_log(
            [
                ("2026-07-23", [roadmap_rel, self.existing_paths[0]]),
            ]
        )
        with self._patched_log():
            out = glg.latest_wiki_nodes_from_log(nodes, max_items=5, window_days=30)
        paths = [item["path"] for item in out]
        self.assertIn(roadmap_rel, paths)
        roadmap_item = next(item for item in out if item["path"] == roadmap_rel)
        self.assertEqual(roadmap_item["type"], "roadmap_page")
        self.assertEqual(roadmap_item["detail_id"], "roadmap-depth-real2sim")
        self.assertEqual(roadmap_item["recency"], "2026-07-23")

    def test_zero_max_items_returns_empty(self) -> None:
        self._write_log([("2026-05-28", self.existing_paths)])
        with self._patched_log():
            out = glg.latest_wiki_nodes_from_log(self.nodes, max_items=0)
        self.assertEqual(out, [])


class ResolveLatestNodesMaxTest(unittest.TestCase):
    """覆盖 CLI / 环境变量 / 默认值的优先级与 clamp 逻辑。"""

    def test_default_when_no_cli_no_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(glg.LATEST_NODES_ENV_VAR, None)
            self.assertEqual(glg.resolve_latest_nodes_max(None), glg.LATEST_NODES_DEFAULT)

    def test_cli_overrides_env(self) -> None:
        with mock.patch.dict(os.environ, {glg.LATEST_NODES_ENV_VAR: "20"}):
            self.assertEqual(glg.resolve_latest_nodes_max(5), 5)

    def test_env_used_when_cli_absent(self) -> None:
        with mock.patch.dict(os.environ, {glg.LATEST_NODES_ENV_VAR: "15"}):
            self.assertEqual(glg.resolve_latest_nodes_max(None), 15)

    def test_invalid_env_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, {glg.LATEST_NODES_ENV_VAR: "not-a-number"}):
            self.assertEqual(glg.resolve_latest_nodes_max(None), glg.LATEST_NODES_DEFAULT)

    def test_cap_is_enforced(self) -> None:
        self.assertEqual(glg.resolve_latest_nodes_max(999), glg.LATEST_NODES_CAP)

    def test_floor_is_one(self) -> None:
        self.assertEqual(glg.resolve_latest_nodes_max(0), 1)
        self.assertEqual(glg.resolve_latest_nodes_max(-3), 1)

    def test_build_graph_data_skips_self_loop_edges(self) -> None:
        nodes, edges = glg._build_graph_data()
        self_loops = [e for e in edges if e["source"] == e["target"]]
        self.assertEqual(self_loops, [])


class PaperHubStatsTest(unittest.TestCase):
    """top_paper_hubs：仅论文节点（type=entity/method 且带 paper 标签）按互链度排序。"""

    def test_build_graph_data_marks_paper_nodes(self) -> None:
        nodes, _ = glg._build_graph_data()
        by_id = {n["id"]: n for n in nodes}
        bfm = by_id.get("wiki/entities/paper-behavior-foundation-model-humanoid.md")
        self.assertIsNotNone(bfm)
        self.assertTrue(bfm.get("_is_paper"))
        # 升格为深度拆解页的论文（type=method + paper 标签）也是论文节点
        sonic = by_id.get("wiki/methods/sonic-motion-tracking.md")
        self.assertIsNotNone(sonic)
        self.assertTrue(sonic.get("_is_paper"))
        # 概念页不是论文节点
        sim2real = by_id.get("wiki/concepts/sim2real.md")
        self.assertIsNotNone(sim2real)
        self.assertFalse(sim2real.get("_is_paper"))
        # 无 paper 标签的 method 页不是论文节点
        rl = by_id.get("wiki/methods/reinforcement-learning.md")
        self.assertIsNotNone(rl)
        self.assertFalse(rl.get("_is_paper"))

    def test_top_paper_hubs_are_papers_sorted_desc(self) -> None:
        nodes, edges = glg._build_graph_data()
        communities, community_meta = glg.assign_communities(nodes, edges)
        paper_ids = {n["id"] for n in nodes if n.get("_is_paper")}
        stats = glg._compute_graph_stats(nodes, edges, communities, community_meta)
        hubs = stats["top_paper_hubs"]
        self.assertTrue(hubs)
        self.assertLessEqual(len(hubs), 10)
        degrees = [h["degree"] for h in hubs]
        self.assertEqual(degrees, sorted(degrees, reverse=True))
        for hub in hubs:
            self.assertIn(hub["id"], paper_ids)

    def test_hub_entries_carry_detail_id_and_type_for_site_links(self) -> None:
        """首页互链枢纽直接消费 hub 条目构造站内链接，须带 detail_id / type。"""
        nodes, edges = glg._build_graph_data()
        communities, community_meta = glg.assign_communities(nodes, edges)
        stats = glg._compute_graph_stats(nodes, edges, communities, community_meta)
        for hub in stats["top_hubs"] + stats["top_paper_hubs"]:
            self.assertEqual(hub["detail_id"], glg._wiki_node_detail_id(hub["id"]))
            self.assertNotIn("/", hub["detail_id"])
            self.assertIn("type", hub)

    def test_hub_entries_include_community_and_optional_repo(self) -> None:
        """互链枢纽行与最新知识节点对齐：带 community_label，开源页带 has_repo。"""
        nodes, edges = glg._build_graph_data()
        communities, community_meta = glg.assign_communities(nodes, edges)
        stats = glg._compute_graph_stats(nodes, edges, communities, community_meta)
        hubs = stats["top_hubs"]
        self.assertTrue(hubs)
        labeled = [h for h in hubs if h.get("community_label")]
        self.assertTrue(labeled, "Top hubs should usually carry community_label")
        for hub in hubs:
            if hub.get("has_repo"):
                self.assertIs(hub["has_repo"], True)
            else:
                self.assertNotIn("has_repo", hub)

    def test_hub_rankings_cover_all_nodes_sorted_desc(self) -> None:
        """完整榜单：全站 / 论文按互链度降序，供 hubs.html 消费。"""
        nodes, edges = glg._build_graph_data()
        communities, community_meta = glg.assign_communities(nodes, edges)
        stats = glg._compute_graph_stats(nodes, edges, communities, community_meta)
        rankings = stats["_hub_rankings"]
        self.assertEqual(len(rankings["all"]), len(nodes))
        paper_ids = {n["id"] for n in nodes if n.get("_is_paper")}
        self.assertEqual(len(rankings["paper"]), len(paper_ids))
        all_degrees = [h["degree"] for h in rankings["all"]]
        self.assertEqual(all_degrees, sorted(all_degrees, reverse=True))
        paper_degrees = [h["degree"] for h in rankings["paper"]]
        self.assertEqual(paper_degrees, sorted(paper_degrees, reverse=True))
        self.assertEqual(
            [h["detail_id"] for h in rankings["all"][:10]],
            [h["detail_id"] for h in stats["top_hubs"]],
        )
        self.assertEqual(
            [h["detail_id"] for h in rankings["paper"][:10]],
            [h["detail_id"] for h in stats["top_paper_hubs"]],
        )


class RoadmapGraphNodesTest(unittest.TestCase):
    """roadmap/ 页面应作为 roadmap_page 节点进入 link-graph。"""

    def test_build_graph_data_includes_roadmap_nodes(self) -> None:
        nodes, edges = glg._build_graph_data()
        by_id = {n["id"]: n for n in nodes}
        motion = by_id.get("roadmap/motion-control.md")
        self.assertIsNotNone(motion)
        self.assertEqual(motion.get("type"), "roadmap_page")
        self.assertEqual(motion.get("detail_id"), "roadmap-motion-control")

    def test_roadmap_links_to_wiki_create_edges(self) -> None:
        nodes, edges = glg._build_graph_data()
        node_ids = {n["id"] for n in nodes}
        motion_edges = [e for e in edges if e["source"] == "roadmap/motion-control.md"]
        self.assertTrue(motion_edges)
        for edge in motion_edges:
            self.assertIn(edge["target"], node_ids)

    def test_wiki_links_to_roadmap_create_edges(self) -> None:
        _, edges = glg._build_graph_data()
        wiki_to_roadmap = [
            e
            for e in edges
            if e["target"] == "roadmap/motion-control.md" and e["source"].startswith("wiki/")
        ]
        self.assertTrue(wiki_to_roadmap)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
