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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
