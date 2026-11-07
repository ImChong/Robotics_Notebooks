"""export_minimal：关联 sources/repos/ 或 frontmatter code: 是代码仓库的页面应写出 has_repo。"""

from __future__ import annotations

import unittest
from pathlib import Path

from export_minimal import build_item, page_has_repo_source

ROOT = Path(__file__).resolve().parents[1]


class ExportHasRepoTests(unittest.TestCase):
    def test_page_has_repo_source_detects_relative_repo_link(self) -> None:
        self.assertTrue(page_has_repo_source("见 [PBHC](../../sources/repos/pbhc.md) 仓库"))
        self.assertFalse(page_has_repo_source("仅有 [wiki](../concepts/sim2real.md)"))

    def test_page_has_repo_source_detects_frontmatter_code_repo(self) -> None:
        self.assertTrue(
            page_has_repo_source(
                "---\ncode: https://github.com/NVlabs/GR00T-WholeBodyControl\n---\n正文"
            )
        )
        self.assertTrue(
            page_has_repo_source('---\ncode: "https://gitlab.com/etherlab.org/ethercat"\n---\n')
        )
        # 项目主页 / 只有组织名 / 写在正文而非 frontmatter：都不算
        self.assertFalse(page_has_repo_source("---\ncode: https://egomimic.github.io/\n---\n"))
        self.assertFalse(page_has_repo_source("---\ncode: https://github.com/NVlabs\n---\n"))
        self.assertFalse(page_has_repo_source("---\ntitle: x\n---\ncode: https://github.com/a/b"))

    def test_build_item_sets_has_repo_for_frontmatter_code_only_page(self) -> None:
        # nvidia-sonic 只在 frontmatter code: 写了官方仓库，没有 sources/repos/ 归档
        item = build_item(ROOT / "wiki" / "entities" / "nvidia-sonic.md")
        self.assertTrue(item.get("has_repo"))

    def test_build_item_sets_has_repo_for_sim2real(self) -> None:
        item = build_item(ROOT / "wiki" / "concepts" / "sim2real.md")
        self.assertTrue(item.get("has_repo"))

    def test_build_item_omits_has_repo_when_no_repo_source(self) -> None:
        # 任选一个不链 sources/repos 的短页；若不存在则跳过
        candidates = sorted((ROOT / "wiki" / "concepts").glob("*.md"))
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if not page_has_repo_source(text):
                item = build_item(path)
                self.assertNotIn("has_repo", item)
                return
        self.skipTest("no wiki/concepts page without sources/repos link")


if __name__ == "__main__":
    unittest.main()
