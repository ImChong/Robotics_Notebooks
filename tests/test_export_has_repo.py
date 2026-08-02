"""export_minimal：关联 sources/repos/ 的页面应写出 has_repo。"""

from __future__ import annotations

import unittest
from pathlib import Path

from export_minimal import build_item, page_has_repo_source

ROOT = Path(__file__).resolve().parents[1]


class ExportHasRepoTests(unittest.TestCase):
    def test_page_has_repo_source_detects_relative_repo_link(self) -> None:
        self.assertTrue(page_has_repo_source("见 [PBHC](../../sources/repos/pbhc.md) 仓库"))
        self.assertFalse(page_has_repo_source("仅有 [wiki](../concepts/sim2real.md)"))

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
