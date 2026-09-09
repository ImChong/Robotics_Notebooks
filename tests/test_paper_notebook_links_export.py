import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from export_minimal import (
    attach_paper_notebook_links,
    build_item,
    build_site_data,
    collect_paper_notebook_links,
    write_page_exports,
)

ROOT = Path(__file__).resolve().parents[1]
ZEROWBC = ROOT / "wiki" / "entities" / "paper-notebook-zerowbc.md"
VMP = ROOT / "wiki" / "entities" / "paper-notebook-vmp.md"
PLANNED = (
    ROOT
    / "wiki"
    / "entities"
    / "paper-notebook-a-21-dof-humanoid-dexterous-hand-with-hybrid-sma.md"
)


class PaperNotebookLinksExportTests(unittest.TestCase):
    def test_collect_paper_notebook_links_prefers_notebook_html(self) -> None:
        item = build_item(ZEROWBC)
        links = collect_paper_notebook_links(item)
        self.assertEqual(len(links), 1)
        self.assertTrue(links[0]["url"].endswith(".html"))
        self.assertIn("ZeroWBC", links[0]["label"])
        self.assertNotIn("progress.json", links[0]["url"])

    def test_collect_paper_notebook_links_ignores_github_progress_json(self) -> None:
        item = build_item(VMP)
        links = collect_paper_notebook_links(item)
        self.assertEqual(links, [])

    def test_planned_stub_without_notebook_html_has_no_links(self) -> None:
        item = build_item(PLANNED)
        links = collect_paper_notebook_links(item)
        self.assertEqual(links, [])

    def test_attach_paper_notebook_links_writes_index_field(self) -> None:
        item = build_item(ZEROWBC)
        attach_paper_notebook_links([item])
        self.assertIn("paper_notebook_links", item)
        self.assertEqual(len(item["paper_notebook_links"]), 1)

    def catalog_entry_for(self, item: dict) -> dict:
        # 单页样本不含 schema/page-aliases.json 的重定向目标，别名校验与本用例无关
        with mock.patch("export_minimal.load_page_aliases", return_value={}):
            payload = build_site_data([item])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_page_exports(payload, out)
            catalog = json.loads((out / "site-catalog-v1.json").read_text(encoding="utf-8"))
        return catalog["pages"]["detail_pages"][item["id"]]

    def test_body_free_catalog_keeps_every_graph_sidebar_field(self) -> None:
        """图谱侧栏改读无正文目录（编号 3），标题/标签/关系/来源必须仍在目录里。"""
        item = build_item(ZEROWBC)
        attach_paper_notebook_links([item])
        entry = self.catalog_entry_for(item)
        self.assertNotIn("content_markdown", entry)
        for field in ("title", "tags", "related", "source_links", "paper_notebook_links"):
            self.assertIn(field, entry)
        self.assertEqual(entry["paper_notebook_links"], item["paper_notebook_links"])

    def test_links_found_only_in_the_body_still_ship_without_the_body(self) -> None:
        """正文里的行内笔记链接由导出期扫描，无正文的消费端不能因此丢链接。"""
        url = (
            "https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/"
            "03_High_Impact_Selection/Demo/Demo.html"
        )
        item = {
            "id": "wiki-concepts-demo",
            "title": "Demo",
            "type": "wiki_page",
            "path": "wiki/concepts/demo.md",
            "summary": "",
            "content_markdown": f"参考 [机器人论文阅读笔记：Demo]({url})",
            "tags": [],
            "related": [],
            "source_links": [],
        }
        attach_paper_notebook_links([item], {})
        self.assertEqual([link["url"] for link in item["paper_notebook_links"]], [url])
        entry = self.catalog_entry_for(item)
        self.assertNotIn("content_markdown", entry)
        self.assertEqual([link["url"] for link in entry["paper_notebook_links"]], [url])


if __name__ == "__main__":
    unittest.main()
