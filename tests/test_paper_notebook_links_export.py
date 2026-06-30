import unittest
from pathlib import Path

from export_minimal import (
    attach_paper_notebook_links,
    build_item,
    collect_paper_notebook_links,
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


if __name__ == "__main__":
    unittest.main()
