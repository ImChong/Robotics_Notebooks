import json
import unittest
from pathlib import Path

from export_minimal import collect_paths
from utils.paths import path_to_id

ROOT = Path(__file__).resolve().parents[1]


class WikiRoadmapsExportTests(unittest.TestCase):
    def test_collect_paths_includes_wiki_roadmaps(self):
        paths = collect_paths()
        rel_paths = {p.relative_to(ROOT).as_posix() for p in paths}
        self.assertIn("wiki/roadmaps/humanoid-control-roadmap.md", rel_paths)

    def test_humanoid_control_roadmap_exports_to_detail_pages(self):
        page_id = path_to_id(
            ROOT / "wiki" / "roadmaps" / "humanoid-control-roadmap.md",
            ROOT,
        )
        self.assertEqual(page_id, "wiki-roadmaps-humanoid-control-roadmap")

        site_data = json.loads((ROOT / "exports" / "site-data-v1.json").read_text(encoding="utf-8"))
        detail_pages = site_data["pages"]["detail_pages"]
        self.assertIn(page_id, detail_pages)
        self.assertTrue(detail_pages[page_id].get("content_markdown"))
