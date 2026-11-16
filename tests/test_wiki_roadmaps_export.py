import json
import unittest
from pathlib import Path

import pytest
from export_minimal import collect_paths
from utils.paths import path_to_id

ROOT = Path(__file__).resolve().parents[1]


class WikiRoadmapsExportTests(unittest.TestCase):
    def test_collect_paths_includes_wiki_roadmaps(self):
        paths = collect_paths()
        rel_paths = {p.relative_to(ROOT).as_posix() for p in paths}
        self.assertIn("wiki/roadmaps/humanoid-practitioner-entry-roadmap.md", rel_paths)

    @pytest.mark.full_only
    def test_humanoid_practitioner_entry_roadmap_exports_to_detail_pages(self):
        page_id = path_to_id(
            ROOT / "wiki" / "roadmaps" / "humanoid-practitioner-entry-roadmap.md",
            ROOT,
        )
        self.assertEqual(page_id, "wiki-roadmaps-humanoid-practitioner-entry-roadmap")

        catalog = json.loads(
            (ROOT / "exports" / "site-catalog-v1.json").read_text(encoding="utf-8")
        )
        detail_pages = catalog["pages"]["detail_pages"]
        self.assertIn(page_id, detail_pages)
        body_path = ROOT / detail_pages[page_id]["content_url"]
        body_data = json.loads(body_path.read_text(encoding="utf-8"))
        self.assertTrue(body_data.get("content_markdown"))
