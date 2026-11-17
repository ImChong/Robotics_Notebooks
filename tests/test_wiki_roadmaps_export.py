import json
import unittest
from pathlib import Path

import pytest
from export_minimal import collect_paths
from utils.paths import path_to_id

ROOT = Path(__file__).resolve().parents[1]


class WikiRoadmapsExportTests(unittest.TestCase):
    def test_collect_paths_includes_hardware_design_roadmap(self):
        paths = collect_paths()
        rel_paths = {p.relative_to(ROOT).as_posix() for p in paths}
        self.assertIn("roadmap/depth-humanoid-hardware-design.md", rel_paths)

    @pytest.mark.full_only
    def test_hardware_design_roadmap_exports_to_detail_and_roadmap_pages(self):
        page_id = path_to_id(
            ROOT / "roadmap" / "depth-humanoid-hardware-design.md",
            ROOT,
        )
        self.assertEqual(page_id, "roadmap-depth-humanoid-hardware-design")

        catalog = json.loads(
            (ROOT / "exports" / "site-catalog-v1.json").read_text(encoding="utf-8")
        )
        roadmap_pages = catalog["pages"]["roadmap_pages"]
        self.assertIn(page_id, roadmap_pages)
        detail_pages = catalog["pages"]["detail_pages"]
        self.assertIn(page_id, detail_pages)
        body_path = ROOT / detail_pages[page_id]["content_url"]
        body_data = json.loads(body_path.read_text(encoding="utf-8"))
        self.assertTrue(body_data.get("content_markdown"))
