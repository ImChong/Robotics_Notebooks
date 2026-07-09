import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP_HTML = ROOT / "docs" / "roadmap.html"
MAIN_JS = ROOT / "docs" / "main.js"


class RoadmapPageTests(unittest.TestCase):
    def test_roadmap_page_contains_required_mount_points(self):
        content = ROADMAP_HTML.read_text(encoding="utf-8")
        required_ids = [
            'id="roadmapTitle"',
            'id="roadmapSummary"',
            'id="roadmapMeta"',
            'id="roadmapMetaUpdated"',
            'id="roadmapMetaStages"',
            'id="roadmapFlowMermaidRoot"',
            'id="roadmapContent"',
            'id="roadmapContentSourceLink"',
            'id="roadmapTocList"',
        ]
        for marker in required_ids:
            self.assertIn(marker, content)

    def test_main_js_contains_roadmap_page_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function renderRoadmapPage",
            "function renderRoadmapMetaPanel",
            "document.getElementById('roadmapTitle')",
            "roadmap.html?id=",
            "roadmap_pages",
            "renderRoadmapMarkdownBody",
            "roadmapContentSourceLink",
            "findRoadmapStageEntryAnchor",
            "bindSelftestMermaidRerender",
            "roadmap-stage-entry-embed",
            "renderDetailToc(tocEl, headings, roadmapMarkdownContext, { maxLevel: 2 })",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_search_results_link_roadmap_pages_to_roadmap_html(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("isRoadmapPageId(resultId, null, item)", content)
        self.assertIn("roadmapHref(resultId)", content)

    def test_timeline_shows_stage_labels_for_depth_roadmaps(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function parseRoadmapTimelineNodeLabel", content)
        self.assertIn("Stage\\s+(\\d+)", content)

    def test_main_roadmap_knowledge_map_includes_depth_branches(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function collectDepthBranchRoadmaps", content)
        self.assertIn("roadmap-kmap-stage-depth", content)
        self.assertIn("#depth-optional-index", content)


if __name__ == "__main__":
    unittest.main()
