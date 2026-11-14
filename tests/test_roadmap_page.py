import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP_HTML = ROOT / "docs" / "roadmap.html"
MAIN_JS = ROOT / "docs" / "main.js"
STYLE_CSS = ROOT / "docs" / "style.css"


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
            'id="roadmapTocSubtitle"',
            'id="detail-inline-link-tooltip"',
            'src="graph-tooltip.js"',
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
            "function buildRoadmapStageTreeBlock",
            "function latestNodeHref",
            "renderDetailMetaDateBadge(updated)",
            "roadmapMetaUpdated",
            "setupDetailInlineLinkPreview",
            "collectInlineLinkPreviewRoots",
            "currentPageIdFromLocation",
            "suppressNativeTitle",
        ]
        self.assertNotIn(
            "var tip = typeLabel + (page.summary ? ' · ' + page.summary : '');",
            content,
        )
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

    def test_roadmap_toc_subtitle_is_derived_from_headings(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function deriveRoadmapTocSubtitle", content)
        self.assertIn("function renderRoadmapTocSubtitle", content)
        self.assertIn("roadmapTocSubtitle", content)
        self.assertNotIn(
            "从 L−1 序言到 L7 出口的全程导航", ROADMAP_HTML.read_text(encoding="utf-8")
        )

    def test_main_roadmap_knowledge_map_includes_depth_branches(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function collectDepthBranchRoadmaps", content)
        self.assertIn("title: '可选路线'", content)
        self.assertIn("#depth-optional-index", content)

    def test_roadmap_knowledge_map_overview_is_top_down(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        # 全局总览：各阶段小树按阅读顺序排开，不再用分列天赋树
        self.assertIn("function drawOverview", content)
        self.assertIn("roadmap-ltree-ov-stage", content)
        self.assertNotIn("renderRoadmapTalentTree", content)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn(".roadmap-ltree-ov {", css)
        self.assertNotIn(".roadmap-ttree", css)
        # 总览节点同样带类型与健康度
        self.assertNotIn(".roadmap-ltree-ov .roadmap-ltree-node-type {\n  display: none;", css)

    def test_depth_roadmap_stages_embed_knowledge_map_nodes(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        # 纵深路线正文各 Stage 复用知识地图同阶段节点框，替换原「本阶段链接」下拉块
        self.assertIn("function embedDepthRoadmapStageNodes", content)
        self.assertIn("function roadmapLtreeNodeHtml", content)
        self.assertIn("embedDepthRoadmapStageNodes(contentEl, roadmapPage, detailPages)", content)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn(".roadmap-stage-kmap {", css)
        self.assertIn("function renderRoadmapStageTrees", content)

    def test_roadmap_trees_pan_instead_of_shrinking(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        # 节点不随列数一路缩小，放不下就拖动查看：顶部单阶段视图与正文小树共用
        self.assertIn("function syncRoadmapLtreePan", content)
        self.assertIn("function bindRoadmapLtreeMousePan", content)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn(".roadmap-ltree-pan {", css)
        self.assertIn(".roadmap-ltree-stage > .roadmap-ltree-tree {", css)

    def test_roadmap_overview_uses_floating_modal_for_mouse_and_touch(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("roadmap-ltree-map", content)
        self.assertIn("bindMapPanZoom", content)
        self.assertIn("touchstart", content)
        self.assertIn("rn-hide-inline-tooltip", content)
        self.assertIn("document.getElementById('roadmap-ltree-map')", content)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn(".roadmap-ltree-map {", css)
        self.assertIn(".roadmap-ltree-map-viewport {", css)
        self.assertIn("touch-action: pan-x pan-y;", css)
        self.assertIn(".roadmap-ltree-map-hint--touch {", css)

    def test_roadmap_knowledge_map_uses_learning_route_layout(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function renderRoadmapLearningTree", content)
        self.assertIn("function layoutRoadmapLearningStage", content)
        self.assertIn("renderRoadmapLearningTree(columns, treeEl, metaEl, detailPages)", content)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn(".roadmap-ltree {", css)
        self.assertIn(".roadmap-ltree-gate {", css)


if __name__ == "__main__":
    unittest.main()
