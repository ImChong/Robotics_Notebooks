"""图谱更新时间与详情目录共用口径，不以文件 mtime 或活动日期替代。"""

import generate_link_graph as glg


def test_graph_updated_matches_detail_page_dates():
    pages = [
        glg.REPO_ROOT / "wiki/concepts/motion-retargeting.md",
        glg.REPO_ROOT / "roadmap/motion-control.md",
    ]
    nodes, _edges = glg._build_graph_data()
    by_path = {node["id"]: node for node in nodes}
    for page in pages:
        node = by_path[str(page.relative_to(glg.REPO_ROOT))]
        expected = glg.resolve_page_updated(page, glg.parse_frontmatter(page.read_text()))
        assert expected
        assert node["updated"] == expected
