"""路线的叙事顺序与图谱邻接边顺序独立。"""

import generate_link_graph as glg


def test_route_order_uses_first_body_mention_and_markdown_semantics(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    roadmap = tmp_path / "roadmap"
    wiki.mkdir()
    roadmap.mkdir()
    for name in ["a", "z", "code", "comment", "related", "ref", "space name"]:
        (wiki / f"{name}.md").write_text("# Node")
    page = roadmap / "test.md"
    page.write_text("# Route")
    monkeypatch.setattr(glg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(glg, "GRAPH_CONTENT_DIRS", (wiki, roadmap))
    content = """---
related:
  - ../wiki/related.md
---
[Z](../wiki/z.md#stage)
[A](../wiki/a.md)
[Z again](../wiki/z.md)
`[code](../wiki/code.md)`
```md
[code](../wiki/code.md)
```
~~~md
[code](../wiki/code.md)
~~~
<!-- [hidden](../wiki/comment.md) -->
![image](../wiki/comment.md)
[Reference][later]

[later]: ../wiki/ref.md
[Space](../wiki/space%20name.md "title")
[Self](test.md)
[Missing](../wiki/missing.md)
[External](https://example.org/wiki/a.md)
[Outside](../other.md)
"""
    (tmp_path / "other.md").write_text("# Outside")
    assert glg.extract_route_order(content, page) == [
        "roadmap/test.md",
        "wiki/z.md",
        "wiki/a.md",
        "wiki/ref.md",
        "wiki/space name.md",
    ]


def test_actual_safe_control_route_starts_at_hub_and_follows_prose():
    page = glg.ROADMAP_DIR / "depth-safe-control.md"
    order = glg.extract_route_order(page.read_text(), page)
    assert len(order) == len(set(order))
    assert order[:3] == [
        "roadmap/depth-safe-control.md",
        "roadmap/motion-control.md",
        "wiki/formalizations/lyapunov.md",
    ]
