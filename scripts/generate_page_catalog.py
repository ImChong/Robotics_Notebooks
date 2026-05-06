#!/usr/bin/env python3
"""
generate_page_catalog.py
从 wiki/、roadmap/、tech-map/ 目录自动收集所有 .md 文件，
生成符合 index.md Page Catalog 格式的 markdown，输出到 stdout。
用法：python generate_page_catalog.py >> ../index.md
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
WIKI = ROOT / "wiki"
ROADMAP = ROOT / "roadmap"
TECHMAP = ROOT / "tech-map"

SECTION_TEMPLATE = """### {section_name}（{count}页）

"""

PAGE_TEMPLATE = """- [{title}]({path}) — {summary} `📅{date}` `{page_type}`
"""

def extract_frontmatter_date(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
        # 优先找 frontmatter date:
        m = re.search(r"^date:\s*(\d{4}-\d{2}-\d{2})", content, re.MULTILINE)
        if m:
            return m.group(1)
        # 其次找 📅 格式
        m = re.search(r"📅(\d{4}-\d{2}-\d{2})", content)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"

def strip_frontmatter(content: str) -> str:
    """去除 YAML frontmatter（--- ... ---）。"""
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content
    return content[end + 4:].lstrip()

def extract_first_sentence(content: str) -> str:
    """取第一段非空、非标题的正文句子作为 summary。"""
    content = strip_frontmatter(content)
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">") or line == "---":
            continue
        # 截断到 100 字符
        s = line[:100].strip("[]()#*`|")
        if s:
            return s
    return "—"

def get_type(path: Path, relative: Path) -> tuple:
    rel_str = str(relative)
    if "entities/" in rel_str:
        return "[entity_page]", "entity"
    if "references/" in rel_str:
        return "[reference_page]", "reference"
    if "roadmap/" in rel_str or "learning-paths/" in rel_str:
        return "[roadmap_page]", "roadmap"
    if "tech-map/" in rel_str and rel_str != "tech-map/":
        return "[tech_map_node]", "tech-map"
    if "concepts/" in rel_str:
        return "[wiki_page]", "concept"
    if "methods/" in rel_str:
        return "[method_page]", "method"
    if "tasks/" in rel_str:
        return "[task_page]", "task"
    if "formalizations/" in rel_str:
        return "[formalization_page]", "formalization"
    if "comparisons/" in rel_str:
        return "[comparison_page]", "comparison"
    if "overview/" in rel_str:
        return "[overview_page]", "overview"
    return "[wiki_page]", "unknown"

def collect_pages(dir_path: Path, base_path: Path) -> list:
    pages = []
    if not dir_path.exists():
        return pages
    for md in sorted(dir_path.rglob("*.md")):
        rel = md.relative_to(base_path)
        content = md.read_text(encoding="utf-8", errors="ignore")

        title_m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else rel.stem

        summary = extract_first_sentence(content)
        date = extract_frontmatter_date(md)
        page_type, _ = get_type(md, rel)

        pages.append({
            "title": title,
            "path": str(rel).replace("\\", "/"),
            "summary": summary,
            "date": date,
            "page_type": page_type,
        })
    return pages

def main():
    sections = [
        ("Entities（实体页）", WIKI / "entities", WIKI),
        ("Wiki Concepts（概念页）", WIKI / "concepts", WIKI),
        ("Wiki Methods（方法页）", WIKI / "methods", WIKI),
        ("Wiki Tasks（任务页）", WIKI / "tasks", WIKI),
        ("Wiki Formalizations（形式化基础）", WIKI / "formalizations", WIKI),
        ("Wiki Comparisons（对比页）", WIKI / "comparisons", WIKI),
        ("Wiki Overview（总览）", WIKI / "overview", WIKI),
        ("Roadmaps（路线页）", ROADMAP, ROADMAP),
        ("Tech-map Nodes（技术栈节点）", TECHMAP, TECHMAP),
        ("References（参考资料页）", ROOT / "references", ROOT),
    ]

    for sec_name, dir_path, base_path in sections:
        pages = collect_pages(dir_path, base_path)
        if not pages:
            continue
        print(f"\n### {sec_name}")
        print()
        for p in pages:
            print(f"- [{p['title']}]({p['path']}) — {p['summary']} `📅{p['date']}` `{p['page_type']}`")

if __name__ == "__main__":
    main()
