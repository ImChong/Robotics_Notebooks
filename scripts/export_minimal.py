#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "exports" / "index-v1.json"
SITE_OUTPUT = ROOT / "exports" / "site-data-v1.json"
DOCS_OUTPUT = ROOT / "docs" / "exports" / "index-v1.json"
DOCS_SITE_OUTPUT = ROOT / "docs" / "exports" / "site-data-v1.json"
SITEMAP_OUTPUT = ROOT / "docs" / "sitemap.xml"
BASE_URL = "https://imchong.github.io/Robotics_Notebooks"


def build_ingest_index() -> Dict[str, str]:
    """Return {wiki_stem: sources_file_rel} for all wiki pages mentioned in sources/papers/*.md."""
    index: Dict[str, str] = {}
    sources_dir = ROOT / "sources" / "papers"
    if not sources_dir.exists():
        return index
    wiki_link_re = re.compile(r'\]\(\S*wiki/[^)]+/([^/)]+)\.md\)')
    for src in sources_dir.glob("*.md"):
        text = src.read_text(encoding="utf-8")
        for m in wiki_link_re.finditer(text):
            stem = m.group(1)
            if stem not in index:
                index[stem] = rel(src)
    return index


_INGEST_INDEX: Dict[str, str] = {}  # populated in main()

TAG_HINTS = {
    "humanoid": ["humanoid", "unitree"],
    "locomotion": ["locomotion", "walking", "gait", "biped", "legged"],
    "control": ["control", "mpc", "wbc", "tsid", "zmp", "lip"],
    "dynamics": ["dynamics", "centroidal", "floating-base", "contact"],
    "optimization": ["optimization", "optimal-control", "crocoddyl", "qp"],
    "rl": ["reinforcement-learning", "rl", "ppo", "sac", "td3"],
    "il": ["imitation-learning", "retarget", "bc", "dagger"],
    "sim2real": ["sim2real", "domain-randomization", "system-identification"],
    "tooling": ["mujoco", "isaac", "pinocchio", "crocoddyl", "framework", "simulation"],
    "hardware": ["hardware", "unitree", "actuator", "sensor"],
}

WIKI_PAGE_TYPES = {
    "concepts": "concept",
    "methods": "method",
    "tasks": "task",
    "comparisons": "comparison",
    "overview": "overview",
    "roadmaps": "roadmap",
    "formalizations": "formalization",
    "queries": "query",
}

REFERENCE_KINDS = {"papers", "repos", "benchmarks"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def clean_summary(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*]\s+", "", text)
    text = re.sub(r">\s*", "", text)
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ：:，,。") + ("。" if text and not text.endswith(("。", "!", "?", ".")) else "")


def extract_summary(text: str) -> str:
    lines = text.splitlines()

    for line in lines[1:10]:
        stripped = line.strip()
        if stripped.startswith("**") and "：" in stripped:
            return clean_summary(stripped)

    in_one_liner = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## 一句话定义":
            in_one_liner = True
            continue
        if in_one_liner:
            if not stripped:
                continue
            if stripped.startswith("## "):
                break
            return clean_summary(stripped)

    for line in lines[1:20]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return clean_summary(stripped)
    return ""


def extract_body_markdown(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def path_to_id(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    stem = path.stem
    if parts[0] == "wiki":
        if parts[1] == "entities":
            return f"entity-{stem}"
        return f"wiki-{parts[1]}-{stem}"
    if parts[0] == "roadmap":
        return f"roadmap-{stem}"
    if parts[0] == "references":
        return f"reference-{parts[1]}-{stem}"
    if parts[0] == "tech-map":
        if len(parts) >= 3 and parts[1] == "modules":
            return f"tech-node-{parts[2]}-{stem}"
        if len(parts) >= 3 and parts[1] == "research-directions":
            return f"tech-node-research-{stem}"
        return f"tech-node-{stem}"
    return slugify("-".join(parts)).removesuffix("-md")


def infer_tags(path: Path, title: str, text: str) -> List[str]:
    tags: List[str] = []
    parts = path.relative_to(ROOT).parts
    if parts[0] == "wiki":
        tags.append(WIKI_PAGE_TYPES.get(parts[1], parts[1]))
        if parts[1] == "entities":
            tags.append("entity")
    elif parts[0] == "roadmap":
        tags.append("roadmap")
    elif parts[0] == "references":
        tags.append(parts[1])
    elif parts[0] == "tech-map":
        tags.append("tech-map")

    haystack = f"{rel(path)} {title} {text[:2000]}".lower()
    for tag, hints in TAG_HINTS.items():
        if any(h in haystack for h in hints):
            tags.append(tag)

    seen = set()
    ordered = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered[:6]


def extract_section_links(text: str, current_path: Path, headings: List[str]) -> List[str]:
    lines = text.splitlines()
    collecting = False
    results: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            heading = re.sub(r"^#+\s*", "", stripped)
            if any(h in heading for h in headings):
                collecting = True
                continue
            if collecting and (stripped.startswith("## ") or stripped.startswith("### ")):
                collecting = False
        if not collecting:
            continue
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", line):
            if target.startswith(("http://", "https://", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target.endswith(".md"):
                continue
            resolved = (current_path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            results.append(path_to_id(resolved))
    return results


def collect_markdown_links(text: str, current_path: Path) -> List[str]:
    priority = extract_section_links(text, current_path, ["关联页面", "继续深挖入口", "关联任务"])
    general = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith("http://") or target.startswith("https://") or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        if not target.endswith(".md"):
            continue
        resolved = (current_path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            continue
        general.append(path_to_id(resolved))
    ordered = []
    seen = set()
    for item in priority + general:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered[:12]


def collect_external_links(text: str) -> List[str]:
    links = re.findall(r"https?://[^)\s>]+", text)
    seen = set()
    out = []
    for link in links:
        link = link.rstrip('.,')
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


def parse_roadmap_stages(text: str) -> List[Dict[str, str]]:
    stages = []
    for line in text.splitlines():
        m = re.match(r"##\s+(L\d+(?:\.\d+)?)\s+(.+)", line.strip())
        if m:
            stages.append({"id": m.group(1).lower(), "title": m.group(2).strip()})
    return stages


def build_item(path: Path) -> Dict:
    text = read_text(path)
    title = extract_title(text, path.stem)
    item = {
        "id": path_to_id(path),
        "title": title,
        "path": rel(path),
        "summary": extract_summary(text),
        "content_markdown": extract_body_markdown(text),
        "tags": infer_tags(path, title, text),
        "related": collect_markdown_links(text, path),
        "source_links": collect_external_links(text),
        "status": "active",
    }

    parts = path.relative_to(ROOT).parts
    if parts[0] == "wiki":
        if parts[1] == "entities":
            item["type"] = "entity_page"
            title_lower = title.lower()
            if any(k in title_lower for k in ["mujoco", "isaac"]):
                item["entity_kind"] = "simulator"
            elif any(k in title_lower for k in ["pinocchio", "crocoddyl"]):
                item["entity_kind"] = "library"
            elif "unitree" in title_lower:
                item["entity_kind"] = "hardware"
            else:
                item["entity_kind"] = "framework"
        else:
            item["type"] = "wiki_page"
            item["page_type"] = WIKI_PAGE_TYPES.get(parts[1], parts[1])
        # ingest coverage: check if any sources/papers/ file mentions this wiki page
        src_file = _INGEST_INDEX.get(path.stem)
        if src_file:
            item["has_ingest"] = True
            item["ingest_source"] = src_file
    elif parts[0] == "roadmap":
        item["type"] = "roadmap_page"
        item["stages"] = parse_roadmap_stages(text)
    elif parts[0] == "references":
        item["type"] = "reference_page"
        item["reference_kind"] = parts[1] if parts[1] in REFERENCE_KINDS else "unknown"
    elif parts[0] == "tech-map":
        item["type"] = "tech_map_node"
        if len(parts) >= 3 and parts[1] == "modules":
            item["node_kind"] = "module"
            item["layer"] = parts[2]
        elif path.name == "dependency-graph.md":
            item["node_kind"] = "dependency_graph"
        elif path.name == "overview.md":
            item["node_kind"] = "overview"
        else:
            item["node_kind"] = "meta"
    else:
        item["type"] = "unknown"
    return item


def collect_paths() -> List[Path]:
    patterns = [
        "wiki/concepts/*.md",
        "wiki/methods/*.md",
        "wiki/tasks/*.md",
        "wiki/comparisons/*.md",
        "wiki/overview/*.md",
        "wiki/formalizations/*.md",
        "wiki/queries/*.md",
        "wiki/entities/*.md",
        "roadmap/*.md",
        "roadmap/learning-paths/*.md",
        "references/papers/*.md",
        "references/repos/*.md",
        "references/benchmarks/*.md",
        "tech-map/overview.md",
        "tech-map/dependency-graph.md",
        "tech-map/modules/*/*.md",
        "tech-map/research-directions/*.md",
    ]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(ROOT.glob(pattern)))
    return [p for p in paths if p.name != "README.md"]


def sort_items(items: List[Dict]) -> List[Dict]:
    return sorted(items, key=lambda item: (item.get("title", ""), item.get("path", "")))


def pick_existing(ids: List[str], item_map: Dict[str, Dict]) -> List[str]:
    seen = set()
    out = []
    for item_id in ids:
        if item_id in item_map and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def build_module_page(module_id: str, title: str, summary: str, tag: str, related_modules: List[str], item_map: Dict[str, Dict]) -> Dict:
    entry_items = sort_items([
        item for item in item_map.values()
        if item.get("type") in {"wiki_page", "entity_page"} and tag in item.get("tags", [])
    ])
    references = sort_items([
        item for item in item_map.values()
        if item.get("type") == "reference_page" and tag in item.get("tags", [])
    ])
    roadmaps = sort_items([
        item for item in item_map.values()
        if item.get("type") == "roadmap_page" and tag in item.get("tags", [])
    ])
    return {
        "module_id": module_id,
        "title": title,
        "summary": summary,
        "tag": tag,
        "entry_items": [item["id"] for item in entry_items[:12]],
        "references": [item["id"] for item in references[:8]],
        "roadmaps": [item["id"] for item in roadmaps[:6]],
        "related_modules": related_modules,
    }


def build_site_data(items: List[Dict]) -> Dict:
    item_map = {item["id"]: item for item in items}

    module_specs = [
        {
            "module_id": "control",
            "title": "控制与优化主链",
            "summary": "从 LIP / ZMP、Centroidal Dynamics 到 MPC、TSID、WBC 的控制主干。",
            "tag": "control",
            "related_modules": ["locomotion", "sim2real", "rl"],
        },
        {
            "module_id": "rl",
            "title": "强化学习主链",
            "summary": "围绕 PPO、策略优化、人形运动控制训练与仿真平台的学习路径。",
            "tag": "rl",
            "related_modules": ["control", "locomotion", "sim2real"],
        },
        {
            "module_id": "il",
            "title": "模仿学习主链",
            "summary": "从行为克隆到技能迁移，连接动作数据、策略表示与人形技能学习。",
            "tag": "il",
            "related_modules": ["rl", "locomotion", "control"],
        },
        {
            "module_id": "sim2real",
            "title": "Sim2Real 主链",
            "summary": "把仿真中的控制与学习方法迁移到真实机器人系统的桥接层。",
            "tag": "sim2real",
            "related_modules": ["control", "rl", "tooling"],
        },
        {
            "module_id": "locomotion",
            "title": "Locomotion 任务主链",
            "summary": "聚焦双足 / 人形 locomotion，串联动力学、控制、学习与部署问题。",
            "tag": "locomotion",
            "related_modules": ["control", "rl", "sim2real"],
        },
        {
            "module_id": "tooling",
            "title": "工具与平台生态",
            "summary": "整理仿真器、动力学库、训练框架与硬件平台的进入方式。",
            "tag": "tooling",
            "related_modules": ["control", "rl", "sim2real"],
        },
    ]
    module_pages = {
        spec["module_id"]: build_module_page(
            spec["module_id"], spec["title"], spec["summary"], spec["tag"], spec["related_modules"], item_map
        )
        for spec in module_specs
    }

    quick_entries = pick_existing(
        [
            "roadmap-route-a-motion-control",
            "roadmap-if-goal-locomotion-rl",
            "roadmap-if-goal-imitation-learning",
        ],
        item_map,
    )
    featured_chain = pick_existing(
        [
            "wiki-concepts-lip-zmp",
            "wiki-concepts-floating-base-dynamics",
            "wiki-concepts-contact-dynamics",
            "wiki-concepts-capture-point-dcm",
            "wiki-concepts-centroidal-dynamics",
            "wiki-methods-trajectory-optimization",
            "wiki-methods-model-predictive-control",
            "wiki-concepts-tsid",
            "wiki-concepts-whole-body-control",
            "wiki-concepts-sim2real",
        ],
        item_map,
    )
    featured_modules = [spec["module_id"] for spec in module_specs if module_pages[spec["module_id"]]["entry_items"]]

    roadmap_items = sort_items([item for item in items if item.get("type") == "roadmap_page"])
    roadmap_pages = {
        item["id"]: {
            "id": item["id"],
            "title": item["title"],
            "summary": item.get("summary", ""),
            "stages": item.get("stages", []),
            "related_items": item.get("related", []),
            "source_links": item.get("source_links", []),
        }
        for item in roadmap_items
    }

    tech_nodes = sort_items([item for item in items if item.get("type") == "tech_map_node"])
    tech_map_page = {
        "graph_meta": {
            "overview_id": "tech-node-overview",
            "dependency_graph_id": "tech-node-dependency-graph",
        },
        "nodes": [
            {
                "id": item["id"],
                "title": item["title"],
                "summary": item.get("summary", ""),
                "layer": item.get("layer"),
                "node_kind": item.get("node_kind"),
                "related": item.get("related", []),
            }
            for item in tech_nodes
        ],
    }

    detail_pages = {
        item["id"]: {
            "id": item["id"],
            "title": item["title"],
            "type": item.get("type"),
            "path": item.get("path"),
            "summary": item.get("summary", ""),
            "content_markdown": item.get("content_markdown", ""),
            "tags": item.get("tags", []),
            "related": item.get("related", []),
            "source_links": item.get("source_links", []),
            "status": item.get("status", "active"),
        }
        for item in items
    }

    home_page = {
        "hero": {
            "title": "Robotics_Notebooks",
            "subtitle": "机器人技术栈知识库 / Robotics research and engineering wiki.",
        },
        "quick_entries": quick_entries,
        "featured_chain": featured_chain,
        "featured_modules": featured_modules,
    }

    return {
        "version": "v1",
        "generated_mode": "script",
        "source_index": "index-v1.json",
        "page_types": ["home_page", "module_page", "roadmap_page", "tech_map_page", "detail_page"],
        "pages": {
            "home_page": home_page,
            "module_pages": module_pages,
            "roadmap_pages": roadmap_pages,
            "tech_map_page": tech_map_page,
            "detail_pages": detail_pages,
        },
    }


def generate_sitemap(items: List[Dict], base_url: str = "https://ImChong.github.io/Robotics_Notebooks") -> str:
    """生成 sitemap.xml，包含首页、预览页和所有 detail_pages。"""
    urls = [
        {"loc": base_url + "/", "priority": "1.0", "changefreq": "weekly"},
        {"loc": base_url + "/docs/index.html", "priority": "0.9", "changefreq": "weekly"},
        {"loc": base_url + "/docs/site-data-preview.html", "priority": "0.8", "changefreq": "daily"},
        {"loc": base_url + "/docs/tech-map.html", "priority": "0.8", "changefreq": "weekly"},
    ]

    detail_pages = [item for item in items if item.get("type") in ("wiki_page", "entity_page")]
    for item in detail_pages:
        item_id = item.get("id", "")
        priority = "0.7"
        page_type = item.get("page_type", "")
        if page_type == "overview":
            priority = "0.9"
        elif page_type == "task":
            priority = "0.7"
        elif page_type == "query":
            priority = "0.6"
        urls.append({
            "loc": f"{base_url}/docs/detail.html?id={item_id}",
            "priority": priority,
            "changefreq": "monthly",
            "lastmod": None,
        })

    sitemap_lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                     '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url in urls:
        line = '  <url>'
        line += f'<loc>{url["loc"]}</loc>'
        line += f'<changefreq>{url["changefreq"]}</changefreq>'
        line += f'<priority>{url["priority"]}</priority>'
        if url.get("lastmod"):
            line += f'<lastmod>{url["lastmod"]}</lastmod>'
        line += '</url>'
        sitemap_lines.append(line)
    sitemap_lines.append('</urlset>')
    return "\n".join(sitemap_lines) + "\n"


def main() -> None:
    global _INGEST_INDEX
    _INGEST_INDEX = build_ingest_index()
    items = [build_item(p) for p in collect_paths()]
    payload = {
        "version": "v1",
        "generated_mode": "script",
        "item_count": len(items),
        "items": items,
    }
    site_payload = build_site_data(items)

    write_json(OUTPUT, payload)
    write_json(SITE_OUTPUT, site_payload)
    write_json(DOCS_OUTPUT, payload)
    write_json(DOCS_SITE_OUTPUT, site_payload)

    sitemap_content = generate_sitemap(items, BASE_URL)
    SITEMAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SITEMAP_OUTPUT.write_text(sitemap_content, encoding="utf-8")

    print(f"Wrote {OUTPUT} with {len(items)} items")
    print(f"Wrote {SITE_OUTPUT} with {len(site_payload['pages']['detail_pages'])} detail pages")
    print(f"Mirrored exports to {DOCS_OUTPUT.parent}")
    print(f"Wrote {SITEMAP_OUTPUT} with {sum(1 for i in items if i.get('type') in {'wiki_page','entity_page'})} wiki/entity URLs")


if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://ImChong.github.io/Robotics_Notebooks"

    _INGEST_INDEX = build_ingest_index()
    paths = collect_paths()
    items = sort_items([build_item(p) for p in paths])

    # index-v1.json
    index_items = []
    for item in items:
        index_items.append({
            "id": item["id"],
            "title": item["title"],
            "path": item["path"],
            "summary": item.get("summary", ""),
            "tags": item.get("tags", []),
            "type": item.get("type"),
            "page_type": item.get("page_type"),
            "entity_kind": item.get("entity_kind"),
            "reference_kind": item.get("reference_kind"),
            "content_markdown": item.get("content_markdown", ""),
            "has_ingest": item.get("has_ingest", False),
            "ingest_source": item.get("ingest_source", ""),
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_SITE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(json.dumps({"items": index_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    SITE_OUTPUT.write_text(json.dumps(build_site_data(items), ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_OUTPUT.write_text(json.dumps({"items": index_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCS_SITE_OUTPUT.write_text(json.dumps(build_site_data(items), ensure_ascii=False, indent=2), encoding="utf-8")

    # sitemap.xml
    sitemap = generate_sitemap(items, base_url)
    sitemap_path = ROOT / "docs" / "sitemap.xml"
    sitemap_path.write_text(sitemap, encoding="utf-8")

    print(f"index-v1.json: {OUTPUT} ({len(index_items)} items)")
    print(f"site-data-v1.json: {SITE_OUTPUT}")
    print(f"sitemap.xml: {sitemap_path} ({len(sitemap)} bytes)")
