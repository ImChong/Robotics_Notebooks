#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from build_search_index import generate_search_index
from search_indexing import parse_frontmatter, strip_frontmatter
from utils.paths import path_to_id

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "exports" / "index-v1.json"
SITE_OUTPUT = ROOT / "exports" / "site-data-v1.json"
DOCS_OUTPUT = ROOT / "docs" / "exports" / "index-v1.json"
DOCS_SITE_OUTPUT = ROOT / "docs" / "exports" / "site-data-v1.json"
SITEMAP_OUTPUT = ROOT / "docs" / "sitemap.xml"
BASE_URL = "https://imchong.github.io/Robotics_Notebooks"


def build_ingest_index() -> Dict[str, str]:
    """Return {wiki_stem: sources_file_rel} for all wiki pages mentioned in sources/papers/*.md.

    Sources files are iterated in sorted order so that wiki pages with multiple
    ingest references get a deterministic single source (the alphabetically first
    one), avoiding spurious diffs in exports/index-v1.json across machines/runs.
    """
    index: Dict[str, str] = {}
    sources_dir = ROOT / "sources" / "papers"
    if not sources_dir.exists():
        return index
    wiki_link_re = re.compile(r"\]\(\S*wiki/[^)]+/([^/)]+)\.md\)")
    for src in sorted(sources_dir.glob("*.md")):
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

# Markdown 链接只有指向这些命名空间下的文件时才会被升格成 detail page id；
# 其它（典型如 sources/）属于原始资料层，按仓库分层不应该作为站内一等节点出现。
INDEXED_LINK_ROOTS = {"wiki", "roadmap", "references", "tech-map"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_indexed_link_target(resolved: Path) -> bool:
    try:
        parts = resolved.relative_to(ROOT).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in INDEXED_LINK_ROOTS


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def clean_summary(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*]\s+", "", text)
    text = re.sub(r">\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ：:，,。") + (
        "。" if text and not text.endswith(("。", "!", "?", ".")) else ""
    )


LABELED_LINE_RE = re.compile(r"^\*\*([^*]+)\*\*[：:]\s*(.*)$")


def _parse_labeled_line(stripped: str) -> tuple[str, str] | None:
    m = LABELED_LINE_RE.match(stripped)
    if not m:
        return None
    return m.group(1).strip(), m.group(2)


ROADMAP_HERO_LABEL = "首屏导读"
ROADMAP_SKIP_SUMMARY_LABELS = {ROADMAP_HERO_LABEL}


def _labeled_section_label(stripped: str) -> str | None:
    parsed = _parse_labeled_line(stripped)
    return parsed[0] if parsed else None


def extract_labeled_bullets(text: str, label: str) -> List[str]:
    lines = text.splitlines()
    for i in range(0, min(len(lines), 40)):
        stripped = lines[i].strip()
        if _labeled_section_label(stripped) != label:
            continue
        _label, _sep, tail = stripped.partition("：")
        bullets: List[str] = []
        if tail.strip():
            bullets.append(clean_summary(tail))
        for j in range(i + 1, min(len(lines), i + 30)):
            s = lines[j].strip()
            if not s:
                if bullets:
                    break
                continue
            if s.startswith("## "):
                break
            if s.startswith("- "):
                bullets.append(clean_summary(s[2:]))
            elif bullets:
                break
        return [b for b in bullets if b]
    return []


def strip_labeled_section(text: str, label: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if _labeled_section_label(stripped) == label:
            skipping = True
            continue
        if skipping:
            if not stripped:
                continue
            if stripped.startswith("## ") or (
                stripped.startswith("**") and "：" in stripped and _labeled_section_label(stripped)
            ):
                skipping = False
            else:
                continue
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def extract_roadmap_hero(text: str) -> tuple[List[str], str]:
    bullets = extract_labeled_bullets(text, ROADMAP_HERO_LABEL)
    if not bullets:
        return [], ""
    short = re.sub(r"\s+", " ", bullets[0].replace("。", ""))
    if len(short) > 120:
        short = short[:117] + "…"
    return bullets, short + ("。" if short and not short.endswith("…") else "")


def extract_summary(text: str, fm: dict[str, Any] | None = None) -> str:
    fm = fm or {}
    yaml_summary = fm.get("summary") or fm.get("description") or ""
    if isinstance(yaml_summary, str) and yaml_summary.strip():
        return clean_summary(yaml_summary.strip())

    lines = text.splitlines()

    for i in range(0, min(len(lines), 40)):
        stripped = lines[i].strip()
        parsed = _parse_labeled_line(stripped)
        if not parsed:
            continue
        label, tail = parsed
        if label in ROADMAP_SKIP_SUMMARY_LABELS:
            continue
        if tail.strip():
            return clean_summary(stripped)
        # 仅「**摘要**：」占一行、正文在后续列表时：合并列表项为全文检索/备用摘要
        bullets: List[str] = []
        for j in range(i + 1, min(len(lines), i + 30)):
            s = lines[j].strip()
            if not s:
                if bullets:
                    break
                continue
            if s.startswith("## "):
                break
            if s.startswith("- "):
                bullets.append(clean_summary(s[2:]))
            elif bullets:
                break
        if bullets:
            merged = "；".join(b for b in bullets if b)
            if len(merged) > 320:
                merged = merged[:317] + "…"
            return merged if merged.endswith(("。", "…", "!", "?", ".")) else merged + "。"
        continue

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
        if stripped and not stripped.startswith("#") and not _parse_labeled_line(stripped):
            return clean_summary(stripped)
    return ""


def extract_body_markdown(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    body = "\n".join(lines)
    if extract_labeled_bullets(body, ROADMAP_HERO_LABEL):
        body = strip_labeled_section(body, ROADMAP_HERO_LABEL)
    while body and not body.splitlines()[0].strip():
        body = "\n".join(body.splitlines()[1:])
    while body and not body.splitlines()[-1].strip():
        body = "\n".join(body.splitlines()[:-1])
    return body


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
            if not _is_indexed_link_target(resolved):
                continue
            results.append(path_to_id(resolved, ROOT))
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
        if not _is_indexed_link_target(resolved):
            continue
        general.append(path_to_id(resolved, ROOT))
    if current_path.parts and "wiki" in current_path.parts:
        wiki_dir = ROOT / "wiki"
        stem_to_path = {p.stem: p for p in wiki_dir.rglob("*.md")}
        for target in re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", text):
            linked_path = stem_to_path.get(target.strip())
            if linked_path:
                general.append(path_to_id(linked_path, ROOT))
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
        link = link.rstrip(".,")
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


def parse_roadmap_stages(text: str, current_path: Path) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    section_lines: List[str] = []

    def flush_current() -> None:
        if not current:
            return
        section_text = "\n".join(section_lines)
        current["related_items"] = collect_markdown_links(section_text, current_path)[:10]
        current["source_links"] = collect_external_links(section_text)[:5]
        stages.append(current)

    for line in text.splitlines():
        m = re.match(r"##\s+(L\d+(?:\.\d+)?)\s+(.+)", line.strip())
        if m:
            flush_current()
            current = {"id": m.group(1).lower(), "title": m.group(2).strip()}
            section_lines = []
            continue
        if current:
            section_lines.append(line)

    flush_current()
    return stages


def build_item(path: Path) -> dict[str, Any]:
    text = read_text(path)
    fm = parse_frontmatter(text)
    body_text = strip_frontmatter(text)
    title = extract_title(body_text, path.stem)
    item: dict[str, Any] = {
        "id": path_to_id(path, ROOT),
        "title": title,
        "path": rel(path),
        "summary": extract_summary(body_text, fm),
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
        item["stages"] = parse_roadmap_stages(text, path)
        hero_items, hero_short = extract_roadmap_hero(text)
        if hero_items:
            item["summary_items"] = hero_items
            item["hero_summary"] = hero_short
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
        "wiki/references/*.md",
        "wiki/roadmaps/*.md",
        "roadmap/*.md",
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


def build_module_page(
    module_id: str,
    title: str,
    summary: str,
    tag: str,
    related_modules: List[str],
    item_map: Dict[str, Dict],
) -> Dict:
    entry_items = sort_items(
        [
            item
            for item in item_map.values()
            if item.get("type") in {"wiki_page", "entity_page"} and tag in item.get("tags", [])
        ]
    )
    references = sort_items(
        [
            item
            for item in item_map.values()
            if item.get("type") == "reference_page" and tag in item.get("tags", [])
        ]
    )
    roadmaps = sort_items(
        [
            item
            for item in item_map.values()
            if item.get("type") == "roadmap_page" and tag in item.get("tags", [])
        ]
    )
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

    module_specs: list[dict[str, Any]] = [
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
            spec["module_id"],
            spec["title"],
            spec["summary"],
            spec["tag"],
            spec["related_modules"],
            item_map,
        )
        for spec in module_specs
    }

    quick_entries = pick_existing(
        [
            "roadmap-motion-control",
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
    featured_modules = [
        spec["module_id"] for spec in module_specs if module_pages[spec["module_id"]]["entry_items"]
    ]

    roadmap_items = sort_items([item for item in items if item.get("type") == "roadmap_page"])
    roadmap_pages = {
        item["id"]: {
            "id": item["id"],
            "title": item["title"],
            "summary": item.get("hero_summary") or item.get("summary", ""),
            "summary_items": item.get("summary_items", []),
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


def generate_sitemap(
    items: List[Dict], base_url: str = "https://ImChong.github.io/Robotics_Notebooks"
) -> str:
    """生成 sitemap.xml，包含首页、预览页和所有 detail_pages。"""
    urls = [
        {"loc": base_url + "/", "priority": "1.0", "changefreq": "weekly"},
        {"loc": base_url + "/docs/index.html", "priority": "0.9", "changefreq": "weekly"},
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
        urls.append(
            {
                "loc": f"{base_url}/docs/detail.html?id={item_id}",
                "priority": priority,
                "changefreq": "monthly",
            }
        )

    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for url in urls:
        line = "  <url>"
        line += f"<loc>{url['loc']}</loc>"
        line += f"<changefreq>{url['changefreq']}</changefreq>"
        line += f"<priority>{url['priority']}</priority>"
        if url.get("lastmod"):
            line += f"<lastmod>{url['lastmod']}</lastmod>"
        line += "</url>"
        sitemap_lines.append(line)
    sitemap_lines.append("</urlset>")
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

    search_payload = generate_search_index()

    print(f"Wrote {OUTPUT} with {len(items)} items")
    print(f"Wrote {SITE_OUTPUT} with {len(site_payload['pages']['detail_pages'])} detail pages")
    print(f"Mirrored exports to {DOCS_OUTPUT.parent}")
    print(
        f"Wrote {SITEMAP_OUTPUT} with {sum(1 for i in items if i.get('type') in {'wiki_page', 'entity_page'})} wiki/entity URLs"
    )
    print(f"Wrote docs/search-index.json with {len(search_payload['docs'])} docs")


if __name__ == "__main__":
    main()
