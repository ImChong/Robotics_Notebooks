#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "exports" / "index-v1.json"

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
    text = re.sub(r"^>\s*", "", text)
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


def main() -> None:
    items = [build_item(p) for p in collect_paths()]
    payload = {
        "version": "v1",
        "generated_mode": "script",
        "item_count": len(items),
        "items": items,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(items)} items")


if __name__ == "__main__":
    main()
