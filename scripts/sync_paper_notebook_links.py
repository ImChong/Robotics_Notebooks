#!/usr/bin/env python3
"""Sync Humanoid_Robot_Learning_Paper_Notebooks deep-read links into wiki pages."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
INDEX_PATH = SCHEMA_DIR / "paper-notebook-index.json"
MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-map.yml"
FULL_MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-full-map.yml"
OVERRIDES_PATH = SCHEMA_DIR / "paper-notebook-wiki-overrides.yml"
AUTO_MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-auto-map.yml"
PAPERS_JSON_URL = (
    "https://raw.githubusercontent.com/ImChong/"
    "Humanoid_Robot_Learning_Paper_Notebooks/main/_data/papers.json"
)
TREE_URL = (
    "https://api.github.com/repos/ImChong/"
    "Humanoid_Robot_Learning_Paper_Notebooks/git/trees/main?recursive=1"
)
BASE_URL = "https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks"
LINK_PREFIX = "机器人论文阅读笔记："


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def clean_display_title(title: str) -> str:
    """Strip markdown link wrappers and stray brackets from Paper Notebooks titles."""
    title = title.strip()
    link = re.match(r"^\[([^\]]+)\]\([^)]+\)$", title)
    if link:
        title = link.group(1).strip()
    if title.lower().startswith("[website],"):
        title = title.split(",", 1)[1].strip()
    if title.startswith("[") and "]" in title:
        title = title[1 : title.index("]")].strip()
    elif title.startswith("["):
        title = title[1:].strip()
    return title


def short_label(title: str) -> str:
    title = clean_display_title(title)
    label = title.split(":")[0].split("(")[0].strip()
    return label or title.strip()


def norm_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_paper_index(force_refresh: bool = False) -> list[dict]:
    if INDEX_PATH.exists() and not force_refresh:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    tree = fetch_json(TREE_URL)["tree"]
    papers_meta: dict[str, dict] = {}
    try:
        papers_json = fetch_json(PAPERS_JSON_URL)
        for _cat, section in papers_json.items():
            for item in section.get("papers", []):
                folder = "/".join(item["path"].split("/")[:-1])
                papers_meta[folder] = item
    except OSError as exc:
        print(f"warning: could not fetch papers.json: {exc}", file=sys.stderr)

    papers: list[dict] = []
    seen_folders: set[str] = set()
    for node in tree:
        path = node["path"]
        if not path.startswith("papers/") or not path.endswith(".md"):
            continue
        if "/todos/" in path or "PROGRESS" in path or path.endswith("README.md"):
            continue
        folder = "/".join(path.split("/")[:-1])
        if folder in seen_folders or folder == "papers":
            continue
        seen_folders.add(folder)
        html_path = path.rsplit(".", 1)[0] + ".html"
        meta = papers_meta.get(folder, {})
        title = meta.get("title") or Path(folder).name.replace("__", ": ").replace("_", " ")
        parts = folder.split("/")
        category = meta.get("_category") or (parts[1] if len(parts) > 1 else "")
        papers.append(
            {
                "folder": folder,
                "dir": Path(folder).name,
                "title": title,
                "arxiv": meta.get("arxiv"),
                "url": BASE_URL + (meta.get("url") or f"/{html_path}"),
                "category": category,
            }
        )

    papers.sort(key=lambda p: p["folder"])
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return papers


def parse_frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    return match.group(1) if match else ""


def wiki_source_paths(wiki_path: Path, frontmatter: str) -> list[str]:
    paths: list[str] = []
    for line in frontmatter.splitlines():
        m = re.match(r"^\s*-\s+(\.\./\.\./sources/[^\n]+\.md)\s*$", line)
        if not m:
            continue
        src = (wiki_path.parent / m.group(1)).resolve()
        if src.exists():
            paths.append(str(src.relative_to(ROOT.resolve())))
    return paths


def collect_wiki_index() -> dict[str, dict[str, set[str]]]:
    arxiv_to_wiki: dict[str, set[str]] = defaultdict(set)
    title_to_wiki: dict[str, set[str]] = defaultdict(set)
    h1_entity_to_wiki: dict[str, set[str]] = defaultdict(set)
    method_slug_to_wiki: dict[str, set[str]] = defaultdict(set)

    for wiki_path in (ROOT / "wiki").rglob("*.md"):
        rel = str(wiki_path.relative_to(ROOT))
        text = wiki_path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(text)
        h1 = re.search(r"^#\s+(.+)$", text, re.M)
        if h1 and rel.startswith("wiki/entities/"):
            h1_entity_to_wiki[norm_title(clean_display_title(h1.group(1)))].add(rel)
        if rel.startswith("wiki/methods/"):
            method_slug_to_wiki[wiki_path.stem.lower()].add(rel)

        arxivs: set[str] = set()
        fm_arxiv = re.search(r'^arxiv:\s*"?([0-9]+\.[0-9]+)"?\s*$', frontmatter, re.M)
        if fm_arxiv:
            arxivs.add(fm_arxiv.group(1))
        for src_rel in wiki_source_paths(wiki_path, frontmatter):
            src_text = (ROOT / src_rel).read_text(encoding="utf-8", errors="replace")
            arxivs.update(re.findall(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", src_text))
            src_h1 = re.search(r"^#\s+(.+)$", src_text, re.M)
            if src_h1:
                title_to_wiki[norm_title(src_h1.group(1))].add(rel)
        for arxiv in arxivs:
            arxiv_to_wiki[arxiv].add(rel)

    return {
        "arxiv": arxiv_to_wiki,
        "title": title_to_wiki,
        "h1_entity": h1_entity_to_wiki,
        "method_slug": method_slug_to_wiki,
    }


def _entity_pick_score(rel: str) -> tuple[int, int, str]:
    score = 0
    if "paper-notebook-" in rel:
        score -= 100
    if rel.startswith("wiki/entities/") and "paper-" not in Path(rel).name:
        score += 30
    if "paper-bfm-" in rel or "paper-hrl-stack-" in rel or "paper-amp-survey-" in rel:
        score -= 40
    if rel.startswith("wiki/methods/"):
        score -= 10
    if "behavior-foundation-model" in rel or rel.endswith(
        "paper-pilot-perceptive-loco-manipulation.md"
    ):
        score += 20
    if (
        "paper-digit-humanoid-locomotion-rl.md" in rel
        or "paper-cassie-biped-versatile-locomotion-rl.md" in rel
    ):
        score += 15
    return (-score, len(rel), rel)


def pick_entity_or_single(candidates: set[str]) -> list[str]:
    if len(candidates) == 1:
        return [next(iter(candidates))]
    deep = [c for c in candidates if c.startswith("wiki/entities/") and "paper-notebook-" not in c]
    if deep:
        return [sorted(deep, key=_entity_pick_score)[0]]
    entity = sorted(c for c in candidates if c.startswith("wiki/entities/"))
    if len(entity) == 1:
        return entity
    methods = sorted(c for c in candidates if c.startswith("wiki/methods/"))
    if len(methods) == 1:
        return methods
    return []


def auto_match(paper: dict, wiki_index: dict) -> list[str]:
    if paper.get("arxiv"):
        picked = pick_entity_or_single(wiki_index["arxiv"].get(paper["arxiv"], set()))
        if picked:
            return picked

    for key in (norm_title(paper["title"]), norm_title(short_label(paper["title"]))):
        picked = pick_entity_or_single(wiki_index["title"].get(key, set()))
        if picked:
            return picked
        picked = pick_entity_or_single(wiki_index["h1_entity"].get(key, set()))
        if picked:
            return picked

    slug = short_label(paper["title"]).lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]+", "", slug.replace("_", "-"))
    if slug in wiki_index["method_slug"]:
        return sorted(wiki_index["method_slug"][slug])

    return []


def load_manual_map() -> dict[str, list[str]]:
    path = OVERRIDES_PATH if OVERRIDES_PATH.exists() else MAP_PATH
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides", {})
    result: dict[str, list[str]] = {}
    for key, value in overrides.items():
        result[key] = [value] if isinstance(value, str) else list(value)
    return result


def link_line(paper: dict) -> str:
    return f"- [{LINK_PREFIX}{short_label(paper['title'])}]({paper['url']})"


def has_notebook_link(text: str, paper: dict) -> bool:
    if "Humanoid_Robot_Learning_Paper_Notebooks" not in text:
        return False
    return paper["url"] in text or paper["dir"] in text


def inject_link(wiki_rel: str, line: str, dry_run: bool) -> bool:
    wiki_path = ROOT / wiki_rel
    if not wiki_path.exists():
        return False
    text = wiki_path.read_text(encoding="utf-8")
    if line in text or has_notebook_link(text, {"url": line, "dir": paper_dir_from_line(line)}):
        return False

    section = "## 推荐继续阅读"
    if section in text:
        parts = text.split(section, 1)
        body = parts[1].lstrip("\n")
        new_text = parts[0] + section + "\n\n" + line + "\n" + body
    else:
        ref_section = "## 参考来源"
        if ref_section in text and wiki_rel.startswith("wiki/methods/"):
            parts = text.split(ref_section, 1)
            new_text = parts[0] + ref_section + "\n" + line + parts[1]
        else:
            new_text = text.rstrip() + f"\n\n{section}\n\n{line}\n"

    if not dry_run:
        wiki_path.write_text(new_text, encoding="utf-8")
    return True


def paper_dir_from_line(line: str) -> str:
    m = re.search(r"/papers/[^/]+/([^/]+)/", line)
    return m.group(1) if m else ""


def fix_stale_urls(papers: list[dict], dry_run: bool) -> int:
    dir_to_url = {p["dir"]: p["url"] for p in papers}
    changed = 0
    for wiki_path in (ROOT / "wiki").rglob("*.md"):
        text = wiki_path.read_text(encoding="utf-8")
        new_text = text
        for match in re.finditer(
            r"https://imchong\.github\.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/[^)\s]+",
            text,
        ):
            old_url = match.group(0)
            dir_match = re.search(r"/papers/[^/]+/([^/]+)/", old_url)
            if not dir_match:
                continue
            new_url = dir_to_url.get(dir_match.group(1))
            if new_url and new_url != old_url:
                new_text = new_text.replace(old_url, new_url)
        if new_text != text:
            changed += 1
            if not dry_run:
                wiki_path.write_text(new_text, encoding="utf-8")
            print(
                f"{'would fix urls in' if dry_run else 'fixed urls in'}: {wiki_path.relative_to(ROOT)}"
            )
    return changed


def load_full_map() -> dict[str, list[str]]:
    if not FULL_MAP_PATH.exists():
        return {}
    data = yaml.safe_load(FULL_MAP_PATH.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides", {})
    result: dict[str, list[str]] = {}
    for key, value in overrides.items():
        result[key] = [value] if isinstance(value, str) else list(value)
    return result


def build_mapping(papers: list[dict], wiki_index: dict) -> dict[str, list[str]]:
    full = load_full_map()
    manual = load_manual_map()
    mapping: dict[str, list[str]] = {}
    for paper in papers:
        key = paper["dir"]
        if key in full:
            mapping[key] = full[key]
            continue
        if key in manual:
            mapping[key] = manual[key]
            continue
        auto = auto_match(paper, wiki_index)
        if auto:
            mapping[key] = auto
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-index", action="store_true", help="Re-download paper notebook index"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned edits without writing files"
    )
    parser.add_argument(
        "--write-map", action="store_true", help="Write auto-generated mapping to schema YAML"
    )
    args = parser.parse_args()

    papers = build_paper_index(force_refresh=args.refresh_index)
    wiki_index = collect_wiki_index()
    mapping = build_mapping(papers, wiki_index)

    if args.write_map:
        auto_only: dict[str, list[str]] = {}
        manual = load_manual_map()
        for paper in papers:
            key = paper["dir"]
            if key in manual:
                continue
            auto = auto_match(paper, wiki_index)
            if auto:
                auto_only[key] = auto
        AUTO_MAP_PATH.write_text(
            yaml.safe_dump({"overrides": auto_only}, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        merged = {**auto_only, **manual}
        MAP_PATH.write_text(
            yaml.safe_dump({"overrides": merged}, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"wrote {AUTO_MAP_PATH} ({len(auto_only)} auto) and "
            f"{MAP_PATH} ({len(merged)} merged); manual={len(manual)}"
        )
        return 0

    url_fixes = fix_stale_urls(papers, args.dry_run)
    changed = 0
    for paper in papers:
        targets = mapping.get(paper["dir"], [])
        line = link_line(paper)
        for wiki_rel in targets:
            if inject_link(wiki_rel, line, args.dry_run):
                changed += 1
                action = "would update" if args.dry_run else "updated"
                print(f"{action}: {wiki_rel} <- {short_label(paper['title'])}")

    print(
        f"{'would change' if args.dry_run else 'changed'} {changed} wiki pages; "
        f"{'would fix' if args.dry_run else 'fixed'} {url_fixes} stale url files; "
        f"mapped papers {len(mapping)} / {len(papers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
