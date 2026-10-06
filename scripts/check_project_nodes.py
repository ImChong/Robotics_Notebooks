#!/usr/bin/env python3
"""Prevent paper / website / implementation splits of the same wiki project.

Only primary frontmatter identities count. Body citations, sources archives and
shared implementation repositories are not sufficient evidence of duplication.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARXIV = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
PROJECT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not match:
        return {}
    data = yaml.safe_load(match[1])
    if not isinstance(data, dict):
        raise ValueError("frontmatter 必须是映射")
    return data


def values(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if item is not None and str(item).strip()]


def paper_ids(meta: dict[str, Any]) -> set[str]:
    return {
        match[1]
        for key in ("arxiv", "paper", "papers")
        for value in values(meta.get(key))
        for match in ARXIV.finditer(value)
    }


def resource_url(value: str, *, repository: bool = False) -> str:
    url = urlsplit(value.strip())
    if url.scheme not in {"http", "https"} or not url.hostname:
        return ""
    host = url.hostname.lower().removeprefix("www.")
    path = unquote(url.path).rstrip("/")
    if host == "github.com":
        parts = path.strip("/").split("/")
        if len(parts) < 2:
            return ""  # An organization is not a project repository.
        path = "/" + "/".join(parts[:2]).lower().removesuffix(".git")
    else:
        path = re.sub(r"/index\.html?$", "", path, flags=re.I)
    # Project anchors may distinguish projects on a shared lab listing page.
    fragment = "#" + url.fragment if url.fragment and not repository else ""
    query = ""
    if not repository and host != "github.com":
        query = urlencode(
            sorted(
                (key, val)
                for key, val in parse_qsl(url.query, keep_blank_values=True)
                if not key.lower().startswith("utm_") and key.lower() not in {"gclid", "fbclid"}
            )
        )
    port = f":{url.port}" if url.port and url.port not in {80, 443} else ""
    return host + port + path + ("?" + query if query else "") + fragment


def added_entities(root: Path, base: str) -> set[str]:
    """Compare the actual filesystem to the base tree, including untracked pages."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", base, "--", "wiki/entities"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    before = set(result.stdout.splitlines())
    now = {p.relative_to(root).as_posix() for p in (root / "wiki/entities").glob("*.md")}
    return now - before


def check(root: Path, *, new_pages: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    identities: dict[tuple[str, str], list[str]] = defaultdict(list)
    repos: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    new_pages = new_pages or set()
    for path in sorted((root / "wiki/entities").glob("*.md")):
        name = path.relative_to(root).as_posix()
        try:
            meta = frontmatter(path)
            project = str(meta.get("project_id", "")).strip()
            if "project_id" in meta and not PROJECT_ID.fullmatch(project):
                errors.append(f"{name}: project_id 必须是非空小写英文 slug")
            if project:
                identities["project_id", project].append(name)
            for paper in paper_ids(meta):
                identities["arxiv", paper].append(name)
            for key in ("project", "url"):
                for value in values(meta.get(key)):
                    normalized = resource_url(value)
                    if not normalized:
                        errors.append(f"{name}: {key} 必须是官方项目的 HTTP(S) 地址")
                    else:
                        identities["project_url", normalized].append(name)
            tags = values(meta.get("tags"))
            is_project = (
                path.stem.startswith(("paper-", "project-", "repo-"))
                or bool(set(tags) & {"paper", "repo", "project"})
                or any(meta.get(k) for k in ("arxiv", "paper", "papers", "code", "project"))
            )
            if name in new_pages and is_project and not project:
                errors.append(f"{name}: 新项目实体缺 project_id；先检索既有项目并复用主节点")
            if (
                name in new_pages
                and is_project
                and not any(
                    meta.get(k)
                    for k in (
                        "arxiv",
                        "paper",
                        "papers",
                        "code",
                        "project",
                        "url",
                        "doi",
                        "openreview",
                    )
                )
            ):
                errors.append(f"{name}: 新项目实体须标注主论文编号、官方 project 或 code 地址")
            if name in new_pages:
                for key in ("arxiv", "paper", "papers"):
                    for value in values(meta.get(key)):
                        if not ARXIV.search(value):
                            errors.append(f"{name}: {key} 的论文编号无效：{value}")
            for value in values(meta.get("code")):
                normalized = resource_url(value, repository=True)
                if normalized:
                    repos[normalized].append((name, meta))
                elif name in new_pages:
                    errors.append(f"{name}: code 必须是官方源码的 HTTP(S) 地址")
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{name}: 项目元数据无效：{exc}")

    for (kind, value), paths in sorted(identities.items()):
        paths = sorted(set(paths))
        if len(paths) > 1:
            errors.append(f"重复 {kind}={value} → {' / '.join(paths)}；合并到一个项目节点")

    # Existing libraries may host several papers. Block new resource-only splits,
    # but allow distinct paper IDs or documented modules of a shared monorepo.
    for repo, entries in sorted(repos.items()):
        for name, meta in entries:
            if name not in new_pages:
                continue
            for other, other_meta in entries:
                if name == other:
                    continue
                ids, other_ids = paper_ids(meta), paper_ids(other_meta)
                if ids and other_ids and ids.isdisjoint(other_ids):
                    continue
                scope = str(meta.get("code_scope", "")).strip()
                other_scope = str(other_meta.get("code_scope", "")).strip()
                if scope and scope != other_scope and meta.get("project_distinction"):
                    continue
                errors.append(
                    f"{name}: 官方源码 {repo} 已由 {other} 使用；"
                    "复用项目节点，独立模块须注明 code_scope 与 project_distinction"
                )
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="检查相对此 git ref 新增实体的身份字段；无效 ref 阻塞检查")
    args = parser.parse_args()
    try:
        new_pages = added_entities(ROOT, args.base) if args.base else set()
        errors = check(ROOT, new_pages=new_pages)
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"无法读取基线 {args.base}: {exc.stderr}\n")
    for error in errors:
        print(f"❌ {error}")
    if errors:
        parser.exit(1, f"项目节点检查失败：{len(errors)} 项\n")
    print("✅ 项目节点身份唯一；论文、项目页与官方源码统一导航")


if __name__ == "__main__":
    main()
