#!/usr/bin/env python3
"""Merge 国内具身开源全景 `cn-os-*` 占位页 into the canonical entity for the same project.

424 项全景 ingest 为每个仓库建了一个 `cn-os-*` 模板占位页；其中一部分项目本库
已有同仓库的深读实体页，于是同一节点在站点/图谱里出现两次（如 DiT4DiT）。
本脚本把占位页合并进 canonical 页：

1. canonical 页承接占位页的策展归属（`open-source` / `china-embodied-opensource`
   标签、全景总表与 424 覆盖索引双向链接、`sources/repos/*.md` 与公众号来源）；
2. 删除占位页，并在 `schema/page-aliases.json` 登记旧 detail 页 ID，保证历史 URL 重定向；
3. `exports/china-opensource-424-mappings.json` 的 slug 改指 canonical、`new` 置 false，
   覆盖表该行由「新建」改为「复用」，复用/新建计数同步；
4. 全库把指向占位页的链接改指 canonical，canonical 自身对占位页的引用则整行删除。

用法：``python3 scripts/dedupe_cn_os_stubs.py``（幂等，已合并过的条目会跳过）。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTITIES = REPO / "wiki" / "entities"
MAPPINGS = REPO / "exports" / "china-opensource-424-mappings.json"
URL_CACHE = REPO / "exports" / "china-opensource-424-repo-urls.json"
ALIASES = REPO / "schema" / "page-aliases.json"
COVERAGE = REPO / "wiki" / "queries" / "china-domestic-opensource-424-coverage.md"
OVERVIEW = (
    REPO / "wiki" / "overview" / "china-domestic-embodied-opensource-76-companies-technology-map.md"
)
LINK_SCRIPT = REPO / "scripts" / "link_china_opensource_repo_sources.py"
PANORAMA_BLOG = "sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md"
TODAY = date.today().isoformat()

# 占位页 slug -> canonical 实体 slug（均位于 wiki/entities/）
MERGE_MAP: dict[str, str] = {
    # canonical 页 frontmatter `code:` 与占位页归档仓库完全一致
    "cn-os-abot-recon": "paper-abot-recon",
    "cn-os-galaxeavla": "paper-galaxea-g05",
    "cn-os-gigaworld-policy": "paper-sa-2607-13960-gigaworld-policy-0-5-a-faster-and-stronger-wam-e",
    "cn-os-humantracker": "paper-humantracker",
    "cn-os-kairos": "paper-kairos-native-world-model-stack",
    "cn-os-lightnav-0": "paper-lightnav-0",
    "cn-os-lingbot-map": "paper-lingbot-map",
    "cn-os-world-in-your-hands": "paper-wiyh",
    # canonical 页未写 `code:`，但正文首段 GitHub 链接与占位页归档仓库一致
    "cn-os-abot-manipulation": "paper-abot-m05-mobile-manipulation-wam",
    "cn-os-abot-navigation": "paper-abot-n1",
    "cn-os-embodiedgen-v2": "paper-embodiedgen-v2-sim-ready-world-engine",
    "cn-os-motubrain": "paper-motubrain",
    "cn-os-opendm": "dexmal-dm05",
    "cn-os-opendw": "dexmal-dw05",
    "cn-os-rynnbrain": "paper-rynnbrain-1-1",
    # 同名同机构同项目（canonical 页无 GitHub 链接，按项目名与机构判定）
    "cn-os-gigaworld-1": "paper-gigaworld-1-policy-evaluation",
    "cn-os-rynnworld-4d": "paper-rynnworld-4d-rgb-depth-flow",
    "cn-os-video-prediction-policy": "paper-shenlan-wm-02-vpp",
}

CARRY_TAGS = ("open-source", "china-embodied-opensource")
SKIP_PARTS = {".git", "node_modules", ".cursor-artifacts", "raw"}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit(f"frontmatter 缺失: {text[:60]!r}")
    return match.group(1), text[match.end() :]


def stub_reference_lines(stub_text: str) -> list[str]:
    """占位页「参考来源」里指向 sources/repos 或全景公众号的条目。"""
    body = stub_text.split("## 参考来源", 1)
    if len(body) < 2:
        return []
    section = body[1].split("\n## ", 1)[0]
    return [
        line.strip()
        for line in section.splitlines()
        if line.startswith("- [") and ("sources/repos/" in line or PANORAMA_BLOG in line)
    ]


def add_tags(frontmatter: str, tags: tuple[str, ...]) -> str:
    inline = re.search(r"^tags:\s*\[(.*?)\]\s*$", frontmatter, re.MULTILINE | re.DOTALL)
    if inline:
        existing = [t.strip() for t in inline.group(1).split(",") if t.strip()]
        merged = existing + [t for t in tags if t not in existing]
        return (
            frontmatter[: inline.start()]
            + f"tags: [{', '.join(merged)}]"
            + frontmatter[inline.end() :]
        )
    block = re.search(r"^tags:\n((?:  - .+\n)+)", frontmatter + "\n", re.MULTILINE)
    if not block:
        raise SystemExit("未识别的 tags 写法")
    existing = [line[4:].strip() for line in block.group(1).splitlines()]
    extra = "".join(f"  - {t}\n" for t in tags if t not in existing)
    return (
        (frontmatter + "\n")[: block.end()].rstrip("\n")
        + "\n"
        + extra
        + (frontmatter + "\n")[block.end() :].rstrip("\n")
    )


def add_list_entries(frontmatter: str, key: str, entries: list[str]) -> str:
    block = re.search(rf"^{key}:\n((?:  - .+\n)+)", frontmatter + "\n", re.MULTILINE)
    if not block:
        return frontmatter
    existing = block.group(1)
    extra = "".join(f"  - {e}\n" for e in entries if f"  - {e}\n" not in existing)
    if not extra:
        return frontmatter
    padded = frontmatter + "\n"
    return (padded[: block.end()] + extra + padded[block.end() :]).rstrip("\n")


def link_target(line: str) -> str:
    match = re.search(r"\]\(([^)]+)\)", line)
    return match.group(1) if match else ""


def append_section_lines(body: str, heading: str, lines: list[str]) -> str:
    """在指定二级标题小节末尾追加条目。

    去重按 **链接目标** 判断：canonical 页往往已用别的措辞引用了同一个
    `sources/…` 归档（如「代码仓」vs 占位页的「源码归档」），此时不再重复追加。
    """
    marker = f"## {heading}\n"
    if marker not in body or not lines:
        return body
    head, rest = body.split(marker, 1)
    parts = rest.split("\n## ", 1)
    section, tail = parts[0], ("\n## " + parts[1] if len(parts) > 1 else "")
    present = {link_target(ln) for ln in section.splitlines() if ln.startswith("- ")}
    new = [ln for ln in lines if ln not in section and link_target(ln) not in present]
    if not new:
        return body
    return head + marker + section.rstrip("\n") + "\n" + "\n".join(new) + "\n" + tail


def merge_into_canonical(stub: str, canonical: str, stub_text: str) -> None:
    path = ENTITIES / f"{canonical}.md"
    frontmatter, body = split_frontmatter(load_text(path))

    frontmatter = add_tags(frontmatter, CARRY_TAGS)
    frontmatter = re.sub(r"^updated:.*$", f"updated: {TODAY}", frontmatter, count=1, flags=re.M)
    frontmatter = add_list_entries(
        frontmatter,
        "related",
        [
            "../overview/china-domestic-embodied-opensource-76-companies-technology-map.md",
            "../queries/china-domestic-opensource-424-coverage.md",
        ],
    )
    stub_fm, _ = split_frontmatter(stub_text)
    stub_sources = re.findall(r"^  - (\.\./\.\./sources/\S+)$", stub_fm, re.M)
    frontmatter = add_list_entries(frontmatter, "sources", stub_sources)

    body = append_section_lines(
        body,
        "关联页面",
        [
            "- [国内具身开源全景（76 家 · 424 项）]"
            "(../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)"
            " — 本页为该清单对应条目的 canonical 详情节点",
            "- [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md)"
            " — 同公司其它开源入口",
        ],
    )
    body = append_section_lines(body, "参考来源", stub_reference_lines(stub_text))

    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def drop_self_references(text: str, stub: str) -> str:
    """canonical 页里指向占位页的引用：独立条目整行删除，句中引用退化为纯文本。

    句中引用若照常改写目标就会变成指向自己的自链接，所以只去掉链接外壳、保留文字，
    具体措辞由 main() 汇报后人工复核。
    """
    kept = []
    for line in text.splitlines(keepends=True):
        if f"{stub}.md" not in line:
            kept.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and stripped.count("](") <= 1:
            continue
        kept.append(re.sub(r"\[([^\]]*)\]\([^)]*" + re.escape(stub) + r"\.md\)", r"\1", line))
    return "".join(kept)


def rewrite_links(text: str, stub: str, canonical: str) -> str:
    """把指向占位页的 markdown 链接目标改指 canonical（只改目标，不改链接文字）。"""
    for pattern, repl in (
        (rf"\./{re.escape(stub)}\.md", f"./{canonical}.md"),
        (rf"\.\./entities/{re.escape(stub)}\.md", f"../entities/{canonical}.md"),
        (rf"wiki/entities/{re.escape(stub)}\.md", f"wiki/entities/{canonical}.md"),
    ):
        text = re.sub(pattern, repl, text)
    return text


def walk_text_files() -> list[Path]:
    out = []
    for path in REPO.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix in {".md", ".py", ".json", ".yml"} and path != Path(__file__):
            out.append(path)
    return out


def update_mappings() -> tuple[int, int]:
    rows = json.loads(load_text(MAPPINGS))
    for row in rows:
        if row["slug"] in MERGE_MAP:
            row["slug"] = MERGE_MAP[row["slug"]]
            row["new"] = False
    MAPPINGS.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new_count = sum(1 for r in rows if r["new"])
    return len(rows) - new_count, new_count


def update_aliases() -> None:
    raw = load_text(ALIASES)
    data = json.loads(raw)
    additions = [
        f'    "entity-{stub}": "entity-{canonical}",\n'
        for stub, canonical in sorted(MERGE_MAP.items())
        if f"entity-{stub}" not in data["aliases"]
    ]
    if not additions:
        return
    anchor = '  "aliases": {\n'
    raw = raw.replace(anchor, anchor + "".join(additions), 1)
    json.loads(raw)
    ALIASES.write_text(raw, encoding="utf-8")


def update_url_cache() -> None:
    if not URL_CACHE.exists():
        return
    cache = json.loads(load_text(URL_CACHE))
    updated = {}
    for key, url in cache.items():
        slug, _, name = key.partition("\x00")
        updated[f"{MERGE_MAP.get(slug, slug)}\x00{name}"] = url
    URL_CACHE.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    text = load_text(LINK_SCRIPT)
    for stub, canonical in MERGE_MAP.items():
        text = text.replace(f'("{stub}",', f'("{canonical}",')
    LINK_SCRIPT.write_text(text, encoding="utf-8")


def update_counts(reused: int, created: int) -> None:
    text = load_text(COVERAGE)
    text = re.sub(r"\| 复用既有实体 \| \d+ \|", f"| 复用既有实体 | {reused} |", text)
    text = re.sub(r"\| 本 ingest 新建实体 \| \d+ \|", f"| 本 ingest 新建实体 | {created} |", text)
    COVERAGE.write_text(text, encoding="utf-8")

    text = load_text(OVERVIEW)
    text = re.sub(r"\*\*复用 \d+\*\* 既有实体", f"**复用 {reused}** 既有实体", text)
    text = re.sub(r"\*\*新建 \d+\*\* `cn-os-\*` 实体", f"**新建 {created}** `cn-os-*` 实体", text)
    OVERVIEW.write_text(text, encoding="utf-8")


def main() -> None:
    merged: list[str] = []
    for stub, canonical in sorted(MERGE_MAP.items()):
        stub_path = ENTITIES / f"{stub}.md"
        if not stub_path.exists():
            continue
        if not (ENTITIES / f"{canonical}.md").exists():
            raise SystemExit(f"canonical 页不存在: {canonical}")
        merge_into_canonical(stub, canonical, load_text(stub_path))
        stub_path.unlink()
        merged.append(stub)

    changed = 0
    delinked: list[str] = []
    for path in walk_text_files():
        try:
            text = original = load_text(path)
        except UnicodeDecodeError:
            continue
        for stub, canonical in MERGE_MAP.items():
            if stub not in text:
                continue
            if path.stem == canonical:
                delinked += [
                    f"{path.relative_to(REPO)}: {line.strip()}"
                    for line in text.splitlines()
                    if f"{stub}.md" in line
                    and not (line.strip().startswith("- ") and line.count("](") <= 1)
                ]
                text = drop_self_references(text, stub)
            text = rewrite_links(text, stub, canonical)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1

    # 覆盖表：合并后的行由「新建」改为「复用」
    coverage = load_text(COVERAGE)
    coverage = re.sub(
        r"\[([^\]]+)\]\(\.\./entities/("
        + "|".join(map(re.escape, MERGE_MAP.values()))
        + r")\.md\) · 新建",
        r"[\1](../entities/\2.md) · 复用",
        coverage,
    )
    COVERAGE.write_text(coverage, encoding="utf-8")

    update_aliases()
    update_url_cache()
    reused, created = update_mappings()
    update_counts(reused, created)

    print(f"合并占位页: {len(merged)}")
    for stub in merged:
        print(f"  - {stub} → {MERGE_MAP[stub]}")
    print(f"改写引用的文件: {changed}")
    print(f"复用/新建计数: {reused}/{created}")

    leftovers = [
        f"{path.relative_to(REPO)}: {line.strip()}"
        for path in walk_text_files()
        if path.suffix == ".md"
        for line in load_text(path).splitlines()
        for stub in MERGE_MAP
        if stub in line
    ]
    if leftovers:
        print(
            f"\n仍残留占位页名称的行（需人工确认文案，链接目标已改指 canonical）：{len(leftovers)}"
        )
        for item in leftovers:
            print(f"  - {item}")
    if delinked:
        print(f"\ncanonical 页内被去链接的句中引用（需人工复核措辞）：{len(delinked)}")
        for item in delinked:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
