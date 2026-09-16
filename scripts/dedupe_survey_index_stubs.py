#!/usr/bin/env python3
"""Merge 策展索引级占位页 into the deep entity that covers the same paper.

几套策展清单（Awesome-* 的 `paper-sa-*`、Loco-Manip 161 的
`paper-loco-manip-161-*`、Paper Notebooks 的 `paper-notebook-*`）都按「一条清单
条目 = 一个节点」建页；当同一篇论文本库另有深读页时，站点与图谱里就出现两个节点。
`lint_wiki` 的 V30 只比对 frontmatter `arxiv:`，深读页普遍没写这一项，所以这类重复
长期没被门禁发现。

本脚本把索引页并入深读页：深读页继承索引页的清单归属（tags / related / sources /
参考来源），索引页删除并在 `schema/page-aliases.json` 登记旧 detail 页 ID，全库指向
索引页的链接改指深读页。

用法：``python3 scripts/dedupe_survey_index_stubs.py``（幂等）。
"""

from __future__ import annotations

import json
import re
from datetime import date

from dedupe_cn_os_stubs import (
    ALIASES,
    ENTITIES,
    REPO,
    add_list_entries,
    append_section_lines,
    drop_self_references,
    load_text,
    rewrite_links,
    split_frontmatter,
    walk_text_files,
)

TODAY = date.today().isoformat()

# 索引页 slug -> 深读页 slug
MERGE_MAP: dict[str, str] = {
    # V30 在补齐深读页 frontmatter arxiv: 后报出的同 ID 重复
    "paper-sa-2604-07607-egoverse": "paper-egoverse",
    "paper-sa-2607-06988-wam-ttt-steering-world-action-models-by-watching": (
        "paper-wam-ttt-human-video-test-time-steering"
    ),
    "paper-sa-2607-11643-xiaomi-robotics-u0-unified-embodied-synthesis-wi": "xiaomi-robotics-u0",
    # 索引页正文自述「本页仅为索引 / 重复索引，深读见 X」
    "paper-loco-manip-161-009-gmt": "paper-gmt",
    "paper-notebook-aero-hand-open": "paper-aero-hand-open",
    "paper-notebook-adamimic": "paper-adamimic",
    "paper-notebook-towards-adaptable-humanoid-control-via-adaptive": "paper-adamimic",
    "paper-notebook-general-motion-tracking-for-humanoid-whole-body": "paper-gmt",
}


def stub_tags(frontmatter: str) -> list[str]:
    inline = re.search(r"^tags:\s*\[(.*?)\]\s*$", frontmatter, re.MULTILINE | re.DOTALL)
    if inline:
        return [t.strip() for t in inline.group(1).split(",") if t.strip()]
    block = re.search(r"^tags:\n((?:  - .+\n)+)", frontmatter + "\n", re.MULTILINE)
    return [line[4:].strip() for line in block.group(1).splitlines()] if block else []


def merge_tags(frontmatter: str, tags: list[str]) -> str:
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
    padded = frontmatter + "\n"
    return (padded[: block.end()] + extra + padded[block.end() :]).rstrip("\n")


def stub_reference_lines(stub_text: str) -> list[str]:
    parts = stub_text.split("## 参考来源", 1)
    if len(parts) < 2:
        return []
    section = parts[1].split("\n## ", 1)[0]
    return [ln.strip() for ln in section.splitlines() if ln.startswith("- [")]


def merge_into_keeper(stub: str, keeper: str, stub_text: str) -> None:
    path = ENTITIES / f"{keeper}.md"
    frontmatter, body = split_frontmatter(load_text(path))
    stub_fm, _ = split_frontmatter(stub_text)

    # 只继承清单归属类标签（策展集合名），不继承通用主题标签，避免噪声
    curated = [
        t
        for t in stub_tags(stub_fm)
        if any(k in t for k in ("curated-index", "awesome", "sun254667", "survey", "notebooks"))
    ]
    frontmatter = merge_tags(frontmatter, curated)
    frontmatter = re.sub(r"^updated:.*$", f"updated: {TODAY}", frontmatter, count=1, flags=re.M)

    stub_related = [
        r for r in re.findall(r"^  - (\.\./\S+|\./\S+)$", stub_fm, re.M) if f"/{keeper}.md" not in r
    ]
    frontmatter = add_list_entries(frontmatter, "related", stub_related)
    frontmatter = add_list_entries(
        frontmatter, "sources", re.findall(r"^  - (\.\./\.\./sources/\S+)$", stub_fm, re.M)
    )

    body = append_section_lines(body, "参考来源", stub_reference_lines(stub_text))
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def normalize_link_texts(text: str, stub: str, keeper: str) -> str:
    """链接目标改指深读页后，把仍写着索引页 slug 的链接文字也换成深读页 slug。

    策展总表 / 分类地图里大量 `[<索引页 slug>](<深读页>)` 这类行，只改目标会让
    文字与落点不一致。纯文本（无链接）的历史叙述不动。
    """
    for text_form in (f"`{stub}.md`", f"`{stub}`", f"{stub}.md", stub):
        keeper_form = text_form.replace(stub, keeper)
        text = text.replace(f"[{text_form}](", f"[{keeper_form}](")
    return text


def update_aliases() -> None:
    raw = load_text(ALIASES)
    data = json.loads(raw)
    additions = [
        f'    "entity-{stub}": "entity-{keeper}",\n'
        for stub, keeper in sorted(MERGE_MAP.items())
        if f"entity-{stub}" not in data["aliases"]
    ]
    if not additions:
        return
    anchor = '  "aliases": {\n'
    raw = raw.replace(anchor, anchor + "".join(additions), 1)
    json.loads(raw)
    ALIASES.write_text(raw, encoding="utf-8")


def main() -> None:
    merged = []
    for stub, keeper in sorted(MERGE_MAP.items()):
        stub_path = ENTITIES / f"{stub}.md"
        if not stub_path.exists():
            continue
        if not (ENTITIES / f"{keeper}.md").exists():
            raise SystemExit(f"深读页不存在: {keeper}")
        merge_into_keeper(stub, keeper, load_text(stub_path))
        stub_path.unlink()
        merged.append(stub)

    changed = 0
    delinked: list[str] = []
    for path in walk_text_files():
        try:
            text = original = load_text(path)
        except UnicodeDecodeError:
            continue
        for stub, keeper in MERGE_MAP.items():
            if stub not in text:
                continue
            if path.stem == keeper:
                delinked += [
                    f"{path.relative_to(REPO)}: {line.strip()}"
                    for line in text.splitlines()
                    if f"{stub}.md" in line
                    and not (line.strip().startswith("- ") and line.count("](") <= 1)
                ]
                text = drop_self_references(text, stub)
            text = rewrite_links(text, stub, keeper)
            text = normalize_link_texts(text, stub, keeper)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1

    update_aliases()

    print(f"合并索引页: {len(merged)}")
    for stub in merged:
        print(f"  - {stub} → {MERGE_MAP[stub]}")
    print(f"改写引用的文件: {changed}")

    leftovers = [
        f"{path.relative_to(REPO)}: {line.strip()}"
        for path in walk_text_files()
        if path.suffix == ".md"
        for line in load_text(path).splitlines()
        for stub in MERGE_MAP
        if stub in line
    ]
    for label, items in (("仍残留索引页名称的行", leftovers), ("被去链接的句中引用", delinked)):
        if items:
            print(f"\n{label}（需人工复核）：{len(items)}")
            for item in items:
                print(f"  - {item}")


if __name__ == "__main__":
    main()
