#!/usr/bin/env python3
"""
wiki_to_marp.py — Wiki 页面转 Marp 幻灯片

将 wiki/*.md 页面转换为 Marp Markdown 幻灯片格式。
每个 H2 节（## 标题）对应一张幻灯片。

用法：
  python3 scripts/wiki_to_marp.py wiki/methods/model-predictive-control.md
  make slides F=wiki/methods/model-predictive-control.md
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SLIDES_DIR = REPO_ROOT / "exports" / "slides"

MARP_HEADER = """\
---
marp: true
theme: default
paginate: true
---

"""


def strip_frontmatter(content: str) -> tuple[str, dict]:
    """去掉 YAML frontmatter，返回 (body, meta)。"""
    meta: dict = {}
    if not content.startswith("---"):
        return content, meta
    end = content.find("\n---", 3)
    if end == -1:
        return content, meta
    fm_block = content[3:end].strip()
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return content[end + 4:].lstrip(), meta


def wiki_to_marp(wiki_path: Path) -> str:
    """将 wiki 页面转换为 Marp Markdown 文本。"""
    raw = wiki_path.read_text(encoding="utf-8")
    body, meta = strip_frontmatter(raw)

    # H1 = 封面幻灯片
    h1_m = re.search(r'^# (.+)', body, re.MULTILINE)
    cover_title = h1_m.group(1) if h1_m else wiki_path.stem
    page_type = meta.get("type", "")
    tags = meta.get("tags", "").strip("[]")

    cover = f"# {cover_title}\n\n"
    if page_type:
        cover += f"**类型：** {page_type}  \n"
    if tags:
        cover += f"**标签：** {tags}\n"

    # 按 H2 分割，每个 H2 节 → 一张幻灯片
    # 去掉 H1 行
    body_no_h1 = re.sub(r'^# .+\n?', '', body, count=1, flags=re.MULTILINE)

    sections = re.split(r'^(?=## )', body_no_h1, flags=re.MULTILINE)
    slides = [cover]
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # 跳过纯链接/导航节（参考来源、关联页面、推荐继续阅读）
        first_line = sec.splitlines()[0] if sec.splitlines() else ""
        if re.match(r'^## .*(参考来源|关联页面|推荐继续)', first_line):
            continue
        slides.append(sec)

    return MARP_HEADER + "\n\n---\n\n".join(slides)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python3 scripts/wiki_to_marp.py <wiki/path/to/page.md>")
        sys.exit(1)

    wiki_path = REPO_ROOT / sys.argv[1]
    if not wiki_path.exists():
        print(f"❌ 文件不存在: {wiki_path}")
        sys.exit(1)

    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SLIDES_DIR / wiki_path.stem
    out_path = out_path.with_suffix(".md")

    marp_content = wiki_to_marp(wiki_path)
    out_path.write_text(marp_content, encoding="utf-8")

    slide_count = marp_content.count("\n---\n")
    print(f"✅ 生成 {out_path.relative_to(REPO_ROOT)}  （{slide_count} 张幻灯片）")


if __name__ == "__main__":
    main()
