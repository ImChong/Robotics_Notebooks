#!/usr/bin/env python3
"""
迁移 wiki/ 内所有 [[wikilink]] 为标准 Markdown [text](path) 内链。

策略：
- 跳过代码块（```...```）和行内代码（`...`）
- 含 `/` 的写法按字面路径解析（如 [[references/repos/simulation]]）
- 不含 `/` 的写法按 stem 查找（在 wiki/, sources/, references/, roadmap/ 中）
- 支持 alias 写法 [[stem|显示文本]]
- 链接文本：alias > basename
"""
import re
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
SEARCH_DIRS = [
    REPO_ROOT / "wiki",
    REPO_ROOT / "sources",
    REPO_ROOT / "references",
    REPO_ROOT / "roadmap",
]

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINECODE_RE = re.compile(r"`[^`\n]+`")


def build_stem_map() -> dict[str, Path]:
    stem_map = {}
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.md"):
            stem = p.stem
            if stem not in stem_map:
                stem_map[stem] = p
    return stem_map


def resolve_target(token: str, stem_map: dict[str, Path]) -> Path | None:
    """token 可能是 'stem' 或 'subdir/stem' 形式。"""
    token = token.strip()
    if "/" in token:
        # 按字面路径解析
        candidate = (REPO_ROOT / (token if token.endswith(".md") else token + ".md"))
        if candidate.exists():
            return candidate.resolve()
        return None
    # stem 查找
    return stem_map.get(token)


def replace_wikilinks_in_text(text: str, source: Path, stem_map: dict[str, Path]) -> tuple[str, list[dict]]:
    """跳过代码块/行内代码，替换其余 [[...]]。"""
    # 用占位符把代码块和行内代码暂时替换出来
    placeholders: dict[str, str] = {}

    def stash(match: re.Match) -> str:
        key = f"\x00PLH{len(placeholders)}\x00"
        placeholders[key] = match.group(0)
        return key

    safe = CODEBLOCK_RE.sub(stash, text)
    safe = INLINECODE_RE.sub(stash, safe)

    changes = []

    def replace(m: re.Match) -> str:
        token = m.group(1).strip()
        alias = m.group(2)
        # 过滤纯数字/矩阵
        if re.match(r"^[\d\.\-\s,]+$", token):
            return m.group(0)
        target = resolve_target(token, stem_map)
        if target is None:
            changes.append({"match": m.group(0), "status": "BROKEN_NO_TARGET"})
            return m.group(0)  # 保留原样，后续报错
        # 计算最短相对路径（从 source 文件所在目录到 target）
        rel_str = os.path.relpath(target, source.parent).replace("\\", "/")
        # 链接文本：alias > basename
        text_label = alias.strip() if alias else (token.split("/")[-1])
        new_link = f"[{text_label}]({rel_str})"
        changes.append({
            "match": m.group(0),
            "status": "OK",
            "new": new_link,
            "target": str(target.relative_to(REPO_ROOT)),
        })
        return new_link

    safe = WIKILINK_RE.sub(replace, safe)

    # 还原占位符
    for key, original in placeholders.items():
        safe = safe.replace(key, original)

    return safe, changes


def main() -> int:
    stem_map = build_stem_map()
    total_changes = 0
    files_modified = 0
    broken = []
    for p in sorted(WIKI_DIR.rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        new_text, changes = replace_wikilinks_in_text(text, p, stem_map)
        if not changes:
            continue
        ok = [c for c in changes if c["status"] == "OK"]
        bad = [c for c in changes if c["status"] != "OK"]
        if bad:
            broken.extend((str(p), b["match"]) for b in bad)
        if ok and new_text != text:
            p.write_text(new_text, encoding="utf-8")
            files_modified += 1
            total_changes += len(ok)
            print(f"✓ {p.relative_to(REPO_ROOT)}: {len(ok)} 处替换")

    print()
    print(f"修改文件 {files_modified} 个，替换 {total_changes} 处 wikilink。")
    if broken:
        print(f"\n⚠️ 仍有 {len(broken)} 处无法解析（保留原样）：")
        for f, m in broken:
            print(f"  {f}: {m}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
