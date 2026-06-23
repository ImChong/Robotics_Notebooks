#!/usr/bin/env python3
"""为可派生机构的 wiki 页写入 frontmatter `institutions:`。

复用 generate_link_graph.derive_node_institutions 的完整派生规则；仅当派生结果
超出当前 tags/显式 institutions 覆盖范围时才写入，且尊重非空显式 institutions 覆盖。

用法：
  python3 scripts/bump_wiki_institutions.py
  python3 scripts/bump_wiki_institutions.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_link_graph as glg  # noqa: E402


def _format_institutions_yaml(institutions: list[str]) -> str:
    if len(institutions) == 1:
        return f"institutions: [{institutions[0]}]"
    lines = ["institutions:"]
    lines.extend(f"  - {inst_id}" for inst_id in institutions)
    return "\n".join(lines)


def bump_institutions(path: Path, dry_run: bool = False) -> bool:
    content = path.read_text(encoding="utf-8")
    page_id = str(path.relative_to(REPO_ROOT))
    explicit = glg.parse_frontmatter_list(content, "institutions")
    if explicit:
        return False

    derived = glg.derive_node_institutions(content, page_id=page_id)
    if not derived:
        return False

    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        return False

    fm_block = fm_match.group(1).rstrip()
    inst_yaml = _format_institutions_yaml(derived)
    new_fm = fm_block + f"\n{inst_yaml}\n"
    new_content = f"---\n{new_fm}\n---" + content[fm_match.end() :]
    if new_content == content:
        return False
    if not dry_run:
        path.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="为 wiki 页补全 frontmatter institutions")
    parser.add_argument("--dry-run", action="store_true", help="只打印将修改的页面")
    args = parser.parse_args()

    changed: list[str] = []
    for page in sorted(WIKI_DIR.rglob("*.md")):
        if page.name == "README.md":
            continue
        if bump_institutions(page, dry_run=args.dry_run):
            changed.append(str(page.relative_to(REPO_ROOT)))

    if changed:
        action = "将写入" if args.dry_run else "已写入"
        print(f"{action} institutions → {len(changed)} 个 wiki 页")
        for rel in changed:
            print(f"  - {rel}")
    else:
        print("无需更新（无可派生机构或已覆盖）")


if __name__ == "__main__":
    main()
