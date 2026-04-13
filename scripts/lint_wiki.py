#!/usr/bin/env python3
"""
lint_wiki.py — 自动化 wiki 健康检查脚本
基于 Karpathy LLM Wiki 模式，检测以下问题：
  1. 孤儿页（无其他 wiki 页面链接到它）
  2. 缺少"关联页面"或"关联"区块的页面
  3. 缺少"参考来源"区块的页面
  4. 内链断链（链接目标文件不存在）
  5. 空壳页面（内容过少，< 200 字）

用法：
  python3 scripts/lint_wiki.py
  python3 scripts/lint_wiki.py --write-log   # 同时追加报告到 log.md
"""

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

# 只扫描 wiki/ 下的 markdown 文件
def get_wiki_pages() -> list[Path]:
    return sorted(WIKI_DIR.rglob("*.md"))

# 移除代码块（``` ... ``` 和 ` ... `），避免提取代码示例中的假链接
def strip_code_blocks(content: str) -> str:
    # 移除围栏代码块
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r'`[^`]+`', '', content)
    return content

# 从文件中提取所有内部链接目标（相对路径 .md 文件）
def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    targets = []
    content = strip_code_blocks(content)
    # 匹配 markdown 链接 [text](path)，只取 .md 文件且不是 http
    for match in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', content):
        href = match.group(2).strip()
        if href.startswith("http") or href.startswith("#"):
            continue
        # 去掉锚点
        href = href.split("#")[0]
        if not href.endswith(".md"):
            continue
        resolved = (source_path.parent / href).resolve()
        targets.append(resolved)
    return targets

def has_section(content: str, patterns: list[str]) -> bool:
    """检查是否存在某个 ## 级别的区块（匹配关键词）"""
    for pat in patterns:
        if re.search(rf'^##\s+.*{pat}', content, re.MULTILINE | re.IGNORECASE):
            return True
    return False

def word_count(content: str) -> int:
    """简单估算字数（中英文混合）"""
    # 中文字符 + 英文单词
    chinese = len(re.findall(r'[\u4e00-\u9fff]', content))
    english = len(re.findall(r'\b[a-zA-Z]+\b', content))
    return chinese + english

def lint() -> dict:
    pages = get_wiki_pages()
    page_set = {p.resolve() for p in pages}

    # 建立每个页面被哪些其他页面链接的索引
    inbound: dict[Path, list[Path]] = {p.resolve(): [] for p in pages}
    broken_links: dict[Path, list[str]] = {}

    for page in pages:
        content = page.read_text(encoding="utf-8")
        links = extract_internal_links(content, page)
        for target in links:
            if target in page_set:
                # wiki 内链：记录 inbound
                inbound[target].append(page.resolve())
            elif target.exists():
                # 链接到 references/ 等其他目录，文件存在，不算断链
                # 但这些页面不在 wiki 内，不参与 inbound 统计
                pass
            else:
                broken_links.setdefault(page.resolve(), []).append(
                    str(target.relative_to(REPO_ROOT)) if target.is_relative_to(REPO_ROOT) else str(target)
                )

    results = {
        "orphan_pages": [],
        "missing_related": [],
        "missing_sources": [],
        "broken_links": [],
        "stub_pages": [],
    }

    for page in pages:
        resolved = page.resolve()
        content = page.read_text(encoding="utf-8")
        rel = page.relative_to(REPO_ROOT)

        # 1. 孤儿页（README 和 index 类页面排除）
        if page.name.lower() not in ("readme.md", "index.md"):
            if not inbound.get(resolved):
                results["orphan_pages"].append(str(rel))

        # 2. 缺少关联页面区块（支持多种命名变体）
        related_patterns = ["关联", "related", "已有页面", "关系"]
        if not has_section(content, related_patterns):
            results["missing_related"].append(str(rel))

        # 3. 缺少参考来源区块
        if not has_section(content, ["参考来源", "sources", "参考"]):
            results["missing_sources"].append(str(rel))

        # 4. 断链
        if resolved in broken_links:
            for broken in broken_links[resolved]:
                results["broken_links"].append(f"{rel} → {broken}")

        # 5. 空壳页面（< 200 字）
        if word_count(content) < 200:
            results["stub_pages"].append(f"{rel} ({word_count(content)} 字)")

    return results

def format_report(results: dict) -> str:
    today = date.today().isoformat()
    lines = [f"## [{today}] lint | health-check | 自动化 wiki 健康检查", ""]

    total_issues = sum(len(v) for v in results.values())
    lines.append(f"共发现 **{total_issues}** 个问题：")
    lines.append("")

    sections = [
        ("orphan_pages",    "孤儿页（无入链）",            "⚠️"),
        ("missing_related", "缺少关联页面区块",            "⚠️"),
        ("missing_sources", "缺少参考来源区块",            "⚠️"),
        ("broken_links",    "断链（内链目标不存在）",       "❌"),
        ("stub_pages",      "空壳页面（< 200 字）",        "⚠️"),
    ]

    for key, label, icon in sections:
        items = results[key]
        lines.append(f"### {icon} {label}（{len(items)} 个）")
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- 无")
        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Robotics_Notebooks wiki lint 检查")
    parser.add_argument("--write-log", action="store_true", help="将结果追加到 log.md")
    args = parser.parse_args()

    print("正在扫描 wiki/ 目录...")
    results = lint()
    report = format_report(results)

    print(report)

    total = sum(len(v) for v in results.values())
    if total == 0:
        print("✅ 所有检查通过！")
        sys.exit(0)
    else:
        print(f"⚠️  共发现 {total} 个问题，请参考上方报告处理。")

    if args.write_log:
        log_path = REPO_ROOT / "log.md"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n---\n\n")
            f.write(report)
        print(f"\n已将报告追加到 {log_path}")

if __name__ == "__main__":
    main()
