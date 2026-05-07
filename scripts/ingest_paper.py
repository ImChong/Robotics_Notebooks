#!/usr/bin/env python3
"""
ingest_paper.py — 快速生成 sources/papers/ 新条目模板

用法:
    python3 scripts/ingest_paper.py <文件名(无.md)> [--title "标题"] [--desc "一句话说明"] [--suggest-updates]

示例:
    python3 scripts/ingest_paper.py diffusion_policy --title "Diffusion Policy 论文" --desc "覆盖扩散策略" --suggest-updates

    生成: sources/papers/diffusion_policy.md
    --suggest-updates：分析标题/描述关键词，输出可能需要更新的 wiki 页面列表
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

TEMPLATE = """\
# {stem}

> 来源归档（ingest）

- **标题：** {title}
- **类型：** paper
- **来源：** arXiv / NeurIPS / ICLR / ICML
- **入库日期：** {today}
- **最后更新：** {today}
- **一句话说明：** {desc}

## 核心论文摘录（MVP）

### 1) [论文标题]（[作者], [年份]）
- **链接：** <https://arxiv.org/abs/XXXX.XXXXX>
- **核心贡献：** TODO — 请填写核心贡献
- **对 wiki 的映射：**
  - [TODO](../../wiki/methods/TODO.md)

### 2) [论文标题]（[作者], [年份]）
- **链接：** <https://arxiv.org/abs/XXXX.XXXXX>
- **核心贡献：** TODO — 请填写核心贡献
- **对 wiki 的映射：**
  - [TODO](../../wiki/concepts/TODO.md)

## 当前提炼状态

- [ ] 论文摘要填写
- [ ] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
"""


def suggest_wiki_updates(title: str, desc: str, repo_root: Path) -> None:
    """分析标题/描述关键词，输出 wiki/ 中可能需要更新的页面。"""
    # 从标题和描述中提取关键词（去停用词）
    STOP = {
        "the",
        "a",
        "an",
        "for",
        "of",
        "in",
        "on",
        "with",
        "and",
        "or",
        "to",
        "from",
        "is",
        "are",
        "by",
        "at",
        "as",
        "its",
        "via",
        "的",
        "了",
        "和",
        "与",
        "或",
        "在",
        "中",
        "上",
        "下",
        "对",
        "等",
    }
    combined = (title + " " + desc).lower()
    # 提取英文单词和中文词
    en_words = re.findall(r"\b[a-z]{3,}\b", combined)
    zh_words = re.findall(r"[\u4e00-\u9fff]{2,}", combined)
    keywords = set(en_words + zh_words) - STOP

    if not keywords:
        print("⚠️  未能从标题/描述中提取关键词，跳过 suggest-updates。")
        return

    wiki_dir = repo_root / "wiki"
    matches: dict[str, list[str]] = {}
    for page in sorted(wiki_dir.rglob("*.md")):
        content = page.read_text(encoding="utf-8").lower()
        hit_kw = [kw for kw in keywords if kw in content]
        if len(hit_kw) >= 2:
            rel = str(page.relative_to(repo_root))
            matches[rel] = hit_kw[:5]

    if matches:
        print("\n💡 --suggest-updates: 以下 wiki 页面可能需要根据新来源更新（匹配关键词 ≥ 2）：")
        for path, kws in sorted(matches.items())[:10]:
            print(f"   • {path}  （命中: {', '.join(kws)}）")
        if len(matches) > 10:
            print(f"   … 共 {len(matches)} 个页面，仅显示前 10 个")
    else:
        print("💡 --suggest-updates: 未找到高度相关的 wiki 页面（关键词匹配 < 2）。")


def main():
    parser = argparse.ArgumentParser(description="生成 sources/papers/ 条目模板")
    parser.add_argument("stem", help="文件名（不含 .md），例如 diffusion_policy")
    parser.add_argument("--title", default="", help="论文集合标题")
    parser.add_argument("--desc", default="TODO — 请填写一句话说明", help="一句话说明")
    parser.add_argument(
        "--suggest-updates",
        action="store_true",
        help="分析关键词，输出可能需要更新的 wiki 页面列表",
    )
    args = parser.parse_args()

    stem = args.stem.strip()
    if not stem:
        print("错误：文件名不能为空", file=sys.stderr)
        sys.exit(1)

    title = args.title or stem.replace("_", " ").title()
    today = date.today().isoformat()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "sources" / "papers" / f"{stem}.md"

    if out_path.exists():
        print(f"⚠️  文件已存在：{out_path}，跳过生成。", file=sys.stderr)
        sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = TEMPLATE.format(stem=stem, title=title, desc=args.desc, today=today)
    out_path.write_text(content, encoding="utf-8")

    print(f"✅ 已生成：{out_path}")
    print("   请编辑文件，填写论文摘录后运行 make lint 验证")

    if args.suggest_updates:
        suggest_wiki_updates(title, args.desc, repo_root)


if __name__ == "__main__":
    main()
