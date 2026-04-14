#!/usr/bin/env python3
"""
ingest_paper.py — 快速生成 sources/papers/ 新条目模板

用法:
    python3 scripts/ingest_paper.py <文件名(无.md)> [--title "标题"] [--desc "一句话说明"]

示例:
    python3 scripts/ingest_paper.py diffusion_policy --title "Diffusion Policy 论文" --desc "覆盖扩散策略及其在机器人操作中的应用"

    生成: sources/papers/diffusion_policy.md
"""

import argparse
import sys
from pathlib import Path
from datetime import date

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


def main():
    parser = argparse.ArgumentParser(description="生成 sources/papers/ 条目模板")
    parser.add_argument("stem", help="文件名（不含 .md），例如 diffusion_policy")
    parser.add_argument("--title", default="", help="论文集合标题")
    parser.add_argument("--desc", default="TODO — 请填写一句话说明", help="一句话说明")
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
    print(f"   请编辑文件，填写论文摘录后运行 make lint 验证")


if __name__ == "__main__":
    main()
