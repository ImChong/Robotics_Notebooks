#!/usr/bin/env python3
"""scaffold_wiki_page.py — 按全库 frontmatter 规范生成 wiki 页面骨架。

把 query 答案 / 选题候选沉淀回 wiki 时，手工拼 frontmatter、速查区块与统一
正文骨架成本不低。本脚本给定 type 与标题，输出符合 lint 结构要求的骨架：
定义 → 英文缩写 → 重要性 → 核心原理 → 工程实践 → 局限 → 关联与来源。

用法：
    python3 scripts/scaffold_wiki_page.py concept "视觉伺服" --slug visual-servoing-intro
    python3 scripts/scaffold_wiki_page.py query "机器人感知选型" --slug perception-pick --dry-run
    python3 scripts/scaffold_wiki_page.py entity "AMASS" --slug amass --dataset --dry-run
    python3 scripts/scaffold_wiki_page.py entity "YAHMP" --slug paper-yahmp --paper --dry-run

`--dataset`（仅 entity 类型）额外附「规模 / 模态 / 许可证 / 适配形态 / 重定向就绪度」
速查块并写入 `dataset` tag，使新建数据集页直接通过 lint 的数据集元数据巡检。
自带 `--dry-run`（只打印不落盘）与生成后结构自检（复用 lint_wiki 判据）。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

from lint_wiki import has_section, word_count
from wiki_abbrev_section import is_abbrev_glossary_well_placed

REPO_ROOT = Path(__file__).resolve().parents[1]

# type → wiki 子目录
TYPE_DIRS: dict[str, str] = {
    "concept": "concepts",
    "method": "methods",
    "task": "tasks",
    "comparison": "comparisons",
    "query": "queries",
    "formalization": "formalizations",
    "entity": "entities",
    "overview": "overview",
}


def slugify(title: str, override: str | None) -> str:
    """从标题或 --slug 生成 kebab-case slug；纯中文标题需显式 --slug。"""
    if override:
        return override.strip().lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return ascii_only


def _frontmatter(page_type: str, title: str, dataset: bool = False) -> str:
    today = date.today().isoformat()
    tags = "  - dataset\n  - TODO\n" if dataset else "  - TODO\n"
    return (
        "---\n"
        f"type: {page_type}\n"
        "tags:\n"
        f"{tags}"
        "status: draft\n"
        f"updated: {today}\n"
        f'summary: "TODO：一句话概括「{title}」是什么、为什么重要。"\n'
        "related:\n"
        "  - ../TODO/related-page.md\n"
        "sources:\n"
        "  - ../../sources/TODO/source.md\n"
        "---\n"
    )


_ABBREV_BLOCK = (
    "## 英文缩写速查\n\n"
    "| 缩写 | 英文全称 | 简要说明 |\n"
    "|------|----------|----------|\n"
    "| TODO | TODO Full Name | TODO 待补全 |\n"
)

_TAIL_BLOCK = (
    "## 关联页面\n\n"
    "- [TODO 相关页面标题](../TODO/related-page.md)\n\n"
    "## 参考来源\n\n"
    "- [TODO 来源标题](../../sources/TODO/source.md)\n\n"
    "## 推荐继续阅读\n\n"
    "- [TODO 外部资料标题](https://example.com)\n"
)


STANDARD_SECTION_ORDER = (
    "## 一句话定义",
    "## 英文缩写速查",
    "## 为什么重要",
    "## 核心原理",
    "## 工程实践",
    "## 局限与风险",
    "## 关联页面",
    "## 参考来源",
    "## 推荐继续阅读",
)


# 数据集实体速查块：五维度覆盖 lint `dataset_metadata_check` 全部关键词命中，
# 新建数据集页即满足元数据巡检（0 缺失），降低后续 ingest 手工拼写成本。
_DATASET_QUICK_BLOCK = (
    "## 数据集速查\n\n"
    "| 维度 | 速查 |\n"
    "|------|------|\n"
    "| 规模 | TODO：序列 / 帧 / 小时 / 被试 / 轨迹等量级（如 X 小时、Y 条序列、Z 名被试）。 |\n"
    "| 模态 | TODO：动捕 / 视频 / 深度 / 点云 / SMPL / IMU 等数据模态。 |\n"
    "| 许可证 | TODO：开源协议 / 注册要求 / 商用约束（如 CC-BY / MIT / non-commercial）。 |\n"
    "| 适配形态 | TODO：面向的机器人形态与骨架（人形 / 机械臂 / 通用），与本体的形态差距。 |\n"
    "| 重定向就绪度 | TODO：能否直接作训练输入，还是需重定向；是否物理可行 / 可部署。 |\n"
)


def _body_dataset(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## 一句话定义\n\n"
        f"**{title}**：TODO 一句话核心定义（这是什么数据集、为机器人学习提供什么）。\n\n"
        f"{_ABBREV_BLOCK}\n"
        f"{_DATASET_QUICK_BLOCK}\n"
        "## 为什么重要\n\n"
        "- **TODO 要点一：** 待补全。\n"
        "- **TODO 要点二：** 待补全。\n\n"
        "## 核心原理\n\n"
        "TODO 说明数据的**采集方式 → 数据构成 → 标注与处理流程**。\n\n"
        "## 工程实践\n\n"
        "TODO 说明下载、清洗、重定向、训练接入与评测方式。\n\n"
        "## 局限与风险\n\n"
        "- **适用边界：** TODO 说明数据覆盖不到的场景或形态。\n"
        "- **工程风险：** TODO 说明许可证、数据偏差或部署差距。\n\n"
        f"{_TAIL_BLOCK}"
    )


def _body_standard(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## 一句话定义\n\n"
        f"**{title}**：TODO 一句话核心定义（在机器人语境下它解决什么问题）。\n\n"
        f"{_ABBREV_BLOCK}\n"
        "## 为什么重要\n\n"
        "- **TODO 要点一：** 待补全。\n"
        "- **TODO 要点二：** 待补全。\n\n"
        "## 核心原理\n\n"
        "TODO 说明**输入 → 关键机制 → 输出**，必要时补公式、流程图或伪代码。\n\n"
        "## 工程实践\n\n"
        "TODO 说明实现步骤、关键参数、调试指标与机器人应用示例。\n\n"
        "## 局限与风险\n\n"
        "- **适用边界：** TODO 说明什么情况下不适用。\n"
        "- **工程风险：** TODO 说明稳定性、实时性、数据或部署风险。\n\n"
        f"{_TAIL_BLOCK}"
    )


def _body_query(title: str) -> str:
    return (
        f"# Query：{title}\n\n"
        "> **Query 产物**：本页由以下问题触发：「TODO 触发问题」\n"
        "> 综合来源：[TODO 页面](../TODO/related-page.md)\n\n"
        f"{_ABBREV_BLOCK}\n"
        "## TL;DR 决策结论\n\n"
        "- TODO 第一刀砍在哪。\n"
        "- TODO 第二刀砍在哪。\n\n"
        "## 详细分析\n\n"
        "TODO 分类对比 / 决策树 / 推荐组合，逐段补全。\n\n"
        "## 关联页面\n\n"
        "- [TODO 相关页面标题](../TODO/related-page.md)\n\n"
        "## 参考来源\n\n"
        "- [TODO 来源标题](../../sources/TODO/source.md)\n"
    )


def _body_paper(title: str) -> str:
    """论文实体骨架：含评测 / 结论 / 对比 / 源码时序图占位，对齐 page-types 附加要求。"""
    return (
        f"# {title}\n\n"
        "## 一句话定义\n\n"
        f"**{title}**：TODO 一句话核心定义（方法主张与平台）。\n\n"
        f"{_ABBREV_BLOCK}\n"
        "## 为什么重要\n\n"
        "- **TODO 要点一：** 待补全。\n"
        "- **TODO 要点二：** 待补全。\n\n"
        "## 核心信息\n\n"
        "| 项 | 内容 |\n"
        "|----|------|\n"
        "| **机构** | TODO |\n"
        "| **平台** | TODO |\n"
        "| **开源** | TODO：已开源 / 部分 / 未开源 |\n\n"
        "## 核心原理\n\n"
        "### 方法栈\n\n"
        "TODO 模块表或分节。\n\n"
        "### 流程总览\n\n"
        "TODO：可用 mermaid flowchart。\n\n"
        "## 源码运行时序图\n\n"
        "TODO：已开源则画 sequenceDiagram；否则写 **不适用**（原因）。\n\n"
        "## 工程实践\n\n"
        "TODO 复现入口、关键超参、部署注意。\n\n"
        "## 实验与评测\n\n"
        "- TODO 主指标与设定。\n\n"
        "## 结论\n\n"
        "**TODO 一句话总判。**\n\n"
        "1. **要点一** — TODO。\n"
        "2. **要点二** — TODO。\n"
        "3. **要点三** — TODO。\n\n"
        "## 与其他工作对比\n\n"
        "| 对照 | 差异读法 |\n"
        "|------|----------|\n"
        "| TODO | TODO |\n\n"
        "## 局限与风险\n\n"
        "- **适用边界：** TODO。\n"
        "- **工程风险：** TODO。\n\n"
        f"{_TAIL_BLOCK}"
    )


def build_skeleton(
    page_type: str, title: str, dataset: bool = False, paper: bool = False
) -> str:
    if dataset:
        return _frontmatter(page_type, title, dataset=True) + "\n" + _body_dataset(title)
    if paper:
        fm = _frontmatter(page_type, title)
        # 论文页默认补 arxiv 占位，便于过 paper 元数据巡检
        fm = fm.replace(
            "status: draft\n",
            'status: draft\narxiv: "TODO"\n',
            1,
        )
        return fm + "\n" + _body_paper(title)
    body = _body_query(title) if page_type == "query" else _body_standard(title)
    return _frontmatter(page_type, title) + "\n" + body


def self_check(content: str, page_type: str) -> list[str]:
    """复用 lint_wiki 判据对骨架做结构自检；返回问题列表（空=通过）。"""
    problems: list[str] = []
    fm = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm:
        problems.append("缺 frontmatter 区块")
    else:
        for key in ("type", "updated", "summary", "related", "sources"):
            if not re.search(rf"^{key}\s*:", fm.group(1), re.MULTILINE):
                problems.append(f"frontmatter 缺 {key}")

    if not has_section(content, ["英文缩写速查"]):
        problems.append("缺「英文缩写速查」区块")
    elif not is_abbrev_glossary_well_placed(content):
        problems.append("「英文缩写速查」位置不规范")
    if not has_section(content, ["关联", "related"]):
        problems.append("缺「关联页面」区块")
    if not has_section(content, ["参考来源", "sources"]):
        problems.append("缺「参考来源」区块")

    if page_type != "query":
        positions: list[int] = []
        for heading in STANDARD_SECTION_ORDER:
            pos = content.find(heading)
            if pos == -1:
                problems.append(f"缺「{heading.removeprefix('## ')}」区块")
            positions.append(pos)
        if all(pos >= 0 for pos in positions) and positions != sorted(positions):
            problems.append("统一阅读骨架顺序不规范")

    if page_type == "query":
        for needle, label in (
            ("**Query 产物**", "Query 产物说明"),
            ("## 参考来源", "## 参考来源"),
            ("## 关联页面", "## 关联页面"),
        ):
            if needle not in content:
                problems.append(f"query 页缺 {label}")

    if "arxiv:" in content or re.search(r"^##\s+结论\b", content, re.M):
        # 论文实体骨架 / 含结论的实体：额外自检
        if not has_section(content, ["结论"]):
            problems.append("论文实体缺「结论」区块")
        if not has_section(content, ["评测", "实验"]):
            problems.append("论文实体缺「实验与评测」类区块")
        if not has_section(content, ["与其他工作", "对比", "比较"]):
            problems.append("论文实体缺「与其他工作对比」类区块")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成符合 lint 规范的 wiki 页面骨架")
    parser.add_argument("type", choices=sorted(TYPE_DIRS), help="页面类型")
    parser.add_argument("title", help="页面标题（h1 与 summary 中使用）")
    parser.add_argument("--slug", help="文件名 slug（纯中文标题必填）")
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="数据集实体骨架：附「规模/模态/许可证/适配形态/重定向就绪度」速查块（仅 entity 类型）",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="论文实体骨架：含实验与评测 / 结论 / 对比 / 源码时序图占位（仅 entity；建议 --slug paper-...）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印骨架，不落盘")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在文件")
    args = parser.parse_args(argv)

    if args.dataset and args.type != "entity":
        print("✗ --dataset 仅适用于 entity 类型", file=sys.stderr)
        return 2
    if args.paper and args.type != "entity":
        print("✗ --paper 仅适用于 entity 类型", file=sys.stderr)
        return 2
    if args.dataset and args.paper:
        print("✗ --dataset 与 --paper 互斥", file=sys.stderr)
        return 2

    slug = slugify(args.title, args.slug)
    if not slug:
        print("✗ 无法从标题推断 slug（纯中文？请用 --slug 指定）", file=sys.stderr)
        return 2

    content = build_skeleton(
        args.type, args.title, dataset=args.dataset, paper=args.paper
    )

    problems = self_check(content, args.type)
    wc = word_count(content)

    target = REPO_ROOT / "wiki" / TYPE_DIRS[args.type] / f"{slug}.md"

    if args.dry_run:
        print(content)
        print(f"--- 自检：{'通过' if not problems else '发现问题'} ---", file=sys.stderr)
    if problems:
        for p in problems:
            print(f"✗ 结构自检：{p}", file=sys.stderr)
        return 1

    print(
        f"✓ 结构自检通过（骨架 {wc} 字，待补全正文以超过 200 字下限）",
        file=sys.stderr,
    )

    if args.dry_run:
        print(f"（--dry-run：未写入 {target.relative_to(REPO_ROOT)}）", file=sys.stderr)
        return 0

    if target.exists() and not args.force:
        print(f"✗ 已存在 {target.relative_to(REPO_ROOT)}（--force 可覆盖）", file=sys.stderr)
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"✓ 已生成 {target.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
