#!/usr/bin/env python3
"""Audit Stage-linked wiki health across all depth roadmaps (Issue #3490).

Reports existing scores; never mutates sources, statuses, or scoring rules.
Usage: python3 scripts/report_depth_health_backlog.py [--output path.md]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import generate_link_graph as graph

ROOT = Path(__file__).resolve().parent.parent
ROADMAPS = ROOT / "roadmap"
WIKI = ROOT / "wiki"
STAGE_RE = re.compile(r"^## (Stage\s+\d+[^\n]*)$", re.MULTILINE)
SECTION_RE = re.compile(r"^## [^\n]*$", re.MULTILINE)
LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")


def roadmap_order() -> list[Path]:
    """README reading order is oldest to newest; process newest first."""
    overview = (ROADMAPS / "README.md").read_text(encoding="utf-8")
    ordered = []
    for match in re.finditer(r"\]\((depth-[^)]+\.md)\)", overview):
        path = ROADMAPS / match.group(1)
        if path.is_file() and path not in ordered:
            ordered.append(path)
    unlisted = sorted(set(ROADMAPS.glob("depth-*.md")) - set(ordered))
    return list(reversed(ordered)) + unlisted


def stage_links(path: Path, content: str) -> list[tuple[str, Path]]:
    """Stage-only links, excluding unrelated footer and cross-roadmap links."""
    result = []
    for heading in STAGE_RE.finditer(content):
        start = heading.end()
        next_section = SECTION_RE.search(content, start)
        body = content[start : next_section.start() if next_section else None]
        for link in LINK_RE.finditer(body):
            href = link.group(1)
            target = (path.parent / href).resolve()
            if target.is_relative_to(WIKI):
                result.append((heading.group(1), target))
    return result


def count_roadmap_links(path: Path, content: str) -> int:
    """Same existing local-link extractor used by link-graph scoring."""
    return len(graph.extract_internal_links(content, path))


def build_report() -> tuple[str, dict[str, int]]:
    routes = roadmap_order()
    associations = defaultdict(list)
    route_summary = []
    broken = []
    for route in routes:
        body = route.read_text(encoding="utf-8")
        rows = stage_links(route, body)
        targets = set()
        for stage, path in rows:
            if not path.is_file():
                broken.append((route.relative_to(ROOT).as_posix(), stage, path))
                continue
            targets.add(path)
            associations[path].append((route.relative_to(ROOT).as_posix(), stage))
        score, gaps = graph.compute_roadmap_health(
            body, count_roadmap_links(route, body)
        )
        route_summary.append((route, score, gaps, targets))

    status = {}
    for path in associations:
        score, gaps = graph.compute_health(
            path.read_text(encoding="utf-8"), path
        )
        status[path] = (score, gaps)
    counts = Counter(score for score, _ in status.values())
    route_counts = Counter(score for _, score, _, _ in route_summary)
    remaining = sum(v for score, v in counts.items() if score < 3)
    summary = {
        "routes": len(routes),
        "unique_nodes": len(status),
        "route_3": route_counts[3],
        "node_0": counts[0],
        "node_1": counts[1],
        "node_2": counts[2],
        "node_3": counts[3],
        "remaining": remaining,
        "broken_links": len(broken),
    }

    out = [
        "# #3490 纵深路线健康度基线（按 README 逆序）",
        "",
        "> 运行 `python3 scripts/report_depth_health_backlog.py --output "
        "docs/plans/depth-health-issue-3490.md` 生成。"
        " 此报告为静态评分与缺项索引，不能替代真实原文通读。",
        "",
        "## 唯一节点总览",
        "",
        f"- 纵深路线：{len(routes)}，路线页 3/3：{route_counts[3]}。",
        f"- Stage 关联唯一 Wiki 节点：{len(status)}。",
        f"- Wiki 0/1/2/3 分：{counts[0]} / {counts[1]} / "
        f"{counts[2]} / {counts[3]}，待提升：{remaining}。",
        f"- 失效 Stage 链接：{len(broken)}。",
        "",
        "## 路线覆盖率（最新里程碑路线优先）",
        "",
        "| 路线 | 路线健康度 | Stage 唯一节点 | 3/3 | 待提升 | 覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, score, _, nodes in route_summary:
        done = sum(status[node][0] == 3 for node in nodes)
        total = len(nodes)
        pct = f"{done / total:.1%}" if total else "—"
        out.append(
            f"| `{route.relative_to(ROOT).as_posix()}` | {score}/3 "
            f"| {total} | {done} | {total - done} | {pct} |"
        )

    out += [
        "",
        "## 不达标的唯一节点（同一页面只维护一次）",
        "",
        "| 节点路径 | 当前评分 | 缺项 | 关联路线 / Stage |",
        "|---|---:|---|---|",
    ]
    gap_names = ((1, "元数据"), (2, "阅读骨架"), (4, "原文通读"), (8, "状态未 complete"))
    for path in sorted(status, key=lambda p: p.as_posix()):
        score, gaps = status[path]
        if score == 3:
            continue
        missing = "、".join(label for flag, label in gap_names if gaps & flag) or "—"
        refs = "；".join(
            f"`{roadmap}` {stage}" for roadmap, stage in associations[path]
        )
        out.append(
            f"| `{path.relative_to(ROOT).as_posix()}` | "
            f"{score}/3 | {missing} | {refs} |"
        )
    if broken:
        out += ["", "## 失效的 Stage 链接", ""]
        for route, stage, missing in broken:
            out.append(
                f"- `{route}` {stage}: `{missing.relative_to(ROOT)}`"
            )
    out += [
        "",
        "## 下一批优先顺序",
        "",
        "按 `roadmap/README.md` 里程碑历史排序**逆序**处理，"
        "先 WAM，再具身数据 / Real2Sim / VLA / Diffusion 等后续路线；"
        "共享节点仅在首次修复时记一次。",
        "",
        "当原文存在付费墙或未读完时保持缺项，禁止仅修改 `original` / "
        "`status` 冒充完成或修改评分阈值。",
        "",
    ]
    return "\n".join(out), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, stats = build_report()
    if args.output:
        destination = (ROOT / args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    else:
        print(report)
    print("Issue #3490 baseline:", stats)


if __name__ == "__main__":
    main()
