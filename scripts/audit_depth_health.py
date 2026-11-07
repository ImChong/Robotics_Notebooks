#!/usr/bin/env python3
"""Audit Issue #3490 against actual depth Stage links and the existing health rules.

Default: print a Markdown audit to stdout (no writes).
--write: refresh docs/quality/depth-health-backlog.{md,csv}.
--latest N: inspect only the N most recent *historical* depth topics (README reverse order).
Never changes health scoring or marks a source as fully read.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_link_graph as graph  # noqa: E402
from original_reading import original_entries  # noqa: E402

WIKI_ROOT = (ROOT / "wiki").resolve()
ROUTE_PATTERN = re.compile(r"\((depth-[^()]+\.md)\)")
STAGE_PATTERN = re.compile(r"^##\s+(Stage\s+\d+[^\n]*)\s*$")
LINK_PATTERN = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")
STATUS_PATTERN = re.compile(r"^status\s*:\s*['\"]?([\w-]+)", re.MULTILINE)


def ordered_depth_routes() -> list[Path]:
    """README is oldest -> newest by topic's starting milestone, not file edit date."""
    listed = ROUTE_PATTERN.findall((ROOT / "roadmap" / "README.md").read_text(encoding="utf-8"))
    seen: set[str] = set()
    ordered: list[Path] = []
    for name in listed:
        if name not in seen:
            seen.add(name)
            ordered.append(ROOT / "roadmap" / name)
    # Include routes missed by the README so auditing cannot silently omit them.
    for extra in sorted((ROOT / "roadmap").glob("depth-*.md")):
        if extra.name not in seen:
            ordered.append(extra)
    return list(reversed(ordered))


def stage_wiki_links(route: Path) -> tuple[str, dict[str, list[str]]]:
    """Only knowledge links inside actual Stage sections count, not footer links."""
    content = route.read_text(encoding="utf-8")
    stages: dict[str, list[str]] = {}
    stage: str | None = None
    for line in content.splitlines():
        heading = STAGE_PATTERN.match(line)
        if heading:
            stage = heading.group(1)
            stages.setdefault(stage, [])
            continue
        if line.startswith("## "):
            stage = None
        if stage is None:
            continue
        for match in LINK_PATTERN.finditer(line):
            href = match.group(1).split("#", 1)[0]
            if href.startswith(("http:", "https:")):
                continue
            target = (route.parent / href).resolve()
            if not target.is_relative_to(WIKI_ROOT):
                continue
            rel = target.relative_to(ROOT).as_posix()
            if rel not in stages[stage]:
                stages[stage].append(rel)
    return content, stages


def check_wiki(rel_path: str) -> dict:
    path = ROOT / rel_path
    if not path.is_file():
        return {"score": -1, "status": "missing", "reason": "节点文件不存在"}
    content = path.read_text(encoding="utf-8")
    score, gaps = graph.compute_health(content, path)
    status_match = STATUS_PATTERN.search(content.split("---", 2)[1] if content.startswith("---") else "")
    status = status_match.group(1) if status_match else "unknown"
    reasons: list[str] = []
    if gaps & 1:
        reasons.append("缺摘要/一手来源")
    if gaps & 2:
        reasons.append("阅读骨架不完整")
    if gaps & 4:
        entries = original_entries(content)
        reasons.append("缺 original 归档" if not entries else "original 归档未全部标记通读完成")
    if gaps & 8:
        reasons.append("status 限分")
    return {"score": score, "status": status, "reason": "；".join(reasons) or "—"}


def collect(latest: int = 0) -> tuple[list[dict], dict[str, dict], list[dict]]:
    routes = ordered_depth_routes()
    if latest > 0:
        routes = routes[:latest]
    records: list[dict] = []
    nodes: dict[str, dict] = {}
    route_stats: list[dict] = []
    for route in routes:
        if not route.is_file():
            continue
        content, stages = stage_wiki_links(route)
        links = graph.extract_internal_links(content, route)
        health, route_gaps = graph.compute_roadmap_health(content, len(links))
        unique: set[str] = set()
        for stage, paths in stages.items():
            for rel in paths:
                unique.add(rel)
                if rel not in nodes:
                    nodes[rel] = check_wiki(rel)
                records.append({
                    "roadmap": route.relative_to(ROOT).as_posix(),
                    "stage": stage,
                    "node": rel,
                    **nodes[rel],
                })
        count = Counter(nodes[rel]["score"] for rel in unique)
        route_stats.append({
            "roadmap": route.relative_to(ROOT).as_posix(),
            "route_health": f"{health}/3",
            "route_gaps": route_gaps,
            "stages": len(stages),
            "unique": len(unique),
            "by_score": count,
            "full": count[3],
            "pending": len(unique) - count[3],
        })
    return records, nodes, route_stats


def markdown(records: list[dict], nodes: dict[str, dict], routes: list[dict]) -> str:
    total = Counter(v["score"] for v in nodes.values())
    lines = [
        "# Issue #3490：纵深路线 Stage 节点健康度清单",
        "",
        "> 按 `roadmap/README.md` 中的**历史起点顺序逆序**（最新纵深主题优先）。",
        "> 数据来自实际 Stage 链接与 `generate_link_graph.compute_health`、",
        "> `compute_roadmap_health`，不改分值规则；同一 wiki 节点全局只计一次。",
        "",
        f"- 路线：**{len(routes)}**；关联唯一 wiki 节点：**{len(nodes)}**",
        f"- Wiki 健康度：**3/3 {total[3]}**，**2/3 {total[2]}**，**1/3 {total[1]}**，**0/3 {total[0]}**，缺失文件 **{total[-1]}**",
        f"- 待推进：**{len(nodes) - total[3]}**（含缺失文件）",
        "",
        "## 路线进度（从最新主题向前）",
        "",
        "| 纵深路线 | 路线健康度 | Stage | 唯一 wiki | 3/3 | 2/3 | 1/3 | 0/3 | 缺失 | 待办 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in routes:
        scores = r["by_score"]
        lines.append(
            f'| [`{r["roadmap"]}`](../../{r["roadmap"]}) | {r["route_health"]} | '
            f'{r["stages"]} | {r["unique"]} | {scores[3]} | {scores[2]} | '
            f'{scores[1]} | {scores[0]} | {scores[-1]} | {r["pending"]} |'
        )
    by_route: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_route[record["roadmap"]][record["stage"]].append(record)
    lines += ["", "## Stage → 节点 → 分值 → 缺项（保留重复引用关系）", ""]
    for r in routes:
        route = r["roadmap"]
        lines.append(f"### {route}（路线 {r['route_health']}，节点 3/3: {r['full']}/{r['unique']}）")
        lines.append("")
        for stage, entries in by_route[route].items():
            lines.extend([f"#### {stage}", "", "| 节点 | 分值 | 状态 | 缺项原因 |", "|---|---:|---|---|"])
            for item in entries:
                score = "缺失" if item["score"] < 0 else f'{item["score"]}/3'
                lines.append(f'| [`{item["node"]}`](../../{item["node"]}) | {score} | '
                             f'{item["status"]} | {item["reason"]} |')
            lines.append("")
    lines += [
        "## 验收说明",
        "",
        "- `original` 只接受实际已经通读并留有可核验来源记录的档案；不能凭补 frontmatter 伪造通读。",
        "- 本表是机器规则的状态快照，不等于语义质量审查；有未读章节或有疑问的来源需人工逐项复核。",
        "- 全量跑法：`python3 scripts/audit_depth_health.py --write`；单看最新主题：`python3 scripts/audit_depth_health.py --latest 1`。",
        "- 新增/调整纵深路线、Stage 引用或 wiki 后应重跑；待所有节点满分且证据可信才可关闭 #3490。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=int, default=0, help="Limit to newest N historical depth topics")
    parser.add_argument("--write", action="store_true", help="Write full Markdown and CSV backlog")
    args = parser.parse_args()
    if args.latest < 0 or (args.write and args.latest):
        parser.error("--latest must be >= 0; --write requires the full audit")
    records, nodes, routes = collect(args.latest)
    report = markdown(records, nodes, routes)
    if args.write:
        out = ROOT / "docs" / "quality"
        out.mkdir(parents=True, exist_ok=True)
        (out / "depth-health-backlog.md").write_text(report + "\n", encoding="utf-8")
        with (out / "depth-health-backlog.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["roadmap", "stage", "node", "score", "status", "reason"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Backlog updated: {len(routes)} routes, {len(nodes)} unique nodes")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
