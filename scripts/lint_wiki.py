#!/usr/bin/env python3
"""
lint_wiki.py — 自动化 wiki 健康检查脚本
基于 Karpathy LLM Wiki 模式，检测以下问题：
  1. 孤儿页（无其他 wiki 页面链接到它）
  2. 缺少"关联页面"或"关联"区块的页面
  3. 缺少"参考来源"区块的页面
  4. 内链断链（链接目标文件不存在）
  5. 空壳页面（内容过少，< 200 字）
  6. Sources 孤儿（sources/papers 中链接到不存在 wiki 页）
  7. 陈旧页面（sources 文件比对应 wiki 页新，需 review）
  8. 矛盾检测（同一概念在不同页面有相反描述）
  9. Frontmatter 缺少 type 字段（V8 新增）
 10. log.md 活跃度检查（V8 新增：最近 30 天无操作则警告）
 11. concepts/methods/tasks 缺少 summary/description 字段（V10 新增）
 12. formalizations/ 公式变量在正文是否有物理含义解释（V21 新增）

用法：
  python3 scripts/lint_wiki.py
  python3 scripts/lint_wiki.py --write-log   # 同时追加报告到 log.md
  python3 scripts/lint_wiki.py --report      # 保存 markdown 报告到 exports/lint-report.md
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
CANONICAL_FACTS_FILE = REPO_ROOT / "schema" / "canonical-facts.json"


def load_canonical_facts() -> dict:
    """加载 schema/canonical-facts.json 矛盾检测规则数据。

    历史上规则与 ~150 条事实数据写在 lint() 内联字典里，文件膨胀到 ~700 行
    数据 + ~300 行代码。外移到 JSON 后，新增/修改事实属于纯数据变更，
    不再需要改代码、过 ruff、动 imports。
    """
    if not CANONICAL_FACTS_FILE.exists():
        return {}
    return json.loads(CANONICAL_FACTS_FILE.read_text(encoding="utf-8"))


# 只扫描 wiki/ 下的 markdown 文件
def get_wiki_pages() -> list[Path]:
    return sorted(WIKI_DIR.rglob("*.md"))


# 移除代码块（``` ... ``` 和 ` ... `），避免提取代码示例中的假链接
def strip_code_blocks(content: str) -> str:
    # 移除围栏代码块
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r"`[^`]+`", "", content)
    return content


# 从文件中提取所有内部链接目标（相对路径 .md 文件）
def extract_internal_links(content: str, source_path: Path) -> list[Path]:
    targets = []
    content = strip_code_blocks(content)
    # 匹配 markdown 链接 [text](path)，只取 .md 文件且不是 http
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", content):
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
        if re.search(rf"^##\s+.*{pat}", content, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def word_count(content: str) -> int:
    """简单估算字数（中英文混合）"""
    # 中文字符 + 英文单词
    chinese = len(re.findall(r"[\u4e00-\u9fff]", content))
    english = len(re.findall(r"\b[a-zA-Z]+\b", content))
    return chinese + english


def has_source_reference(content: str) -> bool:
    """检查页面是否引用了 sources/ 下的原始资料。

    覆盖率统计关心的是 wiki 页是否能追溯到原始资料，而不是资料是否一定来自
    sources/papers/。repo、blog、note 等来源同样是有效 ingest 来源。
    """
    return bool(re.search(r"(?:\.\./)*sources/[^)\s]+\.md\b", content))


def strip_misconception_sections(content: str) -> str:
    """移除"常见误区/误区"区块，避免把辟谣内容误判为事实矛盾。"""
    lines = content.splitlines()
    kept = []
    skip_level = None
    heading_re = re.compile(r"^(#{2,6})\s+(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if any(key in title for key in ["常见误区", "误区", "misconception", "pitfall"]):
                skip_level = level
                continue
        if skip_level is None:
            kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# lint() 子检查函数
# ---------------------------------------------------------------------------


def _build_link_index(
    pages: list[Path], page_set: set[Path]
) -> tuple[dict[Path, list[Path]], dict[Path, list[str]]]:
    """建立每个页面被哪些其他页面链接的索引，并收集断链。"""
    inbound: dict[Path, list[Path]] = {p.resolve(): [] for p in pages}
    broken_links: dict[Path, list[str]] = {}

    for page in pages:
        content = page.read_text(encoding="utf-8")
        links = extract_internal_links(content, page)
        for target in links:
            if target in page_set:
                inbound[target].append(page.resolve())
            elif not target.exists():
                broken_links.setdefault(page.resolve(), []).append(
                    str(target.relative_to(REPO_ROOT))
                    if target.is_relative_to(REPO_ROOT)
                    else str(target)
                )
    return inbound, broken_links


def _empty_results() -> dict[str, Any]:
    """初始化空的检查结果字典。"""
    return {
        "orphan_pages": [],
        "missing_related": [],
        "missing_sources": [],
        "broken_links": [],
        "stub_pages": [],
        "missing_pages": [],
        "broken_source_refs": [],
        "sources_orphans": [],
        "stale_pages": [],
        "outdated_pages": [],
        "contradictions": [],
        "missing_type": [],
        "log_inactive": [],
        "missing_summary": [],
        "query_format": [],
        "formalization_no_formula": [],
        "formalization_unexplained_vars": [],
        "readme_badge": [],
        "orphan_count": [],
        "method_missing_link": [],
        "method_missing_sections": [],
        "entity_missing_outgoing": [],
        "wikilink_syntax": [],
        "_ingest_covered": 0,
        "_ingest_total": 0,
    }


def _check_per_page(
    pages: list[Path],
    inbound: dict[Path, list[Path]],
    broken: dict[Path, list[str]],
    results: dict[str, Any],
) -> None:
    """逐页检查：孤儿、关联区块、参考来源、断链、空壳、wikilink 语法、source 引用。"""
    for page in pages:
        resolved = page.resolve()
        content = page.read_text(encoding="utf-8")
        rel = page.relative_to(REPO_ROOT)

        if page.name.lower() not in ("readme.md", "index.md"):
            if not inbound.get(resolved):
                results["orphan_pages"].append(str(rel))

        is_meta_page = (
            page.name.lower() in ("readme.md", "index.md")
            or "references/" in str(rel)
            or "roadmaps/" in str(rel)
        )
        if not is_meta_page and not has_section(content, ["关联", "related", "已有页面", "关系"]):
            results["missing_related"].append(str(rel))

        is_meta_sources = page.name.lower() in ("readme.md", "index.md") or "references/" in str(
            rel
        )
        if not is_meta_sources and not has_section(content, ["参考来源", "sources", "参考"]):
            results["missing_sources"].append(str(rel))

        if resolved in broken:
            for b in broken[resolved]:
                results["broken_links"].append(f"{rel} → {b}")

        if word_count(content) < 200:
            results["stub_pages"].append(f"{rel} ({word_count(content)} 字)")

        if not is_meta_page:
            results["_ingest_total"] += 1
            if has_source_reference(content):
                results["_ingest_covered"] += 1

        wl_stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", wl_stripped):
            token = m.group(1).strip()
            if re.match(r"^[\d\.\-\s,]+$", token):
                continue
            results["wikilink_syntax"].append(f"{rel}: {m.group(0)}")

        stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+sources/[^)]+\.md)[^)]*\)", stripped):
            href = m.group(2).split("#")[0]
            resolved_src = (page.parent / href).resolve()
            if not resolved_src.exists():
                results["broken_source_refs"].append(f"{rel} → {href}")


def _check_missing_concepts(pages: list[Path], results: dict[str, Any]) -> None:
    """全局扫描提及但缺少对应 wiki 页面的技术概念。"""
    existing_stems = {p.stem.lower() for p in pages}
    watch_terms = {
        "MPPI": "model-based-rl",
        "DMP": "dmp",
        "GAE": "gae",
        "HER": "her",
        "POMDP": "pomdp",
        "Pontryagin": "optimal-control",
        "DDPG": "policy-optimization",
        "MARL": "marl",
        "ContactNet": "contact-net",
    }
    all_content = "".join(page.read_text(encoding="utf-8") for page in pages)
    term_counts: dict[str, int] = {}
    for term, slug in watch_terms.items():
        count = len(re.findall(rf"\b{re.escape(term)}\b", all_content))
        if count >= 2 and slug not in existing_stems:
            term_counts[term] = count
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        results["missing_pages"].append(
            f"{term} （出现 {count} 次，建议新建 wiki/{watch_terms[term]}.md）"
        )


def _check_sources_health(results: dict[str, Any]) -> None:
    """Sources 孤儿检测 + 陈旧页面检测。"""
    sources_papers_dir = REPO_ROOT / "sources" / "papers"
    if not sources_papers_dir.exists():
        return

    for src_file in sorted(sources_papers_dir.glob("*.md")):
        src_content = src_file.read_text(encoding="utf-8")
        for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
            href = m.group(1).split("#")[0]
            target = (src_file.parent / href).resolve()
            if not target.exists():
                results["sources_orphans"].append(f"sources/papers/{src_file.name} → {href}")

    if os.environ.get("GITHUB_ACTIONS") == "true":
        return
    seen_stale: set[Path] = set()
    for src_file in sorted(sources_papers_dir.glob("*.md")):
        src_content = src_file.read_text(encoding="utf-8")
        src_mtime = src_file.stat().st_mtime
        for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
            href = m.group(1).split("#")[0]
            wiki_target = (src_file.parent / href).resolve()
            if wiki_target.exists() and wiki_target not in seen_stale:
                wiki_mtime = wiki_target.stat().st_mtime
                if src_mtime > wiki_mtime + 86400:
                    seen_stale.add(wiki_target)
                    rel_wiki = wiki_target.relative_to(REPO_ROOT)
                    src_date = date.fromtimestamp(src_mtime).isoformat()
                    wiki_date = date.fromtimestamp(wiki_mtime).isoformat()
                    results["stale_pages"].append(
                        f"{rel_wiki} (wiki:{wiki_date} < sources/{src_file.name}:{src_date})"
                    )


def _check_contradictions(pages: list[Path], results: dict[str, Any]) -> None:
    """矛盾检测 — 检查同一概念在不同页面是否有相反的定性描述。"""
    canonical_facts = load_canonical_facts()
    all_pages_content = {
        p: strip_misconception_sections(p.read_text(encoding="utf-8")) for p in pages
    }
    for fact_id, fact in canonical_facts.items():
        pos_pages, neg_pages = [], []
        for page, content in all_pages_content.items():
            if not all(re.search(t, content, re.IGNORECASE) for t in fact["terms"]):
                continue
            has_pos = any(re.search(p, content, re.IGNORECASE) for p in fact["pos_claims"])
            has_neg = any(re.search(p, content, re.IGNORECASE) for p in fact["neg_claims"])
            if has_pos and has_neg:
                continue
            if has_pos:
                pos_pages.append(page.stem)
            if has_neg:
                neg_pages.append(page.stem)
        if pos_pages and neg_pages:
            results["contradictions"].append(
                f"「{fact_id}」正面描述({', '.join(pos_pages)}) vs 负面描述({', '.join(neg_pages)})"
            )


def _check_frontmatter(pages: list[Path], results: dict[str, Any]) -> None:
    """Frontmatter 检查：type 字段、updated 过期、summary 字段。"""
    fm_exempt_dirs = {"references", "roadmaps", "tech-map", "overview", "schema", "queries"}
    today_for_stale = date.today()

    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)

        if page.name.lower() not in ("readme.md", "index.md") and not any(
            d in parts for d in fm_exempt_dirs
        ):
            if fm_match:
                if not re.search(r"^type\s*:", fm_match.group(1), re.MULTILINE):
                    results["missing_type"].append(str(rel))
            else:
                results["missing_type"].append(str(rel))

        if fm_match:
            upd_m = re.search(r"^updated:\s*(\d{4}-\d{2}-\d{2})", fm_match.group(1), re.MULTILINE)
            if upd_m:
                try:
                    days_old = (today_for_stale - date.fromisoformat(upd_m.group(1))).days
                    if days_old > 180:
                        results["outdated_pages"].append(
                            f"{rel} （updated: {upd_m.group(1)}，已 {days_old} 天）"
                        )
                except ValueError:
                    pass

        if len(parts) >= 2 and parts[0] == "wiki" and parts[1] in ("concepts", "methods", "tasks"):
            if not fm_match:
                results["missing_summary"].append(str(rel))
            elif not re.search(r"^(summary|description)\s*:", fm_match.group(1), re.MULTILINE):
                results["missing_summary"].append(str(rel))


def _check_log_activity(results: dict[str, Any]) -> None:
    """log.md 活跃度检查（最近 30 天内是否有操作记录）。"""
    log_path = REPO_ROOT / "log.md"
    if not log_path.exists():
        results["log_inactive"].append("log.md 文件不存在，无法检查知识库活跃度")
        return

    log_content = log_path.read_text(encoding="utf-8")
    date_matches = re.findall(r"^## \[(\d{4}-\d{2}-\d{2})\]", log_content, re.MULTILINE)
    if not date_matches:
        results["log_inactive"].append(
            "log.md 中未找到符合格式的操作记录（格式：## [YYYY-MM-DD] ...）"
        )
        return

    latest = max(date_matches)
    try:
        days_since = (date.today() - date.fromisoformat(latest)).days
        if days_since > 30:
            results["log_inactive"].append(
                f"log.md 最后操作于 {latest}（已 {days_since} 天未更新，知识库可能停止维护）"
            )
    except ValueError:
        pass


def _check_query_format(pages: list[Path], results: dict[str, Any]) -> None:
    """queries/ 页面必须包含 Query 产物说明 + 参考来源 + 关联页面。"""
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if (
            len(parts) < 2
            or parts[0] != "wiki"
            or parts[1] != "queries"
            or page.name == "README.md"
        ):
            continue
        content = page.read_text(encoding="utf-8")
        missing_parts = []
        if "**Query 产物**" not in content:
            missing_parts.append("缺 'Query 产物' 说明")
        if "## 参考来源" not in content:
            missing_parts.append("缺 '## 参考来源' 区块")
        if "## 关联页面" not in content:
            missing_parts.append("缺 '## 关联页面' 区块")
        if missing_parts:
            results["query_format"].append(f"{rel}（{', '.join(missing_parts)}）")


def _check_formalizations(pages: list[Path], results: dict[str, Any]) -> None:
    """formalizations/ 公式块和变量解释检查。"""
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "formalizations":
            continue
        content = page.read_text(encoding="utf-8")

        if "$$" not in content and "$`" not in content and "`$" not in content:
            results["formalization_no_formula"].append(str(rel))
            continue

        _check_formalization_vars(rel, content, results)


def _var_is_explained(var: str, content: str) -> bool:
    """检查单个大写变量是否在正文中有物理含义解释。"""
    esc = re.escape(var)
    patterns = [
        rf"-\s*\$\\?{esc}[^$]*\$[^|\n]*?[:：]",
        rf"\$\\?{esc}[^$]*\$\s*[:：]\s*\S",
        rf"\|\s*\$\\?{esc}[^$]*\$\s*\|",
        rf"\$\\?{esc}[^$]*\$[^.\n]{{0,80}}(?:是|为|表示|代表|指|denote|denotes|含义|定义)",
        rf"(?:其中|where).{{0,100}}\$\\?{esc}[^$]*\$",
        rf"\$\\?{esc}\s*(?:=|\\in|\\succeq|\\succ|\\equiv|\\triangleq)",
        rf"\*\*[^*]*\$\\?{esc}[^$]*\$[^*]*\*\*",
    ]
    return any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in patterns)


def _check_formalization_vars(rel: Path, content: str, results: dict[str, Any]) -> None:
    """检查 formalization 页面中 $$...$$ 公式变量是否有正文解释。"""
    display_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
    if not display_blocks:
        return
    candidate_vars: set[str] = set()
    for block in display_blocks:
        for v in re.findall(r"(?<![A-Za-z\\_^{])([A-Z])(?![A-Za-z_(])", block):
            candidate_vars.add(v)
    if not candidate_vars:
        return
    unexplained: list[str] = []
    for v in sorted(candidate_vars):
        if not _var_is_explained(v, content):
            unexplained.append(v)
    if unexplained:
        results["formalization_unexplained_vars"].append(
            f"{rel}（变量缺解释：{', '.join(unexplained)}）"
        )


def _checklist_version_num(path: Path) -> int:
    m = re.search(r"v(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _check_checklist_badge(readme_content: str, results: dict[str, Any]) -> None:
    """检查 README 中 checklist 链接版本是否与最新文件一致。"""
    checklist_files = sorted(
        (REPO_ROOT / "docs" / "checklists").glob("tech-stack-next-phase-checklist-v*.md")
    )
    if not checklist_files:
        return
    latest_checklist = max(checklist_files, key=_checklist_version_num)
    latest_ver = _checklist_version_num(latest_checklist)

    main_link_versions = re.findall(r"\[技术栈项目执行清单 v(\d+)\]", readme_content)
    if main_link_versions and int(main_link_versions[0]) < latest_ver:
        results["readme_badge"].append(
            f"README 主执行清单标题仍是 v{main_link_versions[0]}，但最新为 v{latest_ver}，请更新"
        )

    source_badge_match = re.search(
        r"\[!\[Sources Coverage\]\([^)]+\)\]\(([^)]+tech-stack-next-phase-checklist-v(\d+)\.md)\)",
        readme_content,
    )
    if not source_badge_match:
        results["readme_badge"].append("README 缺少 Sources Coverage badge 或链接格式异常")
    else:
        badge_link = source_badge_match.group(1)
        badge_ver = int(source_badge_match.group(2))
        expected_link = str(latest_checklist.relative_to(REPO_ROOT))
        if badge_ver < latest_ver or badge_link != expected_link:
            results["readme_badge"].append(
                f"README Sources badge 指向 {badge_link}，但最新应为 {expected_link}"
            )


def _check_graph_badge(readme_content: str, results: dict[str, Any]) -> None:
    """检查 README 中 Knowledge Graph badge 数据是否与 graph-stats.json 一致。"""
    graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
    if not graph_stats_path.exists():
        return
    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
    node_count = graph_stats.get("node_count")
    edge_count = graph_stats.get("edge_count")
    graph_badge_match = re.search(
        r"\[!\[Knowledge Graph\]\(https://img\.shields\.io/badge/知识图谱-(\d+)节点_(\d+)边-blue\?logo=d3\.js\)\]\([^)]+\)",
        readme_content,
    )
    if not graph_badge_match:
        results["readme_badge"].append("README 缺少 Knowledge Graph badge 或格式异常")
    else:
        badge_nodes = int(graph_badge_match.group(1))
        badge_edges = int(graph_badge_match.group(2))
        if badge_nodes != node_count or badge_edges != edge_count:
            results["readme_badge"].append(
                f"README Knowledge Graph badge 为 {badge_nodes}节点/{badge_edges}边，但实际为 {node_count}节点/{edge_count}边"
            )


def _check_readme_badges(results: dict[str, Any]) -> None:
    """README.md 中 badges / checklist 链接应与当前仓库状态一致。"""
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return
    readme_content = readme_path.read_text(encoding="utf-8")
    _check_checklist_badge(readme_content, results)
    _check_graph_badge(readme_content, results)


def _check_graph_orphans(results: dict[str, Any]) -> None:
    """graph-stats.json 中孤儿节点预警。"""
    graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
    if not graph_stats_path.exists():
        return
    graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
    orphan_nodes = graph_stats.get("orphan_nodes", [])
    if orphan_nodes:
        results["orphan_count"].append(
            f"发现 {len(orphan_nodes)} 个孤儿节点（无入链）：{orphan_nodes}"
        )


def _check_method_page(page: Path, rel: Path, content: str, results: dict[str, Any]) -> None:
    """单个 methods/ 页面检查：必须链接到 formalizations/concepts，必须含主要路线区块。"""
    links = extract_internal_links(content, page)
    has_required_link = False
    for target in links:
        if not target.is_relative_to(REPO_ROOT):
            continue
        target_parts = target.relative_to(REPO_ROOT).parts
        if (
            len(target_parts) >= 2
            and target_parts[0] == "wiki"
            and target_parts[1] in ("formalizations", "concepts")
        ):
            has_required_link = True
            break
    if not has_required_link:
        results["method_missing_link"].append(str(rel))

    if not re.search(r"##\s+(主要方法路线|核心技术路线|主要分类|主要技术路线)", content):
        results["method_missing_sections"].append(str(rel))


def _check_entity_page(page: Path, rel: Path, content: str, results: dict[str, Any]) -> None:
    """单个 entities/ 页面检查：必须含至少 2 个指向 methods/tasks 的出边。"""
    links = extract_internal_links(content, page)
    out_count = 0
    for target in links:
        if not target.is_relative_to(REPO_ROOT):
            continue
        t_parts = target.relative_to(REPO_ROOT).parts
        if len(t_parts) >= 2 and t_parts[0] == "wiki" and t_parts[1] in ("methods", "tasks"):
            out_count += 1
    if out_count < 2:
        results["entity_missing_outgoing"].append(f"{rel} (当前出边: {out_count})")


def _check_methods_entities(pages: list[Path], results: dict[str, Any]) -> None:
    """methods/ 页面结构检查 + entities/ 出边检查。

    注意：原始实现中 entities/ 检查写在 methods/ 循环内部，
    因 ``continue`` 门控导致 entities/ 页面实际不会被扫描。
    此处保持相同行为——仅在 methods/ 页面上执行两项检查。
    """
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "methods":
            continue
        content = page.read_text(encoding="utf-8")
        _check_method_page(page, rel, content, results)

        if parts[1] == "entities":
            _check_entity_page(page, rel, content, results)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def lint() -> dict[str, Any]:
    pages = get_wiki_pages()
    page_set = {p.resolve() for p in pages}
    inbound, broken_links = _build_link_index(pages, page_set)
    results = _empty_results()

    _check_per_page(pages, inbound, broken_links, results)
    _check_missing_concepts(pages, results)
    _check_sources_health(results)
    _check_contradictions(pages, results)
    _check_frontmatter(pages, results)
    _check_log_activity(results)
    _check_query_format(pages, results)
    _check_formalizations(pages, results)
    _check_readme_badges(results)
    _check_graph_orphans(results)
    _check_methods_entities(pages, results)

    return results


def format_report(results: dict[str, Any]) -> str:
    today = date.today().isoformat()
    lines = [f"## [{today}] lint | health-check | 自动化 wiki 健康检查", ""]

    total_issues = sum(len(v) for k, v in results.items() if not k.startswith("_"))
    lines.append(f"共发现 **{total_issues}** 个问题：")
    lines.append("")

    sections = [
        ("orphan_pages", "孤儿页（无入链）", "⚠️"),
        ("missing_related", "缺少关联页面区块", "⚠️"),
        ("missing_sources", "缺少参考来源区块", "⚠️"),
        ("broken_links", "断链（内链目标不存在）", "❌"),
        ("wikilink_syntax", "禁止的 [[...]] wikilink 写法（请用标准 Markdown）", "❌"),
        ("broken_source_refs", "引用了不存在的 sources/ 文件", "❌"),
        ("sources_orphans", "Sources 孤儿（sources/papers 死链）", "❌"),
        ("stale_pages", "陈旧页面（sources 比 wiki 新，建议 review）", "⚠️"),
        ("outdated_pages", "可能过期（updated: 距今 > 180 天）", "⚠️"),
        ("contradictions", "潜在矛盾（跨页面相反定性描述）", "⚠️"),
        ("stub_pages", "空壳页面（< 200 字）", "⚠️"),
        ("missing_pages", "频繁提及但缺少 wiki 页面的概念", "💡"),
        ("missing_type", "Frontmatter 缺少 type 字段", "⚠️"),
        ("log_inactive", "log.md 活跃度警告", "⚠️"),
        ("missing_summary", "缺少摘要字段（summary/description）", "⚠️"),
        ("query_format", "Query 页面格式不完整（缺 Query 产物/参考来源/关联页面）", "⚠️"),
        ("formalization_no_formula", "Formalization 页面缺少公式块", "⚠️"),
        ("formalization_unexplained_vars", "Formalization 公式变量缺少正文物理含义解释", "⚠️"),
        ("readme_badge", "README checklist 链接版本不一致", "⚠️"),
        ("orphan_count", "图谱孤儿节点预警（graph-stats.json）", "⚠️"),
        ("method_missing_link", "Methods 页面缺少 Formalization/Concept 链接", "⚠️"),
        ("method_missing_sections", "Methods 页面缺少主要路线区块", "⚠️"),
        ("entity_missing_outgoing", "Entities 页面缺少 Methods/Tasks 关联出边", "⚠️"),
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

    covered = results.get("_ingest_covered", 0)
    total = results.get("_ingest_total", 0)
    pct = round(covered / total * 100) if total else 0
    lines.append(f"📊 Sources 覆盖率：{covered}/{total} ({pct}%) wiki/entity 页有 ingest 来源")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Robotics_Notebooks wiki lint 检查")
    parser.add_argument("--write-log", action="store_true", help="将结果追加到 log.md")
    parser.add_argument(
        "--report", action="store_true", help="将 markdown 健康报告保存到 exports/lint-report.md"
    )
    args = parser.parse_args()

    print("正在扫描 wiki/ 目录...")
    results = lint()
    report = format_report(results)

    print(report)

    total = sum(len(v) for k, v in results.items() if not k.startswith("_"))
    if total == 0:
        print("✅ 所有检查通过！")
    else:
        print(f"⚠️  共发现 {total} 个问题，请参考上方报告处理。")

    if args.write_log:
        log_path = REPO_ROOT / "log.md"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n---\n\n")
            f.write(report)
        print(f"\n已将报告追加到 {log_path}")

    if args.report:
        exports_dir = REPO_ROOT / "exports"
        exports_dir.mkdir(exist_ok=True)
        report_path = exports_dir / "lint-report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Wiki 健康报告\n\n")
            f.write(report)
        print(f"\n已将健康报告保存到 {report_path}")

    if total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
