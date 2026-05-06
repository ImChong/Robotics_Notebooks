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
    """移除“常见误区/误区”区块，避免把辟谣内容误判为事实矛盾。"""
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


def lint() -> dict[str, Any]:
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
                    str(target.relative_to(REPO_ROOT))
                    if target.is_relative_to(REPO_ROOT)
                    else str(target)
                )

    # 已有 wiki 页面的 stem 集合（用于"提及但缺页"检测）
    existing_stems = {p.stem.lower() for p in pages}

    results: dict[str, Any] = {
        "orphan_pages": [],
        "missing_related": [],
        "missing_sources": [],
        "broken_links": [],
        "stub_pages": [],
        "missing_pages": [],  # 提及但缺少对应 wiki 页面的技术概念
        "broken_source_refs": [],  # 引用了不存在的 sources/ 文件
        "sources_orphans": [],  # P3.3: sources/papers 中的死链（wiki 目标不存在）
        "stale_pages": [],  # P3.2: wiki 页面比对应 sources 文件旧
        "outdated_pages": [],  # V9: frontmatter updated: 字段距今 > 180 天
        "contradictions": [],  # P3.1: 同一概念跨页面矛盾描述
        "missing_type": [],  # V8: wiki 页面缺少 frontmatter type 字段
        "log_inactive": [],  # V8: log.md 最近 30 天无操作记录
        "missing_summary": [],  # V10: concepts/methods/tasks 缺少 summary/description
        "query_format": [],  # V11: queries/ 缺少 Query 产物说明/参考来源/关联页面
        "formalization_no_formula": [],  # V11: formalizations/ 缺少公式块
        "formalization_unexplained_vars": [],  # V21: formalizations/ 公式变量缺少正文物理含义解释
        "readme_badge": [],  # V11: README checklist 链接版本不一致
        "orphan_count": [],  # V13: graph-stats.json 中孤儿节点（无入链）预警
        "method_missing_link": [],  # V15: methods/ 页面缺少指向 formalizations/ 或 concepts/ 的链接
        "method_missing_sections": [],  # V17: methods/ 页面缺少标准区块
        "entity_missing_outgoing": [],  # V20: entities/ 页面缺少指向 methods/ 或 tasks/ 的出边
        "wikilink_syntax": [],  # V22: 禁止使用 [[...]] Obsidian wikilink，必须用标准 Markdown 内链
        "_ingest_covered": 0,  # 内部统计：有 ingest 来源的页面数
        "_ingest_total": 0,  # 内部统计：扫描的页面总数
    }

    for page in pages:
        resolved = page.resolve()
        content = page.read_text(encoding="utf-8")
        rel = page.relative_to(REPO_ROOT)

        # 1. 孤儿页（README 和 index 类页面排除）
        if page.name.lower() not in ("readme.md", "index.md"):
            if not inbound.get(resolved):
                results["orphan_pages"].append(str(rel))

        # 2. 缺少关联页面区块（README、references/、roadmaps/ 元页面豁免）
        is_meta_page = (
            page.name.lower() in ("readme.md", "index.md")
            or "references/" in str(rel)
            or "roadmaps/" in str(rel)
        )
        related_patterns = ["关联", "related", "已有页面", "关系"]
        if not is_meta_page and not has_section(content, related_patterns):
            results["missing_related"].append(str(rel))

        # 3. 缺少参考来源区块（README、references/ 元页面豁免）
        is_meta_sources = page.name.lower() in ("readme.md", "index.md") or "references/" in str(
            rel
        )
        if not is_meta_sources and not has_section(content, ["参考来源", "sources", "参考"]):
            results["missing_sources"].append(str(rel))

        # 4. 断链
        if resolved in broken_links:
            for broken in broken_links[resolved]:
                results["broken_links"].append(f"{rel} → {broken}")

        # 5. 空壳页面（< 200 字）
        if word_count(content) < 200:
            results["stub_pages"].append(f"{rel} ({word_count(content)} 字)")

        # Ingest coverage: count non-meta wiki pages with any sources/ raw-material link
        if not is_meta_page:
            results["_ingest_total"] += 1
            if has_source_reference(content):
                results["_ingest_covered"] += 1

        # 5a. 禁止使用 Obsidian [[wikilink]] 语法（AGENTS.md 第 156 行规定）
        #     代码块和行内代码内的 [[...]] 是合法的（如 numpy 矩阵字面量），跳过
        wl_stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", wl_stripped):
            token = m.group(1).strip()
            # 过滤数字/矩阵字面量残留（理论上 strip_code_blocks 已处理，双保险）
            if re.match(r"^[\d\.\-\s,]+$", token):
                continue
            results["wikilink_syntax"].append(f"{rel}: {m.group(0)}")

        # 5b. 引用了不存在的 sources/ 文件（检测 sources/ 路径的内链）
        stripped = strip_code_blocks(content)
        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+sources/[^)]+\.md)[^)]*\)", stripped):
            href = m.group(2).split("#")[0]
            resolved_src = (page.parent / href).resolve()
            if not resolved_src.exists():
                results["broken_source_refs"].append(f"{rel} → {href}")

    # 6. 提及但缺少对应 wiki 页面的技术概念（全局扫描）
    WATCH_TERMS = {
        # key: 术语名，value: 期望覆盖该术语的 wiki 页面 stem
        # 已有覆盖的术语不在此列（EKF→ekf.md, HQP→hqp.md, SAC→policy-optimization.md,
        #   InEKF→ekf.md, LQR→lqr.md, NMPC→model-predictive-control.md）
        "MPPI": "model-based-rl",  # 已在 model-based-rl.md 中覆盖
        "DMP": "dmp",
        "GAE": "gae",
        "HER": "her",
        "POMDP": "pomdp",
        "Pontryagin": "optimal-control",  # 已在 optimal-control.md 有专节
        "DDPG": "policy-optimization",  # 已在 policy-optimization.md 提及
        "MARL": "marl",
        "ContactNet": "contact-net",
    }
    term_counts: dict[str, int] = {}
    all_content = ""
    for page in pages:
        all_content += page.read_text(encoding="utf-8")
    for term in WATCH_TERMS:
        count = len(re.findall(rf"\b{re.escape(term)}\b", all_content))
        slug = WATCH_TERMS[term]
        if count >= 2 and slug not in existing_stems:
            term_counts[term] = count
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        results["missing_pages"].append(
            f"{term} （出现 {count} 次，建议新建 wiki/{WATCH_TERMS[term]}.md）"
        )

    # P3.3: Sources 孤儿检测 — sources/papers/*.md 中链接到不存在的 wiki 页
    sources_papers_dir = REPO_ROOT / "sources" / "papers"
    if sources_papers_dir.exists():
        for src_file in sorted(sources_papers_dir.glob("*.md")):
            src_content = src_file.read_text(encoding="utf-8")
            for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
                href = m.group(1).split("#")[0]
                target = (src_file.parent / href).resolve()
                if not target.exists():
                    results["sources_orphans"].append(f"sources/papers/{src_file.name} → {href}")

    # P3.2: 陈旧页面检测 — sources 文件比对应 wiki 页更新时，标记需 review
    # 注意：在 GitHub Actions 中 mtime 不可靠（checkout 会重置 mtime），故跳过
    if sources_papers_dir.exists() and os.environ.get("GITHUB_ACTIONS") != "true":
        seen_stale = set()
        for src_file in sorted(sources_papers_dir.glob("*.md")):
            src_content = src_file.read_text(encoding="utf-8")
            src_mtime = src_file.stat().st_mtime
            for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", src_content):
                href = m.group(1).split("#")[0]
                wiki_target = (src_file.parent / href).resolve()
                if wiki_target.exists() and wiki_target not in seen_stale:
                    wiki_mtime = wiki_target.stat().st_mtime
                    if src_mtime > wiki_mtime + 86400:  # 1天容差，避免同批次误报
                        seen_stale.add(wiki_target)
                        rel_wiki = wiki_target.relative_to(REPO_ROOT)
                        src_date = date.fromtimestamp(src_mtime).isoformat()
                        wiki_date = date.fromtimestamp(wiki_mtime).isoformat()
                        results["stale_pages"].append(
                            f"{rel_wiki} (wiki:{wiki_date} < sources/{src_file.name}:{src_date})"
                        )

    # P3.1: 矛盾检测 — 检查同一概念在不同页面是否有相反的定性描述
    # CANONICAL_FACTS: {fact_id: {terms, pos_claims, neg_claims}}
    # 当 pos_claims 和 neg_claims 同时出现在不同页面时，报告潜在矛盾
    CANONICAL_FACTS = load_canonical_facts()
    all_pages_content = {
        p: strip_misconception_sections(p.read_text(encoding="utf-8")) for p in pages
    }
    for fact_id, fact in CANONICAL_FACTS.items():
        pos_pages, neg_pages = [], []
        for page, content in all_pages_content.items():
            if not all(re.search(t, content, re.IGNORECASE) for t in fact["terms"]):
                continue
            has_pos = any(re.search(p, content, re.IGNORECASE) for p in fact["pos_claims"])
            has_neg = any(re.search(p, content, re.IGNORECASE) for p in fact["neg_claims"])
            # 同一页面同时命中正反模式时，多半是比较/讨论页或正则重叠，
            # 而非真矛盾。跳过该页避免 self-vs-self 误报（如「Impedance 柔顺性」）。
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

    # V8: Frontmatter type 字段一致性检查
    # 豁免：references/、roadmaps/、tech-map/、overview/ 目录及 README/index 文件
    fm_exempt_dirs = {"references", "roadmaps", "tech-map", "overview", "schema", "queries"}
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if page.name.lower() in ("readme.md", "index.md"):
            continue
        if any(d in parts for d in fm_exempt_dirs):
            continue
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            if not re.search(r"^type\s*:", fm_text, re.MULTILINE):
                results["missing_type"].append(str(rel))
        else:
            results["missing_type"].append(str(rel))

    # V9: frontmatter updated: 字段过期检测（距今 > 180 天）
    today_for_stale = date.today()
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            continue
        upd_m = re.search(r"^updated:\s*(\d{4}-\d{2}-\d{2})", fm_match.group(1), re.MULTILINE)
        if not upd_m:
            continue
        try:
            upd_dt = date.fromisoformat(upd_m.group(1))
            days_old = (today_for_stale - upd_dt).days
            if days_old > 180:
                results["outdated_pages"].append(
                    f"{rel} （updated: {upd_m.group(1)}，已 {days_old} 天）"
                )
        except ValueError:
            pass

    # V8: log.md 活跃度检查（最近 30 天内是否有操作记录）
    log_path = REPO_ROOT / "log.md"
    if log_path.exists():
        log_content = log_path.read_text(encoding="utf-8")
        today_dt = date.today()
        # 解析所有 ## [YYYY-MM-DD] 条目
        date_matches = re.findall(r"^## \[(\d{4}-\d{2}-\d{2})\]", log_content, re.MULTILINE)
        if date_matches:
            latest = max(date_matches)
            try:
                latest_dt = date.fromisoformat(latest)
                days_since = (today_dt - latest_dt).days
                if days_since > 30:
                    results["log_inactive"].append(
                        f"log.md 最后操作于 {latest}（已 {days_since} 天未更新，知识库可能停止维护）"
                    )
            except ValueError:
                pass
        else:
            results["log_inactive"].append(
                "log.md 中未找到符合格式的操作记录（格式：## [YYYY-MM-DD] ...）"
            )
    else:
        results["log_inactive"].append("log.md 文件不存在，无法检查知识库活跃度")

    # V10: concepts/methods/tasks frontmatter 摘要字段完整性检查
    summary_dirs = {"concepts", "methods", "tasks"}
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] not in summary_dirs:
            continue
        content = page.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            results["missing_summary"].append(str(rel))
            continue
        fm_text = fm_match.group(1)
        if not re.search(r"^(summary|description)\s*:", fm_text, re.MULTILINE):
            results["missing_summary"].append(str(rel))

    # V11: queries/ 页面必须包含 Query 产物说明 + 参考来源 + 关联页面
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

    # V11: formalizations/ 页面必须包含至少一个公式块
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "formalizations":
            continue
        content = page.read_text(encoding="utf-8")
        if "$$" not in content and "$`" not in content and "`$" not in content:
            results["formalization_no_formula"].append(str(rel))

    # V21: formalizations/ 公式变量必须有物理含义解释
    # 启发式：从 $$...$$ 显示公式抽取“单字母拉丁大写变量”，排除函数调用形式（X(...)）。
    # 对每个变量，检查正文是否含有定义/解释模式（动词、表格行、条目冒号、其中子句、等式定义等）。
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "formalizations":
            continue
        content = page.read_text(encoding="utf-8")
        display_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
        if not display_blocks:
            continue
        candidate_vars: set[str] = set()
        for block in display_blocks:
            for v in re.findall(r"(?<![A-Za-z\\_^{])([A-Z])(?![A-Za-z_(])", block):
                candidate_vars.add(v)
        if not candidate_vars:
            continue
        unexplained: list[str] = []
        for v in sorted(candidate_vars):
            esc = re.escape(v)
            patterns = [
                # 列表条目： - $X$（可带下标/上标）：含义
                rf"-\s*\$\\?{esc}[^$]*\$[^|\n]*?[:：]",
                # 行内冒号定义： $X$：含义
                rf"\$\\?{esc}[^$]*\$\s*[:：]\s*\S",
                # 表格行 | $X...$ |
                rf"\|\s*\$\\?{esc}[^$]*\$\s*\|",
                # 动词解释： $X...$ 是/为/表示/代表/指/denote/含义/定义
                rf"\$\\?{esc}[^$]*\$[^.\n]{{0,80}}(?:是|为|表示|代表|指|denote|denotes|含义|定义)",
                # “其中 $X$” / “where $X$”
                rf"(?:其中|where).{{0,100}}\$\\?{esc}[^$]*\$",
                # 等式/集合/序关系定义： $X = / \in / \succeq / \succ / \equiv / \triangleq
                rf"\$\\?{esc}\s*(?:=|\\in|\\succeq|\\succ|\\equiv|\\triangleq)",
                # 粗体段落中带变量： **... $X$ ...**
                rf"\*\*[^*]*\$\\?{esc}[^$]*\$[^*]*\*\*",
            ]
            if not any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in patterns):
                unexplained.append(v)
        if unexplained:
            results["formalization_unexplained_vars"].append(
                f"{rel}（变量缺解释：{', '.join(unexplained)}）"
            )

    # V11: README.md 中 badges / checklist 链接应与当前仓库状态一致
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")

        checklist_files = sorted(
            (REPO_ROOT / "docs" / "checklists").glob("tech-stack-next-phase-checklist-v*.md")
        )
        if checklist_files:

            def _checklist_version_num(path: Path) -> int:
                m = re.search(r"v(\d+)", path.stem)
                return int(m.group(1)) if m else -1

            latest_checklist = max(checklist_files, key=_checklist_version_num)
            latest_ver = _checklist_version_num(latest_checklist)

            main_link_versions = re.findall(r"\[技术栈项目执行清单 v(\d+)\]", readme_content)
            if main_link_versions:
                main_ver = int(main_link_versions[0])
                if main_ver < latest_ver:
                    results["readme_badge"].append(
                        f"README 主执行清单标题仍是 v{main_ver}，但最新为 v{latest_ver}，请更新"
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

        graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
        if graph_stats_path.exists():
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

    # V13: 孤儿节点计数检测（读取 graph-stats.json）
    graph_stats_path = REPO_ROOT / "exports" / "graph-stats.json"
    if graph_stats_path.exists():
        graph_stats = json.loads(graph_stats_path.read_text(encoding="utf-8"))
        orphan_nodes = graph_stats.get("orphan_nodes", [])
        if orphan_nodes:
            results["orphan_count"].append(
                f"发现 {len(orphan_nodes)} 个孤儿节点（无入链）：{orphan_nodes}"
            )

    # V15: methods/ 页面必须包含指向 formalizations/ 或 concepts/ 的链接
    for page in pages:
        rel = page.relative_to(REPO_ROOT)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "methods":
            continue
        content = page.read_text(encoding="utf-8")
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

        # V17: methods/ 页面必须包含主要方法路线区块
        if not re.search(r"##\s+(主要方法路线|核心技术路线|主要分类|主要技术路线)", content):
            results["method_missing_sections"].append(str(rel))

        # V20: entities/ 页面必须包含至少 2 个指向 methods/ 或 tasks/ 的出边
        if len(parts) >= 2 and parts[0] == "wiki" and parts[1] == "entities":
            links = extract_internal_links(content, page)
            out_count = 0
            for target in links:
                if not target.is_relative_to(REPO_ROOT):
                    continue
                t_parts = target.relative_to(REPO_ROOT).parts
                if (
                    len(t_parts) >= 2
                    and t_parts[0] == "wiki"
                    and t_parts[1] in ("methods", "tasks")
                ):
                    out_count += 1
            if out_count < 2:
                results["entity_missing_outgoing"].append(f"{rel} (当前出边: {out_count})")

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
