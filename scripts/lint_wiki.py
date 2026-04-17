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

用法：
  python3 scripts/lint_wiki.py
  python3 scripts/lint_wiki.py --write-log   # 同时追加报告到 log.md
  python3 scripts/lint_wiki.py --report      # 保存 markdown 报告到 exports/lint-report.md
"""

import argparse
import os
import re
import sys
from datetime import date, datetime
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

    # 已有 wiki 页面的 stem 集合（用于"提及但缺页"检测）
    existing_stems = {p.stem.lower() for p in pages}

    results = {
        "orphan_pages": [],
        "missing_related": [],
        "missing_sources": [],
        "broken_links": [],
        "stub_pages": [],
        "missing_pages": [],          # 提及但缺少对应 wiki 页面的技术概念
        "broken_source_refs": [],     # 引用了不存在的 sources/ 文件
        "sources_orphans": [],        # P3.3: sources/papers 中的死链（wiki 目标不存在）
        "stale_pages": [],            # P3.2: wiki 页面比对应 sources 文件旧
        "outdated_pages": [],         # V9: frontmatter updated: 字段距今 > 180 天
        "contradictions": [],         # P3.1: 同一概念跨页面矛盾描述
        "missing_type": [],           # V8: wiki 页面缺少 frontmatter type 字段
        "log_inactive": [],           # V8: log.md 最近 30 天无操作记录
        "_ingest_covered": 0,         # 内部统计：有 ingest 来源的页面数
        "_ingest_total": 0,           # 内部统计：扫描的页面总数
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
        is_meta_sources = (
            page.name.lower() in ("readme.md", "index.md")
            or "references/" in str(rel)
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

        # Ingest coverage: count non-meta wiki pages and those with sources/papers/ links
        if not is_meta_page:
            results["_ingest_total"] += 1
            if "sources/papers/" in content:
                results["_ingest_covered"] += 1

        # 5b. 引用了不存在的 sources/ 文件（检测 sources/ 路径的内链）
        stripped = strip_code_blocks(content)
        for m in re.finditer(r'\[([^\]]*)\]\(([^)]+sources/[^)]+\.md)[^)]*\)', stripped):
            href = m.group(2).split("#")[0]
            resolved_src = (page.parent / href).resolve()
            if not resolved_src.exists():
                results["broken_source_refs"].append(
                    f"{rel} → {href}"
                )

    # 6. 提及但缺少对应 wiki 页面的技术概念（全局扫描）
    WATCH_TERMS = {
        # key: 术语名，value: 期望覆盖该术语的 wiki 页面 stem
        # 已有覆盖的术语不在此列（EKF→ekf.md, HQP→hqp.md, SAC→policy-optimization.md,
        #   InEKF→ekf.md, LQR→lqr.md, NMPC→model-predictive-control.md）
        "MPPI": "model-based-rl",   # 已在 model-based-rl.md 中覆盖
        "DMP": "dmp",
        "GAE": "gae",
        "HER": "her",
        "POMDP": "pomdp",
        "Pontryagin": "optimal-control",  # 已在 optimal-control.md 有专节
        "DDPG": "policy-optimization",    # 已在 policy-optimization.md 提及
        "MARL": "marl",
        "ContactNet": "contact-net",
    }
    term_counts: dict[str, int] = {}
    all_content = ""
    for page in pages:
        all_content += page.read_text(encoding="utf-8")
    for term in WATCH_TERMS:
        count = len(re.findall(rf'\b{re.escape(term)}\b', all_content))
        slug = WATCH_TERMS[term]
        if count >= 2 and slug not in existing_stems:
            term_counts[term] = count
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        results["missing_pages"].append(f"{term} （出现 {count} 次，建议新建 wiki/{WATCH_TERMS[term]}.md）")

    # P3.3: Sources 孤儿检测 — sources/papers/*.md 中链接到不存在的 wiki 页
    sources_papers_dir = REPO_ROOT / "sources" / "papers"
    if sources_papers_dir.exists():
        for src_file in sorted(sources_papers_dir.glob("*.md")):
            src_content = src_file.read_text(encoding="utf-8")
            for m in re.finditer(r'\]\(([^)]*wiki/[^)]+\.md)\)', src_content):
                href = m.group(1).split("#")[0]
                target = (src_file.parent / href).resolve()
                if not target.exists():
                    results["sources_orphans"].append(
                        f"sources/papers/{src_file.name} → {href}"
                    )

    # P3.2: 陈旧页面检测 — sources 文件比对应 wiki 页更新时，标记需 review
    if sources_papers_dir.exists():
        seen_stale = set()
        for src_file in sorted(sources_papers_dir.glob("*.md")):
            src_content = src_file.read_text(encoding="utf-8")
            src_mtime = src_file.stat().st_mtime
            for m in re.finditer(r'\]\(([^)]*wiki/[^)]+\.md)\)', src_content):
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
    CANONICAL_FACTS = {
        "PPO 样本效率": {
            "terms": ["PPO"],
            "pos_claims": [r"PPO.*样本效率.*高|高.*样本效率.*PPO|PPO.*sample.efficient"],
            "neg_claims": [r"PPO.*样本效率.*低|PPO.*sample.inefficient|PPO.*样本效率差"],
        },
        "MPC 实时性": {
            "terms": ["MPC", "model.predictive"],
            "pos_claims": [r"MPC.*实时|实时.*MPC|MPC.*real.?time|MPC.*online"],
            "neg_claims": [r"MPC.*无法实时|MPC.*not real.?time|MPC.*计算量.*过大.*实时"],
        },
        "Domain Randomization 必要性": {
            "terms": ["domain.randomization", "域随机"],
            "pos_claims": [r"必须|必要|sim2real.*必|是.*sim2real.*关键"],
            "neg_claims": [r"降低.*in.distribution|DR.*不.*必要|不需要.*domain.*random|domain.*random.*unnecessary"],
        },
        "RL 推理速度": {
            "terms": ["policy", "RL", "强化学习"],
            "pos_claims": [r"推理.*快|推理延迟.*低|inference.*fast|low.*latency"],
            "neg_claims": [r"推理.*慢|推理.*延迟.*高|inference.*slow|latency.*high"],
        },
        "WBC 计算复杂度": {
            "terms": ["WBC", "whole.body"],
            "pos_claims": [r"实时|real.?time|efficient|高效|fast"],
            "neg_claims": [r"WBC.*计算量大|WBC.*computationally expensive|WBC.*not real.?time|WBC.*无法实时"],
        },
        "接触力估计精度": {
            "terms": ["contact", "接触力"],
            "pos_claims": [r"精确.*估计|accurate.*estimation|高精度"],
            "neg_claims": [r"估计不准|inaccurate|sim2real.*gap.*contact|接触.*仿真.*差距"],
        },
        "TSID 基于 QP": {
            "terms": ["TSID"],
            "pos_claims": [r"基于.*QP|QP.*框架|QP.*求解|二次规划.*求解"],
            "neg_claims": [r"不.*基于.*QP|独立.*于.*QP|非.*QP.*方法|TSID.*不.*基于"],
        },
        "WBC 多接触优势": {
            "terms": ["WBC", "whole.body"],
            "pos_claims": [r"优于|多接触.*优|必要|必须.*控制|统一.*优化"],
            "neg_claims": [r"WBC.*多接触.*无优势|WBC.*不必要|独立关节.*足够|WBC.*可选"],
        },
        "MuJoCo 接触精度": {
            "terms": ["mujoco", "MuJoCo"],
            "pos_claims": [r"精确|accurate|高精度|精度.*高|接触.*真实"],
            "neg_claims": [r"不精确|不适合.*接触|接触.*gap.*大|contact.*inaccurate"],
        },
        "仿真频率对接触稳定性": {
            "terms": ["仿真频率|simulation.*frequenc|sim.*freq"],
            "pos_claims": [r"关键|重要|必须|稳定.*必要|stability.*critical|高频.*稳定"],
            "neg_claims": [r"频率.*无关|低频.*足够|频率.*不重要|不影响.*稳定"],
        },
        "GAE λ 范围": {
            "terms": ["GAE", "广义优势"],
            "pos_claims": [r"λ.*\(0.*1\)|lambda.*0.*1|λ.*介于|0.*<.*λ.*<.*1"],
            "neg_claims": [r"λ.*>.*1|λ.*<.*0|GAE.*无.*λ|GAE.*不.*用.*lambda"],
        },
        "EKF 线性化误差": {
            "terms": ["EKF", "Extended Kalman"],
            "pos_claims": [r"线性化|lineariz|一阶近似|first.order"],
            "neg_claims": [r"EKF.*精确非线性|EKF.*无线性化误差|EKF.*exact"],
        },
        "Model-Based RL 样本效率": {
            "terms": ["model.based.*rl|model.based.*reinforcement|基于模型.*强化"],
            "pos_claims": [r"样本效率.*高|sample.*efficient|数据效率.*高|更.*efficient"],
            "neg_claims": [r"model.based.*样本效率.*低|model.based.*inefficient|数据效率.*差"],
        },
        "Diffusion Policy 推理延迟": {
            "terms": ["diffusion.policy", "扩散策略"],
            "pos_claims": [r"去噪.*步数|denoising.*step|推理.*多步|multi.*step.*inference"],
            "neg_claims": [r"diffusion.*单步|diffusion.*instant|扩散.*无延迟|denoising.*free"],
        },
        "WBC 优先级约束": {
            "terms": ["WBC", "whole.body"],
            "pos_claims": [r"优先级|priority|层次|hierarchy|安全.*约束.*首"],
            "neg_claims": [r"WBC.*无优先级|WBC.*等权|WBC.*flat.*weight"],
        },
        "Sim2Real 主流手段": {
            "terms": ["sim2real", "sim.to.real"],
            "pos_claims": [r"domain.*randomization|域随机|DR.*|adaptive.*domain"],
            "neg_claims": [r"sim2real.*无需随机|sim2real.*only.*real.*data"],
        },
        "Retargeting 运动学约束": {
            "terms": ["retarget", "重定向"],
            "pos_claims": [r"运动学.*约束|kinematic.*constraint|关节.*限制.*重定向|需要.*匹配.*运动学"],
            "neg_claims": [r"重定向.*仅.*角度|retarget.*just.*angle|retarget.*no.*constraint"],
        },
        "PPO Clip 范围": {
            "terms": ["PPO", "clip"],
            "pos_claims": [r"clip.*ratio|ε.*=.*0\.[12]|epsilon.*clip|裁剪.*比率"],
            "neg_claims": [r"PPO.*无.*clip|PPO.*不.*裁剪|clip.*unnecessary"],
        },
        "接触互补性条件": {
            "terms": ["complementarity|互补性|接触.*互补"],
            "pos_claims": [r"φ.*≥.*0|λ.*≥.*0|非负|non.negative|互补.*约束"],
            "neg_claims": [r"接触.*无约束|contact.*unconstrained|互补.*不必要"],
        },
    }
    all_pages_content = {p: p.read_text(encoding="utf-8") for p in pages}
    for fact_id, fact in CANONICAL_FACTS.items():
        pos_pages, neg_pages = [], []
        for page, content in all_pages_content.items():
            if not all(re.search(t, content, re.IGNORECASE) for t in fact["terms"]):
                continue
            has_pos = any(re.search(p, content, re.IGNORECASE) for p in fact["pos_claims"])
            has_neg = any(re.search(p, content, re.IGNORECASE) for p in fact["neg_claims"])
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
            results["log_inactive"].append("log.md 中未找到符合格式的操作记录（格式：## [YYYY-MM-DD] ...）")
    else:
        results["log_inactive"].append("log.md 文件不存在，无法检查知识库活跃度")

    return results

def format_report(results: dict) -> str:
    today = date.today().isoformat()
    lines = [f"## [{today}] lint | health-check | 自动化 wiki 健康检查", ""]

    total_issues = sum(len(v) for k, v in results.items() if not k.startswith("_"))
    lines.append(f"共发现 **{total_issues}** 个问题：")
    lines.append("")

    sections = [
        ("orphan_pages",       "孤儿页（无入链）",                           "⚠️"),
        ("missing_related",    "缺少关联页面区块",                           "⚠️"),
        ("missing_sources",    "缺少参考来源区块",                           "⚠️"),
        ("broken_links",       "断链（内链目标不存在）",                      "❌"),
        ("broken_source_refs", "引用了不存在的 sources/ 文件",                "❌"),
        ("sources_orphans",    "Sources 孤儿（sources/papers 死链）",         "❌"),
        ("stale_pages",        "陈旧页面（sources 比 wiki 新，建议 review）", "⚠️"),
        ("outdated_pages",     "可能过期（updated: 距今 > 180 天）",          "⚠️"),
        ("contradictions",     "潜在矛盾（跨页面相反定性描述）",              "⚠️"),
        ("stub_pages",         "空壳页面（< 200 字）",                       "⚠️"),
        ("missing_pages",      "频繁提及但缺少 wiki 页面的概念",              "💡"),
        ("missing_type",       "Frontmatter 缺少 type 字段",                 "⚠️"),
        ("log_inactive",       "log.md 活跃度警告",                          "⚠️"),
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
    parser.add_argument("--report", action="store_true",
                        help="将 markdown 健康报告保存到 exports/lint-report.md")
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
            f.write(f"# Wiki 健康报告\n\n")
            f.write(report)
        print(f"\n已将健康报告保存到 {report_path}")

    if total > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
