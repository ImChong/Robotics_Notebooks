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


def strip_misconception_sections(content: str) -> str:
    """移除“常见误区/误区”区块，避免把辟谣内容误判为事实矛盾。"""
    lines = content.splitlines()
    kept = []
    skip_level = None
    heading_re = re.compile(r'^(#{2,6})\s+(.*)$')
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
        "missing_summary": [],        # V10: concepts/methods/tasks 缺少 summary/description
        "query_format": [],           # V11: queries/ 缺少 Query 产物说明/参考来源/关联页面
        "formalization_no_formula": [],  # V11: formalizations/ 缺少公式块
        "readme_badge": [],           # V11: README checklist 链接版本不一致
        "orphan_count": [],           # V13: graph-stats.json 中孤儿节点（无入链）预警
        "method_missing_link": [],    # V15: methods/ 页面缺少指向 formalizations/ 或 concepts/ 的链接
        "method_missing_sections": [], # V17: methods/ 页面缺少标准区块
        "entity_missing_outgoing": [], # V20: entities/ 页面缺少指向 methods/ 或 tasks/ 的出边
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
            "neg_claims": [r"WBC.*完全没用|WBC.*没有.*多接触优势|WBC.*没必要.*多接触"],
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
        "DAgger 数据效率": {
            "terms": ["DAgger"],
            "pos_claims": [r"比.*BC.*数据效率.*高|解决.*分布漂移|处理.*covariate shift|缓解.*covariate shift"],
            "neg_claims": [r"与.*BC.*等价|不处理.*covariate shift|不处理.*分布漂移"],
        },
        "VLA 推理延迟": {
            "terms": ["VLA|Vision.Language.Action|RT-2|π₀|pi0"],
            "pos_claims": [r"50ms\+|50ms以上|推理延迟.*高|控制频率.*低|latency.*50"],
            "neg_claims": [r"实时性.*传统控制器相当|与.*传统控制器相当|低延迟.*1kHz|高频闭环.*无需缓冲"],
        },
        "行为克隆 compounding error": {
            "terms": ["Behavior Cloning|行为克隆|\bBC\b"],
            "pos_claims": [r"compounding error|错误累积|累积误差|covariate shift"],
            "neg_claims": [r"累积误差.*无关|序列长度.*无关|不会.*错误累积"],
        },
        "接触力摩擦约束": {
            "terms": ["接触|contact|摩擦锥|friction cone"],
            "pos_claims": [r"\|f_\{?xy\}?\|.*<=.*(μ|mu).*f_z|\|f_t\|.*<=.*mu.*f_n|摩擦锥.*约束|friction cone"],
            "neg_claims": [r"无需.*摩擦锥|不需要.*摩擦锥|contact force.*unconstrained"],
        },
        "腿足地形感知": {
            "terms": ["地形|terrain|腿足|locomotion|高度图|point cloud"],
            "pos_claims": [r"高度图|点云|height map|point cloud|地形感知"],
            "neg_claims": [r"不需要.*地形感知|无需.*高度图|无需.*点云"],
        },
        "MiniLM 向量维度": {
            "terms": ["MiniLM|all-MiniLM-L6-v2"],
            "pos_claims": [r"384.*维|384-dim|384维"],
            "neg_claims": [r"768.*维|768-dim|768维"],
        },
        "Marp 幻灯片格式": {
            "terms": ["Marp"],
            "pos_claims": [r"Markdown.*frontmatter|frontmatter.*Markdown|生成.*幻灯片"],
            "neg_claims": [r"需要.*LaTeX|需要.*Beamer|必须.*LaTeX"],
        },
        "sentence-transformers CPU": {
            "terms": ["sentence-transformers|SentenceTransformer"],
            "pos_claims": [r"CPU.*运行|可在.*CPU.*运行|无需.*GPU|without GPU"],
            "neg_claims": [r"必须.*GPU|only.*GPU|cannot.*CPU"],
        },
        "VLA 训练数据规模": {
            "terms": ["VLA|RT-1|RT-2|π₀|pi0"],
            "pos_claims": [r"大量.*演示|数千\+|130k\+|大规模.*数据|多样化.*演示"],
            "neg_claims": [r"十条.*演示|10条.*演示|少量.*演示.*收敛"],
        },
        "BM25 参数含义": {
            "terms": ["BM25|k1|\bb\b"],
            "pos_claims": [r"k1.*词频.*饱和|b.*长度归一化|长度归一化.*b|词频饱和.*k1"],
            "neg_claims": [r"b.*与词频无关|b.*控制词频|k1.*长度归一化"],
        },
        "π₀ Flow Matching": {
            "terms": ["π₀|pi0|Flow Matching|flow matching"],
            "pos_claims": [r"Flow Matching|flow matching|连续动作.*生成"],
            "neg_claims": [r"不.*用.*Flow Matching|仅.*Transformer.*直接回归|without.*flow matching"],
        },
        "CBF 安全集条件": {
            "terms": ["CBF|Control Barrier Function|控制障碍函数"],
            "pos_claims": [r"h\(x\).*≥.*0|安全集.*h.*≥|维持.*安全集|超平面.*h.*≥.*0"],
            "neg_claims": [r"CBF.*直接优化.*性能|CBF.*最大化.*奖励|CBF.*优化.*目标函数"],
        },
        "CLF 指数衰减条件": {
            "terms": ["CLF|Control Lyapunov Function|控制李雅普诺夫"],
            "pos_claims": [r"V̇.*≤.*-.*α.*V|衰减.*条件|指数.*衰减|dot.*V.*≤.*-.*alpha"],
            "neg_claims": [r"CLF.*无需.*衰减|Lyapunov.*无衰减条件|CLF.*不要求.*导数.*负"],
        },
        "双臂闭链约束": {
            "terms": ["双臂|bimanual|dual.arm"],
            "pos_claims": [r"闭链|closed.loop.*kinematic|closed-loop.*kinematic|闭环运动学|闭链约束"],
            "neg_claims": [r"双臂.*独立.*单臂|bimanual.*two.*independent|双臂.*各自独立"],
        },
        "Anki TSV 字段分隔符": {
            "terms": ["Anki|anki", "TSV|tsv"],
            "pos_claims": [r"制表符|Tab.*分隔|tab.separated|\t.*分隔符"],
            "neg_claims": [r"逗号.*分隔|comma.*separated|CSV.*Anki|Anki.*CSV"],
        },
        "WBC QP 求解时间": {
            "terms": ["WBC|whole.body", "QP|二次规划"],
            "pos_claims": [r"50.*200.*μs|OSQP.*微秒|QP.*求解.*50|QP.*μs"],
            "neg_claims": [r"WBC.*QP.*秒级|WBC.*求解.*几秒|QP.*second.*WBC"],
        },
        "特权训练非对称 Actor": {
            "terms": ["特权训练|privileged.*training|teacher.student|asymmetric.*actor"],
            "pos_claims": [r"非对称|asymmetric.*actor|Teacher.*有.*特权|Student.*无.*特权|训练时.*额外"],
            "neg_claims": [r"特权训练.*对称|teacher.*student.*相同.*观测|privileged.*symmetric"],
        },
        "Foundation Policy 预训练数据规模": {
            "terms": ["foundation.*policy|基础策略模型|RT-1|Octo"],
            "pos_claims": [r"130k\+|大规模.*演示|多机器人.*形态|800k|跨.*任务.*预训练"],
            "neg_claims": [r"RT-1.*少量.*数据|foundation.*policy.*百条|基础策略.*低数据"],
        },
        "模仿学习分布外泛化": {
            "terms": ["imitation.*learning|模仿学习|behavior.*cloning|行为克隆"],
            "pos_claims": [r"分布外.*失败|out.*of.*distribution|分布偏移|covariate shift|compounding"],
            "neg_claims": [r"模仿学习.*分布外.*稳定|BC.*不受.*分布偏移.*影响|IL.*robust.*distribution"],
        },
        "HQP 优先级层次": {
            "terms": ["HQP|Hierarchical.*QP|层次化.*QP"],
            "pos_claims": [r"优先级|priority|层次|hierarchy|高优先级.*先满足"],
            "neg_claims": [r"HQP.*无优先级|HQP.*等权.*优化|hierarchical.*no.*priority"],
        },
        "PPO on-policy 特性": {
            "terms": ["PPO"],
            "pos_claims": [r"PPO.*on.policy|PPO.*每次.*更新.*丢弃|PPO.*不.*复用.*历史数据"],
            "neg_claims": [r"PPO 是 off.policy|PPO 属于 off.policy|PPO 本质上.*off.policy|PPO.*replay buffer.*来.*复用"],
        },
        "Contact-rich 接触力建模": {
            "terms": ["contact.rich|接触丰富|接触.*丰富"],
            "pos_claims": [r"显式.*建模.*接触|接触力.*建模|force.*model|顺应性.*控制|impedance"],
            "neg_claims": [r"contact.rich.*忽略.*接触|接触丰富.*不需要.*力控|contact.*force.*ignored"],
        },
        "CLF 收敛驱动目标": {
            "terms": ["CLF|Control Lyapunov|控制李雅普诺夫"],
            "pos_claims": [r"CLF.*收敛.*平衡|CLF.*驱动.*收敛|CLF.*渐近稳定|Control Lyapunov.*stabiliz"],
            "neg_claims": [r"CLF 用于安全边界|CLF 就是 CBF|CLF 等同于 CBF|CLF.*safety barrier function"],
        },
        "CBF vs CLF 功能区别": {
            "terms": ["CBF|CBF.*CLF|CLF.*CBF"],
            "pos_claims": [r"CBF.*安全.*CLF.*收敛|安全集.*CBF.*稳定性.*CLF|CBF.*维持.*CLF.*驱动"],
            "neg_claims": [r"CBF.*CLF.*功能相同|CBF.*与.*CLF.*等价|CBF.*替代.*CLF"],
        },
        "VLA 输出空间": {
            "terms": ["VLA|Vision.Language.Action"],
            "pos_claims": [r"末端执行器.*位姿|end.effector.*pose|EE.*pose|目标位姿.*输出"],
            "neg_claims": [r"VLA.*直接.*关节.*力矩|VLA.*joint.*torque.*direct|VLA.*关节级.*指令"],
        },
        "双臂内力协调": {
            "terms": ["双臂|bimanual|dual.arm"],
            "pos_claims": [r"协调规划|内力.*避免|avoid.*internal.*force|closed.loop.*coordination|协调.*内力"],
            "neg_claims": [r"双臂.*各自.*独立.*规划|bimanual.*no.*coordination|双臂.*内力.*无需处理"],
        },
        "Isaac Lab GPU 并行规模": {
            "terms": ["Isaac Lab|IsaacLab|isaac.*lab"],
            "pos_claims": [r"GPU.*并行|并行.*环境|数千.*环境|thousands.*enviro|massive.*parallel"],
            "neg_claims": [r"Isaac Lab.*不.*并行|Isaac Lab.*单环境|IsaacLab.*no.*GPU"],
        },
        "Domain Randomization 随机扰动方式": {
            "terms": ["domain.randomization|域随机|DR.*参数|随机化.*参数"],
            "pos_claims": [r"随机.*扰动|随机化.*参数|参数.*分布|parameter.*distribution|uniform.*range"],
            "neg_claims": [r"DR.*固定.*参数|domain.*random.*fixed.*param|随机化.*无效"],
        },
        "Diffusion Policy 多模态能力": {
            "terms": ["diffusion.policy|扩散策略"],
            "pos_claims": [r"多模态|multimodal|复杂.*操作|multi.modal.*action|多峰.*分布"],
            "neg_claims": [r"diffusion.*单模态|diffusion.*unimodal|扩散.*单峰.*分布"],
        },
        "VLA 高延迟异步执行": {
            "terms": ["VLA|Vision.Language.Action"],
            "pos_claims": [r"100ms|异步.*执行|async.*execution|action.*chunk|动作.*块.*缓冲"],
            "neg_claims": [r"VLA.*无需异步|VLA.*同步.*控制|VLA.*低延迟.*同步"],
        },
        "接触丰富顺应性控制": {
            "terms": ["contact.rich|接触丰富", "force.*control|impedance|力控|阻抗"],
            "pos_claims": [r"力控|阻抗控制|impedance.*control|force.*control|顺应性.*控制|compliance"],
            "neg_claims": [r"contact.rich.*纯位置控制|接触丰富.*位置控制.*足够|contact.*pure.*position.*control"],
        },
        "Impedance Control 模型": {
            "terms": ["impedance.*control|阻抗控制"],
            "pos_claims": [r"质量.*弹簧.*阻尼|mass.*spring.*damper|M_d.*B_d.*K_d|虚拟.*物理系统"],
            "neg_claims": [r"阻抗控制.*无阻尼|阻抗控制.*纯刚性|阻抗控制.*等同PID"],
        },
        "Impedance vs Admittance": {
            "terms": ["impedance|阻抗", "admittance|导纳"],
            "pos_claims": [r"阻抗.*位移.*力|导纳.*力.*位移|输入.*误差.*输出.*力矩"],
            "neg_claims": [r"阻抗.*导纳.*相同|阻抗控制.*输入力|导纳控制.*输出力"],
        },
        "Impedance 柔顺性": {
            "terms": ["impedance.*control|阻抗控制"],
            "pos_claims": [r"主动柔顺|active.*compliance|接触力.*安全|退让"],
            "neg_claims": [r"阻抗控制.*位置.*绝对|阻抗控制.*不具备柔顺|缺乏柔顺"],
        },
        "Impedance 阻尼调参": {
            "terms": ["impedance.*control|阻抗控制", "阻尼|damper|B_d"],
            "pos_claims": [r"临界阻尼|critical.*damping|防止.*震荡|往复振荡"],
            "neg_claims": [r"阻尼.*越大越好|阻尼.*无需调整|阻尼.*导致.*震荡"],
        },
        "Trajectory Optimization 离线": {
            "terms": ["trajectory.*optimization|轨迹优化"],
            "pos_claims": [r"离线规划|offline|长时域|开环|非线性规划|NLP"],
            "neg_claims": [r"轨迹优化.*必须在线|轨迹优化.*等于.*MPC|完全实时"],
        },
        "Direct Collocation 约束": {
            "terms": ["direct.*collocation|直接配点"],
            "pos_claims": [r"状态.*控制.*同时优化|动力学.*等式约束|隐式.*接触"],
            "neg_claims": [r"只优化控制|只优化状态|无动力学约束"],
        },
        "ANYmal SEA": {
            "terms": ["ANYmal|anymal"],
            "pos_claims": [r"串联弹性执行器|SEA|力控执行器|碰撞缓冲"],
            "neg_claims": [r"ANYmal.*纯刚性电机|ANYmal.*准直接驱动|ANYmal.*QDD"],
        },
        "ANYmal 鲁棒性": {
            "terms": ["ANYmal|anymal"],
            "pos_claims": [r"防水防尘|IP67|工业环境|极端.*地形|RMA|ANYmal.*C"],
            "neg_claims": [r"ANYmal.*仅限室内|ANYmal.*实验室.*环境|ANYmal.*无感知"],
        },
        "WBC 权重分配": {
            "terms": ["WBC|whole.body.control"],
            "pos_claims": [r"躯干姿态.*高优先级|相对比例|支撑足.*硬约束|Orientation.*Support"],
            "neg_claims": [r"WBC.*所有任务.*等权|躯干姿态.*低优先级|权重.*绝对值"],
        },
        "WBC 约束松弛": {
            "terms": ["WBC|whole.body.control", "约束|constraint"],
            "pos_claims": [r"松弛变量|slack.*variable|惩罚项|软约束"],
            "neg_claims": [r"WBC.*必须硬约束|WBC.*不可松弛|松弛变量.*无用"],
        },
        "WBC 力矩正则化": {
            "terms": ["WBC|whole.body.control", "正则|regularization"],
            "pos_claims": [r"惩罚.*力矩|稳定.*数值|Hessian.*正定|阻尼项"],
            "neg_claims": [r"WBC.*无需正则化|力矩正则化.*有害|雅可比.*永远满秩"],
        },
        "Boston Dynamics 控制": {
            "terms": ["boston.*dynamics|波士顿动力|Atlas"],
            "pos_claims": [r"基于优化|optimization.*based|WBC.*MPC|全身控制"],
            "neg_claims": [r"纯.*强化学习|完全弃用.*控制|波士顿动力.*只用.*RL"],
        },
        "Boston Dynamics 硬件": {
            "terms": ["boston.*dynamics|波士顿动力|Atlas"],
            "pos_claims": [r"高功率密度|液压.*全电|定制电机"],
            "neg_claims": [r"Atlas.*一直液压|全电版.*未发布|低成本电机"],
        },
        "Reward Shaping 平滑": {
            "terms": ["reward.*shaping|奖励函数", "smoothness|平滑|抖动"],
            "pos_claims": [r"惩罚.*功耗|惩罚.*加速度|action.*rate|防止.*振荡"],
            "neg_claims": [r"平滑项.*不重要|只需.*正奖励|无需惩罚.*加速度"],
        },
        "Reward Shaping Air Time": {
            "terms": ["air.*time|腾空时间"],
            "pos_claims": [r"落地后.*结算|诱导.*迈步|产生.*节奏"],
            "neg_claims": [r"Air Time.*随时给|Air Time.*导致.*站立|不需要.*落地结算"],
        },
        "Locomotion 原地踏步": {
            "terms": ["locomotion.*failure|失败模式|原地踏步"],
            "pos_claims": [r"前进奖励.*低|惩罚.*过重|curriculum|增加.*速度权重"],
            "neg_claims": [r"原地踏步.*是好现象|无需.*调整奖励|惩罚.*越重越好"],
        },
        "Locomotion 策略崩溃": {
            "terms": ["locomotion.*failure|策略崩溃|policy.*collapse"],
            "pos_claims": [r"学习率.*过大|随机性.*太大|DR.*范围|减小.*学习率"],
            "neg_claims": [r"策略崩溃.*学习率.*太小|DR.*范围.*必须.*最大|无需.*衰减"],
        },
        "Locomotion 双脚起跳": {
            "terms": ["locomotion.*failure|失败模式|双脚起跳|Bunny.*Hop"],
            "pos_claims": [r"Air Time.*滥用|缺少.*步态约束|限制.*最大值|对角步态"],
            "neg_claims": [r"双脚起跳.*正常|增加.*Air Time|不需要.*约束"],
        },
        "ZMP 稳定性": {
            "terms": ["ZMP|Zero.*Moment.*Point"],
            "pos_claims": [r"支撑多边形|support.*polygon|落在.*内|不翻倒"],
            "neg_claims": [r"ZMP.*可以.*外|支撑多边形.*无关|落在边缘.*安全"],
        },
        "CMDP 拉格朗日法": {
            "terms": ["CMDP|Constrained.*MDP|Lagrangian|拉格朗日"],
            "pos_claims": [r"转化为.*惩罚项|交替更新|权重|乘子"],
            "neg_claims": [r"不使用.*乘子|拉格朗日.*无效|只.*CPO"],
        },
        "VLA Action Tokenization": {
            "terms": ["VLA|Vision.Language.Action", "分词|tokeniz"],
            "pos_claims": [r"离散.*Token|映射|Scalar.*Binning|分箱|Codebook|量化"],
            "neg_claims": [r"VLA.*不使用.*Token|VLA.*纯连续输出|无需量化"],
        },
        "Cross-modal Attention 作用": {
            "terms": ["Cross.modal.*Attention|跨模态.*注意力"],
            "pos_claims": [r"关联|视.*语.*控| RGB.*指令|对齐|拼接到.*序列"],
            "neg_claims": [r"模态间.*独立|注意力.*无关|无需关注指令"],
        },
        "Allegro Hand 自由度": {
            "terms": ["Allegro.*Hand|灵巧手"],
            "pos_claims": [r"16.*自由度|16.*DOFs|4.*指|4.*fingers"],
            "neg_claims": [r"Allegro.*拥有.*20.*自由度|Allegro.*具备.*5.*指"],
        },
        "In-hand Reorientation 定义": {
            "terms": ["In.hand.*Reorientation|手内重定向"],
            "pos_claims": [r"不借助于.*外部环境|不利用.*桌面|手指间的协同|改变.*位姿"],
            "neg_claims": [r"依赖.*桌面|必须放回桌面|非手内"],
        },
        "Preempt_RT 实时性": {
            "terms": ["PREEMPT_RT|硬实时"],
            "pos_claims": [r"补丁|内核源码|SCHED_FIFO|消除.*抖动|消除.*Jitter"],
            "neg_claims": [r"普通.*Ubuntu.*足够|无需内核补丁|实时性.*无关"],
        },
        "CPU Isolation 效果": {
            "terms": ["isolcpus|核心隔离"],
            "pos_claims": [r"GRUB|预留.*核心|屏蔽.*中断|零中断|独占"],
            "neg_claims": [r"CPU.*自动平衡|隔离.*无用|无需隔离核心"],
        },
        "LCM 协议特征": {
            "terms": ["LCM|UDP.*组播|组播"],
            "pos_claims": [r"UDP.*组播|Multicast|极低延迟|1000Hz|无.*重传"],
            "neg_claims": [r"LCM.*基于.*TCP|LCM.*高开销|保证.*重传"],
        },
        "Pinocchio 核心优势": {
            "terms": ["Pinocchio|刚体动力学库"],
            "pos_claims": [r"解析导数|Analytical.*Derivatives|模板元编程|Eigen|Featherstone"],
            "neg_claims": [r"Pinocchio.*无法.*计算.*导数|Pinocchio.*严重依赖.*动态内存分配"],
        },
        "MuJoCo 接触模型": {
            "terms": ["MuJoCo|mjcf"],
            "pos_claims": [r"凸优化|convex.*optimization|稳定|平滑.*离散化|解析.*求导"],
            "neg_claims": [r"MuJoCo.*底层.*是.*惩罚冲量|MuJoCo.*接触.*极不稳定|MuJoCo.*不支持.*解析.*求导"],
        },
        "Drake 轨迹优化": {
            "terms": ["Drake|Tedrake"],
            "pos_claims": [r"为优化而生|直接配点|Direct.*Collocation|NLP.*构建|TRI"],
            "neg_claims": [r"Drake.*仅仿真|Drake.*无优化|不支持.*OCP"],
        },
        "混合中间件架构": {
            "terms": ["ROS.*2", "LCM"],
            "pos_claims": [r"分层|ROS.*2.*大脑|LCM.*小脑|ROS.*2.*中高层|LCM.*底层"],
            "neg_claims": [r"底层.*必须.*ROS.*2|LCM.*负责感知|LCM.*和.*ROS.*2.*竞争"],
        },
        "GelSight 触觉原理": {
            "terms": ["GelSight|视觉.*触觉"],
            "pos_claims": [r"摄像头|硅胶|形变图像|分辨率|千万像素"],
            "neg_claims": [r"GelSight.*电阻|GelSight.*无摄像头|仅测量压力"],
        },
        "BC Loss 协变量偏移": {
            "terms": ["Behavior.*Cloning|BC.*Loss", "偏移|shift"],
            "pos_claims": [r"协变量偏移|Covariate.*Shift|错误累积|Compounding.*Error"],
            "neg_claims": [r"BC.*无偏移|BC.*绝对鲁棒|数据越多越好"],
        },
        "概率流 ODE 效率": {
            "terms": ["概率流", "Probability.Flow"],
            "pos_claims": [r"1-3.*步|加速|确定性|ODE.*求解器|实时性"],
            "neg_claims": [r"概率流.*慢于.*扩散|概率流.*需要.*100.*步"],
        },
        "流匹配目标函数": {
            "terms": ["流匹配|Flow.*Matching"],
            "pos_claims": [r"回归.*速度场|velocity.*field|最小化.*速度.*差异|线性插值"],
            "neg_claims": [r"流匹配.*最小化.*Q值|流匹配.*是.*分类任务"],
        },
        "潜空间想象优势": {
            "terms": ["潜空间想象|Latent.*Imagination|Dreamer"],
            "pos_claims": [r"无需.*真实.*交互|快.*100-1000.*倍|过滤.*背景.*噪声|脑海.*预演"],
            "neg_claims": [r"想象.*必须.*真实采样|潜空间.*比.*物理仿真.*慢"],
        },
        "RSSM 构成": {
            "terms": ["RSSM|Recurrent.*State.*Space.*Model"],
            "pos_claims": [r"Transition|Observation|Reward|确定性.*随机.*状态"],
            "neg_claims": [r"RSSM.*无随机项|RSSM.*仅包含.*卷积"],
        },
        "生成式世界模型定义": {
            "terms": ["生成式世界模型|Generative.*World.*Models"],
            "pos_claims": [r"以生成代替计算|海量视频数据|模拟.*运动规律|反事实推演"],
            "neg_claims": [r"底层.*基于.*力学方程|必须.*解析.*计算.*接触力"],
        },
        "Video-as-Simulation 差异": {
            "terms": ["Video.as.Simulation"],
            "pos_claims": [r"算像素|Next-frame.*Prediction|无鸿沟|无需.*手工.*建模"],
            "neg_claims": [r"必须.*建模.*几何体|Video.as.Simulation.*计算.*冲量"],
        },
        "电池压降风险": {
            "terms": ["电池.*压降|Voltage.*Sag"],
            "pos_claims": [r"瞬时.*掉电|主板.*重启|欠压|独立.*供电"],
            "neg_claims": [r"压降.*无影响|电池.*电压.*恒定"],
        },
        "关节过热对策": {
            "terms": ["关节.*过热|散热|Thermal"],
            "pos_claims": [r"能效项|主动散热|导热硅脂|减少.*对抗.*出力"],
            "neg_claims": [r"过热.*必须.*停机|散热.*无关.*控制"],
        },
        "野外感知失效": {
            "terms": ["野外|草地|户外", "感知|LiDAR|相机"],
            "pos_claims": [r"过曝|镜面反射|点云.*嘈杂|强度过滤"],
            "neg_claims": [r"户外.*感知.*更准|感知.*无需.*增强"],
        },
        "地面硬度不匹配": {
            "terms": ["地面硬度|草地|刚性地面"],
            "pos_claims": [r"阻尼.*弹性|深陷泥潭|降低.*Kp|刚度随机化"],
            "neg_claims": [r"草地.*等同.*硬地|Kp.*越高越好"],
        },
        "SE(3) 6D 表示优点": {
            "terms": ["6D.*连续表示|6D.*Rep"],
            "pos_claims": [r"姿态空间连续|适合.*回归|平滑.*梯度"],
            "neg_claims": [r"6D.*表示.*不连续|不如.*欧拉角"],
        },
        "Oculus Quest 遥操作": {
            "terms": ["Quest|VR.*遥操作"],
            "pos_claims": [r"低成本|Inside-out.*Tracking|沉浸式|6D.*位姿"],
            "neg_claims": [r"Quest.*极其昂贵|Quest.*不支持.*手部追踪"],
        },
        "Omniverse 核心底座": {
            "terms": ["Omniverse|USD"],
            "pos_claims": [r"实时协作|数字化孪生|OpenUSD|高保真.*渲染"],
            "neg_claims": [r"Omniverse.*仅限游戏|Omniverse.*不支持.*物理"],
        },
        "生成式数据增强重点": {
            "terms": ["生成式数据增强|Generative.*Data.*Augmentation"],
            "pos_claims": [r"长尾合成|物体变幻|场景编辑|解决.*瓶颈"],
            "neg_claims": [r"仅限.*裁剪翻转|增强.*无效"],
        },
        "ELBO 目标": {
            "terms": ["ELBO|证据下界|变分"],
            "pos_claims": [r"重建项|正则项|最小化.*KL散度|一致性"],
            "neg_claims": [r"ELBO.*最大化.*误差|ELBO.*不含.*KL"],
        },
        "开源人形硬件方案": {
            "terms": ["Berkeley.*Humanoid|Roboto.*Origin|ODRI"],
            "pos_claims": [r"低成本|验证.*算法|QDD.*电机|12-14.*自由度"],
            "neg_claims": [r"开源硬件.*昂贵|仅限.*工业.*应用"],
        },
        "WBC 奇异点保护": {
            "terms": ["奇异点|Singularity"],
            "pos_claims": [r"膝关节.*伸直|虚拟弹簧力|雅可比.*退化|力矩.*爆炸"],
            "neg_claims": [r"奇异点.*无需保护|奇异点.*力矩.*最小"],
        },
        "具身数据清洗": {
            "terms": ["数据清洗|Data.*Cleaning"],
            "pos_claims": [r"时序对齐|剔除.*无效动作|修复.*重定向误差|动作分段"],
            "neg_claims": [r"原始数据.*完美|无需清洗"],
        },
        "MuJoCo 选型理由": {
            "terms": ["MuJoCo|mjcf"],
            "pos_claims": [r"基于模型.*控制|Model.based.*Control|推导动力学|MPC|iLQR|轨迹优化"],
            "neg_claims": [r"MuJoCo.*只适合.*Model.free|MuJoCo.*不支持.*MPC"],
        },
        "Isaac Sim 选型理由": {
            "terms": ["Isaac.*Sim|Isaac.*Gym"],
            "pos_claims": [r"千万级并行|千万.*机器人|GPU.*采样|经验采样速度|光线追踪|RTX"],
            "neg_claims": [r"Isaac.*Sim.*采样慢|Isaac.*Sim.*仅限 CPU|Isaac.*Sim.*无渲染"],
        },
        "Single Shooting 缺点": {
            "terms": ["Single.*Shooting|单重打靶"],
            "pos_claims": [r"高度非线性|容易陷入.*局部最优|蝴蝶效应|难以处理.*中间.*状态.*约束"],
            "neg_claims": [r"单重打靶.*处理约束.*极佳|单重打靶.*是线性系统"],
        },
        "Direct Collocation 优点": {
            "terms": ["Direct.*Collocation|直接配点"],
            "pos_claims": [r"处理.*极多.*状态约束|初始猜测.*物理不可行.*依然收敛|隐式.*动力学"],
            "neg_claims": [r"直接配点.*无法处理约束|初始猜测.*必须.*可行"],
        },
        "Contact-Implicit 核心": {
            "terms": ["Contact.Implicit|接触隐式"],
            "pos_claims": [r"互补.*约束|complementarity|无需.*手工.*编排.*时序|自主发现.*步态"],
            "neg_claims": [r"接触隐式.*需要.*预设.*时序|手动编排.*落地"],
        },
        "VLA 真机分层": {
            "terms": ["VLA|Vision.Language.Action", "真机|部署"],
            "pos_claims": [r"中高层策略|不要直接.*驱动.*执行器|低频.*高频.*分层|action.*chunk"],
            "neg_claims": [r"VLA.*直接.*1kHz.*闭环|VLA.*控制.*脊髓"],
        },
        "SCHED_FIFO 风险": {
            "terms": ["SCHED_FIFO|实时线程"],
            "pos_claims": [r"死循环.*卡死|忘记.*sleep|优先级|拔电源"],
            "neg_claims": [r"SCHED_FIFO.*绝对安全|系统.*自动.*恢复"],
        },
        "CPU Isolation 配置": {
            "terms": ["isolcpus|nohz_full"],
            "pos_claims": [r"GRUB|屏蔽.*tick|RCU.*回调|屏蔽.*中断"],
            "neg_claims": [r"isolcpus.*无法屏蔽中断|isolcpus.*内核级.*无关"],
        },
        "RT-1 动作分词": {
            "terms": ["RT-1|Action.*Tokenization"],
            "pos_claims": [r"标量分箱|Scalar.*Binning|256.*bin|独立量化"],
            "neg_claims": [r"RT-1.*连续回归|RT-1.*不分词"],
        },
        "In-hand Reorientation 挑战": {
            "terms": ["In.hand.*Reorientation|手内重定向"],
            "pos_claims": [r"频繁.*接触切换|感知遮挡|滑移|Slip|离散非线性"],
            "neg_claims": [r"重定向.*无遮挡|接触.*恒定|重定向.*是线性系统"],
        },
        "π₀ 核心技术": {
            "terms": ["π₀|Pi-zero"],
            "pos_claims": [r"流匹配|Flow.*Matching|Scaling.*Law|规模法则|后训练"],
            "neg_claims": [r"π₀.*基于.*DQN|π₀.*无预训练"],
        },
        "Allegro Hand 优势": {
            "terms": ["Allegro.*Hand|灵巧手"],
            "pos_claims": [r"结构精简|4指|易于集成触觉|HandLib|DexDex"],
            "neg_claims": [r"Allegro.*太复杂|Allegro.*无法加触觉"],
        },
        "Embodied Data Cleaning 重点": {
            "terms": ["Data.*Cleaning|数据清洗"],
            "pos_claims": [r"时序对齐|剔除.*异常|重定向误差|插值|重新采样"],
            "neg_claims": [r"数据.*无需清洗|脏数据.*直接训练"],
        },
        "Actuator Network 作用": {
            "terms": ["Actuator.*Network|执行器网络"],
            "pos_claims": [r"拟合.*真实电机|非线性特性|反向电动势|摩擦|延迟特性|TorchScript"],
            "neg_claims": [r"执行器网络.*拟合传感器|执行器网络.*代替策略"],
        },
    }
    all_pages_content = {p: strip_misconception_sections(p.read_text(encoding="utf-8")) for p in pages}
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
        if len(parts) < 2 or parts[0] != "wiki" or parts[1] != "queries" or page.name == "README.md":
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

    # V11: README.md 中 badges / checklist 链接应与当前仓库状态一致
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")

        checklist_files = sorted((REPO_ROOT / "docs" / "checklists").glob("tech-stack-next-phase-checklist-v*.md"))
        if checklist_files:
            latest_checklist = max(
                checklist_files,
                key=lambda p: int(re.search(r"v(\d+)", p.stem).group(1))
            )
            latest_ver = int(re.search(r"v(\d+)", latest_checklist.stem).group(1))

            main_link_versions = re.findall(
                r"\[技术栈项目执行清单 v(\d+)\]", readme_content
            )
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
            if len(target_parts) >= 2 and target_parts[0] == "wiki" and target_parts[1] in ("formalizations", "concepts"):
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
                if len(t_parts) >= 2 and t_parts[0] == "wiki" and t_parts[1] in ("methods", "tasks"):
                    out_count += 1
            if out_count < 2:
                results["entity_missing_outgoing"].append(f"{rel} (当前出边: {out_count})")

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
        ("missing_summary",    "缺少摘要字段（summary/description）",         "⚠️"),
        ("query_format",       "Query 页面格式不完整（缺 Query 产物/参考来源/关联页面）", "⚠️"),
        ("formalization_no_formula", "Formalization 页面缺少公式块",              "⚠️"),
        ("readme_badge",       "README checklist 链接版本不一致",               "⚠️"),
        ("orphan_count",       "图谱孤儿节点预警（graph-stats.json）",           "⚠️"),
        ("method_missing_link","Methods 页面缺少 Formalization/Concept 链接", "⚠️"),
        ("method_missing_sections","Methods 页面缺少主要路线区块", "⚠️"),
        ("entity_missing_outgoing","Entities 页面缺少 Methods/Tasks 关联出边", "⚠️"),
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
