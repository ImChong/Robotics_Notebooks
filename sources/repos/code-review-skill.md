# Code Review Skill（awesome-skills/code-review-skill）

> 原始资料归档

- **标题：** Code Review Skill / 模块化代码审查技能
- **类型：** repo
- **作者：** awesome-skills
- **链接：** https://github.com/awesome-skills/code-review-skill
- **项目主页：** https://awesome-skills.github.io/code-review-skill/
- **代码：** https://github.com/awesome-skills/code-review-skill（已开源，MIT）
- **入库日期：** 2026-11-11；按当日 main 公开文件核对，未锁定具体 commit
- **核对范围：** 通读根目录 `SKILL.md`；阅读 `README.md` 的中英功能/安装/分发/指南清单；阅读 `scripts/pr-analyzer.py` 运行入口、解析器、复杂度与风险算法、输出逻辑；检查 `scripts/test_pr_analyzer.py` 测试分类、`LICENSE` 和项目官方主页。未逐篇通读 `reference/` 下全部语言指南，未执行上游测试。
- **通读：** 已完成
- **一句话说明：** 面向 AI 编码代理的四阶段 PR 审查 Skill，通过按需参考指南、分级反馈以及可选 Git diff 启发式分析器帮助审阅代码。
- **为什么值得保留：** 为机器人常见的 C++、Python、Qt 代码审阅提供结构化检查框架；强调语义问题与自动格式检查分离。
- **沉淀到 wiki：** [Code Review Skill](../../wiki/entities/code-review-skill.md)
- **项目站归档：** [官方展示页](../sites/code-review-skill.md)

## 官方原文：`SKILL.md`

- **Phase 1 — Context Gathering：** 阅读 PR 背景、关联 issue、规模、CI 状态、需求和架构决策。对大型 diff 可先运行 `scripts/pr-analyzer.py`。
- **Phase 2 — High-Level Review：** 查看架构/性能、文件组织及测试策略；按需调阅 `reference/architecture-review-guide.md` 和 `reference/performance-review-guide.md`。
- **Phase 3 — Line-by-Line：** 检查逻辑和边界、安全、并发、性能、可维护性及已有模块复用；语言指南从 `reference/` 按需加载。
- **Phase 4 — Summary & Decision：** 汇总问题、指出优点，给出 Approve / Comment / Request Changes 结论。
- **分级标签：** `blocking`、`important`、`nit` 代表三种优先程度；`suggestion`、`learning`、`praise` 是非阻塞反馈类别。
- **自动化分工：** 格式、import 排序、常规 lint 留给工具，评审重点是行为、设计和可维护性。Skill 是文本式指南，并不自动运行 CI。

## 源码阅读：`scripts/pr-analyzer.py`

- **入口：** `git diff main...HEAD | python scripts/pr-analyzer.py` 或 `python scripts/pr-analyzer.py --diff-file changes.diff --stats`。
- `main()` 从标准输入或 `--diff-file` 读取文本，校验非空，调用 `analyze_pr(diff_content)`。
- `parse_diff` 识别 `diff --git` 头、`+/-` 变更行，建立 `FileStats`；`detect_language` 根据文件扩展名识别语言；`is_test_file` 和 `is_config_file` 依路径规则识别文件类型。
- `calculate_complexity` 以 0.4 改动行规模 + 0.2 文件规模 + 0.2 非测试行比例 + 0.2 语言数量多样性计算 0–1 **启发式**得分；`estimate_review_time` 由变更行与复杂度估算 5–120 分钟，并非实测耗时。
- `identify_risk_factors` 按改动超过 400 行、缺测试修改、低测试比例、安全敏感文件名关键词、迁移/SQL 及配置改动提示风险；`generate_suggestions` 补充语言相关关注项。
- `print_analysis` 将报告输出到 stdout，`--stats` 补充按语言分组的文件统计。
- **测试：** `test_pr_analyzer.py` 包含 diff 头/重命名路径、语言/测试文件、风险规则和端到端解析等用例；本次只静态核对，未运行。
- **源码边界：** 工具没有 GitHub PR 评论 API 调用、AST 分析、真正的漏洞检测或自动批准逻辑；敏感路径提醒不能代替安全审查。

## 上游来源与许可证

- [SKILL.md（完整评审规约）](https://github.com/awesome-skills/code-review-skill/blob/main/SKILL.md)
- [README.md（中英说明）](https://github.com/awesome-skills/code-review-skill/blob/main/README.md)
- [pr-analyzer.py（Python CLI）](https://github.com/awesome-skills/code-review-skill/blob/main/scripts/pr-analyzer.py)
- [test_pr_analyzer.py（测试）](https://github.com/awesome-skills/code-review-skill/blob/main/scripts/test_pr_analyzer.py)
- [reference/（20+ 语言/跨领域指南集合）](https://github.com/awesome-skills/code-review-skill/tree/main/reference)
- [LICENSE（MIT）](https://github.com/awesome-skills/code-review-skill/blob/main/LICENSE)

上游宣称覆盖 20+ 种语言/框架与 21,000+ 行指南；这是作者自述，未重新逐文件统计，也无缺陷检测准确率基准。对机器人代码仍需另检 ROS 2 调度、硬实时延迟和物理安全等专属边界。
