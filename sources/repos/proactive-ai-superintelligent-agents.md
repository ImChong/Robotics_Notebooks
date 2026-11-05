# Proactive AI for Superintelligent Agents 官方仓库

> 来源归档（repo）

- **标题：** EmpathYang/Proactive-AI-for-Superintelligent-Agents
- **类型：** repo（综述配套资源）
- **链接：** <https://github.com/EmpathYang/Proactive-AI-for-Superintelligent-Agents>
- **许可：** MIT（`LICENSE`，Copyright 2026 EmpathYang）
- **论文：** [sources/papers/proactive_ai_superintelligent_agents.md](../papers/proactive_ai_superintelligent_agents.md)
- **项目页：** [proactive-ai-superintelligent-agents-site.md](../sites/proactive-ai-superintelligent-agents-site.md)
- **入库日期：** 2026-11-05
- **一句话说明：** 综述的文献库与网站资源：按六项能力与 13 个应用领域整理其所引 1,410 篇文献；没有主动式智能体实现。

## 目录与内容（浅克隆，静态阅读）

| 路径 | 内容 |
|------|------|
| `README.md` | 引用 BibTeX、目录、按六项能力 / 13 领域分组的文献表（由 `scripts/papers.py readme` 生成） |
| `data/papers.json` | 文献库：`count` = 1,410，`updated` = 2026-10-07；1,402 条 `verified`、8 条 `needs-metadata` |
| `data/taxonomy.json` `evaluation.json` `maturity.json` `applications.json` `site.json` | 六项能力分类、评估层级、13 领域成熟度、代表系统在六项能力上的映射、站点文案与 BibTeX |
| `scripts/papers.py` | 仅用标准库：`validate`（校验条目与分类）、`readme [--check]`（重新生成 README 的文献表）、`stats`（按能力 / 领域 / 年份 / 状态计数） |
| `index.html` `site/` `classic/` | 网站与视频概览 |
| `Proactive-AI-for-Superintelligent-Agents.pdf` `paper.html` | 论文 PDF 与 HTML |

## 开放范围结论

- 开放的是**综述与文献整理**，不含主动式智能体、训练代码、模型或评测基准的实现。
- **未运行任何仓库代码**（本环境拒绝执行下载的仓库脚本）；以上是静态阅读结果。
