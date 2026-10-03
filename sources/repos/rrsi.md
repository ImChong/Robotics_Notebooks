# RRSI（Google Research 官方代码仓库）

> 来源归档

- **标题：** RRSI: Regularized Recursive Self-Improvement of Agent Harnesses
- **类型：** repo / agent-harness / recursive-self-improvement
- **组织：** Google Research
- **链接：** <https://github.com/google-research/rrsi>
- **项目页：** <https://regularized-rsi.com/>
- **论文：** [arXiv:2609.24972v2](https://arxiv.org/abs/2609.24972v2)
- **许可证：** Apache-2.0
- **语言：** Python
- **入库日期：** 2026-10-03
- **一句话说明：** 公开的 agent harness RSI 研究实现：提供 proposal / critic / evaluation / selection 闭环，以及 coding、agentic workspace、engineering design 三种实例入口。
- **沉淀到 wiki：** [wiki/entities/paper-rrsi-2609-24972.md](../../wiki/entities/paper-rrsi-2609-24972.md)

---

## 开源与运行状态核查（2026-10-03）

| 项 | 核查结论 |
|----|----------|
| **仓库访问** | 公开仓库，Apache-2.0 |
| **核心代码** | rrsi.py CLI 与 rrsi/ 搜索模块已发布 |
| **测试** | README 提供 python3 -m pytest tests |
| **运行入口** | smoke、baseline、run、status；run 可恢复，并将候选放进独立 git worktree |
| **任务实例** | coding、workspace、eng，分别对接 terminal coding、agentic workspace 与工程设计评测 |
| **依赖边界** | benchmark runners 使用各自环境；策略 / 提案 / 分析 / critic 需要模型服务配置。README 示例以 Vertex AI Claude Opus 4.8 为默认设置，亦描述 LiteLLM 模型标识 |
| **许可证** | Apache License 2.0 |

## 关键入口

- rrsi.py — 域选择与 smoke / baseline / run / status 命令入口
- rrsi/ — 分析、提案、critique、评测、历史、调度、selection 等核心模块
- domains/<name>/ — 域 adapter、起始 harness、演化配置与 prompt
- tests/ — 搜索环测试

> 版本提醒：以上依据 2026-10-03 可见的默认分支 README 与仓库文件；复现前核对该仓库的最新 README 与模型服务要求。

## 与论文的对应

- 论文归档：[sources/papers/rrsi_arxiv_2609_24972.md](../papers/rrsi_arxiv_2609_24972.md)
- 项目页归档：[sources/sites/regularized-rsi-com.md](../sites/regularized-rsi-com.md)
