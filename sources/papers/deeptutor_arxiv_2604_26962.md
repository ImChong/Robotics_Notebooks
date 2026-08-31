# DeepTutor: Towards Agentic Personalized Tutoring（arXiv:2604.26962）

> 来源归档（ingest）

- **标题：** DeepTutor: Towards Agentic Personalized Tutoring
- **缩写 / 框架：** **DeepTutor**；**TutorBench**；agentic personalized tutoring
- **类型：** paper / tech-report / cs.CY / cs.AI / education / llm-agents
- **arXiv：** <https://arxiv.org/abs/2604.26962>（v3 2026-07-09；PDF：<https://arxiv.org/pdf/2604.26962>）
- **项目页：** <https://deeptutor.info/>
- **代码：** <https://github.com/HKUDS/DeepTutor>（**已开源**，Apache-2.0）
- **作者：** Bingxi Zhao、Jiahao Zhang、Xubin Ren、Zirui Guo、Tianzhe Chu、Yi Ma、Chao Huang
- **机构：** 香港大学（HKU）/ HKUDS 开源组织语境（以作者与 GitHub org 为准）
- **入库日期：** 2026-08-31
- **一句话说明：** 提出 **全开源 agentic 辅导框架**，统一引用落地的解题辅导与难度校准出题，并以静态知识 grounding + 动态 learner memory 的混合个性化引擎驱动自适应工作流；配套 **TutorBench** 与 profile-driven 学生模拟评测协议。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-08-31） |
|----|-------------------|
| **项目页** | [deeptutor.info](https://deeptutor.info/) 链代码仓与文档 |
| **代码** | [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor) 公开；arXiv comment 写明 *Code available at …* |
| **结论** | **已开源**（论文主张与平台实现一致）。本库以 **工具实体页** 升格为主，不另建完整 `paper-*` 深读页。 |

## 摘录 1：问题与主张（Abstract）

- **痛点：** 现有 LLM 依赖静态预训练、难以适应个体学习者；传统 RAG 不足以提供 **个性化、引导式** 反馈。
- **主张：** **DeepTutor** 作为全开源 agentic 框架，统一 **引用落地（citation-grounded）解题辅导** 与 **难度校准出题**。
- **个性化：** 混合引擎耦合 **静态知识 grounding** 与 **动态 learner memory**，持续适应学生演化需求。
- **扩展：** 同一个性化基底延伸到自适应学习流、交互式书本与主动多通道 TutorBot/Partner。
- **评测：** **TutorBench** — 跨五大学科、带定制 learner profile 的交互基准；另提出 LLM **第一人称交互评测**（profile-driven student simulator）。
- **结果（论文报告）：** 个性化指标平均 +10.8%；五个 backbone 上通用 agentic reasoning +29.4%（以论文为准）。

**对 wiki 的映射：** 写入 [`wiki/entities/deeptutor.md`](../../wiki/entities/deeptutor.md)「为什么重要 / 核心原理」；TutorBench 作为局限与评测语境一句带过。

## 摘录 2：与工程实现的对应（README 交叉）

- v1.0.0-beta.1 起 **agent-native 架构重写**（Tools + Capabilities 插件、CLI & SDK、TutorBot、Co-Writer、Guided Learning、持久 memory）与论文「agentic framework」叙事一致。
- **Knowledge Center** 多引擎 RAG、**Book** living compiler、**Partner** 多通道 IM 对应论文「adaptive workflows / interactive books / proactive multi-channel tutoring agents」。
- **My Agents / consult_subagent** 体现「agentic」不只单模型回复，而是可编排子代理与工具环。

**对 wiki 的映射：** 实体页「流程总览」Mermaid 与「工程实践」安装表。

## 对 wiki 的映射

- 主升格页：[wiki/entities/deeptutor.md](../../wiki/entities/deeptutor.md)
- 仓库归档：[sources/repos/hkuds_deeptutor.md](../repos/hkuds_deeptutor.md)
- 站点归档：[sources/sites/deeptutor-info.md](../sites/deeptutor-info.md)
