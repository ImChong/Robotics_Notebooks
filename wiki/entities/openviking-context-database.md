---
type: entity
project_id: openviking-context-database
project: https://openviking.ai/
code: https://github.com/volcengine/OpenViking
tags:
  - llm-agents
  - agent-memory
  - context-database
  - agent-infrastructure
status: complete
updated: 2026-10-15
related:
  - ./hermes-agent.md
  - ./agent-reach.md
  - ../concepts/agentic-coding-software-fundamentals.md
sources:
  - ../../sources/repos/volcengine-openviking.md
  - ../../sources/sites/openviking-ai.md
summary: "OpenViking 是火山引擎 Viking 团队开源的 AI Agent 上下文数据库：以 viking:// 虚拟文件系统统一组织知识资源、跨会话记忆和技能，通过目录浏览、分层摘要与范围检索按需提供上下文。"
---

# OpenViking 上下文数据库

OpenViking 是面向 AI Agent 的开源上下文数据库，将知识资源、用户记忆和技能统一组织为可浏览、可检索的 viking:// 虚拟文件系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAG | Retrieval-Augmented Generation | 结合检索结果补充生成上下文；OpenViking 将检索范围组织到目录层级 |
| URI | Uniform Resource Identifier | viking:// 地址用于定位上下文目录或资源 |
| LLM | Large Language Model | 语言模型代理从上下文库按需读取信息 |
| AGPL | GNU Affero General Public License | OpenViking 仓库采用的开源许可证 |

## 为什么重要（对机器人研究与工程）

机器人研发 Agent 会跨论文、代码仓库、实验记录和技能说明工作。OpenViking 提供一种把这些上下文作为可检查目录管理的工程方案：Agent 可先查看项目或目录摘要，再按需读取原始内容，并将会话中的经验整理为后续任务可复用的记忆。

它是 Agent 上下文基础设施，不是运动策略或机器人控制框架。可作为机器人科研与编码 Agent 的上下文存储选型候选，具体适配性仍取决于数据接入、权限隔离和部署方式。

## 核心原理

OpenViking 将内容映射到统一的 viking:// 命名空间。README 展示的基本分区包括：

- **resources**：项目文档、仓库、网页等外部或共享资料。
- **user/{user_id}/memories**：从交互中积累的偏好和经验。
- **user/{user_id}/resources**：用户自己的项目资料。
- **user/{user_id}/skills**：代理可复用的任务技能。
- **sessions**：可归档和检查的会话内容。

它把传统上不可见的向量索引包装成文件式操作：代理可以列目录、查看树、读取或搜索条目。目录摘要支持三层渐进式读取：

| 层级 | 内容 | 使用方式 |
|------|------|----------|
| L0 | Abstract，一句话摘要 | 快速筛选相关目录 |
| L1 | Overview，结构和要点概览 | 判断是否值得继续深入 |
| L2 | 原始详情 | 只在任务需要时读取完整内容 |

检索可限定在某个项目或记忆子目录内，避免每次都在一个扁平集合中搜索。会话提交后可归档对话并抽取记忆；官方文档还介绍了将来源材料编译为 wiki、知识图谱或报告的流程。

## 工程实践

| 项目 | 说明 |
|------|------|
| 资料组织 | 为机器人项目、论文、仿真环境和实验日志规划稳定目录，再把数据接入资源树 |
| 检索策略 | 先按目录缩小范围，再依据任务读取摘要、概览或原文 |
| 记忆维护 | 将跨会话偏好和执行经验作为可检查内容管理，定期核对抽取结果 |
| 集成边界 | 评估 Agent harness、权限模型、部署形态和数据生命周期；按官方文档验证具体接口 |
| 许可证 | AGPL-3.0；集成到服务或产品前应评估许可证义务 |

本仓库归档的资料不足以证明其对机器人工作负载的检索质量、延迟或可靠性优于其他方案；应基于目标数据与任务自行评测。

## 局限与风险

- **不是机器人专用系统。** 现有项目定位是通用 AI Agent 上下文数据库，机器人任务所需的时序、坐标系、传感器同步等语义仍需上层建模。
- **分层摘要有信息损失。** L0/L1 适合筛选和规划，关键实验参数与安全约束应回到 L2 原文核实。
- **记忆抽取需要审查。** 自动归纳可能遗漏上下文或固化错误经验；重要实验结论应保留原始记录并可追溯。
- **许可证需要评估。** 仓库采用 AGPL-3.0，商用集成前需结合部署方式咨询合规意见。

## 关联页面

- [Hermes Agent](./hermes-agent.md) — 常驻 Agent 运行时，也包含持久记忆机制；可比较运行时内建记忆与独立上下文数据库。
- [Agent Reach](./agent-reach.md) — 提供外部资料读取和检索渠道，可与 OpenViking 的长期上下文组织互补。
- [Agentic Coding 软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 研发 Agent 的工程环境与上下文管理实践。
- [LLM Wiki（Karpathy 模式）](../references/llm-wiki-karpathy.md) — 面向人类策展的持久知识编译方式。

## 参考来源

- [OpenViking 仓库源归档（本站）](../../sources/repos/volcengine-openviking.md)
- [OpenViking 官方项目页归档（本站）](../../sources/sites/openviking-ai.md)
- [volcengine/OpenViking（GitHub）](https://github.com/volcengine/OpenViking)
- [OpenViking 官方文档：Architecture](https://docs.openviking.ai/en/concepts/01-architecture)
- [OpenViking 官方文档：Context Layers](https://docs.openviking.ai/en/concepts/03-context-layers)
- [OpenViking 官方文档：Retrieval](https://docs.openviking.ai/en/concepts/07-retrieval)

## 推荐继续阅读

- [OpenViking README](https://github.com/volcengine/OpenViking/blob/main/README.md) — 目录模型、上下文类型与快速开始
- [OpenViking 官方文档](https://docs.openviking.ai/) — 架构、检索、会话与集成说明
