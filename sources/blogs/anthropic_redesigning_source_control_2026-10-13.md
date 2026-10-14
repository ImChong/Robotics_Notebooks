# Redesigning source control at Anthropic

> 来源归档

- **标题：** Redesigning source control at Anthropic
- **类型：** blog（Anthropic Engineering）
- **作者：** Nick G. 与 Anthropic source control team
- **链接：** https://www.anthropic.com/engineering/redesigning-source-control
- **发布日期：** 2026-10-13
- **入库日期：** 2026-10-14
- **一句话说明：** Anthropic 分享内部版本控制系统 Scion 的设计与迁移：针对 agent 持续在线、超大单仓与每日数万次合并，采用服务端权威 changeset、内容寻址对象、分层缓存、稀疏/虚拟检出和 API 化元数据操作。
- **开源状态：** 未开源。文章介绍的是 Anthropic 内部系统 Scion 与代码评审工具 Antique；没有提供公开源码、数据或可复现部署入口。
- **沉淀到 wiki：** 交叉补充 → [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 规模背景

- Anthropic 报告称，主分支历史接近 **100 GB**，压缩工作副本超过 **25 GB**；CI 每天约 **2500 万次 checkout**。
- 合并量从 2026 年 5 月平均约 **2500 个变更/天** 增长到 9 月超过 **30,000 个/天**。数百个 Git 镜像曾缓解流量，但随着仓库增长与合并速率上升，镜像维护成本和容量限制越来越突出。
- 七人团队在五个月内构建 Scion、代码评审工具 Antique 及配套系统，并完成主单仓迁移。
- 文章的关键判断是：在 agent 与任务默认在线的场景里，Git 为离线分布式开发提供的完整本地仓库模型，可能成为超大单仓吞吐瓶颈；这不是对一般 Git 工作流的否定。

## Scion 的设计摘要

1. **服务端执行版本控制操作。** 服务端处理历史、blame、提交、合并和 rebase；客户端主要管理工作副本、监测文件变化并维护有界对象缓存。
2. **changeset 增量记录。** 每个 changeset 记录相对首个父提交发生变化的文件及指向压缩文件内容的内容寻址指针。Blame、路径历史、提交图等由派生索引支持。
3. **按需取树和 blob。** 客户端从根树递归请求自己缺少且需要的不可变对象，不协商整个仓库的 packfile；稀疏检出只请求所需目录，虚拟检出按需物化文件。
4. **分层、可横向扩展的缓存。** 节点级 daemon、区域缓存和权威服务分层提供对象；缓存命中率均超过 99%，隔离层级也用于限制故障扩散。
5. **优先通过 API 访问仓库元数据。** CI 或自动化工具可直接查询 diff、merge base、blame 等信息，避免为了回答元数据问题而维护完整本地仓库。

## 报告的工程结果与边界

- Anthropic 报告 Scion 基础设施在 CPU 与内存需求上比此前 Git 镜像基础设施小 **200 倍**；CI 与研究任务的 checkout 快约 **两个数量级**，CI 单独释放约 **25% CPU 容量**。
- 这些是 Anthropic 针对其工作负载报告的内部结果，不是独立基准，也不能直接推断到常规规模仓库。
- Anthropic 明确表示 Scion **不是通用 Git 替代品**：它偏向超大单仓、极高频 agent 写入场景；采用它需要迁移代码评审、代码搜索和 CI 等配套开发生命周期工具，系统本身也没有作为可直接复用的公共实现发布。
- 文章同时强调，提升代码生成速度后，源码控制、构建与 CI 会成为新瓶颈；工程师仍需定义架构约束、审查变更意图，并用测试、模糊测试和运行监控守住系统行为。

## 对 agentic coding 的启示

把 agent 扩展到团队规模时，需要一起评估提交/合并吞吐、checkout 成本、CI 容量、代码审查语义和故障隔离。Scion 是面向 Anthropic 特定规模的架构案例；对大多数团队，可迁移的思路更可能是让服务端承担可扩展的元数据查询、组合稀疏检出能力，并提前测量 agent 工作负载给现有 Git 与 CI 带来的压力，而非照搬其内部系统。
