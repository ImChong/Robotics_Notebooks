# 阿里云 AI Agent Handbook（官方开源仓库）
- **类型：** repo / 企业级 Agent 工程白皮书与站点源码
- **官方仓库：** https://github.com/aliyun/ai-agent-handbook
- **上游身份：** aliyun 组织 GitHub 原仓库（REST `fork: false`）
- **核对版本：** `a1dd02ec080c4e18930660c0154da1ef589df639`（main；2026-11-11）
- **仓库创建：** 2026-09-11（GitHub created_at，非正式发布日）
- **发布资料：** 阿里云开发者社区 2026-10-23 文章《阿里云发布 Alibaba Cloud AI Agent Handbook，阐述“智能面积”》
- **入库日期：** 2026-11-11
- **维护方：** 阿里云（aliyun）与社区贡献者
- **协议：** Apache-2.0（GitHub 仓库 LICENSE）
- **开源状态：** 已公开 Markdown 书稿、VitePress 站点、脚本与测试；**并非**一套自带生产 Agent Loop / 模型权重的可执行 Agent 框架
- **知识节点：** [AI Agent Handbook](../../wiki/entities/aliyun-ai-agent-handbook.md)
- **站点归档：** [官方在线站点](../sites/aliyun-ai-agent-handbook.md)
- **核对范围：** 已完整阅读官方 README、`00-preface/00-preface.md`、架构/构建/运行/治理/调优/实践六篇导读，以及第 21 章《Agent 黄金数据集》；另核查第 04、05、06、13 章的标题结构、开头、重点片段与结尾。核对官方站点首页、仓库目录、GitHub 创建时间和 commit。**未通读 01–30 章全部逐章全文**，未运行示例、测试或站点构建；本归档的“通读”仅指上述列明的核心一手材料。
- **通读：** 已完成

## 官方一手资料定位

- [README（固定 commit）](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/README.md)：项目定位、每周更新、读者、目录、共建规划及背景。
- [前言](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/00-preface/00-preface.md)：为何从 2025 年白皮书转向开源持续维护。
- [架构篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/01-architecture)：第 01–02 章，应用形态、成熟度、参考架构。
- [构建篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/02-build)：第 03–06 章，Harness 构建入口与任务、信息、行动契约。
- [运行篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/03-run)：第 07–12 章，沙箱、状态、网关、异步与多 Agent。
- [治理篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/04-governance)：第 13–16 章，可观测、安全、资产治理、上线前模拟验证。
- [调优篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization)：第 17–24 章，模型调优与 Trace/Trajectory/黄金数据集/Badcase 数据飞轮。
- [实践篇](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/06-case-study)：企业实践，另含 Jev、Personal Agent、FDE 专题。
- [总结](https://github.com/aliyun/ai-agent-handbook/tree/a1dd02ec080c4e18930660c0154da1ef589df639/07-conclusion)：第 30 章，从 Agentic Application 到 Agentic OS。
- [第 21 章 Agent 黄金数据集](https://github.com/aliyun/ai-agent-handbook/blob/a1dd02ec080c4e18930660c0154da1ef589df639/05-optimization/%E7%AC%AC%2021%20%E7%AB%A0%E3%80%80Agent%20%E9%BB%84%E9%87%91%E6%95%B0%E6%8D%AE%E9%9B%86.md)：候选集、参考答案、rubric、专家确认、回归评估、版本留痕。
- [阿里云官方发布文章（2026-10-23）](https://developer.aliyun.com/article/1774447)。
- [2025 年 9 月旧版白皮书](https://developer.aliyun.com/ebook/8479)：属于前身，不应误作本仓库的发布日。

## 从原文提取的工程结论

1. **架构：** Agent 并非纯模型调用；Model + Harness 才是完整工程对象。Workflow、单 Agent、Long-Horizon 与 Multi-Agent 是按任务和风险选择的形态，不是单向必经升级链。
2. **构建：** 构建篇将 Harness 组织为任务/信息/行动三类契约；Agent Loop 以 Prepare → Model → Act → Observe → Verify 推进，完成须依据外部证据而不是模型主动停止。
3. **运行：** 沙箱与工作区承载受控执行；任务状态、Checkpoint、事件、Artifact、记忆需分层存储；网关区分 LLM、MCP、Agent 入口。
4. **治理：** 使用 Metrics、Logs、Trace 与事件定位行动，结合安全授权、资产版本管理、仿真验证；未经授权的工具调用不应被事后“结果正确”抵消。
5. **优化：** 从 Trace 提取 Trajectory，经数据处理、黄金样本、评估与 Badcase 归因，决定修 Harness、工具、上下文还是调模型；第 21 章特别强调黄金题要有经过业务确认的判据，并区分用于调优的题和独立验收题。

## 更新频率与可信边界

README 的“**每周会收录最新的实践，并对已发布的内容进行不定期更新**”是维护承诺，而不是本次独立统计得出的“每周必然有一次 GitHub commit”；今后应看仓库历史验证实绩。2026-09-11 为仓库创建日，2026-10-23 为找到的官方发布文章日期，2025-09 是前身白皮书。所有章节、案例数量可能随上游继续变化。

## 对本知识库的映射

[AI Agent Handbook](../../wiki/entities/aliyun-ai-agent-handbook.md) 作为知识实体，向 [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)、[Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md) 连边。机器人侧的真机执行权限、环境终态、仿真回归属于**本库迁移思考**，不是阿里云书稿的真机实验结果。
