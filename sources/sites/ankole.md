# Ankole 官方站点与中文文档

> 来源归档（site）

- **标题：** Ankole — 企业数字总部和 AI Agent Harness
- **类型：** site
- **主页：** https://ankole.agentbull.com/
- **官方中文介绍：** https://ankole.agentbull.com/zh-Hans-CN/docs/introduction/
- **快速开始：** https://ankole.agentbull.com/zh-Hans-CN/docs/quickstart/
- **代码：** https://github.com/AgentBull/ankole
- **核对范围：** 核查官方中文「介绍」及「快速开始」的架构与部署条件，对照同仓 README.zh-Hans.md 全文；快速开始的详细适配器子链接未逐一阅读。
- **通读：** 已完成
- **入库日期：** 2026-11-18
- **沉淀到 wiki：** [Ankole](../../wiki/entities/ankole.md)

## 官方介绍的能力划分

- Pod Documents / Base / Applets / Pod Chat：人在环协作的业务界面和状态。
- Meta Ankole：企业知识图（Company Brain）、Skills 与团队 Agent 规范。
- Signals / Sessions / AIGateway：事件收集、任务状态与模型路由。
- Agent Computer Worker：每个 Agent 自己的执行容器；高风险写入需遵守权限及人工审批边界。
- Applet Actions：确定性自动化任务，不需要每次消耗 LLM 推理 Token。

## 对部署承诺的限定

网站强调自托管；部署依赖 PostgreSQL、HTTPS、存储、模型密钥与每位 Agent 的 Worker。是否达到特定企业安全要求须单独审计；官方已知限制表明一个站点内不同租户共享受信任站点运维方。

## 交叉来源

- [GitHub README 与源码核对](../repos/ankole.md)
- [Ankole wiki 实体](../../wiki/entities/ankole.md)
