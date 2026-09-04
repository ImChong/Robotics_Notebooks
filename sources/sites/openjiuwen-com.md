# openJiuwen 官网

> 来源归档

- **标题：** openJiuwen 智能体平台
- **类型：** site / product
- **链接：** <https://www.openjiuwen.com>
- **GitHub org：** <https://github.com/openJiuwen-ai>
- **AtomGit org：** <https://atomgit.com/openJiuwen>
- **入库日期：** 2026-09-04
- **一句话说明：** 华为 2012 实验室、华为云、终端与计算联合共建的开源 AI Agent 平台入口：WorkSwarm 蜂群办公智能体、协同工程（Coordination Engineering）、模型广场与社区；ScienceDiscovery / JiuwenSymbiosis 等垂直仓从本站生态分叉，但不都在首页一等入口。
- **沉淀到 wiki：** [`wiki/entities/openjiuwen.md`](../../wiki/entities/openjiuwen.md)；科研工作台见 [`wiki/entities/sciencediscovery.md`](../../wiki/entities/sciencediscovery.md)

## 开源状态（步骤 2.5）

**已开源（平台级）**：官网指向 GitHub / AtomGit；核心仓 Apache-2.0（如 [jiuwenswarm](https://github.com/openJiuwen-ai/jiuwenswarm)、[agent-core](https://github.com/openJiuwen-ai/agent-core)、[sciencediscovery](https://github.com/openJiuwen-ai/sciencediscovery)）。文档中心 [openJiuwen-ai/docs](https://github.com/openJiuwen-ai/docs) 为 CC-BY-4.0。

截至入库日首页可见的产品叙事（动态统计/日历区为前端加载，抓取时为空）：

| 区块 | 官网陈述 |
|------|----------|
| 定位 | 「精准 · 易用 · 高效，打造生产级 AI Agent」 |
| WorkSwarm | 三种模式：个人助手（Claw 类）、编码（Coding）、集群（Swarm）；HOTS（Human on the Swarm）/ HITS（Human in the Swarm） |
| 频道 | 小艺、飞书、钉钉、Telegram 等 |
| Coordination Engineering | Agent Swarm 分工协商；Swarm Skills 流程沉淀；Swarm Skills Hub 共享 |
| 自演进 | 执行轨迹 + 用户反馈 → 团队技能与成员技能双层演进；多参数归因的「文本梯度」优化提示/上下文 |
| 算力 | KV Cache 主动协同；多模型分级与通算/智算统一调度；叙事上亲和昇腾/鲲鹏 |
| 联系 | `contact@openjiuwen.com`；Bug 走 AtomGit / GitCode Issue |

公开新闻稿（非本站正文、仅作机构归属旁证）：openJiuwen 由华为 2012 实验室、华为云、终端、计算联合共建；WorkSwarm 为原 JiuwenSwarm 的办公向升级，上架鸿蒙 PC 并兼容 Windows / Mac / Ubuntu。**评测数字（如 PinchBench）以官方文档或仓库为准，官网首页未给可复核表。**

## 与本库相关的公开仓（未全部升格）

一次 ingest 只深挖用户指定的 ScienceDiscovery；下表供导航，避免把整个 org 一次吃完。

| 仓 | 角色 | 本库状态 |
|----|------|----------|
| [sciencediscovery](https://github.com/openJiuwen-ai/sciencediscovery) | 本地 AI 科研工作台 | **本次升格** |
| [jiuwenswarm](https://github.com/openJiuwen-ai/jiuwenswarm) | 蜂群办公 / WorkSwarm 上游 | 仅登记，未单独 wiki |
| [jiuwensymbiosis](https://github.com/openJiuwen-ai/jiuwensymbiosis) | 跨本体具身 Agent（Piper / SO-101 / Cruzr） | 仅登记；机器人读者优先跟进 |
| [agent-core](https://github.com/openJiuwen-ai/agent-core) | Agent SDK | 仅登记 |
| [skillhub](https://github.com/openJiuwen-ai/skillhub) | 技能托管（可兼容 ClawHub） | 仅登记 |
| [agent-protocol](https://github.com/openJiuwen-ai/agent-protocol) | MCP / A2A C++ SDK | 仅登记 |

## 对本库的意义

- **科研自动化工作台** 对照 [AI Auto-Research](../../wiki/concepts/ai-auto-research.md)：文献 MCP + 沙箱实验 + 审阅/provenance，而不是本库 `ingest` 把知识编译进 wiki。
- **具身控制平面** 对照 [OpenClaw](../../wiki/entities/openclaw.md) / [Philia](../../wiki/entities/philia.md)：官网 WorkSwarm 自称 Claw 类个人助手 + Swarm；真机适配在 **JiuwenSymbiosis**（`ActionSpec` + 本体 adapter），不在 ScienceDiscovery。
- **MCP 科学数据源** 对照 [Model Context Protocol](../../wiki/concepts/model-context-protocol.md)：ScienceDiscovery 把 PubMed / PDB 等收进治理过的 MCP broker。

## 参考来源（原始）

- 官网首页（2026-09-04 抓取）：<https://www.openjiuwen.com>
- GitHub org：<https://github.com/openJiuwen-ai>
- AtomGit org：<https://atomgit.com/openJiuwen>
