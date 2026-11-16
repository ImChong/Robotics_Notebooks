# Ansys 2026 R1：首批智能体工程能力（官方资料归档）

- **英文标题：** Synopsys Launches Ansys 2026 R1 to Re-Engineer Engineering with Joint Solutions and AI-Powered Products
- **中文标题：** 新思科技推出 Ansys 2026 R1：首批智能体工程与 AI 驱动仿真功能
- **资料类型：** site（Synopsys / Ansys 官方发布公告）
- **发布方：** 新思科技（Synopsys，Ansys 品牌所属公司）
- **原始发布日期：** 2026-03-11（中文新闻页面路径为 2026-03-13，正文仍标 3 月 11 日）
- **英文原文：** <https://news.synopsys.com/2026-03-11-Synopsys-Launches-Ansys-2026-R1-to-Re-Engineer-Engineering-with-Joint-Solutions-and-AI-Powered-Products>
- **中文原文：** <https://cn.news.synopsys.com/2026-03-13>
- **产品发布亮点：** <https://ansys.synopsys.com/zh-CN/Products/Release-Highlights>
- **入库与核对日期：** 2026-11-16
- **核对范围：** 通读 Synopsys 官方英文公告及对应中文公告正文，覆盖整篇新版本发布、AI/智能体、跨工具集成、数字孪生和系统工程说明；并查阅 Ansys 2026 R1 官方发布亮点页中版本定位及产品亮点。下方聚焦官方公告关于智能体的段落，未获得产品内部实现文档。
- **通读：** 已完成

## 一手资料摘录（中文归纳）

### 明确命名的两项智能体

| 名称 | 所属软件 | 官方发布的能力 | 公告所述开放阶段 |
|------|----------|----------------|------------------|
| Mesh Agent | Ansys Mechanical | 协助诊断、排查模型前处理的网格划分失败，引导工程师采用已验证的修复步骤 | available for exploratory use（探索性使用） |
| Discovery Validation Agent | Ansys Discovery | 根据工程上下文与行业最佳实践，主动发现仿真设置问题 | early customer evaluations（早期客户评估） |

> 官方中文公告把第二项称为「Discovery 验证智能体（Validation Agent）」，英文公告使用完整名称 **Discovery Validation Agent**。

### 同版 AI 功能，但不应全部称为 Agent

- **Ansys GeomAI：** 从参考设计学习，用生成式 AI 辅助几何概念生成、评估与优化。
- **Ansys SimAI：** 公告区分 SimAI Premium SaaS 与需本地数据存储的 SimAI Pro；可通过 optiSLang 连接器串起训练数据生成、模型训练、优化与设计研究。
- **Ansys Engineering Copilot：** 集成于 medini analyze、ModelCenter 和 Rocky 的界面内 AI 辅助功能。
- **optiSLang + Discovery：** 敏感性分析与一键优化接口，支持在进入 Mechanical、Fluent 或 Icepak 的进一步验证前筛选设计。
- **其他版本方向：** TwinAI 的仿真/传感器/测试融合、ROM 向导及 CoSim 协同仿真属于本次版本更新，但非本条目的两项 Agent。

## 来源边界与开放性

1. 官方公告证实的是**辅助网格故障处理**与**主动检查设置**；未发布可逐项审计的内部模型架构、工具调用链、自治权限或成功率评测数据。
2. 以上两份公告和发布亮点页面**未列出 Mesh Agent / Discovery Validation Agent 的官方公开源码仓库或可运行入口**。截至 2026-11-16，只能按厂商产品内功能归档，不将第三方 Ansys MCP 项目冒认为官方智能体源码。
3. 网格、边界条件、物性和结果的工程审核不能因智能体提示而省略；是否可用及授权以实际安装产品和厂商渠道为准。

## 对 wiki 的映射

- [Ansys 2026 R1 智能体工程实体](../../wiki/entities/ansys-2026-r1-agentic-engineering.md)
- [CAE / CFD 代理工具与技能总览](../../wiki/overview/cae-cfd-agent-skills-landscape.md)

## 官方链接

- [Synopsys 英文原始公告](https://news.synopsys.com/2026-03-11-Synopsys-Launches-Ansys-2026-R1-to-Re-Engineer-Engineering-with-Joint-Solutions-and-AI-Powered-Products)
- [Synopsys 中文公告](https://cn.news.synopsys.com/2026-03-13)
- [Ansys 2026 R1 官方产品亮点](https://ansys.synopsys.com/zh-CN/Products/Release-Highlights)
