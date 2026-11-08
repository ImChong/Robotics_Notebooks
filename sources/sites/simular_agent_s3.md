# Agent S3：Simular 官方技术文章／项目入口

> 来源归档（site / official blog）

- **类型：** 官方项目技术文章
- **项目文章：** <https://www.simular.ai/articles/agent-s3>
- **后续结果说明：** <https://www.simular.ai/articles/simulars-computer-use-agent-outperforms-humans>
- **主论文：** <https://arxiv.org/abs/2510.02250>
- **代码：** <https://github.com/simular-ai/Agent-S>（`gui_agents/s3/`）
- **入库日期：** 2026-11-08
- **核对范围：** 官方文章的项目定位、方法概述、benchmark 链接及与论文／仓库的互相指向；细节和精确评测口径以论文修订版和代码为准。
- **通读：** 已完成
- **沉淀到 wiki：** [Agent S3](../../wiki/entities/paper-agent-s3.md)

## 官方陈述与核实边界

- 论文／官方站将 Agent S3 描述为桌面计算机操作智能体；主线是单轨轻量 Worker，以及配合 Behavior Judge 的多轨迹择优。
- 72.6% 为 OSWorld 条件下多轨迹选择结果；实际部署依赖模型、操作系统、截屏／输入权限以及外部 grounding 端点。
- 在“全平台兼容”表述中，三大桌面系统是官方支持目标，不构成所有应用、分辨率、权限环境都保证正常工作的测试结论。
- 仓库已公开源码（Apache-2.0），但模型 API 与运行基础设施可能收费。

## 交叉来源

- [论文全文归档](../papers/agent_s3_arxiv_2510_02250.md)
- [源码归档](../repos/simular_agent_s3.md)
- [Agent S3 统一实体](../../wiki/entities/paper-agent-s3.md)
