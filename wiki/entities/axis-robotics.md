---
type: entity
tags: [physical-ai, data-engine, teleoperation, sim2real, manipulation, axis-robotics, company, china-embodied]
status: complete
updated: 2026-09-27
topic: [manipulation]
related:
  - ./axis-composable-capability-library.md
  - ../concepts/data-flywheel.md
  - ../concepts/sim2real.md
  - ../concepts/recursive-self-improvement.md
  - ../methods/vla.md
  - ../methods/multi-expert-distillation.md
  - ../overview/embodied-infra-2026-panorama.md
sources:
  - ../../sources/sites/axisrobotics-ai.md
  - ../../sources/sites/axisaiorg-github-io.md
  - ../../sources/repos/axisaiorg.md
  - ../../sources/blogs/axis_composable_library_robotic_capabilities_2026-09-25.md
summary: "Axis Robotics（AXIS ROBOTICS）：去中心化 Physical AI 数据引擎——浏览器 MuJoCo-WASM 遥操作、GPU 仿真增广与训练部署管线；GitHub AxisAIOrg 部分开源；2026-09 博客提出 Grounded RSI、proxy 选轨迹与可组合 Expert 能力库路线。"
---

# Axis Robotics（AXIS ROBOTICS）

**Axis Robotics** 是面向 **Physical AI** 的 **可复利数据基础设施** 公司（[官网](https://axisrobotics.ai/)）：用 **去中心化贡献者网络**（宣称 10 万+）、**浏览器仿真遥操作** 与 **模型-centric 数据处理**，把任务生成、演示采集、训练评测与失败驱动优化连成闭环；与 [AXIS-V1 数据引擎](https://axisaiorg.github.io/AXIS-V1/)（arXiv:2607.21588）及 [AxisAIOrg](https://github.com/AxisAIOrg) 开源模块同属一条技术线。

## 一句话定义

**以「任务生成 → 采集 → 训练 → 优化」复利环扩展机器人数据多样性，并向下游 VLA/Expert 策略与 Sim2Real 部署提供跨仿真器、跨机器的统一平台。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AXIS | A Growable Community-Driven Data Engine（项目名） | 浏览器采集 + 自动任务/质检 + benchmark 的数据引擎 |
| VLA | Vision-Language-Action | 文内规划将数千 Expert 蒸馏进统一基础策略 |
| RSI | Recursive Self-Improvement | 博客 **Grounded RSI**：部署策略产生真机数据训后继者 |
| WASM | WebAssembly | 浏览器端 MuJoCo 轻量仿真遥操作 |
| Sim2Real | Simulation to Real | 平台叙事含 web 演示 → GPU 增广 → 真机部署 |

## 为什么重要

- **数据瓶颈叙事与工程对齐：** 与 [具身数据飞轮](../concepts/data-flywheel.md) 同向，但强调 **社区规模采集 + 自动质检** 而非仅实验室遥操作。
- **部分开源可复现触点：** [AxisDataCleaning](https://github.com/AxisAIOrg/AxisDataCleaning)、[AxisWebInfra](https://github.com/AxisAIOrg/AxisWebInfra) 等可独立研究 **轨迹清洗与 Web 采集**；全栈 Expert/RSI **未** 随 2026-09 博客开放。
- **Scaling 单位变化：** [可组合能力库](./axis-composable-capability-library.md) 把基本单元从「更多轨迹」改为 **fully solved、可链式的 Expert**，与 [Multi-Expert Distillation](../methods/multi-expert-distillation.md) 的「分技能专家 → 合成」形成 **操作域组合** 对照。

## 核心信息

| 字段 | 内容 |
|------|------|
| **机构** | 轴心机器人（Axis Robotics） |
| **官网** | <https://axisrobotics.ai/> |
| **技术报告** | <https://techreport.axisrobotics.ai/> |
| **GitHub** | <https://github.com/AxisAIOrg>（**部分开源**） |
| **开源结论** | 数据平台与 AXIS-V1 模块 **已部分开源**；博客 Expert/proxy/RSI **待发布**（见 [站点归档](../../sources/sites/axisrobotics-ai.md)） |

## 平台架构（压缩）

```mermaid
flowchart LR
  web[浏览器 MuJoCo-WASM<br/>遥操作采集]
  upload[统一轨迹格式上传]
  gpu[Linux GPU · IsaacSim 等<br/>域随机 /  photoreal 增广]
  clean[清洗 / 精炼管线]
  train[训练 + Sim2Real 评测]
  real[真机部署]
  web --> upload --> gpu --> clean --> train --> real
  real -.->|失败挖掘 · 新任务| web
```

## 工程实践

1. **先读开源边界：** 复现 **Web 采集 + 离线轨迹 metrics** 从 AxisAIOrg 入手；**不要假设** Expert 训练或 Grounded RSI 共训脚本已公开。
2. **与 AXIS 论文分工：** arXiv:2607.21588 侧重 **社区数据引擎与 benchmark**；2026-09-25 博客侧重 **Expert 成本、proxy 选数据、真机 RSI 环**——站内 **paper-AXIS 实体待后续 ingest**（见 [infra 全景](../overview/embodied-infra-2026-panorama.md) 登记）。
3. **RSI 用语：** 博客 **Grounded RSI** 属于 [递归自改进](../concepts/recursive-self-improvement.md) 语境下的 **有界、真机数据闭环**，不是「模型自主设计下一代架构」的 ignition。

## 局限与风险

- **数字与成本**（22%→52%、$5–10/任务）来自 **官方博客**，无独立 peer-review 或公开复现包。
- **链式 Expert** 依赖 **操作域重叠**；未披露安全监控、失败恢复与长 horizon 任务成功率。
- **VLA 蒸馏路径** 与 Expert 库路线 **并行叙事**；何者为主产品未在单篇博客定稿。

## 关联页面

- [可组合 robotic 能力库（2026-09-25 博客）](./axis-composable-capability-library.md)
- [Sim2Real](../concepts/sim2real.md) · [具身数据飞轮](../concepts/data-flywheel.md)
- [递归自改进（RSI）](../concepts/recursive-self-improvement.md)
- [VLA](../methods/vla.md) · [Multi-Expert Distillation](../methods/multi-expert-distillation.md)
- [2026 具身 infra 全景](../overview/embodied-infra-2026-panorama.md)

## 参考来源

- [axisrobotics.ai 站点归档](../../sources/sites/axisrobotics-ai.md)
- [AXIS-V1 项目页归档](../../sources/sites/axisaiorg-github-io.md)
- [AxisAIOrg 仓库索引](../../sources/repos/axisaiorg.md)
- [Composable Library 博客归档](../../sources/blogs/axis_composable_library_robotic_capabilities_2026-09-25.md)

## 推荐继续阅读

- 博客原文：<https://axisrobotics.ai/blogs/blog/beyond-more-tasks-axis-is-building-a-composable-library-of-robotic-capabilities>
- AXIS 论文：<https://arxiv.org/abs/2607.21588>
- 平台技术报告：<https://techreport.axisrobotics.ai/>
